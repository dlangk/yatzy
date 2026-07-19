//! Axum HTTP server: stateless endpoints for the Yatzy frontend.
//!
//! All endpoints are stateless lookups against the precomputed `YatzyContext`.
//! The context is shared as `Arc<YatzyContext>` across async handlers.
//!
//! ## Endpoints
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | GET | `/health` | Health check |
//! | GET | `/state_value` | Look up E_table[S] for a given state |
//! | POST | `/evaluate` | All mask EVs + category EVs for one roll |
//! | POST | `/density` | Score distribution percentiles from a mid-game state |

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use tokio::sync::Semaphore;
use tower_http::cors::{Any, CorsLayer};

use crate::api_computations::compute_roll_response;
use crate::batched_solver::{precompute_exp_scores, ExpScores};
use crate::constants::state_index;
use crate::simulation::engine::{extract_percentiles_i32, PERCENTILE_KEYS};
use crate::types::{PercentileEntry, PolicyOracle, StateValues, YatzyContext};

/// A non-neutral risk detent: its table (log-domain L(S)) plus precomputed
/// exp(theta*score). Neutral (theta=0) is served from the shared context.
pub struct Detent {
    pub theta: f32,
    pub sv: StateValues,
    pub exp_scores: Box<ExpScores>,
}

/// Shared server state: context + optional oracle + optional percentile table.
pub struct ServerState {
    pub ctx: Arc<YatzyContext>,
    pub oracle: Option<Arc<PolicyOracle>>,
    pub percentile_table: Option<HashMap<u32, PercentileEntry>>,
    /// The four non-neutral risk detents (theta = -0.08, -0.04, +0.04, +0.08).
    pub detents: Vec<Detent>,
    /// Limits concurrent MC density simulations to 1 at a time.
    pub density_semaphore: Arc<Semaphore>,
}

pub type AppState = Arc<ServerState>;

pub fn create_router(ctx: Arc<YatzyContext>) -> Router {
    let state = Arc::new(ServerState {
        ctx,
        oracle: None,
        percentile_table: None,
        detents: Vec::new(),
        density_semaphore: Arc::new(Semaphore::new(1)),
    });
    create_router_with_state(state)
}

pub fn create_router_full(
    ctx: Arc<YatzyContext>,
    oracle: Option<PolicyOracle>,
    percentile_table: Option<HashMap<u32, PercentileEntry>>,
    detents: Vec<Detent>,
) -> Router {
    let state = Arc::new(ServerState {
        ctx,
        oracle: oracle.map(Arc::new),
        percentile_table,
        detents,
        density_semaphore: Arc::new(Semaphore::new(1)),
    });
    create_router_with_state(state)
}

/// Load the four non-neutral detent tables from disk. Missing files are logged
/// and skipped (the endpoint then returns fewer detents). Neutral is not here.
pub fn load_detents(ctx: &YatzyContext) -> Vec<Detent> {
    const NON_NEUTRAL_THETAS: [f32; 4] = [-0.08, -0.04, 0.04, 0.08];
    let mut detents = Vec::new();
    for &theta in &NON_NEUTRAL_THETAS {
        let path = crate::storage::state_file_path(theta);
        match crate::storage::load_state_values_standalone(&path) {
            Some(sv) => {
                let exp_scores = precompute_exp_scores(ctx, theta);
                detents.push(Detent {
                    theta,
                    sv,
                    exp_scores,
                });
                println!("Loaded risk detent theta={} from {}", theta, path);
            }
            None => {
                eprintln!("Risk detent theta={} not loaded (missing {})", theta, path);
            }
        }
    }
    detents
}

#[cfg(test)]
mod detent_tests {
    use super::*;
    use crate::phase0_tables;

    #[test]
    #[ignore] // needs data/strategy_tables/all_states_theta_*.bin
    fn test_load_detents_finds_all_four() {
        // `cargo test` runs with cwd = the package root (solver/), but the
        // theta tables live in the repo root's data/strategy_tables/ (same
        // fallback pattern as test_precomputed.rs::resolve_state_file()).
        let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
        let _ = std::env::set_current_dir(&base_path);
        if !crate::storage::file_exists(&crate::storage::state_file_path(-0.08)) {
            let _ = std::env::set_current_dir("..");
        }

        let mut ctx = YatzyContext::new_boxed();
        phase0_tables::precompute_lookup_tables(&mut ctx);
        let detents = load_detents(&ctx);
        assert_eq!(detents.len(), 4, "expected all four detent tables present");
        let thetas: Vec<f32> = detents.iter().map(|d| d.theta).collect();
        assert_eq!(thetas, vec![-0.08, -0.04, 0.04, 0.08]);
    }
}

fn create_router_with_state(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(handle_health_check))
        .route("/state_value", get(handle_get_state_value))
        .route("/evaluate", post(handle_evaluate))
        .route("/density", post(handle_density))
        .layer(cors)
        .with_state(state)
}

// ── Request/Response types ──────────────────────────────────────────

#[derive(Deserialize)]
struct EvaluateRequest {
    dice: [i32; 5],
    upper_score: i32,
    scored_categories: i32,
    rerolls_remaining: i32,
}

#[derive(Deserialize)]
struct StateValueQuery {
    upper_score: i32,
    scored_categories: i32,
}

#[derive(Deserialize)]
struct DensityRequest {
    upper_score: i32,
    scored_categories: i32,
    accumulated_score: i32,
}

fn error_response(status: StatusCode, msg: &str) -> (StatusCode, Json<serde_json::Value>) {
    (status, Json(serde_json::json!({ "error": msg })))
}

// ── GET handlers ────────────────────────────────────────────────────

async fn handle_health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "OK" }))
}

async fn handle_get_state_value(
    State(state): State<AppState>,
    Query(params): Query<StateValueQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    if params.upper_score < 0 || params.upper_score > 63 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "upper_score must be 0-63",
        ));
    }
    if params.scored_categories < 0 || params.scored_categories >= (1 << 15) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "scored_categories must be a valid 15-bit bitmask",
        ));
    }
    let val = state
        .ctx
        .get_state_value(params.upper_score, params.scored_categories);
    Ok(Json(serde_json::json!({
        "upper_score": params.upper_score,
        "scored_categories": params.scored_categories,
        "expected_final_score": val,
    })))
}

// ── POST handler ────────────────────────────────────────────────────

async fn handle_evaluate(
    State(state): State<AppState>,
    Json(req): Json<EvaluateRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    if req.rerolls_remaining < 0 || req.rerolls_remaining > 2 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "rerolls_remaining must be 0, 1, or 2",
        ));
    }
    if req.upper_score < 0 || req.upper_score > 63 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "upper_score must be 0-63",
        ));
    }
    if req.scored_categories < 0 || req.scored_categories >= (1 << 15) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "scored_categories must be a valid 15-bit bitmask",
        ));
    }
    for &d in &req.dice {
        if d < 1 || d > 6 {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "Each die must be between 1 and 6",
            ));
        }
    }

    // Serialize one RollResponse into the flat JSON object shape.
    // Off-neutral detents, these _ev-named fields hold certainty-equivalent values, not EVs.
    let to_obj = |resp: &crate::api_computations::RollResponse, theta: f32| -> serde_json::Value {
        let categories: Vec<serde_json::Value> = resp
            .categories
            .iter()
            .map(|cat| {
                serde_json::json!({
                    "id": cat.id,
                    "name": cat.name,
                    "score": cat.score,
                    "available": cat.available,
                    "ev_if_scored": if cat.ev_if_scored.is_finite() { cat.ev_if_scored } else { 0.0 },
                })
            })
            .collect();
        let mut obj = serde_json::json!({
            "theta": theta,
            "categories": categories,
            "optimal_category": resp.optimal_category,
            "optimal_category_ev": resp.optimal_category_ev,
            "state_ev": resp.state_ev,
        });
        if let Some(mask_evs) = resp.mask_evs {
            obj["mask_evs"] = serde_json::json!(mask_evs);
            obj["optimal_mask"] = serde_json::json!(resp.optimal_mask.unwrap());
            obj["optimal_mask_ev"] = serde_json::json!(resp.optimal_mask_ev.unwrap());
        }
        obj
    };

    // Neutral (theta = 0): the existing EV path, unchanged.
    let neutral = compute_roll_response(
        &state.ctx,
        req.upper_score,
        req.scored_categories,
        &req.dice,
        req.rerolls_remaining,
    );

    // Build EXACTLY five detents in fixed theta order. A missing detent table
    // falls back to the neutral response (that risk level then behaves as
    // Neutral), so the array is always length 5. The frontend relies on that.
    const DETENT_THETAS: [f32; 5] = [-0.08, -0.04, 0.0, 0.04, 0.08];
    let detent_objs: Vec<serde_json::Value> = DETENT_THETAS
        .iter()
        .map(|&theta| {
            if theta == 0.0 {
                return to_obj(&neutral, 0.0);
            }
            match state
                .detents
                .iter()
                .find(|d| (d.theta - theta).abs() < 1e-6)
            {
                Some(d) => {
                    let r = crate::api_computations::compute_roll_response_theta(
                        &state.ctx,
                        d.sv.as_slice(),
                        &d.exp_scores,
                        d.theta,
                        req.upper_score,
                        req.scored_categories,
                        &req.dice,
                        req.rerolls_remaining,
                    );
                    to_obj(&r, d.theta)
                }
                // Missing table: emit neutral values under this detent's theta,
                // so the array stays length 5 and the level degrades to Neutral.
                None => to_obj(&neutral, theta),
            }
        })
        .collect();

    // Flat fields = neutral (backward compatible), plus the detents array.
    let mut result = to_obj(&neutral, 0.0);
    result["detents"] = serde_json::Value::Array(detent_objs);

    Ok(Json(result))
}

// ── POST /density handler ──────────────────────────────────────────
//
// Three-tier approach:
// - turns_scored <= 4: O(1) lookup from precomputed percentile table
// - turns_scored 5-14: real-time MC simulation (~10K games)
// - turns_scored == 15: degenerate (all percentiles = accumulated_score)

async fn handle_density(
    State(state): State<AppState>,
    Json(req): Json<DensityRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    if req.upper_score < 0 || req.upper_score > 63 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "upper_score must be 0-63",
        ));
    }
    if req.scored_categories < 0 || req.scored_categories >= (1 << 15) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "scored_categories must be a valid 15-bit bitmask",
        ));
    }
    if req.accumulated_score < 0 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "accumulated_score must be non-negative",
        ));
    }

    let turns_scored = (req.scored_categories as u32).count_ones();

    // Turn 15: game over — all percentiles equal accumulated score
    if turns_scored >= 15 {
        let final_score = req.accumulated_score + if req.upper_score >= 63 { 50 } else { 0 };
        let mut percentiles = HashMap::new();
        for &k in &PERCENTILE_KEYS {
            percentiles.insert(format!("p{}", k), final_score);
        }
        return Ok(Json(serde_json::json!({
            "mean": final_score as f64,
            "std_dev": 0.0,
            "percentiles": percentiles,
        })));
    }

    // Turns 0-4: precomputed percentile lookup (O(1))
    if turns_scored <= 4 {
        if let Some(ref table) = state.percentile_table {
            let si = state_index(req.upper_score as usize, req.scored_categories as usize) as u32;
            if let Some(entry) = table.get(&si) {
                let mut percentiles = HashMap::new();
                for (i, &k) in PERCENTILE_KEYS.iter().enumerate() {
                    percentiles.insert(
                        format!("p{}", k),
                        entry.percentiles[i] + req.accumulated_score,
                    );
                }
                let mean = entry.mean + req.accumulated_score as f64;
                return Ok(Json(serde_json::json!({
                    "mean": mean,
                    "std_dev": entry.std_dev,
                    "percentiles": percentiles,
                })));
            }
        }
        // Fall through to MC if table not loaded or state not found
    }

    // Turns 5-14: real-time MC simulation (no oracle needed)
    // Acquire semaphore to limit concurrent simulations to 1.
    let permit = match tokio::time::timeout(
        std::time::Duration::from_secs(5),
        Arc::clone(&state.density_semaphore).acquire_owned(),
    )
    .await
    {
        Ok(Ok(permit)) => permit,
        Ok(Err(_)) => {
            return Err(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Density semaphore closed",
            ));
        }
        Err(_) => {
            return Err(error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "Server busy — density simulation already running",
            ));
        }
    };

    let ctx = Arc::clone(&state.ctx);
    let upper_score = req.upper_score;
    let scored_categories = req.scored_categories;
    let accumulated_score = req.accumulated_score;

    let computation = tokio::task::spawn_blocking(move || {
        let _permit = permit; // hold permit until simulation completes
        let num_games: usize = 1_000;
        let seed = (upper_score as u64)
            .wrapping_mul(65537)
            .wrapping_add(scored_categories as u64);
        let sorted = crate::simulation::engine::simulate_remaining_scores(
            &ctx,
            upper_score,
            scored_categories,
            num_games,
            seed,
        );
        let (mean, std_dev, pcts) = extract_percentiles_i32(&sorted);

        let mut percentiles = HashMap::new();
        for (i, &k) in PERCENTILE_KEYS.iter().enumerate() {
            percentiles.insert(format!("p{}", k), pcts[i] + accumulated_score);
        }
        let mean = mean + accumulated_score as f64;

        serde_json::json!({
            "mean": mean,
            "std_dev": std_dev,
            "percentiles": percentiles,
        })
    });

    match tokio::time::timeout(std::time::Duration::from_secs(10), computation).await {
        Ok(Ok(result)) => Ok(Json(result)),
        Ok(Err(_)) => Err(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Density computation failed",
        )),
        Err(_) => Err(error_response(
            StatusCode::REQUEST_TIMEOUT,
            "Density computation timed out (>10s)",
        )),
    }
}
