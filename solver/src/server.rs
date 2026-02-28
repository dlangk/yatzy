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
//! | POST | `/density` | Exact score distribution from a mid-game state |

use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use tower_http::cors::{Any, CorsLayer};

use crate::api_computations::compute_roll_response;
use crate::types::{PolicyOracle, YatzyContext};

/// Shared server state: context + optional oracle for density endpoint.
pub struct ServerState {
    pub ctx: Arc<YatzyContext>,
    pub oracle: Option<Arc<PolicyOracle>>,
}

pub type AppState = Arc<ServerState>;

pub fn create_router(ctx: Arc<YatzyContext>) -> Router {
    let state = Arc::new(ServerState { ctx, oracle: None });
    create_router_with_state(state)
}

pub fn create_router_with_oracle(ctx: Arc<YatzyContext>, oracle: Option<PolicyOracle>) -> Router {
    let state = Arc::new(ServerState {
        ctx,
        oracle: oracle.map(Arc::new),
    });
    create_router_with_state(state)
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

    let resp = compute_roll_response(
        &state.ctx,
        req.upper_score,
        req.scored_categories,
        &req.dice,
        req.rerolls_remaining,
    );

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

    let mut result = serde_json::json!({
        "categories": categories,
        "optimal_category": resp.optimal_category,
        "optimal_category_ev": resp.optimal_category_ev,
        "state_ev": resp.state_ev,
    });

    if let Some(mask_evs) = resp.mask_evs {
        result["mask_evs"] = serde_json::json!(mask_evs);
        result["optimal_mask"] = serde_json::json!(resp.optimal_mask.unwrap());
        result["optimal_mask_ev"] = serde_json::json!(resp.optimal_mask_ev.unwrap());
    }

    Ok(Json(result))
}

// ── POST /density handler ──────────────────────────────────────────

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

    // Guard: reject density with >11 remaining turns. Forward DP cost grows
    // combinatorially; on a 2-CPU/1GB server, turns 0-3 exceed 30s or OOM.
    // Turn 4 (11 remaining) takes ~20s; turn 5+ is <3s.
    let turns_scored = (req.scored_categories as u32).count_ones();
    if turns_scored < 4 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "Density requires at least 4 scored categories (≤11 remaining turns).",
        ));
    }

    let oracle = match &state.oracle {
        Some(o) => Arc::clone(o),
        None => {
            return Err(error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "Oracle not loaded. Run yatzy-precompute --oracle first.",
            ))
        }
    };

    let ctx = Arc::clone(&state.ctx);
    let upper_score = req.upper_score as usize;
    let scored_categories = req.scored_categories as usize;
    let accumulated_score = req.accumulated_score as usize;

    let computation = tokio::task::spawn_blocking(move || {
        crate::density::forward::density_evolution_from_state(
            &ctx,
            &oracle,
            upper_score,
            scored_categories,
            accumulated_score,
        )
    });

    let result = match tokio::time::timeout(std::time::Duration::from_secs(30), computation).await
    {
        Ok(Ok(r)) => r,
        Ok(Err(_)) => {
            return Err(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Density computation failed",
            ))
        }
        Err(_) => {
            return Err(error_response(
                StatusCode::REQUEST_TIMEOUT,
                "Density computation timed out (>30s)",
            ))
        }
    };

    Ok(Json(serde_json::json!({
        "mean": result.mean,
        "std_dev": result.std_dev,
        "percentiles": result.percentiles,
    })))
}
