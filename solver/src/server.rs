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
//! | GET | `/score_histogram` | Binned score distribution from CSV |
//! | GET | `/statistics` | Aggregated game statistics from simulation |
//! | POST | `/evaluate` | All mask EVs + category EVs for one roll |

use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use tower_http::cors::{Any, CorsLayer};

use crate::api_computations::compute_roll_response;
use crate::types::YatzyContext;

pub type AppState = Arc<YatzyContext>;

pub fn create_router(ctx: Arc<YatzyContext>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(handle_health_check))
        .route("/state_value", get(handle_get_state_value))
        .route("/score_histogram", get(handle_get_score_histogram))
        .route("/statistics", get(handle_get_statistics))
        .route("/evaluate", post(handle_evaluate))
        .layer(cors)
        .with_state(ctx)
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

fn error_response(status: StatusCode, msg: &str) -> (StatusCode, Json<serde_json::Value>) {
    (status, Json(serde_json::json!({ "error": msg })))
}

// ── GET handlers ────────────────────────────────────────────────────

async fn handle_health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "OK" }))
}

async fn handle_get_state_value(
    State(ctx): State<AppState>,
    Query(params): Query<StateValueQuery>,
) -> impl IntoResponse {
    let val = ctx.get_state_value(params.upper_score, params.scored_categories);
    Json(serde_json::json!({
        "upper_score": params.upper_score,
        "scored_categories": params.scored_categories,
        "expected_final_score": val,
    }))
}

async fn handle_get_score_histogram(
    State(_ctx): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let file_path = "data/score_histogram.csv";
    let content = match std::fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(_) => {
            return Err(error_response(
                StatusCode::NOT_FOUND,
                "Could not open score histogram file",
            ))
        }
    };

    let min_ev = 100;
    let max_ev = 380;
    let bin_count = 56;
    let bin_width = (max_ev - min_ev) as f64 / bin_count as f64;
    let mut bins = vec![0i32; bin_count];

    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue; // skip header
        }
        if let Ok(score) = line.trim().parse::<i32>() {
            if score >= min_ev && score <= max_ev {
                let bin_index = ((score - min_ev) as f64 / bin_width) as usize;
                if bin_index < bin_count {
                    bins[bin_index] += 1;
                }
            }
        }
    }

    Ok(Json(serde_json::json!({
        "min_ev": min_ev,
        "max_ev": max_ev,
        "bin_count": bin_count,
        "bins": bins,
    })))
}

async fn handle_get_statistics(
    State(_ctx): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let file_path = "analytics/results/aggregates/json/game_statistics.json";
    let content = match std::fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(_) => {
            return Err(error_response(
                StatusCode::NOT_FOUND,
                "Statistics not found. Run yatzy-simulate --output results first.",
            ))
        }
    };

    let json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => {
            return Err(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to parse statistics JSON",
            ))
        }
    };

    Ok(Json(json))
}

// ── POST handler ────────────────────────────────────────────────────

async fn handle_evaluate(
    State(ctx): State<AppState>,
    Json(req): Json<EvaluateRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    if req.rerolls_remaining < 0 || req.rerolls_remaining > 2 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "rerolls_remaining must be 0, 1, or 2",
        ));
    }

    let resp = compute_roll_response(
        &ctx,
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
