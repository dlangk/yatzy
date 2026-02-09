//! Axum HTTP server: 9 endpoints for the Yatzy frontend.
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
//! | POST | `/evaluate_category_score` | Score for placing dice in a category |
//! | POST | `/available_categories` | List categories with scores and validity |
//! | POST | `/evaluate_all_categories` | EV for each available category (0 rerolls) |
//! | POST | `/evaluate_actions` | EV for all 32 reroll masks |
//! | POST | `/suggest_optimal_action` | Best reroll mask or best category |
//! | POST | `/evaluate_user_action` | EV of a user's chosen action |

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

use crate::api_computations::*;
use crate::constants::*;
use crate::dice_mechanics::{find_dice_set_index, sort_dice_set};
use crate::game_mechanics::update_upper_score;
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
        .route(
            "/evaluate_category_score",
            post(handle_evaluate_category_score),
        )
        .route("/available_categories", post(handle_available_categories))
        .route(
            "/evaluate_all_categories",
            post(handle_evaluate_all_categories),
        )
        .route("/evaluate_actions", post(handle_evaluate_actions))
        .route(
            "/suggest_optimal_action",
            post(handle_suggest_optimal_action),
        )
        .route("/evaluate_user_action", post(handle_evaluate_user_action))
        .layer(cors)
        .with_state(ctx)
}

// ── Request/Response types ──────────────────────────────────────────

#[derive(Deserialize)]
struct DiceRequest {
    dice: [i32; 5],
    #[serde(default)]
    category_id: Option<i32>,
    #[serde(default)]
    scored_categories: Option<i32>,
    #[serde(default)]
    upper_score: Option<i32>,
    #[serde(default)]
    rerolls_remaining: Option<i32>,
    #[serde(default)]
    user_action: Option<serde_json::Value>,
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

// ── POST handlers ───────────────────────────────────────────────────

async fn handle_evaluate_category_score(
    State(ctx): State<AppState>,
    Json(req): Json<DiceRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let cat_id = req
        .category_id
        .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "Missing category_id"))?
        as usize;

    let mut dice = req.dice;
    sort_dice_set(&mut dice);
    let ds_index = find_dice_set_index(&ctx, &dice);
    let score = ctx.precomputed_scores[ds_index][cat_id];

    Ok(Json(serde_json::json!({
        "category_id": cat_id,
        "category_name": CATEGORY_NAMES[cat_id],
        "score": score,
    })))
}

async fn handle_available_categories(
    State(ctx): State<AppState>,
    Json(req): Json<DiceRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let scored_categories = req
        .scored_categories
        .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "Missing scored_categories"))?;

    let mut dice = req.dice;
    sort_dice_set(&mut dice);
    let ds_index = find_dice_set_index(&ctx, &dice);

    let mut categories = Vec::new();
    for c in 0..CATEGORY_COUNT {
        let scr = ctx.precomputed_scores[ds_index][c];
        let valid = !is_category_scored(scored_categories, c) && scr > 0;
        categories.push(serde_json::json!({
            "id": c,
            "name": CATEGORY_NAMES[c],
            "score": scr,
            "valid": valid,
        }));
    }

    Ok(Json(serde_json::json!({ "categories": categories })))
}

async fn handle_evaluate_all_categories(
    State(ctx): State<AppState>,
    Json(req): Json<DiceRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let upper_score = req
        .upper_score
        .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "Missing upper_score"))?;
    let scored_categories = req
        .scored_categories
        .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "Missing scored_categories"))?;
    let rerolls = req
        .rerolls_remaining
        .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "Missing rerolls_remaining"))?;

    if rerolls != 0 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "rerolls_remaining must be 0 for this endpoint",
        ));
    }

    let mut dice = req.dice;
    sort_dice_set(&mut dice);
    let ds_index = find_dice_set_index(&ctx, &dice);

    let mut categories = Vec::new();
    for c in 0..CATEGORY_COUNT {
        if !is_category_scored(scored_categories, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            let new_up = update_upper_score(upper_score, c, scr);
            let new_scored = scored_categories | (1 << c);
            let ev = scr as f64 + ctx.get_state_value(new_up, new_scored);
            categories.push(serde_json::json!({
                "id": c,
                "name": CATEGORY_NAMES[c],
                "score": scr,
                "expected_value_if_chosen": ev,
            }));
        }
    }

    Ok(Json(serde_json::json!({ "categories": categories })))
}

async fn handle_evaluate_actions(
    State(ctx): State<AppState>,
    Json(req): Json<DiceRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let upper_score = req
        .upper_score
        .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "Missing upper_score"))?;
    let scored_categories = req
        .scored_categories
        .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "Missing scored_categories"))?;
    let rerolls = req
        .rerolls_remaining
        .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "Missing rerolls_remaining"))?;

    if rerolls <= 0 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "rerolls_remaining must be > 0",
        ));
    }

    let mut dice = req.dice;
    sort_dice_set(&mut dice);

    let mut e_ds_for_masks = [0.0f64; 252];
    compute_expected_values(
        &ctx,
        upper_score,
        scored_categories,
        rerolls,
        &mut e_ds_for_masks,
    );
    let ds_index = find_dice_set_index(&ctx, &dice);

    let mut actions = Vec::new();
    for mask in 0..32 {
        let mut distribution = [EVProbabilityPair {
            ev: 0.0,
            probability: 0.0,
            ds2_index: 0,
        }; 252];
        compute_distribution_for_reroll_mask(
            &ctx,
            ds_index,
            &e_ds_for_masks,
            mask,
            &mut distribution,
        );
        let ev = compute_ev_from_distribution(&distribution, 252);
        actions.push(serde_json::json!({
            "mask": mask,
            "expected_value": ev,
        }));
    }

    Ok(Json(serde_json::json!({ "actions": actions })))
}

async fn handle_suggest_optimal_action(
    State(ctx): State<AppState>,
    Json(req): Json<DiceRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let upper_score = req
        .upper_score
        .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "Missing upper_score"))?;
    let scored_categories = req
        .scored_categories
        .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "Missing scored_categories"))?;
    let rerolls = req
        .rerolls_remaining
        .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "Missing rerolls_remaining"))?;

    let mut dice = req.dice;
    sort_dice_set(&mut dice);

    let mut resp = serde_json::json!({});

    if rerolls > 0 {
        let mut best_mask = 0;
        let mut ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            upper_score,
            scored_categories,
            &dice,
            rerolls,
            &mut best_mask,
            &mut ev,
        );
        let mask_binary: String = (0..5)
            .map(|i| if best_mask & (1 << i) != 0 { '1' } else { '0' })
            .collect();
        resp["best_reroll"] = serde_json::json!({
            "id": best_mask,
            "expected_value": ev,
            "mask_binary": mask_binary,
        });
    } else {
        let mut best_ev = 0.0;
        let best_category = choose_best_category_no_rerolls(
            &ctx,
            upper_score,
            scored_categories,
            &dice,
            &mut best_ev,
        );
        if best_category >= 0 {
            resp["best_category"] = serde_json::json!({
                "id": best_category,
                "name": CATEGORY_NAMES[best_category as usize],
                "expected_value": best_ev,
            });
        }
    }

    Ok(Json(resp))
}

async fn handle_evaluate_user_action(
    State(ctx): State<AppState>,
    Json(req): Json<DiceRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let upper_score = req
        .upper_score
        .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "Missing upper_score"))?;
    let scored_categories = req
        .scored_categories
        .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "Missing scored_categories"))?;
    let rerolls = req
        .rerolls_remaining
        .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "Missing rerolls_remaining"))?;

    let user_action = req
        .user_action
        .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "Missing user_action"))?;

    let mut dice = req.dice;

    if rerolls > 0 {
        let mask = user_action
            .get("best_reroll")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "Invalid best_reroll"))?
            as i32;

        let ev =
            evaluate_chosen_reroll_mask(&ctx, upper_score, scored_categories, &dice, mask, rerolls);
        let ev = if ev.is_nan() || ev.is_infinite() {
            0.0
        } else {
            ev
        };
        Ok(Json(serde_json::json!({ "expected_value": ev })))
    } else {
        let category_id = user_action
            .get("best_category")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "Invalid best_category"))?
            as usize;

        sort_dice_set(&mut dice);
        let ev = evaluate_chosen_category(&ctx, upper_score, scored_categories, &dice, category_id);
        let ev = if ev.is_nan() || ev.is_infinite() {
            0.0
        } else {
            ev
        };
        Ok(Json(serde_json::json!({ "expected_value": ev })))
    }
}
