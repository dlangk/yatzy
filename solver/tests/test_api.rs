//! Integration tests for the HTTP API endpoints.
//!
//! Uses axum's oneshot pattern (via tower::ServiceExt) — no TCP binding needed.
//! The setup computes state values inline (~1s), so these tests are slower than unit tests.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;

use yatzy::phase0_tables;
use yatzy::server::{create_router, create_router_full};
use yatzy::state_computation::compute_all_state_values;
use yatzy::types::YatzyContext;

fn setup_ctx() -> Arc<YatzyContext> {
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);
    compute_all_state_values(&mut ctx);
    Arc::new(*ctx)
}

/// Parse response body as JSON.
async fn body_json(body: Body) -> serde_json::Value {
    let bytes = body.collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

// Shared context (computed once per test binary via lazy_static pattern).
// Each test gets its own Router but shares the expensive context.
static CTX: std::sync::OnceLock<Arc<YatzyContext>> = std::sync::OnceLock::new();

fn get_ctx() -> Arc<YatzyContext> {
    CTX.get_or_init(setup_ctx).clone()
}

fn app() -> axum::Router {
    create_router(get_ctx())
}

fn app_with_density() -> axum::Router {
    // Create router with state_values loaded (no oracle, no percentile table)
    // — density endpoint will use real-time MC simulation for turns 5+
    create_router_full(get_ctx(), None, None)
}

// ── GET /health ──────────────────────────────────────────────────────

#[tokio::test]
async fn health_returns_200() {
    let resp = app()
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp.into_body()).await;
    assert_eq!(json["status"], "OK");
}

// ── GET /state_value ─────────────────────────────────────────────────

#[tokio::test]
async fn state_value_valid() {
    let resp = app()
        .oneshot(
            Request::get("/state_value?upper_score=0&scored_categories=0")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp.into_body()).await;
    let ev = json["expected_final_score"].as_f64().unwrap();
    assert!(ev > 230.0 && ev < 290.0, "EV={ev} out of expected range");
}

#[tokio::test]
async fn state_value_negative_upper_score() {
    let resp = app()
        .oneshot(
            Request::get("/state_value?upper_score=-1&scored_categories=0")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn state_value_upper_score_64() {
    let resp = app()
        .oneshot(
            Request::get("/state_value?upper_score=64&scored_categories=0")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn state_value_scored_categories_overflow() {
    let resp = app()
        .oneshot(
            Request::get("/state_value?upper_score=0&scored_categories=32768")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ── POST /evaluate ───────────────────────────────────────────────────

fn evaluate_request(body: serde_json::Value) -> Request<Body> {
    Request::post("/evaluate")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

#[tokio::test]
async fn evaluate_valid_with_rerolls() {
    let body = serde_json::json!({
        "dice": [3, 3, 4, 5, 6],
        "upper_score": 0,
        "scored_categories": 0,
        "rerolls_remaining": 2,
    });
    let resp = app().oneshot(evaluate_request(body)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let json = body_json(resp.into_body()).await;
    let cats = json["categories"].as_array().unwrap();
    assert_eq!(cats.len(), 15);
    assert!(json.get("mask_evs").is_some());
    assert!(json.get("optimal_mask").is_some());
    assert!(json.get("optimal_category").is_some());
    assert!(json.get("state_ev").is_some());
}

#[tokio::test]
async fn evaluate_no_rerolls_has_no_masks() {
    let body = serde_json::json!({
        "dice": [1, 2, 3, 4, 5],
        "upper_score": 0,
        "scored_categories": 0,
        "rerolls_remaining": 0,
    });
    let resp = app().oneshot(evaluate_request(body)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let json = body_json(resp.into_body()).await;
    assert!(json.get("mask_evs").is_none());
    assert!(json.get("optimal_mask").is_none());
}

#[tokio::test]
async fn evaluate_invalid_dice() {
    let body = serde_json::json!({
        "dice": [0, 1, 2, 3, 4],
        "upper_score": 0,
        "scored_categories": 0,
        "rerolls_remaining": 2,
    });
    let resp = app().oneshot(evaluate_request(body)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn evaluate_invalid_rerolls() {
    let body = serde_json::json!({
        "dice": [1, 2, 3, 4, 5],
        "upper_score": 0,
        "scored_categories": 0,
        "rerolls_remaining": 3,
    });
    let resp = app().oneshot(evaluate_request(body)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ── POST /density ───────────────────────────────────────────────────

#[tokio::test]
async fn density_game_over_returns_degenerate() {
    // All 15 categories scored — game over, should return degenerate response
    let body = serde_json::json!({
        "upper_score": 63,
        "scored_categories": 32767,
        "accumulated_score": 280,
    });
    let resp = app_with_density()
        .oneshot(
            Request::post("/density")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp.into_body()).await;
    // With upper_score=63, bonus = 50, so final = 280 + 50 = 330
    let mean = json["mean"].as_f64().unwrap();
    assert_eq!(mean, 330.0);
    let p50 = json["percentiles"]["p50"].as_i64().unwrap();
    assert_eq!(p50, 330);
}

#[tokio::test]
async fn density_mc_returns_percentiles() {
    // 10 categories scored — should use real-time MC simulation
    let body = serde_json::json!({
        "upper_score": 30,
        "scored_categories": 1023,
        "accumulated_score": 150,
    });
    let resp = app_with_density()
        .oneshot(
            Request::post("/density")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp.into_body()).await;
    // Should have mean, std_dev, percentiles
    assert!(json.get("mean").is_some());
    assert!(json.get("std_dev").is_some());
    let pcts = json["percentiles"].as_object().unwrap();
    assert!(pcts.contains_key("p50"));
    assert!(pcts.contains_key("p10"));
    assert!(pcts.contains_key("p90"));
    // p50 should be reasonable (150 accumulated + some remaining)
    let p50 = pcts["p50"].as_i64().unwrap();
    assert!(p50 > 150, "p50={} should be > accumulated 150", p50);
    assert!(p50 < 374, "p50={} should be < max possible 374", p50);
}

// ── Determinism ──────────────────────────────────────────────────────

#[tokio::test]
async fn evaluate_deterministic() {
    let body = serde_json::json!({
        "dice": [2, 3, 4, 5, 6],
        "upper_score": 10,
        "scored_categories": 0,
        "rerolls_remaining": 1,
    });
    let resp1 = app().oneshot(evaluate_request(body.clone())).await.unwrap();
    let json1 = body_json(resp1.into_body()).await;

    let resp2 = app().oneshot(evaluate_request(body)).await.unwrap();
    let json2 = body_json(resp2.into_body()).await;

    assert_eq!(json1, json2);
}

// ── Optimal category is argmax ───────────────────────────────────────

#[tokio::test]
async fn optimal_category_is_argmax() {
    let body = serde_json::json!({
        "dice": [5, 5, 5, 6, 6],
        "upper_score": 0,
        "scored_categories": 0,
        "rerolls_remaining": 0,
    });
    let resp = app().oneshot(evaluate_request(body)).await.unwrap();
    let json = body_json(resp.into_body()).await;

    let optimal_id = json["optimal_category"].as_i64().unwrap();
    let cats = json["categories"].as_array().unwrap();

    let mut best_ev = f64::NEG_INFINITY;
    let mut best_id = -1i64;
    for cat in cats {
        if cat["available"].as_bool().unwrap() {
            let ev = cat["ev_if_scored"].as_f64().unwrap();
            if ev > best_ev {
                best_ev = ev;
                best_id = cat["id"].as_i64().unwrap();
            }
        }
    }
    assert_eq!(optimal_id, best_id);
}

// ── Category fields ──────────────────────────────────────────────────

#[tokio::test]
async fn category_fields_present() {
    let body = serde_json::json!({
        "dice": [1, 1, 2, 3, 4],
        "upper_score": 0,
        "scored_categories": 0,
        "rerolls_remaining": 0,
    });
    let resp = app().oneshot(evaluate_request(body)).await.unwrap();
    let json = body_json(resp.into_body()).await;

    for cat in json["categories"].as_array().unwrap() {
        assert!(cat.get("id").is_some());
        assert!(cat.get("name").is_some());
        assert!(cat.get("score").is_some());
        assert!(cat.get("available").is_some());
        assert!(cat.get("ev_if_scored").is_some());
    }
}
