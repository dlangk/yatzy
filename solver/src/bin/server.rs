use std::sync::Arc;

use yatzy::phase0_tables;
use yatzy::server::create_router_full;
use yatzy::state_computation::compute_all_state_values;
use yatzy::storage::{
    load_all_state_values, load_oracle, load_percentile_table, ORACLE_FILE_PATH,
    PERCENTILE_FILE_PATH,
};
use yatzy::types::YatzyContext;

#[tokio::main]
async fn main() {
    let _base = yatzy::env_config::init_base_path();
    let port = yatzy::env_config::server_port();
    println!("Starting yatzy API server...");

    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    if !load_all_state_values(&mut ctx, "data/strategy_tables/all_states.bin") {
        println!("No precomputed data found, computing (run yatzy-precompute to avoid this)...");
        compute_all_state_values(&mut ctx);
    }

    // Load oracle for density endpoint (optional — skip if file missing)
    let oracle = load_oracle(ORACLE_FILE_PATH);
    if oracle.is_some() {
        println!("Oracle loaded — /density endpoint available with exact DP for high-turn states");
    } else {
        println!("Oracle not found — /density endpoint will use MC simulation only");
    }

    // Load percentile table for fast turn 0-4 density (optional)
    let percentile_table = load_percentile_table(PERCENTILE_FILE_PATH);
    if percentile_table.is_some() {
        println!("Percentile table loaded — /density O(1) lookup for turns 0-4");
    } else {
        println!("Percentile table not found — /density will use MC for all turns");
    }

    let ctx = Arc::new(*ctx);
    let app = create_router_full(ctx, oracle, percentile_table);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
        .await
        .unwrap();
    println!("Server is running on port {}. Press Ctrl+C to stop.", port);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();

    println!("\nStopping server...");
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C signal handler");
}
