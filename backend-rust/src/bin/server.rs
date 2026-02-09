use std::sync::Arc;

use yatzy::phase0_tables;
use yatzy::server::create_router;
use yatzy::state_computation::compute_all_state_values;
use yatzy::storage::load_all_state_values;
use yatzy::types::YatzyContext;

fn set_working_directory() {
    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    println!("YATZY_BASE_PATH={}", base_path);
    if std::env::set_current_dir(&base_path).is_err() {
        eprintln!("Failed to change directory to {}", base_path);
        std::process::exit(1);
    }
    if let Ok(cwd) = std::env::current_dir() {
        println!("Working directory changed to: {}", cwd.display());
    }
}

#[tokio::main]
async fn main() {
    set_working_directory();
    println!("Starting yatzy API server...");

    let mut ctx = Box::new(YatzyContext::new());
    phase0_tables::precompute_lookup_tables(&mut ctx);

    if !load_all_state_values(&mut ctx, "data/all_states.bin") {
        println!("No precomputed data found, computing (run yatzy-precompute to avoid this)...");
        compute_all_state_values(&mut ctx);
    }

    let ctx = Arc::new(*ctx);
    let app = create_router(ctx);

    let port = 9000;
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
