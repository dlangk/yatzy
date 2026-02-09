use yatzy::phase0_tables;
use yatzy::state_computation::compute_all_state_values;
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

fn main() {
    set_working_directory();
    println!("Yatzy precomputation tool (Rust)");

    // Configure rayon thread pool
    let num_threads = std::env::var("RAYON_NUM_THREADS")
        .or_else(|_| std::env::var("OMP_NUM_THREADS"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();
    println!("Using {} threads", num_threads);

    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);
    compute_all_state_values(&mut ctx);

    println!("Precomputation complete.");
}
