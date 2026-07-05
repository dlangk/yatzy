//! Shared environment configuration for all Yatzy binaries.
//!
//! Consolidates `YATZY_BASE_PATH`, `RAYON_NUM_THREADS`, and `YATZY_PORT`
//! reads shared by all 28 binaries.

use std::path::PathBuf;

/// Read `YATZY_BASE_PATH` (default `"."`), chdir, print path. Exits on failure.
pub fn init_base_path() -> PathBuf {
    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    println!("YATZY_BASE_PATH={}", base_path);
    let path = PathBuf::from(&base_path);
    if std::env::set_current_dir(&base_path).is_err() {
        eprintln!("Failed to change directory to {}", base_path);
        std::process::exit(1);
    }
    if let Ok(cwd) = std::env::current_dir() {
        println!("Working directory: {}", cwd.display());
    }
    path
}

/// Default thread count when `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` are unset.
/// Uses `std::thread::available_parallelism()` so it scales to whatever host
/// we run on (M1 Max: 10, M5 Max: 18, large servers: more). Falls back to 8.
fn default_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
}

fn resolve_thread_count() -> usize {
    std::env::var("RAYON_NUM_THREADS")
        .or_else(|_| std::env::var("OMP_NUM_THREADS"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(default_thread_count)
}

/// Read `RAYON_NUM_THREADS` (fallback `OMP_NUM_THREADS`, default = all cores).
/// Builds rayon global thread pool. Returns thread count.
pub fn init_rayon_threads() -> usize {
    let num_threads = resolve_thread_count();
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();
    println!("Rayon threads: {}", num_threads);
    num_threads
}

/// Like [`init_rayon_threads`] but tolerates an already-initialized pool.
/// Returns thread count.
pub fn init_rayon_threads_lenient() -> usize {
    let num_threads = resolve_thread_count();
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .ok(); // May fail if already initialized
    println!("Rayon threads: {}", num_threads);
    num_threads
}

/// Read `YATZY_PORT` (default 9000).
pub fn server_port() -> u16 {
    std::env::var("YATZY_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9000)
}
