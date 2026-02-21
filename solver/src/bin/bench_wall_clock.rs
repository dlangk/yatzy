//! Wall-clock performance benchmarks for critical Yatzy computational routines.
//!
//! Usage:
//!   yatzy-bench --record     # Run benchmarks, save baseline
//!   yatzy-bench --check      # Run benchmarks, compare against baseline
//!   yatzy-bench              # Run benchmarks, print results only
//!
//! Baseline is saved to .overhaul/performance-baseline.json.
//! Failure threshold: max(mean + 3σ, mean × 1.05).

use std::path::Path;
use std::time::Instant;

use yatzy::api_computations::compute_roll_response;
use yatzy::phase0_tables;
use yatzy::simulation::lockstep::simulate_batch_lockstep;
use yatzy::state_computation::compute_all_state_values_nocache;
use yatzy::storage::{load_all_state_values, state_file_path};
use yatzy::types::YatzyContext;

// ── Data structures ────────────────────────────────────────────────────────

#[derive(Clone)]
struct BenchResult {
    name: String,
    #[allow(dead_code)]
    times_ms: Vec<f64>,
    mean_ms: f64,
    std_ms: f64,
    p95_ms: f64,
    threshold_ms: f64,
}

struct Baseline {
    entries: Vec<BaselineEntry>,
}

#[derive(Clone)]
#[allow(dead_code)]
struct BaselineEntry {
    name: String,
    mean_ms: f64,
    std_ms: f64,
    p95_ms: f64,
    threshold_ms: f64,
}

#[derive(PartialEq)]
enum Mode {
    Run,
    Record,
    Check,
}

// ── Benchmark runner ───────────────────────────────────────────────────────

fn run_bench<F: FnMut()>(name: &str, iterations: usize, mut f: F) -> BenchResult {
    // Warmup
    f();

    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        times.push(elapsed);
    }

    let n = times.len() as f64;
    let mean = times.iter().sum::<f64>() / n;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95_idx = ((n * 0.95) as usize).min(sorted.len() - 1);
    let p95 = sorted[p95_idx];

    // Threshold: max(mean + 3σ, mean × 1.05)
    let threshold = (mean + 3.0 * std).max(mean * 1.05);

    let result = BenchResult {
        name: name.to_string(),
        times_ms: times,
        mean_ms: mean,
        std_ms: std,
        p95_ms: p95,
        threshold_ms: threshold,
    };

    if mean >= 1.0 {
        println!(
            "  {:<32} {:>8.1} ms  (σ={:.1}, p95={:.1}, threshold={:.1})",
            name, mean, std, p95, threshold
        );
    } else {
        // Sub-millisecond: show in microseconds
        println!(
            "  {:<32} {:>8.0} μs  (σ={:.0}, p95={:.0}, threshold={:.0})",
            name,
            mean * 1000.0,
            std * 1000.0,
            p95 * 1000.0,
            threshold * 1000.0
        );
    }

    result
}

// ── Baseline I/O ───────────────────────────────────────────────────────────

fn save_baseline(results: &[BenchResult], path: &str) {
    let entries: Vec<String> = results
        .iter()
        .map(|r| {
            format!(
                "    {{\n      \"name\": \"{}\",\n      \"mean_ms\": {:.4},\n      \"std_ms\": {:.4},\n      \"p95_ms\": {:.4},\n      \"threshold_ms\": {:.4}\n    }}",
                r.name, r.mean_ms, r.std_ms, r.p95_ms, r.threshold_ms
            )
        })
        .collect();

    let json = format!(
        "{{\n  \"machine\": \"Apple M1 Max\",\n  \"date\": \"{}\",\n  \"rust_version\": \"{}\",\n  \"threads\": {},\n  \"entries\": [\n{}\n  ]\n}}\n",
        get_date(),
        get_rust_version(),
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "8".into()),
        entries.join(",\n")
    );

    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(path, json).unwrap_or_else(|e| {
        eprintln!("Failed to save baseline to {}: {}", path, e);
        std::process::exit(1);
    });
    println!("\nBaseline saved to {}", path);
}

fn load_baseline(path: &str) -> Option<Baseline> {
    let content = std::fs::read_to_string(path).ok()?;
    let mut entries = Vec::new();
    let mut i = 0;
    while let Some(pos) = content[i..].find("\"name\"") {
        let key_start = i + pos;
        // Find the colon after "name"
        let after_key = key_start + 6; // len("\"name\"")
        if let Some(colon) = content[after_key..].find(':') {
            let after_colon = after_key + colon + 1;
            // Find the opening quote of the value
            if let Some(q1) = content[after_colon..].find('"') {
                let val_start = after_colon + q1 + 1;
                // Find the closing quote
                if let Some(q2) = content[val_start..].find('"') {
                    let val_end = val_start + q2;
                    let name = content[val_start..val_end].to_string();

                    let mean_val = extract_json_number(&content, key_start, "\"mean_ms\"");
                    let std_val = extract_json_number(&content, key_start, "\"std_ms\"");
                    let p95_val = extract_json_number(&content, key_start, "\"p95_ms\"");
                    let threshold_val =
                        extract_json_number(&content, key_start, "\"threshold_ms\"");

                    if threshold_val > 0.0 || mean_val > 0.0 {
                        entries.push(BaselineEntry {
                            name,
                            mean_ms: mean_val,
                            std_ms: std_val,
                            p95_ms: p95_val,
                            threshold_ms: threshold_val,
                        });
                    }
                    i = val_end + 1;
                    continue;
                }
            }
        }
        i = key_start + 6;
    }

    if entries.is_empty() {
        None
    } else {
        Some(Baseline { entries })
    }
}

fn extract_json_number(content: &str, search_start: usize, key: &str) -> f64 {
    // Search within a reasonable window (500 chars) from search_start
    let window_end = (search_start + 500).min(content.len());
    let window = &content[search_start..window_end];
    if let Some(pos) = window.find(key) {
        let after_key = pos + key.len();
        if let Some(colon) = window[after_key..].find(':') {
            let val_start = after_key + colon + 1;
            let val_str: String = window[val_start..]
                .chars()
                .take_while(|c| *c != ',' && *c != '}' && *c != '\n')
                .collect();
            return val_str.trim().parse().unwrap_or(0.0);
        }
    }
    0.0
}

fn get_date() -> String {
    let output = std::process::Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .ok();
    output
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".into())
        .trim()
        .to_string()
}

fn get_rust_version() -> String {
    let output = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok();
    output
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".into())
        .trim()
        .to_string()
}

fn check_baseline(results: &[BenchResult], baseline: &Baseline) -> bool {
    println!(
        "\n  {:<32} {:>10} {:>10} {:>10}  {}",
        "Benchmark", "Measured", "Threshold", "Baseline", "Status"
    );
    println!("  {}", "-".repeat(85));

    let mut all_pass = true;
    for result in results {
        if let Some(entry) = baseline.entries.iter().find(|e| e.name == result.name) {
            let pass = result.mean_ms <= entry.threshold_ms;
            let status = if pass { "PASS" } else { "FAIL" };
            if !pass {
                all_pass = false;
            }
            if entry.mean_ms >= 1.0 {
                println!(
                    "  {:<32} {:>8.1} ms {:>8.1} ms {:>8.1} ms  {}",
                    result.name, result.mean_ms, entry.threshold_ms, entry.mean_ms, status
                );
            } else {
                println!(
                    "  {:<32} {:>7.0} μs  {:>7.0} μs  {:>7.0} μs   {}",
                    result.name,
                    result.mean_ms * 1000.0,
                    entry.threshold_ms * 1000.0,
                    entry.mean_ms * 1000.0,
                    status
                );
            }
        } else {
            println!(
                "  {:<32} {:>8.1} ms {:>10} {:>10}  SKIP (no baseline)",
                result.name, result.mean_ms, "-", "-"
            );
        }
    }

    all_pass
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    if std::env::set_current_dir(&base_path).is_err() {
        eprintln!("Failed to change directory to {}", base_path);
        std::process::exit(1);
    }

    let args: Vec<String> = std::env::args().collect();
    let mode = if args.iter().any(|a| a == "--record") {
        Mode::Record
    } else if args.iter().any(|a| a == "--check") {
        Mode::Check
    } else {
        Mode::Run
    };

    let baseline_path = ".overhaul/performance-baseline.json";

    // Configure rayon
    let num_threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    println!("Yatzy Wall-Clock Performance Benchmarks");
    println!("  Threads: {}", num_threads);
    println!("  Rust:    {}", get_rust_version());
    println!();

    let mut results = Vec::new();

    // ── Benchmark 1: Phase 0 lookup tables ─────────────────────────────────
    println!("Phase 0: Lookup tables");
    results.push(run_bench("phase0_tables", 10, || {
        let mut ctx = YatzyContext::new_boxed();
        phase0_tables::precompute_lookup_tables(&mut ctx);
    }));
    println!();

    // ── Benchmark 2: Full precompute θ=0 (tables + backward induction) ────
    // Uses nocache variant to always recompute (never loads from mmap).
    println!("Precomputation (θ=0, force recompute)");
    results.push(run_bench("precompute_ev", 5, || {
        let mut ctx = YatzyContext::new_boxed();
        phase0_tables::precompute_lookup_tables(&mut ctx);
        compute_all_state_values_nocache(&mut ctx);
    }));
    println!();

    // ── Benchmark 3: Simulation (requires strategy table on disk) ──────────
    let state_file = state_file_path(0.0);
    let have_tables = Path::new(&state_file).exists();

    if have_tables {
        println!("Simulation (lockstep, θ=0)");
        let mut sim_ctx = YatzyContext::new_boxed();
        phase0_tables::precompute_lookup_tables(&mut sim_ctx);
        load_all_state_values(&mut sim_ctx, &state_file);

        results.push(run_bench("simulate_lockstep_10k", 10, || {
            simulate_batch_lockstep(&sim_ctx, 10_000, 42);
        }));

        results.push(run_bench("simulate_lockstep_100k", 5, || {
            simulate_batch_lockstep(&sim_ctx, 100_000, 42);
        }));
        println!();

        // ── Benchmark 4: API compute latency ───────────────────────────────
        println!("API computation latency");

        // Mid-game: 3 categories scored, upper_score=12
        let dice = [1, 2, 3, 4, 5];

        results.push(run_bench("api_evaluate_2rerolls", 200, || {
            std::hint::black_box(compute_roll_response(&sim_ctx, 12, 0b111, &dice, 2));
        }));

        results.push(run_bench("api_evaluate_0rerolls", 200, || {
            std::hint::black_box(compute_roll_response(&sim_ctx, 12, 0b111, &dice, 0));
        }));

        // Late-game: 12 categories scored (heavier computation)
        results.push(run_bench("api_evaluate_late_game", 200, || {
            std::hint::black_box(compute_roll_response(
                &sim_ctx,
                45,
                0b0111_1111_1111_11,
                &[6, 6, 5, 5, 4],
                2,
            ));
        }));
        println!();
    } else {
        println!(
            "SKIP: Simulation and API benchmarks (no strategy table at {})",
            state_file
        );
        println!("  Run `just precompute` first, then re-run benchmarks.");
        println!();
    }

    // ── Results summary ────────────────────────────────────────────────────
    match mode {
        Mode::Record => {
            save_baseline(&results, baseline_path);
        }
        Mode::Check => {
            if let Some(baseline) = load_baseline(baseline_path) {
                let pass = check_baseline(&results, &baseline);
                println!();
                if pass {
                    println!("ALL BENCHMARKS PASSED");
                } else {
                    println!("BENCHMARK REGRESSION DETECTED");
                    std::process::exit(1);
                }
            } else {
                eprintln!(
                    "No baseline found at {}. Run with --record first.",
                    baseline_path
                );
                std::process::exit(1);
            }
        }
        Mode::Run => {
            println!("Done. Use --record to save baseline, --check to compare.");
        }
    }
}
