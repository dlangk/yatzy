//! Per-category statistics across all theta values.
//!
//! For each theta in the dense sweep, simulates N games and outputs per-category
//! statistics as CSV: theta, category, mean_score, zero_rate, mean_fill_turn,
//! score_pct_ceiling, hit_rate.

use std::io::Write;
use std::time::Instant;

use yatzy::constants::{CATEGORY_COUNT, CATEGORY_NAMES};
use yatzy::phase0_tables;
use yatzy::simulation::engine::simulate_batch_summaries;
use yatzy::storage::{load_all_state_values, state_file_path};
use yatzy::types::YatzyContext;

/// Maximum score per category (Ones..Sixes, OnePair..Yatzy).
const CATEGORY_MAX_SCORES: [u8; CATEGORY_COUNT] = [
    5, 10, 15, 20, 25, 30, // Ones–Sixes
    12, 22, 18, 24, 15, 20, 28, 30, 50, // One Pair–Yatzy
];

/// All 37 theta values: progressive spacing, dense near 0, sparse at tails.
const THETAS: [f32; 37] = [
    -3.00, -2.00, -1.50, -1.00, -0.75, -0.50, -0.30, -0.200, -0.150, -0.100, -0.070, -0.050,
    -0.040, -0.030, -0.020, -0.015, -0.010, -0.005, 0.000, 0.005, 0.010, 0.015, 0.020, 0.030,
    0.040, 0.050, 0.070, 0.100, 0.150, 0.200, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00,
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 1_000_000usize;
    let mut seed = 42u64;
    let mut output_path = String::from("outputs/aggregates/csv/category_stats.csv");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                i += 1;
                num_games = args[i].parse().expect("Invalid --games");
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("Invalid --seed");
            }
            "--output" => {
                i += 1;
                output_path = args[i].clone();
            }
            "--help" | "-h" => {
                println!("Usage: yatzy-category-sweep [--games N] [--seed S] [--output FILE]");
                println!(
                    "  Simulates N games for each of 33 theta values and outputs per-category CSV."
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Resolve output path to absolute before changing directory
    let output_path = if std::path::Path::new(&output_path).is_absolute() {
        output_path
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&output_path).to_string_lossy().to_string())
            .unwrap_or(output_path)
    };

    // Change to base path for loading state files
    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    if std::env::set_current_dir(&base_path).is_err() {
        eprintln!("Failed to change directory to {}", base_path);
        std::process::exit(1);
    }

    // Configure rayon
    let num_threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    // Precompute lookup tables once
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    // CSV header
    let mut rows: Vec<String> = Vec::with_capacity(THETAS.len() * CATEGORY_COUNT + 1);
    rows.push("theta,category_id,category_name,mean_score,zero_rate,mean_fill_turn,score_pct_ceiling,hit_rate".to_string());

    let total_start = Instant::now();

    for &theta in &THETAS {
        let theta_start = Instant::now();

        // Load state values for this theta
        ctx.theta = theta;
        let file = state_file_path(theta);
        if !load_all_state_values(&mut ctx, &file) {
            eprintln!("Skipping θ={:.3}: failed to load {}", theta, file);
            continue;
        }

        // Simulate with lightweight summaries (no dice recording)
        let summaries = simulate_batch_summaries(&ctx, num_games, seed);
        let n = summaries.len() as f64;

        // Aggregate per-category stats
        let mut score_sum = [0u64; CATEGORY_COUNT];
        let mut zero_count = [0u64; CATEGORY_COUNT];
        let mut turn_sum = [0u64; CATEGORY_COUNT];
        let mut max_count = [0u64; CATEGORY_COUNT];

        for summary in &summaries {
            for (turn_idx, turn) in summary.turns.iter().enumerate() {
                let cat = turn.category as usize;
                let scr = turn.score;
                score_sum[cat] += scr as u64;
                if scr == 0 {
                    zero_count[cat] += 1;
                }
                if scr == CATEGORY_MAX_SCORES[cat] {
                    max_count[cat] += 1;
                }
                turn_sum[cat] += turn_idx as u64;
            }
        }

        // Output CSV rows
        for c in 0..CATEGORY_COUNT {
            let mean_score = score_sum[c] as f64 / n;
            let zero_rate = zero_count[c] as f64 / n;
            let mean_fill_turn = turn_sum[c] as f64 / n + 1.0; // 1-indexed
            let ceiling = CATEGORY_MAX_SCORES[c] as f64;
            let score_pct_ceiling = if ceiling > 0.0 {
                mean_score / ceiling * 100.0
            } else {
                0.0
            };
            let hit_rate = 1.0 - zero_rate;

            rows.push(format!(
                "{:.3},{},{},{:.4},{:.6},{:.4},{:.2},{:.6}",
                theta,
                c,
                CATEGORY_NAMES[c],
                mean_score,
                zero_rate,
                mean_fill_turn,
                score_pct_ceiling,
                hit_rate,
            ));
        }

        let elapsed = theta_start.elapsed();
        println!(
            "  θ={:.3}: {:.1}s ({:.0} games/s)",
            theta,
            elapsed.as_secs_f64(),
            num_games as f64 / elapsed.as_secs_f64(),
        );
    }

    let total_elapsed = total_start.elapsed();
    println!(
        "\nTotal: {:.1}s for {} thetas × {} games",
        total_elapsed.as_secs_f64(),
        THETAS.len(),
        num_games,
    );

    // Write CSV
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = std::fs::File::create(&output_path).expect("Failed to create output file");
    for row in &rows {
        writeln!(f, "{}", row).expect("Failed to write CSV row");
    }
    println!("Output: {} ({} rows)", output_path, rows.len() - 1);
}
