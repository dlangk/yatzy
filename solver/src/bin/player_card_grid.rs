//! Pre-compute player card grid: simulate games for a 4D parameter grid
//! (θ × β × γ × d) and write summary statistics to JSON.
//!
//! Usage:
//!   YATZY_BASE_PATH=. solver/target/release/yatzy-player-card-grid [OPTIONS]
//!
//! Options:
//!   --games N      Games per parameter combo (default: 100000)
//!   --seed S       Random seed (default: 42)
//!   --output PATH  Output JSON file (default: profiler/data/player_card_grid.json)

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::Serialize;

use yatzy::phase0_tables;
use yatzy::profiling::player_card::simulate_game_profiled;
use yatzy::profiling::qvalues::sigma_for_depth;
use yatzy::simulation::engine::simulate_batch;
use yatzy::simulation::sweep::ensure_strategy_table;
use yatzy::storage::{load_all_state_values, state_file_path};
use yatzy::types::YatzyContext;

// ── Parameter grids ──

const THETA_GRID: [f32; 6] = [-0.05, -0.02, 0.0, 0.02, 0.05, 0.1];
const BETA_GRID: [f32; 6] = [0.5, 1.0, 2.0, 4.0, 7.0, 10.0];
const GAMMA_GRID: [f32; 6] = [0.3, 0.6, 0.8, 0.9, 0.95, 1.0];
const D_GRID: [u32; 3] = [8, 20, 999];

// ── Output types ──

#[derive(Serialize)]
struct GridStats {
    mean: f32,
    std: f32,
    p5: i32,
    p10: i32,
    p25: i32,
    p50: i32,
    p75: i32,
    p90: i32,
    p95: i32,
    p99: i32,
    bonus_rate: f32,
}

#[derive(Serialize)]
struct OutputRoot {
    theta_values: Vec<f32>,
    beta_values: Vec<f32>,
    gamma_values: Vec<f32>,
    d_values: Vec<u32>,
    games_per_combo: usize,
    optimal: GridStats,
    grid: Vec<GridStats>,
}

fn percentile(sorted: &[i32], p: f32) -> i32 {
    let idx = ((p / 100.0) * (sorted.len() - 1) as f32).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn compute_stats(scores: &mut [i32]) -> GridStats {
    scores.sort_unstable();
    let n = scores.len() as f64;
    let mean = scores.iter().map(|&s| s as f64).sum::<f64>() / n;
    let variance = scores
        .iter()
        .map(|&s| {
            let d = s as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;

    GridStats {
        mean: mean as f32,
        std: variance.sqrt() as f32,
        p5: percentile(scores, 5.0),
        p10: percentile(scores, 10.0),
        p25: percentile(scores, 25.0),
        p50: percentile(scores, 50.0),
        p75: percentile(scores, 75.0),
        p90: percentile(scores, 90.0),
        p95: percentile(scores, 95.0),
        p99: percentile(scores, 99.0),
        bonus_rate: 0.0, // filled in by caller
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 100_000usize;
    let mut seed = 42u64;
    let mut output_path = String::from("profiler/data/player_card_grid.json");

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
                println!("Usage: yatzy-player-card-grid [OPTIONS]");
                println!("  --games N      Games per combo (default: 100000)");
                println!("  --seed S       Random seed (default: 42)");
                println!(
                    "  --output PATH  Output JSON (default: profiler/data/player_card_grid.json)"
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

    let _base = yatzy::env_config::init_base_path();
    let _threads = yatzy::env_config::init_rayon_threads();

    let total_start = Instant::now();
    let total_combos = THETA_GRID.len() * BETA_GRID.len() * GAMMA_GRID.len() * D_GRID.len();
    println!(
        "Player Card Grid: {} combos × {} games = {} total games",
        total_combos,
        num_games,
        total_combos * num_games
    );

    // Phase 0: precompute lookup tables
    println!("Precomputing lookup tables...");
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    // Load θ=0 and simulate optimal baseline
    let file0 = state_file_path(0.0);
    if !load_all_state_values(&mut ctx, &file0) {
        eprintln!("Cannot load {}", file0);
        std::process::exit(1);
    }
    ctx.theta = 0.0;
    println!("Loaded θ=0 state values");

    println!("Simulating optimal baseline ({} games)...", num_games);
    let opt_result = simulate_batch(&ctx, num_games, seed);
    let mut opt_scores = opt_result.scores;
    let optimal = compute_stats(&mut opt_scores);
    println!(
        "  Optimal: mean={:.1}, std={:.1}, p50={}",
        optimal.mean, optimal.std, optimal.p50
    );

    // Ensure all θ strategy tables exist
    for &theta in &THETA_GRID {
        if theta != 0.0 {
            ensure_strategy_table(&mut ctx, theta);
        }
    }

    // Simulate grid: iterate over θ (outer) to minimize strategy table reloads
    let mut grid: Vec<GridStats> = Vec::with_capacity(total_combos);
    let mut combo_idx = 0usize;

    for &theta in &THETA_GRID {
        // Load this θ's strategy table
        let file = state_file_path(theta);
        if !load_all_state_values(&mut ctx, &file) {
            eprintln!("Cannot load {}", file);
            std::process::exit(1);
        }
        ctx.theta = theta;

        for &beta in &BETA_GRID {
            for &gamma in &GAMMA_GRID {
                for &d in &D_GRID {
                    let sigma_d = sigma_for_depth(d);
                    let combo_seed = seed.wrapping_add(combo_idx as u64 * 1000003);

                    // Simulate games in parallel
                    let mut scores: Vec<i32> = (0..num_games)
                        .into_par_iter()
                        .map(|game_i| {
                            let mut rng =
                                SmallRng::seed_from_u64(combo_seed.wrapping_add(game_i as u64));
                            simulate_game_profiled(&ctx, &mut rng, beta, gamma, sigma_d)
                        })
                        .collect();

                    let stats = compute_stats(&mut scores);

                    if combo_idx.is_multiple_of(50) || combo_idx == total_combos - 1 {
                        println!(
                            "  [{:>3}/{}] θ={:>6.3} β={:>4.1} γ={:>4.2} d={:>3} → mean={:.1}",
                            combo_idx + 1,
                            total_combos,
                            theta,
                            beta,
                            gamma,
                            d,
                            stats.mean,
                        );
                    }

                    grid.push(stats);
                    combo_idx += 1;
                }
            }
        }
    }

    let output = OutputRoot {
        theta_values: THETA_GRID.to_vec(),
        beta_values: BETA_GRID.to_vec(),
        gamma_values: GAMMA_GRID.to_vec(),
        d_values: D_GRID.to_vec(),
        games_per_combo: num_games,
        optimal,
        grid,
    };

    // Ensure output directory exists
    if let Some(parent) = Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let json = serde_json::to_string(&output).expect("JSON serialization failed");
    let mut file = std::fs::File::create(&output_path).expect("Cannot create output file");
    file.write_all(json.as_bytes()).expect("Write failed");

    println!(
        "\nDone in {:.1}s. Wrote {} grid entries to {}",
        total_start.elapsed().as_secs_f64(),
        total_combos,
        output_path
    );
}
