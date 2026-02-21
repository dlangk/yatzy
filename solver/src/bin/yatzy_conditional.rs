//! Conditional Yatzy hit-rate analysis across θ values and score bands.
//!
//! Tests the hypothesis: does risk-seeking (high θ) sacrifice Yatzy universally,
//! or only dump it in games already going poorly?
//!
//! For each θ (plus max-policy), simulates N games and computes Yatzy (category 14)
//! hit rates conditioned on the game's total score falling in various bands, plus a
//! dynamic top-5% band per strategy.
//!
//! Output CSV columns:
//!   theta, band, n_games, yatzy_hit_count, yatzy_hit_rate,
//!   mean_score_all, mean_score_hit, mean_score_miss

use std::io::Write;
use std::time::Instant;

use yatzy::constants::CATEGORY_YATZY;
use yatzy::phase0_tables;
use yatzy::simulation::engine::{simulate_batch_summaries, GameSummary};
use yatzy::storage::{load_all_state_values, state_file_path};
use yatzy::types::YatzyContext;

/// θ values to sweep (sparse set covering the interesting range).
const THETAS: [f32; 6] = [0.0, 0.05, 0.10, 0.20, 0.50, 1.10];

/// Score band boundaries (upper bound exclusive, except last which is unbounded).
const BAND_EDGES: [i32; 7] = [200, 220, 240, 260, 280, 300, i32::MAX];
const BAND_LABELS: [&str; 7] = [
    "<200", "200-220", "220-240", "240-260", "260-280", "280-300", "300+",
];

fn band_index(score: i32) -> usize {
    if score < 200 {
        return 0;
    }
    for (i, &edge) in BAND_EDGES.iter().enumerate().skip(1) {
        if score < edge {
            return i;
        }
    }
    BAND_EDGES.len() - 1
}

/// Accumulator for one score band.
#[derive(Default, Clone)]
struct BandStats {
    n: u64,
    yatzy_hits: u64,
    score_sum: u64,
    score_sum_hit: u64,
    score_sum_miss: u64,
    n_hit: u64,
    n_miss: u64,
}

/// Aggregate summaries into band stats and emit CSV rows.
/// Returns (overall_hit_pct, top5_hit_pct) for logging.
fn aggregate_and_emit(
    label: &str,
    summaries: &[GameSummary],
    rows: &mut Vec<String>,
) -> (f64, f64) {
    // Extract per-game: (total_score, yatzy_hit)
    let game_data: Vec<(i32, bool)> = summaries
        .iter()
        .map(|s| {
            let yatzy_hit = s
                .turns
                .iter()
                .any(|t| t.category == CATEGORY_YATZY as u8 && t.score > 0);
            (s.total_score, yatzy_hit)
        })
        .collect();

    // Sort scores to find p95 threshold
    let mut sorted_scores: Vec<i32> = game_data.iter().map(|&(s, _)| s).collect();
    sorted_scores.sort_unstable();
    let p95_idx = (sorted_scores.len() as f64 * 0.95).floor() as usize;
    let p95_threshold = sorted_scores[p95_idx.min(sorted_scores.len() - 1)];

    // Accumulate into bands
    let num_bands = BAND_LABELS.len();
    let mut bands = vec![BandStats::default(); num_bands + 2]; // +2 for top5pct and all
    let top5_idx = num_bands;
    let all_idx = num_bands + 1;

    for &(score, yatzy_hit) in &game_data {
        let bi = band_index(score);

        // Update specific band
        bands[bi].n += 1;
        bands[bi].score_sum += score as u64;
        if yatzy_hit {
            bands[bi].yatzy_hits += 1;
            bands[bi].score_sum_hit += score as u64;
            bands[bi].n_hit += 1;
        } else {
            bands[bi].score_sum_miss += score as u64;
            bands[bi].n_miss += 1;
        }

        // Update top-5% band
        if score >= p95_threshold {
            bands[top5_idx].n += 1;
            bands[top5_idx].score_sum += score as u64;
            if yatzy_hit {
                bands[top5_idx].yatzy_hits += 1;
                bands[top5_idx].score_sum_hit += score as u64;
                bands[top5_idx].n_hit += 1;
            } else {
                bands[top5_idx].score_sum_miss += score as u64;
                bands[top5_idx].n_miss += 1;
            }
        }

        // Update all band
        bands[all_idx].n += 1;
        bands[all_idx].score_sum += score as u64;
        if yatzy_hit {
            bands[all_idx].yatzy_hits += 1;
            bands[all_idx].score_sum_hit += score as u64;
            bands[all_idx].n_hit += 1;
        } else {
            bands[all_idx].score_sum_miss += score as u64;
            bands[all_idx].n_miss += 1;
        }
    }

    // Emit CSV rows
    let band_names: Vec<&str> = BAND_LABELS
        .iter()
        .copied()
        .chain(std::iter::once("top5pct"))
        .chain(std::iter::once("all"))
        .collect();

    for (bi, name) in band_names.iter().enumerate() {
        let b = &bands[bi];
        if b.n == 0 {
            continue;
        }
        let hit_rate = b.yatzy_hits as f64 / b.n as f64;
        let mean_all = b.score_sum as f64 / b.n as f64;
        let mean_hit = if b.n_hit > 0 {
            b.score_sum_hit as f64 / b.n_hit as f64
        } else {
            0.0
        };
        let mean_miss = if b.n_miss > 0 {
            b.score_sum_miss as f64 / b.n_miss as f64
        } else {
            0.0
        };

        rows.push(format!(
            "{},{},{},{},{:.6},{:.2},{:.2},{:.2}",
            label, name, b.n, b.yatzy_hits, hit_rate, mean_all, mean_hit, mean_miss,
        ));
    }

    let overall = if bands[all_idx].n > 0 {
        bands[all_idx].yatzy_hits as f64 / bands[all_idx].n as f64 * 100.0
    } else {
        0.0
    };
    let top5 = if bands[top5_idx].n > 0 {
        bands[top5_idx].yatzy_hits as f64 / bands[top5_idx].n as f64 * 100.0
    } else {
        0.0
    };
    (overall, top5)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 1_000_000usize;
    let mut seed = 42u64;
    let mut output_path = String::from("outputs/aggregates/csv/yatzy_conditional.csv");

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
                println!("Usage: yatzy-conditional [--games N] [--seed S] [--output FILE]");
                println!(
                    "  Simulates N games for each theta (+ max-policy) and outputs conditional Yatzy hit rates."
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
    let _base = yatzy::env_config::init_base_path();

    // Configure rayon
    let _threads = yatzy::env_config::init_rayon_threads();

    // Precompute lookup tables once
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    // CSV header
    let mut rows: Vec<String> = Vec::with_capacity((THETAS.len() + 1) * 9 + 1);
    rows.push(
        "theta,band,n_games,yatzy_hit_count,yatzy_hit_rate,\
         mean_score_all,mean_score_hit,mean_score_miss"
            .to_string(),
    );

    let total_start = Instant::now();

    // θ sweep
    for &theta in &THETAS {
        let theta_start = Instant::now();

        ctx.theta = theta;
        ctx.max_policy = false;
        let file = state_file_path(theta);
        if !load_all_state_values(&mut ctx, &file) {
            eprintln!("Skipping θ={:.3}: failed to load {}", theta, file);
            continue;
        }

        let summaries = simulate_batch_summaries(&ctx, num_games, seed);
        let label = format!("{:.3}", theta);
        let (overall, top5) = aggregate_and_emit(&label, &summaries, &mut rows);

        let elapsed = theta_start.elapsed();
        println!(
            "  θ={:.3}: {:.1}s ({:.0} games/s) — Yatzy hit={:.1}%, top5% Yatzy={:.1}%",
            theta,
            elapsed.as_secs_f64(),
            num_games as f64 / elapsed.as_secs_f64(),
            overall,
            top5,
        );
    }

    // Max-policy run (uses θ=0 state values with max-outcome chance nodes)
    {
        let mp_start = Instant::now();

        ctx.theta = 0.0;
        ctx.max_policy = true;
        let file = state_file_path(0.0);
        if !load_all_state_values(&mut ctx, &file) {
            eprintln!("Skipping max-policy: failed to load {}", file);
        } else {
            let summaries = simulate_batch_summaries(&ctx, num_games, seed);
            let (overall, top5) = aggregate_and_emit("max_policy", &summaries, &mut rows);

            let elapsed = mp_start.elapsed();
            println!(
                "  max-policy: {:.1}s ({:.0} games/s) — Yatzy hit={:.1}%, top5% Yatzy={:.1}%",
                elapsed.as_secs_f64(),
                num_games as f64 / elapsed.as_secs_f64(),
                overall,
                top5,
            );
        }
        ctx.max_policy = false;
    }

    let total_elapsed = total_start.elapsed();
    println!(
        "\nTotal: {:.1}s for {} thetas + max-policy × {} games",
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
