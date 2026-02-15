//! Two-phase adaptive percentile sweep: finds the θ that maximizes each percentile.
//!
//! Phase 1 (coarse): [-0.08, +0.40] step 0.01, N₁ games each.
//!   → identifies approximate peak θ* per percentile.
//!
//! Phase 2 (fine): [θ*−0.015, θ*+0.015] step 0.002, N₂ games each.
//!   → pinpoints peaks with high precision.
//!
//! Precomputes missing θ strategy tables on the fly (~7s each).
//!
//! Output: `percentile_sweep.csv` with per-θ percentile values from the finest available data.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::time::Instant;

use yatzy::phase0_tables;
use yatzy::simulation::engine::simulate_batch;
use yatzy::state_computation::compute_all_state_values;
use yatzy::storage::{load_all_state_values, state_file_path};
use yatzy::types::YatzyContext;

/// Percentiles to track. Ordered from left-peak to right-peak.
const PERCENTILES: &[(&str, f64)] = &[
    ("p1", 1.0),
    ("p5", 5.0),
    ("p10", 10.0),
    ("p25", 25.0),
    ("p50", 50.0),
    ("p75", 75.0),
    ("p90", 90.0),
    ("p95", 95.0),
    ("p99", 99.0),
    ("p999", 99.9),
    ("p9999", 99.99),
];

/// Per-θ result: all percentiles + basic stats.
#[derive(Clone)]
struct SweepResult {
    theta: f32,
    mean: f64,
    std_dev: f64,
    percentiles: Vec<i32>, // same order as PERCENTILES
    num_games: usize,
    phase: u8, // 1 or 2
}

/// Generate coarse grid: [-0.08, +0.40] step 0.01
fn coarse_grid() -> Vec<f32> {
    let mut grid = Vec::new();
    let mut t = -0.08f32;
    while t <= 0.401 {
        // Round to avoid float drift
        let rounded = (t * 1000.0).round() / 1000.0;
        grid.push(rounded);
        t += 0.01;
    }
    grid
}

/// Generate fine grid around a peak: [center-0.015, center+0.015] step 0.002
fn fine_grid_around(center: f32) -> Vec<f32> {
    let mut grid = Vec::new();
    let mut t = center - 0.015;
    while t <= center + 0.0151 {
        let rounded = (t * 10000.0).round() / 10000.0;
        grid.push(rounded);
        t += 0.002;
    }
    grid
}

/// Compute percentiles from sorted scores.
fn compute_percentiles(scores: &[i32]) -> Vec<i32> {
    PERCENTILES
        .iter()
        .map(|&(_, pct)| {
            let idx = ((pct / 100.0) * scores.len() as f64) as usize;
            scores[idx.min(scores.len() - 1)]
        })
        .collect()
}

/// Ensure a θ strategy table exists, precomputing if necessary.
/// Returns false if precomputation fails.
fn ensure_table(ctx: &mut Box<YatzyContext>, theta: f32) -> bool {
    let file = state_file_path(theta);
    if std::path::Path::new(&file).exists() {
        return true;
    }

    println!("    Precomputing θ={:.4}...", theta);
    let t0 = Instant::now();
    ctx.theta = theta;

    // Reset state values to owned for computation
    ctx.state_values = yatzy::types::StateValues::Owned(vec![0.0f32; yatzy::constants::NUM_STATES]);

    // Re-initialize terminal states for this theta
    phase0_tables::initialize_final_states(ctx);

    // Run DP
    compute_all_state_values(ctx);

    println!(
        "    Precomputed θ={:.4} in {:.1}s",
        theta,
        t0.elapsed().as_secs_f64()
    );
    true
}

/// Simulate games for a given θ, returning a SweepResult.
fn simulate_theta(
    ctx: &mut Box<YatzyContext>,
    theta: f32,
    num_games: usize,
    seed: u64,
    phase: u8,
) -> Option<SweepResult> {
    let file = state_file_path(theta);
    if !load_all_state_values(ctx, &file) {
        eprintln!("    Failed to load θ={:.4}: {}", theta, file);
        return None;
    }
    ctx.theta = theta;

    let result = simulate_batch(ctx, num_games, seed);

    let percentiles = compute_percentiles(&result.scores);

    Some(SweepResult {
        theta,
        mean: result.mean,
        std_dev: result.std_dev,
        percentiles,
        num_games,
        phase,
    })
}

struct Args {
    games_coarse: usize,
    games_fine: usize,
    seed: u64,
    output: Option<String>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut games_coarse = 1_000_000usize;
    let mut games_fine = 10_000_000usize;
    let mut seed = 42u64;
    let mut output: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games-coarse" => {
                i += 1;
                if i < args.len() {
                    games_coarse = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid --games-coarse value: {}", args[i]);
                        std::process::exit(1);
                    });
                }
            }
            "--games-fine" => {
                i += 1;
                if i < args.len() {
                    games_fine = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid --games-fine value: {}", args[i]);
                        std::process::exit(1);
                    });
                }
            }
            "--seed" => {
                i += 1;
                if i < args.len() {
                    seed = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid --seed value: {}", args[i]);
                        std::process::exit(1);
                    });
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    output = Some(args[i].clone());
                }
            }
            "--help" | "-h" => {
                println!(
                    "Usage: yatzy-percentile-sweep [--games-coarse N] [--games-fine N] [--seed S] [--output DIR]"
                );
                println!();
                println!("Two-phase adaptive sweep to find θ that maximizes each percentile.");
                println!("  --games-coarse N  Games per θ in phase 1 (default: 1000000)");
                println!("  --games-fine N    Games per θ in phase 2 (default: 10000000)");
                println!("  --seed S          RNG seed (default: 42)");
                println!("  --output DIR      Write CSV results to DIR");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Args {
        games_coarse,
        games_fine,
        seed,
        output,
    }
}

fn main() {
    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    if std::env::set_current_dir(&base_path).is_err() {
        eprintln!("Failed to change directory to {}", base_path);
        std::process::exit(1);
    }

    let args = parse_args();

    // Configure rayon
    let num_threads = std::env::var("RAYON_NUM_THREADS")
        .or_else(|_| std::env::var("OMP_NUM_THREADS"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Percentile Sweep: Find θ* that Maximizes Each Percentile");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Phase 1 (coarse): 1M games, step 0.01");
    println!(
        "  Phase 2 (fine):   {}M games, step 0.002",
        args.games_fine / 1_000_000
    );
    println!("  Seed:             {}", args.seed);
    println!("  Threads:          {}", num_threads);
    if let Some(ref dir) = args.output {
        println!("  Output:           {}", dir);
    }
    println!();

    // ── Setup ────────────────────────────────────────────────────────────────

    let t0 = Instant::now();
    let mut ctx = yatzy::types::YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);
    println!(
        "  Phase 0 tables: {:.1} ms\n",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // All results indexed by theta (use BTreeMap for sorted order)
    let mut all_results: BTreeMap<i64, SweepResult> = BTreeMap::new();
    let theta_key = |t: f32| -> i64 { (t * 100000.0).round() as i64 };

    // ── Phase 1: Coarse sweep ──────────────────────────────────────────────

    let coarse = coarse_grid();
    println!(
        "── Phase 1: Coarse Sweep ({} θ values × {} games) ──",
        coarse.len(),
        args.games_coarse
    );

    // Ensure all tables exist
    let t_precompute = Instant::now();
    for &theta in &coarse {
        ensure_table(&mut ctx, theta);
    }
    let precompute_time = t_precompute.elapsed().as_secs_f64();
    if precompute_time > 1.0 {
        println!("  Precomputation: {:.1}s\n", precompute_time);
    }

    // Header
    print!("  {:>7}", "θ");
    for &(name, _) in PERCENTILES {
        print!(" {:>6}", name);
    }
    print!(" {:>8}", "mean");
    println!();
    println!("  {}", "─".repeat(7 + PERCENTILES.len() * 7 + 9));

    let t_phase1 = Instant::now();
    for &theta in &coarse {
        if let Some(r) = simulate_theta(&mut ctx, theta, args.games_coarse, args.seed, 1) {
            print!("  {:>+7.3}", r.theta);
            for &p in &r.percentiles {
                print!(" {:>6}", p);
            }
            print!(" {:>8.2}", r.mean);
            println!();
            all_results.insert(theta_key(theta), r);
        }
    }
    println!(
        "  Phase 1 done in {:.1}s\n",
        t_phase1.elapsed().as_secs_f64()
    );

    // ── Find peaks from phase 1 ────────────────────────────────────────────

    println!("── Phase 1 Peaks ─────────────────────────────────────────────────");
    let mut fine_thetas: Vec<f32> = Vec::new();

    for (pi, &(name, _)) in PERCENTILES.iter().enumerate() {
        let best = all_results
            .values()
            .max_by_key(|r| r.percentiles[pi])
            .unwrap();
        println!(
            "  {:<6} peak at θ={:+.3} (value={})",
            name, best.theta, best.percentiles[pi]
        );

        // Add fine grid around this peak
        for t in fine_grid_around(best.theta) {
            if !fine_thetas.iter().any(|&x| (x - t).abs() < 1e-5) {
                fine_thetas.push(t);
            }
        }
    }

    fine_thetas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Remove any that already have phase-2 quality data
    let new_fine: Vec<f32> = fine_thetas
        .iter()
        .copied()
        .filter(|&t| all_results.get(&theta_key(t)).is_none_or(|r| r.phase < 2))
        .collect();

    println!(
        "\n  {} unique fine-grid θ values ({} new)\n",
        fine_thetas.len(),
        new_fine.len()
    );

    // ── Phase 2: Fine sweep ────────────────────────────────────────────────

    println!(
        "── Phase 2: Fine Sweep ({} θ values × {} games) ──",
        new_fine.len(),
        args.games_fine
    );

    // Ensure all fine tables exist
    let t_precompute2 = Instant::now();
    for &theta in &new_fine {
        ensure_table(&mut ctx, theta);
    }
    let precompute_time2 = t_precompute2.elapsed().as_secs_f64();
    if precompute_time2 > 1.0 {
        println!("  Precomputation: {:.1}s\n", precompute_time2);
    }

    // Header
    print!("  {:>7}", "θ");
    for &(name, _) in PERCENTILES {
        print!(" {:>6}", name);
    }
    print!(" {:>8}", "mean");
    println!();
    println!("  {}", "─".repeat(7 + PERCENTILES.len() * 7 + 9));

    let t_phase2 = Instant::now();
    for &theta in &new_fine {
        if let Some(r) = simulate_theta(&mut ctx, theta, args.games_fine, args.seed, 2) {
            print!("  {:>+7.4}", r.theta);
            for &p in &r.percentiles {
                print!(" {:>6}", p);
            }
            print!(" {:>8.2}", r.mean);
            println!();
            all_results.insert(theta_key(theta), r);
        }
    }
    println!(
        "  Phase 2 done in {:.1}s\n",
        t_phase2.elapsed().as_secs_f64()
    );

    // ── Final results ──────────────────────────────────────────────────────

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  PEAK PERCENTILE VALUES (best from all phases)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!(
        "  {:<6} {:>+8} {:>6} {:>8} {:>6}",
        "Pctile", "θ*", "Value", "Mean@θ*", "Phase"
    );
    println!("  {}", "─".repeat(40));

    for (pi, &(name, _)) in PERCENTILES.iter().enumerate() {
        let best = all_results
            .values()
            .max_by(|a, b| {
                a.percentiles[pi].cmp(&b.percentiles[pi]).then_with(|| {
                    // Tie-break: prefer phase 2, then closer to 0
                    b.phase
                        .cmp(&a.phase)
                        .then_with(|| a.theta.abs().partial_cmp(&b.theta.abs()).unwrap())
                })
            })
            .unwrap();
        println!(
            "  {:<6} {:>+8.4} {:>6} {:>8.2} {:>4}",
            name, best.theta, best.percentiles[pi], best.mean, best.phase
        );
    }
    println!();

    // ── CSV output ──────────────────────────────────────────────────────────

    if let Some(ref output_dir) = args.output {
        fs::create_dir_all(output_dir).unwrap_or_else(|e| {
            eprintln!("Failed to create output directory: {}", e);
            std::process::exit(1);
        });

        // Full sweep data
        let sweep_path = format!("{}/percentile_sweep.csv", output_dir);
        let mut f = std::io::BufWriter::new(fs::File::create(&sweep_path).unwrap());
        write!(f, "theta,phase,games,mean,std").unwrap();
        for &(name, _) in PERCENTILES {
            write!(f, ",{}", name).unwrap();
        }
        writeln!(f).unwrap();

        for r in all_results.values() {
            write!(
                f,
                "{:.4},{},{},{:.4},{:.4}",
                r.theta, r.phase, r.num_games, r.mean, r.std_dev
            )
            .unwrap();
            for &p in &r.percentiles {
                write!(f, ",{}", p).unwrap();
            }
            writeln!(f).unwrap();
        }
        drop(f);
        println!("  Wrote {}", sweep_path);

        // Peaks summary
        let peaks_path = format!("{}/percentile_peaks.csv", output_dir);
        let mut f = std::io::BufWriter::new(fs::File::create(&peaks_path).unwrap());
        writeln!(f, "percentile,theta_star,value,mean_at_theta,phase").unwrap();
        for (pi, &(name, _)) in PERCENTILES.iter().enumerate() {
            let best = all_results
                .values()
                .max_by(|a, b| {
                    a.percentiles[pi]
                        .cmp(&b.percentiles[pi])
                        .then_with(|| b.phase.cmp(&a.phase))
                })
                .unwrap();
            writeln!(
                f,
                "{},{:.4},{},{:.4},{}",
                name, best.theta, best.percentiles[pi], best.mean, best.phase
            )
            .unwrap();
        }
        drop(f);
        println!("  Wrote {}", peaks_path);
    }

    let total = t0.elapsed().as_secs_f64();
    println!(
        "\n  Total: {:.1}s ({} θ values, {} total games)",
        total,
        all_results.len(),
        all_results
            .values()
            .map(|r| r.num_games as u64)
            .sum::<u64>()
    );
}
