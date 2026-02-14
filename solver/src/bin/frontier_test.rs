//! Frontier test: compare state-dependent θ(s) policies against the constant-θ Pareto frontier.
//!
//! Runs all adaptive policies and a set of constant-θ baselines at N games each,
//! then prints a table showing (mean, σ) for each and whether any adaptive policy
//! beats the interpolated constant-θ frontier.
//!
//! With `--output DIR`, writes:
//!   - `frontier_results.csv` — all policies with stats and frontier comparison
//!   - `frontier_scores_<name>.csv` — raw sorted scores per policy (for CDFs/histograms)

use std::fs;
use std::io::Write;
use std::time::Instant;

use yatzy::phase0_tables;
use yatzy::simulation::adaptive::{
    make_policy, simulate_batch_adaptive, ThetaTable, POLICY_CONFIGS,
};
use yatzy::storage::{load_state_values_standalone, state_file_path};
use yatzy::types::YatzyContext;

/// Constant-θ values to simulate as baselines.
const BASELINE_THETAS: &[f32] = &[0.0, 0.03, 0.05, 0.07, 0.10, 0.15];

/// Result for one policy or baseline run.
struct RunResult {
    name: String,
    kind: &'static str, // "baseline" or "adaptive"
    theta: Option<f32>,
    mean: f64,
    std_dev: f64,
    p1: i32,
    p5: i32,
    p10: i32,
    p25: i32,
    p50: i32,
    p75: i32,
    p90: i32,
    p95: i32,
    p99: i32,
    min: i32,
    max: i32,
    frontier_mean: Option<f64>,
    delta_mu: Option<f64>,
    scores: Vec<i32>,
}

fn compute_stats(scores: &[i32]) -> RunResult {
    let n = scores.len() as f64;
    let sum: f64 = scores.iter().map(|&s| s as f64).sum();
    let mean = sum / n;
    let var: f64 = scores
        .iter()
        .map(|&s| (s as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    let std_dev = var.sqrt();

    let pct = |p: f64| scores[(scores.len() as f64 * p) as usize];

    RunResult {
        name: String::new(),
        kind: "baseline",
        theta: None,
        mean,
        std_dev,
        p1: pct(0.01),
        p5: pct(0.05),
        p10: pct(0.10),
        p25: pct(0.25),
        p50: scores[scores.len() / 2],
        p75: pct(0.75),
        p90: pct(0.90),
        p95: pct(0.95),
        p99: pct(0.99),
        min: scores[0],
        max: scores[scores.len() - 1],
        frontier_mean: None,
        delta_mu: None,
        scores: scores.to_vec(),
    }
}

struct Args {
    num_games: usize,
    seed: u64,
    output: Option<String>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 1_000_000usize;
    let mut seed = 42u64;
    let mut output: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                i += 1;
                if i < args.len() {
                    num_games = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid --games value: {}", args[i]);
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
                println!("Usage: yatzy-frontier-test [--games N] [--seed S] [--output DIR]");
                println!();
                println!("Compare adaptive θ(s) policies against constant-θ Pareto frontier.");
                println!("  --games N     Number of games per policy (default: 1000000)");
                println!("  --seed S      RNG seed (default: 42)");
                println!("  --output DIR  Write CSV results to DIR");
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
        num_games,
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
    let num_games = args.num_games;
    let seed = args.seed;

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
    println!("  Frontier Test: State-Dependent θ(s) vs Constant-θ Pareto Frontier");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Games per policy: {}", num_games);
    println!("  Seed: {}", seed);
    println!("  Threads: {}", num_threads);
    if let Some(ref dir) = args.output {
        println!("  Output: {}", dir);
    }
    println!();

    // ── Setup ────────────────────────────────────────────────────────────────

    let t0 = Instant::now();
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);
    println!(
        "  Phase 0 tables: {:.1} ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Collect all unique θ values needed and verify they exist
    let mut all_thetas: Vec<f32> = BASELINE_THETAS.to_vec();
    for pc in POLICY_CONFIGS.iter() {
        for &t in pc.thetas {
            if !all_thetas.iter().any(|&x| (x - t).abs() < 1e-6) {
                all_thetas.push(t);
            }
        }
    }
    all_thetas.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!(
        "  Checking {} θ tables: {:?}",
        all_thetas.len(),
        all_thetas
    );
    for &theta in &all_thetas {
        let file = state_file_path(theta);
        if !std::path::Path::new(&file).exists() {
            eprintln!("  MISSING: θ={:.3} ({})", theta, file);
            eprintln!("  Run: yatzy-precompute --theta {}", theta);
            std::process::exit(1);
        }
    }
    println!("  All tables present.");
    println!();

    let mut results: Vec<RunResult> = Vec::new();

    // ── Constant-θ baselines ─────────────────────────────────────────────────

    println!("── Constant-θ Baselines ──────────────────────────────────────────");
    for &theta in BASELINE_THETAS {
        let single_table = vec![ThetaTable {
            theta,
            sv: load_state_values_standalone(&state_file_path(theta)).unwrap(),
            minimize: theta < 0.0,
        }];

        struct AlwaysFixed;
        impl yatzy::simulation::adaptive::AdaptivePolicy for AlwaysFixed {
            fn name(&self) -> &str {
                "fixed"
            }
            fn select_theta_index(&self, _: i32, _: i32, _: usize) -> usize {
                0
            }
        }
        let policy = AlwaysFixed;

        let t_sim = Instant::now();
        let scores = simulate_batch_adaptive(&ctx, &single_table, &policy, num_games, seed);
        let elapsed = t_sim.elapsed();

        let mut r = compute_stats(&scores);
        r.name = if theta == 0.0 {
            "ev".to_string()
        } else {
            format!("theta_{:.3}", theta)
        };
        r.kind = "baseline";
        r.theta = Some(theta);
        println!(
            "  {:<18} mean={:.2}  σ={:.1}  p5={}  p95={}  ({:.0} ms)",
            r.name,
            r.mean,
            r.std_dev,
            r.p5,
            r.p95,
            elapsed.as_secs_f64() * 1000.0
        );
        results.push(r);
    }
    println!();

    // ── Adaptive policies ────────────────────────────────────────────────────

    println!("── Adaptive Policies ─────────────────────────────────────────────");
    for pc in POLICY_CONFIGS.iter() {
        if pc.name == "always-ev" {
            continue; // redundant with θ=0 baseline
        }

        let mut policy_tables: Vec<ThetaTable> = Vec::new();
        for &t in pc.thetas {
            let sv = load_state_values_standalone(&state_file_path(t)).unwrap();
            policy_tables.push(ThetaTable {
                theta: t,
                sv,
                minimize: t < 0.0,
            });
        }

        let policy = make_policy(pc.name, &policy_tables).unwrap();

        let t_sim = Instant::now();
        let scores =
            simulate_batch_adaptive(&ctx, &policy_tables, policy.as_ref(), num_games, seed);
        let elapsed = t_sim.elapsed();

        let mut r = compute_stats(&scores);
        r.name = pc.name.to_string();
        r.kind = "adaptive";
        r.theta = None;
        println!(
            "  {:<18} mean={:.2}  σ={:.1}  p5={}  p95={}  ({:.0} ms)",
            pc.name,
            r.mean,
            r.std_dev,
            r.p5,
            r.p95,
            elapsed.as_secs_f64() * 1000.0
        );
        results.push(r);
    }
    println!();

    // ── Analysis: compare against frontier ───────────────────────────────────

    // Build constant-θ frontier points (from baseline results)
    let baseline_points: Vec<(f64, f64)> = results
        .iter()
        .filter(|r| r.kind == "baseline")
        .map(|r| (r.std_dev, r.mean))
        .collect();

    // Interpolate frontier: given a target σ, find the expected mean on the frontier
    let interpolate_frontier = |target_sigma: f64| -> Option<f64> {
        let mut sorted = baseline_points.clone();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        if target_sigma < sorted.first()?.0 || target_sigma > sorted.last()?.0 {
            return None;
        }

        for i in 0..sorted.len() - 1 {
            let (s0, m0) = sorted[i];
            let (s1, m1) = sorted[i + 1];
            if target_sigma >= s0 && target_sigma <= s1 {
                let t = (target_sigma - s0) / (s1 - s0);
                return Some(m0 + t * (m1 - m0));
            }
        }
        None
    };

    // Compute frontier comparison for adaptive policies
    for r in results.iter_mut() {
        if r.kind == "adaptive" {
            if let Some(fm) = interpolate_frontier(r.std_dev) {
                r.frontier_mean = Some(fm);
                r.delta_mu = Some(r.mean - fm);
            }
        }
    }

    // ── Print summary ────────────────────────────────────────────────────────

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RESULTS TABLE");
    println!("═══════════════════════════════════════════════════════════════════");
    println!(
        "  {:<18} {:>8} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "Policy", "Mean", "σ", "p5", "p50", "p95", "p99"
    );
    println!("  {}", "─".repeat(62));
    for r in &results {
        println!(
            "  {:<18} {:>8.2} {:>6.1} {:>6} {:>6} {:>6} {:>6}",
            r.name, r.mean, r.std_dev, r.p5, r.p50, r.p95, r.p99
        );
    }
    println!();

    println!("── Frontier Comparison ────────────────────────────────────────────");
    println!(
        "  {:<18} {:>6} {:>8} {:>10} {:>10} {:>8}",
        "Policy", "σ", "Mean", "Frontier μ", "Δμ", "Verdict"
    );
    println!("  {}", "─".repeat(66));

    let se = results
        .iter()
        .find(|r| r.kind == "baseline" && r.theta == Some(0.0))
        .map(|r| r.std_dev / (num_games as f64).sqrt())
        .unwrap_or(0.05);

    let mut any_beat = false;
    for r in &results {
        if r.kind != "adaptive" {
            continue;
        }
        if let (Some(fm), Some(delta)) = (r.frontier_mean, r.delta_mu) {
            let verdict = if delta > 1.0 {
                any_beat = true;
                "BEATS"
            } else if delta > 0.0 {
                "above"
            } else if delta > -1.0 {
                "on frontier"
            } else {
                "below"
            };
            println!(
                "  {:<18} {:>6.1} {:>8.2} {:>10.2} {:>+10.2} {:>8}",
                r.name, r.std_dev, r.mean, fm, delta, verdict
            );
        } else {
            println!(
                "  {:<18} {:>6.1} {:>8.2} {:>10} {:>10} {:>8}",
                r.name, r.std_dev, r.mean, "N/A", "N/A", "outside"
            );
        }
    }
    println!();

    println!("  SE(mean) = {:.3} at {} games", se, num_games);
    println!("  Threshold: Δμ >= 1.0 to declare H₁ (frontier beaten)");
    println!();

    if any_beat {
        println!("  *** H₁ SUPPORTED: At least one adaptive policy beats the frontier! ***");
    } else {
        println!("  H₀ holds: No adaptive policy significantly beats the constant-θ frontier.");
        println!("  The frontier appears tight for single-player Yatzy.");
    }
    println!();

    // ── Write CSV output ─────────────────────────────────────────────────────

    if let Some(ref output_dir) = args.output {
        fs::create_dir_all(output_dir).unwrap_or_else(|e| {
            eprintln!("Failed to create output directory: {}", e);
            std::process::exit(1);
        });

        // 1. Summary CSV
        let summary_path = format!("{}/frontier_results.csv", output_dir);
        let mut f = std::io::BufWriter::new(fs::File::create(&summary_path).unwrap());
        writeln!(
            f,
            "policy,kind,theta,mean,std,min,p1,p5,p10,p25,p50,p75,p90,p95,p99,max,frontier_mean,delta_mu,games,seed"
        )
        .unwrap();
        for r in &results {
            writeln!(
                f,
                "{},{},{},{:.4},{:.4},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                r.name,
                r.kind,
                r.theta.map_or("".to_string(), |t| format!("{:.3}", t)),
                r.mean,
                r.std_dev,
                r.min,
                r.p1,
                r.p5,
                r.p10,
                r.p25,
                r.p50,
                r.p75,
                r.p90,
                r.p95,
                r.p99,
                r.max,
                r.frontier_mean
                    .map_or("".to_string(), |v| format!("{:.4}", v)),
                r.delta_mu
                    .map_or("".to_string(), |v| format!("{:.4}", v)),
                num_games,
                seed,
            )
            .unwrap();
        }
        drop(f);
        println!("  Wrote {}", summary_path);

        // 2. Per-policy score distributions
        for r in &results {
            let safe_name = r.name.replace(':', "_").replace(' ', "_");
            let scores_path = format!("{}/frontier_scores_{}.csv", output_dir, safe_name);
            let mut f = std::io::BufWriter::new(fs::File::create(&scores_path).unwrap());
            writeln!(f, "score").unwrap();
            for &s in &r.scores {
                writeln!(f, "{}", s).unwrap();
            }
            drop(f);
        }
        println!(
            "  Wrote {} score distribution files to {}/",
            results.len(),
            output_dir
        );
    }
}
