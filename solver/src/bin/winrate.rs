//! Head-to-head win rate experiment: does a constant-θ strategy beat θ=0?
//!
//! Simulates games under θ=0 (baseline) and each challenger θ, builds PMFs from
//! the score distributions, and computes exact win rates via O(n²) convolution:
//!   win_rate = Σ_x Σ_y P_chal[x] * P_opp[y] * I(x > y)
//!
//! Also decomposes wins by opponent score band to identify where the advantage comes from.
//!
//! With `--output DIR`, writes:
//!   - `winrate_results.csv` — per-θ win/draw/loss rates + distribution stats
//!   - `winrate_conditional.csv` — win rate by opponent score band

use std::fs;
use std::io::Write;
use std::time::Instant;

use yatzy::phase0_tables;
use yatzy::simulation::engine::simulate_batch;
use yatzy::storage::{load_all_state_values, state_file_path};
use yatzy::types::YatzyContext;

/// Score range [0, PMF_SIZE). Yatzy max is 374.
const PMF_SIZE: usize = 500;

/// Challenger θ values: dense in [-0.10, +0.10], plus out-of-range controls.
const CHALLENGER_THETAS: &[f32] = &[
    -0.20, -0.15, -0.10, -0.07, -0.05, -0.04, -0.03, -0.02, -0.015, -0.01, -0.005, 0.005, 0.01,
    0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.15, 0.20,
];

/// Opponent score bands for conditional win rate analysis.
const BANDS: &[(usize, usize)] = &[
    (0, 200),
    (200, 220),
    (220, 240),
    (240, 250),
    (250, 260),
    (260, 280),
    (280, 500),
];

/// Build a PMF (histogram of counts) from a sorted score vector.
fn build_pmf(scores: &[i32]) -> [u64; PMF_SIZE] {
    let mut pmf = [0u64; PMF_SIZE];
    for &s in scores {
        let idx = s as usize;
        if idx < PMF_SIZE {
            pmf[idx] += 1;
        }
    }
    pmf
}

/// Compute exact win/draw/loss rates from two PMFs.
/// Returns (win_rate, draw_rate, loss_rate) as fractions in [0, 1].
fn compute_win_rate(chal: &[u64; PMF_SIZE], opp: &[u64; PMF_SIZE]) -> (f64, f64, f64) {
    let n_chal: u64 = chal.iter().sum();
    let n_opp: u64 = opp.iter().sum();
    let total = n_chal as f64 * n_opp as f64;

    let mut wins: f64 = 0.0;
    let mut draws: f64 = 0.0;

    // For each challenger score x, count how many opponent scores y < x (wins)
    // and y == x (draws). Use cumulative sum of opponent PMF for efficiency.
    let mut opp_cum = 0u64; // Σ opp[0..x]
    for x in 0..PMF_SIZE {
        if chal[x] == 0 {
            opp_cum += opp[x];
            continue;
        }
        // Wins: challenger at x, opponent at any y < x (sum of opp[0..x-1] = opp_cum before adding opp[x])
        wins += chal[x] as f64 * opp_cum as f64;
        // Draws: both at x
        draws += chal[x] as f64 * opp[x] as f64;
        opp_cum += opp[x];
    }

    (
        wins / total,
        draws / total,
        1.0 - wins / total - draws / total,
    )
}

/// Result for one opponent score band.
struct BandResult {
    lo: usize,
    hi: usize,
    win_rate: f64,
    n_opponent: u64,
}

/// Compute conditional win rates: for each band [lo, hi), what is the challenger's
/// win rate when the opponent scores in that band?
fn conditional_win_rates(
    chal: &[u64; PMF_SIZE],
    opp: &[u64; PMF_SIZE],
    bands: &[(usize, usize)],
) -> Vec<BandResult> {
    let n_chal: u64 = chal.iter().sum();

    // Precompute challenger CDF: chal_cum[x] = Σ chal[0..x] (exclusive of x)
    let mut chal_cum = vec![0u64; PMF_SIZE + 1];
    for x in 0..PMF_SIZE {
        chal_cum[x + 1] = chal_cum[x] + chal[x];
    }

    bands
        .iter()
        .map(|&(lo, hi)| {
            let n_opp_band: u64 = opp[lo..hi.min(PMF_SIZE)].iter().sum();
            if n_opp_band == 0 || n_chal == 0 {
                return BandResult {
                    lo,
                    hi,
                    win_rate: 0.0,
                    n_opponent: 0,
                };
            }

            // For each opponent score y in [lo, hi), challenger wins if x > y.
            // Count of challenger scores > y = n_chal - chal_cum[y+1].
            let mut wins: f64 = 0.0;
            for y in lo..hi.min(PMF_SIZE) {
                if opp[y] == 0 {
                    continue;
                }
                let chal_above = n_chal - chal_cum[y + 1];
                wins += opp[y] as f64 * chal_above as f64;
            }

            let total = n_chal as f64 * n_opp_band as f64;
            BandResult {
                lo,
                hi,
                win_rate: wins / total,
                n_opponent: n_opp_band,
            }
        })
        .collect()
}

/// Distribution stats computed from sorted scores.
struct DistStats {
    mean: f64,
    std_dev: f64,
    p5: i32,
    p50: i32,
    p95: i32,
    p99: i32,
}

fn compute_dist_stats(scores: &[i32]) -> DistStats {
    let n = scores.len() as f64;
    let sum: f64 = scores.iter().map(|&s| s as f64).sum();
    let mean = sum / n;
    let var: f64 = scores
        .iter()
        .map(|&s| (s as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    let pct = |p: f64| scores[(scores.len() as f64 * p) as usize];
    DistStats {
        mean,
        std_dev: var.sqrt(),
        p5: pct(0.05),
        p50: scores[scores.len() / 2],
        p95: pct(0.95),
        p99: pct(0.99),
    }
}

struct Args {
    num_games: usize,
    seed: u64,
    output: Option<String>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 10_000_000usize;
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
                println!("Usage: yatzy-winrate [--games N] [--seed S] [--output DIR]");
                println!();
                println!("Head-to-head win rate: constant-θ vs θ=0 (EV-optimal).");
                println!("  --games N     Games per θ (default: 10000000)");
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
    println!("  Head-to-Head Win Rate: Constant-θ vs θ=0 (EV-Optimal)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Games per θ: {:>12}", num_games);
    println!("  Seed:        {:>12}", seed);
    println!("  Threads:     {:>12}", num_threads);
    println!("  Challengers: {:>12}", CHALLENGER_THETAS.len());
    if let Some(ref dir) = args.output {
        println!("  Output:      {}", dir);
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

    // Check all required θ table files
    let mut all_thetas: Vec<f32> = vec![0.0];
    for &t in CHALLENGER_THETAS {
        all_thetas.push(t);
    }

    println!("  Checking {} θ tables...", all_thetas.len());
    let mut missing = Vec::new();
    for &theta in &all_thetas {
        let file = state_file_path(theta);
        if !std::path::Path::new(&file).exists() {
            missing.push(theta);
        }
    }

    if !missing.is_empty() {
        eprintln!("  MISSING θ tables:");
        for &t in &missing {
            eprintln!("    θ={:.3} ({})", t, state_file_path(t));
            eprintln!("    Run: yatzy-precompute --theta {}", t);
        }
        eprintln!();
        eprintln!(
            "  {} of {} tables missing. Continuing with available thetas.",
            missing.len(),
            all_thetas.len()
        );
    }

    // Filter to available thetas
    let available_challengers: Vec<f32> = CHALLENGER_THETAS
        .iter()
        .copied()
        .filter(|t| !missing.contains(t))
        .collect();

    if available_challengers.is_empty() {
        eprintln!("No challenger θ tables available. Nothing to do.");
        std::process::exit(1);
    }

    if missing.contains(&0.0) {
        eprintln!("Baseline θ=0 table missing. Cannot continue.");
        std::process::exit(1);
    }

    println!("  {} challengers available.", available_challengers.len());
    println!();

    // ── Baseline: θ=0 ───────────────────────────────────────────────────────

    println!("── Baseline (θ=0) ────────────────────────────────────────────────");
    let t_sim = Instant::now();

    // Load θ=0 state values
    let baseline_file = state_file_path(0.0);
    if !load_all_state_values(&mut ctx, &baseline_file) {
        eprintln!("Failed to load baseline state values: {}", baseline_file);
        std::process::exit(1);
    }
    ctx.theta = 0.0;

    let baseline_result = simulate_batch(&ctx, num_games, seed);
    let baseline_stats = compute_dist_stats(&baseline_result.scores);
    let baseline_pmf = build_pmf(&baseline_result.scores);
    println!(
        "  θ=0.000  mean={:.2}  σ={:.1}  p5={}  p50={}  p95={}  ({:.1}s)",
        baseline_stats.mean,
        baseline_stats.std_dev,
        baseline_stats.p5,
        baseline_stats.p50,
        baseline_stats.p95,
        t_sim.elapsed().as_secs_f64()
    );
    println!();

    // ── Challengers ─────────────────────────────────────────────────────────

    struct ChalResult {
        theta: f32,
        win_rate: f64,
        draw_rate: f64,
        loss_rate: f64,
        stats: DistStats,
        band_results: Vec<BandResult>,
    }

    println!("── Challengers ───────────────────────────────────────────────────");
    println!(
        "  {:>7} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6}",
        "θ", "Win%", "Draw%", "Loss%", "Mean", "p5", "p95"
    );
    println!("  {}", "─".repeat(58));

    let mut results: Vec<ChalResult> = Vec::new();

    for &theta in &available_challengers {
        let t_chal = Instant::now();

        // Load challenger state values
        let chal_file = state_file_path(theta);
        if !load_all_state_values(&mut ctx, &chal_file) {
            eprintln!("  Failed to load θ={:.3}: {}", theta, chal_file);
            continue;
        }
        ctx.theta = theta;

        let chal_result = simulate_batch(&ctx, num_games, seed);
        let chal_stats = compute_dist_stats(&chal_result.scores);
        let chal_pmf = build_pmf(&chal_result.scores);

        let (win, draw, loss) = compute_win_rate(&chal_pmf, &baseline_pmf);
        let bands = conditional_win_rates(&chal_pmf, &baseline_pmf, BANDS);

        let elapsed = t_chal.elapsed().as_secs_f64();
        let win_marker = if win > 0.51 {
            " **"
        } else if win > 0.50 {
            " *"
        } else {
            ""
        };

        println!(
            "  {:>+7.3} {:>7.2}% {:>7.2}% {:>7.2}% {:>8.2} {:>6} {:>6}  ({:.1}s){}",
            theta,
            win * 100.0,
            draw * 100.0,
            loss * 100.0,
            chal_stats.mean,
            chal_stats.p5,
            chal_stats.p95,
            elapsed,
            win_marker
        );

        results.push(ChalResult {
            theta,
            win_rate: win,
            draw_rate: draw,
            loss_rate: loss,
            stats: chal_stats,
            band_results: bands,
        });
    }
    println!();

    // ── Summary ─────────────────────────────────────────────────────────────

    // Find best win rate
    let best = results
        .iter()
        .max_by(|a, b| a.win_rate.partial_cmp(&b.win_rate).unwrap());

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RESULTS");
    println!("═══════════════════════════════════════════════════════════════════");
    println!(
        "  Baseline: θ=0, mean={:.2}, σ={:.1}",
        baseline_stats.mean, baseline_stats.std_dev
    );
    println!();

    if let Some(best) = best {
        println!(
            "  Best win rate: θ={:+.3} → {:.2}% (draw {:.2}%, loss {:.2}%)",
            best.theta,
            best.win_rate * 100.0,
            best.draw_rate * 100.0,
            best.loss_rate * 100.0
        );
        println!(
            "  Best θ stats:  mean={:.2}, σ={:.1}, p5={}, p95={}",
            best.stats.mean, best.stats.std_dev, best.stats.p5, best.stats.p95
        );
        println!(
            "  Mean cost:     {:.2} points",
            baseline_stats.mean - best.stats.mean
        );
        println!();

        let se = (best.win_rate * (1.0 - best.win_rate)
            / (num_games as f64 * num_games as f64).sqrt())
        .sqrt();
        let margin = best.win_rate - 0.5;
        let z = if se > 0.0 { margin / se } else { f64::INFINITY };

        println!("  SE(win_rate) ≈ {:.4}%", se * 100.0);
        println!("  Margin over 50%: {:.2}pp  ({:.1}σ)", margin * 100.0, z);
        println!();

        if best.win_rate > 0.51 {
            println!(
                "  *** H₁ SUPPORTED: θ={:+.3} achieves >{:.0}% win rate! ***",
                best.theta, 51.0
            );
        } else if best.win_rate > 0.50 {
            println!(
                "  H₁ marginal: best win rate {:.2}% (between 50-51%).",
                best.win_rate * 100.0
            );
        } else {
            println!("  H₀ holds: no θ achieves >50% win rate.");
        }
        println!();

        // Conditional breakdown for best θ
        println!(
            "── Conditional Win Rate (θ={:+.3}) by Opponent Band ──",
            best.theta
        );
        println!("  {:>10} {:>10} {:>8}", "Opp Band", "N(opp)", "Win%");
        println!("  {}", "─".repeat(32));
        for br in &best.band_results {
            if br.n_opponent > 0 {
                println!(
                    "  {:>4}-{:<4} {:>10} {:>7.2}%",
                    br.lo,
                    br.hi,
                    br.n_opponent,
                    br.win_rate * 100.0
                );
            }
        }
    }
    println!();

    // ── CSV output ──────────────────────────────────────────────────────────

    if let Some(ref output_dir) = args.output {
        fs::create_dir_all(output_dir).unwrap_or_else(|e| {
            eprintln!("Failed to create output directory: {}", e);
            std::process::exit(1);
        });

        // 1. Main results CSV
        let results_path = format!("{}/winrate_results.csv", output_dir);
        let mut f = std::io::BufWriter::new(fs::File::create(&results_path).unwrap());
        writeln!(
            f,
            "theta,win_rate,draw_rate,loss_rate,mean,std,p5,p50,p95,p99,games"
        )
        .unwrap();
        // Include baseline as θ=0
        writeln!(
            f,
            "0.000,0.5000,,,{:.4},{:.4},{},{},{},{},{}",
            baseline_stats.mean,
            baseline_stats.std_dev,
            baseline_stats.p5,
            baseline_stats.p50,
            baseline_stats.p95,
            baseline_stats.p99,
            num_games,
        )
        .unwrap();
        for r in &results {
            writeln!(
                f,
                "{:.3},{:.6},{:.6},{:.6},{:.4},{:.4},{},{},{},{},{}",
                r.theta,
                r.win_rate,
                r.draw_rate,
                r.loss_rate,
                r.stats.mean,
                r.stats.std_dev,
                r.stats.p5,
                r.stats.p50,
                r.stats.p95,
                r.stats.p99,
                num_games,
            )
            .unwrap();
        }
        drop(f);
        println!("  Wrote {}", results_path);

        // 2. Conditional win rates CSV
        let cond_path = format!("{}/winrate_conditional.csv", output_dir);
        let mut f = std::io::BufWriter::new(fs::File::create(&cond_path).unwrap());
        writeln!(f, "theta,band_lo,band_hi,win_rate,n_opponent_in_band").unwrap();
        for r in &results {
            for br in &r.band_results {
                writeln!(
                    f,
                    "{:.3},{},{},{:.6},{}",
                    r.theta, br.lo, br.hi, br.win_rate, br.n_opponent
                )
                .unwrap();
            }
        }
        drop(f);
        println!("  Wrote {}", cond_path);
    }

    let total_elapsed = t0.elapsed().as_secs_f64();
    println!();
    println!(
        "  Total: {:.1}s ({} simulations × {} games = {} total games)",
        total_elapsed,
        available_challengers.len() + 1,
        num_games,
        (available_challengers.len() + 1) as u64 * num_games as u64
    );
}
