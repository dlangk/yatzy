use std::path::PathBuf;
use std::time::Instant;

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

use yatzy::constants::*;
use yatzy::dice_mechanics::{count_faces, sort_dice_set};
use yatzy::game_mechanics::{calculate_category_score, update_upper_score};
use yatzy::phase0_tables;
use yatzy::simulation::engine::simulate_game;
use yatzy::simulation::heuristic::{heuristic_pick_category, heuristic_reroll_mask};
use yatzy::simulation::multiplayer::simulate_multiplayer;
use yatzy::simulation::strategy::Strategy;
use yatzy::storage::load_all_state_values;
use yatzy::types::YatzyContext;

fn set_working_directory() -> PathBuf {
    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    let path = PathBuf::from(&base_path);
    if std::env::set_current_dir(&base_path).is_err() {
        eprintln!("Failed to change directory to {}", base_path);
        std::process::exit(1);
    }
    path
}

struct Args {
    num_games: u32,
    seed: u64,
    output: Option<String>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 100_000u32;
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
                println!("Usage: yatzy-human-baseline [OPTIONS]");
                println!();
                println!("Simulate human (heuristic) vs EV-optimal play and report statistics.");
                println!();
                println!("Options:");
                println!("  --games N    Number of games (default: 100000)");
                println!("  --seed S     RNG seed (default: 42)");
                println!("  --output DIR Save CSV results to DIR");
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

/// Roll 5 random dice and sort them.
fn roll_dice(rng: &mut SmallRng) -> [i32; 5] {
    use rand::Rng;
    let mut dice = [0i32; 5];
    for d in &mut dice {
        *d = rng.random_range(1..=6);
    }
    sort_dice_set(&mut dice);
    dice
}

/// Apply a reroll mask.
fn apply_reroll(dice: &mut [i32; 5], mask: i32, rng: &mut SmallRng) {
    use rand::Rng;
    for (i, d) in dice.iter_mut().enumerate() {
        if mask & (1 << i) != 0 {
            *d = rng.random_range(1..=6);
        }
    }
    sort_dice_set(dice);
}

/// Simulate one game using the heuristic strategy.
fn simulate_game_heuristic(rng: &mut SmallRng) -> i32 {
    let mut upper_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut total_score: i32 = 0;

    for _turn in 0..CATEGORY_COUNT {
        let mut dice = roll_dice(rng);

        // Reroll 1
        let fc1 = count_faces(&dice);
        let mask1 = heuristic_reroll_mask(&dice, &fc1, scored, upper_score);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }

        // Reroll 2
        let fc2 = count_faces(&dice);
        let mask2 = heuristic_reroll_mask(&dice, &fc2, scored, upper_score);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }

        // Pick category
        let fc_final = count_faces(&dice);
        let cat = heuristic_pick_category(&dice, &fc_final, scored, upper_score);
        let scr = calculate_category_score(&dice, cat);

        upper_score = update_upper_score(upper_score, cat, scr);
        scored |= 1 << cat;
        total_score += scr;
    }

    if upper_score >= 63 {
        total_score += 50;
    }

    total_score
}

fn percentile(scores: &[i32], p: f64) -> i32 {
    let idx = ((scores.len() as f64 - 1.0) * p / 100.0).round() as usize;
    scores[idx.min(scores.len() - 1)]
}

fn main() {
    let base_path = set_working_directory();
    let args = parse_args();

    let num_threads = std::env::var("RAYON_NUM_THREADS")
        .or_else(|_| std::env::var("OMP_NUM_THREADS"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    // Phase 0: build context
    let t0 = Instant::now();
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);
    println!("Phase 0 tables: {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0);

    // Load EV-optimal state values
    let t1 = Instant::now();
    if !load_all_state_values(&mut ctx, "data/strategy_tables/all_states.bin") {
        eprintln!("Failed to load state values. Run yatzy-precompute first.");
        std::process::exit(1);
    }
    println!("State values loaded: {:.1} ms", t1.elapsed().as_secs_f64() * 1000.0);
    println!();

    let n = args.num_games;

    // ── 1. Solo simulations ──────────────────────────────────────────────

    println!("=== Human Baseline Experiment ===");
    println!("Games: {} | Seed: {} | Threads: {}", n, args.seed, num_threads);
    println!();

    // Heuristic (human) games
    let t2 = Instant::now();
    let mut human_scores: Vec<i32> = (0..n as u64)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(args.seed.wrapping_add(i));
            simulate_game_heuristic(&mut rng)
        })
        .collect();
    let human_ms = t2.elapsed().as_secs_f64() * 1000.0;
    human_scores.sort_unstable();

    // Optimal games (same seeds for fair comparison)
    let t3 = Instant::now();
    let mut optimal_scores: Vec<i32> = (0..n as u64)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(args.seed.wrapping_add(i));
            simulate_game(&ctx, &mut rng)
        })
        .collect();
    let optimal_ms = t3.elapsed().as_secs_f64() * 1000.0;
    optimal_scores.sort_unstable();

    // Stats
    let human_mean = human_scores.iter().map(|&s| s as f64).sum::<f64>() / n as f64;
    let human_std = (human_scores
        .iter()
        .map(|&s| (s as f64 - human_mean).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();

    let optimal_mean = optimal_scores.iter().map(|&s| s as f64).sum::<f64>() / n as f64;
    let optimal_std = (optimal_scores
        .iter()
        .map(|&s| (s as f64 - optimal_mean).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();

    println!("Solo Performance:");
    println!(
        "  {:>12}  {:>7}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>8}",
        "Strategy", "Mean", "Std", "Min", "p5", "p50", "p95", "Max"
    );
    println!(
        "  {:>12}  {:>7.1}  {:>5.1}  {:>5}  {:>5}  {:>5}  {:>5}  {:>8}",
        "human",
        human_mean,
        human_std,
        human_scores[0],
        percentile(&human_scores, 5.0),
        percentile(&human_scores, 50.0),
        percentile(&human_scores, 95.0),
        human_scores[human_scores.len() - 1],
    );
    println!(
        "  {:>12}  {:>7.1}  {:>5.1}  {:>5}  {:>5}  {:>5}  {:>5}  {:>8}",
        "ev-optimal",
        optimal_mean,
        optimal_std,
        optimal_scores[0],
        percentile(&optimal_scores, 5.0),
        percentile(&optimal_scores, 50.0),
        percentile(&optimal_scores, 95.0),
        optimal_scores[optimal_scores.len() - 1],
    );
    println!();
    println!(
        "  Value of optimization: {:.1} points ({:.1}%)",
        optimal_mean - human_mean,
        (optimal_mean - human_mean) / human_mean * 100.0
    );
    println!(
        "  Simulation time: human {:.0} ms, optimal {:.0} ms",
        human_ms, optimal_ms
    );
    println!();

    // ── 2. Head-to-head ─────────────────────────────────────────────────

    println!("Head-to-Head (same dice rolls):");

    // Compare game by game (same seeds → same dice)
    let mut human_wins = 0u32;
    let mut optimal_wins = 0u32;
    let mut draws = 0u32;
    let mut margin_sum_when_human_wins = 0.0f64;
    let mut margin_sum_when_optimal_wins = 0.0f64;

    // We need to re-simulate pairwise with same RNG
    let paired: Vec<(i32, i32)> = (0..n as u64)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(args.seed.wrapping_add(i));
            let h = simulate_game_heuristic(&mut rng);
            let mut rng2 = SmallRng::seed_from_u64(args.seed.wrapping_add(i));
            let o = simulate_game(&ctx, &mut rng2);
            (h, o)
        })
        .collect();

    for &(h, o) in &paired {
        if h > o {
            human_wins += 1;
            margin_sum_when_human_wins += (h - o) as f64;
        } else if o > h {
            optimal_wins += 1;
            margin_sum_when_optimal_wins += (o - h) as f64;
        } else {
            draws += 1;
        }
    }

    println!(
        "  Human wins:   {:>7} ({:.1}%) avg margin {:.1}",
        human_wins,
        human_wins as f64 / n as f64 * 100.0,
        if human_wins > 0 {
            margin_sum_when_human_wins / human_wins as f64
        } else {
            0.0
        }
    );
    println!(
        "  Optimal wins: {:>7} ({:.1}%) avg margin {:.1}",
        optimal_wins,
        optimal_wins as f64 / n as f64 * 100.0,
        if optimal_wins > 0 {
            margin_sum_when_optimal_wins / optimal_wins as f64
        } else {
            0.0
        }
    );
    println!(
        "  Draws:        {:>7} ({:.1}%)",
        draws,
        draws as f64 / n as f64 * 100.0
    );
    println!();

    // ── 3. Multiplayer framework head-to-head ───────────────────────────

    println!("Multiplayer framework (interleaved turns, shared RNG):");

    let strategies = vec![
        Strategy::from_spec("ev", &base_path, &ctx).unwrap(),
        Strategy::from_spec("human", &base_path, &ctx).unwrap(),
    ];

    let t4 = Instant::now();
    let mp_result = simulate_multiplayer(&ctx, &strategies, n, args.seed);
    let mp_ms = t4.elapsed().as_secs_f64() * 1000.0;

    for i in 0..2 {
        println!(
            "  {:>12}: win {:.1}%, mean {:.1}, std {:.1}, margin {:.1}",
            mp_result.strategies[i],
            mp_result.win_rates[i],
            mp_result.score_means[i],
            mp_result.score_stds[i],
            mp_result.avg_margin_when_winning[i],
        );
    }
    println!(
        "  Draws: {} ({:.1}%)",
        mp_result.draws,
        mp_result.draws as f64 / n as f64 * 100.0
    );
    println!("  Elapsed: {:.0} ms", mp_ms);

    // ── 4. Save CSV ─────────────────────────────────────────────────────

    if let Some(ref output_dir) = args.output {
        std::fs::create_dir_all(output_dir).unwrap_or_else(|e| {
            eprintln!("Failed to create output directory: {}", e);
            std::process::exit(1);
        });

        let csv_path = format!("{}/human_baseline_results.csv", output_dir);
        let mut csv = String::new();
        csv.push_str("strategy,mean,std,min,p5,p50,p95,max,win_rate_vs_optimal\n");
        csv.push_str(&format!(
            "human,{:.1},{:.1},{},{},{},{},{},{:.1}\n",
            human_mean,
            human_std,
            human_scores[0],
            percentile(&human_scores, 5.0),
            percentile(&human_scores, 50.0),
            percentile(&human_scores, 95.0),
            human_scores[human_scores.len() - 1],
            human_wins as f64 / n as f64 * 100.0,
        ));
        csv.push_str(&format!(
            "ev-optimal,{:.1},{:.1},{},{},{},{},{},{:.1}\n",
            optimal_mean,
            optimal_std,
            optimal_scores[0],
            percentile(&optimal_scores, 5.0),
            percentile(&optimal_scores, 50.0),
            percentile(&optimal_scores, 95.0),
            optimal_scores[optimal_scores.len() - 1],
            optimal_wins as f64 / n as f64 * 100.0,
        ));
        std::fs::write(&csv_path, csv).unwrap();
        println!();
        println!("Results saved to {}", csv_path);
    }
}
