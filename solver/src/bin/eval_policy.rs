//! Evaluate a skill ladder (JSON rule set) via Monte Carlo simulation.
//!
//! Simulates N games using the rule-based policy and compares against the
//! EV-optimal oracle (245.87 expected score).

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::time::Instant;

use yatzy::phase0_tables;
use yatzy::rosetta::policy::{simulate_game_category_rules_only, simulate_game_with_rules, SkillLadder};
use yatzy::storage::{load_all_state_values, state_file_path};
use yatzy::types::YatzyContext;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 1_000_000usize;
    let mut seed = 42u64;
    let mut json_path = String::from("outputs/rosetta/skill_ladder.json");
    let mut output_path: Option<String> = None;
    let mut category_only = false;

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
                output_path = Some(args[i].clone());
            }
            "--category-only" => {
                category_only = true;
            }
            "--help" | "-h" => {
                println!("Usage: yatzy-eval-policy [OPTIONS] [SKILL_LADDER_JSON]");
                println!("  --games N        Number of games to simulate (default: 1000000)");
                println!("  --seed S         Random seed (default: 42)");
                println!("  --output PATH    Output JSON path (default: outputs/rosetta/eval_results.json)");
                println!("  --category-only  Use optimal rerolls, only apply rules for category selection");
                println!("  POSITIONAL       Path to skill_ladder.json (default: outputs/rosetta/skill_ladder.json)");
                std::process::exit(0);
            }
            other => {
                if other.starts_with('-') {
                    eprintln!("Unknown argument: {}", other);
                    std::process::exit(1);
                }
                json_path = other.to_string();
            }
        }
        i += 1;
    }

    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    if std::env::set_current_dir(&base_path).is_err() {
        eprintln!("Failed to change directory to {}", base_path);
        std::process::exit(1);
    }

    let output_path = output_path.unwrap_or_else(|| "outputs/rosetta/eval_results.json".to_string());

    let num_threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    let total_start = Instant::now();

    // Load skill ladder
    println!("Loading skill ladder from {}...", json_path);
    let json_str = std::fs::read_to_string(&json_path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {}", json_path, e);
        std::process::exit(1);
    });
    let json_val: serde_json::Value = serde_json::from_str(&json_str).unwrap_or_else(|e| {
        eprintln!("Failed to parse JSON: {}", e);
        std::process::exit(1);
    });
    let ladder = SkillLadder::from_json(&json_val).unwrap_or_else(|e| {
        eprintln!("Failed to parse skill ladder: {}", e);
        std::process::exit(1);
    });
    println!(
        "  {} category rules, {} reroll1 rules, {} reroll2 rules",
        ladder.category_rules.len(),
        ladder.reroll1_rules.len(),
        ladder.reroll2_rules.len(),
    );

    // Load context (needed for precomputed_scores in category selection)
    println!("Loading lookup tables...");
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    // We don't strictly need state values for rule-based play,
    // but load them for oracle EV comparison
    let file0 = state_file_path(0.0);
    let has_sv = load_all_state_values(&mut ctx, &file0);
    if has_sv {
        let oracle_ev = ctx.state_values.as_slice()[0]; // V*(start state)
        println!("  Oracle EV: {:.2}", oracle_ev);
    }

    // Simulate
    println!(
        "Simulating {} games (seed={}, threads={})...",
        num_games, seed, num_threads
    );
    let sim_start = Instant::now();

    if category_only {
        println!("  Mode: category-only (optimal rerolls + rule-based category)");
    } else {
        println!("  Mode: full rules (rule-based rerolls + category)");
    }

    let scores: Vec<i32> = (0..num_games)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
            if category_only {
                simulate_game_category_rules_only(&ctx, &ladder, &mut rng)
            } else {
                simulate_game_with_rules(&ctx, &ladder, &mut rng)
            }
        })
        .collect();

    let sim_time = sim_start.elapsed().as_secs_f64();
    println!(
        "  Done in {:.1}s ({:.0} games/s)",
        sim_time,
        num_games as f64 / sim_time
    );

    // Statistics
    let n = scores.len();
    let sum: f64 = scores.iter().map(|&s| s as f64).sum();
    let mean = sum / n as f64;
    let var: f64 = scores.iter().map(|&s| (s as f64 - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = var.sqrt();
    let min = *scores.iter().min().unwrap();
    let max = *scores.iter().max().unwrap();

    let mut sorted = scores.clone();
    sorted.sort();
    let p5 = sorted[n * 5 / 100];
    let p25 = sorted[n * 25 / 100];
    let p50 = sorted[n / 2];
    let p75 = sorted[n * 75 / 100];
    let p95 = sorted[n * 95 / 100];

    let oracle_ev = if has_sv {
        ctx.state_values.as_slice()[0] as f64
    } else {
        245.87 // known θ=0 oracle EV
    };

    println!("\n=== Skill Ladder Evaluation ===");
    println!("  Games:      {:>10}", n);
    println!("  Policy EV:  {:>10.2}", mean);
    println!("  Oracle EV:  {:>10.2}", oracle_ev);
    println!("  EV Gap:     {:>10.2}", oracle_ev - mean);
    println!("  Std Dev:    {:>10.2}", std_dev);
    println!("  Min:        {:>10}", min);
    println!("  p5:         {:>10}", p5);
    println!("  p25:        {:>10}", p25);
    println!("  p50:        {:>10}", p50);
    println!("  p75:        {:>10}", p75);
    println!("  p95:        {:>10}", p95);
    println!("  Max:        {:>10}", max);

    let total_rules = ladder.category_rules.len()
        + ladder.reroll1_rules.len()
        + ladder.reroll2_rules.len();
    println!("  Total rules: {}", total_rules);

    // Write results JSON
    let results = serde_json::json!({
        "policy_ev": mean,
        "oracle_ev": oracle_ev,
        "ev_gap": oracle_ev - mean,
        "std_dev": std_dev,
        "min": min,
        "p5": p5,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p95": p95,
        "max": max,
        "num_games": n,
        "total_rules": total_rules,
        "category_rules": ladder.category_rules.len(),
        "reroll1_rules": ladder.reroll1_rules.len(),
        "reroll2_rules": ladder.reroll2_rules.len(),
    });

    let out_dir = std::path::Path::new(&output_path).parent().unwrap();
    let _ = std::fs::create_dir_all(out_dir);
    std::fs::write(&output_path, serde_json::to_string_pretty(&results).unwrap())
        .expect("Failed to write results");
    println!("\nWrote {}", output_path);
    println!("Total time: {:.1}s", total_start.elapsed().as_secs_f64());
}
