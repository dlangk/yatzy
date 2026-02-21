use std::time::Instant;

use yatzy::phase0_tables;
use yatzy::simulation::multiplayer::{
    aggregate_from_records, simulate_multiplayer, simulate_multiplayer_with_recording,
};
use yatzy::simulation::raw_storage::save_multiplayer_recording;
use yatzy::simulation::strategy::Strategy;
use yatzy::types::YatzyContext;

struct Args {
    strategies: Vec<String>,
    num_games: u32,
    seed: u64,
    output: Option<String>,
    record: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut strategies = Vec::new();
    let mut num_games = 10_000u32;
    let mut seed = 42u64;
    let mut output: Option<String> = None;
    let mut record = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--strategy" | "-s" => {
                i += 1;
                if i < args.len() {
                    strategies.push(args[i].clone());
                }
            }
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
            "--record" => {
                record = true;
            }
            "--help" | "-h" => {
                println!("Usage: yatzy-multiplayer [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --strategy SPEC  Add a player strategy (repeatable, min 2)");
                println!("  --games N        Number of games to simulate (default: 10000)");
                println!("  --seed S         RNG seed (default: 42)");
                println!("  --output DIR     Save JSON results to DIR");
                println!("  --record         Save per-game binary recording (requires --output, 2 players)");
                println!();
                println!("Strategy specs:");
                println!("  ev               EV-optimal (θ=0)");
                println!("  theta:0.05       Fixed θ=0.05");
                println!("  adaptive:bonus   Bonus-adaptive policy");
                println!("  adaptive:phase   Phase-based policy");
                println!("  adaptive:combined Combined policy");
                println!("  mp:trailing      Risk-seeking when trailing (threshold=15)");
                println!("  mp:trailing:20   Custom trailing threshold");
                println!("  mp:underdog      Continuous ramp: θ scales with EV deficit (default θ_max=0.05, scale=50)");
                println!("  mp:underdog:0.07 Custom θ_max");
                println!("  mp:underdog:0.05:60 Custom θ_max and scale");
                println!();
                println!("Example:");
                println!(
                    "  yatzy-multiplayer --strategy ev --strategy \"theta:0.05\" --games 100000"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                eprintln!("Run with --help for usage.");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if strategies.len() < 2 {
        eprintln!(
            "Error: need at least 2 strategies (got {}). Use --strategy SPEC.",
            strategies.len()
        );
        std::process::exit(1);
    }

    if record && output.is_none() {
        eprintln!("Error: --record requires --output DIR.");
        std::process::exit(1);
    }
    if record && strategies.len() != 2 {
        eprintln!(
            "Error: --record only supports 2 players (got {}).",
            strategies.len()
        );
        std::process::exit(1);
    }

    Args {
        strategies,
        num_games,
        seed,
        output,
        record,
    }
}

fn main() {
    let base_path = yatzy::env_config::init_base_path();
    let args = parse_args();
    let num_threads = yatzy::env_config::init_rayon_threads();

    // Phase 0: build context
    let t0 = Instant::now();
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);
    let phase0_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Phase 0 tables: {:.1} ms", phase0_ms);

    // Parse strategies (each loads its own state-value tables)
    let t1 = Instant::now();
    let mut strategies: Vec<Strategy> = Vec::with_capacity(args.strategies.len());
    for spec in &args.strategies {
        match Strategy::from_spec(spec, &base_path, &ctx) {
            Ok(s) => {
                println!("  Strategy '{}' loaded", s.name);
                strategies.push(s);
            }
            Err(e) => {
                eprintln!("Error parsing strategy '{}': {}", spec, e);
                std::process::exit(1);
            }
        }
    }
    let load_ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!("Strategy loading: {:.1} ms", load_ms);
    println!();

    // Run simulation
    let n = strategies.len();
    println!(
        "Multiplayer Yatzy: {} players \u{00d7} {} games ({} threads)",
        n, args.num_games, num_threads
    );

    let t2 = Instant::now();
    let (result, records) = if args.record {
        let recs =
            simulate_multiplayer_with_recording(&ctx, &strategies, args.num_games, args.seed);
        let names: Vec<String> = strategies.iter().map(|s| s.name.clone()).collect();
        let res = aggregate_from_records(&recs, &names);
        (res, Some(recs))
    } else {
        let res = simulate_multiplayer(&ctx, &strategies, args.num_games, args.seed);
        (res, None)
    };
    let sim_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let per_game_us = t2.elapsed().as_secs_f64() * 1e6 / args.num_games as f64;
    println!(
        "  Elapsed: {:.1} ms ({:.1} \u{00b5}s/game)",
        sim_ms, per_game_us
    );
    println!();

    // Print results table
    let name_width = result
        .strategies
        .iter()
        .map(|s| s.len())
        .max()
        .unwrap_or(8)
        .max(12);

    println!(
        "{:<width$}  {:>8}  {:>6}  {:>7}  {:>6}  {:>7}",
        "Strategy",
        "Wins",
        "Win%",
        "Mean",
        "Std",
        "Margin",
        width = name_width
    );
    println!(
        "{:─<width$}  {:─>8}  {:─>6}  {:─>7}  {:─>6}  {:─>7}",
        "",
        "",
        "",
        "",
        "",
        "",
        width = name_width
    );

    for i in 0..n {
        println!(
            "{:<width$}  {:>8}  {:>5.1}%  {:>7.1}  {:>6.1}  {:>+6.1}",
            result.strategies[i],
            result.wins[i],
            result.win_rates[i],
            result.score_means[i],
            result.score_stds[i],
            result.avg_margin_when_winning[i],
            width = name_width
        );
    }

    println!();
    println!(
        "Draws: {} ({:.2}%)",
        result.draws,
        result.draws as f64 / args.num_games as f64 * 100.0
    );

    // Head-to-head matrix
    if n <= 6 {
        println!();
        println!("Head-to-Head (row beats col):");

        // Header row
        let short_names: Vec<String> = result
            .strategies
            .iter()
            .enumerate()
            .map(|(i, _)| format!("P{}", i + 1))
            .collect();

        print!("{:<width$}", "", width = name_width);
        for sn in &short_names {
            print!("  {:>7}", sn);
        }
        println!();

        for i in 0..n {
            print!("{:<width$}", result.strategies[i], width = name_width);
            for j in 0..n {
                if i == j {
                    print!("  {:>7}", "—");
                } else {
                    let total = result.head_to_head[i][j] + result.head_to_head[j][i];
                    let pct = if total > 0 {
                        result.head_to_head[i][j] as f64 / total as f64 * 100.0
                    } else {
                        50.0
                    };
                    print!("  {:>6.1}%", pct);
                }
            }
            println!();
        }
    }

    // Save outputs
    if let Some(ref output_dir) = args.output {
        std::fs::create_dir_all(output_dir).unwrap_or_else(|e| {
            eprintln!("Failed to create output directory '{}': {}", output_dir, e);
            std::process::exit(1);
        });

        // Save binary recording if available
        if let Some(ref recs) = records {
            let bin_path = format!("{}/multiplayer_raw.bin", output_dir);
            save_multiplayer_recording(recs, args.seed, n as u8, &bin_path);
            let size_mb = (recs.len() * 64 + 32) as f64 / 1e6;
            println!();
            println!("Recording saved to {} ({:.1} MB)", bin_path, size_mb);
        }

        let json_path = format!("{}/multiplayer_results.json", output_dir);
        let json = serde_json::to_string_pretty(&result).unwrap();
        std::fs::write(&json_path, json).unwrap_or_else(|e| {
            eprintln!("Failed to write {}: {}", json_path, e);
            std::process::exit(1);
        });
        println!("Results saved to {}", json_path);
    }
}
