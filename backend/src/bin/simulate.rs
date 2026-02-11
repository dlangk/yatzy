use std::time::Instant;

use yatzy::phase0_tables;
use yatzy::simulation::{
    aggregate_statistics, save_raw_simulation, save_statistics, simulate_batch,
    simulate_batch_with_recording,
};
use yatzy::storage::load_all_state_values;
use yatzy::types::YatzyContext;

fn set_working_directory() {
    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    if std::env::set_current_dir(&base_path).is_err() {
        eprintln!("Failed to change directory to {}", base_path);
        std::process::exit(1);
    }
}

fn parse_args() -> (usize, u64, Option<String>) {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 1000usize;
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
                println!("Usage: yatzy-simulate [--games N] [--seed S] [--output DIR]");
                println!();
                println!("Options:");
                println!("  --games N     Number of games to simulate (default: 1000)");
                println!("  --seed S      RNG seed (default: 42)");
                println!("  --output DIR  Write raw data and statistics to DIR");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                eprintln!("Usage: yatzy-simulate [--games N] [--seed S] [--output DIR]");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    (num_games, seed, output)
}

fn main() {
    set_working_directory();
    let (num_games, seed, output) = parse_args();

    // Configure rayon thread pool
    let num_threads = std::env::var("RAYON_NUM_THREADS")
        .or_else(|_| std::env::var("OMP_NUM_THREADS"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    println!("Yatzy Simulation ({} games)", num_games);

    let t0 = Instant::now();
    let mut ctx = YatzyContext::new_boxed();
    let alloc_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    phase0_tables::precompute_lookup_tables(&mut ctx);
    let tables_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let t2 = Instant::now();
    if !load_all_state_values(&mut ctx, "data/all_states.bin") {
        eprintln!("Failed to load state values from data/all_states.bin");
        eprintln!("Run yatzy-precompute first.");
        std::process::exit(1);
    }
    let mmap_ms = t2.elapsed().as_secs_f64() * 1000.0;

    println!("  Context alloc:  {:.1} ms", alloc_ms);
    println!("  Phase 0 tables: {:.1} ms", tables_ms);
    println!("  State values:   {:.1} ms (mmap)", mmap_ms);

    let starting_ev = ctx.get_state_value(0, 0);
    println!("  Starting EV:   {:.4}", starting_ev);
    println!();

    if let Some(ref output_dir) = output {
        // Recording mode: capture full per-step data
        println!(
            "Simulating {} games with recording ({} threads)...",
            num_games, num_threads
        );
        let sim_start = Instant::now();
        let records = simulate_batch_with_recording(&ctx, num_games, seed);
        let sim_elapsed = sim_start.elapsed();

        let per_game_us = sim_elapsed.as_secs_f64() * 1e6 / num_games as f64;
        let throughput = num_games as f64 / sim_elapsed.as_secs_f64();

        println!(
            "  Elapsed:     {:.1} ms",
            sim_elapsed.as_secs_f64() * 1000.0
        );
        println!("  Per game:    {:.1} \u{00b5}s", per_game_us);
        println!("  Throughput:  {:.0} games/sec", throughput);
        println!();

        // Save raw binary
        let raw_path = format!("{}/simulation_raw.bin", output_dir);
        let t_raw = Instant::now();
        save_raw_simulation(&records, seed, starting_ev as f32, &raw_path);
        let raw_ms = t_raw.elapsed().as_secs_f64() * 1000.0;

        let raw_size_mb =
            (std::fs::metadata(&raw_path).map(|m| m.len()).unwrap_or(0)) as f64 / 1024.0 / 1024.0;
        println!(
            "  Raw data:    {} ({:.1} MB, {:.1} ms)",
            raw_path, raw_size_mb, raw_ms
        );

        // Aggregate statistics
        let t_agg = Instant::now();
        let stats = aggregate_statistics(&records, starting_ev, seed);
        let agg_ms = t_agg.elapsed().as_secs_f64() * 1000.0;

        let json_path = format!("{}/game_statistics.json", output_dir);
        save_statistics(&stats, &json_path);
        println!(
            "  Statistics:  {} ({:.1} ms aggregation)",
            json_path, agg_ms
        );
        println!();

        // Print summary
        println!("Results:");
        println!(
            "  Mean score:  {:.2} (expected EV: {:.2}, delta: {:+.2})",
            stats.total_score.mean,
            starting_ev,
            stats.total_score.mean - starting_ev,
        );
        println!("  Std dev:     {:.1}", stats.total_score.std_dev);
        println!("  Min:         {}", stats.total_score.min);
        println!("  Max:         {}", stats.total_score.max);
        println!("  Median:      {}", stats.total_score.median);
        println!(
            "  Bonus rate:  {:.1}%",
            stats.upper_section.bonus_rate * 100.0
        );

        let se = stats.total_score.std_dev / (num_games as f64).sqrt();
        let z = (stats.total_score.mean - starting_ev) / se;
        println!();
        println!(
            "  Std error:   {:.3}  (z = {:+.2}, |z| < 3.0 expected)",
            se, z
        );
        if z.abs() > 3.5 {
            eprintln!(
                "WARNING: Mean deviates from EV by {:.1} standard errors — possible bug!",
                z.abs()
            );
        }
    } else {
        // Non-recording mode: existing lightweight behavior
        println!(
            "Simulating {} games ({} threads)...",
            num_games, num_threads
        );
        let result = simulate_batch(&ctx, num_games, seed);

        let per_game_us = result.elapsed.as_secs_f64() * 1e6 / num_games as f64;
        let throughput = num_games as f64 / result.elapsed.as_secs_f64();

        println!(
            "  Elapsed:     {:.1} ms",
            result.elapsed.as_secs_f64() * 1000.0
        );
        println!("  Per game:    {:.1} \u{00b5}s", per_game_us);
        println!("  Throughput:  {:.0} games/sec", throughput);
        println!();

        println!("Results:");
        println!(
            "  Mean score:  {:.2} (expected EV: {:.2}, delta: {:+.2})",
            result.mean,
            starting_ev,
            result.mean - starting_ev
        );
        println!("  Std dev:     {:.1}", result.std_dev);
        println!("  Min:         {}", result.min);
        println!("  Max:         {}", result.max);
        println!("  Median:      {}", result.median);

        let se = result.std_dev / (num_games as f64).sqrt();
        let z = (result.mean - starting_ev) / se;
        println!();
        println!(
            "  Std error:   {:.3}  (z = {:+.2}, |z| < 3.0 expected)",
            se, z
        );
        if z.abs() > 3.5 {
            eprintln!(
                "WARNING: Mean deviates from EV by {:.1} standard errors — possible bug!",
                z.abs()
            );
        }
    }
}
