use std::fs;
use std::io::Write;
use std::time::Instant;

use yatzy::phase0_tables;
use yatzy::simulation::{
    aggregate_statistics, save_raw_simulation, save_statistics, simulate_batch,
    simulate_batch_with_recording,
};
use yatzy::storage::{load_all_state_values, state_file_path};
use yatzy::types::YatzyContext;

/// Save scores as flat binary: u32 count + i32[count].
fn save_scores_binary(scores: &[i32], path: &str) {
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut f = fs::File::create(path).unwrap_or_else(|e| {
        eprintln!("Failed to create {}: {}", path, e);
        std::process::exit(1);
    });
    let count = scores.len() as u32;
    f.write_all(&count.to_le_bytes()).unwrap();
    let bytes =
        unsafe { std::slice::from_raw_parts(scores.as_ptr() as *const u8, scores.len() * 4) };
    f.write_all(bytes).unwrap();
}

fn set_working_directory() {
    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    if std::env::set_current_dir(&base_path).is_err() {
        eprintln!("Failed to change directory to {}", base_path);
        std::process::exit(1);
    }
}

fn parse_args() -> (usize, u64, Option<String>, f32, bool) {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 1000usize;
    let mut seed = 42u64;
    let mut output: Option<String> = None;
    let mut theta = 0.0f32;
    let mut max_policy = false;

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
            "--theta" => {
                i += 1;
                if i < args.len() {
                    theta = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid --theta value: {}", args[i]);
                        std::process::exit(1);
                    });
                }
            }
            "--max-policy" => {
                max_policy = true;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: yatzy-simulate [--games N] [--seed S] [--output DIR] [--theta FLOAT] [--max-policy]"
                );
                println!();
                println!("Options:");
                println!("  --games N      Number of games to simulate (default: 1000)");
                println!("  --seed S       RNG seed (default: 42)");
                println!("  --output DIR   Write raw data and statistics to DIR");
                println!("  --theta FLOAT  Risk parameter (default: 0.0, risk-neutral)");
                println!("  --max-policy   Max-policy mode (chance nodes use max, not EV)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                eprintln!(
                    "Usage: yatzy-simulate [--games N] [--seed S] [--output DIR] [--theta FLOAT] [--max-policy]"
                );
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if max_policy && theta != 0.0 {
        eprintln!("Error: --max-policy and --theta are mutually exclusive");
        std::process::exit(1);
    }

    (num_games, seed, output, theta, max_policy)
}

fn main() {
    set_working_directory();
    let (num_games, seed, output, theta, max_policy) = parse_args();

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
    if max_policy {
        println!("  Mode: max-policy (chance nodes use max, not EV)");
    } else if theta != 0.0 {
        println!("  Risk parameter θ = {:.4}", theta);
    }

    let t0 = Instant::now();
    let mut ctx = YatzyContext::new_boxed();
    ctx.theta = theta;
    ctx.max_policy = max_policy;
    let alloc_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    phase0_tables::precompute_lookup_tables(&mut ctx);
    let tables_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let state_file = if max_policy {
        "data/all_states_max.bin".to_string()
    } else {
        state_file_path(theta)
    };
    let t2 = Instant::now();
    if !load_all_state_values(&mut ctx, &state_file) {
        eprintln!("Failed to load state values from {}", state_file);
        if max_policy {
            eprintln!("Run yatzy-precompute --max-policy first.");
        } else {
            eprintln!("Run yatzy-precompute --theta {} first.", theta);
        }
        std::process::exit(1);
    }
    let mmap_ms = t2.elapsed().as_secs_f64() * 1000.0;

    println!("  Context alloc:  {:.1} ms", alloc_ms);
    println!("  Phase 0 tables: {:.1} ms", tables_ms);
    println!("  State values:   {:.1} ms (mmap)", mmap_ms);

    let starting_ev = ctx.get_state_value(0, 0);
    println!("  Starting EV:   {:.4}", starting_ev);
    println!();

    if max_policy {
        // Max-policy mode: lightweight batch (no recording), save scores binary
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
            "  Mean score:  {:.2} (precomputed max-value: {:.2})",
            result.mean, starting_ev
        );
        println!("  Std dev:     {:.1}", result.std_dev);
        println!("  Min:         {}", result.min);
        println!("  Max:         {}", result.max);
        println!("  Median:      {}", result.median);

        if let Some(ref output_dir) = output {
            let scores_path = format!("{}/scores.bin", output_dir);
            save_scores_binary(&result.scores, &scores_path);
            let size_mb = (std::fs::metadata(&scores_path)
                .map(|m| m.len())
                .unwrap_or(0)) as f64
                / 1024.0
                / 1024.0;
            println!();
            println!("  Scores saved: {} ({:.1} MB)", scores_path, size_mb);
        }
    } else if let Some(ref output_dir) = output {
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
