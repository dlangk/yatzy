use std::time::Instant;

use yatzy::phase0_tables;
use yatzy::simulation::lockstep::{simulate_batch_lockstep, simulate_batch_lockstep_oracle};
use yatzy::simulation::{
    aggregate_statistics, make_policy, policy_thetas, save_raw_simulation, save_scores,
    save_statistics, simulate_batch, simulate_batch_adaptive_with_recording,
    simulate_batch_with_recording, ThetaTable, POLICY_CONFIGS,
};
use yatzy::storage::{
    load_all_state_values, load_oracle, load_state_values_standalone, state_file_path,
    ORACLE_FILE_PATH,
};
use yatzy::types::YatzyContext;

struct Args {
    num_games: usize,
    seed: u64,
    output: Option<String>,
    theta: f32,
    max_policy: bool,
    policy: Option<String>,
    full_recording: bool,
    lockstep: bool,
    use_oracle: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 1000usize;
    let mut seed = 42u64;
    let mut output: Option<String> = None;
    let mut theta = 0.0f32;
    let mut max_policy = false;
    let mut policy: Option<String> = None;
    let mut full_recording = false;
    let mut lockstep = false;
    let mut use_oracle = false;

    let policy_names: Vec<&str> = POLICY_CONFIGS.iter().map(|p| p.name).collect();

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
            "--full-recording" => {
                full_recording = true;
            }
            "--lockstep" => {
                lockstep = true;
            }
            "--oracle" => {
                use_oracle = true;
            }
            "--policy" => {
                i += 1;
                if i < args.len() {
                    policy = Some(args[i].clone());
                }
            }
            "--help" | "-h" => {
                println!(
                    "Usage: yatzy-simulate [--games N] [--seed S] [--output DIR] [--theta FLOAT] [--max-policy] [--full-recording] [--policy NAME] [--oracle]"
                );
                println!();
                println!("Options:");
                println!("  --games N          Number of games to simulate (default: 1000)");
                println!("  --seed S           RNG seed (default: 42)");
                println!("  --output DIR       Write data and statistics to DIR");
                println!("  --theta FLOAT      Risk parameter (default: 0.0, risk-neutral)");
                println!("  --max-policy       Max-policy mode (chance nodes use max, not EV)");
                println!("  --full-recording   Save full per-turn GameRecord data (289 B/game)");
                println!(
                    "  --policy NAME      Adaptive policy: {}",
                    policy_names.join(", ")
                );
                println!("  --oracle           Use precomputed policy oracle (θ=0 lockstep only)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                eprintln!(
                    "Usage: yatzy-simulate [--games N] [--seed S] [--output DIR] [--theta FLOAT] [--max-policy] [--full-recording] [--policy NAME]"
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

    if policy.is_some() && (theta != 0.0 || max_policy) {
        eprintln!("Error: --policy is mutually exclusive with --theta and --max-policy");
        std::process::exit(1);
    }

    if let Some(ref p) = policy {
        if policy_thetas(p).is_none() {
            eprintln!(
                "Unknown policy: '{}'. Available: {}",
                p,
                policy_names.join(", ")
            );
            std::process::exit(1);
        }
    }

    if use_oracle && (theta != 0.0 || max_policy) {
        eprintln!("Error: --oracle only works with θ=0 EV mode");
        std::process::exit(1);
    }

    Args {
        num_games,
        seed,
        output,
        theta,
        max_policy,
        policy,
        full_recording,
        lockstep,
        use_oracle,
    }
}

fn main() {
    let _base = yatzy::env_config::init_base_path();
    let args = parse_args();
    let Args {
        num_games,
        seed,
        output,
        theta,
        max_policy,
        policy: policy_name,
        full_recording,
        lockstep,
        use_oracle,
    } = args;

    let num_threads = yatzy::env_config::init_rayon_threads();

    println!("Yatzy Simulation ({} games)", num_games);

    // ── Adaptive policy mode ────────────────────────────────────────────────
    if let Some(ref pname) = policy_name {
        println!("  Mode: adaptive policy '{}'", pname);

        let t0 = Instant::now();
        let mut ctx = YatzyContext::new_boxed();
        let alloc_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        phase0_tables::precompute_lookup_tables(&mut ctx);
        let tables_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // Load θ=0 into the main context (for starting EV)
        let t2 = Instant::now();
        if !load_all_state_values(&mut ctx, &state_file_path(0.0)) {
            eprintln!("Failed to load θ=0 state values. Run yatzy-precompute first.");
            std::process::exit(1);
        }
        let mmap_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let starting_ev = ctx.get_state_value(0, 0);

        // Load all required θ tables
        let required_thetas = policy_thetas(pname).unwrap();
        println!(
            "  Loading {} θ tables: {:?}",
            required_thetas.len(),
            required_thetas
        );

        let t3 = Instant::now();
        let mut theta_tables: Vec<ThetaTable> = Vec::new();
        for &t in required_thetas {
            let file = state_file_path(t);
            let sv = if t == 0.0 {
                // Reuse the already-loaded θ=0 from ctx
                load_state_values_standalone(&file)
            } else {
                load_state_values_standalone(&file)
            };
            match sv {
                Some(sv) => {
                    println!("    θ={:.3}: loaded from {}", t, file);
                    theta_tables.push(ThetaTable {
                        theta: t,
                        minimize: t < 0.0,
                        sv,
                    });
                }
                None => {
                    eprintln!("Failed to load state values from {}", file);
                    eprintln!("Run yatzy-precompute --theta {} first.", t);
                    std::process::exit(1);
                }
            }
        }
        let load_ms = t3.elapsed().as_secs_f64() * 1000.0;

        let policy = make_policy(pname, &theta_tables).unwrap();

        println!("  Context alloc:  {:.1} ms", alloc_ms);
        println!("  Phase 0 tables: {:.1} ms", tables_ms);
        println!("  θ=0 state vals: {:.1} ms (mmap)", mmap_ms);
        println!("  Extra tables:   {:.1} ms (mmap)", load_ms);
        println!("  Starting EV:   {:.4} (θ=0 baseline)", starting_ev);
        println!();

        if let Some(ref output_dir) = output {
            // Recording mode
            println!(
                "Simulating {} games with recording ({} threads)...",
                num_games, num_threads
            );
            let sim_start = Instant::now();
            let records = simulate_batch_adaptive_with_recording(
                &ctx,
                &theta_tables,
                policy.as_ref(),
                num_games,
                seed,
            );
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

            // Save raw binary (adaptive always records full data for statistics)
            let raw_path = format!("{}/simulation_raw.bin", output_dir);
            let t_raw = Instant::now();
            save_raw_simulation(&records, seed, starting_ev as f32, &raw_path);
            let raw_ms = t_raw.elapsed().as_secs_f64() * 1000.0;

            let raw_size_mb = (std::fs::metadata(&raw_path).map(|m| m.len()).unwrap_or(0)) as f64
                / 1024.0
                / 1024.0;
            println!(
                "  Raw data:    {} ({:.1} MB, {:.1} ms)",
                raw_path, raw_size_mb, raw_ms
            );

            // Also save compact scores.bin
            let adaptive_scores: Vec<i32> = records.iter().map(|r| r.total_score as i32).collect();
            let scores_path = format!("{}/scores.bin", output_dir);
            save_scores(
                &adaptive_scores,
                seed,
                starting_ev as f32,
                0.0,
                &scores_path,
            );
            let scores_size_mb = (std::fs::metadata(&scores_path)
                .map(|m| m.len())
                .unwrap_or(0)) as f64
                / 1024.0
                / 1024.0;
            println!("  Scores:      {} ({:.1} MB)", scores_path, scores_size_mb);

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
            println!("Results (policy: {}):", pname);
            println!(
                "  Mean score:  {:.2} (θ=0 EV: {:.2}, delta: {:+.2})",
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
        } else {
            // Non-recording mode: just print summary
            println!(
                "Simulating {} games ({} threads)...",
                num_games, num_threads
            );
            let sim_start = Instant::now();
            let scores = yatzy::simulation::simulate_batch_adaptive(
                &ctx,
                &theta_tables,
                policy.as_ref(),
                num_games,
                seed,
            );
            let sim_elapsed = sim_start.elapsed();

            let sum: f64 = scores.iter().map(|&s| s as f64).sum();
            let mean = sum / num_games as f64;
            let variance: f64 = scores
                .iter()
                .map(|&s| (s as f64 - mean).powi(2))
                .sum::<f64>()
                / num_games as f64;
            let std_dev = variance.sqrt();

            let per_game_us = sim_elapsed.as_secs_f64() * 1e6 / num_games as f64;
            let throughput = num_games as f64 / sim_elapsed.as_secs_f64();

            println!(
                "  Elapsed:     {:.1} ms",
                sim_elapsed.as_secs_f64() * 1000.0
            );
            println!("  Per game:    {:.1} \u{00b5}s", per_game_us);
            println!("  Throughput:  {:.0} games/sec", throughput);
            println!();

            println!("Results (policy: {}):", pname);
            println!(
                "  Mean score:  {:.2} (θ=0 EV: {:.2}, delta: {:+.2})",
                mean,
                starting_ev,
                mean - starting_ev,
            );
            println!("  Std dev:     {:.1}", std_dev);
            println!("  Min:         {}", scores.first().unwrap_or(&0));
            println!("  Max:         {}", scores.last().unwrap_or(&0));
            println!("  Median:      {}", scores[num_games / 2]);
        }

        return;
    }

    // ── Standard (non-adaptive) simulation modes ────────────────────────────
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
        "data/strategy_tables/all_states_max.bin".to_string()
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

    if let Some(ref output_dir) = output {
        if full_recording {
            // Full recording mode: capture per-turn GameRecord data (289 B/game)
            println!(
                "Simulating {} games with full recording ({} threads)...",
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

            // Save full GameRecord binary
            let raw_path = format!("{}/simulation_raw.bin", output_dir);
            let t_raw = Instant::now();
            save_raw_simulation(&records, seed, starting_ev as f32, &raw_path);
            let raw_ms = t_raw.elapsed().as_secs_f64() * 1000.0;

            let raw_size_mb = (std::fs::metadata(&raw_path).map(|m| m.len()).unwrap_or(0)) as f64
                / 1024.0
                / 1024.0;
            println!(
                "  Raw data:    {} ({:.1} MB, {:.1} ms)",
                raw_path, raw_size_mb, raw_ms
            );

            // Also save compact scores.bin
            let scores: Vec<i32> = records.iter().map(|r| r.total_score as i32).collect();
            let scores_path = format!("{}/scores.bin", output_dir);
            save_scores(&scores, seed, starting_ev as f32, theta, &scores_path);
            let scores_size_mb = (std::fs::metadata(&scores_path)
                .map(|m| m.len())
                .unwrap_or(0)) as f64
                / 1024.0
                / 1024.0;
            println!("  Scores:      {} ({:.1} MB)", scores_path, scores_size_mb);

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
            // Default: scores-only output (compact, ~2 MB per 1M games)
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

            // Save compact scores.bin
            let scores_path = format!("{}/scores.bin", output_dir);
            save_scores(
                &result.scores,
                seed,
                starting_ev as f32,
                theta,
                &scores_path,
            );
            let scores_size_mb = (std::fs::metadata(&scores_path)
                .map(|m| m.len())
                .unwrap_or(0)) as f64
                / 1024.0
                / 1024.0;
            println!("  Scores:      {} ({:.1} MB)", scores_path, scores_size_mb);

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
    } else {
        // No output: lightweight batch, print results only
        if lockstep {
            println!("  Mode: lockstep (horizontal processing)");
        }
        if use_oracle {
            println!("  Mode: oracle-based lockstep");
        }

        // Load oracle if requested
        let oracle = if use_oracle {
            let t_orc = Instant::now();
            match load_oracle(ORACLE_FILE_PATH) {
                Some(o) => {
                    println!(
                        "  Oracle:         {:.1} ms (loaded)",
                        t_orc.elapsed().as_secs_f64() * 1000.0
                    );
                    Some(o)
                }
                None => {
                    eprintln!("Failed to load oracle. Run yatzy-precompute --oracle first.");
                    std::process::exit(1);
                }
            }
        } else {
            None
        };

        println!(
            "Simulating {} games ({} threads)...",
            num_games, num_threads
        );
        let result = if use_oracle {
            simulate_batch_lockstep_oracle(&ctx, oracle.as_ref().unwrap(), num_games, seed)
        } else if lockstep && theta == 0.0 && !max_policy {
            simulate_batch_lockstep(&ctx, num_games, seed)
        } else {
            simulate_batch(&ctx, num_games, seed)
        };

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
