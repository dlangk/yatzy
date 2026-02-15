//! yatzy-sweep: Simulate games for each theta in a grid, store raw scores.
//!
//! Resumable and incremental: skips thetas where existing scores.bin has >= --games.
//! Each theta is atomic — safe to Ctrl+C and re-run.

use std::time::Instant;

use yatzy::phase0_tables;
use yatzy::simulation::sweep::{
    ensure_strategy_table, format_theta_dir, range_grid, resolve_grid, scan_inventory,
    theta_eq, theta_scores_path,
};
use yatzy::simulation::{save_scores, simulate_batch};
use yatzy::storage::load_all_state_values;
use yatzy::storage::state_file_path;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut grid_name = "all".to_string();
    let mut thetas_csv: Option<String> = None;
    let mut range_args: Option<(f32, f32, f32)> = None;
    let mut games: u32 = 1_000_000;
    let mut seed: u64 = 42;
    let mut force = false;
    let mut list_mode = false;

    // Parse args
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--grid" => {
                i += 1;
                grid_name = args[i].clone();
            }
            "--thetas" => {
                i += 1;
                thetas_csv = Some(args[i].clone());
            }
            "--range" => {
                let lo: f32 = args[i + 1].parse().expect("Invalid --range lo");
                let hi: f32 = args[i + 2].parse().expect("Invalid --range hi");
                let step: f32 = args[i + 3].parse().expect("Invalid --range step");
                range_args = Some((lo, hi, step));
                i += 3;
            }
            "--games" => {
                i += 1;
                games = args[i].parse().expect("Invalid --games");
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("Invalid --seed");
            }
            "--force" => {
                force = true;
            }
            "--list" => {
                list_mode = true;
            }
            "--help" | "-h" => {
                print_usage();
                return;
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // List mode: just show existing inventory
    if list_mode {
        let inventory = scan_inventory();
        if inventory.is_empty() {
            println!("No existing scores found.");
        } else {
            println!(
                "{:>8}  {:>10}  {:>6}  path",
                "theta", "games", "seed"
            );
            println!("{}", "-".repeat(70));
            for e in &inventory {
                println!(
                    "{:>8}  {:>10}  {:>6}  {}",
                    format_theta_dir(e.theta),
                    e.num_games,
                    e.seed,
                    e.path
                );
            }
            println!("\n{} thetas with scores.", inventory.len());
        }
        return;
    }

    // Resolve theta grid
    let thetas = if let Some(csv) = thetas_csv {
        csv.split(',')
            .map(|s| s.trim().parse::<f32>().expect("Invalid theta value"))
            .collect::<Vec<_>>()
    } else if let Some((lo, hi, step)) = range_args {
        range_grid(lo, hi, step)
    } else {
        resolve_grid(&grid_name).unwrap_or_else(|| {
            eprintln!(
                "Unknown grid '{}'. Available: all, dense, sparse",
                grid_name
            );
            std::process::exit(1);
        })
    };

    println!("=== yatzy-sweep ===");
    println!("Grid: {} thetas, {} games/theta, seed={}", thetas.len(), games, seed);

    // Scan existing inventory
    let inventory = scan_inventory();
    let mut skip_count = 0;
    let mut to_simulate: Vec<f32> = Vec::new();

    for &t in &thetas {
        let existing = inventory.iter().find(|e| theta_eq(e.theta, t));
        if let Some(e) = existing {
            if e.num_games >= games && !force {
                skip_count += 1;
                continue;
            }
        }
        to_simulate.push(t);
    }

    println!(
        "Found {} existing, {} requested, {} to simulate",
        inventory.len(),
        thetas.len(),
        to_simulate.len()
    );
    if skip_count > 0 {
        println!("Skipping {} thetas (already have >= {} games)", skip_count, games);
    }

    if to_simulate.is_empty() {
        println!("Nothing to do.");
        return;
    }

    // Build phase0 tables once
    println!("\nBuilding lookup tables...");
    let mut ctx = yatzy::types::YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    let total = to_simulate.len();
    let t_total = Instant::now();

    for (idx, &theta) in to_simulate.iter().enumerate() {
        let t0 = Instant::now();

        // Ensure strategy table exists
        if !ensure_strategy_table(&mut ctx, theta) {
            eprintln!("Failed to precompute strategy table for θ={}", theta);
            continue;
        }

        // Load strategy table
        let file = state_file_path(theta);
        if !load_all_state_values(&mut ctx, &file) {
            eprintln!("Failed to load strategy table: {}", file);
            continue;
        }
        ctx.theta = theta;

        // Simulate
        let result = simulate_batch(&ctx, games as usize, seed);

        // Save scores
        let path = theta_scores_path(theta);
        save_scores(&result.scores, seed, result.mean as f32, theta, &path);

        let elapsed = t0.elapsed().as_secs_f64();
        println!(
            "[{}/{}] θ={:>8}  {} games  mean={:.1}  std={:.1}  {:.1}s",
            idx + 1,
            total,
            format_theta_dir(theta),
            games,
            result.mean,
            result.std_dev,
            elapsed,
        );
    }

    let total_elapsed = t_total.elapsed().as_secs_f64();
    println!(
        "\nDone. {}/{} thetas simulated in {:.1}s.",
        total,
        thetas.len(),
        total_elapsed
    );
}

fn print_usage() {
    println!(
        "yatzy-sweep: Simulate games for each theta in a grid, store raw scores.

USAGE:
    yatzy-sweep [OPTIONS]

OPTIONS:
    --grid <NAME>         Named grid: all (37), dense (19), sparse (17) [default: all]
    --thetas <LIST>       Comma-separated theta values (overrides --grid)
    --range <LO HI STEP>  Generate range, e.g. --range -0.1 0.4 0.01
    --games <N>           Games per theta [default: 1000000]
    --seed <S>            Base RNG seed [default: 42]
    --force               Re-simulate even if scores exist with >= games
    --list                List existing thetas and exit
    -h, --help            Print this help"
    );
}
