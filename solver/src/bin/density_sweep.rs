//! yatzy-density: Compute exact score distributions via forward density evolution.
//!
//! Produces mathematically perfect PMFs with zero variance, replacing Monte Carlo
//! simulation for sweep statistics.

use std::fs;
use std::io::Write;
use std::time::Instant;

use yatzy::density::forward::{density_evolution, density_evolution_oracle};
use yatzy::phase0_tables;
use yatzy::simulation::sweep::{ensure_strategy_table, format_theta_dir, resolve_grid};
use yatzy::storage::{load_all_state_values, load_oracle, state_file_path, ORACLE_FILE_PATH};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut grid_name = "dense".to_string();
    let mut thetas_csv: Option<String> = None;
    let mut output_dir = "outputs/density".to_string();
    let mut single_theta: Option<f32> = None;
    let mut use_oracle = false;

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
            "--theta" => {
                i += 1;
                single_theta = Some(args[i].parse().expect("Invalid --theta value"));
            }
            "--output" => {
                i += 1;
                output_dir = args[i].clone();
            }
            "--oracle" => {
                use_oracle = true;
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

    // Set working directory
    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    if std::env::set_current_dir(&base_path).is_err() {
        eprintln!("Failed to change directory to {}", base_path);
        std::process::exit(1);
    }

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

    // Resolve theta list
    let thetas = if let Some(t) = single_theta {
        vec![t]
    } else if let Some(csv) = thetas_csv {
        csv.split(',')
            .map(|s| s.trim().parse::<f32>().expect("Invalid theta value"))
            .collect()
    } else {
        resolve_grid(&grid_name).unwrap_or_else(|| {
            eprintln!("Unknown grid '{}'", grid_name);
            std::process::exit(1);
        })
    };

    println!("=== yatzy-density ===");
    println!(
        "{} thetas, {} threads, output: {}",
        thetas.len(),
        num_threads,
        output_dir
    );

    // Create output directory
    fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    // Build phase0 tables once
    let mut ctx = yatzy::types::YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    // Load oracle if requested
    let oracle = if use_oracle {
        let t_orc = Instant::now();
        match load_oracle(ORACLE_FILE_PATH) {
            Some(o) => {
                println!("Oracle loaded in {:.2}s", t_orc.elapsed().as_secs_f64());
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

    let t_total = Instant::now();

    for (idx, &theta) in thetas.iter().enumerate() {
        let t0 = Instant::now();

        println!(
            "\n[{}/{}] θ={}",
            idx + 1,
            thetas.len(),
            format_theta_dir(theta)
        );

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

        // Run density evolution
        let result = if use_oracle && theta == 0.0 {
            density_evolution_oracle(&ctx, oracle.as_ref().unwrap())
        } else {
            density_evolution(&ctx, theta)
        };

        // Write JSON output
        let out_path = format!("{}/density_{}.json", output_dir, format_theta_dir(theta));

        let mut json = String::new();
        json.push_str("{\n");
        json.push_str(&format!("  \"theta\": {},\n", theta));
        json.push_str(&format!("  \"mean\": {:.10},\n", result.mean));
        json.push_str(&format!("  \"variance\": {:.10},\n", result.variance));
        json.push_str(&format!("  \"std_dev\": {:.10},\n", result.std_dev));
        json.push_str("  \"percentiles\": {\n");

        let pct_keys = ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"];
        for (j, key) in pct_keys.iter().enumerate() {
            let val = result.percentiles.get(*key).copied().unwrap_or(0);
            let comma = if j < pct_keys.len() - 1 { "," } else { "" };
            json.push_str(&format!("    \"{}\": {}{}\n", key, val, comma));
        }
        json.push_str("  },\n");

        json.push_str("  \"pmf\": [\n");
        for (j, &(score, prob)) in result.pmf.iter().enumerate() {
            let comma = if j < result.pmf.len() - 1 { "," } else { "" };
            json.push_str(&format!("    [{}, {:.15}]{}\n", score, prob, comma));
        }
        json.push_str("  ]\n");
        json.push_str("}\n");

        let mut f = fs::File::create(&out_path).expect("Failed to create output file");
        f.write_all(json.as_bytes()).expect("Failed to write");

        let elapsed = t0.elapsed().as_secs_f64();
        println!(
            "  Wrote {} ({} PMF entries, mean={:.2}, std={:.2}) in {:.1}s",
            out_path,
            result.pmf.len(),
            result.mean,
            result.std_dev,
            elapsed
        );
    }

    let total_elapsed = t_total.elapsed().as_secs_f64();
    println!(
        "\nDone. {} thetas in {:.1}s ({:.1}s/theta avg).",
        thetas.len(),
        total_elapsed,
        total_elapsed / thetas.len() as f64
    );
}

fn print_usage() {
    println!(
        "yatzy-density: Exact score distributions via forward density evolution.

USAGE:
    yatzy-density [OPTIONS]

OPTIONS:
    --theta <FLOAT>       Single theta value
    --thetas <LIST>       Comma-separated theta values
    --grid <NAME>         Named grid: all, dense, sparse [default: dense]
    --output <DIR>        Output directory [default: outputs/density]
    -h, --help            Print this help

EXAMPLES:
    yatzy-density --theta 0                # Single exact PMF for θ=0
    yatzy-density --grid dense             # All 19 dense-grid thetas
    yatzy-density --thetas 0,0.05,0.1      # Specific thetas"
    );
}
