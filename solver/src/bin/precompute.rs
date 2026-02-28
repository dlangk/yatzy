use yatzy::phase0_tables;
use yatzy::state_computation::{compute_all_state_values, compute_all_state_values_with_oracle};
use yatzy::storage::{save_oracle, ORACLE_FILE_PATH};
use yatzy::types::YatzyContext;

fn parse_args() -> (f32, bool, bool, bool) {
    let args: Vec<String> = std::env::args().collect();
    let mut theta = 0.0f32;
    let mut max_policy = false;
    let mut build_oracle = false;
    let mut build_percentiles = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
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
            "--oracle" => {
                build_oracle = true;
            }
            "--percentiles" => {
                build_percentiles = true;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: yatzy-precompute [--theta FLOAT] [--max-policy] [--oracle] [--percentiles]"
                );
                println!();
                println!("Options:");
                println!("  --theta FLOAT   Risk parameter (default: 0.0, risk-neutral)");
                println!("  --max-policy    Max-policy mode (chance nodes use max, not EV)");
                println!("  --oracle        Build policy oracle (~3.17 GB, θ=0 only)");
                println!(
                    "  --percentiles   Build percentile table for turns 0-4 (requires oracle)"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }
    if max_policy && theta != 0.0 {
        eprintln!("Error: --max-policy and --theta are mutually exclusive");
        std::process::exit(1);
    }
    if build_oracle && (theta != 0.0 || max_policy) {
        eprintln!("Error: --oracle only works with θ=0 EV mode");
        std::process::exit(1);
    }
    if build_percentiles && (theta != 0.0 || max_policy) {
        eprintln!("Error: --percentiles only works with θ=0 EV mode");
        std::process::exit(1);
    }
    (theta, max_policy, build_oracle, build_percentiles)
}

fn main() {
    let _base = yatzy::env_config::init_base_path();
    let (theta, max_policy, build_oracle, build_percentiles) = parse_args();

    println!("Yatzy precomputation tool (Rust)");
    if max_policy {
        println!("Mode: max-policy (chance nodes use max, not EV)");
    } else if theta != 0.0 {
        println!("Risk parameter θ = {:.4}", theta);
    }
    if build_oracle {
        println!("Mode: building policy oracle (~3.17 GB)");
    }
    if build_percentiles {
        println!("Mode: building percentile table for turns 0-4");
    }

    let _threads = yatzy::env_config::init_rayon_threads();

    if build_percentiles {
        build_percentile_table();
        return;
    }

    let mut ctx = YatzyContext::new_boxed();
    ctx.theta = theta;
    ctx.max_policy = max_policy;
    phase0_tables::precompute_lookup_tables(&mut ctx);

    if build_oracle {
        let oracle = compute_all_state_values_with_oracle(&mut ctx, true);
        if let Some(ref orc) = oracle {
            save_oracle(orc, ORACLE_FILE_PATH);
        }
    } else {
        compute_all_state_values(&mut ctx);
    }

    println!("Precomputation complete.");
}

/// Build percentile table for states with 0-4 scored categories.
///
/// Loads strategy table + oracle from disk, then simulates 100K games per state
/// to compute remaining-score percentiles. Results saved to percentiles.bin.
fn build_percentile_table() {
    use rayon::prelude::*;
    use std::time::Instant;
    use yatzy::constants::*;
    use yatzy::simulation::engine::{extract_percentiles_i32, simulate_remaining_scores_oracle};
    use yatzy::storage::{
        load_all_state_values, load_oracle, save_percentile_table, PERCENTILE_FILE_PATH,
    };
    use yatzy::types::PercentileEntry;

    let start = Instant::now();

    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    if !load_all_state_values(&mut ctx, "data/strategy_tables/all_states.bin") {
        eprintln!("Error: strategy table not found. Run yatzy-precompute first.");
        std::process::exit(1);
    }

    let oracle = match load_oracle(ORACLE_FILE_PATH) {
        Some(o) => o,
        None => {
            eprintln!("Error: oracle not found. Run yatzy-precompute --oracle first.");
            std::process::exit(1);
        }
    };

    // Enumerate all reachable states with 0-4 scored categories
    let mut states: Vec<(i32, i32)> = Vec::new(); // (upper_score, scored_categories)
    for scored in 0..(1i32 << CATEGORY_COUNT) {
        let n = scored.count_ones();
        if n > 4 {
            continue;
        }
        // Check which upper scores are reachable for this mask
        let upper_mask = scored & 0x3F; // bits 0-5 are upper categories
        for up in 0..=63i32 {
            if ctx.reachable[upper_mask as usize][up as usize] {
                states.push((up, scored));
            }
        }
    }

    println!(
        "Found {} reachable states with 0-4 scored categories",
        states.len()
    );

    let num_games: usize = 100_000;
    let ctx_ref = &*ctx;
    let oracle_ref = &oracle;

    let entries: Vec<(u32, PercentileEntry)> = states
        .par_iter()
        .map(|&(up, scored)| {
            let si = state_index(up as usize, scored as usize) as u32;
            let sorted = simulate_remaining_scores_oracle(
                ctx_ref,
                oracle_ref,
                up,
                scored,
                num_games,
                si as u64 * 12345, // deterministic seed per state
            );
            let (mean, std_dev, pcts) = extract_percentiles_i32(&sorted);
            (
                si,
                PercentileEntry {
                    mean,
                    std_dev,
                    percentiles: pcts,
                },
            )
        })
        .collect();

    save_percentile_table(&entries, num_games as u32, PERCENTILE_FILE_PATH);

    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "Percentile table complete: {} entries in {:.1}s",
        entries.len(),
        elapsed
    );
}
