use yatzy::phase0_tables;
use yatzy::state_computation::{compute_all_state_values, compute_all_state_values_with_oracle};
use yatzy::storage::{save_oracle, ORACLE_FILE_PATH};
use yatzy::types::YatzyContext;

fn parse_args() -> (f32, bool, bool) {
    let args: Vec<String> = std::env::args().collect();
    let mut theta = 0.0f32;
    let mut max_policy = false;
    let mut build_oracle = false;
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
            "--help" | "-h" => {
                println!("Usage: yatzy-precompute [--theta FLOAT] [--max-policy] [--oracle]");
                println!();
                println!("Options:");
                println!("  --theta FLOAT  Risk parameter (default: 0.0, risk-neutral)");
                println!("  --max-policy   Max-policy mode (chance nodes use max, not EV)");
                println!("  --oracle       Build policy oracle (~3.17 GB, θ=0 only)");
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
    (theta, max_policy, build_oracle)
}

fn main() {
    let _base = yatzy::env_config::init_base_path();
    let (theta, max_policy, build_oracle) = parse_args();

    println!("Yatzy precomputation tool (Rust)");
    if max_policy {
        println!("Mode: max-policy (chance nodes use max, not EV)");
    } else if theta != 0.0 {
        println!("Risk parameter θ = {:.4}", theta);
    }
    if build_oracle {
        println!("Mode: building policy oracle (~3.17 GB)");
    }

    let _threads = yatzy::env_config::init_rayon_threads();

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
