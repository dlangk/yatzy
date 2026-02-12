use yatzy::phase0_tables;
use yatzy::state_computation::compute_all_state_values;
use yatzy::types::YatzyContext;

fn set_working_directory() {
    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    println!("YATZY_BASE_PATH={}", base_path);
    if std::env::set_current_dir(&base_path).is_err() {
        eprintln!("Failed to change directory to {}", base_path);
        std::process::exit(1);
    }
    if let Ok(cwd) = std::env::current_dir() {
        println!("Working directory changed to: {}", cwd.display());
    }
}

fn parse_args() -> (f32, bool) {
    let args: Vec<String> = std::env::args().collect();
    let mut theta = 0.0f32;
    let mut max_policy = false;
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
            "--help" | "-h" => {
                println!("Usage: yatzy-precompute [--theta FLOAT] [--max-policy]");
                println!();
                println!("Options:");
                println!("  --theta FLOAT  Risk parameter (default: 0.0, risk-neutral)");
                println!("  --max-policy   Max-policy mode (chance nodes use max, not EV)");
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
    (theta, max_policy)
}

fn main() {
    set_working_directory();
    let (theta, max_policy) = parse_args();

    println!("Yatzy precomputation tool (Rust)");
    if max_policy {
        println!("Mode: max-policy (chance nodes use max, not EV)");
    } else if theta != 0.0 {
        println!("Risk parameter θ = {:.4}", theta);
    }

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
    println!("Using {} threads", num_threads);

    let mut ctx = YatzyContext::new_boxed();
    ctx.theta = theta;
    ctx.max_policy = max_policy;
    phase0_tables::precompute_lookup_tables(&mut ctx);
    compute_all_state_values(&mut ctx);

    println!("Precomputation complete.");
}
