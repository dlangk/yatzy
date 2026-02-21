//! Parameter sweep for the underdog adaptive strategy.
//!
//! Grid-searches over threshold and θ parameters, running each configuration
//! against EV-optimal play and reporting win rates. Top configurations are
//! re-tested at higher game counts for confirmation.

use std::time::Instant;

use yatzy::phase0_tables;
use yatzy::simulation::multiplayer::simulate_multiplayer;
use yatzy::simulation::strategy::Strategy;
use yatzy::types::YatzyContext;

struct Args {
    games: u32,
    confirm_games: u32,
    top_n: usize,
    confirm_n: usize,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut games = 100_000u32;
    let mut confirm_games = 1_000_000u32;
    let mut top_n = 20usize;
    let mut confirm_n = 5usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                i += 1;
                if i < args.len() {
                    games = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid --games value: {}", args[i]);
                        std::process::exit(1);
                    });
                }
            }
            "--confirm-games" => {
                i += 1;
                if i < args.len() {
                    confirm_games = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid --confirm-games value: {}", args[i]);
                        std::process::exit(1);
                    });
                }
            }
            "--top" => {
                i += 1;
                if i < args.len() {
                    top_n = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid --top value: {}", args[i]);
                        std::process::exit(1);
                    });
                }
            }
            "--confirm" => {
                i += 1;
                if i < args.len() {
                    confirm_n = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid --confirm value: {}", args[i]);
                        std::process::exit(1);
                    });
                }
            }
            "--help" | "-h" => {
                println!("Usage: yatzy-underdog-sweep [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --games N         Games per sweep config (default: 100000)");
                println!("  --confirm-games N Games for confirmation run (default: 1000000)");
                println!("  --top N           Show top N results (default: 20)");
                println!("  --confirm N       Confirm top N at higher game count (default: 5)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Args {
        games,
        confirm_games,
        top_n,
        confirm_n,
    }
}

#[derive(Clone)]
struct SweepConfig {
    threshold_lead: f32,
    threshold_even: f32,
    threshold_aggressive: f32,
    theta_protect: f32,
    theta_mild: f32,
    theta_aggressive: f32,
}

impl SweepConfig {
    fn spec(&self) -> String {
        format!(
            "mp:underdog:{}:{}:{}:{}:{}:{}",
            self.threshold_lead,
            self.threshold_even,
            self.threshold_aggressive,
            self.theta_protect,
            self.theta_mild,
            self.theta_aggressive,
        )
    }
}

struct SweepResult {
    config: SweepConfig,
    win_rate: f64,
    mean_score: f64,
    ev_mean: f64,
}

fn main() {
    let base_path = yatzy::env_config::init_base_path();
    let args = parse_args();
    let num_threads = yatzy::env_config::init_rayon_threads();

    let t0 = Instant::now();
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);
    println!(
        "Phase 0 tables: {:.1} ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Build parameter grid
    let threshold_leads = [0.1, 0.2, 0.3, 0.5, 0.7];
    let threshold_evens = [0.1, 0.2, 0.3, 0.5];
    let threshold_aggressives = [0.5, 0.7, 1.0, 1.5, 2.0];
    let theta_protects = [-0.02, -0.03, -0.05];
    let theta_milds = [0.03, 0.05, 0.07];
    let theta_aggressives = [0.07, 0.10, 0.15];

    let mut configs: Vec<SweepConfig> = Vec::new();
    for &tl in &threshold_leads {
        for &te in &threshold_evens {
            for &ta in &threshold_aggressives {
                for &tp in &theta_protects {
                    for &tm in &theta_milds {
                        for &tag in &theta_aggressives {
                            // Skip configs where mild θ >= aggressive θ
                            if tm >= tag {
                                continue;
                            }
                            configs.push(SweepConfig {
                                threshold_lead: tl,
                                threshold_even: te,
                                threshold_aggressive: ta,
                                theta_protect: tp,
                                theta_mild: tm,
                                theta_aggressive: tag,
                            });
                        }
                    }
                }
            }
        }
    }

    let total = configs.len();
    println!(
        "\nSweeping {} configurations × {} games each ({} threads)",
        total, args.games, num_threads
    );

    let t1 = Instant::now();
    let mut results: Vec<SweepResult> = Vec::with_capacity(total);

    for (i, config) in configs.iter().enumerate() {
        let spec = config.spec();
        let ev_strategy = match Strategy::from_spec("ev", &base_path, &ctx) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to load ev strategy: {}", e);
                std::process::exit(1);
            }
        };
        let underdog_strategy = match Strategy::from_spec(&spec, &base_path, &ctx) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to load underdog strategy '{}': {}", spec, e);
                continue;
            }
        };

        let strategies = vec![ev_strategy, underdog_strategy];
        let result = simulate_multiplayer(&ctx, &strategies, args.games, 42);

        results.push(SweepResult {
            config: config.clone(),
            win_rate: result.win_rates[1], // underdog is player 1
            mean_score: result.score_means[1],
            ev_mean: result.score_means[0],
        });

        if (i + 1) % 100 == 0 || i + 1 == total {
            let elapsed = t1.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            let remaining = (total - i - 1) as f64 / rate;
            eprint!(
                "\r  [{}/{}] {:.1} configs/s, ETA {:.0}s    ",
                i + 1,
                total,
                rate,
                remaining
            );
        }
    }
    eprintln!();

    let sweep_secs = t1.elapsed().as_secs_f64();
    println!("Sweep completed in {:.1}s", sweep_secs);

    // Sort by win rate descending
    results.sort_by(|a, b| b.win_rate.partial_cmp(&a.win_rate).unwrap());

    // Print top N
    let show_n = args.top_n.min(results.len());
    println!(
        "\n{:>4}  {:>5}  {:>5}  {:>5}  {:>6}  {:>5}  {:>5}  {:>7}  {:>7}  {:>7}",
        "Rank", "t_led", "t_evn", "t_agg", "θ_prot", "θ_mld", "θ_agg", "Win%", "Mean", "EvMean"
    );
    println!("{}", "─".repeat(78));

    for (i, r) in results.iter().take(show_n).enumerate() {
        println!(
            "{:>4}  {:>5.2}  {:>5.2}  {:>5.2}  {:>6.3}  {:>5.3}  {:>5.3}  {:>6.1}%  {:>7.1}  {:>7.1}",
            i + 1,
            r.config.threshold_lead,
            r.config.threshold_even,
            r.config.threshold_aggressive,
            r.config.theta_protect,
            r.config.theta_mild,
            r.config.theta_aggressive,
            r.win_rate,
            r.mean_score,
            r.ev_mean,
        );
    }

    // Confirmation run for top N
    if args.confirm_n > 0 && !results.is_empty() {
        let confirm_n = args.confirm_n.min(results.len());
        println!(
            "\n── Confirming top {} at {} games ──",
            confirm_n, args.confirm_games
        );

        for (i, r) in results.iter().take(confirm_n).enumerate() {
            let spec = r.config.spec();
            let ev_strategy = Strategy::from_spec("ev", &base_path, &ctx).unwrap();
            let underdog_strategy = Strategy::from_spec(&spec, &base_path, &ctx).unwrap();
            let strategies = vec![ev_strategy, underdog_strategy];

            let t = Instant::now();
            let confirmed = simulate_multiplayer(&ctx, &strategies, args.confirm_games, 12345);
            let ms = t.elapsed().as_secs_f64() * 1000.0;

            println!(
                "  #{}: Win {:.2}%, Mean {:.1} vs {:.1}, Draws {:.2}% ({:.0}ms)",
                i + 1,
                confirmed.win_rates[1],
                confirmed.score_means[1],
                confirmed.score_means[0],
                confirmed.draws as f64 / args.confirm_games as f64 * 100.0,
                ms,
            );
        }
    }

    println!("\nDone.");
}
