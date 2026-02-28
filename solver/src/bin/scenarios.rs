//! Unified scenario generation pipeline.
//!
//! Subcommands:
//!   collect   — Simulate games, collect + validate candidates → candidate_pool.json
//!   select    — Select scenarios from pool (difficulty or diagnostic mode)
//!   enrich    — Add θ-sensitivity or Q-grid data to selected scenarios
//!   all       — Run full pipeline: collect → select → enrich
//!
//! Usage:
//!   YATZY_BASE_PATH=. solver/target/release/yatzy-scenarios collect [OPTIONS]
//!   YATZY_BASE_PATH=. solver/target/release/yatzy-scenarios select --mode difficulty [OPTIONS]
//!   YATZY_BASE_PATH=. solver/target/release/yatzy-scenarios enrich --sensitivity [OPTIONS]

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use yatzy::phase0_tables;
use yatzy::scenarios::actions::truncate_str;
use yatzy::scenarios::classify::classify_phase;
use yatzy::scenarios::collect::{collect_candidates, filter_and_validate, NoisyParams};
use yatzy::scenarios::enrich::{compute_q_grid, evaluate_scenario_sensitivity, ScenarioParams};
use yatzy::scenarios::io::{discover_theta_files, load_theta_entries};
use yatzy::scenarios::select::{
    assemble_quiz, build_master_pool, build_profiling_scenarios, select_by_difficulty,
};
use yatzy::scenarios::types::*;
use yatzy::storage::{load_state_values_standalone, state_file_path};
use yatzy::types::{StateValues, YatzyContext};

// ── Output JSON types ──

#[derive(Serialize, Deserialize)]
struct DifficultScenarioJson {
    rank: usize,
    upper_score: i32,
    scored_categories: i32,
    dice: [i32; 5],
    turn: usize,
    decision_type: String,
    category_scores: [i32; 15],
    visit_count: usize,
    visit_fraction: f64,
    best_action: ActionInfo,
    runner_up_action: ActionInfo,
    ev_gap: f32,
    difficulty_score: f64,
    all_actions: Vec<ActionInfo>,
    description: String,
}

#[derive(Serialize)]
struct SensitivityScenarioJson {
    rank: usize,
    upper_score: i32,
    scored_categories: i32,
    dice: [i32; 5],
    turn: usize,
    decision_type: String,
    game_phase: String,
    visit_count: usize,
    difficulty_score: f64,
    description: String,
    theta_0_action: String,
    theta_0_action_id: i32,
    has_flip: bool,
    flip_theta: f32,
    flip_action: String,
    flip_action_id: i32,
    gap_at_flip: f32,
    gap_at_theta0: f32,
    theta_results: Vec<ThetaResult>,
}

#[derive(Serialize)]
struct OutputAction {
    id: i32,
    label: String,
    ev_theta0: f32,
}

#[derive(Serialize)]
struct QGridJson {
    theta_values: Vec<f32>,
    gamma_values: Vec<f32>,
    d_values: Vec<u32>,
    q_values: HashMap<String, Vec<f32>>,
}

#[derive(Serialize)]
struct ProfileScenarioJson {
    id: usize,
    upper_score: i32,
    scored_categories: i32,
    dice: [i32; 5],
    rerolls_remaining: u8,
    turn: usize,
    decision_type: String,
    quadrant: String,
    actions: Vec<OutputAction>,
    optimal_action_id: i32,
    gap: f32,
    description: String,
    q_grid: QGridJson,
}

#[derive(Serialize)]
struct ProfileOutputRoot {
    scenarios: Vec<ProfileScenarioJson>,
}

#[derive(Serialize)]
struct PoolCandidate {
    upper_score: i32,
    scored_categories: i32,
    dice: [i32; 5],
    turn: usize,
    decision_type: String,
    phase: String,
    tension: String,
    visit_count: usize,
    ev_gap: f32,
    s_theta: f32,
    s_gamma: f32,
    s_d: f32,
    s_beta: f32,
}

#[derive(Serialize)]
struct PoolRoot {
    total_candidates: usize,
    candidates: Vec<PoolCandidate>,
}

// ── CLI parsing ──

struct Args {
    subcommand: String,
    num_games: usize,
    seed: u64,
    output_dir: String,
    mode: String,     // "difficulty" or "diagnostic"
    top_n: usize,     // for difficulty mode
    quiz_size: usize, // for diagnostic mode
    pool_size: usize, // max per bucket
    min_visits: usize,
    noisy: bool,
    beta: f32,
    gamma: f32,
    sigma_d: f32,
    sensitivity: bool,
    q_grid: bool,
    theta_min: Option<f32>,
    theta_max: Option<f32>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut a = Args {
        subcommand: String::new(),
        num_games: 1_000_000,
        seed: 42,
        output_dir: String::from("outputs/scenarios"),
        mode: String::from("difficulty"),
        top_n: 200,
        quiz_size: 30,
        pool_size: 20,
        min_visits: 10,
        noisy: false,
        beta: 3.0,
        gamma: 0.85,
        sigma_d: 4.0,
        sensitivity: false,
        q_grid: false,
        theta_min: None,
        theta_max: None,
    };

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    a.subcommand = args[1].clone();
    if a.subcommand == "--help" || a.subcommand == "-h" {
        print_usage();
        std::process::exit(0);
    }

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                i += 1;
                a.num_games = args[i].parse().expect("Invalid --games");
            }
            "--seed" => {
                i += 1;
                a.seed = args[i].parse().expect("Invalid --seed");
            }
            "--output" => {
                i += 1;
                a.output_dir = args[i].clone();
            }
            "--mode" => {
                i += 1;
                a.mode = args[i].clone();
            }
            "--top" => {
                i += 1;
                a.top_n = args[i].parse().expect("Invalid --top");
            }
            "--quiz-size" => {
                i += 1;
                a.quiz_size = args[i].parse().expect("Invalid --quiz-size");
            }
            "--pool-size" => {
                i += 1;
                a.pool_size = args[i].parse().expect("Invalid --pool-size");
            }
            "--min-visits" => {
                i += 1;
                a.min_visits = args[i].parse().expect("Invalid --min-visits");
            }
            "--noisy" => {
                a.noisy = true;
            }
            "--beta" => {
                i += 1;
                a.beta = args[i].parse().expect("Invalid --beta");
            }
            "--gamma" => {
                i += 1;
                a.gamma = args[i].parse().expect("Invalid --gamma");
            }
            "--sigma-d" => {
                i += 1;
                a.sigma_d = args[i].parse().expect("Invalid --sigma-d");
            }
            "--sensitivity" => {
                a.sensitivity = true;
            }
            "--q-grid" => {
                a.q_grid = true;
            }
            "--theta-min" => {
                i += 1;
                a.theta_min = Some(args[i].parse().expect("Invalid --theta-min"));
            }
            "--theta-max" => {
                i += 1;
                a.theta_max = Some(args[i].parse().expect("Invalid --theta-max"));
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    a
}

fn print_usage() {
    println!("Usage: yatzy-scenarios <COMMAND> [OPTIONS]");
    println!();
    println!("Commands:");
    println!("  collect     Simulate games, collect candidates → candidate_pool.json");
    println!("  select      Select scenarios from pool (--mode difficulty|diagnostic)");
    println!("  enrich      Add θ-sensitivity or Q-grid to selected scenarios");
    println!("  all         Full pipeline: collect → select → enrich");
    println!();
    println!("Options:");
    println!("  --games N        Games to simulate (default: 1000000)");
    println!("  --seed S         Random seed (default: 42)");
    println!("  --output DIR     Output directory (default: outputs/scenarios)");
    println!("  --mode MODE      Selection mode: difficulty, diagnostic (default: difficulty)");
    println!("  --top N          Scenarios for difficulty mode (default: 200)");
    println!("  --quiz-size N    Scenarios for diagnostic mode (default: 30)");
    println!("  --pool-size N    Max candidates per bucket (default: 20)");
    println!("  --min-visits N   Minimum visit count (default: 10)");
    println!("  --noisy          Use noisy simulation (for diagnostic mode)");
    println!("  --beta F         Noisy sim β (default: 3.0)");
    println!("  --gamma F        Noisy sim γ (default: 0.85)");
    println!("  --sigma-d F      Noisy sim σ_d (default: 4.0)");
    println!("  --sensitivity    Enrich with θ-sensitivity");
    println!("  --q-grid         Enrich with Q-value grids");
    println!("  --theta-min F    Min θ for sensitivity (default: all)");
    println!("  --theta-max F    Max θ for sensitivity (default: all)");
}

fn resolve_output_dir(output_dir: &str) -> String {
    if std::path::Path::new(output_dir).is_absolute() {
        output_dir.to_string()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(output_dir).to_string_lossy().to_string())
            .unwrap_or_else(|_| output_dir.to_string())
    }
}

fn setup(args: &Args) -> Box<YatzyContext> {
    let _base = yatzy::env_config::init_base_path();
    let _threads = yatzy::env_config::init_rayon_threads_lenient();

    println!("Precomputing lookup tables...");
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    let file0 = state_file_path(0.0);
    match load_state_values_standalone(&file0) {
        Some(sv) => {
            ctx.state_values = sv;
            println!("Loaded θ=0 state values from {}", file0);
        }
        None => {
            eprintln!("Failed to load {}. Run yatzy-precompute first.", file0);
            std::process::exit(1);
        }
    }

    let _ = args; // used for uniformity
    ctx
}

// ── Subcommands ──

fn cmd_collect(args: &Args, ctx: &YatzyContext) -> Vec<(RawDecision, usize)> {
    let noisy_params = if args.noisy {
        println!(
            "Simulating {} noisy games (β={}, γ={}, σ_d={})...",
            args.num_games, args.beta, args.gamma, args.sigma_d
        );
        Some(NoisyParams {
            beta: args.beta,
            gamma: args.gamma,
            sigma_d: args.sigma_d,
        })
    } else {
        println!("Simulating {} optimal games...", args.num_games);
        None
    };

    let sim_start = Instant::now();
    let candidates = collect_candidates(ctx, args.num_games, args.seed, noisy_params.as_ref());
    println!(
        "Collected {} unique decisions in {:.1}s",
        candidates.len(),
        sim_start.elapsed().as_secs_f64(),
    );

    println!("Filtering (min_visits={})...", args.min_visits);
    let filtered = filter_and_validate(ctx, candidates, args.min_visits, !args.noisy);
    println!("  {} surviving candidates", filtered.len());

    filtered
}

fn cmd_select_difficulty(
    args: &Args,
    ctx: &YatzyContext,
    candidates: &[(RawDecision, usize)],
    output_dir: &str,
) {
    println!(
        "Selecting top {} by difficulty, stratified by turn...",
        args.top_n
    );
    let scenarios = select_by_difficulty(ctx, candidates, args.top_n);

    // Count total decisions by type for visit_fraction
    let mut type_totals: HashMap<&str, usize> = HashMap::new();
    for (d, count) in candidates {
        *type_totals.entry(d.decision_type.as_str()).or_default() += count;
    }

    // Build JSON output
    let json_scenarios: Vec<DifficultScenarioJson> = scenarios
        .iter()
        .enumerate()
        .map(|(rank, c)| {
            let total_of_type = type_totals
                .get(c.decision.decision_type.as_str())
                .copied()
                .unwrap_or(1);
            let visit_fraction = c.visit_count as f64 / total_of_type.max(1) as f64;

            let best = c.actions[0].clone();
            let runner_up = if c.actions.len() > 1 {
                c.actions[1].clone()
            } else {
                best.clone()
            };

            let desc = format!(
                "Turn {} ({}), dice {:?}, {} remaining categories. {}: {} (EV {:.2}) vs {} (EV {:.2}), gap {:.3}",
                c.decision.turn + 1,
                c.decision.decision_type.phase_label(),
                c.decision.dice,
                yatzy::constants::CATEGORY_COUNT - (c.decision.scored.count_ones() as usize),
                c.decision.decision_type.phase_label(),
                best.label,
                best.ev,
                runner_up.label,
                runner_up.ev,
                c.ev_gap,
            );

            DifficultScenarioJson {
                rank: rank + 1,
                upper_score: c.decision.upper_score,
                scored_categories: c.decision.scored,
                dice: c.decision.dice,
                turn: c.decision.turn,
                decision_type: c.decision.decision_type.as_str().to_string(),
                category_scores: c.decision.category_scores,
                visit_count: c.visit_count,
                visit_fraction,
                best_action: best,
                runner_up_action: runner_up,
                ev_gap: c.ev_gap,
                difficulty_score: c.difficulty_score,
                all_actions: c.actions.clone(),
                description: desc,
            }
        })
        .collect();

    let _ = std::fs::create_dir_all(output_dir);
    let json_path = format!("{}/difficult_scenarios.json", output_dir);
    let json = serde_json::to_string_pretty(&json_scenarios).expect("JSON serialization failed");
    let mut f = std::fs::File::create(&json_path).expect("Failed to create JSON");
    f.write_all(json.as_bytes()).unwrap();
    println!("Wrote {}", json_path);

    // Console summary
    println!(
        "\n=== Top 20 Most Difficult Scenarios ===\n{:>4} {:>4} {:>10} {:>14} {:>6} {:>9} {:>8} {:>20} {:>20}",
        "Rank", "Turn", "Type", "Dice", "Visits", "Gap", "DScore", "Best Action", "Runner-Up"
    );
    println!("{}", "-".repeat(105));
    for s in json_scenarios.iter().take(20) {
        println!(
            "{:>4} {:>4} {:>10} {:>14} {:>6} {:>9.4} {:>8.0} {:>20} {:>20}",
            s.rank,
            s.turn + 1,
            s.decision_type,
            format!("{:?}", s.dice),
            s.visit_count,
            s.ev_gap,
            s.difficulty_score,
            truncate_str(&s.best_action.label, 20),
            truncate_str(&s.runner_up_action.label, 20),
        );
    }

    // Stratification summary
    let mut by_turn: HashMap<usize, usize> = HashMap::new();
    for s in &json_scenarios {
        *by_turn.entry(s.turn).or_default() += 1;
    }
    println!(
        "\nStratified: {} scenarios across {} turns",
        json_scenarios.len(),
        by_turn.len()
    );
}

fn cmd_select_diagnostic(
    args: &Args,
    ctx: &YatzyContext,
    candidates: Vec<(RawDecision, usize)>,
    output_dir: &str,
) {
    // Load theta tables for classification
    println!("Loading θ strategy tables for diagnostics...");
    let all_theta_files = discover_theta_files();
    println!("  Found {} .bin files", all_theta_files.len());

    let classification_thetas: Vec<f32> = vec![-0.05, -0.02, 0.0, 0.02, 0.05, 0.1];
    let mut theta_tables: Vec<(f32, StateValues)> = Vec::new();

    for &target_theta in &classification_thetas {
        if let Some((actual_theta, path)) = all_theta_files.iter().min_by(|(a, _), (b, _)| {
            (a - target_theta)
                .abs()
                .partial_cmp(&(b - target_theta).abs())
                .unwrap()
        }) {
            if (actual_theta - target_theta).abs() < 0.02
                && !theta_tables
                    .iter()
                    .any(|(t, _)| (t - actual_theta).abs() < 0.001)
            {
                if let Some(sv) = load_state_values_standalone(path) {
                    println!("  Loaded θ={:.3} from {}", actual_theta, path);
                    theta_tables.push((*actual_theta, sv));
                }
            }
        }
    }

    let file0 = state_file_path(0.0);
    if !theta_tables.iter().any(|(t, _)| *t == 0.0) {
        if let Some(sv) = load_state_values_standalone(&file0) {
            theta_tables.push((0.0, sv));
        }
    }

    println!(
        "Building master pool (max {} per bucket)...",
        args.pool_size
    );
    let pool = build_master_pool(ctx, candidates, &theta_tables, args.pool_size);

    println!("Assembling {} quiz scenarios...", args.quiz_size);
    let selected = assemble_quiz(&pool, args.quiz_size);
    println!("Selected {} scenarios", selected.len());

    let scenarios = build_profiling_scenarios(ctx, &pool, &selected);

    // Q-grid computation
    println!("Computing Q-value grids...");
    let grid_start = Instant::now();

    let q_theta_values: Vec<f32> = vec![-0.05, -0.02, 0.0, 0.02, 0.05, 0.1];
    let q_gamma_values: Vec<f32> = vec![0.3, 0.6, 0.8, 0.9, 0.95, 1.0];
    let q_d_values: Vec<u32> = vec![8, 20, 999];

    let grid_theta_tables: Vec<(f32, StateValues)> = {
        let mut tables: Vec<(f32, StateValues)> = Vec::new();
        for &target in &q_theta_values {
            if let Some((actual, path)) = all_theta_files.iter().min_by(|(a, _), (b, _)| {
                (a - target).abs().partial_cmp(&(b - target).abs()).unwrap()
            }) {
                if (actual - target).abs() < 0.02
                    && !tables.iter().any(|(t, _)| (t - actual).abs() < 0.001)
                {
                    if let Some(sv) = load_state_values_standalone(path) {
                        tables.push((*actual, sv));
                    }
                }
            }
        }
        tables
    };

    let mut output_scenarios: Vec<ProfileScenarioJson> = Vec::with_capacity(scenarios.len());
    for scenario in &scenarios {
        let q_grid_map = compute_q_grid(
            ctx,
            scenario,
            &grid_theta_tables,
            &q_gamma_values,
            &q_d_values,
        );

        let rerolls_remaining = match scenario.decision_type {
            DecisionType::Reroll1 => 2,
            DecisionType::Reroll2 => 1,
            DecisionType::Category => 0,
        };

        let actions: Vec<OutputAction> = scenario
            .actions
            .iter()
            .take(5)
            .map(|a| OutputAction {
                id: a.id,
                label: a.label.clone(),
                ev_theta0: a.ev,
            })
            .collect();

        output_scenarios.push(ProfileScenarioJson {
            id: scenario.id,
            upper_score: scenario.upper_score,
            scored_categories: scenario.scored_categories,
            dice: scenario.dice,
            rerolls_remaining,
            turn: scenario.turn,
            decision_type: scenario.decision_type.as_str().to_string(),
            quadrant: scenario.quadrant.as_str().to_string(),
            actions,
            optimal_action_id: scenario.optimal_action_id,
            gap: scenario.ev_gap,
            description: scenario.description.clone(),
            q_grid: QGridJson {
                theta_values: grid_theta_tables.iter().map(|(t, _)| *t).collect(),
                gamma_values: q_gamma_values.clone(),
                d_values: q_d_values.clone(),
                q_values: q_grid_map,
            },
        });
    }

    println!(
        "  Q-grid done in {:.1}s",
        grid_start.elapsed().as_secs_f64()
    );

    let _ = std::fs::create_dir_all(output_dir);
    let json_path = format!("{}/scenarios.json", output_dir);
    let output = ProfileOutputRoot {
        scenarios: output_scenarios,
    };
    let json = serde_json::to_string_pretty(&output).expect("JSON serialization failed");
    let json_size = json.len();
    let mut f = std::fs::File::create(&json_path).expect("Failed to create JSON");
    f.write_all(json.as_bytes()).unwrap();
    println!("Wrote {} ({:.1} KB)", json_path, json_size as f64 / 1024.0);

    // Write master_pool.json
    let pool_json_path = format!("{}/master_pool.json", output_dir);
    let pool_output = PoolRoot {
        total_candidates: pool.len(),
        candidates: pool
            .iter()
            .map(|sc| PoolCandidate {
                upper_score: sc.decision.upper_score,
                scored_categories: sc.decision.scored,
                dice: sc.decision.dice,
                turn: sc.decision.turn,
                decision_type: sc.decision.decision_type.as_str().to_string(),
                phase: sc.bucket.phase.as_str().to_string(),
                tension: sc.bucket.tension.as_str().to_string(),
                visit_count: sc.visit_count,
                ev_gap: sc.ev_gap,
                s_theta: sc.scores.s_theta,
                s_gamma: sc.scores.s_gamma,
                s_d: sc.scores.s_d,
                s_beta: sc.scores.s_beta,
            })
            .collect(),
    };

    let pool_json =
        serde_json::to_string_pretty(&pool_output).expect("Pool JSON serialization failed");
    let mut pf = std::fs::File::create(&pool_json_path).expect("Failed to create pool JSON");
    pf.write_all(pool_json.as_bytes()).unwrap();
    println!("Wrote {}", pool_json_path);

    // Copy to profiler/data/ if exists
    let profiler_data_dir = "profiler/data";
    if std::path::Path::new(profiler_data_dir).exists() {
        let profiler_dest = format!("{}/scenarios.json", profiler_data_dir);
        if std::fs::copy(&json_path, &profiler_dest).is_ok() {
            println!("Copied to {}", profiler_dest);
        }
    }
}

fn cmd_enrich_sensitivity(args: &Args, output_dir: &str) {
    // Read difficult_scenarios.json
    let input_path = format!("{}/difficult_scenarios.json", output_dir);
    println!("Reading scenarios from {}...", input_path);

    let json_str = match std::fs::read_to_string(&input_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read {}: {}", input_path, e);
            std::process::exit(1);
        }
    };
    let scenarios: Vec<DifficultScenarioJson> = match serde_json::from_str(&json_str) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to parse JSON: {}", e);
            std::process::exit(1);
        }
    };
    println!("  {} scenarios loaded", scenarios.len());

    // Load θ tables
    let all_theta_files = discover_theta_files();
    let theta_entries = load_theta_entries(&all_theta_files, args.theta_min, args.theta_max);
    println!("  Loaded {} θ tables", theta_entries.len());

    if theta_entries.is_empty() || !theta_entries.iter().any(|e| e.theta == 0.0) {
        eprintln!("θ=0 table required but not found.");
        std::process::exit(1);
    }

    // Reload ctx for evaluation
    let _base = yatzy::env_config::init_base_path();

    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    println!(
        "Evaluating {} scenarios × {} θ values...",
        scenarios.len(),
        theta_entries.len(),
    );
    let eval_start = Instant::now();

    let output_scenarios: Vec<SensitivityScenarioJson> = scenarios
        .par_iter()
        .map(|s| {
            let params = ScenarioParams {
                upper_score: s.upper_score,
                scored_categories: s.scored_categories,
                dice: &s.dice,
                turn: s.turn,
                decision_type: &s.decision_type,
            };
            let (
                theta_results,
                has_flip,
                flip_theta,
                flip_action,
                flip_action_id,
                gap_at_flip,
                gap_at_theta0,
            ) = evaluate_scenario_sensitivity(&ctx, &theta_entries, &params);

            let theta0_result = theta_results
                .iter()
                .find(|r| r.theta == 0.0)
                .expect("θ=0 missing");

            SensitivityScenarioJson {
                rank: s.rank,
                upper_score: s.upper_score,
                scored_categories: s.scored_categories,
                dice: s.dice,
                turn: s.turn,
                decision_type: s.decision_type.clone(),
                game_phase: classify_phase(s.turn).as_str().to_string(),
                visit_count: s.visit_count,
                difficulty_score: s.difficulty_score,
                description: s.description.clone(),
                theta_0_action: theta0_result.action.clone(),
                theta_0_action_id: theta0_result.action_id,
                has_flip,
                flip_theta,
                flip_action,
                flip_action_id,
                gap_at_flip,
                gap_at_theta0,
                theta_results,
            }
        })
        .collect();

    println!("  Done in {:.1}s", eval_start.elapsed().as_secs_f64());

    let flip_count = output_scenarios.iter().filter(|s| s.has_flip).count();
    println!(
        "  {} scenarios with flips ({:.1}%)",
        flip_count,
        100.0 * flip_count as f64 / output_scenarios.len() as f64,
    );

    let json_path = format!("{}/difficult_scenarios_sensitivity.json", output_dir);
    let json = serde_json::to_string_pretty(&output_scenarios).expect("JSON serialization failed");
    let mut f = std::fs::File::create(&json_path).expect("Failed to create JSON");
    f.write_all(json.as_bytes()).unwrap();
    println!("Wrote {}", json_path);
}

fn main() {
    let args = parse_args();
    let output_dir = resolve_output_dir(&args.output_dir);
    let total_start = Instant::now();

    match args.subcommand.as_str() {
        "collect" => {
            let ctx = setup(&args);
            let filtered = cmd_collect(&args, &ctx);
            println!("{} candidates ready for selection", filtered.len());
        }
        "select" => {
            let ctx = setup(&args);
            let filtered = cmd_collect(&args, &ctx);
            match args.mode.as_str() {
                "difficulty" => {
                    cmd_select_difficulty(&args, &ctx, &filtered, &output_dir);
                }
                "diagnostic" => {
                    cmd_select_diagnostic(&args, &ctx, filtered, &output_dir);
                }
                other => {
                    eprintln!("Unknown mode: {}. Use 'difficulty' or 'diagnostic'.", other);
                    std::process::exit(1);
                }
            }
        }
        "enrich" => {
            if args.sensitivity {
                cmd_enrich_sensitivity(&args, &output_dir);
            } else {
                eprintln!("Specify --sensitivity or --q-grid");
                std::process::exit(1);
            }
        }
        "all" => {
            let ctx = setup(&args);
            let filtered = cmd_collect(&args, &ctx);
            match args.mode.as_str() {
                "difficulty" => {
                    cmd_select_difficulty(&args, &ctx, &filtered, &output_dir);
                    if args.sensitivity {
                        cmd_enrich_sensitivity(&args, &output_dir);
                    }
                }
                "diagnostic" => {
                    cmd_select_diagnostic(&args, &ctx, filtered, &output_dir);
                }
                other => {
                    eprintln!("Unknown mode: {}", other);
                    std::process::exit(1);
                }
            }
        }
        other => {
            eprintln!(
                "Unknown subcommand: {}. Use collect, select, enrich, or all.",
                other
            );
            print_usage();
            std::process::exit(1);
        }
    }

    println!("\nTotal: {:.1}s", total_start.elapsed().as_secs_f64());
}
