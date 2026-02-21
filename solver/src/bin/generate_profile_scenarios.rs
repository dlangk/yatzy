//! Generate profiling scenarios for cognitive player profiling.
//!
//! Produces `outputs/profiling/scenarios.json` with 30 diagnostic scenarios
//! selected via diversity-constrained stratified sampling, plus pre-computed
//! Q-value grids. Also writes `master_pool.json` with ~200-400 candidates.
//!
//! Usage:
//!   YATZY_BASE_PATH=. solver/target/release/yatzy-profile-scenarios [OPTIONS]
//!
//! Options:
//!   --games N        Number of games to simulate (default: 100000)
//!   --seed S         Random seed (default: 42)
//!   --output DIR     Output directory (default: outputs/profiling)
//!   --quiz-size N    Number of quiz scenarios (default: 20)
//!   --pool-size N    Max candidates per bucket (default: 20)
//!   --no-noisy       Use optimal play instead of noisy simulation

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

use serde::Serialize;

use yatzy::phase0_tables;
use yatzy::profiling::scenarios::{
    assemble_quiz, build_master_pool, build_profiling_scenarios, collect_candidates,
    collect_candidates_noisy, compute_q_grid,
};
use yatzy::storage::{load_state_values_standalone, state_file_path};
use yatzy::types::{StateValues, YatzyContext};

// ── Output JSON types ──

#[derive(Serialize)]
struct OutputAction {
    id: i32,
    label: String,
    ev_theta0: f32,
}

#[derive(Serialize)]
struct QGrid {
    theta_values: Vec<f32>,
    gamma_values: Vec<f32>,
    d_values: Vec<u32>,
    /// Key: "theta,gamma,d" → array of Q-values per action
    q_values: HashMap<String, Vec<f32>>,
}

#[derive(Serialize)]
struct OutputScenario {
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
    q_grid: QGrid,
}

#[derive(Serialize)]
struct OutputRoot {
    scenarios: Vec<OutputScenario>,
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

fn discover_theta_files() -> Vec<(f32, String)> {
    let dir = "data/strategy_tables";
    let mut entries: Vec<(f32, String)> = Vec::new();

    let read_dir = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) => {
            eprintln!("Cannot read {}: {}", dir, e);
            return entries;
        }
    };

    for entry in read_dir.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".bin") {
            continue;
        }
        if name == "all_states.bin" {
            entries.push((0.0, format!("{}/{}", dir, name)));
        } else if let Some(rest) = name.strip_prefix("all_states_theta_") {
            if let Some(theta_str) = rest.strip_suffix(".bin") {
                if let Ok(theta) = theta_str.parse::<f32>() {
                    entries.push((theta, format!("{}/{}", dir, name)));
                }
            }
        }
    }

    entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    entries
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 100_000usize;
    let mut seed = 42u64;
    let mut output_dir = String::from("outputs/profiling");
    let mut quiz_size = 30usize;
    let mut pool_size = 20usize; // max per bucket
    let mut use_noisy = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                i += 1;
                num_games = args[i].parse().expect("Invalid --games");
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("Invalid --seed");
            }
            "--output" => {
                i += 1;
                output_dir = args[i].clone();
            }
            "--quiz-size" => {
                i += 1;
                quiz_size = args[i].parse().expect("Invalid --quiz-size");
            }
            "--pool-size" => {
                i += 1;
                pool_size = args[i].parse().expect("Invalid --pool-size");
            }
            "--no-noisy" => {
                use_noisy = false;
            }
            "--help" | "-h" => {
                println!("Usage: yatzy-profile-scenarios [OPTIONS]");
                println!("  --games N        Number of games to simulate (default: 100000)");
                println!("  --seed S         Random seed (default: 42)");
                println!("  --output DIR     Output directory (default: outputs/profiling)");
                println!("  --quiz-size N    Number of quiz scenarios (default: 20)");
                println!("  --pool-size N    Max candidates per bucket (default: 20)");
                println!("  --no-noisy       Use optimal play instead of noisy simulation");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Resolve paths
    let output_dir = if std::path::Path::new(&output_dir).is_absolute() {
        output_dir
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&output_dir).to_string_lossy().to_string())
            .unwrap_or(output_dir)
    };

    let _base = yatzy::env_config::init_base_path();
    let _threads = yatzy::env_config::init_rayon_threads();

    let total_start = Instant::now();

    // Phase 0: precompute lookup tables
    println!("Precomputing lookup tables...");
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    // Load θ=0 state values
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

    // Load theta tables for classification and Q-grid
    println!("Loading θ strategy tables...");
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
            if (actual_theta - target_theta).abs() < 0.02 {
                if !theta_tables
                    .iter()
                    .any(|(t, _)| (t - actual_theta).abs() < 0.001)
                {
                    match load_state_values_standalone(path) {
                        Some(sv) => {
                            println!("  Loaded θ={:.3} from {}", actual_theta, path);
                            theta_tables.push((*actual_theta, sv));
                        }
                        None => {
                            eprintln!("  WARNING: Failed to load {}", path);
                        }
                    }
                }
            }
        }
    }

    if !theta_tables.iter().any(|(t, _)| *t == 0.0) {
        if let Some(sv) = load_state_values_standalone(&file0) {
            theta_tables.push((0.0, sv));
        }
    }

    println!("  Using {} θ tables", theta_tables.len());

    // Step 1: Collect candidates
    if use_noisy {
        println!(
            "\nSimulating {} noisy games (β=3.0, γ=0.85, σ_d=4.0)...",
            num_games
        );
    } else {
        println!("\nSimulating {} optimal games...", num_games);
    }
    let sim_start = Instant::now();
    let candidates = if use_noisy {
        collect_candidates_noisy(&ctx, num_games, seed, 3.0, 0.85, 4.0)
    } else {
        collect_candidates(&ctx, num_games, seed)
    };
    println!(
        "Collected {} unique decisions in {:.1}s",
        candidates.len(),
        sim_start.elapsed().as_secs_f64(),
    );

    // Step 2: Build master pool with diagnostic scoring
    println!("\nBuilding master pool (max {} per bucket)...", pool_size);
    let pool_start = Instant::now();
    let pool = build_master_pool(&ctx, candidates, &theta_tables, pool_size);
    println!(
        "Master pool: {} candidates in {:.1}s",
        pool.len(),
        pool_start.elapsed().as_secs_f64(),
    );

    // Step 3: Assemble quiz
    println!("\nAssembling {} quiz scenarios...", quiz_size);
    let selected = assemble_quiz(&pool, quiz_size);
    println!("Selected {} scenarios", selected.len());

    // Print coverage stats
    {
        let mut phase_counts = HashMap::new();
        let mut dtype_counts = HashMap::new();
        let mut quad_counts = HashMap::new();
        for &idx in &selected {
            let sc = &pool[idx];
            *phase_counts
                .entry(sc.bucket.phase.as_str())
                .or_insert(0usize) += 1;
            *dtype_counts
                .entry(sc.bucket.dtype.as_str())
                .or_insert(0usize) += 1;
            let q = if sc.scores.s_theta >= 0.5 {
                "theta"
            } else if sc.scores.s_gamma >= 0.5 {
                "gamma"
            } else if sc.scores.s_d > 0.0 {
                "depth"
            } else {
                "beta"
            };
            *quad_counts.entry(q).or_insert(0usize) += 1;
        }
        println!("  Phases: {:?}", phase_counts);
        println!("  Types: {:?}", dtype_counts);
        println!("  Quadrants: {:?}", quad_counts);
        println!(
            "  Turns: {:?}",
            selected
                .iter()
                .map(|&i| pool[i].decision.turn + 1)
                .collect::<Vec<_>>()
        );
    }

    // Step 4: Build ProfilingScenarios
    let scenarios = build_profiling_scenarios(&ctx, &pool, &selected);

    // Step 5: Compute Q-value grids
    println!("\nComputing Q-value grids...");
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

    println!(
        "  {} θ × {} γ × {} d = {} grid points per scenario",
        grid_theta_tables.len(),
        q_gamma_values.len(),
        q_d_values.len(),
        grid_theta_tables.len() * q_gamma_values.len() * q_d_values.len(),
    );

    // Build output
    let mut output_scenarios: Vec<OutputScenario> = Vec::with_capacity(scenarios.len());

    for scenario in &scenarios {
        let q_grid_map = compute_q_grid(
            &ctx,
            scenario,
            &grid_theta_tables,
            &q_gamma_values,
            &q_d_values,
        );

        let rerolls_remaining = match scenario.decision_type {
            yatzy::profiling::scenarios::DecisionType::Reroll1 => 2,
            yatzy::profiling::scenarios::DecisionType::Reroll2 => 1,
            yatzy::profiling::scenarios::DecisionType::Category => 0,
        };

        let actions: Vec<OutputAction> = scenario
            .actions
            .iter()
            .take(5)
            .map(|a| OutputAction {
                id: a.id,
                label: a.label.clone(),
                ev_theta0: a.ev_theta0,
            })
            .collect();

        let q_grid = QGrid {
            theta_values: grid_theta_tables.iter().map(|(t, _)| *t).collect(),
            gamma_values: q_gamma_values.clone(),
            d_values: q_d_values.clone(),
            q_values: q_grid_map,
        };

        output_scenarios.push(OutputScenario {
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
            q_grid,
        });
    }

    println!(
        "  Q-grid computation done in {:.1}s",
        grid_start.elapsed().as_secs_f64()
    );

    // Write scenarios.json
    let _ = std::fs::create_dir_all(&output_dir);
    let json_path = format!("{}/scenarios.json", output_dir);

    let output = OutputRoot {
        scenarios: output_scenarios,
    };

    let json = serde_json::to_string_pretty(&output).expect("JSON serialization failed");
    let json_size = json.len();
    let mut f = std::fs::File::create(&json_path).expect("Failed to create JSON");
    f.write_all(json.as_bytes()).unwrap();

    println!(
        "\nWrote {} ({:.1} KB)",
        json_path,
        json_size as f64 / 1024.0
    );

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
    let pool_json_size = pool_json.len();
    let mut pf = std::fs::File::create(&pool_json_path).expect("Failed to create pool JSON");
    pf.write_all(pool_json.as_bytes()).unwrap();

    println!(
        "Wrote {} ({:.1} KB)",
        pool_json_path,
        pool_json_size as f64 / 1024.0
    );

    // Copy to blog/data/
    let blog_data_dir = "blog/data";
    if std::path::Path::new(blog_data_dir).exists() {
        let blog_dest = format!("{}/scenarios.json", blog_data_dir);
        if std::fs::copy(&json_path, &blog_dest).is_ok() {
            println!("Copied to {}", blog_dest);
        }
    }

    // Summary
    println!("\n=== Profiling Scenarios Summary ===");
    println!("Scenarios: {}", output.scenarios.len());
    for q in &["theta", "gamma", "depth", "beta"] {
        let count = output.scenarios.iter().filter(|s| s.quadrant == *q).count();
        let avg_gap: f32 = output
            .scenarios
            .iter()
            .filter(|s| s.quadrant == *q)
            .map(|s| s.gap)
            .sum::<f32>()
            / count.max(1) as f32;
        println!("  {}: {} scenarios, avg gap {:.3}", q, count, avg_gap);
    }

    let avg_actions: f32 = output
        .scenarios
        .iter()
        .map(|s| s.actions.len() as f32)
        .sum::<f32>()
        / output.scenarios.len() as f32;
    println!("Avg actions per scenario: {:.1}", avg_actions);

    let total_grid_entries: usize = output
        .scenarios
        .iter()
        .map(|s| s.q_grid.q_values.len())
        .sum();
    println!("Total Q-grid entries: {}", total_grid_entries);
    println!("Master pool: {} candidates", pool_output.total_candidates);

    println!("\nTotal: {:.1}s", total_start.elapsed().as_secs_f64(),);
}
