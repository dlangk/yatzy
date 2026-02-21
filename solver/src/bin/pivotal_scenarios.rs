//! Pivotal scenario generator for θ estimation.
//!
//! Simulates games under θ=0, extracts every category decision point, then checks
//! which scenarios produce different optimal-category choices across a grid of θ values.
//! These "pivotal" scenarios are where different risk preferences lead to different
//! actions — ideal for a Bayesian questionnaire to infer a human's hidden θ.
//!
//! Output: JSON array of ~200 diverse, high-information scenarios.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

use yatzy::constants::*;
use yatzy::dice_mechanics::{find_dice_set_index, sort_dice_set};
use yatzy::game_mechanics::update_upper_score;
use yatzy::phase0_tables;
use yatzy::storage::{load_state_values_standalone, state_file_path};
use yatzy::types::{StateValues, YatzyContext};
use yatzy::widget_solver::{choose_best_reroll_mask, compute_max_ev_for_n_rerolls};

/// θ grid for scenario generation: progressive spacing, dense near 0, sparse at tails.
/// 37 values from -3.0 to +3.0, symmetric around 0.
const THETA_GRID: [f32; 37] = [
    -3.00, -2.00, -1.50, -1.00, -0.75, -0.50, -0.30, -0.200, -0.150, -0.100, -0.070, -0.050,
    -0.040, -0.030, -0.020, -0.015, -0.010, -0.005, 0.000, 0.005, 0.010, 0.015, 0.020, 0.030,
    0.040, 0.050, 0.070, 0.100, 0.150, 0.200, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00,
];

/// Target number of scenarios to output.
const TARGET_SCENARIOS: usize = 200;

/// A raw scenario extracted from simulation: the state + dice at a category decision point.
#[derive(Clone)]
struct RawScenario {
    upper_score: i32,
    scored: i32,
    dice: [i32; 5],
    turn: usize,                // 0-indexed turn number
    category_scores: [i32; 15], // score placed in each scored category (-1 = not yet scored)
}

/// Per-category value under a specific θ.
#[derive(Clone, Serialize)]
struct CategoryOption {
    id: usize,
    name: String,
    score: i32,
    /// V_θ(c, s) = score(c) + sv_θ[successor] for each θ in the grid
    values: HashMap<String, f32>,
}

/// A pivotal scenario where the optimal category changes across θ values.
#[derive(Clone, Serialize)]
struct PivotalScenario {
    id: usize,
    upper_score: i32,
    scored_categories: i32,
    scored_details: Vec<ScoredDetail>,
    dice: [i32; 5],
    turn: usize,
    available: Vec<CategoryOption>,
    switch_theta: f32,
    theta_optimal: HashMap<String, usize>,
    fisher_score: f32,
}

#[derive(Clone, Serialize)]
struct ScoredDetail {
    id: usize,
    name: String,
    score: i32,
}

/// Loaded θ table: state values + metadata.
struct ThetaEntry {
    theta: f32,
    sv: StateValues,
}

// ── Simulation helpers (extracted from engine.rs, simplified for scenario extraction) ──

fn roll_dice(rng: &mut SmallRng) -> [i32; 5] {
    let mut dice = [0i32; 5];
    for d in &mut dice {
        *d = rng.random_range(1..=6);
    }
    sort_dice_set(&mut dice);
    dice
}

fn apply_reroll(dice: &mut [i32; 5], mask: i32, rng: &mut SmallRng) {
    for i in 0..5 {
        if mask & (1 << i) != 0 {
            dice[i] = rng.random_range(1..=6);
        }
    }
    sort_dice_set(dice);
}

/// Simulate one game under θ=0, collecting all category decision points.
fn simulate_game_collecting(ctx: &YatzyContext, rng: &mut SmallRng) -> Vec<RawScenario> {
    let sv = ctx.state_values.as_slice();
    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut category_scores = [-1i32; 15]; // -1 = not yet scored
    let mut scenarios = Vec::with_capacity(15);

    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;
        let mut dice = roll_dice(rng);

        // Compute EVs for reroll decisions
        compute_group6(ctx, sv, up_score, scored, &mut e_ds_0);
        compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);

        let mut best_ev = 0.0;
        let mask1 = choose_best_reroll_mask(ctx, &e_ds_1, &dice, &mut best_ev);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }

        let mask2 = choose_best_reroll_mask(ctx, &e_ds_0, &dice, &mut best_ev);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }

        // Record the scenario (state + final dice) before scoring
        scenarios.push(RawScenario {
            upper_score: up_score,
            scored,
            dice,
            turn,
            category_scores,
        });

        // Score optimally under θ=0
        let ds_index = find_dice_set_index(ctx, &dice);
        let (cat, scr) = if is_last_turn {
            find_best_category_final(ctx, up_score, scored, ds_index)
        } else {
            find_best_category(ctx, sv, up_score, scored, ds_index)
        };

        category_scores[cat] = scr;
        up_score = update_upper_score(up_score, cat, scr);
        scored |= 1 << cat;
    }

    scenarios
}

/// Standard Group 6 computation (copied from engine.rs for self-containedness).
#[inline(always)]
fn compute_group6(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    e_ds_0: &mut [f32; 252],
) {
    let mut lower_succ_ev = [0.0f32; CATEGORY_COUNT];
    for c in 6..CATEGORY_COUNT {
        if !is_category_scored(scored, c) {
            lower_succ_ev[c] = unsafe {
                *sv.get_unchecked(state_index(up_score as usize, (scored | (1 << c)) as usize))
            };
        }
    }

    for ds_i in 0..252 {
        let mut best_val = f32::NEG_INFINITY;
        for c in 0..6 {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_i][c];
                let new_up = update_upper_score(up_score, c, scr);
                let new_scored = scored | (1 << c);
                let val = scr as f32
                    + unsafe {
                        *sv.get_unchecked(state_index(new_up as usize, new_scored as usize))
                    };
                if val > best_val {
                    best_val = val;
                }
            }
        }
        for c in 6..CATEGORY_COUNT {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_i][c];
                let val = scr as f32 + unsafe { *lower_succ_ev.get_unchecked(c) };
                if val > best_val {
                    best_val = val;
                }
            }
        }
        e_ds_0[ds_i] = best_val;
    }
}

fn find_best_category(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    ds_index: usize,
) -> (usize, i32) {
    let mut best_val = f32::NEG_INFINITY;
    let mut best_cat = 0usize;
    let mut best_score = 0i32;

    for c in 0..6 {
        if !is_category_scored(scored, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            let new_up = update_upper_score(up_score, c, scr);
            let new_scored = scored | (1 << c);
            let val = scr as f32
                + unsafe { *sv.get_unchecked(state_index(new_up as usize, new_scored as usize)) };
            if val > best_val {
                best_val = val;
                best_cat = c;
                best_score = scr;
            }
        }
    }

    for c in 6..CATEGORY_COUNT {
        if !is_category_scored(scored, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            let new_scored = scored | (1 << c);
            let val = scr as f32
                + unsafe { *sv.get_unchecked(state_index(up_score as usize, new_scored as usize)) };
            if val > best_val {
                best_val = val;
                best_cat = c;
                best_score = scr;
            }
        }
    }

    (best_cat, best_score)
}

fn find_best_category_final(
    ctx: &YatzyContext,
    up_score: i32,
    scored: i32,
    ds_index: usize,
) -> (usize, i32) {
    let mut best_val = i32::MIN;
    let mut best_cat = 0usize;
    let mut best_score = 0i32;

    for c in 0..CATEGORY_COUNT {
        if !is_category_scored(scored, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            let bonus = if c < 6 {
                let new_up = update_upper_score(up_score, c, scr);
                if new_up >= 63 && up_score < 63 {
                    50
                } else {
                    0
                }
            } else {
                0
            };
            let val = scr + bonus;
            if val > best_val {
                best_val = val;
                best_cat = c;
                best_score = scr;
            }
        }
    }

    (best_cat, best_score)
}

/// For a given scenario + θ table, compute V(c, s) = score(c) + sv_θ[successor] for each
/// available category, and return the optimal category index.
fn compute_category_values(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    ds_index: usize,
    theta: f32,
    is_last_turn: bool,
) -> Vec<(usize, i32, f32)> {
    // (category, score, value)
    let mut results = Vec::new();

    if is_last_turn {
        // Last turn: no successor, value = score + bonus
        for c in 0..CATEGORY_COUNT {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let bonus = if c < 6 {
                    let new_up = update_upper_score(up_score, c, scr);
                    if new_up >= 63 && up_score < 63 {
                        50
                    } else {
                        0
                    }
                } else {
                    0
                };
                let val = if theta == 0.0 {
                    (scr + bonus) as f32
                } else {
                    theta * (scr + bonus) as f32
                };
                results.push((c, scr, val));
            }
        }
    } else if theta == 0.0 {
        // EV mode
        for c in 0..6 {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let new_up = update_upper_score(up_score, c, scr);
                let new_scored = scored | (1 << c);
                let val = scr as f32
                    + unsafe {
                        *sv.get_unchecked(state_index(new_up as usize, new_scored as usize))
                    };
                results.push((c, scr, val));
            }
        }
        for c in 6..CATEGORY_COUNT {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let new_scored = scored | (1 << c);
                let val = scr as f32
                    + unsafe {
                        *sv.get_unchecked(state_index(up_score as usize, new_scored as usize))
                    };
                results.push((c, scr, val));
            }
        }
    } else {
        // Risk-sensitive mode: val = θ·score + sv_θ[successor]
        for c in 0..6 {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let new_up = update_upper_score(up_score, c, scr);
                let new_scored = scored | (1 << c);
                let val = theta * scr as f32
                    + unsafe {
                        *sv.get_unchecked(state_index(new_up as usize, new_scored as usize))
                    };
                results.push((c, scr, val));
            }
        }
        for c in 6..CATEGORY_COUNT {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let new_scored = scored | (1 << c);
                let val = theta * scr as f32
                    + unsafe {
                        *sv.get_unchecked(state_index(up_score as usize, new_scored as usize))
                    };
                results.push((c, scr, val));
            }
        }
    }

    results
}

/// Check whether a scenario looks like something a human would encounter.
///
/// Optimal θ=0 play aggressively dumps categories (e.g. scoring 0 in Sixes or Full House
/// in turns 1-2), producing states no human would reach. This filter rejects such scenarios.
fn is_realistic(scenario: &RawScenario) -> bool {
    let turn = scenario.turn; // 0-indexed

    // Count zero-score categories among those already scored
    let mut zero_count = 0;
    let mut scored_sum = 0i32;
    let mut upper_scored_count = 0;
    for c in 0..CATEGORY_COUNT {
        if is_category_scored(scenario.scored, c) {
            let scr = scenario.category_scores[c];
            if scr == 0 {
                zero_count += 1;
            }
            scored_sum += scr;
            if c < 6 {
                upper_scored_count += 1;
            }
        }
    }

    // Zero-dump limit: at most 1 zero per 3 turns played
    let max_zeros = (turn + 3) / 3; // turn 0-2 → 1, turn 3-5 → 2, etc.
    if zero_count > max_zeros {
        return false;
    }

    // Upper section sanity: if ≤5 turns played and ≥3 upper categories scored,
    // upper_score shouldn't be 0 (a human wouldn't zero-out 3+ upper categories early)
    if turn <= 5 && upper_scored_count >= 3 && scenario.upper_score == 0 {
        return false;
    }

    // Minimum scored sum: at least ~5 pts per turn played (very conservative)
    if turn >= 3 && scored_sum < (turn as i32) * 5 {
        return false;
    }

    true
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 100_000usize;
    let mut seed = 42u64;
    let mut output_path = String::from("outputs/scenarios/pivotal_scenarios.json");

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
                output_path = args[i].clone();
            }
            "--help" | "-h" => {
                println!("Usage: pivotal-scenarios [--games N] [--seed S] [--output FILE]");
                println!("  Generates pivotal scenarios for θ estimation questionnaire.");
                println!("  Requires precomputed state files for θ=0.00 through θ=0.20.");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Resolve output path to absolute before changing directory
    let output_path = if std::path::Path::new(&output_path).is_absolute() {
        output_path
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&output_path).to_string_lossy().to_string())
            .unwrap_or(output_path)
    };

    // Change to base path
    let _base = yatzy::env_config::init_base_path();

    // Configure rayon
    let _threads = yatzy::env_config::init_rayon_threads();

    let total_start = Instant::now();

    // Phase 0: precompute lookup tables
    println!("Precomputing lookup tables...");
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    // Load all θ tables
    println!("Loading {} θ tables...", THETA_GRID.len());
    let mut theta_entries: Vec<ThetaEntry> = Vec::new();
    for &theta in &THETA_GRID {
        let file = state_file_path(theta);
        match load_state_values_standalone(&file) {
            Some(sv) => {
                println!("  θ={:.2}: loaded {}", theta, file);
                theta_entries.push(ThetaEntry { theta, sv });
            }
            None => {
                eprintln!(
                    "Failed to load {}. Run yatzy-precompute --theta {} first.",
                    file, theta
                );
                std::process::exit(1);
            }
        }
    }

    // Load θ=0 into main context for simulation
    let file0 = state_file_path(0.0);
    if !yatzy::storage::load_all_state_values(&mut ctx, &file0) {
        eprintln!("Failed to load θ=0 state values");
        std::process::exit(1);
    }

    // Step 1: Simulate games under θ=0, collecting all category decision points
    println!("\nSimulating {} games to collect scenarios...", num_games);
    let sim_start = Instant::now();

    let all_scenarios: Vec<RawScenario> = (0..num_games)
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
            simulate_game_collecting(&ctx, &mut rng)
        })
        .collect();

    println!(
        "  Collected {} scenarios in {:.1}s",
        all_scenarios.len(),
        sim_start.elapsed().as_secs_f64()
    );

    // Step 2: Deduplicate scenarios by (upper_score, scored, dice)
    println!("Deduplicating scenarios...");
    let mut unique_map: HashMap<(i32, i32, [i32; 5]), RawScenario> = HashMap::new();
    for s in &all_scenarios {
        let key = (s.upper_score, s.scored, s.dice);
        unique_map.entry(key).or_insert_with(|| s.clone());
    }
    let unique_scenarios: Vec<RawScenario> = unique_map.into_values().collect();
    println!("  {} unique scenarios", unique_scenarios.len());

    // Step 3: For each unique scenario, compute optimal category under each θ
    println!(
        "Finding pivotal scenarios (checking {} × {} θ values)...",
        unique_scenarios.len(),
        THETA_GRID.len()
    );
    let analysis_start = Instant::now();

    let pivotal_candidates: Vec<PivotalScenario> = unique_scenarios
        .par_iter()
        .enumerate()
        .filter_map(|(idx, scenario)| {
            let ds_index = find_dice_set_index(&ctx, &scenario.dice);
            let is_last = scenario.turn == CATEGORY_COUNT - 1;

            // Count available categories
            let mut num_available = 0;
            for c in 0..CATEGORY_COUNT {
                if !is_category_scored(scenario.scored, c) {
                    num_available += 1;
                }
            }
            // Skip trivial scenarios (only 1 category left, or last turn)
            if num_available <= 1 || is_last {
                return None;
            }

            // Skip unrealistic scenarios (too many zero-dumps, etc.)
            if !is_realistic(scenario) {
                return None;
            }

            // Compute optimal category under each θ
            let mut theta_optimal: HashMap<String, usize> = HashMap::new();
            let mut all_category_values: Vec<Vec<(usize, i32, f32)>> = Vec::new();

            for entry in &theta_entries {
                let sv = entry.sv.as_slice();
                let values = compute_category_values(
                    &ctx,
                    sv,
                    scenario.upper_score,
                    scenario.scored,
                    ds_index,
                    entry.theta,
                    is_last,
                );

                // Find optimal: for θ>0 maximize, for θ<0 minimize, θ=0 maximize
                let best = if entry.theta < 0.0 {
                    values.iter().min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
                } else {
                    values.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
                };

                if let Some(&(cat, _, _)) = best {
                    theta_optimal.insert(format!("{:.3}", entry.theta), cat);
                }

                all_category_values.push(values);
            }

            // Check if the optimal category changes across the θ grid
            let opt_cats: Vec<usize> = theta_optimal.values().copied().collect();
            let first = opt_cats[0];
            let has_switch = opt_cats.iter().any(|&c| c != first);

            if !has_switch {
                return None;
            }

            // Find the switch point: lowest θ where optimal category changes from θ=0's choice
            let theta0_cat = theta_optimal[&format!("{:.3}", 0.0f32)];
            let mut switch_theta = 0.20f32;
            for (ti, entry) in theta_entries.iter().enumerate() {
                if ti > 0 {
                    let cat = theta_optimal[&format!("{:.3}", entry.theta)];
                    if cat != theta0_cat {
                        switch_theta = entry.theta;
                        break;
                    }
                }
            }

            // Fisher information proxy: 1/|gap| at switch point, weighted by position
            let gap = {
                let switch_idx = theta_entries
                    .iter()
                    .position(|e| e.theta == switch_theta)
                    .unwrap_or(0);
                if switch_idx > 0 && switch_idx < all_category_values.len() {
                    let vals = &all_category_values[switch_idx];
                    let mut sorted_vals: Vec<f32> = vals.iter().map(|v| v.2).collect();
                    sorted_vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    if sorted_vals.len() >= 2 {
                        (sorted_vals[0] - sorted_vals[1]).abs()
                    } else {
                        f32::INFINITY
                    }
                } else {
                    f32::INFINITY
                }
            };

            // Score: smaller gap = more informative (action is close to indifferent)
            // Weight by distance from edges (middle of grid is better)
            let edge_weight = 1.0 - ((switch_theta - 0.10).abs() / 0.10).min(1.0);
            let fisher_score = if gap > 0.001 {
                (1.0 / gap) * (0.5 + edge_weight)
            } else {
                1000.0 // Very close gap = highly informative
            };

            // Build the available categories with per-θ values
            let mut available: Vec<CategoryOption> = Vec::new();
            // Use the θ=0 values as base, but include all θ values
            for c in 0..CATEGORY_COUNT {
                if !is_category_scored(scenario.scored, c) {
                    let score = ctx.precomputed_scores[ds_index][c];
                    let mut values_map = HashMap::new();
                    for (ti, entry) in theta_entries.iter().enumerate() {
                        if let Some(cv) = all_category_values[ti].iter().find(|v| v.0 == c) {
                            values_map.insert(format!("{:.3}", entry.theta), cv.2);
                        }
                    }
                    available.push(CategoryOption {
                        id: c,
                        name: CATEGORY_NAMES[c].to_string(),
                        score,
                        values: values_map,
                    });
                }
            }

            // Build scored details with actual per-category scores
            let scored_details: Vec<ScoredDetail> = (0..CATEGORY_COUNT)
                .filter(|&c| is_category_scored(scenario.scored, c))
                .map(|c| ScoredDetail {
                    id: c,
                    name: CATEGORY_NAMES[c].to_string(),
                    score: scenario.category_scores[c],
                })
                .collect();

            Some(PivotalScenario {
                id: idx,
                upper_score: scenario.upper_score,
                scored_categories: scenario.scored,
                scored_details,
                dice: scenario.dice,
                turn: scenario.turn + 1, // 1-indexed for display
                available,
                switch_theta,
                theta_optimal,
                fisher_score,
            })
        })
        .collect();

    println!(
        "  Found {} pivotal scenarios in {:.1}s",
        pivotal_candidates.len(),
        analysis_start.elapsed().as_secs_f64()
    );

    // Step 4: Diverse selection — spread across game phases and switch thetas
    println!("Selecting {} diverse scenarios...", TARGET_SCENARIOS);

    let mut selected = select_diverse(&pivotal_candidates, TARGET_SCENARIOS);

    // Renumber IDs
    for (i, s) in selected.iter_mut().enumerate() {
        s.id = i;
    }

    // Step 5: Output JSON
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let json = serde_json::to_string_pretty(&selected).expect("JSON serialization failed");
    let mut f = std::fs::File::create(&output_path).expect("Failed to create output file");
    f.write_all(json.as_bytes()).expect("Failed to write JSON");

    println!("\nOutput: {} ({} scenarios)", output_path, selected.len());

    // Print summary stats
    let mut switch_hist: HashMap<String, usize> = HashMap::new();
    let mut phase_hist = [0usize; 3]; // early (1-5), mid (6-10), late (11-15)
    for s in &selected {
        *switch_hist
            .entry(format!("{:.2}", s.switch_theta))
            .or_default() += 1;
        let phase = if s.turn <= 5 {
            0
        } else if s.turn <= 10 {
            1
        } else {
            2
        };
        phase_hist[phase] += 1;
    }
    println!(
        "  Game phases: early={}, mid={}, late={}",
        phase_hist[0], phase_hist[1], phase_hist[2]
    );

    let mut switches: Vec<(String, usize)> = switch_hist.into_iter().collect();
    switches.sort_by(|a, b| a.0.cmp(&b.0));
    let switch_summary: Vec<String> = switches
        .iter()
        .map(|(t, n)| format!("θ*={}:{}", t, n))
        .collect();
    println!("  Switch points: {}", switch_summary.join(", "));

    println!("\nTotal: {:.1}s", total_start.elapsed().as_secs_f64());
}

/// Select a diverse subset of pivotal scenarios.
///
/// Strategy: partition by (game_phase, switch_theta_bucket), then pick top-scoring
/// from each bucket. This ensures coverage across different game states and θ ranges.
fn select_diverse(candidates: &[PivotalScenario], target: usize) -> Vec<PivotalScenario> {
    if candidates.len() <= target {
        return candidates.to_vec();
    }

    // Bucket by (phase, switch_theta_bucket)
    // Phase: 3 buckets (1-5, 6-10, 11-15)
    // Switch theta: 4 buckets (0.00-0.05, 0.05-0.10, 0.10-0.15, 0.15-0.20)
    let num_buckets = 3 * 4;
    let per_bucket = target / num_buckets + 1;

    let mut buckets: Vec<Vec<&PivotalScenario>> = vec![Vec::new(); num_buckets];

    for s in candidates {
        let phase = if s.turn <= 5 {
            0
        } else if s.turn <= 10 {
            1
        } else {
            2
        };
        let theta_bucket = ((s.switch_theta / 0.05).floor() as usize).min(3);
        let bucket_idx = phase * 4 + theta_bucket;
        buckets[bucket_idx].push(s);
    }

    // Sort each bucket by fisher_score (descending)
    for bucket in &mut buckets {
        bucket.sort_by(|a, b| b.fisher_score.partial_cmp(&a.fisher_score).unwrap());
    }

    // Take top per_bucket from each bucket
    let mut selected: Vec<PivotalScenario> = Vec::new();
    for bucket in &buckets {
        for &s in bucket.iter().take(per_bucket) {
            selected.push(s.clone());
        }
    }

    // If we have too many, sort by fisher_score and trim
    if selected.len() > target {
        selected.sort_by(|a, b| b.fisher_score.partial_cmp(&a.fisher_score).unwrap());
        selected.truncate(target);
    }

    // If we have too few (some buckets empty), fill from remaining top-scoring
    if selected.len() < target {
        let selected_set: std::collections::HashSet<usize> =
            selected.iter().map(|s| s.id).collect();
        let mut remaining: Vec<&PivotalScenario> = candidates
            .iter()
            .filter(|s| !selected_set.contains(&s.id))
            .collect();
        remaining.sort_by(|a, b| b.fisher_score.partial_cmp(&a.fisher_score).unwrap());
        for s in remaining.iter().take(target - selected.len()) {
            selected.push((*s).clone());
        }
    }

    selected
}
