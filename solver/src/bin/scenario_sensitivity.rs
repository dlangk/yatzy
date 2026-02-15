//! Scenario θ-Sensitivity: Re-evaluate difficult scenarios across all θ tables.
//!
//! Pipeline:
//! 1. Load phase0 tables (ctx)
//! 2. Scan `data/strategy_tables/` for all `all_states*.bin`, parse θ from filenames
//! 3. Load each as ThetaEntry { theta, sv } via mmap
//! 4. Read `outputs/scenarios/difficult_scenarios.json` (serde deserialize)
//! 5. For each scenario × each θ: compute best/runner-up using risk-sensitive logic
//! 6. Detect flips (best action differs from θ=0)
//! 7. Output `difficult_scenarios_sensitivity.json`

#![allow(clippy::needless_range_loop)]

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::time::Instant;

use yatzy::constants::*;
use yatzy::dice_mechanics::find_dice_set_index;
use yatzy::game_mechanics::update_upper_score;
use yatzy::phase0_tables;
use yatzy::storage::load_state_values_standalone;
use yatzy::types::{StateValues, YatzyContext};
use yatzy::widget_solver::{compute_max_ev_for_n_rerolls, compute_opt_lse_for_n_rerolls};

// ── Input types (from difficult_scenarios.json) ──

#[derive(Deserialize)]
#[allow(dead_code)]
struct InputAction {
    action_id: i32,
    action_name: String,
    ev: f32,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct InputScenario {
    rank: usize,
    upper_score: i32,
    scored_categories: i32,
    dice: [i32; 5],
    turn: usize,
    decision_type: String,
    visit_count: usize,
    #[serde(default)]
    visit_fraction: f64,
    best_action: InputAction,
    runner_up_action: InputAction,
    ev_gap: f32,
    difficulty_score: f64,
    #[serde(default)]
    all_actions: Vec<InputAction>,
    description: String,
}

// ── Output types ──

#[derive(Clone, Serialize)]
struct ThetaResult {
    theta: f32,
    action: String,
    action_id: i32,
    value: f32,
    runner_up: String,
    runner_up_id: i32,
    runner_up_value: f32,
    gap: f32,
}

#[derive(Serialize)]
struct OutputScenario {
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

/// Loaded θ table.
struct ThetaEntry {
    theta: f32,
    sv: StateValues,
}

// ── Risk-sensitive computation (adapted from decision_sensitivity.rs) ──

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

#[inline(always)]
fn compute_group6_risk(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    theta: f32,
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
                let val = theta * scr as f32
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
                let val = theta * scr as f32 + unsafe { *lower_succ_ev.get_unchecked(c) };
                if val > best_val {
                    best_val = val;
                }
            }
        }
        e_ds_0[ds_i] = best_val;
    }
}

/// Find best and runner-up reroll mask.
fn find_best_and_runner_up_mask(
    ctx: &YatzyContext,
    e_ds: &[f32; 252],
    dice: &[i32; 5],
    is_risk: bool,
) -> (i32, f32, i32, f32) {
    let ds_index = find_dice_set_index(ctx, dice);
    let kt = &ctx.keep_table;
    let vals = &kt.vals;
    let cols = &kt.cols;

    let mut options: Vec<(i32, f32)> = Vec::with_capacity(32);

    // mask=0: keep all
    options.push((0, e_ds[ds_index]));

    for j in 0..kt.unique_count[ds_index] as usize {
        let kid = kt.unique_keep_ids[ds_index][j] as usize;
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;

        let ev = if is_risk {
            // LSE for stochastic node
            let mut max_x = f32::NEG_INFINITY;
            for k in start..end {
                let v = unsafe { *e_ds.get_unchecked(*cols.get_unchecked(k) as usize) };
                if v > max_x {
                    max_x = v;
                }
            }
            let mut sum: f32 = 0.0;
            for k in start..end {
                unsafe {
                    let v = *e_ds.get_unchecked(*cols.get_unchecked(k) as usize);
                    sum += (*vals.get_unchecked(k) as f32) * (v - max_x).exp();
                }
            }
            max_x + sum.ln()
        } else {
            let mut ev: f32 = 0.0;
            for k in start..end {
                unsafe {
                    ev += (*vals.get_unchecked(k) as f32)
                        * e_ds.get_unchecked(*cols.get_unchecked(k) as usize);
                }
            }
            ev
        };

        let mask = kt.keep_to_mask[ds_index * 32 + j];
        options.push((mask, ev));
    }

    options.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let (best_mask, best_ev) = options[0];
    let (ru_mask, ru_ev) = if options.len() > 1 {
        options[1]
    } else {
        (0, best_ev)
    };

    (best_mask, best_ev, ru_mask, ru_ev)
}

/// Find best and runner-up category.
fn find_best_and_runner_up_category(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    ds_index: usize,
    theta: f32,
    is_last_turn: bool,
) -> (usize, f32, usize, f32) {
    let mut options: Vec<(usize, f32)> = Vec::with_capacity(CATEGORY_COUNT);

    if is_last_turn {
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
                options.push((c, val));
            }
        }
    } else if theta == 0.0 {
        for c in 0..6 {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let new_up = update_upper_score(up_score, c, scr);
                let new_scored = scored | (1 << c);
                let val = scr as f32
                    + unsafe {
                        *sv.get_unchecked(state_index(new_up as usize, new_scored as usize))
                    };
                options.push((c, val));
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
                options.push((c, val));
            }
        }
    } else {
        for c in 0..6 {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let new_up = update_upper_score(up_score, c, scr);
                let new_scored = scored | (1 << c);
                let val = theta * scr as f32
                    + unsafe {
                        *sv.get_unchecked(state_index(new_up as usize, new_scored as usize))
                    };
                options.push((c, val));
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
                options.push((c, val));
            }
        }
    }

    options.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let (best_cat, best_val) = options[0];
    let (ru_cat, ru_val) = if options.len() > 1 {
        options[1]
    } else {
        (best_cat, best_val)
    };

    (best_cat, best_val, ru_cat, ru_val)
}

/// Format a reroll mask as human-readable string.
fn format_mask(mask: i32, dice: &[i32; 5]) -> String {
    if mask == 0 {
        return "keep all".to_string();
    }
    if mask == 31 {
        return "reroll all".to_string();
    }
    let mut kept: Vec<i32> = Vec::new();
    for i in 0..5 {
        if mask & (1 << i) == 0 {
            kept.push(dice[i]);
        }
    }
    if kept.is_empty() {
        "reroll all".to_string()
    } else {
        format!("keep {:?}", kept)
    }
}

fn game_phase(turn: usize) -> &'static str {
    if turn < 5 {
        "early"
    } else if turn < 10 {
        "mid"
    } else {
        "late"
    }
}

/// Evaluate a single scenario under a single θ.
fn evaluate_scenario_at_theta(
    ctx: &YatzyContext,
    sv: &[f32],
    theta: f32,
    s: &InputScenario,
) -> ThetaResult {
    let ds_index = find_dice_set_index(ctx, &s.dice);
    let is_last = s.turn == CATEGORY_COUNT - 1;
    let is_risk = theta != 0.0;

    let (action_str, action_id, value, ru_str, ru_id, ru_value) = match s.decision_type.as_str() {
        "reroll1" => {
            let mut e_ds_0 = [0.0f32; 252];
            let mut e_ds_1 = [0.0f32; 252];
            if is_risk {
                compute_group6_risk(ctx, sv, s.upper_score, s.scored_categories, theta, &mut e_ds_0);
                compute_opt_lse_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1, false);
            } else {
                compute_group6(ctx, sv, s.upper_score, s.scored_categories, &mut e_ds_0);
                compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);
            }
            let (bm, bv, rm, rv) =
                find_best_and_runner_up_mask(ctx, &e_ds_1, &s.dice, is_risk);
            (
                format_mask(bm, &s.dice),
                bm,
                bv,
                format_mask(rm, &s.dice),
                rm,
                rv,
            )
        }
        "reroll2" => {
            let mut e_ds_0 = [0.0f32; 252];
            if is_risk {
                compute_group6_risk(ctx, sv, s.upper_score, s.scored_categories, theta, &mut e_ds_0);
            } else {
                compute_group6(ctx, sv, s.upper_score, s.scored_categories, &mut e_ds_0);
            }
            let (bm, bv, rm, rv) =
                find_best_and_runner_up_mask(ctx, &e_ds_0, &s.dice, is_risk);
            (
                format_mask(bm, &s.dice),
                bm,
                bv,
                format_mask(rm, &s.dice),
                rm,
                rv,
            )
        }
        "category" => {
            let (bc, bv, rc, rv) = find_best_and_runner_up_category(
                ctx,
                sv,
                s.upper_score,
                s.scored_categories,
                ds_index,
                theta,
                is_last,
            );
            (
                CATEGORY_NAMES[bc].to_string(),
                bc as i32,
                bv,
                CATEGORY_NAMES[rc].to_string(),
                rc as i32,
                rv,
            )
        }
        _ => panic!("Unknown decision type: {}", s.decision_type),
    };

    let gap = value - ru_value;

    ThetaResult {
        theta,
        action: action_str,
        action_id,
        value,
        runner_up: ru_str,
        runner_up_id: ru_id,
        runner_up_value: ru_value,
        gap,
    }
}

/// Scan data/strategy_tables/ for all .bin files and parse θ from filenames.
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
    let mut input_path = String::from("outputs/scenarios/difficult_scenarios.json");
    let mut output_dir = String::from("outputs/scenarios");
    let mut theta_min: Option<f32> = None;
    let mut theta_max: Option<f32> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                i += 1;
                input_path = args[i].clone();
            }
            "--output" => {
                i += 1;
                output_dir = args[i].clone();
            }
            "--theta-min" => {
                i += 1;
                theta_min = Some(args[i].parse().expect("Invalid --theta-min"));
            }
            "--theta-max" => {
                i += 1;
                theta_max = Some(args[i].parse().expect("Invalid --theta-max"));
            }
            "--help" | "-h" => {
                println!("Usage: yatzy-scenario-sensitivity [OPTIONS]");
                println!("  --input PATH      Input JSON (default: outputs/scenarios/difficult_scenarios.json)");
                println!("  --output DIR      Output directory (default: outputs/scenarios)");
                println!("  --theta-min F     Min θ to include (default: all)");
                println!("  --theta-max F     Max θ to include (default: all)");
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

    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    if std::env::set_current_dir(&base_path).is_err() {
        eprintln!("Failed to change directory to {}", base_path);
        std::process::exit(1);
    }

    let total_start = Instant::now();

    // Phase 0: precompute lookup tables
    println!("Precomputing lookup tables...");
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    // Discover and load θ tables
    println!("Scanning for θ strategy tables...");
    let all_theta_files = discover_theta_files();
    println!("  Found {} .bin files", all_theta_files.len());

    // Filter by theta range
    let theta_files: Vec<(f32, String)> = all_theta_files
        .into_iter()
        .filter(|(theta, _)| {
            if let Some(tmin) = theta_min {
                if *theta < tmin - 1e-6 {
                    return false;
                }
            }
            if let Some(tmax) = theta_max {
                if *theta > tmax + 1e-6 {
                    return false;
                }
            }
            true
        })
        .collect();

    println!("Loading {} θ tables...", theta_files.len());
    let mut theta_entries: Vec<ThetaEntry> = Vec::new();
    for (theta, path) in &theta_files {
        match load_state_values_standalone(path) {
            Some(sv) => {
                theta_entries.push(ThetaEntry { theta: *theta, sv });
            }
            None => {
                eprintln!("  WARNING: Failed to load {}", path);
            }
        }
    }
    println!(
        "  Loaded {} θ tables in {:.1}s",
        theta_entries.len(),
        total_start.elapsed().as_secs_f64(),
    );

    if theta_entries.is_empty() {
        eprintln!("No θ tables found. Run yatzy-precompute first.");
        std::process::exit(1);
    }

    // Ensure θ=0 is present
    if !theta_entries.iter().any(|e| e.theta == 0.0) {
        eprintln!("θ=0 table not found — required as baseline.");
        std::process::exit(1);
    }

    // Read input scenarios
    println!("Reading scenarios from {}...", input_path);
    let json_str = match std::fs::read_to_string(&input_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read {}: {}", input_path, e);
            std::process::exit(1);
        }
    };
    let scenarios: Vec<InputScenario> = match serde_json::from_str(&json_str) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to parse JSON: {}", e);
            std::process::exit(1);
        }
    };
    println!("  {} scenarios loaded", scenarios.len());

    // Evaluate each scenario across all θ values (parallel over scenarios)
    println!(
        "Evaluating {} scenarios × {} θ values...",
        scenarios.len(),
        theta_entries.len(),
    );
    let eval_start = Instant::now();

    let output_scenarios: Vec<OutputScenario> = scenarios
        .par_iter()
        .map(|s| {
            let mut theta_results: Vec<ThetaResult> = Vec::with_capacity(theta_entries.len());

            for entry in &theta_entries {
                let sv = entry.sv.as_slice();
                let tr = evaluate_scenario_at_theta(&ctx, sv, entry.theta, s);
                theta_results.push(tr);
            }

            // Find θ=0 baseline
            let theta0_result = theta_results
                .iter()
                .find(|r| r.theta == 0.0)
                .expect("θ=0 missing from results");
            let theta0_action = theta0_result.action.clone();
            let theta0_action_id = theta0_result.action_id;
            let theta0_gap = theta0_result.gap;

            // Detect first flip (action differs from θ=0)
            let mut has_flip = false;
            let mut flip_theta: f32 = 0.0;
            let mut flip_action = String::new();
            let mut flip_action_id: i32 = 0;
            let mut gap_at_flip: f32 = 0.0;

            for tr in &theta_results {
                if tr.theta != 0.0 && !has_flip && tr.action_id != theta0_action_id {
                    has_flip = true;
                    flip_theta = tr.theta;
                    flip_action = tr.action.clone();
                    flip_action_id = tr.action_id;
                    gap_at_flip = tr.gap;
                }
            }

            OutputScenario {
                rank: s.rank,
                upper_score: s.upper_score,
                scored_categories: s.scored_categories,
                dice: s.dice,
                turn: s.turn,
                decision_type: s.decision_type.clone(),
                game_phase: game_phase(s.turn).to_string(),
                visit_count: s.visit_count,
                difficulty_score: s.difficulty_score,
                description: s.description.clone(),
                theta_0_action: theta0_action,
                theta_0_action_id: theta0_action_id,
                has_flip,
                flip_theta,
                flip_action,
                flip_action_id,
                gap_at_flip,
                gap_at_theta0: theta0_gap,
                theta_results,
            }
        })
        .collect();

    println!(
        "  Done in {:.1}s",
        eval_start.elapsed().as_secs_f64(),
    );

    // Stats
    let flip_count = output_scenarios.iter().filter(|s| s.has_flip).count();
    println!(
        "  {} scenarios with flips ({:.1}%)",
        flip_count,
        100.0 * flip_count as f64 / output_scenarios.len() as f64,
    );

    // Output JSON
    let _ = std::fs::create_dir_all(&output_dir);
    let json_path = format!("{}/difficult_scenarios_sensitivity.json", output_dir);
    {
        let json = serde_json::to_string_pretty(&output_scenarios).expect("JSON serialization failed");
        let mut f = std::fs::File::create(&json_path).expect("Failed to create JSON");
        f.write_all(json.as_bytes()).unwrap();
    }
    println!("Wrote {}", json_path);

    // Console summary
    println!("\n=== Top 20 Flip Scenarios ===");
    println!(
        "{:>4} {:>4} {:>10} {:>14} {:>20} {:>20} {:>8} {:>8}",
        "Rank", "Turn", "Type", "Dice", "θ=0 Action", "Flip Action", "Flip θ", "θ=0 Gap"
    );
    println!("{}", "-".repeat(100));
    let mut flips: Vec<&OutputScenario> = output_scenarios.iter().filter(|s| s.has_flip).collect();
    flips.sort_by(|a, b| {
        a.gap_at_theta0
            .abs()
            .partial_cmp(&b.gap_at_theta0.abs())
            .unwrap()
    });
    for s in flips.iter().take(20) {
        println!(
            "{:>4} {:>4} {:>10} {:>14} {:>20} {:>20} {:>8.3} {:>8.4}",
            s.rank,
            s.turn + 1,
            s.decision_type,
            format!("{:?}", s.dice),
            truncate_str(&s.theta_0_action, 20),
            truncate_str(&s.flip_action, 20),
            s.flip_theta,
            s.gap_at_theta0,
        );
    }

    // θ distribution
    println!("\n=== θ Table Coverage ===");
    let thetas: Vec<f32> = theta_entries.iter().map(|e| e.theta).collect();
    let min_theta = thetas.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_theta = thetas.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  {} θ values from {:.3} to {:.3}",
        thetas.len(),
        min_theta,
        max_theta,
    );

    println!(
        "\nTotal: {:.1}s ({} scenarios × {} θ values)",
        total_start.elapsed().as_secs_f64(),
        output_scenarios.len(),
        theta_entries.len(),
    );
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len - 1])
    }
}
