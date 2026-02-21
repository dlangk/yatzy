//! Difficult Scenarios Collector (Piece 1 of Human Skill Estimation)
//!
//! Identifies the most frequently-occurring, difficult decision scenarios
//! from EV-optimal play. "Difficult" = small EV gap between best and runner-up.
//! "Frequent" = arises often during optimal play.
//!
//! Ranking: difficulty_score = visit_count / max(gap, 0.01)
//!
//! Pipeline:
//! 1. Load θ=0 state table (mmap)
//! 2. Simulate 1M games in 100K batches, dedup into HashMap after each batch
//! 3. Filter: is_realistic() + visit_count ≥ 10
//! 4. Compute all action EVs + gap for each surviving scenario
//! 5. Rank by difficulty_score, take top N
//! 6. Output JSON + CSV + console summary

#![allow(clippy::needless_range_loop)]

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
use yatzy::types::YatzyContext;
use yatzy::widget_solver::{choose_best_reroll_mask, compute_max_ev_for_n_rerolls};

// ── Types ──

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum DecisionType {
    Reroll1,
    Reroll2,
    Category,
}

impl DecisionType {
    fn as_str(&self) -> &'static str {
        match self {
            DecisionType::Reroll1 => "reroll1",
            DecisionType::Reroll2 => "reroll2",
            DecisionType::Category => "category",
        }
    }
}

#[derive(Clone)]
struct RawDecision {
    upper_score: i32,
    scored: i32,
    dice: [i32; 5],
    turn: usize,
    decision_type: DecisionType,
    category_scores: [i32; 15],
}

type DecisionKey = (i32, i32, [i32; 5], DecisionType);

fn decision_key(d: &RawDecision) -> DecisionKey {
    (d.upper_score, d.scored, d.dice, d.decision_type)
}

#[derive(Clone, Serialize)]
struct ActionDetail {
    action_id: i32,
    action_name: String,
    ev: f32,
}

#[derive(Clone, Serialize)]
struct DifficultScenario {
    rank: usize,
    upper_score: i32,
    scored_categories: i32,
    dice: [i32; 5],
    turn: usize,
    decision_type: String,
    visit_count: usize,
    visit_fraction: f64,
    best_action: ActionDetail,
    runner_up_action: ActionDetail,
    ev_gap: f32,
    difficulty_score: f64,
    all_actions: Vec<ActionDetail>,
    description: String,
}

// ── Simulation helpers (adapted from decision_sensitivity.rs) ──

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

fn is_realistic(upper_score: i32, scored: i32, turn: usize, category_scores: &[i32; 15]) -> bool {
    let mut zero_count = 0;
    let mut scored_sum = 0i32;
    let mut upper_scored_count = 0;
    for c in 0..CATEGORY_COUNT {
        if is_category_scored(scored, c) {
            let scr = category_scores[c];
            if scr == 0 {
                zero_count += 1;
            }
            scored_sum += scr;
            if c < 6 {
                upper_scored_count += 1;
            }
        }
    }
    let max_zeros = (turn + 3) / 3;
    if zero_count > max_zeros {
        return false;
    }
    if turn <= 5 && upper_scored_count >= 3 && upper_score == 0 {
        return false;
    }
    if turn >= 3 && scored_sum < (turn as i32) * 5 {
        return false;
    }
    true
}

/// Simulate one game under θ=0, collecting all 3 decision types per turn.
fn simulate_game_collecting(ctx: &YatzyContext, rng: &mut SmallRng) -> Vec<RawDecision> {
    let sv = ctx.state_values.as_slice();
    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut category_scores = [-1i32; 15];
    let mut decisions = Vec::with_capacity(45);

    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;
        let mut dice = roll_dice(rng);

        compute_group6(ctx, sv, up_score, scored, &mut e_ds_0);
        compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);

        // Decision 1: Reroll1
        decisions.push(RawDecision {
            upper_score: up_score,
            scored,
            dice,
            turn,
            decision_type: DecisionType::Reroll1,
            category_scores,
        });

        let mut best_ev = 0.0;
        let mask1 = choose_best_reroll_mask(ctx, &e_ds_1, &dice, &mut best_ev);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }

        // Decision 2: Reroll2
        decisions.push(RawDecision {
            upper_score: up_score,
            scored,
            dice,
            turn,
            decision_type: DecisionType::Reroll2,
            category_scores,
        });

        let mask2 = choose_best_reroll_mask(ctx, &e_ds_0, &dice, &mut best_ev);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }

        // Decision 3: Category
        decisions.push(RawDecision {
            upper_score: up_score,
            scored,
            dice,
            turn,
            decision_type: DecisionType::Category,
            category_scores,
        });

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

    decisions
}

// ── Action enumeration (ALL legal actions with EVs) ──

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

/// Enumerate ALL reroll options with EVs for a given dice set and e_ds values.
fn enumerate_reroll_actions(
    ctx: &YatzyContext,
    e_ds: &[f32; 252],
    dice: &[i32; 5],
) -> Vec<ActionDetail> {
    let ds_index = find_dice_set_index(ctx, dice);
    let kt = &ctx.keep_table;

    let mut actions: Vec<ActionDetail> = Vec::with_capacity(32);

    // mask=0: keep all
    actions.push(ActionDetail {
        action_id: 0,
        action_name: "keep all".to_string(),
        ev: e_ds[ds_index],
    });

    for j in 0..kt.unique_count[ds_index] as usize {
        let kid = kt.unique_keep_ids[ds_index][j] as usize;
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;

        let mut ev: f32 = 0.0;
        for k in start..end {
            unsafe {
                ev += (*kt.vals.get_unchecked(k) as f32)
                    * e_ds.get_unchecked(*kt.cols.get_unchecked(k) as usize);
            }
        }

        let mask = kt.keep_to_mask[ds_index * 32 + j];
        actions.push(ActionDetail {
            action_id: mask,
            action_name: format_mask(mask, dice),
            ev,
        });
    }

    actions.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap());
    actions
}

/// Enumerate ALL category options with EVs.
fn enumerate_category_actions(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    ds_index: usize,
    is_last_turn: bool,
) -> Vec<ActionDetail> {
    let mut actions: Vec<ActionDetail> = Vec::with_capacity(CATEGORY_COUNT);

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
                actions.push(ActionDetail {
                    action_id: c as i32,
                    action_name: CATEGORY_NAMES[c].to_string(),
                    ev: (scr + bonus) as f32,
                });
            }
        }
    } else {
        for c in 0..6 {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let new_up = update_upper_score(up_score, c, scr);
                let new_scored = scored | (1 << c);
                let val = scr as f32
                    + unsafe {
                        *sv.get_unchecked(state_index(new_up as usize, new_scored as usize))
                    };
                actions.push(ActionDetail {
                    action_id: c as i32,
                    action_name: CATEGORY_NAMES[c].to_string(),
                    ev: val,
                });
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
                actions.push(ActionDetail {
                    action_id: c as i32,
                    action_name: CATEGORY_NAMES[c].to_string(),
                    ev: val,
                });
            }
        }
    }

    actions.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap());
    actions
}

/// Build a human-readable description for a scenario.
fn describe_scenario(
    dice: &[i32; 5],
    turn: usize,
    decision_type: &DecisionType,
    scored: i32,
    best: &ActionDetail,
    runner_up: &ActionDetail,
    gap: f32,
) -> String {
    let scored_cats: Vec<&str> = (0..CATEGORY_COUNT)
        .filter(|&c| is_category_scored(scored, c))
        .map(|c| CATEGORY_NAMES[c])
        .collect();
    let remaining = CATEGORY_COUNT - scored_cats.len();

    let phase = match decision_type {
        DecisionType::Reroll1 => "1st reroll",
        DecisionType::Reroll2 => "2nd reroll",
        DecisionType::Category => "category choice",
    };

    format!(
        "Turn {} ({}), dice {:?}, {} remaining categories. {}: {} (EV {:.2}) vs {} (EV {:.2}), gap {:.3}",
        turn + 1,
        phase,
        dice,
        remaining,
        phase,
        best.action_name,
        best.ev,
        runner_up.action_name,
        runner_up.ev,
        gap,
    )
}

/// Analyze a single deduped decision: enumerate all actions, compute gap, build scenario.
fn analyze_scenario(
    ctx: &YatzyContext,
    d: &RawDecision,
    visit_count: usize,
    total_decisions_of_type: usize,
    rank: usize,
) -> DifficultScenario {
    let sv = ctx.state_values.as_slice();
    let ds_index = find_dice_set_index(ctx, &d.dice);
    let is_last = d.turn == CATEGORY_COUNT - 1;

    let all_actions = match d.decision_type {
        DecisionType::Reroll1 => {
            let mut e_ds_0 = [0.0f32; 252];
            let mut e_ds_1 = [0.0f32; 252];
            compute_group6(ctx, sv, d.upper_score, d.scored, &mut e_ds_0);
            compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);
            enumerate_reroll_actions(ctx, &e_ds_1, &d.dice)
        }
        DecisionType::Reroll2 => {
            let mut e_ds_0 = [0.0f32; 252];
            compute_group6(ctx, sv, d.upper_score, d.scored, &mut e_ds_0);
            enumerate_reroll_actions(ctx, &e_ds_0, &d.dice)
        }
        DecisionType::Category => {
            enumerate_category_actions(ctx, sv, d.upper_score, d.scored, ds_index, is_last)
        }
    };

    let best = all_actions[0].clone();
    let runner_up = if all_actions.len() > 1 {
        all_actions[1].clone()
    } else {
        best.clone()
    };
    let ev_gap = best.ev - runner_up.ev;
    let visit_fraction = visit_count as f64 / total_decisions_of_type.max(1) as f64;
    let difficulty_score = visit_count as f64 / (ev_gap as f64).max(0.01);

    let description = describe_scenario(
        &d.dice,
        d.turn,
        &d.decision_type,
        d.scored,
        &best,
        &runner_up,
        ev_gap,
    );

    DifficultScenario {
        rank,
        upper_score: d.upper_score,
        scored_categories: d.scored,
        dice: d.dice,
        turn: d.turn,
        decision_type: d.decision_type.as_str().to_string(),
        visit_count,
        visit_fraction,
        best_action: best,
        runner_up_action: runner_up,
        ev_gap,
        difficulty_score,
        all_actions,
        description,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 1_000_000usize;
    let mut top_n = 200usize;
    let mut seed = 42u64;
    let mut output_dir = String::from("outputs/scenarios");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                i += 1;
                num_games = args[i].parse().expect("Invalid --games");
            }
            "--top" => {
                i += 1;
                top_n = args[i].parse().expect("Invalid --top");
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("Invalid --seed");
            }
            "--output" => {
                i += 1;
                output_dir = args[i].clone();
            }
            "--help" | "-h" => {
                println!("Usage: yatzy-difficult-scenarios [OPTIONS]");
                println!("  --games N      Number of games to simulate (default: 1000000)");
                println!("  --top N        Number of scenarios to output (default: 200)");
                println!("  --seed S       Random seed (default: 42)");
                println!("  --output DIR   Output directory (default: outputs/scenarios)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Resolve output path
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

    // Batched simulation: 100K games per batch, dedup after each
    let batch_size = 100_000usize;
    let num_batches = num_games.div_ceil(batch_size);
    println!(
        "\nSimulating {} games in {} batches of {}...",
        num_games, num_batches, batch_size
    );
    let sim_start = Instant::now();

    let mut visit_counts: HashMap<DecisionKey, (RawDecision, usize)> = HashMap::new();
    let mut total_decisions = 0usize;
    let mut games_done = 0usize;

    for batch in 0..num_batches {
        let batch_start = batch * batch_size;
        let batch_end = (batch_start + batch_size).min(num_games);
        let batch_games = batch_end - batch_start;

        let batch_decisions: Vec<RawDecision> = (batch_start..batch_end)
            .into_par_iter()
            .flat_map_iter(|i| {
                let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
                simulate_game_collecting(&ctx, &mut rng)
            })
            .collect();

        total_decisions += batch_decisions.len();

        for d in &batch_decisions {
            let key = decision_key(d);
            visit_counts
                .entry(key)
                .and_modify(|e| e.1 += 1)
                .or_insert_with(|| (d.clone(), 1));
        }

        games_done += batch_games;
        if (batch + 1) % 2 == 0 || batch == num_batches - 1 {
            println!(
                "  Batch {}/{}: {} games, {} unique scenarios, {:.1}s",
                batch + 1,
                num_batches,
                games_done,
                visit_counts.len(),
                sim_start.elapsed().as_secs_f64(),
            );
        }
    }

    println!(
        "Simulation complete: {} total decisions, {} unique scenarios in {:.1}s",
        total_decisions,
        visit_counts.len(),
        sim_start.elapsed().as_secs_f64(),
    );

    // Count decisions by type for visit_fraction
    let mut type_totals: HashMap<DecisionType, usize> = HashMap::new();
    for ((_, _, _, dt), (_, count)) in &visit_counts {
        *type_totals.entry(*dt).or_default() += count;
    }

    // Filter: is_realistic + visit_count ≥ 10
    let min_visits = 10usize;
    println!("Filtering (min_visits={}, is_realistic)...", min_visits);

    let filtered: Vec<(RawDecision, usize)> = visit_counts
        .into_values()
        .filter(|(d, count)| {
            *count >= min_visits
                && is_realistic(d.upper_score, d.scored, d.turn, &d.category_scores)
        })
        .collect();
    println!("  {} surviving scenarios", filtered.len());

    // Analyze each scenario (compute all action EVs + gap)
    println!("Computing action EVs for {} scenarios...", filtered.len());
    let analysis_start = Instant::now();

    // First pass: compute gap + difficulty_score for ranking (parallel)
    struct ScoredEntry {
        raw: RawDecision,
        visit_count: usize,
        #[allow(dead_code)]
        num_actions: usize,
        #[allow(dead_code)]
        ev_gap: f32,
        difficulty_score: f64,
    }

    let scored_entries: Vec<ScoredEntry> = filtered
        .par_iter()
        .filter_map(|(d, count)| {
            let sv = ctx.state_values.as_slice();
            let ds_index = find_dice_set_index(&ctx, &d.dice);
            let is_last = d.turn == CATEGORY_COUNT - 1;

            // Compute all actions to get count + gap for ranking
            let actions = match d.decision_type {
                DecisionType::Reroll1 => {
                    let mut e_ds_0 = [0.0f32; 252];
                    let mut e_ds_1 = [0.0f32; 252];
                    compute_group6(&ctx, sv, d.upper_score, d.scored, &mut e_ds_0);
                    compute_max_ev_for_n_rerolls(&ctx, &e_ds_0, &mut e_ds_1);
                    enumerate_reroll_actions(&ctx, &e_ds_1, &d.dice)
                }
                DecisionType::Reroll2 => {
                    let mut e_ds_0 = [0.0f32; 252];
                    compute_group6(&ctx, sv, d.upper_score, d.scored, &mut e_ds_0);
                    enumerate_reroll_actions(&ctx, &e_ds_0, &d.dice)
                }
                DecisionType::Category => {
                    enumerate_category_actions(&ctx, sv, d.upper_score, d.scored, ds_index, is_last)
                }
            };

            // Skip trivial decisions (only 1 legal action)
            if actions.len() < 2 {
                return None;
            }

            let best_ev = actions[0].ev;
            let runner_up_ev = actions[1].ev;
            let gap = best_ev - runner_up_ev;
            let difficulty_score = *count as f64 / (gap as f64).max(0.01);

            Some(ScoredEntry {
                raw: d.clone(),
                visit_count: *count,
                num_actions: actions.len(),
                ev_gap: gap,
                difficulty_score,
            })
        })
        .collect();

    println!(
        "  Gap computation done in {:.1}s",
        analysis_start.elapsed().as_secs_f64()
    );

    // Stratified selection: equal allocation per turn, hardest within each
    let mut by_turn: HashMap<usize, Vec<ScoredEntry>> = HashMap::new();
    for entry in scored_entries {
        by_turn.entry(entry.raw.turn).or_default().push(entry);
    }
    // Sort each turn's entries by difficulty_score descending
    for entries in by_turn.values_mut() {
        entries.sort_by(|a, b| b.difficulty_score.partial_cmp(&a.difficulty_score).unwrap());
    }

    let active_turns = by_turn.len();
    let base_per_turn = top_n / active_turns.max(1);
    let mut remainder = top_n - base_per_turn * active_turns;

    // First pass: take base_per_turn from each turn
    let mut selected: Vec<ScoredEntry> = Vec::with_capacity(top_n);
    let mut leftover: Vec<ScoredEntry> = Vec::new();
    let mut turn_keys: Vec<usize> = by_turn.keys().copied().collect();
    turn_keys.sort();

    for turn in &turn_keys {
        let entries = by_turn.remove(turn).unwrap();
        let take = if entries.len() < base_per_turn {
            // This turn has fewer than its allocation — take all, add surplus to remainder
            remainder += base_per_turn - entries.len();
            entries.len()
        } else {
            base_per_turn
        };
        let (taken, rest) = entries.into_iter().enumerate().fold(
            (Vec::new(), Vec::new()),
            |(mut taken, mut rest), (i, e)| {
                if i < take {
                    taken.push(e);
                } else {
                    rest.push(e);
                }
                (taken, rest)
            },
        );
        selected.extend(taken);
        leftover.extend(rest);
    }

    // Second pass: fill remainder from leftover pool, sorted by difficulty_score
    leftover.sort_by(|a, b| b.difficulty_score.partial_cmp(&a.difficulty_score).unwrap());
    selected.extend(leftover.into_iter().take(remainder));

    // Final sort by difficulty_score for ranking
    selected.sort_by(|a, b| b.difficulty_score.partial_cmp(&a.difficulty_score).unwrap());
    selected.truncate(top_n);
    let sorted_entries = selected;

    // Print stratification summary
    {
        let mut turn_counts: HashMap<usize, usize> = HashMap::new();
        for e in &sorted_entries {
            *turn_counts.entry(e.raw.turn).or_default() += 1;
        }
        println!(
            "Stratified selection: {} scenarios across {} turns",
            sorted_entries.len(),
            turn_counts.len()
        );
        let mut tc: Vec<_> = turn_counts.iter().collect();
        tc.sort_by_key(|(t, _)| *t);
        for (t, n) in tc {
            println!("  Turn {:>2}: {}", t + 1, n);
        }
    }

    // Full analysis (with all_actions) only for top N
    println!(
        "Building full scenario details for top {}...",
        sorted_entries.len()
    );

    let scenarios: Vec<DifficultScenario> = sorted_entries
        .iter()
        .enumerate()
        .map(|(idx, entry)| {
            let total_of_type = type_totals
                .get(&entry.raw.decision_type)
                .copied()
                .unwrap_or(1);
            analyze_scenario(&ctx, &entry.raw, entry.visit_count, total_of_type, idx + 1)
        })
        .collect();

    // Output JSON
    let _ = std::fs::create_dir_all(&output_dir);
    let json_path = format!("{}/difficult_scenarios.json", output_dir);
    {
        let json = serde_json::to_string_pretty(&scenarios).expect("JSON serialization failed");
        let mut f = std::fs::File::create(&json_path).expect("Failed to create JSON");
        f.write_all(json.as_bytes()).unwrap();
    }
    println!("Wrote {}", json_path);

    // Output CSV
    let csv_path = format!("{}/difficult_scenarios.csv", output_dir);
    {
        let mut f = std::fs::File::create(&csv_path).expect("Failed to create CSV");
        writeln!(
            f,
            "rank,turn,decision_type,upper_score,scored_hex,dice,visit_count,visit_fraction,\
             best_action,best_ev,runner_up_action,runner_up_ev,ev_gap,difficulty_score,\
             num_actions,description"
        )
        .unwrap();
        for s in &scenarios {
            let dice_str = format!(
                "{} {} {} {} {}",
                s.dice[0], s.dice[1], s.dice[2], s.dice[3], s.dice[4]
            );
            let q = |s: &str| -> String {
                if s.contains(',') || s.contains('"') {
                    format!("\"{}\"", s.replace('"', "\"\""))
                } else {
                    s.to_string()
                }
            };
            writeln!(
                f,
                "{},{},{},{},0x{:04X},{},{},{:.6},{},{:.3},{},{:.3},{:.4},{:.1},{},{}",
                s.rank,
                s.turn + 1,
                s.decision_type,
                s.upper_score,
                s.scored_categories,
                dice_str,
                s.visit_count,
                s.visit_fraction,
                q(&s.best_action.action_name),
                s.best_action.ev,
                q(&s.runner_up_action.action_name),
                s.runner_up_action.ev,
                s.ev_gap,
                s.difficulty_score,
                s.all_actions.len(),
                q(&s.description),
            )
            .unwrap();
        }
    }
    println!("Wrote {}", csv_path);

    // Console summary
    println!("\n=== Top 20 Most Difficult Scenarios ===");
    println!(
        "{:>4} {:>4} {:>10} {:>14} {:>6} {:>9} {:>8} {:>20} {:>20}",
        "Rank", "Turn", "Type", "Dice", "Visits", "Gap", "DScore", "Best Action", "Runner-Up"
    );
    println!("{}", "-".repeat(105));
    for s in scenarios.iter().take(20) {
        println!(
            "{:>4} {:>4} {:>10} {:>14} {:>6} {:>9.4} {:>8.0} {:>20} {:>20}",
            s.rank,
            s.turn + 1,
            s.decision_type,
            format!("{:?}", s.dice),
            s.visit_count,
            s.ev_gap,
            s.difficulty_score,
            truncate_str(&s.best_action.action_name, 20),
            truncate_str(&s.runner_up_action.action_name, 20),
        );
    }

    // Distribution summary
    let mut by_type: HashMap<&str, usize> = HashMap::new();
    let mut by_turn: HashMap<usize, usize> = HashMap::new();
    for s in &scenarios {
        *by_type.entry(s.decision_type.as_str()).or_default() += 1;
        *by_turn.entry(s.turn).or_default() += 1;
    }

    println!("\n=== Distribution ===");
    println!("By decision type:");
    for dt in &["reroll1", "reroll2", "category"] {
        println!("  {}: {}", dt, by_type.get(dt).unwrap_or(&0));
    }
    println!("By turn:");
    let mut turns: Vec<_> = by_turn.iter().collect();
    turns.sort_by_key(|(t, _)| *t);
    for (t, count) in turns {
        println!("  Turn {}: {}", t + 1, count);
    }

    let gap_stats: Vec<f32> = scenarios.iter().map(|s| s.ev_gap).collect();
    if !gap_stats.is_empty() {
        let min_gap = gap_stats.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_gap = gap_stats.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_gap = gap_stats.iter().sum::<f32>() / gap_stats.len() as f32;
        println!(
            "\nEV gap: min={:.4}, max={:.4}, mean={:.4}",
            min_gap, max_gap, mean_gap
        );
    }

    println!(
        "\nTotal: {:.1}s ({} scenarios output)",
        total_start.elapsed().as_secs_f64(),
        scenarios.len()
    );
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len - 1])
    }
}
