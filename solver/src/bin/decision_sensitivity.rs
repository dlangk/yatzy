//! Decision sensitivity analysis: find game decisions where optimal play flips
//! under small risk-preference changes (θ ∈ [0, 0.2]).
//!
//! Pipeline:
//! 1. Load 12 θ state tables (mmap)
//! 2. Simulate 100K games under θ=0, collecting all 3 decision types per turn
//! 3. Dedup by (upper_score, scored, dice, decision_type), count visits
//! 4. Filter to ≥0.1% visit rate + is_realistic()
//! 5. Analyze each surviving decision across 12 θ values (rayon)
//! 6. Output CSV + JSON

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
use yatzy::widget_solver::{
    choose_best_reroll_mask, compute_max_ev_for_n_rerolls, compute_opt_lse_for_n_rerolls,
};

/// θ grid for sensitivity analysis: 12 values in [0, 0.2].
const THETA_GRID: [f32; 12] = [
    0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2,
];

/// Decision type at each point in a turn.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum DecisionType {
    Reroll1,  // After initial roll, before 1st reroll
    Reroll2,  // After 1st reroll, before 2nd reroll
    Category, // After final dice, choosing category
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

/// Game phase based on turn number.
fn game_phase(turn: usize) -> &'static str {
    if turn < 5 {
        "early"
    } else if turn < 10 {
        "mid"
    } else {
        "late"
    }
}

/// A raw decision point extracted from simulation.
#[derive(Clone)]
struct RawDecision {
    upper_score: i32,
    scored: i32,
    dice: [i32; 5],
    turn: usize,
    decision_type: DecisionType,
    category_scores: [i32; 15],
}

/// Dedup key for decision points.
type DecisionKey = (i32, i32, [i32; 5], DecisionType);

fn decision_key(d: &RawDecision) -> DecisionKey {
    (d.upper_score, d.scored, d.dice, d.decision_type)
}

/// Per-θ analysis result for a decision.
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

/// Full analysis of a single decision point.
#[derive(Clone, Serialize)]
struct DecisionAnalysis {
    upper_score: i32,
    scored_categories: i32,
    dice: [i32; 5],
    turn: usize,
    decision_type: String,
    game_phase: String,
    theta_0_action: String,
    theta_0_action_id: i32,
    has_flip: bool,
    flip_theta: f32,
    flip_action: String,
    flip_action_id: i32,
    gap_at_flip: f32,
    gap_at_theta0: f32,
    visit_count: usize,
    state_frequency: usize,
    state_fraction: f64,
    theta_results: Vec<ThetaResult>,
}

/// Loaded θ table.
struct ThetaEntry {
    theta: f32,
    sv: StateValues,
}

// ── Simulation helpers (from engine.rs / pivotal_scenarios.rs) ──

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

#[inline(always)]
fn compute_group6_risk(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    theta: f32,
    e_ds_0: &mut [f32; 252],
) {
    // For θ>0, decision nodes maximize (risk-seeking)
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

/// Check whether a scenario looks like something a human would encounter.
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

        // Decision 1: Reroll1 (before first reroll)
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

        // Decision 2: Reroll2 (before second reroll)
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

        // Decision 3: Category (after final dice)
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

/// Find best and runner-up reroll mask for a given dice set and e_ds values.
/// Returns (best_mask, best_ev, runner_up_mask, runner_up_ev).
fn find_best_and_runner_up_mask(
    ctx: &YatzyContext,
    e_ds: &[f32; 252],
    dice: &[i32; 5],
    theta: f32,
    is_risk: bool,
) -> (i32, f32, i32, f32) {
    let ds_index = find_dice_set_index(ctx, dice);
    let kt = &ctx.keep_table;
    let vals = &kt.vals;
    let cols = &kt.cols;

    // Collect (mask, ev) for all options including keep-all
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
                    sum += *vals.get_unchecked(k) * (v - max_x).exp();
                }
            }
            max_x + sum.ln()
        } else {
            // Standard EV
            let mut ev: f32 = 0.0;
            for k in start..end {
                unsafe {
                    ev += *vals.get_unchecked(k)
                        * e_ds.get_unchecked(*cols.get_unchecked(k) as usize);
                }
            }
            ev
        };

        let mask = kt.keep_to_mask[ds_index * 32 + j];
        options.push((mask, ev));
    }

    // Sort descending by value (θ>0 always maximizes in our grid)
    options.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let (best_mask, best_ev) = options[0];
    let (ru_mask, ru_ev) = if options.len() > 1 {
        options[1]
    } else {
        (0, best_ev)
    };

    let _ = theta; // used in future extensions
    (best_mask, best_ev, ru_mask, ru_ev)
}

/// Find best and runner-up category for given state and dice.
/// Returns (best_cat, best_val, runner_up_cat, runner_up_val).
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
        // Risk-sensitive: val = θ·score + sv_θ[successor]
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

    // Sort descending (θ≥0 always maximizes)
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

/// Analyze a decision point across all θ values.
fn analyze_decision(
    ctx: &YatzyContext,
    theta_entries: &[ThetaEntry],
    d: &RawDecision,
    visit_count: usize,
    state_frequency: usize,
    state_fraction: f64,
) -> DecisionAnalysis {
    let ds_index = find_dice_set_index(ctx, &d.dice);
    let is_last = d.turn == CATEGORY_COUNT - 1;
    let mut theta_results = Vec::with_capacity(THETA_GRID.len());

    let mut theta0_action = String::new();
    let mut theta0_action_id: i32 = 0;
    let mut theta0_gap: f32 = 0.0;

    let mut has_flip = false;
    let mut flip_theta: f32 = 0.0;
    let mut flip_action = String::new();
    let mut flip_action_id: i32 = 0;
    let mut gap_at_flip: f32 = 0.0;

    for entry in theta_entries {
        let sv = entry.sv.as_slice();
        let theta = entry.theta;
        let is_risk = theta != 0.0;

        let (action_str, action_id, value, ru_str, ru_id, ru_value) = match d.decision_type {
            DecisionType::Reroll1 => {
                // Need e_ds_1 (after Group 5 propagation)
                let mut e_ds_0 = [0.0f32; 252];
                let mut e_ds_1 = [0.0f32; 252];
                if is_risk {
                    compute_group6_risk(ctx, sv, d.upper_score, d.scored, theta, &mut e_ds_0);
                    compute_opt_lse_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1, false);
                } else {
                    compute_group6(ctx, sv, d.upper_score, d.scored, &mut e_ds_0);
                    compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);
                }
                let (bm, bv, rm, rv) =
                    find_best_and_runner_up_mask(ctx, &e_ds_1, &d.dice, theta, is_risk);
                (
                    format_mask(bm, &d.dice),
                    bm,
                    bv,
                    format_mask(rm, &d.dice),
                    rm,
                    rv,
                )
            }
            DecisionType::Reroll2 => {
                // Need e_ds_0 (Group 6 output)
                let mut e_ds_0 = [0.0f32; 252];
                if is_risk {
                    compute_group6_risk(ctx, sv, d.upper_score, d.scored, theta, &mut e_ds_0);
                } else {
                    compute_group6(ctx, sv, d.upper_score, d.scored, &mut e_ds_0);
                }
                let (bm, bv, rm, rv) =
                    find_best_and_runner_up_mask(ctx, &e_ds_0, &d.dice, theta, is_risk);
                (
                    format_mask(bm, &d.dice),
                    bm,
                    bv,
                    format_mask(rm, &d.dice),
                    rm,
                    rv,
                )
            }
            DecisionType::Category => {
                let (bc, bv, rc, rv) = find_best_and_runner_up_category(
                    ctx,
                    sv,
                    d.upper_score,
                    d.scored,
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
        };

        let gap = value - ru_value;

        theta_results.push(ThetaResult {
            theta,
            action: action_str.clone(),
            action_id,
            value,
            runner_up: ru_str.clone(),
            runner_up_id: ru_id,
            runner_up_value: ru_value,
            gap,
        });

        if theta == 0.0 {
            theta0_action = action_str.clone();
            theta0_action_id = action_id;
            theta0_gap = gap;
        }

        // Check for flip: action differs from θ=0
        if theta != 0.0 && !has_flip && action_id != theta0_action_id {
            has_flip = true;
            flip_theta = theta;
            flip_action = action_str;
            flip_action_id = action_id;
            gap_at_flip = gap;
        }
    }

    DecisionAnalysis {
        upper_score: d.upper_score,
        scored_categories: d.scored,
        dice: d.dice,
        turn: d.turn,
        decision_type: d.decision_type.as_str().to_string(),
        game_phase: game_phase(d.turn).to_string(),
        theta_0_action: theta0_action,
        theta_0_action_id: theta0_action_id,
        has_flip,
        flip_theta,
        flip_action,
        flip_action_id,
        gap_at_flip,
        gap_at_theta0: theta0_gap,
        visit_count,
        state_frequency,
        state_fraction,
        theta_results,
    }
}

/// Summary statistics.
#[derive(Serialize)]
struct SummaryStats {
    total_decisions: usize,
    unique_decisions: usize,
    analyzed_decisions: usize,
    flip_count: usize,
    flip_rate: f32,
    by_decision_type: Vec<TypeFlipStats>,
    by_game_phase: Vec<PhaseFlipStats>,
    flip_theta_histogram: Vec<ThetaHistBin>,
    top_20_tightest: Vec<TightDecision>,
}

#[derive(Serialize)]
struct TypeFlipStats {
    decision_type: String,
    total: usize,
    flips: usize,
    flip_rate: f32,
}

#[derive(Serialize)]
struct PhaseFlipStats {
    game_phase: String,
    decision_type: String,
    total: usize,
    flips: usize,
    flip_rate: f32,
}

#[derive(Serialize)]
struct ThetaHistBin {
    theta: f32,
    count: usize,
}

#[derive(Serialize)]
struct TightDecision {
    dice: [i32; 5],
    turn: usize,
    decision_type: String,
    theta_0_action: String,
    flip_action: String,
    flip_theta: f32,
    gap_at_theta0: f32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 100_000usize;
    let mut seed = 42u64;
    let mut output_dir = String::from("outputs/scenarios");
    let mut min_visit_frac = 0.001f64;

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
            "--min-visits" => {
                i += 1;
                min_visit_frac = args[i].parse().expect("Invalid --min-visits");
            }
            "--help" | "-h" => {
                println!("Usage: yatzy-decision-sensitivity [OPTIONS]");
                println!("  --games N        Number of games to simulate (default: 100000)");
                println!("  --seed S         Random seed (default: 42)");
                println!("  --output DIR     Output directory (default: outputs/scenarios)");
                println!("  --min-visits F   Minimum visit fraction (default: 0.001)");
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

    // Load all θ tables
    println!("Loading {} θ tables...", THETA_GRID.len());
    let mut theta_entries: Vec<ThetaEntry> = Vec::new();
    for &theta in &THETA_GRID {
        let file = state_file_path(theta);
        match load_state_values_standalone(&file) {
            Some(sv) => {
                println!("  θ={:.3}: loaded {}", theta, file);
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

    // Step 1: Simulate games under θ=0, collecting all decision types
    println!(
        "\nSimulating {} games (3 decisions/turn × 15 turns)...",
        num_games
    );
    let sim_start = Instant::now();

    let all_decisions: Vec<RawDecision> = (0..num_games)
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
            simulate_game_collecting(&ctx, &mut rng)
        })
        .collect();

    let total_raw = all_decisions.len();
    println!(
        "  Collected {} decision points in {:.1}s",
        total_raw,
        sim_start.elapsed().as_secs_f64()
    );

    // Step 2: Dedup by (upper_score, scored, dice, decision_type), count visits
    println!("Deduplicating...");
    let mut visit_counts: HashMap<DecisionKey, (RawDecision, usize)> = HashMap::new();
    for d in &all_decisions {
        let key = decision_key(d);
        visit_counts
            .entry(key)
            .and_modify(|e| e.1 += 1)
            .or_insert_with(|| (d.clone(), 1));
    }
    let total_unique = visit_counts.len();
    println!("  {} unique decision points", total_unique);

    // Aggregate by board state: (upper_score, scored_categories) ignoring dice/dtype
    let mut board_state_freq: HashMap<(i32, i32), usize> = HashMap::new();
    for (key, (_raw, count)) in &visit_counts {
        *board_state_freq.entry((key.0, key.1)).or_default() += count;
    }
    println!("  {} unique board states", board_state_freq.len());

    // Step 3: Filter by visit rate and realism
    let min_visits = (num_games as f64 * min_visit_frac).ceil() as usize;
    println!("Filtering (min_visits={}, is_realistic)...", min_visits);

    let filtered: Vec<(RawDecision, usize)> = visit_counts
        .into_values()
        .filter(|(d, count)| {
            *count >= min_visits
                && is_realistic(d.upper_score, d.scored, d.turn, &d.category_scores)
        })
        .collect();
    let total_filtered = filtered.len();
    println!("  {} surviving decision points", total_filtered);

    // Step 4: Analyze each decision across all θ values
    println!(
        "Analyzing {} decisions × {} θ values...",
        total_filtered,
        THETA_GRID.len()
    );
    let analysis_start = Instant::now();

    let analyses: Vec<DecisionAnalysis> = filtered
        .par_iter()
        .map(|(d, count)| {
            let sf = board_state_freq
                .get(&(d.upper_score, d.scored))
                .copied()
                .unwrap_or(0);
            let sf_frac = sf as f64 / total_raw as f64;
            analyze_decision(&ctx, &theta_entries, d, *count, sf, sf_frac)
        })
        .collect();

    println!("  Done in {:.1}s", analysis_start.elapsed().as_secs_f64());

    let flip_count = analyses.iter().filter(|a| a.has_flip).count();
    println!(
        "  {} flips out of {} decisions ({:.1}%)",
        flip_count,
        total_filtered,
        100.0 * flip_count as f64 / total_filtered as f64
    );

    // Step 5: Output CSV (all decisions)
    let _ = std::fs::create_dir_all(&output_dir);
    let csv_path = format!("{}/decision_sensitivity.csv", output_dir);
    {
        let mut f = std::fs::File::create(&csv_path).expect("Failed to create CSV");
        writeln!(
            f,
            "upper_score,scored_categories,dice,turn,decision_type,game_phase,\
             theta_0_action,theta_0_action_id,has_flip,flip_theta,flip_action,\
             flip_action_id,gap_at_flip,gap_at_theta0,visit_count,\
             state_frequency,state_fraction"
        )
        .unwrap();
        for a in &analyses {
            // Format dice as space-separated for CSV: "1 2 3 4 5"
            let dice_str = format!(
                "{} {} {} {} {}",
                a.dice[0], a.dice[1], a.dice[2], a.dice[3], a.dice[4]
            );
            // Quote fields that may contain commas
            let q = |s: &str| -> String {
                if s.contains(',') {
                    format!("\"{}\"", s)
                } else {
                    s.to_string()
                }
            };
            writeln!(
                f,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6}",
                a.upper_score,
                a.scored_categories,
                dice_str,
                a.turn,
                a.decision_type,
                a.game_phase,
                q(&a.theta_0_action),
                a.theta_0_action_id,
                a.has_flip,
                a.flip_theta,
                q(&a.flip_action),
                a.flip_action_id,
                a.gap_at_flip,
                a.gap_at_theta0,
                a.visit_count,
                a.state_frequency,
                a.state_fraction,
            )
            .unwrap();
        }
    }
    println!("Wrote {}", csv_path);

    // Step 5b: Output state_frequency.csv (board-state level aggregation)
    let freq_csv_path = format!("{}/state_frequency.csv", output_dir);
    {
        let mut freq_entries: Vec<_> = board_state_freq.iter().collect();
        freq_entries.sort_by(|a, b| b.1.cmp(a.1)); // descending by visit count

        let mut f = std::fs::File::create(&freq_csv_path).expect("Failed to create freq CSV");
        writeln!(f, "turn,scored_categories,upper_score,visit_count,fraction").unwrap();
        for (&(upper_score, scored), &count) in &freq_entries {
            let turn = (scored as u32).count_ones() as usize;
            let frac = count as f64 / total_raw as f64;
            writeln!(
                f,
                "{},{},{},{},{:.6}",
                turn, scored, upper_score, count, frac
            )
            .unwrap();
        }
    }
    println!("Wrote {}", freq_csv_path);

    // Step 6: Output flips JSON (only flip decisions, with full θ breakdown)
    let flips_json_path = format!("{}/decision_sensitivity_flips.json", output_dir);
    {
        let flips: Vec<&DecisionAnalysis> = analyses.iter().filter(|a| a.has_flip).collect();
        let json = serde_json::to_string_pretty(&flips).expect("JSON serialization failed");
        let mut f = std::fs::File::create(&flips_json_path).expect("Failed to create flips JSON");
        f.write_all(json.as_bytes()).unwrap();
    }
    println!("Wrote {}", flips_json_path);

    // Step 7: Output summary JSON
    let summary_json_path = format!("{}/decision_sensitivity_summary.json", output_dir);
    {
        // Flip rates by decision type
        let mut by_type: HashMap<String, (usize, usize)> = HashMap::new();
        for a in &analyses {
            let entry = by_type.entry(a.decision_type.clone()).or_insert((0, 0));
            entry.0 += 1;
            if a.has_flip {
                entry.1 += 1;
            }
        }
        let by_decision_type: Vec<TypeFlipStats> = ["reroll1", "reroll2", "category"]
            .iter()
            .filter_map(|dt| {
                by_type.get(*dt).map(|(total, flips)| TypeFlipStats {
                    decision_type: dt.to_string(),
                    total: *total,
                    flips: *flips,
                    flip_rate: if *total > 0 {
                        *flips as f32 / *total as f32
                    } else {
                        0.0
                    },
                })
            })
            .collect();

        // Flip rates by game_phase × decision_type
        let mut by_phase: HashMap<(String, String), (usize, usize)> = HashMap::new();
        for a in &analyses {
            let entry = by_phase
                .entry((a.game_phase.clone(), a.decision_type.clone()))
                .or_insert((0, 0));
            entry.0 += 1;
            if a.has_flip {
                entry.1 += 1;
            }
        }
        let mut by_game_phase: Vec<PhaseFlipStats> = by_phase
            .into_iter()
            .map(|((phase, dt), (total, flips))| PhaseFlipStats {
                game_phase: phase,
                decision_type: dt,
                total,
                flips,
                flip_rate: if total > 0 {
                    flips as f32 / total as f32
                } else {
                    0.0
                },
            })
            .collect();
        by_game_phase.sort_by(|a, b| {
            a.game_phase
                .cmp(&b.game_phase)
                .then(a.decision_type.cmp(&b.decision_type))
        });

        // Flip-θ histogram
        let mut theta_hist: HashMap<String, usize> = HashMap::new();
        for a in analyses.iter().filter(|a| a.has_flip) {
            *theta_hist
                .entry(format!("{:.3}", a.flip_theta))
                .or_default() += 1;
        }
        let mut flip_theta_histogram: Vec<ThetaHistBin> = theta_hist
            .into_iter()
            .map(|(t, count)| ThetaHistBin {
                theta: t.parse().unwrap(),
                count,
            })
            .collect();
        flip_theta_histogram.sort_by(|a, b| a.theta.partial_cmp(&b.theta).unwrap());

        // Top 20 tightest gaps (among flips)
        let mut flips_sorted: Vec<&DecisionAnalysis> =
            analyses.iter().filter(|a| a.has_flip).collect();
        flips_sorted.sort_by(|a, b| {
            a.gap_at_theta0
                .abs()
                .partial_cmp(&b.gap_at_theta0.abs())
                .unwrap()
        });
        let top_20_tightest: Vec<TightDecision> = flips_sorted
            .iter()
            .take(20)
            .map(|a| TightDecision {
                dice: a.dice,
                turn: a.turn,
                decision_type: a.decision_type.clone(),
                theta_0_action: a.theta_0_action.clone(),
                flip_action: a.flip_action.clone(),
                flip_theta: a.flip_theta,
                gap_at_theta0: a.gap_at_theta0,
            })
            .collect();

        let summary = SummaryStats {
            total_decisions: total_raw,
            unique_decisions: total_unique,
            analyzed_decisions: total_filtered,
            flip_count,
            flip_rate: if total_filtered > 0 {
                flip_count as f32 / total_filtered as f32
            } else {
                0.0
            },
            by_decision_type,
            by_game_phase,
            flip_theta_histogram,
            top_20_tightest,
        };

        let json = serde_json::to_string_pretty(&summary).expect("JSON serialization failed");
        let mut f =
            std::fs::File::create(&summary_json_path).expect("Failed to create summary JSON");
        f.write_all(json.as_bytes()).unwrap();
    }
    println!("Wrote {}", summary_json_path);

    // Print top-20 to console
    println!("\n=== Top 20 Most Sensitive Decisions (smallest θ=0 gap among flips) ===");
    println!(
        "{:>5} {:>12} {:>10} {:>20} {:>20} {:>10} {:>10}",
        "Turn", "Dice", "Type", "θ=0 Action", "Flip Action", "Flip θ", "θ=0 Gap"
    );
    println!("{}", "-".repeat(92));
    let mut flips_sorted: Vec<&DecisionAnalysis> = analyses.iter().filter(|a| a.has_flip).collect();
    flips_sorted.sort_by(|a, b| {
        a.gap_at_theta0
            .abs()
            .partial_cmp(&b.gap_at_theta0.abs())
            .unwrap()
    });
    for a in flips_sorted.iter().take(20) {
        println!(
            "{:>5} {:>12} {:>10} {:>20} {:>20} {:>10.3} {:>10.3}",
            a.turn + 1,
            format!("{:?}", a.dice),
            a.decision_type,
            a.theta_0_action,
            a.flip_action,
            a.flip_theta,
            a.gap_at_theta0,
        );
    }

    println!("\nTotal: {:.1}s", total_start.elapsed().as_secs_f64());
}
