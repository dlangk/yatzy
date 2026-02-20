//! Action enumeration and EV computation for scenarios.
//!
//! Consolidates `compute_group6`, `compute_group6_risk`, `find_best_category`,
//! `enumerate_reroll_actions`, `enumerate_category_actions`, `format_mask`,
//! `roll_dice`, `apply_reroll` from three separate files.

#![allow(clippy::needless_range_loop)]

use rand::rngs::SmallRng;
use rand::Rng;

use crate::constants::*;
use crate::dice_mechanics::{find_dice_set_index, sort_dice_set};
use crate::game_mechanics::update_upper_score;
use crate::scenarios::types::{ActionInfo, DecisionType, RawDecision};
use crate::types::YatzyContext;
use crate::widget_solver::compute_max_ev_for_n_rerolls;

// ── Dice helpers ──

pub fn roll_dice(rng: &mut SmallRng) -> [i32; 5] {
    let mut dice = [0i32; 5];
    for d in &mut dice {
        *d = rng.random_range(1..=6);
    }
    sort_dice_set(&mut dice);
    dice
}

pub fn apply_reroll(dice: &mut [i32; 5], mask: i32, rng: &mut SmallRng) {
    for i in 0..5 {
        if mask & (1 << i) != 0 {
            dice[i] = rng.random_range(1..=6);
        }
    }
    sort_dice_set(dice);
}

// ── Format helpers ──

/// Format a reroll mask as human-readable string (lowercase).
pub fn format_mask(mask: i32, dice: &[i32; 5]) -> String {
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

/// Format a reroll mask as human-readable string (capitalized, for profiling quiz).
pub fn format_mask_capitalized(mask: i32, dice: &[i32; 5]) -> String {
    if mask == 0 {
        return "Keep all".to_string();
    }
    if mask == 31 {
        return "Reroll all".to_string();
    }
    let mut kept: Vec<i32> = Vec::new();
    for i in 0..5 {
        if mask & (1 << i) == 0 {
            kept.push(dice[i]);
        }
    }
    if kept.is_empty() {
        "Reroll all".to_string()
    } else {
        format!("Keep {:?}", kept)
    }
}

pub fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len - 1])
    }
}

// ── Group 6 computation (best category EV for each dice set) ──

#[inline(always)]
pub fn compute_group6(
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

/// Risk-sensitive variant: uses `theta * scr` instead of `scr`.
#[inline(always)]
pub fn compute_group6_risk(
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

// ── Category selection ──

pub fn find_best_category(
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

pub fn find_best_category_final(
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

// ── Action enumeration ──

/// Enumerate all reroll options with EVs for a given dice set and e_ds values.
pub fn enumerate_reroll_actions(
    ctx: &YatzyContext,
    e_ds: &[f32; 252],
    dice: &[i32; 5],
    capitalize: bool,
) -> Vec<ActionInfo> {
    let ds_index = find_dice_set_index(ctx, dice);
    let kt = &ctx.keep_table;
    let mut actions: Vec<ActionInfo> = Vec::with_capacity(32);

    let fmt = if capitalize {
        format_mask_capitalized
    } else {
        format_mask
    };

    actions.push(ActionInfo {
        id: 0,
        label: fmt(0, dice),
        ev: e_ds[ds_index],
    });

    for j in 0..kt.unique_count[ds_index] as usize {
        let kid = kt.unique_keep_ids[ds_index][j] as usize;
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;

        let mut ev: f32 = 0.0;
        for k in start..end {
            unsafe {
                ev += *kt.vals.get_unchecked(k)
                    * e_ds.get_unchecked(*kt.cols.get_unchecked(k) as usize);
            }
        }

        let mask = kt.keep_to_mask[ds_index * 32 + j];
        actions.push(ActionInfo {
            id: mask,
            label: fmt(mask, dice),
            ev,
        });
    }

    actions.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap());
    actions
}

/// Enumerate all category options with EVs.
pub fn enumerate_category_actions(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    ds_index: usize,
    is_last_turn: bool,
) -> Vec<ActionInfo> {
    let mut actions: Vec<ActionInfo> = Vec::with_capacity(CATEGORY_COUNT);

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
                actions.push(ActionInfo {
                    id: c as i32,
                    label: CATEGORY_NAMES[c].to_string(),
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
                actions.push(ActionInfo {
                    id: c as i32,
                    label: CATEGORY_NAMES[c].to_string(),
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
                actions.push(ActionInfo {
                    id: c as i32,
                    label: CATEGORY_NAMES[c].to_string(),
                    ev: val,
                });
            }
        }
    }

    actions.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap());
    actions
}

/// Get all actions for a decision point (dispatches to reroll or category).
pub fn get_actions(
    ctx: &YatzyContext,
    sv: &[f32],
    d: &RawDecision,
    capitalize: bool,
) -> Vec<ActionInfo> {
    let ds_index = find_dice_set_index(ctx, &d.dice);
    let is_last = d.turn == CATEGORY_COUNT - 1;
    match d.decision_type {
        DecisionType::Reroll1 => {
            let mut e_ds_0 = [0.0f32; 252];
            let mut e_ds_1 = [0.0f32; 252];
            compute_group6(ctx, sv, d.upper_score, d.scored, &mut e_ds_0);
            compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);
            enumerate_reroll_actions(ctx, &e_ds_1, &d.dice, capitalize)
        }
        DecisionType::Reroll2 => {
            let mut e_ds_0 = [0.0f32; 252];
            compute_group6(ctx, sv, d.upper_score, d.scored, &mut e_ds_0);
            enumerate_reroll_actions(ctx, &e_ds_0, &d.dice, capitalize)
        }
        DecisionType::Category => {
            enumerate_category_actions(ctx, sv, d.upper_score, d.scored, ds_index, is_last)
        }
    }
}
