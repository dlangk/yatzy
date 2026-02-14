//! Game simulation engine — plays N games using optimal strategy.
//!
//! Uses the precomputed state values (E_table) to make optimal decisions at each
//! turn: roll → reroll → reroll → score. The simulation measures the resulting score
//! distribution and verifies the mean converges to the starting-state EV (~245.87).
//!
//! ## Recording mode
//!
//! `simulate_game_with_recording` captures full per-step data (dice, reroll masks,
//! category, score) into a compact `GameRecord` for offline aggregation.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

use crate::constants::*;
use crate::dice_mechanics::{find_dice_set_index, sort_dice_set};
use crate::game_mechanics::update_upper_score;
use crate::types::YatzyContext;
use crate::widget_solver::{
    choose_best_reroll_mask, choose_best_reroll_mask_max, choose_best_reroll_mask_risk,
    compute_max_ev_for_n_rerolls, compute_max_outcome_for_n_rerolls, compute_opt_lse_for_n_rerolls,
};

/// Results of a batch simulation.
pub struct SimulationResult {
    pub scores: Vec<i32>,
    pub mean: f64,
    pub std_dev: f64,
    pub min: i32,
    pub max: i32,
    pub median: i32,
    pub elapsed: std::time::Duration,
}

/// Per-turn record capturing all dice and decisions.
#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
pub struct TurnRecord {
    /// Sorted dice [1-6] after initial roll
    pub dice_initial: [u8; 5],
    /// First reroll mask (0-31)
    pub mask1: u8,
    /// Sorted dice after first reroll
    pub dice_after_reroll1: [u8; 5],
    /// Second reroll mask (0-31)
    pub mask2: u8,
    /// Sorted dice after second reroll (final dice)
    pub dice_final: [u8; 5],
    /// Category scored (0-14)
    pub category: u8,
    /// Score awarded (0-50 fits in u8)
    pub score: u8,
}

/// Per-game record: 15 turns plus summary.
#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
pub struct GameRecord {
    pub turns: [TurnRecord; 15],
    /// Total score including bonus (0-374)
    pub total_score: u16,
    /// Upper section total (0-63 capped)
    pub upper_total: u8,
    /// Whether the upper bonus was awarded (0 or 1)
    pub got_bonus: u8,
}

/// Roll 5 random dice and sort them.
#[inline(always)]
fn roll_dice(rng: &mut SmallRng) -> [i32; 5] {
    let mut dice = [0i32; 5];
    for d in &mut dice {
        *d = rng.random_range(1..=6);
    }
    sort_dice_set(&mut dice);
    dice
}

/// Apply a reroll mask: bits set in `mask` indicate dice positions to reroll.
/// After rerolling, the dice are re-sorted.
#[inline(always)]
fn apply_reroll(dice: &mut [i32; 5], mask: i32, rng: &mut SmallRng) {
    for i in 0..5 {
        if mask & (1 << i) != 0 {
            dice[i] = rng.random_range(1..=6);
        }
    }
    sort_dice_set(dice);
}

/// Compute Group 6 (e_ds_0): best category EV for each of the 252 dice sets.
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

/// Compute Group 6 for risk-sensitive mode: val = θ·scr + sv[successor].
/// Decision node: min when θ < 0 (risk-averse), max when θ > 0 (risk-seeking).
#[inline(always)]
fn compute_group6_risk(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    theta: f32,
    minimize: bool,
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
        let mut best_val = if minimize {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };

        for c in 0..6 {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_i][c];
                let new_up = update_upper_score(up_score, c, scr);
                let new_scored = scored | (1 << c);
                let val = theta * scr as f32
                    + unsafe {
                        *sv.get_unchecked(state_index(new_up as usize, new_scored as usize))
                    };
                let better = if minimize {
                    val < best_val
                } else {
                    val > best_val
                };
                if better {
                    best_val = val;
                }
            }
        }

        for c in 6..CATEGORY_COUNT {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_i][c];
                let val = theta * scr as f32 + unsafe { *lower_succ_ev.get_unchecked(c) };
                let better = if minimize {
                    val < best_val
                } else {
                    val > best_val
                };
                if better {
                    best_val = val;
                }
            }
        }

        e_ds_0[ds_i] = best_val;
    }
}

/// Find the best category in risk-sensitive mode: val = θ·scr + sv[successor].
/// Decision node: min when θ < 0 (risk-averse), max when θ > 0 (risk-seeking).
#[inline(always)]
fn find_best_category_risk(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    ds_index: usize,
    theta: f32,
    minimize: bool,
) -> (usize, i32) {
    let mut best_val = if minimize {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };
    let mut best_cat = 0usize;
    let mut best_score = 0i32;

    for c in 0..6 {
        if !is_category_scored(scored, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            let new_up = update_upper_score(up_score, c, scr);
            let new_scored = scored | (1 << c);
            let val = theta * scr as f32
                + unsafe { *sv.get_unchecked(state_index(new_up as usize, new_scored as usize)) };
            let better = if minimize {
                val < best_val
            } else {
                val > best_val
            };
            if better {
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
            let val = theta * scr as f32
                + unsafe { *sv.get_unchecked(state_index(up_score as usize, new_scored as usize)) };
            let better = if minimize {
                val < best_val
            } else {
                val > best_val
            };
            if better {
                best_val = val;
                best_cat = c;
                best_score = scr;
            }
        }
    }

    (best_cat, best_score)
}

/// Find the best category for the LAST turn in risk-sensitive mode.
/// Optimizes θ·(scr + bonus) to match the precomputed log-domain values.
/// On the last turn only 1 category remains, so min/max doesn't matter,
/// but we include the flag for correctness.
#[inline(always)]
fn find_best_category_final_risk(
    ctx: &YatzyContext,
    up_score: i32,
    scored: i32,
    ds_index: usize,
    theta: f32,
    minimize: bool,
) -> (usize, i32) {
    let mut best_val = if minimize {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };
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
            let val = theta * (scr + bonus) as f32;
            let better = if minimize {
                val < best_val
            } else {
                val > best_val
            };
            if better {
                best_val = val;
                best_cat = c;
                best_score = scr;
            }
        }
    }

    (best_cat, best_score)
}

/// Find the best category to score for the given dice, returning (category_index, score).
#[inline(always)]
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

/// Find the best category for the LAST turn (no successor state — just maximize score).
#[inline(always)]
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

/// Convert i32 dice to u8 array for recording.
#[inline(always)]
fn dice_to_u8(dice: &[i32; 5]) -> [u8; 5] {
    [
        dice[0] as u8,
        dice[1] as u8,
        dice[2] as u8,
        dice[3] as u8,
        dice[4] as u8,
    ]
}

/// Simulate one full game, returning the final score.
pub fn simulate_game(ctx: &YatzyContext, rng: &mut SmallRng) -> i32 {
    if ctx.max_policy {
        return simulate_game_max(ctx, rng);
    }
    let theta = ctx.theta;
    if theta != 0.0 {
        return simulate_game_risk(ctx, rng, theta);
    }

    let sv = ctx.state_values.as_slice();
    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut total_score: i32 = 0;

    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;

        let mut dice = roll_dice(rng);

        if is_last_turn {
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

            let ds_index = find_dice_set_index(ctx, &dice);
            let (cat, scr) = find_best_category_final(ctx, up_score, scored, ds_index);

            up_score = update_upper_score(up_score, cat, scr);
            scored |= 1 << cat;
            total_score += scr;
        } else {
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

            let ds_index = find_dice_set_index(ctx, &dice);
            let (cat, scr) = find_best_category(ctx, sv, up_score, scored, ds_index);

            up_score = update_upper_score(up_score, cat, scr);
            scored |= 1 << cat;
            total_score += scr;
        }
    }

    if up_score >= 63 {
        total_score += 50;
    }

    total_score
}

/// Risk-sensitive simulation: decisions use log-domain values, but scoring is still integer.
/// For θ < 0 (risk-averse), decision nodes use min instead of max.
fn simulate_game_risk(ctx: &YatzyContext, rng: &mut SmallRng, theta: f32) -> i32 {
    let sv = ctx.state_values.as_slice();
    let minimize = theta < 0.0;
    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut total_score: i32 = 0;

    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;

        let mut dice = roll_dice(rng);

        compute_group6_risk(ctx, sv, up_score, scored, theta, minimize, &mut e_ds_0);
        compute_opt_lse_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1, minimize);

        let mut best_ev = 0.0;
        let mask1 = choose_best_reroll_mask_risk(ctx, &e_ds_1, &dice, &mut best_ev, minimize);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }

        let mask2 = choose_best_reroll_mask_risk(ctx, &e_ds_0, &dice, &mut best_ev, minimize);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }

        let ds_index = find_dice_set_index(ctx, &dice);
        let (cat, scr) = if is_last_turn {
            find_best_category_final_risk(ctx, up_score, scored, ds_index, theta, minimize)
        } else {
            find_best_category_risk(ctx, sv, up_score, scored, ds_index, theta, minimize)
        };

        up_score = update_upper_score(up_score, cat, scr);
        scored |= 1 << cat;
        total_score += scr;
    }

    if up_score >= 63 {
        total_score += 50;
    }

    total_score
}

/// Max-policy simulation: decisions use max-outcome values, but actual dice are random.
/// Chance nodes in the decision evaluation use max instead of Σ P·x.
fn simulate_game_max(ctx: &YatzyContext, rng: &mut SmallRng) -> i32 {
    let sv = ctx.state_values.as_slice();
    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut total_score: i32 = 0;

    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;

        let mut dice = roll_dice(rng);

        // Group 6: standard (decision node, unchanged)
        compute_group6(ctx, sv, up_score, scored, &mut e_ds_0);
        // Groups 5: max-outcome propagation
        compute_max_outcome_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);

        let mut best_ev = 0.0;
        let mask1 = choose_best_reroll_mask_max(ctx, &e_ds_1, &dice, &mut best_ev);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }

        // Groups 3: already in e_ds_0 from Group 6 (reroll decision uses max-outcome)
        let mask2 = choose_best_reroll_mask_max(ctx, &e_ds_0, &dice, &mut best_ev);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }

        let ds_index = find_dice_set_index(ctx, &dice);
        let (cat, scr) = if is_last_turn {
            find_best_category_final(ctx, up_score, scored, ds_index)
        } else {
            find_best_category(ctx, sv, up_score, scored, ds_index)
        };

        up_score = update_upper_score(up_score, cat, scr);
        scored |= 1 << cat;
        total_score += scr;
    }

    if up_score >= 63 {
        total_score += 50;
    }

    total_score
}

/// Simulate one full game with recording, returning a `GameRecord`.
pub fn simulate_game_with_recording(ctx: &YatzyContext, rng: &mut SmallRng) -> GameRecord {
    let sv = ctx.state_values.as_slice();
    let theta = ctx.theta;
    let use_risk = theta != 0.0;
    let minimize = theta < 0.0;
    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut total_score: i32 = 0;
    let mut record = GameRecord::default();

    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;

        // Roll 5 dice
        let mut dice = roll_dice(rng);
        let dice_initial = dice_to_u8(&dice);

        // Compute EVs (dispatch based on theta)
        if use_risk {
            compute_group6_risk(ctx, sv, up_score, scored, theta, minimize, &mut e_ds_0);
            compute_opt_lse_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1, minimize);
        } else {
            compute_group6(ctx, sv, up_score, scored, &mut e_ds_0);
            compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);
        }

        // First reroll
        let mut best_ev = 0.0;
        let mask1 = if use_risk {
            choose_best_reroll_mask_risk(ctx, &e_ds_1, &dice, &mut best_ev, minimize)
        } else {
            choose_best_reroll_mask(ctx, &e_ds_1, &dice, &mut best_ev)
        };
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }
        let dice_after_reroll1 = dice_to_u8(&dice);

        // Second reroll
        let mask2 = if use_risk {
            choose_best_reroll_mask_risk(ctx, &e_ds_0, &dice, &mut best_ev, minimize)
        } else {
            choose_best_reroll_mask(ctx, &e_ds_0, &dice, &mut best_ev)
        };
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }
        let dice_final = dice_to_u8(&dice);

        // Score
        let ds_index = find_dice_set_index(ctx, &dice);
        let (cat, scr) = if use_risk {
            if is_last_turn {
                find_best_category_final_risk(ctx, up_score, scored, ds_index, theta, minimize)
            } else {
                find_best_category_risk(ctx, sv, up_score, scored, ds_index, theta, minimize)
            }
        } else if is_last_turn {
            find_best_category_final(ctx, up_score, scored, ds_index)
        } else {
            find_best_category(ctx, sv, up_score, scored, ds_index)
        };

        up_score = update_upper_score(up_score, cat, scr);
        scored |= 1 << cat;
        total_score += scr;

        record.turns[turn] = TurnRecord {
            dice_initial,
            mask1: mask1 as u8,
            dice_after_reroll1,
            mask2: mask2 as u8,
            dice_final,
            category: cat as u8,
            score: scr as u8,
        };
    }

    let got_bonus = up_score >= 63;
    if got_bonus {
        total_score += 50;
    }

    record.total_score = total_score as u16;
    record.upper_total = up_score.min(63) as u8;
    record.got_bonus = got_bonus as u8;

    record
}

/// Per-turn summary: just category + score + turn index (no dice/masks).
#[derive(Clone, Copy, Default)]
pub struct TurnSummary {
    pub category: u8,
    pub score: u8,
}

/// Lightweight game summary: 15 turn summaries + total.
#[derive(Clone, Copy, Default)]
pub struct GameSummary {
    pub turns: [TurnSummary; 15],
    pub total_score: i32,
}

/// Simulate one game, returning only category/score per turn (no dice recording).
pub fn simulate_game_summary(ctx: &YatzyContext, rng: &mut SmallRng) -> GameSummary {
    if ctx.max_policy {
        return simulate_game_summary_max(ctx, rng);
    }

    let sv = ctx.state_values.as_slice();
    let theta = ctx.theta;
    let use_risk = theta != 0.0;
    let minimize = theta < 0.0;
    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut total_score: i32 = 0;
    let mut summary = GameSummary::default();

    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;
        let mut dice = roll_dice(rng);

        if use_risk {
            compute_group6_risk(ctx, sv, up_score, scored, theta, minimize, &mut e_ds_0);
            compute_opt_lse_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1, minimize);
        } else {
            compute_group6(ctx, sv, up_score, scored, &mut e_ds_0);
            compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);
        }

        let mut best_ev = 0.0;
        let mask1 = if use_risk {
            choose_best_reroll_mask_risk(ctx, &e_ds_1, &dice, &mut best_ev, minimize)
        } else {
            choose_best_reroll_mask(ctx, &e_ds_1, &dice, &mut best_ev)
        };
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }

        let mask2 = if use_risk {
            choose_best_reroll_mask_risk(ctx, &e_ds_0, &dice, &mut best_ev, minimize)
        } else {
            choose_best_reroll_mask(ctx, &e_ds_0, &dice, &mut best_ev)
        };
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }

        let ds_index = find_dice_set_index(ctx, &dice);
        let (cat, scr) = if use_risk {
            if is_last_turn {
                find_best_category_final_risk(ctx, up_score, scored, ds_index, theta, minimize)
            } else {
                find_best_category_risk(ctx, sv, up_score, scored, ds_index, theta, minimize)
            }
        } else if is_last_turn {
            find_best_category_final(ctx, up_score, scored, ds_index)
        } else {
            find_best_category(ctx, sv, up_score, scored, ds_index)
        };

        up_score = update_upper_score(up_score, cat, scr);
        scored |= 1 << cat;
        total_score += scr;

        summary.turns[turn] = TurnSummary {
            category: cat as u8,
            score: scr as u8,
        };
    }

    if up_score >= 63 {
        total_score += 50;
    }
    summary.total_score = total_score;
    summary
}

/// Max-policy summary: chance nodes use max instead of E[x] for reroll evaluation.
fn simulate_game_summary_max(ctx: &YatzyContext, rng: &mut SmallRng) -> GameSummary {
    let sv = ctx.state_values.as_slice();
    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut total_score: i32 = 0;
    let mut summary = GameSummary::default();

    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;
        let mut dice = roll_dice(rng);

        compute_group6(ctx, sv, up_score, scored, &mut e_ds_0);
        compute_max_outcome_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);

        let mut best_ev = 0.0;
        let mask1 = choose_best_reroll_mask_max(ctx, &e_ds_1, &dice, &mut best_ev);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }

        let mask2 = choose_best_reroll_mask_max(ctx, &e_ds_0, &dice, &mut best_ev);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }

        let ds_index = find_dice_set_index(ctx, &dice);
        let (cat, scr) = if is_last_turn {
            find_best_category_final(ctx, up_score, scored, ds_index)
        } else {
            find_best_category(ctx, sv, up_score, scored, ds_index)
        };

        up_score = update_upper_score(up_score, cat, scr);
        scored |= 1 << cat;
        total_score += scr;

        summary.turns[turn] = TurnSummary {
            category: cat as u8,
            score: scr as u8,
        };
    }

    if up_score >= 63 {
        total_score += 50;
    }
    summary.total_score = total_score;
    summary
}

/// Simulate N games in parallel, returning lightweight summaries.
pub fn simulate_batch_summaries(
    ctx: &YatzyContext,
    num_games: usize,
    seed: u64,
) -> Vec<GameSummary> {
    (0..num_games)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
            simulate_game_summary(ctx, &mut rng)
        })
        .collect()
}

/// Simulate N games in parallel, returning aggregate statistics.
pub fn simulate_batch(ctx: &YatzyContext, num_games: usize, seed: u64) -> SimulationResult {
    let start = Instant::now();

    let mut scores: Vec<i32> = (0..num_games)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
            simulate_game(ctx, &mut rng)
        })
        .collect();

    let elapsed = start.elapsed();

    let sum: f64 = scores.iter().map(|&s| s as f64).sum();
    let mean = sum / num_games as f64;
    let variance: f64 = scores
        .iter()
        .map(|&s| (s as f64 - mean).powi(2))
        .sum::<f64>()
        / num_games as f64;
    let std_dev = variance.sqrt();
    let min = *scores.iter().min().unwrap_or(&0);
    let max = *scores.iter().max().unwrap_or(&0);

    scores.sort_unstable();
    let median = scores[num_games / 2];

    SimulationResult {
        scores,
        mean,
        std_dev,
        min,
        max,
        median,
        elapsed,
    }
}

/// Simulate N games in parallel with full recording, returning all GameRecords.
pub fn simulate_batch_with_recording(
    ctx: &YatzyContext,
    num_games: usize,
    seed: u64,
) -> Vec<GameRecord> {
    (0..num_games)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
            simulate_game_with_recording(ctx, &mut rng)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0_tables;

    fn make_ctx() -> Box<YatzyContext> {
        let mut ctx = YatzyContext::new_boxed();
        phase0_tables::precompute_lookup_tables(&mut ctx);
        ctx
    }

    #[test]
    fn test_roll_dice_range() {
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..100 {
            let dice = roll_dice(&mut rng);
            for &d in &dice {
                assert!(d >= 1 && d <= 6);
            }
            for i in 0..4 {
                assert!(dice[i] <= dice[i + 1]);
            }
        }
    }

    #[test]
    fn test_apply_reroll() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut dice = [1, 2, 3, 4, 5];

        let original = dice;
        apply_reroll(&mut dice, 0, &mut rng);
        assert_eq!(dice, original);

        apply_reroll(&mut dice, 31, &mut rng);
        for &d in &dice {
            assert!(d >= 1 && d <= 6);
        }
        for i in 0..4 {
            assert!(dice[i] <= dice[i + 1]);
        }
    }

    #[test]
    fn test_simulate_game_produces_valid_score() {
        let ctx = make_ctx();
        let mut rng = SmallRng::seed_from_u64(42);
        let score = simulate_game(&ctx, &mut rng);
        assert!(score >= 0, "Score {} should be non-negative", score);
        assert!(score <= 374, "Score {} exceeds maximum possible", score);
    }

    #[test]
    fn test_simulate_game_deterministic() {
        let ctx = make_ctx();
        let mut rng1 = SmallRng::seed_from_u64(123);
        let mut rng2 = SmallRng::seed_from_u64(123);
        let score1 = simulate_game(&ctx, &mut rng1);
        let score2 = simulate_game(&ctx, &mut rng2);
        assert_eq!(score1, score2, "Same seed should produce same score");
    }

    #[test]
    fn test_turn_record_size() {
        assert_eq!(std::mem::size_of::<TurnRecord>(), 19);
    }

    #[test]
    fn test_game_record_size() {
        // 15 * 19 = 285 + 2 + 1 + 1 = 289
        assert_eq!(std::mem::size_of::<GameRecord>(), 289);
    }

    #[test]
    fn test_simulate_game_with_recording_valid() {
        let ctx = make_ctx();
        let mut rng = SmallRng::seed_from_u64(42);
        let record = simulate_game_with_recording(&ctx, &mut rng);

        // Copy packed fields to locals to avoid unaligned references
        let total = record.total_score;
        let upper = record.upper_total;
        let bonus = record.got_bonus;
        let turns = record.turns;

        // Total score should be valid
        assert!(total <= 374);

        // All 15 categories should be used exactly once
        let mut seen = [false; CATEGORY_COUNT];
        let mut upper_sum: i32 = 0;
        for t in 0..15 {
            let turn = turns[t];
            let cat = turn.category;
            let scr = turn.score;

            assert!((cat as usize) < CATEGORY_COUNT, "Invalid category {}", cat);
            assert!(!seen[cat as usize], "Category {} scored twice", cat);
            seen[cat as usize] = true;

            if (cat as usize) < 6 {
                upper_sum += scr as i32;
            }

            // Dice values should be in [1,6]
            for &d in &turn.dice_initial {
                assert!(d >= 1 && d <= 6);
            }
            for &d in &turn.dice_after_reroll1 {
                assert!(d >= 1 && d <= 6);
            }
            for &d in &turn.dice_final {
                assert!(d >= 1 && d <= 6);
            }
        }
        assert!(seen.iter().all(|&s| s), "Not all categories scored");

        // Upper total should match
        assert_eq!(upper as i32, upper_sum.min(63), "Upper total mismatch");

        // Bonus consistency
        if upper >= 63 {
            assert_eq!(bonus, 1);
        } else {
            assert_eq!(bonus, 0);
        }
    }

    #[test]
    fn test_recording_matches_non_recording() {
        // With the same seed, the recorded game should produce the same total score
        // as the non-recording version.
        let ctx = make_ctx();
        let mut rng1 = SmallRng::seed_from_u64(999);
        let mut rng2 = SmallRng::seed_from_u64(999);

        let score = simulate_game(&ctx, &mut rng1);
        let record = simulate_game_with_recording(&ctx, &mut rng2);

        let recorded_total = record.total_score;
        assert_eq!(
            score, recorded_total as i32,
            "Recording should produce same score: {} vs {}",
            score, recorded_total
        );
    }
}
