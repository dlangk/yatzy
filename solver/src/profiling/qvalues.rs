//! Q-value computation for profiling scenarios.
//!
//! Computes Q(s, a | θ, γ, σ_d) for both category and reroll decisions.
//! Uses deterministic hash-based noise to approximate bounded-depth play.

#![allow(clippy::needless_range_loop)]

use rand::Rng;

use crate::constants::*;
use crate::dice_mechanics::find_dice_set_index;
use crate::game_mechanics::update_upper_score;
use crate::types::YatzyContext;
use crate::widget_solver::{compute_max_ev_for_n_rerolls, compute_opt_lse_for_n_rerolls};

/// Deterministic noise for simulating bounded-depth value estimation.
/// Returns V_solver(s) + σ_d · z where z is a deterministic pseudo-uniform in [-1, 1].
#[inline]
pub fn perturbed_state_value(sv: &[f32], idx: usize, sigma_d: f32) -> f32 {
    if sigma_d == 0.0 {
        return sv[idx];
    }
    let hash = (idx as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let z = ((hash >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32;
    sv[idx] + sigma_d * z
}

/// σ_d lookup from surrogate tree calibration data.
/// Maps a depth parameter to the noise standard deviation.
pub fn sigma_for_depth(d: u32) -> f32 {
    match d {
        0..=5 => 25.0,
        6..=7 => 20.0,
        8 => 15.0,
        9 => 12.0,
        10 => 10.0,
        11..=12 => 7.0,
        13..=14 => 5.0,
        15..=17 => 4.0,
        18..=19 => 3.0,
        20..=30 => 2.0,
        _ => 0.0, // d=999 means perfect
    }
}

/// Compute Group 6 (best category EV for each dice set) with γ-discounting and depth noise.
///
/// For the profiling model:
/// Q_cat(s, c | γ, σ_d) = R(s,c) + γ · V_d(successor)
///
/// where V_d(s') = V_solver(s') + σ_d · hash_noise(s')
#[inline]
pub fn compute_group6_profiling(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    gamma: f32,
    sigma_d: f32,
    e_ds_0: &mut [f32; 252],
) {
    let mut lower_succ_ev = [0.0f32; CATEGORY_COUNT];
    for c in 6..CATEGORY_COUNT {
        if !is_category_scored(scored, c) {
            let idx = state_index(up_score as usize, (scored | (1 << c)) as usize);
            lower_succ_ev[c] = perturbed_state_value(sv, idx, sigma_d);
        }
    }
    for ds_i in 0..252 {
        let mut best_val = f32::NEG_INFINITY;
        for c in 0..6 {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_i][c];
                let new_up = update_upper_score(up_score, c, scr);
                let new_scored = scored | (1 << c);
                let idx = state_index(new_up as usize, new_scored as usize);
                let val = scr as f32 + gamma * perturbed_state_value(sv, idx, sigma_d);
                if val > best_val {
                    best_val = val;
                }
            }
        }
        for c in 6..CATEGORY_COUNT {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_i][c];
                let val = scr as f32 + gamma * unsafe { *lower_succ_ev.get_unchecked(c) };
                if val > best_val {
                    best_val = val;
                }
            }
        }
        e_ds_0[ds_i] = best_val;
    }
}

/// Compute Group 6 with risk-sensitive (θ-scaled) scoring, γ-discounting, and depth noise.
#[inline]
pub fn compute_group6_risk_profiling(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    theta: f32,
    gamma: f32,
    sigma_d: f32,
    e_ds_0: &mut [f32; 252],
) {
    let mut lower_succ_ev = [0.0f32; CATEGORY_COUNT];
    for c in 6..CATEGORY_COUNT {
        if !is_category_scored(scored, c) {
            let idx = state_index(up_score as usize, (scored | (1 << c)) as usize);
            lower_succ_ev[c] = perturbed_state_value(sv, idx, sigma_d);
        }
    }
    for ds_i in 0..252 {
        let mut best_val = f32::NEG_INFINITY;
        for c in 0..6 {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_i][c];
                let new_up = update_upper_score(up_score, c, scr);
                let new_scored = scored | (1 << c);
                let idx = state_index(new_up as usize, new_scored as usize);
                let val = theta * scr as f32 + gamma * perturbed_state_value(sv, idx, sigma_d);
                if val > best_val {
                    best_val = val;
                }
            }
        }
        for c in 6..CATEGORY_COUNT {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_i][c];
                let val = theta * scr as f32 + gamma * unsafe { *lower_succ_ev.get_unchecked(c) };
                if val > best_val {
                    best_val = val;
                }
            }
        }
        e_ds_0[ds_i] = best_val;
    }
}

/// Q-values for all category actions at a given state.
///
/// Returns Vec<(category_index, q_value)> for all open categories.
pub fn compute_q_categories(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    ds_index: usize,
    gamma: f32,
    sigma_d: f32,
    is_last_turn: bool,
) -> Vec<(usize, f32)> {
    let mut result = Vec::with_capacity(CATEGORY_COUNT);

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
                result.push((c, (scr + bonus) as f32));
            }
        }
    } else {
        for c in 0..6 {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let new_up = update_upper_score(up_score, c, scr);
                let new_scored = scored | (1 << c);
                let idx = state_index(new_up as usize, new_scored as usize);
                let val = scr as f32 + gamma * perturbed_state_value(sv, idx, sigma_d);
                result.push((c, val));
            }
        }
        for c in 6..CATEGORY_COUNT {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let new_scored = scored | (1 << c);
                let idx = state_index(up_score as usize, new_scored as usize);
                let val = scr as f32 + gamma * perturbed_state_value(sv, idx, sigma_d);
                result.push((c, val));
            }
        }
    }

    result
}

/// Q-values for all reroll actions at a given state.
///
/// For reroll decisions, θ affects the stochastic node (CE) but γ doesn't
/// directly apply within a turn. σ_d affects the Group 6 values that feed
/// into the reroll computation.
///
/// Returns Vec<(mask, q_value)> for all legal reroll masks.
pub fn compute_q_rerolls(
    ctx: &YatzyContext,
    e_ds: &[f32; 252],
    dice: &[i32; 5],
    is_risk: bool,
) -> Vec<(i32, f32)> {
    let ds_index = find_dice_set_index(ctx, dice);
    let kt = &ctx.keep_table;

    let mut result: Vec<(i32, f32)> = Vec::with_capacity(32);

    // mask=0: keep all
    result.push((0, e_ds[ds_index]));

    for j in 0..kt.unique_count[ds_index] as usize {
        let kid = kt.unique_keep_ids[ds_index][j] as usize;
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;

        let ev = if is_risk {
            // LSE for stochastic node
            let mut max_x = f32::NEG_INFINITY;
            for k in start..end {
                let v = unsafe { *e_ds.get_unchecked(*kt.cols.get_unchecked(k) as usize) };
                if v > max_x {
                    max_x = v;
                }
            }
            let mut sum: f32 = 0.0;
            for k in start..end {
                unsafe {
                    let v = *e_ds.get_unchecked(*kt.cols.get_unchecked(k) as usize);
                    sum += (*kt.vals.get_unchecked(k) as f32) * (v - max_x).exp();
                }
            }
            max_x + sum.ln()
        } else {
            let mut ev: f32 = 0.0;
            for k in start..end {
                unsafe {
                    ev += (*kt.vals.get_unchecked(k) as f32)
                        * e_ds.get_unchecked(*kt.cols.get_unchecked(k) as usize);
                }
            }
            ev
        };

        let mask = kt.keep_to_mask[ds_index * 32 + j];
        result.push((mask, ev));
    }

    result
}

/// Sample an action index from softmax(β · Q) distribution.
///
/// Given a list of (action_id, q_value) pairs, returns the action_id chosen
/// by sampling from the softmax distribution with temperature 1/β.
/// Higher β means more deterministic (closer to argmax).
pub fn softmax_sample(q_values: &[(i32, f32)], beta: f32, rng: &mut impl Rng) -> i32 {
    if q_values.is_empty() {
        return 0;
    }
    if q_values.len() == 1 {
        return q_values[0].0;
    }

    // Compute max for numerical stability
    let max_q = q_values
        .iter()
        .map(|(_, q)| *q)
        .fold(f32::NEG_INFINITY, f32::max);

    // Compute unnormalized softmax weights
    let weights: Vec<f32> = q_values
        .iter()
        .map(|(_, q)| (beta * (q - max_q)).exp())
        .collect();
    let total: f32 = weights.iter().sum();

    // Sample from the distribution
    let mut r: f32 = rng.random::<f32>() * total;
    for (i, w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return q_values[i].0;
        }
    }
    // Fallback to last (rounding)
    q_values.last().unwrap().0
}

/// Compute full e_ds buffers for a scenario at given parameters.
///
/// Returns (e_ds_group6, e_ds_reroll1) where:
/// - e_ds_group6 is used for reroll2 decisions and category decisions
/// - e_ds_reroll1 is used for reroll1 decisions
pub fn compute_eds_for_scenario(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    theta: f32,
    gamma: f32,
    sigma_d: f32,
) -> ([f32; 252], [f32; 252]) {
    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    let is_risk = theta != 0.0 && theta != 1.0;

    if is_risk {
        compute_group6_risk_profiling(
            ctx,
            sv,
            up_score,
            scored,
            theta,
            gamma,
            sigma_d,
            &mut e_ds_0,
        );
        compute_opt_lse_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1, false);
    } else {
        compute_group6_profiling(ctx, sv, up_score, scored, gamma, sigma_d, &mut e_ds_0);
        compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);
    }

    (e_ds_0, e_ds_1)
}
