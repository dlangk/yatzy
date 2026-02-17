//! Streamlined noisy simulation for player card grid pre-computation.
//!
//! `simulate_game_profiled` plays one full game using the noisy agent model
//! (β, γ, σ_d) and returns only the final score. No decision collection.
//!
//! The θ parameter only affects which strategy table (state values) is loaded.
//! Q-value computation always uses EV-mode (matching `simulate_game_collecting_noisy`).

#![allow(clippy::needless_range_loop)]

use rand::rngs::SmallRng;
use rand::Rng;

use crate::constants::*;
use crate::dice_mechanics::{find_dice_set_index, sort_dice_set};
use crate::game_mechanics::update_upper_score;
use crate::profiling::qvalues::{
    compute_group6_profiling, compute_q_categories, compute_q_rerolls, softmax_sample,
};
use crate::types::YatzyContext;
use crate::widget_solver::compute_max_ev_for_n_rerolls;

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

/// Apply a reroll mask: re-roll dice at positions where mask bit is set.
#[inline(always)]
fn apply_reroll(dice: &mut [i32; 5], mask: i32, rng: &mut SmallRng) {
    for i in 0..5 {
        if mask & (1 << i) != 0 {
            dice[i] = rng.random_range(1..=6);
        }
    }
    sort_dice_set(dice);
}

/// Find best category on the last turn (greedy: maximize immediate score + bonus).
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

/// Simulate one full game with noisy agent model and return total score (including bonus).
///
/// Uses softmax(β·Q) action selection with γ-discounted future values and
/// σ_d depth noise. Q-value computation is always in EV-mode — the θ parameter
/// only affects which strategy table (state values) backs the decisions.
pub fn simulate_game_profiled(
    ctx: &YatzyContext,
    rng: &mut SmallRng,
    beta: f32,
    gamma: f32,
    sigma_d: f32,
) -> i32 {
    let sv = ctx.state_values.as_slice();

    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut total_score: i32 = 0;

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;
        let mut dice = roll_dice(rng);

        // Compute Group 6 EV buffer (always EV-mode, matching scenarios.rs)
        let mut e_ds_0 = [0.0f32; 252];
        let mut e_ds_1 = [0.0f32; 252];
        compute_group6_profiling(ctx, sv, up_score, scored, gamma, sigma_d, &mut e_ds_0);
        compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);

        // Reroll 1
        let reroll1_qs = compute_q_rerolls(ctx, &e_ds_1, &dice, false);
        let mask1 = softmax_sample(&reroll1_qs, beta, rng);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }

        // Reroll 2
        let reroll2_qs = compute_q_rerolls(ctx, &e_ds_0, &dice, false);
        let mask2 = softmax_sample(&reroll2_qs, beta, rng);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }

        // Category selection
        let ds_index = find_dice_set_index(ctx, &dice);
        let (cat, scr) = if is_last_turn {
            find_best_category_final(ctx, up_score, scored, ds_index)
        } else {
            let cat_qs =
                compute_q_categories(ctx, sv, up_score, scored, ds_index, gamma, sigma_d, false);
            let chosen_cat = softmax_sample(
                &cat_qs
                    .iter()
                    .map(|(c, v)| (*c as i32, *v))
                    .collect::<Vec<_>>(),
                beta,
                rng,
            ) as usize;
            let scr = ctx.precomputed_scores[ds_index][chosen_cat];
            (chosen_cat, scr)
        };

        total_score += scr;
        up_score = update_upper_score(up_score, cat, scr);
        scored |= 1 << cat;
    }

    // Add upper bonus if earned
    if up_score >= 63 {
        total_score += 50;
    }

    total_score
}
