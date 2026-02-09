//! API-facing computation wrappers.
//!
//! These functions compose the widget solver primitives for use by HTTP handlers.
//! Unlike the precomputation hot path (which only computes E(S)), these also track
//! optimal masks, evaluate user-chosen actions, and build outcome distributions.
//!
//! Each function partially re-solves a widget on demand: it builds Group 6 (category
//! scoring) and optionally Groups 5/3 (reroll levels), then queries the result for
//! the specific dice/mask/category the API caller requested.

use crate::constants::*;
use crate::dice_mechanics::{find_dice_set_index, sort_dice_set};
use crate::game_mechanics::update_upper_score;
use crate::types::YatzyContext;
use crate::widget_solver::*;

/// (EV, probability, dice-set index) triple for outcome distributions.
#[derive(Clone, Copy)]
pub struct EVProbabilityPair {
    pub ev: f64,
    pub probability: f64,
    pub ds2_index: usize,
}

/// Compute best reroll mask for a specific game situation.
///
/// Partially solves the widget: builds Group 6, optionally Group 5, then finds
/// the argmax mask for the given dice and reroll count.
pub fn compute_best_reroll_strategy(
    ctx: &YatzyContext,
    upper_score: i32,
    scored_categories: i32,
    dice: &[i32; 5],
    rerolls_remaining: i32,
    best_mask: &mut i32,
    best_ev: &mut f64,
) {
    // Group 6: E(S, r, 0) for all r
    let mut e_ds_0 = [0.0f64; 252];
    let state = crate::types::YatzyState {
        upper_score,
        scored_categories,
    };
    for ds_i in 0..252 {
        e_ds_0[ds_i] =
            compute_best_scoring_value_for_dice_set(ctx, &state, &ctx.all_dice_sets[ds_i]);
    }

    if rerolls_remaining == 1 {
        *best_mask = choose_best_reroll_mask(ctx, &e_ds_0, dice, best_ev);
        return;
    }

    let mut e_ds_1 = [0.0f64; 252];
    let mut dummy_mask = [0i32; 252];
    compute_expected_values_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1, &mut dummy_mask);
    *best_mask = choose_best_reroll_mask(ctx, &e_ds_1, dice, best_ev);
}

/// Choose best category when no rerolls remain.
///
/// Group 6 only: E(S, r, 0) = max_{c ∉ C, s>0} [s(S,r,c) + E_table[n(S,r,c)]]
/// Skips categories that would score 0 (no point in wasting a category).
pub fn choose_best_category_no_rerolls(
    ctx: &YatzyContext,
    upper_score: i32,
    scored_categories: i32,
    dice: &[i32; 5],
    best_ev: &mut f64,
) -> i32 {
    let ds_index = find_dice_set_index(ctx, dice);
    let sv = ctx.state_values.as_slice();
    let mut best_val = f64::NEG_INFINITY;
    let mut best_category = -1i32;

    for c in 0..CATEGORY_COUNT {
        if is_category_scored(scored_categories, c) {
            continue;
        }
        let scr = ctx.precomputed_scores[ds_index][c];
        if scr == 0 {
            continue;
        }
        let new_up = update_upper_score(upper_score, c, scr);
        let new_scored = scored_categories | (1 << c);
        let val = scr as f64 + sv[state_index(new_up as usize, new_scored as usize)] as f64;
        if val > best_val {
            best_val = val;
            best_category = c as i32;
        }
    }

    *best_ev = best_val;
    best_category
}

/// Evaluate the EV of a user-chosen reroll mask.
///
/// Used by the "evaluate user action" endpoint to score a player's reroll decision
/// against the optimal strategy. Builds the necessary widget levels, then computes
/// the expected value of the specific mask the user chose.
pub fn evaluate_chosen_reroll_mask(
    ctx: &YatzyContext,
    upper_score: i32,
    scored_categories: i32,
    dice: &[i32; 5],
    chosen_mask: i32,
    rerolls_remaining: i32,
) -> f64 {
    // Group 6: E(S, r, 0) for all r
    let mut e_ds_0 = [0.0f64; 252];
    let state = crate::types::YatzyState {
        upper_score,
        scored_categories,
    };
    for ds_i in 0..252 {
        e_ds_0[ds_i] =
            compute_best_scoring_value_for_dice_set(ctx, &state, &ctx.all_dice_sets[ds_i]);
    }

    let mut e_ds_1 = [0.0f64; 252];
    let mut dummy_mask = [0i32; 252];
    if rerolls_remaining == 2 {
        compute_expected_values_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1, &mut dummy_mask);
    }

    let e_ds_for_masks = if rerolls_remaining == 1 {
        &e_ds_0
    } else {
        &e_ds_1
    };

    let mut sorted_dice = *dice;
    sort_dice_set(&mut sorted_dice);
    let ds_index = find_dice_set_index(ctx, &sorted_dice);

    compute_expected_value_for_reroll_mask(ctx, ds_index, e_ds_for_masks, chosen_mask)
}

/// Evaluate the EV of a user-chosen category assignment.
///
/// Returns s(S,r,c) + E_table[n(S,r,c)] for the chosen category, or NEG_INFINITY
/// if the category is already scored or would score 0.
pub fn evaluate_chosen_category(
    ctx: &YatzyContext,
    upper_score: i32,
    scored_categories: i32,
    dice: &[i32; 5],
    chosen_category: usize,
) -> f64 {
    if is_category_scored(scored_categories, chosen_category) {
        return f64::NEG_INFINITY;
    }
    let ds_index = find_dice_set_index(ctx, dice);
    let score = ctx.precomputed_scores[ds_index][chosen_category];
    if score == 0 {
        return f64::NEG_INFINITY;
    }
    let sv = ctx.state_values.as_slice();
    let new_up = update_upper_score(upper_score, chosen_category, score);
    let new_scored = scored_categories | (1 << chosen_category);
    let future_val = sv[state_index(new_up as usize, new_scored as usize)] as f64;
    score as f64 + future_val
}

/// Compute E(S, r, n) for all 252 dice sets r, given n rerolls remaining.
///
/// Builds the full Group 6 → Group 5 → Group 3 chain up to the requested reroll
/// depth. Used by the `/evaluate_actions` endpoint to produce EV for all 32 masks.
pub fn compute_expected_values(
    ctx: &YatzyContext,
    upper_score: i32,
    scored_categories: i32,
    rerolls: i32,
    out_e_ds: &mut [f64; 252],
) {
    let mut e_ds = [[0.0f64; 252]; 3];
    let sv = ctx.state_values.as_slice();

    // Level 0: E(S, r, 0) = max_c [s(S,r,c) + E(n(S,r,c))]
    for ds_i in 0..252 {
        let mut best_val = f64::NEG_INFINITY;
        for c in 0..CATEGORY_COUNT {
            if !is_category_scored(scored_categories, c) {
                let scr = ctx.precomputed_scores[ds_i][c];
                let new_up = update_upper_score(upper_score, c, scr);
                let new_scored = scored_categories | (1 << c);
                let val = scr as f64 + sv[state_index(new_up as usize, new_scored as usize)] as f64;
                if val > best_val {
                    best_val = val;
                }
            }
        }
        e_ds[0][ds_i] = best_val;
    }

    // Levels 1..rerolls (with keep EV dedup)
    let kt = &ctx.keep_table;
    let vals = &kt.vals;
    let cols = &kt.cols;
    for n in 1..=rerolls as usize {
        // Step 1: compute EV for each unique keep-multiset (462 dot products)
        let mut keep_ev = [0.0f64; NUM_KEEP_MULTISETS];
        for kid in 0..NUM_KEEP_MULTISETS {
            let start = kt.row_start[kid] as usize;
            let end = kt.row_start[kid + 1] as usize;
            let mut ev = 0.0;
            for k in start..end {
                unsafe {
                    ev += vals.get_unchecked(k)
                        * e_ds[n - 1].get_unchecked(*cols.get_unchecked(k) as usize);
                }
            }
            keep_ev[kid] = ev;
        }

        // Step 2: for each dice set, find max over its unique keeps
        for ds_i in 0..252 {
            let mut best_val = e_ds[n - 1][ds_i]; // mask=0: keep all
            for j in 0..kt.unique_count[ds_i] as usize {
                let kid = kt.unique_keep_ids[ds_i][j] as usize;
                if keep_ev[kid] > best_val {
                    best_val = keep_ev[kid];
                }
            }
            e_ds[n][ds_i] = best_val;
        }
    }

    // Copy result to caller's buffer
    out_e_ds.copy_from_slice(&e_ds[rerolls as usize]);
}

/// Build outcome distribution for a specific reroll mask.
///
/// For mask=0 (keep all): deterministic — 100% probability on current dice set.
/// For mask>0: distributes probability across all 252 possible outcomes using
/// the keep-multiset's sparse CSR row.
pub fn compute_distribution_for_reroll_mask(
    ctx: &YatzyContext,
    ds_index: usize,
    e_ds_for_masks: &[f64; 252],
    mask: i32,
    out_distribution: &mut [EVProbabilityPair; 252],
) {
    for ds2_i in 0..252 {
        out_distribution[ds2_i] = EVProbabilityPair {
            ev: e_ds_for_masks[ds2_i],
            probability: 0.0,
            ds2_index: ds2_i,
        };
    }

    if mask == 0 {
        out_distribution[ds_index].probability = 1.0;
    } else {
        let kt = &ctx.keep_table;
        let kid = kt.mask_to_keep[ds_index * 32 + mask as usize] as usize;
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;
        for k in start..end {
            out_distribution[kt.cols[k] as usize].probability = kt.vals[k]; // not hot path
        }
    }
}

/// Weighted sum of EV over a probability distribution.
pub fn compute_ev_from_distribution(distribution: &[EVProbabilityPair], size: usize) -> f64 {
    let mut ev = 0.0;
    for i in 0..size {
        ev += distribution[i].ev * distribution[i].probability;
    }
    ev
}
