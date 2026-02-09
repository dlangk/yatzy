use crate::constants::*;
use crate::dice_mechanics::{find_dice_set_index, sort_dice_set};
use crate::game_mechanics::update_upper_score;
use crate::types::{YatzyContext, YatzyState};

/// Group 6, inner loop: best category for one roll r.
/// E(S, r, 0) = max_{c not in C} [s(S,r,c) + E(n(S,r,c))]
pub fn compute_best_scoring_value_for_dice_set(
    ctx: &YatzyContext,
    state: &YatzyState,
    dice: &[i32; 5],
) -> f64 {
    let ds_index = find_dice_set_index(ctx, dice);
    compute_best_scoring_value_for_dice_set_by_index(
        ctx,
        state.upper_score,
        state.scored_categories,
        ds_index,
    )
}

/// Group 6, inner loop variant that takes ds_index directly.
#[inline]
pub fn compute_best_scoring_value_for_dice_set_by_index(
    ctx: &YatzyContext,
    up_score: i32,
    scored: i32,
    ds_index: usize,
) -> f64 {
    let mut best_val = f64::NEG_INFINITY;

    // Upper categories (0-5): affect upper score
    for c in 0..6 {
        if !is_category_scored(scored, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            let new_up = update_upper_score(up_score, c, scr);
            let new_scored = scored | (1 << c);
            let val = scr as f64 + ctx.get_state_value(new_up, new_scored);
            if val > best_val {
                best_val = val;
            }
        }
    }

    // Lower categories (6-14): upper score unchanged
    for c in 6..CATEGORY_COUNT {
        if !is_category_scored(scored, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            let new_scored = scored | (1 << c);
            let val = scr as f64 + ctx.get_state_value(up_score, new_scored);
            if val > best_val {
                best_val = val;
            }
        }
    }

    best_val
}

/// Single mask evaluation: sum P(r'->r'') * E_prev[r''].
#[inline]
pub fn compute_expected_value_for_reroll_mask(
    ctx: &YatzyContext,
    ds_index: usize,
    e_ds_for_masks: &[f64; 252],
    mask: i32,
) -> f64 {
    if mask == 0 {
        return e_ds_for_masks[ds_index];
    }
    let kt = &ctx.keep_table;
    let kid = kt.mask_to_keep[ds_index * 32 + mask as usize] as usize;
    let start = kt.row_start[kid] as usize;
    let end = kt.row_start[kid + 1] as usize;
    let mut ev = 0.0;
    for k in start..end {
        ev += kt.vals[k] * e_ds_for_masks[kt.cols[k] as usize];
    }
    ev
}

/// Find argmax mask: best reroll decision for a specific dice set.
pub fn choose_best_reroll_mask(
    ctx: &YatzyContext,
    e_ds_for_masks: &[f64; 252],
    dice: &[i32; 5],
    best_ev: &mut f64,
) -> i32 {
    let mut sorted_dice = *dice;
    sort_dice_set(&mut sorted_dice);
    let ds_index = find_dice_set_index(ctx, &sorted_dice);

    let kt = &ctx.keep_table;

    // mask=0: keep all
    let mut best_val = e_ds_for_masks[ds_index];
    let mut best_mask = 0i32;

    for j in 0..kt.unique_count[ds_index] as usize {
        let kid = kt.unique_keep_ids[ds_index][j] as usize;
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;
        let mut ev = 0.0;
        for k in start..end {
            ev += kt.vals[k] * e_ds_for_masks[kt.cols[k] as usize];
        }
        if ev > best_val {
            best_val = ev;
            best_mask = kt.keep_to_mask[ds_index * 32 + j];
        }
    }

    *best_ev = best_val;
    best_mask
}

/// Groups 5 & 3 (API path): propagate expected values with mask tracking.
pub fn compute_expected_values_for_n_rerolls(
    ctx: &YatzyContext,
    e_ds_prev: &[f64; 252],
    e_ds_current: &mut [f64; 252],
    best_mask_for_n: &mut [i32; 252],
) {
    let kt = &ctx.keep_table;

    for ds_i in 0..252 {
        let mut best_val = e_ds_prev[ds_i]; // mask=0: keep all
        let mut best_mask = 0i32;
        for j in 0..kt.unique_count[ds_i] as usize {
            let kid = kt.unique_keep_ids[ds_i][j] as usize;
            let start = kt.row_start[kid] as usize;
            let end = kt.row_start[kid + 1] as usize;
            let mut ev = 0.0;
            for k in start..end {
                ev += kt.vals[k] * e_ds_prev[kt.cols[k] as usize];
            }
            if ev > best_val {
                best_val = ev;
                best_mask = kt.keep_to_mask[ds_i * 32 + j];
            }
        }
        e_ds_current[ds_i] = best_val;
        best_mask_for_n[ds_i] = best_mask;
    }
}

/// DP-only variant: computes E_ds_current without tracking optimal masks.
fn compute_max_ev_for_n_rerolls(
    ctx: &YatzyContext,
    e_ds_prev: &[f64; 252],
    e_ds_current: &mut [f64; 252],
) {
    let kt = &ctx.keep_table;

    for ds_i in 0..252 {
        let mut best_val = e_ds_prev[ds_i]; // mask=0: keep all
        for j in 0..kt.unique_count[ds_i] as usize {
            let kid = kt.unique_keep_ids[ds_i][j] as usize;
            let start = kt.row_start[kid] as usize;
            let end = kt.row_start[kid + 1] as usize;
            let mut ev = 0.0;
            for k in start..end {
                ev += kt.vals[k] * e_ds_prev[kt.cols[k] as usize];
            }
            if ev > best_val {
                best_val = ev;
            }
        }
        e_ds_current[ds_i] = best_val;
    }
}

/// SOLVE_WIDGET(S): compute E(S) for one turn-start state.
///
/// Evaluates the widget bottom-up using ping-pong buffers:
///   1. Group 6: E[0][r] = best category score for each final roll
///   2. Group 5: E[1][r] = best reroll from E[0] (1 reroll remaining)
///   3. Group 3: E[0][r] = best reroll from E[1] (2 rerolls remaining)
///   4. Group 1: E(S) = sum P(empty->r) * E[0][r]
pub fn compute_expected_state_value(ctx: &YatzyContext, state: &YatzyState) -> f64 {
    let mut e = [[0.0f64; 252]; 2]; // ping-pong buffers

    let up_score = state.upper_score;
    let scored = state.scored_categories;

    // Group 6: E(S, r, 0) for all r
    for ds_i in 0..252 {
        e[0][ds_i] = compute_best_scoring_value_for_dice_set_by_index(ctx, up_score, scored, ds_i);
    }

    // Group 5: E(S, r, 1) = max_{mask} sum P(r'->r'') * E[0][r'']
    let (e0, e1) = e.split_at_mut(1);
    compute_max_ev_for_n_rerolls(ctx, array_ref(&e0[0]), array_mut(&mut e1[0]));

    // Group 3: E(S, r, 2) = max_{mask} sum P(r'->r'') * E[1][r'']
    let (e0, e1) = e.split_at_mut(1);
    compute_max_ev_for_n_rerolls(ctx, array_ref(&e1[0]), array_mut(&mut e0[0]));

    // Group 1: E(S) = sum P(empty->r) * E[0][r]
    let mut e_s = 0.0;
    for ds_i in 0..252 {
        e_s += ctx.dice_set_probabilities[ds_i] * e[0][ds_i];
    }

    e_s
}

// Helper to borrow a [f64; 252] from a slice
fn array_ref(slice: &[f64; 252]) -> &[f64; 252] {
    slice
}

fn array_mut(slice: &mut [f64; 252]) -> &mut [f64; 252] {
    slice
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0_tables;

    fn make_ctx() -> Box<YatzyContext> {
        let mut ctx = Box::new(YatzyContext::new());
        phase0_tables::precompute_lookup_tables(&mut ctx);
        ctx
    }

    #[test]
    fn test_late_game_scoring() {
        let ctx = make_ctx();
        let all_but_yatzy = ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_YATZY);

        let state = YatzyState {
            upper_score: 0,
            scored_categories: all_but_yatzy,
        };
        let d1 = [6, 6, 6, 6, 6];
        let ev1 = compute_best_scoring_value_for_dice_set(&ctx, &state, &d1);
        assert!((ev1 - 50.0).abs() < 1e-9);

        let d2 = [1, 2, 3, 4, 5];
        let ev2 = compute_best_scoring_value_for_dice_set(&ctx, &state, &d2);
        assert!((ev2 - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_late_game_chance_only() {
        let ctx = make_ctx();
        let all_but_chance = ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_CHANCE);

        let state = YatzyState {
            upper_score: 0,
            scored_categories: all_but_chance,
        };
        let d1 = [6, 6, 6, 6, 6];
        let ev1 = compute_best_scoring_value_for_dice_set(&ctx, &state, &d1);
        assert!((ev1 - 30.0).abs() < 1e-9);

        let d2 = [1, 1, 1, 1, 1];
        let ev2 = compute_best_scoring_value_for_dice_set(&ctx, &state, &d2);
        assert!((ev2 - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_keep_all_mask() {
        let ctx = make_ctx();
        let all_but_yatzy = ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_YATZY);

        let mut e_ds_0 = [0.0f64; 252];
        let state = YatzyState {
            upper_score: 0,
            scored_categories: all_but_yatzy,
        };
        for ds_i in 0..252 {
            e_ds_0[ds_i] =
                compute_best_scoring_value_for_dice_set(&ctx, &state, &ctx.all_dice_sets[ds_i]);
        }

        let ev_keep = compute_expected_value_for_reroll_mask(&ctx, 251, &e_ds_0, 0);
        assert!((ev_keep - e_ds_0[251]).abs() < 1e-9);

        let ev_keep0 = compute_expected_value_for_reroll_mask(&ctx, 0, &e_ds_0, 0);
        assert!((ev_keep0 - e_ds_0[0]).abs() < 1e-9);
    }

    #[test]
    fn test_choose_best_reroll() {
        let ctx = make_ctx();
        let all_but_yatzy = ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_YATZY);

        let mut e_ds_0 = [0.0f64; 252];
        let state = YatzyState {
            upper_score: 0,
            scored_categories: all_but_yatzy,
        };
        for ds_i in 0..252 {
            e_ds_0[ds_i] =
                compute_best_scoring_value_for_dice_set(&ctx, &state, &ctx.all_dice_sets[ds_i]);
        }

        for ds in (0..252).step_by(50) {
            let mut best_ev = 0.0;
            let mask = choose_best_reroll_mask(&ctx, &e_ds_0, &ctx.all_dice_sets[ds], &mut best_ev);
            assert!(mask >= 0 && mask < 32);
            let ev_keep = compute_expected_value_for_reroll_mask(&ctx, ds, &e_ds_0, 0);
            assert!(best_ev >= ev_keep - 1e-9);
        }
    }

    #[test]
    fn bench_widget() {
        let ctx = make_ctx();
        use std::time::Instant;

        println!("\n--- Widget Solver Benchmarks ---");

        let all_but_yatzy = ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_YATZY);
        let state = YatzyState {
            upper_score: 0,
            scored_categories: all_but_yatzy,
        };

        let t0 = Instant::now();
        let mut e_ds_0 = [0.0f64; 252];
        for ds_i in 0..252 {
            e_ds_0[ds_i] =
                compute_best_scoring_value_for_dice_set(&ctx, &state, &ctx.all_dice_sets[ds_i]);
        }
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        println!("  {:<42} {:>8.3} ms", "Group 6 (inline loop)", dt);

        let t0 = Instant::now();
        let mut e_ds_1 = [0.0f64; 252];
        let mut best_masks = [0i32; 252];
        compute_expected_values_for_n_rerolls(&ctx, &e_ds_0, &mut e_ds_1, &mut best_masks);
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        println!("  {:<42} {:>8.3} ms", "Group 5 (N-rerolls, 1 level)", dt);

        let early = YatzyState {
            upper_score: 0,
            scored_categories: 1,
        };
        let t0 = Instant::now();
        let ev = compute_expected_state_value(&ctx, &early);
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  {:<42} {:>8.3} ms  (EV={:.4})",
            "Full widget solve (|C|=1)", dt, ev
        );

        let empty = YatzyState {
            upper_score: 0,
            scored_categories: 0,
        };
        let t0 = Instant::now();
        let ev0 = compute_expected_state_value(&ctx, &empty);
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  {:<42} {:>8.3} ms  (EV={:.4})",
            "Full widget solve (|C|=0)", dt, ev0
        );
    }
}
