//! SOLVE_WIDGET implementation — the DP hot path.
//!
//! Implements the pseudocode's `SOLVE_WIDGET(S)` which computes E(S) for a single
//! turn-start state by evaluating 6 groups bottom-up:
//!
//! ```text
//! Group 6 → E(S, r, 0)   : best category for each final roll
//! Group 5 → E(S, r, 1)   : best keep after 1st reroll (uses Group 6 results)
//! Group 4 → E(S, r', 1)  : expected value over 1st reroll outcomes
//! Group 3 → E(S, r, 2)   : best keep from initial roll (uses Group 4/5 results)
//! Group 2 → E(S, r', 2)  : expected value over initial roll outcomes
//! Group 1 → E(S)          : weighted sum P(⊥→r) · E(S, r, 2)
//! ```
//!
//! Groups 4+5 and 2+3 are fused into single passes. Uses ping-pong buffers `e[0]`
//! and `e[1]` (each 252 × f32) to avoid allocation.
//!
//! ## Precision
//!
//! All internal computation uses f32. State values are stored as f32, and empirical
//! testing shows f32 accumulation changes at most 0.00046 points in any state value
//! (0.0002% of typical game score), with zero impact on optimal play decisions.
//!
//! ## Performance optimizations
//!
//! - `sv` slice cached once per widget (avoids enum match per state lookup)
//! - `#[inline(always)]` on all hot functions
//! - `get_unchecked` in inner loops (indices validated by precomputation)
//! - Iterates deduplicated keep-multisets (avg 16.3 vs 31 raw masks per dice set)

use crate::constants::*;
use crate::dice_mechanics::{find_dice_set_index, sort_dice_set};
use crate::game_mechanics::update_upper_score;
use crate::types::{YatzyContext, YatzyState};

#[cfg(feature = "timing")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "timing")]
use std::time::Instant;

#[cfg(feature = "timing")]
pub static TIMING_GROUP6_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "timing")]
pub static TIMING_GROUP53_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "timing")]
pub static TIMING_GROUP1_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "timing")]
pub static TIMING_WIDGET_COUNT: AtomicU64 = AtomicU64::new(0);

/// Reset all timing counters to zero.
#[cfg(feature = "timing")]
pub fn reset_timing() {
    TIMING_GROUP6_NS.store(0, Ordering::Relaxed);
    TIMING_GROUP53_NS.store(0, Ordering::Relaxed);
    TIMING_GROUP1_NS.store(0, Ordering::Relaxed);
    TIMING_WIDGET_COUNT.store(0, Ordering::Relaxed);
}

/// Print timing breakdown and return totals.
#[cfg(feature = "timing")]
pub fn print_timing() {
    let g6 = TIMING_GROUP6_NS.load(Ordering::Relaxed) as f64 / 1e9;
    let g53 = TIMING_GROUP53_NS.load(Ordering::Relaxed) as f64 / 1e9;
    let g1 = TIMING_GROUP1_NS.load(Ordering::Relaxed) as f64 / 1e9;
    let count = TIMING_WIDGET_COUNT.load(Ordering::Relaxed);
    let total = g6 + g53 + g1;
    println!("  Widget timing ({} widgets):", count);
    println!(
        "    Group 6  (sv lookups):  {:>8.3}s ({:.1}%)",
        g6,
        g6 / total * 100.0
    );
    println!(
        "    Group 53 (dot prods):   {:>8.3}s ({:.1}%)",
        g53,
        g53 / total * 100.0
    );
    println!(
        "    Group 1  (weighted sum): {:>7.3}s ({:.1}%)",
        g1,
        g1 / total * 100.0
    );
    println!("    Total widget time:      {:>8.3}s", total);
}

/// Group 6 entry point: best category score for one roll r (public API path).
///
/// Pseudocode: E(S, r, 0) = max_{c ∉ C} [s(S,r,c) + E(n(S,r,c))]
/// Sorts and looks up the dice set index, then delegates to the by-index variant.
pub fn compute_best_scoring_value_for_dice_set(
    ctx: &YatzyContext,
    state: &YatzyState,
    dice: &[i32; 5],
) -> f64 {
    let ds_index = find_dice_set_index(ctx, dice);
    let sv = ctx.state_values.as_slice();
    compute_best_scoring_value_for_dice_set_by_index(
        ctx,
        sv,
        state.upper_score,
        state.scored_categories,
        ds_index,
    )
}

/// Group 6 hot path: best category for dice set r (by index).
///
/// Pseudocode: E(S, r, 0) = max_{c ∉ C} [s(S,r,c) + E_table[n(S,r,c)]]
///
/// Accepts a pre-extracted `sv` slice to avoid the `StateValues` enum match overhead
/// on every lookup (called ~billions of times during precomputation).
/// Split into upper (0–5) and lower (6–14) loops because upper categories affect
/// `upper_score` while lower categories leave it unchanged.
#[inline(always)]
pub fn compute_best_scoring_value_for_dice_set_by_index(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    ds_index: usize,
) -> f64 {
    let mut best_val = f32::NEG_INFINITY;

    // Upper categories (0-5): affect upper score
    for c in 0..6 {
        if !is_category_scored(scored, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            let new_up = update_upper_score(up_score, c, scr);
            let new_scored = scored | (1 << c);
            let val = scr as f32
                + unsafe { *sv.get_unchecked(state_index(new_up as usize, new_scored as usize)) };
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
            let val = scr as f32
                + unsafe { *sv.get_unchecked(state_index(up_score as usize, new_scored as usize)) };
            if val > best_val {
                best_val = val;
            }
        }
    }

    best_val as f64
}

/// Evaluate a single reroll mask: E(S, r', n) = Σ P(r'→r'') · E_prev[r''].
///
/// Pseudocode: "After keeping dice — expected value over reroll outcomes."
/// Uses sparse dot product over the keep-multiset's CSR row.
#[inline(always)]
pub fn compute_expected_value_for_reroll_mask(
    ctx: &YatzyContext,
    ds_index: usize,
    e_ds_for_masks: &[f32; 252],
    mask: i32,
) -> f64 {
    if mask == 0 {
        return e_ds_for_masks[ds_index] as f64;
    }
    let kt = &ctx.keep_table;
    let vals = &kt.vals;
    let cols = &kt.cols;
    let kid = kt.mask_to_keep[ds_index * 32 + mask as usize] as usize;
    let start = kt.row_start[kid] as usize;
    let end = kt.row_start[kid + 1] as usize;
    let mut ev: f32 = 0.0;
    for k in start..end {
        unsafe {
            ev += (*vals.get_unchecked(k) as f32)
                * e_ds_for_masks.get_unchecked(*cols.get_unchecked(k) as usize);
        }
    }
    ev as f64
}

/// Find argmax mask: best reroll decision for a specific dice set.
///
/// Pseudocode: E(S, r, n) = max_{r' ⊆ r} E(S, r', n)
/// Returns the representative mask for the best keep and writes the EV to `best_ev`.
/// Iterates only deduplicated keep-multisets (via `unique_keep_ids`).
pub fn choose_best_reroll_mask(
    ctx: &YatzyContext,
    e_ds_for_masks: &[f32; 252],
    dice: &[i32; 5],
    best_ev: &mut f64,
) -> i32 {
    let mut sorted_dice = *dice;
    sort_dice_set(&mut sorted_dice);
    let ds_index = find_dice_set_index(ctx, &sorted_dice);

    let kt = &ctx.keep_table;
    let vals = &kt.vals;
    let cols = &kt.cols;

    // mask=0: keep all
    let mut best_val = e_ds_for_masks[ds_index];
    let mut best_mask = 0i32;

    for j in 0..kt.unique_count[ds_index] as usize {
        let kid = kt.unique_keep_ids[ds_index][j] as usize;
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;
        let mut ev: f32 = 0.0;
        for k in start..end {
            unsafe {
                ev += (*vals.get_unchecked(k) as f32)
                    * e_ds_for_masks.get_unchecked(*cols.get_unchecked(k) as usize);
            }
        }
        if ev > best_val {
            best_val = ev;
            best_mask = kt.keep_to_mask[ds_index * 32 + j];
        }
    }

    *best_ev = best_val as f64;
    best_mask
}

/// Groups 5 & 3 (API path): propagate EV across one reroll level with mask tracking.
///
/// For each dice set ds, finds: E(S, ds, n) = max_{r' ⊆ ds} Σ P(r'→r'') · E_prev[r'']
/// Records both the best EV and the best mask per dice set (needed for API responses).
///
/// **Optimization**: Same keep EV dedup as the DP variant — 462 dot products instead
/// of 4,108 redundant ones.
pub fn compute_expected_values_for_n_rerolls(
    ctx: &YatzyContext,
    e_ds_prev: &[f32; 252],
    e_ds_current: &mut [f32; 252],
    best_mask_for_n: &mut [i32; 252],
) {
    let kt = &ctx.keep_table;
    let vals = &kt.vals;
    let cols = &kt.cols;

    // Step 1: compute EV for each unique keep-multiset (462 dot products)
    let mut keep_ev = [0.0f32; NUM_KEEP_MULTISETS];
    for kid in 0..NUM_KEEP_MULTISETS {
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;
        let mut ev: f32 = 0.0;
        for k in start..end {
            unsafe {
                ev += (*vals.get_unchecked(k) as f32)
                    * e_ds_prev.get_unchecked(*cols.get_unchecked(k) as usize);
            }
        }
        keep_ev[kid] = ev;
    }

    // Step 2: for each dice set, find max over its unique keeps
    for ds_i in 0..252 {
        let mut best_val = e_ds_prev[ds_i]; // mask=0: keep all
        let mut best_mask = 0i32;
        for j in 0..kt.unique_count[ds_i] as usize {
            let kid = kt.unique_keep_ids[ds_i][j] as usize;
            let ev = keep_ev[kid];
            if ev > best_val {
                best_val = ev;
                best_mask = kt.keep_to_mask[ds_i * 32 + j];
            }
        }
        e_ds_current[ds_i] = best_val;
        best_mask_for_n[ds_i] = best_mask;
    }
}

/// DP-only variant of Groups 5/3: computes E_ds_current without tracking masks.
///
/// Same logic as [`compute_expected_values_for_n_rerolls`] but omits the
/// `best_mask_for_n` output. Used in the precomputation hot path where only
/// the EV matters (strategy is not recorded).
///
/// **Optimization**: Computes each keep's EV once (462 unique dot products),
/// then distributes results to all 252 dice sets via lookup. This eliminates
/// ~8.9x redundant dot products (4,108 → 462 per call).
#[inline(always)]
fn compute_max_ev_for_n_rerolls(
    ctx: &YatzyContext,
    e_ds_prev: &[f32; 252],
    e_ds_current: &mut [f32; 252],
) {
    let kt = &ctx.keep_table;
    let vals = &kt.vals;
    let cols = &kt.cols;

    // Step 1: compute EV for each unique keep-multiset (462 dot products)
    let mut keep_ev = [0.0f32; NUM_KEEP_MULTISETS];
    for kid in 0..NUM_KEEP_MULTISETS {
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;
        let mut ev: f32 = 0.0;
        for k in start..end {
            unsafe {
                ev += (*vals.get_unchecked(k) as f32)
                    * e_ds_prev.get_unchecked(*cols.get_unchecked(k) as usize);
            }
        }
        keep_ev[kid] = ev;
    }

    // Step 2: for each dice set, find max over its unique keeps (O(1) lookup each)
    for ds_i in 0..252 {
        let mut best_val = e_ds_prev[ds_i]; // mask=0: keep all
        for j in 0..kt.unique_count[ds_i] as usize {
            let kid = unsafe { *kt.unique_keep_ids[ds_i].get_unchecked(j) } as usize;
            let ev = unsafe { *keep_ev.get_unchecked(kid) };
            if ev > best_val {
                best_val = ev;
            }
        }
        e_ds_current[ds_i] = best_val;
    }
}

/// SOLVE_WIDGET(S): compute E(S) for one turn-start state.
///
/// Pseudocode: `SOLVE_WIDGET(S)` from Phase 2a.
///
/// Evaluates the widget bottom-up using ping-pong buffers `e[0]`/`e[1]`:
///
/// 1. **Group 6** → `e[0][r]`: best category for each final roll (n=0 rerolls)
/// 2. **Group 5** → `e[1][r]`: best keep after seeing each roll (n=1 reroll left)
/// 3. **Group 3** → `e[0][r]`: best keep from initial roll (n=2 rerolls left)
/// 4. **Group 1** → `E(S)`:    weighted sum P(⊥→r) · e[0][r]
///
/// The `sv` slice is extracted once here and passed to inner functions to avoid
/// repeatedly matching the `StateValues` enum (the main optimization for this path).
pub fn compute_expected_state_value(ctx: &YatzyContext, state: &YatzyState) -> f64 {
    let mut e = [[0.0f32; 252]; 2]; // ping-pong buffers

    let up_score = state.upper_score;
    let scored = state.scored_categories;
    let sv = ctx.state_values.as_slice();

    // Group 6: E(S, r, 0) for all r
    #[cfg(feature = "timing")]
    let t0 = Instant::now();

    // Preload lower-category successor EVs: for categories 6-14, the successor
    // state value sv[state_index(up, scored|(1<<c))] is constant across all 252
    // dice sets (only depends on up and scored, not the roll). Read once here
    // instead of 252 times in the inner loop.
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

        // Upper categories (0-5): sv lookup varies per dice set (new_up depends on score)
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

        // Lower categories (6-14): use preloaded successor EV (no sv read!)
        for c in 6..CATEGORY_COUNT {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_i][c];
                let val = scr as f32 + unsafe { *lower_succ_ev.get_unchecked(c) };
                if val > best_val {
                    best_val = val;
                }
            }
        }

        e[0][ds_i] = best_val;
    }

    #[cfg(feature = "timing")]
    TIMING_GROUP6_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

    // Group 5: E(S, r, 1) = max_{mask} sum P(r'->r'') * E[0][r'']
    #[cfg(feature = "timing")]
    let t1 = Instant::now();

    let (e0, e1) = e.split_at_mut(1);
    compute_max_ev_for_n_rerolls(ctx, &e0[0], &mut e1[0]);

    // Group 3: E(S, r, 2) = max_{mask} sum P(r'->r'') * E[1][r'']
    let (e0, e1) = e.split_at_mut(1);
    compute_max_ev_for_n_rerolls(ctx, &e1[0], &mut e0[0]);

    #[cfg(feature = "timing")]
    TIMING_GROUP53_NS.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

    // Group 1: E(S) = sum P(empty->r) * E[0][r]
    #[cfg(feature = "timing")]
    let t2 = Instant::now();

    let mut e_s: f32 = 0.0;
    for ds_i in 0..252 {
        e_s += ctx.dice_set_probabilities[ds_i] as f32 * e[0][ds_i];
    }

    #[cfg(feature = "timing")]
    TIMING_GROUP1_NS.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
    #[cfg(feature = "timing")]
    TIMING_WIDGET_COUNT.fetch_add(1, Ordering::Relaxed);

    e_s as f64
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
    fn test_late_game_scoring() {
        let ctx = make_ctx();
        let all_but_yatzy = ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_YATZY);

        let state = YatzyState {
            upper_score: 0,
            scored_categories: all_but_yatzy,
        };
        let d1 = [6, 6, 6, 6, 6];
        let ev1 = compute_best_scoring_value_for_dice_set(&ctx, &state, &d1);
        assert!((ev1 - 50.0).abs() < 1e-6);

        let d2 = [1, 2, 3, 4, 5];
        let ev2 = compute_best_scoring_value_for_dice_set(&ctx, &state, &d2);
        assert!((ev2 - 0.0).abs() < 1e-6);
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
        assert!((ev1 - 30.0).abs() < 1e-6);

        let d2 = [1, 1, 1, 1, 1];
        let ev2 = compute_best_scoring_value_for_dice_set(&ctx, &state, &d2);
        assert!((ev2 - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_keep_all_mask() {
        let ctx = make_ctx();
        let all_but_yatzy = ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_YATZY);

        let mut e_ds_0 = [0.0f32; 252];
        let state = YatzyState {
            upper_score: 0,
            scored_categories: all_but_yatzy,
        };
        for ds_i in 0..252 {
            e_ds_0[ds_i] =
                compute_best_scoring_value_for_dice_set(&ctx, &state, &ctx.all_dice_sets[ds_i])
                    as f32;
        }

        let ev_keep = compute_expected_value_for_reroll_mask(&ctx, 251, &e_ds_0, 0);
        assert!((ev_keep - e_ds_0[251] as f64).abs() < 1e-6);

        let ev_keep0 = compute_expected_value_for_reroll_mask(&ctx, 0, &e_ds_0, 0);
        assert!((ev_keep0 - e_ds_0[0] as f64).abs() < 1e-6);
    }

    #[test]
    fn test_choose_best_reroll() {
        let ctx = make_ctx();
        let all_but_yatzy = ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_YATZY);

        let mut e_ds_0 = [0.0f32; 252];
        let state = YatzyState {
            upper_score: 0,
            scored_categories: all_but_yatzy,
        };
        for ds_i in 0..252 {
            e_ds_0[ds_i] =
                compute_best_scoring_value_for_dice_set(&ctx, &state, &ctx.all_dice_sets[ds_i])
                    as f32;
        }

        for ds in (0..252).step_by(50) {
            let mut best_ev = 0.0;
            let mask = choose_best_reroll_mask(&ctx, &e_ds_0, &ctx.all_dice_sets[ds], &mut best_ev);
            assert!(mask >= 0 && mask < 32);
            let ev_keep = compute_expected_value_for_reroll_mask(&ctx, ds, &e_ds_0, 0);
            assert!(best_ev >= ev_keep - 1e-6);
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
        let mut e_ds_0 = [0.0f32; 252];
        for ds_i in 0..252 {
            e_ds_0[ds_i] =
                compute_best_scoring_value_for_dice_set(&ctx, &state, &ctx.all_dice_sets[ds_i])
                    as f32;
        }
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        println!("  {:<42} {:>8.3} ms", "Group 6 (inline loop)", dt);

        let t0 = Instant::now();
        let mut e_ds_1 = [0.0f32; 252];
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
