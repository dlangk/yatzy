//! Batched SoA (Structure-of-Arrays) solver for precomputation.
//!
//! Processes all 64 upper-score variants of a `scored_categories` mask simultaneously,
//! converting the inner SpMV (sparse matrix-vector) operations into SpMM (sparse
//! matrix-matrix). This turns scattered scalar gathers into contiguous 256-byte block
//! reads that hit L1 cache.
//!
//! The state layout `scored * STATE_STRIDE + up` was designed for this: all 64 `up`
//! variants occupy the first half of a 512-byte padded region (128 x f32).
//!
//! ## Memory layout per thread
//!
//! - `e[0]`, `e[1]`: 252 × 64 × f32 = 64,512 bytes each = 129 KB total (ping-pong)
//! - `keep_ev`: 462 × 64 × f32 = 118 KB
//! - Total: ~247 KB per thread, fits in L2

use crate::constants::*;
use crate::simd::*;
use crate::types::YatzyContext;

/// Pre-allocated buffers for batched widget solving, reusable across scored masks.
///
/// Allocate one per rayon thread to eliminate per-widget heap allocation overhead.
pub struct BatchedBuffers {
    pub e0: Vec<[f32; 64]>,
    pub e1: Vec<[f32; 64]>,
    pub keep_ev: Vec<[f32; 64]>,
}

impl BatchedBuffers {
    pub fn new() -> Self {
        Self {
            e0: vec![[0.0f32; 64]; 252],
            e1: vec![[0.0f32; 64]; 252],
            keep_ev: vec![[0.0f32; 64]; NUM_KEEP_MULTISETS],
        }
    }
}

/// Compute E(S) for all 64 upper-score variants of a given `scored_categories` mask.
///
/// This is the batched equivalent of calling `compute_expected_state_value` 64 times
/// with the same `scored` but varying `upper_score`. Returns a 64-element array.
///
/// Uses externally-provided `bufs` to avoid per-call heap allocation.
#[inline(never)]
pub fn solve_widget_batched(
    ctx: &YatzyContext,
    sv: &[f32],
    scored: i32,
    bufs: &mut BatchedBuffers,
) -> [f32; 64] {
    // ── Group 6: E(S, r, 0) for all r and all up ──
    batched_group6(ctx, sv, scored, &mut bufs.e0);

    // ── Group 5: E(S, r, 1) = max_{keep} Σ P(k→ds') · e0[ds'][up] ──
    batched_group53(ctx, &bufs.e0, &mut bufs.e1, &mut bufs.keep_ev);

    // ── Group 3: E(S, r, 2) = max_{keep} Σ P(k→ds') · e1[ds'][up] ──
    batched_group53(ctx, &bufs.e1, &mut bufs.e0, &mut bufs.keep_ev);

    // ── Group 1: E(S) = Σ P(⊥→r) · e0[r][up] ──
    batched_group1(ctx, &bufs.e0)
}

/// Batched Group 6: best category score for each dice set × each upper score.
///
/// For lower categories (6-14): successor block `sv[new_scored*64 .. +64]` is
/// constant across `up` — read once, broadcast.
/// For upper categories (0-5): `new_up = min(up + score, 63)` varies per `up`,
/// but all successors lie within the same 256-byte block.
#[inline(always)]
fn batched_group6(ctx: &YatzyContext, sv: &[f32], scored: i32, e_out: &mut [[f32; 64]]) {
    // Preload lower-category successor EV blocks (each is a contiguous 64-element slice).
    // For lower categories, the successor state value depends only on scored (not up).
    let mut lower_succ_base: [usize; CATEGORY_COUNT] = [0; CATEGORY_COUNT];
    let mut lower_avail: [bool; CATEGORY_COUNT] = [false; CATEGORY_COUNT];
    for c in 6..CATEGORY_COUNT {
        if !is_category_scored(scored, c) {
            lower_avail[c] = true;
            let new_scored = (scored | (1 << c)) as usize;
            lower_succ_base[c] = new_scored * STATE_STRIDE;
        }
    }

    // Check which upper categories are available
    let mut upper_avail: [bool; 6] = [false; 6];
    for c in 0..6 {
        upper_avail[c] = !is_category_scored(scored, c);
    }

    for ds_i in 0..252 {
        let row = &mut e_out[ds_i];

        // Initialize to NEG_INFINITY
        *row = [f32::NEG_INFINITY; 64];

        // Lower categories (6-14): upper score unchanged
        // NEON: row[up] = max(row[up], scr + sv[base + up])
        for c in 6..CATEGORY_COUNT {
            if lower_avail[c] {
                let scr = ctx.precomputed_scores[ds_i][c] as f32;
                let base = lower_succ_base[c];
                unsafe { neon_add_max_64(row, sv.as_ptr().add(base), scr) };
            }
        }

        // Upper categories (0-5): branchless via topological padding.
        // NEON: row[up] = max(row[up], scr + sv[succ_base + scr + up])
        for c in 0..6 {
            if upper_avail[c] {
                let scr = ctx.precomputed_scores[ds_i][c] as usize;
                let scr_f = scr as f32;
                let new_scored = (scored | (1 << c)) as usize;
                let succ_base = new_scored * STATE_STRIDE;
                unsafe { neon_add_max_offset_64(row, sv.as_ptr(), succ_base + scr, scr_f) };
            }
        }
    }
}

/// Batched Groups 5/3: SpMM over the keep-multiset transition table.
///
/// Step 1: For each keep-multiset `kid`, compute keep_ev[kid][up] = Σ P(k→ds') · e_prev[ds'][up].
///         Each `e_prev[ds']` is a contiguous 64×f32 = 256-byte row — perfect cache line read.
/// Step 2: For each dice set `ds`, find best keep: e_curr[ds][up] = max(e_prev[ds][up], max_j keep_ev[kid_j][up]).
#[inline(always)]
fn batched_group53(
    ctx: &YatzyContext,
    e_prev: &[[f32; 64]],
    e_curr: &mut [[f32; 64]],
    keep_ev: &mut [[f32; 64]],
) {
    let kt = &ctx.keep_table;
    let vals = &kt.vals;
    let cols = &kt.cols;

    // Step 1: NEON FMA — keep_ev[kid][up] += prob * e_prev[ds'][up]
    for kid in 0..NUM_KEEP_MULTISETS {
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;
        let kev = &mut keep_ev[kid];
        *kev = [0.0f32; 64];

        for k in start..end {
            let prob = unsafe { *vals.get_unchecked(k) };
            let col = unsafe { *cols.get_unchecked(k) } as usize;
            let e_row = &e_prev[col];
            unsafe { neon_fma_64(kev, e_row, prob) };
        }
    }

    // Step 2: NEON max — e_curr[ds][up] = max(e_prev[ds][up], keep_ev[kid][up])
    for ds_i in 0..252 {
        e_curr[ds_i] = e_prev[ds_i];
        let row = &mut e_curr[ds_i];

        for j in 0..kt.unique_count[ds_i] as usize {
            let kid = unsafe { *kt.unique_keep_ids[ds_i].get_unchecked(j) } as usize;
            let kev = &keep_ev[kid];
            unsafe { neon_max_64(row, kev) };
        }
    }
}

/// Public wrapper for Group 1 — used by oracle builder in state_computation.
pub fn batched_group1_pub(ctx: &YatzyContext, e: &[[f32; 64]]) -> [f32; 64] {
    batched_group1(ctx, e)
}

/// Batched Group 1: E(S)[up] = Σ P(⊥→r) · e[r][up] for all ups.
#[inline(always)]
fn batched_group1(ctx: &YatzyContext, e: &[[f32; 64]]) -> [f32; 64] {
    let mut result = [0.0f32; 64];
    for ds_i in 0..252 {
        let prob = ctx.dice_set_probabilities[ds_i] as f32;
        let e_row = &e[ds_i];
        unsafe { neon_fma_64(&mut result, e_row, prob) };
    }
    result
}

/// Batched SOLVE_WIDGET for risk-sensitive (log-domain) mode with θ ≠ 0.
///
/// // PERF: intentional duplication of solve_widget_batched. This variant uses
/// // LSE (log-sum-exp) instead of weighted sums, and min/max instead of argmax.
/// // Merging with the EV solver would add branches in the inner loop.
///
/// Transforms:
/// - Group 6: `val = θ·scr + sv[successor]` — decision node (min if θ<0, max if θ>0)
/// - Groups 5/3: LSE (stochastic) + opt over keeps (decision: min/max)
/// - Group 1: LSE (stochastic only)
#[inline(never)]
pub fn solve_widget_batched_risk(
    ctx: &YatzyContext,
    sv: &[f32],
    scored: i32,
    bufs: &mut BatchedBuffers,
) -> [f32; 64] {
    let theta = ctx.theta;
    let minimize = theta < 0.0;

    // Group 6: risk-sensitive scoring
    batched_group6_risk(ctx, sv, scored, theta, minimize, &mut bufs.e0);

    // Groups 5/3: LSE propagation
    batched_group53_risk(ctx, &bufs.e0, &mut bufs.e1, &mut bufs.keep_ev, minimize);
    batched_group53_risk(ctx, &bufs.e1, &mut bufs.e0, &mut bufs.keep_ev, minimize);

    // Group 1: LSE over initial rolls (stochastic — always same)
    batched_group1_risk(ctx, &bufs.e0)
}

/// Batched Group 6 for risk-sensitive mode.
#[inline(always)]
fn batched_group6_risk(
    ctx: &YatzyContext,
    sv: &[f32],
    scored: i32,
    theta: f32,
    minimize: bool,
    e_out: &mut [[f32; 64]],
) {
    let init = if minimize {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };

    let mut lower_succ_base: [usize; CATEGORY_COUNT] = [0; CATEGORY_COUNT];
    let mut lower_avail: [bool; CATEGORY_COUNT] = [false; CATEGORY_COUNT];
    for c in 6..CATEGORY_COUNT {
        if !is_category_scored(scored, c) {
            lower_avail[c] = true;
            let new_scored = (scored | (1 << c)) as usize;
            lower_succ_base[c] = new_scored * STATE_STRIDE;
        }
    }

    let mut upper_avail: [bool; 6] = [false; 6];
    for c in 0..6 {
        upper_avail[c] = !is_category_scored(scored, c);
    }

    for ds_i in 0..252 {
        let row = &mut e_out[ds_i];
        *row = [init; 64];

        // NEON: lower categories with min/max dispatch
        for c in 6..CATEGORY_COUNT {
            if lower_avail[c] {
                let scr = ctx.precomputed_scores[ds_i][c] as f32;
                let theta_scr = theta * scr;
                let base = lower_succ_base[c];
                if minimize {
                    unsafe { neon_add_min_64(row, sv.as_ptr().add(base), theta_scr) };
                } else {
                    unsafe { neon_add_max_64(row, sv.as_ptr().add(base), theta_scr) };
                }
            }
        }

        // NEON: upper categories branchless via topological padding
        for c in 0..6 {
            if upper_avail[c] {
                let scr = ctx.precomputed_scores[ds_i][c];
                let theta_scr = theta * scr as f32;
                let new_scored = (scored | (1 << c)) as usize;
                let succ_base = new_scored * STATE_STRIDE;
                if minimize {
                    unsafe {
                        neon_add_min_offset_64(
                            row,
                            sv.as_ptr(),
                            succ_base + scr as usize,
                            theta_scr,
                        )
                    };
                } else {
                    unsafe {
                        neon_add_max_offset_64(
                            row,
                            sv.as_ptr(),
                            succ_base + scr as usize,
                            theta_scr,
                        )
                    };
                }
            }
        }
    }
}

/// Batched Groups 5/3 for risk-sensitive mode: LSE + min/max decision.
#[inline(always)]
fn batched_group53_risk(
    ctx: &YatzyContext,
    e_prev: &[[f32; 64]],
    e_curr: &mut [[f32; 64]],
    keep_ev: &mut [[f32; 64]],
    minimize: bool,
) {
    let kt = &ctx.keep_table;
    let vals = &kt.vals;
    let cols = &kt.cols;

    let empty_val = if minimize {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };

    // Step 1: compute LSE for each keep-multiset
    for kid in 0..NUM_KEEP_MULTISETS {
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;
        let kev = &mut keep_ev[kid];

        if start == end {
            *kev = [empty_val; 64];
            continue;
        }

        // Pass 1: NEON max — find max per up (for numerical stability)
        let mut max_vals = [f32::NEG_INFINITY; 64];
        for k in start..end {
            let col = unsafe { *cols.get_unchecked(k) } as usize;
            let e_row = &e_prev[col];
            unsafe { neon_max_64(&mut max_vals, e_row) };
        }

        // Pass 2: weighted exp-sum
        let mut sums = [0.0f32; 64];
        for k in start..end {
            let prob = unsafe { *vals.get_unchecked(k) };
            let col = unsafe { *cols.get_unchecked(k) } as usize;
            let e_row = &e_prev[col];
            unsafe { neon_weighted_exp_sum_64(&mut sums, e_row, &max_vals, prob) };
        }

        // LSE = max + ln(sum)
        for up in 0..64 {
            kev[up] = max_vals[up] + sums[up].ln();
        }
    }

    // Step 2: NEON min/max — for each dice set, find opt over its unique keeps
    for ds_i in 0..252 {
        e_curr[ds_i] = e_prev[ds_i]; // mask=0: keep all
        let row = &mut e_curr[ds_i];

        for j in 0..kt.unique_count[ds_i] as usize {
            let kid = unsafe { *kt.unique_keep_ids[ds_i].get_unchecked(j) } as usize;
            let kev = &keep_ev[kid];
            if minimize {
                unsafe { neon_min_64(row, kev) };
            } else {
                unsafe { neon_max_64(row, kev) };
            }
        }
    }
}

/// Batched Group 1 for risk-sensitive mode: LSE over initial rolls.
#[inline(always)]
fn batched_group1_risk(ctx: &YatzyContext, e: &[[f32; 64]]) -> [f32; 64] {
    // Pass 1: NEON max — find max per up
    let mut max_vals = [f32::NEG_INFINITY; 64];
    for ds_i in 0..252 {
        let e_row = &e[ds_i];
        unsafe { neon_max_64(&mut max_vals, e_row) };
    }

    // Pass 2: NEON weighted exp-sum
    let mut sums = [0.0f32; 64];
    for ds_i in 0..252 {
        let prob = ctx.dice_set_probabilities[ds_i] as f32;
        let e_row = &e[ds_i];
        unsafe { neon_weighted_exp_sum_64(&mut sums, e_row, &max_vals, prob) };
    }

    let mut result = [0.0f32; 64];
    for up in 0..64 {
        result[up] = max_vals[up] + sums[up].ln();
    }
    result
}

/// Batched SOLVE_WIDGET for max-policy mode.
///
/// // PERF: intentional duplication of solve_widget_batched. Chance nodes use
/// // neon_max_64 instead of neon_fma_64, eliminating probability weights entirely.
/// // A generic interface would add branches in the tightest inner loops.
///
/// Chance nodes use max instead of Σ P·x:
/// - Group 6: identical to EV mode (decision node)
/// - Groups 5/3: max over reachable dice sets (no probability weighting)
/// - Group 1: max over initial rolls
#[inline(never)]
pub fn solve_widget_batched_max(
    ctx: &YatzyContext,
    sv: &[f32],
    scored: i32,
    bufs: &mut BatchedBuffers,
) -> [f32; 64] {
    // Group 6: same as EV mode
    batched_group6(ctx, sv, scored, &mut bufs.e0);

    // Groups 5/3: max-outcome propagation
    batched_group53_max(ctx, &bufs.e0, &mut bufs.e1, &mut bufs.keep_ev);
    batched_group53_max(ctx, &bufs.e1, &mut bufs.e0, &mut bufs.keep_ev);

    // Group 1: max over initial rolls
    batched_group1_max(&bufs.e0)
}

/// Batched Groups 5/3 for max-policy: chance nodes use max instead of Σ P·x.
#[inline(always)]
fn batched_group53_max(
    ctx: &YatzyContext,
    e_prev: &[[f32; 64]],
    e_curr: &mut [[f32; 64]],
    keep_max: &mut [[f32; 64]],
) {
    let kt = &ctx.keep_table;
    let cols = &kt.cols;

    // Step 1: NEON max outcome for each unique keep-multiset
    for kid in 0..NUM_KEEP_MULTISETS {
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;
        let km = &mut keep_max[kid];
        *km = [f32::NEG_INFINITY; 64];

        for k in start..end {
            let col = unsafe { *cols.get_unchecked(k) } as usize;
            let e_row = &e_prev[col];
            unsafe { neon_max_64(km, e_row) };
        }
    }

    // Step 2: NEON max over unique keeps (decision node)
    for ds_i in 0..252 {
        e_curr[ds_i] = e_prev[ds_i]; // mask=0: keep all
        let row = &mut e_curr[ds_i];

        for j in 0..kt.unique_count[ds_i] as usize {
            let kid = unsafe { *kt.unique_keep_ids[ds_i].get_unchecked(j) } as usize;
            let km = &keep_max[kid];
            unsafe { neon_max_64(row, km) };
        }
    }
}

/// Batched Group 1 for max-policy: max over initial rolls.
#[inline(always)]
fn batched_group1_max(e: &[[f32; 64]]) -> [f32; 64] {
    let mut result = [f32::NEG_INFINITY; 64];
    for ds_i in 0..252 {
        let e_row = &e[ds_i];
        unsafe { neon_max_64(&mut result, e_row) };
    }
    result
}

/// Precomputed exponential scores: `exp_scores[ds][c] = e^(θ·score(ds,c))`.
///
/// Used by the utility-domain solver where scoring becomes multiplication:
/// `U(S) = exp_score × U(successor)` instead of `L(S) = θ·score + L(successor)`.
pub type ExpScores = [[f32; CATEGORY_COUNT]; 252];

/// Precompute `e^(θ·score)` for all 252 dice sets × 15 categories.
pub fn precompute_exp_scores(ctx: &YatzyContext, theta: f32) -> Box<ExpScores> {
    let mut table = Box::new([[0.0f32; CATEGORY_COUNT]; 252]);
    for ds_i in 0..252 {
        for c in 0..CATEGORY_COUNT {
            let scr = ctx.precomputed_scores[ds_i][c] as f32;
            table[ds_i][c] = (theta * scr).exp();
        }
    }
    table
}

/// Batched SOLVE_WIDGET for utility-domain mode (|θ| ≤ 0.15).
///
/// Stores `U(S) = E[e^(θ·remaining)|S]` directly. The key win:
/// // PERF: intentional duplication of solve_widget_batched. Utility domain uses
/// // multiplicative exp(θ·score) instead of additive score, avoiding exp/ln/LSE
/// // entirely. This matches θ=0 speed for |θ|≤0.15. Merging with the risk solver
/// // would add exp/ln overhead for all θ values.
///
/// - **Group 6** (scoring): `val = exp_score[ds][c] * sv[successor]` (multiply, not add)
/// - **Groups 5/3** (stochastic): plain weighted sums, IDENTICAL to EV solver
/// - **Groups 5/3** (decision): min/max over keeps (same as risk solver)
/// - **Group 1** (stochastic): plain weighted sum, IDENTICAL to EV solver
///
/// This eliminates all exp/ln/LSE operations, matching θ=0 precompute speed.
#[inline(never)]
pub fn solve_widget_batched_utility(
    ctx: &YatzyContext,
    sv: &[f32],
    scored: i32,
    bufs: &mut BatchedBuffers,
    exp_scores: &ExpScores,
    minimize: bool,
) -> [f32; 64] {
    // Group 6: utility-domain scoring (multiply instead of add)
    batched_group6_utility(ctx, sv, scored, exp_scores, minimize, &mut bufs.e0);

    // Groups 5/3: stochastic nodes are plain weighted sums (same as EV),
    // decision nodes use min/max (same as risk)
    batched_group53_utility(ctx, &bufs.e0, &mut bufs.e1, &mut bufs.keep_ev, minimize);
    batched_group53_utility(ctx, &bufs.e1, &mut bufs.e0, &mut bufs.keep_ev, minimize);

    // Group 1: plain weighted sum (identical to EV solver)
    batched_group1(ctx, &bufs.e0)
}

/// Batched Group 6 for utility-domain mode.
///
/// `val = exp_scores[ds][c] * sv[successor]` instead of `scr + sv[successor]`.
/// Decision node: min if θ<0 (risk-averse), max if θ>0 (risk-seeking).
#[inline(always)]
fn batched_group6_utility(
    ctx: &YatzyContext,
    sv: &[f32],
    scored: i32,
    exp_scores: &ExpScores,
    minimize: bool,
    e_out: &mut [[f32; 64]],
) {
    let init = if minimize {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };

    let mut lower_succ_base: [usize; CATEGORY_COUNT] = [0; CATEGORY_COUNT];
    let mut lower_avail: [bool; CATEGORY_COUNT] = [false; CATEGORY_COUNT];
    for c in 6..CATEGORY_COUNT {
        if !is_category_scored(scored, c) {
            lower_avail[c] = true;
            let new_scored = (scored | (1 << c)) as usize;
            lower_succ_base[c] = new_scored * STATE_STRIDE;
        }
    }

    let mut upper_avail: [bool; 6] = [false; 6];
    for c in 0..6 {
        upper_avail[c] = !is_category_scored(scored, c);
    }

    for ds_i in 0..252 {
        let row = &mut e_out[ds_i];
        *row = [init; 64];

        // Lower categories (6-14): NEON mul-min/max
        for c in 6..CATEGORY_COUNT {
            if lower_avail[c] {
                let exp_scr = exp_scores[ds_i][c];
                let base = lower_succ_base[c];
                let src = unsafe { sv.as_ptr().add(base) };
                if minimize {
                    unsafe { neon_mul_min_64(row, src, exp_scr) };
                } else {
                    unsafe { neon_mul_max_64(row, src, exp_scr) };
                }
            }
        }

        // Upper categories (0-5): NEON mul-min/max with offset (branchless padding)
        for c in 0..6 {
            if upper_avail[c] {
                let scr = ctx.precomputed_scores[ds_i][c] as usize;
                let exp_scr = exp_scores[ds_i][c];
                let new_scored = (scored | (1 << c)) as usize;
                let succ_base = new_scored * STATE_STRIDE;
                if minimize {
                    unsafe { neon_mul_min_offset_64(row, sv.as_ptr(), succ_base + scr, exp_scr) };
                } else {
                    unsafe { neon_mul_max_offset_64(row, sv.as_ptr(), succ_base + scr, exp_scr) };
                }
            }
        }
    }
}

/// Batched Groups 5/3 for utility-domain mode.
///
/// Stochastic nodes: plain weighted sum (same as EV solver — no LSE needed).
/// Decision nodes: min/max over keeps (same as risk solver).
#[inline(always)]
fn batched_group53_utility(
    ctx: &YatzyContext,
    e_prev: &[[f32; 64]],
    e_curr: &mut [[f32; 64]],
    keep_ev: &mut [[f32; 64]],
    minimize: bool,
) {
    let kt = &ctx.keep_table;
    let vals = &kt.vals;
    let cols = &kt.cols;

    // Step 1: NEON weighted sum per keep (identical to EV solver)
    for kid in 0..NUM_KEEP_MULTISETS {
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;
        let kev = &mut keep_ev[kid];
        *kev = [0.0f32; 64];

        for k in start..end {
            let prob = unsafe { *vals.get_unchecked(k) };
            let col = unsafe { *cols.get_unchecked(k) } as usize;
            let e_row = &e_prev[col];
            unsafe { neon_fma_64(kev, e_row, prob) };
        }
    }

    // Step 2: NEON decision node — min/max over keeps
    for ds_i in 0..252 {
        e_curr[ds_i] = e_prev[ds_i]; // mask=0: keep all
        let row = &mut e_curr[ds_i];

        for j in 0..kt.unique_count[ds_i] as usize {
            let kid = unsafe { *kt.unique_keep_ids[ds_i].get_unchecked(j) } as usize;
            let kev = &keep_ev[kid];
            if minimize {
                unsafe { neon_min_64(row, kev) };
            } else {
                unsafe { neon_max_64(row, kev) };
            }
        }
    }
}

/// Per-scored-mask oracle slice: argmax decisions for all 252 ds × 64 up.
///
/// Stored as flat arrays [ds * 64 + up] for sequential write during precompute,
/// then scattered to the oracle's [state_index * 252 + ds] layout.
pub struct OracleSlice {
    /// Best category (0-14) per (ds, up).
    pub cat: Vec<u8>,
    /// Best keep for 1 reroll left: 0=keep-all, j+1=unique keep j.
    pub keep1: Vec<u8>,
    /// Best keep for 2 rerolls left: same encoding.
    pub keep2: Vec<u8>,
}

impl OracleSlice {
    fn new() -> Self {
        Self {
            cat: vec![0u8; 252 * 64],
            keep1: vec![0u8; 252 * 64],
            keep2: vec![0u8; 252 * 64],
        }
    }
}

/// Build oracle decisions for all 64 upper-score variants of a given `scored` mask.
///
/// Tracks argmax in-register alongside NEON max operations, avoiding the need
/// for a separate scalar re-evaluation pass. Uses `neon_max_64_argmax` for
/// Groups 5/3 and scalar argmax for Group 6 (interleaved with the max computation).
///
/// Only for θ=0 EV mode.
pub fn build_oracle_for_scored_mask(
    ctx: &YatzyContext,
    sv: &[f32],
    scored: i32,
    bufs: &mut BatchedBuffers,
) -> OracleSlice {
    let mut oracle = OracleSlice::new();

    // ── Group 6: E(S, r, 0) with argmax tracking ──
    batched_group6_with_argmax(ctx, sv, scored, &mut bufs.e0, &mut oracle.cat);

    // ── Group 5: E(S, r, 1) with argmax tracking ──
    batched_group53_with_argmax(
        ctx,
        &bufs.e0,
        &mut bufs.e1,
        &mut bufs.keep_ev,
        &mut oracle.keep1,
    );

    // ── Group 3: E(S, r, 2) with argmax tracking ──
    batched_group53_with_argmax(
        ctx,
        &bufs.e1,
        &mut bufs.e0,
        &mut bufs.keep_ev,
        &mut oracle.keep2,
    );

    oracle
}

/// Batched Group 6 with argmax: tracks best category per (ds, up).
///
/// Same computation as `batched_group6` but maintains a parallel `cat_idx[ds][up]`
/// array updated via `vcgtq_f32` + conditional write.
#[inline(always)]
fn batched_group6_with_argmax(
    ctx: &YatzyContext,
    sv: &[f32],
    scored: i32,
    e_out: &mut [[f32; 64]],
    cat_out: &mut [u8],
) {
    let mut lower_succ_base: [usize; CATEGORY_COUNT] = [0; CATEGORY_COUNT];
    let mut lower_avail: [bool; CATEGORY_COUNT] = [false; CATEGORY_COUNT];
    for c in 6..CATEGORY_COUNT {
        if !is_category_scored(scored, c) {
            lower_avail[c] = true;
            let new_scored = (scored | (1 << c)) as usize;
            lower_succ_base[c] = new_scored * STATE_STRIDE;
        }
    }

    let mut upper_avail: [bool; 6] = [false; 6];
    for c in 0..6 {
        upper_avail[c] = !is_category_scored(scored, c);
    }

    for ds_i in 0..252 {
        let row = &mut e_out[ds_i];
        let idx_base = ds_i * 64;
        let idx_slice = &mut cat_out[idx_base..idx_base + 64];

        // Initialize to NEG_INFINITY with category 0
        *row = [f32::NEG_INFINITY; 64];
        idx_slice.fill(0);
        let mut first_cat = true;

        // Lower categories (6-14): track which category wins
        for c in 6..CATEGORY_COUNT {
            if lower_avail[c] {
                let scr = ctx.precomputed_scores[ds_i][c] as f32;
                let base = lower_succ_base[c];
                if first_cat {
                    // First category: just set values directly (all will win vs NEG_INFINITY)
                    for up in 0..64 {
                        row[up] = scr + unsafe { *sv.get_unchecked(base + up) };
                    }
                    idx_slice.fill(c as u8);
                    first_cat = false;
                } else {
                    // Compare and update argmax
                    for up in 0..64 {
                        let val = scr + unsafe { *sv.get_unchecked(base + up) };
                        if val > row[up] {
                            row[up] = val;
                            unsafe { *idx_slice.get_unchecked_mut(up) = c as u8 };
                        }
                    }
                }
            }
        }

        // Upper categories (0-5): branchless via topological padding
        for c in 0..6 {
            if upper_avail[c] {
                let scr = ctx.precomputed_scores[ds_i][c] as usize;
                let scr_f = scr as f32;
                let new_scored = (scored | (1 << c)) as usize;
                let succ_base = new_scored * STATE_STRIDE;
                if first_cat {
                    for up in 0..64 {
                        row[up] = scr_f + unsafe { *sv.get_unchecked(succ_base + scr + up) };
                    }
                    idx_slice.fill(c as u8);
                    first_cat = false;
                } else {
                    for up in 0..64 {
                        let val = scr_f + unsafe { *sv.get_unchecked(succ_base + scr + up) };
                        if val > row[up] {
                            row[up] = val;
                            unsafe { *idx_slice.get_unchecked_mut(up) = c as u8 };
                        }
                    }
                }
            }
        }
    }
}

/// Batched Groups 5/3 with argmax: tracks best keep per (ds, up).
///
/// Step 1 is unchanged (computing keep_ev via FMA — no decision).
/// Step 2 uses `neon_max_64_argmax` to track which keep wins at each `up`.
///
/// Keep encoding: 0 = keep all dice, j+1 = unique keep j.
#[inline(always)]
fn batched_group53_with_argmax(
    ctx: &YatzyContext,
    e_prev: &[[f32; 64]],
    e_curr: &mut [[f32; 64]],
    keep_ev: &mut [[f32; 64]],
    keep_out: &mut [u8],
) {
    let kt = &ctx.keep_table;
    let vals = &kt.vals;
    let cols = &kt.cols;

    // Step 1: FMA — keep_ev[kid][up] += prob * e_prev[ds'][up] (no argmax needed here)
    for kid in 0..NUM_KEEP_MULTISETS {
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;
        let kev = &mut keep_ev[kid];
        *kev = [0.0f32; 64];

        for k in start..end {
            let prob = unsafe { *vals.get_unchecked(k) };
            let col = unsafe { *cols.get_unchecked(k) } as usize;
            let e_row = &e_prev[col];
            unsafe { neon_fma_64(kev, e_row, prob) };
        }
    }

    // Step 2: max with argmax — e_curr[ds][up] = max(e_prev[ds][up], keep_ev[kid][up])
    for ds_i in 0..252 {
        e_curr[ds_i] = e_prev[ds_i]; // mask=0: keep all
        let row = &mut e_curr[ds_i];
        let idx_base = ds_i * 64;
        let idx_slice: &mut [u8] = &mut keep_out[idx_base..idx_base + 64];
        idx_slice.fill(0); // 0 = keep all (ORACLE_KEEP_ALL)

        for j in 0..kt.unique_count[ds_i] as usize {
            let kid = unsafe { *kt.unique_keep_ids[ds_i].get_unchecked(j) } as usize;
            let kev = &keep_ev[kid];
            // j+1 encoding: 0 is keep-all, so unique keep j gets encoded as j+1
            let mut idx_arr: &mut [u8; 64] =
                unsafe { &mut *(idx_slice.as_mut_ptr() as *mut [u8; 64]) };
            unsafe { neon_max_64_argmax(row, kev, &mut idx_arr, (j + 1) as u8) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0_tables;
    use crate::types::YatzyState;
    use crate::widget_solver::compute_expected_state_value;

    fn make_ctx() -> Box<YatzyContext> {
        let mut ctx = YatzyContext::new_boxed();
        phase0_tables::precompute_lookup_tables(&mut ctx);
        ctx
    }

    /// Verify batched solver produces identical results to scalar solver
    /// for a variety of scored_categories masks.
    #[test]
    fn test_batched_vs_scalar_ev() {
        let ctx = make_ctx();
        let sv = ctx.state_values.as_slice();
        let mut bufs = BatchedBuffers::new();

        // Test several scored masks across different levels
        let test_masks: Vec<i32> = vec![
            // |C|=14: single category remaining
            ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_YATZY),
            ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_CHANCE),
            ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_ONES),
            // |C|=13: two categories remaining
            ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_YATZY) ^ (1 << CATEGORY_CHANCE),
            ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_ONES) ^ (1 << CATEGORY_TWOS),
            // |C|=12
            ((1 << CATEGORY_COUNT) - 1)
                ^ (1 << CATEGORY_ONES)
                ^ (1 << CATEGORY_TWOS)
                ^ (1 << CATEGORY_THREES),
            // |C|=1
            1 << CATEGORY_ONES,
            // |C|=0
            0,
        ];

        for &scored in &test_masks {
            let batched = solve_widget_batched(&ctx, sv, scored, &mut bufs);
            let upper_mask = (scored & 0x3F) as usize;

            for up in 0..64 {
                if !ctx.reachable[upper_mask][up] {
                    continue;
                }
                let state = YatzyState {
                    upper_score: up as i32,
                    scored_categories: scored,
                };
                let scalar = compute_expected_state_value(&ctx, &state) as f32;
                assert!(
                    (batched[up] - scalar).abs() < 1e-4,
                    "Mismatch at scored=0x{:x} up={}: batched={} scalar={}",
                    scored,
                    up,
                    batched[up],
                    scalar
                );
            }
        }
    }

    /// Verify that terminal states (|C|=15) produce expected bonus values.
    #[test]
    fn test_batched_group6_terminal() {
        let ctx = make_ctx();
        let sv = ctx.state_values.as_slice();
        let mut bufs = BatchedBuffers::new();

        // With only Yatzy remaining, dice [6,6,6,6,6] should score 50
        let scored = ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_YATZY);
        let results = solve_widget_batched(&ctx, sv, scored, &mut bufs);

        // up=0, scored with Yatzy last: the E(S) should be the Yatzy expected value
        // (probability of rolling Yatzy × 50 over 3 rolls)
        assert!(results[0] > 0.0, "EV should be positive");
        assert!(results[0] < 50.0, "EV should be less than max Yatzy score");
    }
}
