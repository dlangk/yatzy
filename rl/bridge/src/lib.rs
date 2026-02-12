//! FFI bridge: exposes batch Yatzy turn simulation for Python RL training.
//!
//! Loaded via ctypes from Python. Provides:
//! - `rl_bridge_init`: load theta tables and build context
//! - `rl_bridge_free`: deallocate context
//! - `rl_bridge_batch_turn`: simulate one turn for N games in parallel
//! - `rl_bridge_batch_game`: simulate full games for N seeds (for evaluation)

use std::ffi::CStr;
use std::os::raw::c_char;
use std::slice;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use yatzy::constants::*;
use yatzy::dice_mechanics::{find_dice_set_index, sort_dice_set};
use yatzy::game_mechanics::update_upper_score;
use yatzy::phase0_tables;
use yatzy::storage::{load_state_values_standalone, state_file_path};
use yatzy::types::{StateValues, YatzyContext};
use yatzy::widget_solver::{
    choose_best_reroll_mask, choose_best_reroll_mask_risk, compute_max_ev_for_n_rerolls,
    compute_opt_lse_for_n_rerolls,
};

/// Opaque context holding precomputed tables and loaded theta state values.
pub struct BridgeContext {
    /// Shared precomputed tables (dice sets, scores, keep table, etc.)
    ctx: Box<YatzyContext>,
    /// State values for each theta: (theta, StateValues)
    theta_tables: Vec<(f32, StateValues)>,
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

/// Apply a reroll mask: bits set indicate positions to reroll.
#[inline(always)]
fn apply_reroll(dice: &mut [i32; 5], mask: i32, rng: &mut SmallRng) {
    for i in 0..5 {
        if mask & (1 << i) != 0 {
            dice[i] = rng.random_range(1..=6);
        }
    }
    sort_dice_set(dice);
}

/// Group 6: best category EV for each dice set (EV mode, θ=0).
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

/// Group 6 risk-sensitive: val = θ·scr + sv[successor].
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
                let better = if minimize { val < best_val } else { val > best_val };
                if better {
                    best_val = val;
                }
            }
        }
        for c in 6..CATEGORY_COUNT {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_i][c];
                let val = theta * scr as f32 + unsafe { *lower_succ_ev.get_unchecked(c) };
                let better = if minimize { val < best_val } else { val > best_val };
                if better {
                    best_val = val;
                }
            }
        }
        e_ds_0[ds_i] = best_val;
    }
}

/// Find the best category to score.
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
                + unsafe {
                    *sv.get_unchecked(state_index(up_score as usize, new_scored as usize))
                };
            if val > best_val {
                best_val = val;
                best_cat = c;
                best_score = scr;
            }
        }
    }
    (best_cat, best_score)
}

/// Find best category for the last turn (no successor state).
fn find_best_category_final(
    ctx: &YatzyContext,
    up_score: i32,
    scored: i32,
    ds_index: usize,
) -> (usize, i32) {
    let mut best_val = i32::MIN;
    let mut best_cat = 0;
    let mut best_score = 0;
    for c in 0..CATEGORY_COUNT {
        if !is_category_scored(scored, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            let bonus = if c < 6 {
                let new_up = update_upper_score(up_score, c, scr);
                if new_up >= 63 && up_score < 63 { 50 } else { 0 }
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

/// Find best category in risk-sensitive mode.
fn find_best_category_risk(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    ds_index: usize,
    theta: f32,
    minimize: bool,
) -> (usize, i32) {
    let mut best_val = if minimize { f32::INFINITY } else { f32::NEG_INFINITY };
    let mut best_cat = 0;
    let mut best_score = 0;

    for c in 0..6 {
        if !is_category_scored(scored, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            let new_up = update_upper_score(up_score, c, scr);
            let new_scored = scored | (1 << c);
            let val = theta * scr as f32
                + unsafe { *sv.get_unchecked(state_index(new_up as usize, new_scored as usize)) };
            let better = if minimize { val < best_val } else { val > best_val };
            if better { best_val = val; best_cat = c; best_score = scr; }
        }
    }
    for c in 6..CATEGORY_COUNT {
        if !is_category_scored(scored, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            let new_scored = scored | (1 << c);
            let val = theta * scr as f32
                + unsafe { *sv.get_unchecked(state_index(up_score as usize, new_scored as usize)) };
            let better = if minimize { val < best_val } else { val > best_val };
            if better { best_val = val; best_cat = c; best_score = scr; }
        }
    }
    (best_cat, best_score)
}

/// Find best category for the last turn in risk-sensitive mode.
fn find_best_category_final_risk(
    ctx: &YatzyContext,
    up_score: i32,
    scored: i32,
    ds_index: usize,
    theta: f32,
    minimize: bool,
) -> (usize, i32) {
    let mut best_val = if minimize { f32::INFINITY } else { f32::NEG_INFINITY };
    let mut best_cat = 0;
    let mut best_score = 0;
    for c in 0..CATEGORY_COUNT {
        if !is_category_scored(scored, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            let bonus = if c < 6 {
                let new_up = update_upper_score(up_score, c, scr);
                if new_up >= 63 && up_score < 63 { 50 } else { 0 }
            } else { 0 };
            let val = theta * (scr + bonus) as f32;
            let better = if minimize { val < best_val } else { val > best_val };
            if better { best_val = val; best_cat = c; best_score = scr; }
        }
    }
    (best_cat, best_score)
}

/// Simulate one turn for a single game.
fn simulate_turn(
    bctx: &BridgeContext,
    theta_index: usize,
    up_score: i32,
    scored: i32,
    turn: i32,
    rng: &mut SmallRng,
) -> (i32, i32, i32) {
    let (theta, ref sv_enum) = bctx.theta_tables[theta_index];
    let sv = sv_enum.as_slice();
    let ctx = &bctx.ctx;
    let is_last = turn == (CATEGORY_COUNT as i32 - 1);
    let use_risk = theta != 0.0;
    let minimize = theta < 0.0;

    let mut dice = roll_dice(rng);

    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

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

    // Second reroll
    let mask2 = if use_risk {
        choose_best_reroll_mask_risk(ctx, &e_ds_0, &dice, &mut best_ev, minimize)
    } else {
        choose_best_reroll_mask(ctx, &e_ds_0, &dice, &mut best_ev)
    };
    if mask2 != 0 {
        apply_reroll(&mut dice, mask2, rng);
    }

    // Choose category
    let ds_index = find_dice_set_index(ctx, &dice);
    let (cat, scr) = if use_risk {
        if is_last {
            find_best_category_final_risk(ctx, up_score, scored, ds_index, theta, minimize)
        } else {
            find_best_category_risk(ctx, sv, up_score, scored, ds_index, theta, minimize)
        }
    } else if is_last {
        find_best_category_final(ctx, up_score, scored, ds_index)
    } else {
        find_best_category(ctx, sv, up_score, scored, ds_index)
    };

    let new_up = update_upper_score(up_score, cat, scr);
    (cat as i32, scr, new_up)
}

// ============================================================================
// FFI exports
// ============================================================================

/// Initialize: load theta tables and build context.
///
/// # Arguments
/// - `base_path`: C string, path to backend directory (e.g., "backend")
/// - `thetas`: pointer to float array of theta values
/// - `n_thetas`: number of theta values
///
/// # Returns
/// Opaque pointer to BridgeContext (must be freed with `rl_bridge_free`).
#[no_mangle]
pub unsafe extern "C" fn rl_bridge_init(
    base_path: *const c_char,
    thetas: *const f32,
    n_thetas: i32,
) -> *mut BridgeContext {
    let base = CStr::from_ptr(base_path).to_str().unwrap_or("backend");
    let theta_slice = slice::from_raw_parts(thetas, n_thetas as usize);

    // Build lookup tables
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    // Load theta tables
    let mut theta_tables = Vec::with_capacity(theta_slice.len());
    for &theta in theta_slice {
        let path = format!("{}/{}", base, state_file_path(theta));
        match load_state_values_standalone(&path) {
            Some(sv) => {
                eprintln!("Loaded theta={:.3} from {}", theta, path);
                theta_tables.push((theta, sv));
            }
            None => {
                eprintln!("ERROR: Failed to load {}", path);
                return std::ptr::null_mut();
            }
        }
    }

    let bctx = Box::new(BridgeContext { ctx, theta_tables });
    Box::into_raw(bctx)
}

/// Free bridge context.
#[no_mangle]
pub unsafe extern "C" fn rl_bridge_free(ptr: *mut BridgeContext) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// Batch simulate one turn for N games.
///
/// Each game can use a different theta table. All games are simulated in parallel.
///
/// # Arguments
/// - `ptr`: context from `rl_bridge_init`
/// - `n`: number of games in batch
/// - `theta_indices`: [n] which theta table index each game uses
/// - `upper_scores`: [n] current upper score per game
/// - `scored_cats`: [n] current scored-categories bitmask per game
/// - `seeds`: [n] RNG seed per game (for reproducibility)
/// - `turn`: current turn number (0-14)
/// - `out_categories`: [n] output: category scored
/// - `out_scores`: [n] output: score awarded
/// - `out_new_upper_scores`: [n] output: new upper score
#[no_mangle]
pub unsafe extern "C" fn rl_bridge_batch_turn(
    ptr: *const BridgeContext,
    n: i32,
    theta_indices: *const i32,
    upper_scores: *const i32,
    scored_cats: *const i32,
    seeds: *const u64,
    turn: i32,
    out_categories: *mut i32,
    out_scores: *mut i32,
    out_new_upper_scores: *mut i32,
) {
    let bctx = &*ptr;
    let n = n as usize;
    let ti = slice::from_raw_parts(theta_indices, n);
    let ups = slice::from_raw_parts(upper_scores, n);
    let scs = slice::from_raw_parts(scored_cats, n);
    let sd = slice::from_raw_parts(seeds, n);
    let out_cat = slice::from_raw_parts_mut(out_categories, n);
    let out_scr = slice::from_raw_parts_mut(out_scores, n);
    let out_nup = slice::from_raw_parts_mut(out_new_upper_scores, n);

    // Parallel via rayon
    (0..n).into_par_iter().for_each(|i| {
        let mut rng = SmallRng::seed_from_u64(sd[i]);
        let (cat, scr, new_up) =
            simulate_turn(bctx, ti[i] as usize, ups[i], scs[i], turn, &mut rng);
        // Safety: each index i is unique, no data races
        let out_cat = out_cat.as_ptr() as *mut i32;
        let out_scr = out_scr.as_ptr() as *mut i32;
        let out_nup = out_nup.as_ptr() as *mut i32;
        *out_cat.add(i) = cat;
        *out_scr.add(i) = scr;
        *out_nup.add(i) = new_up;
    });
}

/// Batch simulate full games for N seeds, returning final scores.
///
/// Each game uses a fixed theta table (all turns use the same theta).
/// Used for evaluation.
///
/// # Arguments
/// - `ptr`: context from `rl_bridge_init`
/// - `n`: number of games
/// - `theta_index`: which theta table to use (same for all games)
/// - `seeds`: [n] RNG seeds
/// - `out_scores`: [n] output: final game scores (including bonus)
#[no_mangle]
pub unsafe extern "C" fn rl_bridge_batch_game(
    ptr: *const BridgeContext,
    n: i32,
    theta_index: i32,
    seeds: *const u64,
    out_scores: *mut i32,
) {
    let bctx = &*ptr;
    let n = n as usize;
    let sd = slice::from_raw_parts(seeds, n);
    let out = slice::from_raw_parts_mut(out_scores, n);

    (0..n).into_par_iter().for_each(|i| {
        let mut rng = SmallRng::seed_from_u64(sd[i]);
        let mut up_score = 0i32;
        let mut scored = 0i32;
        let mut total = 0i32;

        for turn in 0..CATEGORY_COUNT as i32 {
            let (cat, scr, new_up) =
                simulate_turn(bctx, theta_index as usize, up_score, scored, turn, &mut rng);
            up_score = new_up;
            scored |= 1 << cat;
            total += scr;
        }
        if up_score >= 63 {
            total += 50;
        }

        let out_ptr = out.as_ptr() as *mut i32;
        *out_ptr.add(i) = total;
    });
}

/// Get the number of loaded theta tables.
#[no_mangle]
pub unsafe extern "C" fn rl_bridge_n_thetas(ptr: *const BridgeContext) -> i32 {
    (*ptr).theta_tables.len() as i32
}

/// Get the theta value at the given index.
#[no_mangle]
pub unsafe extern "C" fn rl_bridge_theta_at(ptr: *const BridgeContext, index: i32) -> f32 {
    let bctx = &*ptr;
    bctx.theta_tables[index as usize].0
}

/// Get the starting EV (state_values[0]) for the given theta index.
#[no_mangle]
pub unsafe extern "C" fn rl_bridge_starting_ev(ptr: *const BridgeContext, index: i32) -> f32 {
    let bctx = &*ptr;
    bctx.theta_tables[index as usize].1.as_slice()[0]
}

// ============================================================================
// Per-decision FFI exports (for Approach B: direct action RL)
// ============================================================================

/// Batch roll initial dice for N games.
///
/// # Arguments
/// - `n`: number of games
/// - `seeds`: [n] RNG seeds
/// - `out_dice`: [n*5] output: sorted dice values (row-major)
#[no_mangle]
pub unsafe extern "C" fn rl_bridge_batch_roll(
    _ptr: *const BridgeContext,
    n: i32,
    seeds: *const u64,
    out_dice: *mut i32,
) {
    let n = n as usize;
    let sd = slice::from_raw_parts(seeds, n);
    let out_d = slice::from_raw_parts_mut(out_dice, n * 5);

    (0..n).into_par_iter().for_each(|i| {
        let mut rng = SmallRng::seed_from_u64(sd[i]);
        let dice = roll_dice(&mut rng);
        let out = out_d.as_ptr() as *mut i32;
        for j in 0..5 {
            *out.add(i * 5 + j) = dice[j];
        }
    });
}

/// Batch apply reroll masks to dice for N games.
///
/// # Arguments
/// - `n`: number of games
/// - `dice`: [n*5] current dice (row-major)
/// - `masks`: [n] reroll masks (bits set = reroll that position)
/// - `seeds`: [n] RNG seeds
/// - `out_dice`: [n*5] output: new sorted dice (row-major)
#[no_mangle]
pub unsafe extern "C" fn rl_bridge_batch_apply_reroll(
    _ptr: *const BridgeContext,
    n: i32,
    dice: *const i32,
    masks: *const i32,
    seeds: *const u64,
    out_dice: *mut i32,
) {
    let n = n as usize;
    let dice_arr = slice::from_raw_parts(dice, n * 5);
    let masks_arr = slice::from_raw_parts(masks, n);
    let sd = slice::from_raw_parts(seeds, n);
    let out_d = slice::from_raw_parts_mut(out_dice, n * 5);

    (0..n).into_par_iter().for_each(|i| {
        let mut rng = SmallRng::seed_from_u64(sd[i]);
        let mut d = [0i32; 5];
        for j in 0..5 {
            d[j] = dice_arr[i * 5 + j];
        }
        apply_reroll(&mut d, masks_arr[i], &mut rng);
        let out = out_d.as_ptr() as *mut i32;
        for j in 0..5 {
            *out.add(i * 5 + j) = d[j];
        }
    });
}

/// Batch score a category for N games, returning score + updated state.
///
/// # Arguments
/// - `ptr`: context
/// - `n`: number of games
/// - `dice`: [n*5] current dice (row-major, sorted)
/// - `categories`: [n] category to score (0-14)
/// - `upper_scores`: [n] current upper score
/// - `scored_cats`: [n] current scored-categories bitmask
/// - `out_scores`: [n] output: score awarded
/// - `out_new_upper_scores`: [n] output: new upper score
#[no_mangle]
pub unsafe extern "C" fn rl_bridge_batch_score_category(
    ptr: *const BridgeContext,
    n: i32,
    dice: *const i32,
    categories: *const i32,
    upper_scores: *const i32,
    scored_cats: *const i32,
    out_scores: *mut i32,
    out_new_upper_scores: *mut i32,
) {
    let bctx = &*ptr;
    let ctx = &bctx.ctx;
    let n = n as usize;
    let dice_arr = slice::from_raw_parts(dice, n * 5);
    let cats = slice::from_raw_parts(categories, n);
    let ups = slice::from_raw_parts(upper_scores, n);
    let _scs = slice::from_raw_parts(scored_cats, n);

    let out_s = slice::from_raw_parts_mut(out_scores, n);
    let out_u = slice::from_raw_parts_mut(out_new_upper_scores, n);

    (0..n).into_par_iter().for_each(|i| {
        let mut d = [0i32; 5];
        for j in 0..5 {
            d[j] = dice_arr[i * 5 + j];
        }
        let ds_index = find_dice_set_index(ctx, &d);
        let cat = cats[i] as usize;
        let scr = ctx.precomputed_scores[ds_index][cat];
        let new_up = update_upper_score(ups[i], cat, scr);

        let out_sp = out_s.as_ptr() as *mut i32;
        let out_up = out_u.as_ptr() as *mut i32;
        *out_sp.add(i) = scr;
        *out_up.add(i) = new_up;
    });
}

/// Batch get expert reroll mask for N games (using theta=0 table).
///
/// Computes the optimal reroll mask given current dice, game state,
/// and rerolls remaining. Used for behavioral cloning.
///
/// # Arguments
/// - `ptr`: context
/// - `n`: number of games
/// - `dice`: [n*5] current dice (row-major, sorted)
/// - `upper_scores`: [n] current upper score
/// - `scored_cats`: [n] current scored-categories bitmask
/// - `rerolls_remaining`: 1 or 2
/// - `theta_index`: which theta table to use for expert decisions
/// - `out_masks`: [n] output: optimal reroll mask
#[no_mangle]
pub unsafe extern "C" fn rl_bridge_batch_expert_reroll(
    ptr: *const BridgeContext,
    n: i32,
    dice: *const i32,
    upper_scores: *const i32,
    scored_cats: *const i32,
    rerolls_remaining: i32,
    theta_index: i32,
    out_masks: *mut i32,
) {
    let bctx = &*ptr;
    let ctx = &bctx.ctx;
    let (theta, ref sv_enum) = bctx.theta_tables[theta_index as usize];
    let sv = sv_enum.as_slice();
    let n = n as usize;
    let dice_arr = slice::from_raw_parts(dice, n * 5);
    let ups = slice::from_raw_parts(upper_scores, n);
    let scs = slice::from_raw_parts(scored_cats, n);
    let use_risk = theta != 0.0;
    let minimize = theta < 0.0;
    let out_m = slice::from_raw_parts_mut(out_masks, n);

    (0..n).into_par_iter().for_each(|i| {
        let mut d = [0i32; 5];
        for j in 0..5 {
            d[j] = dice_arr[i * 5 + j];
        }

        let mut e_ds_0 = [0.0f32; 252];
        if use_risk {
            compute_group6_risk(ctx, sv, ups[i], scs[i], theta, minimize, &mut e_ds_0);
        } else {
            compute_group6(ctx, sv, ups[i], scs[i], &mut e_ds_0);
        }

        let mask = if rerolls_remaining == 2 {
            let mut e_ds_1 = [0.0f32; 252];
            if use_risk {
                compute_opt_lse_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1, minimize);
                let mut best_ev = 0.0f64;
                choose_best_reroll_mask_risk(ctx, &e_ds_1, &d, &mut best_ev, minimize)
            } else {
                compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);
                let mut best_ev = 0.0f64;
                choose_best_reroll_mask(ctx, &e_ds_1, &d, &mut best_ev)
            }
        } else {
            if use_risk {
                let mut best_ev = 0.0f64;
                choose_best_reroll_mask_risk(ctx, &e_ds_0, &d, &mut best_ev, minimize)
            } else {
                let mut best_ev = 0.0f64;
                choose_best_reroll_mask(ctx, &e_ds_0, &d, &mut best_ev)
            }
        };

        let out_p = out_m.as_ptr() as *mut i32;
        *out_p.add(i) = mask;
    });
}

/// Batch get expert category for N games (using theta table).
///
/// # Arguments
/// - `ptr`: context
/// - `n`: number of games
/// - `dice`: [n*5] current dice (row-major, sorted)
/// - `upper_scores`: [n] current upper score
/// - `scored_cats`: [n] current scored-categories bitmask
/// - `turn`: current turn number (0-14, used to detect last turn)
/// - `theta_index`: which theta table to use
/// - `out_categories`: [n] output: optimal category
#[no_mangle]
pub unsafe extern "C" fn rl_bridge_batch_expert_category(
    ptr: *const BridgeContext,
    n: i32,
    dice: *const i32,
    upper_scores: *const i32,
    scored_cats: *const i32,
    turn: i32,
    theta_index: i32,
    out_categories: *mut i32,
) {
    let bctx = &*ptr;
    let ctx = &bctx.ctx;
    let (theta, ref sv_enum) = bctx.theta_tables[theta_index as usize];
    let sv = sv_enum.as_slice();
    let n = n as usize;
    let dice_arr = slice::from_raw_parts(dice, n * 5);
    let ups = slice::from_raw_parts(upper_scores, n);
    let scs = slice::from_raw_parts(scored_cats, n);
    let is_last = turn == (CATEGORY_COUNT as i32 - 1);
    let use_risk = theta != 0.0;
    let minimize = theta < 0.0;
    let out_c = slice::from_raw_parts_mut(out_categories, n);

    (0..n).into_par_iter().for_each(|i| {
        let mut d = [0i32; 5];
        for j in 0..5 {
            d[j] = dice_arr[i * 5 + j];
        }
        let ds_index = find_dice_set_index(ctx, &d);

        let (cat, _scr) = if use_risk {
            if is_last {
                find_best_category_final_risk(ctx, ups[i], scs[i], ds_index, theta, minimize)
            } else {
                find_best_category_risk(ctx, sv, ups[i], scs[i], ds_index, theta, minimize)
            }
        } else if is_last {
            find_best_category_final(ctx, ups[i], scs[i], ds_index)
        } else {
            find_best_category(ctx, sv, ups[i], scs[i], ds_index)
        };

        let out = out_c.as_ptr() as *mut i32;
        *out.add(i) = cat as i32;
    });
}
