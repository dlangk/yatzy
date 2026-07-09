//! Lockstep simulation — process all games through each turn together.
//!
//! Instead of simulating each game independently (vertical), this module
//! processes all N games through Turn 1, then Turn 2, etc. (horizontal).
//!
//! **Key optimizations**:
//! 1. Games sharing the same (upper_score, scored_categories) state need only
//!    ONE call to `compute_group6` + `compute_max_ev_for_n_rerolls`.
//!    At turn 0 all games share state (0, 0). At turn 1 they split into ~21 states.
//!    The amortization is strongest in early turns where few unique states exist.
//!
//! 2. Radix sort (2-pass counting sort) replaces HashMap for grouping games by
//!    state. O(2N) with no hashing overhead, produces contiguous groups for
//!    cache-friendly sequential access.
//!
//! 3. SplitMix64 PRNG replaces SmallRng — single u64 state (8 bytes vs 128 bytes),
//!    ~2 cycles per output, and extracts 5 dice from a single u64.

use rayon::prelude::*;
use std::time::Instant;

use crate::constants::*;
use crate::dice_mechanics::{find_dice_set_index, sort_dice_set};
use crate::game_mechanics::update_upper_score;
use crate::types::YatzyContext;
use crate::widget_solver::{
    choose_best_reroll_mask_by_ds, choose_best_reroll_mask_by_keep_ev, compute_keep_ev,
    compute_max_ev_and_keep_ev,
};

use super::engine::SimulationResult;
use super::fast_prng::SplitMix64;
use super::radix_sort::{radix_sort_by_state_into_parallel, RadixScratch};

/// Per-game mutable state during lockstep simulation.
struct GameState {
    upper_score: i32,
    scored: i32,
    total_score: i32,
    rng: SplitMix64,
}

/// Cached per-state computation: e_ds_0[252], e_ds_1[252], and per-keep EVs.
struct StateCache {
    e_ds_0: [f32; 252],
    e_ds_1: [f32; 252],
    /// Per-keep EVs against e_ds_0 (second-reroll decisions). Free by-product
    /// of computing e_ds_1; bit-identical to the per-game dot products.
    keep_ev_0: [f32; NUM_KEEP_MULTISETS],
    /// Per-keep EVs against e_ds_1 (first-reroll decisions). Costs one extra
    /// 210-row pass, so only computed for groups with at least
    /// [`KEEP_EV_THRESHOLD`] games.
    keep_ev_1: [f32; NUM_KEEP_MULTISETS],
    has_keep_ev_1: bool,
    /// Precomputed argmax decisions, built only for groups with at least
    /// [`TABLE_THRESHOLD`] games (see `simulate_batch_lockstep` Step 3).
    table: Option<Box<DecisionTable>>,
}

/// Per-dice-set argmax decisions for one state (the local, on-the-fly
/// analogue of the PolicyOracle): reroll masks and category choice depend
/// only on (state, dice set), so within a group every game rolling the same
/// dice set makes the identical decision.
struct DecisionTable {
    /// Best reroll mask with 2 rerolls left (from e_ds_1), per dice set.
    first_mask: [u8; 252],
    /// Best reroll mask with 1 reroll left (from e_ds_0), per dice set.
    second_mask: [u8; 252],
    /// Best category per final dice set.
    cat: [u8; 252],
    /// Score of that category (max single-category score is 50, fits u8).
    score: [u8; 252],
}

// The macOS tiny-zone allocator ceiling is 1008 bytes: one more byte moves
// Box<DecisionTable> to the contended small-zone allocator (measured 27x
// slower per alloc under 18-thread contention, ~2ms/turn hidden cost).
// Growing this struct needs an allocation-strategy rethink, not just a field.
const _: () = assert!(std::mem::size_of::<DecisionTable>() <= 1008);

/// Minimum games in a group before building a DecisionTable pays for itself.
/// Break-even is ~252 games (table build costs 252 argmaxes per decision
/// level vs one per game). Measured at 1M games on M5 Max: 384 and 256
/// perform identically, 1024 leaves ~50 ms on the table; 384 chosen for
/// the lower table-build overhead.
const TABLE_THRESHOLD: u32 = 384;

/// Minimum games in a group before the extra keep_ev_1 pass (210 CSR dot
/// products, ~2.5us) beats per-game first-reroll argmaxes (~1us each).
const KEEP_EV_THRESHOLD: u32 = 3;

/// One game's full turn: two reroll decisions, category choice, state update.
///
/// Table path: pure lookups (groups >= TABLE_THRESHOLD). Fallback path:
/// keep_ev lookup argmaxes (identical values and tie-breaking as the direct
/// dot-product argmax; see widget_solver::compute_keep_ev).
#[inline(always)]
fn process_game_turn(
    ctx: &YatzyContext,
    sv: &[f32],
    cache: &StateCache,
    g: &mut GameState,
    dice: &mut [i32; 5],
    is_last_turn: bool,
) {
    let (cat, scr) = if let Some(t) = &cache.table {
        // Table path: dice are canonical (sorted), decisions are
        // pure lookups. Identical argmax as the fallback path.
        let ds0 = find_dice_set_index(ctx, dice);
        let mask1 = t.first_mask[ds0] as i32;
        let ds1 = if mask1 != 0 {
            apply_reroll(dice, mask1, &mut g.rng);
            find_dice_set_index(ctx, dice)
        } else {
            ds0
        };
        let mask2 = t.second_mask[ds1] as i32;
        let ds2 = if mask2 != 0 {
            apply_reroll(dice, mask2, &mut g.rng);
            find_dice_set_index(ctx, dice)
        } else {
            ds1
        };
        (t.cat[ds2] as usize, t.score[ds2] as i32)
    } else {
        // Fallback path: dice are canonical (sorted), so the
        // dice-set index threads through instead of re-sorting.
        let mut ds = find_dice_set_index(ctx, dice);

        // First reroll: keep_ev_1 lookups when the group paid for
        // them, else the per-game dot-product argmax (identical
        // decisions either way).
        let mask1 = if cache.has_keep_ev_1 {
            choose_best_reroll_mask_by_keep_ev(ctx, &cache.keep_ev_1, &cache.e_ds_1, ds)
        } else {
            choose_best_reroll_mask_by_ds(ctx, &cache.e_ds_1, ds).0
        };
        if mask1 != 0 {
            apply_reroll(dice, mask1, &mut g.rng);
            ds = find_dice_set_index(ctx, dice);
        }

        // Second reroll: keep_ev_0 is always available.
        let mask2 = choose_best_reroll_mask_by_keep_ev(ctx, &cache.keep_ev_0, &cache.e_ds_0, ds);
        if mask2 != 0 {
            apply_reroll(dice, mask2, &mut g.rng);
            ds = find_dice_set_index(ctx, dice);
        }

        // Score
        if is_last_turn {
            find_best_category_final(ctx, g.upper_score, g.scored, ds)
        } else {
            find_best_category(ctx, sv, g.upper_score, g.scored, ds)
        }
    };

    g.upper_score = update_upper_score(g.upper_score, cat, scr);
    g.scored |= 1 << cat;
    g.total_score += scr;
}

/// Roll 5 random dice and sort them.
#[inline(always)]
fn roll_dice(rng: &mut SplitMix64) -> [i32; 5] {
    let mut dice = rng.roll_5_dice();
    sort_dice_set(&mut dice);
    dice
}

/// Apply a reroll mask.
#[inline(always)]
fn apply_reroll(dice: &mut [i32; 5], mask: i32, rng: &mut SplitMix64) {
    for i in 0..5 {
        if mask & (1 << i) != 0 {
            dice[i] = rng.roll_die();
        }
    }
    sort_dice_set(dice);
}

/// Find the best category (non-final turn).
// PERF: intentional duplication of widget_solver Group 6 scoring logic
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

    // Upper categories (0-5): affect upper_score tracking, so the successor
    // state index depends on update_upper_score(up_score, c, scr).
    for c in 0..6 {
        if !is_category_scored(scored, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            let new_up = update_upper_score(up_score, c, scr);
            let new_scored = scored | (1 << c);
            // SAFETY: new_up = update_upper_score(..) is in 0..64,
            // new_scored = scored | (1<<c) < 2^15. So
            // state_index(new_up, new_scored) < NUM_STATES = sv.len().
            let val = scr as f32
                + unsafe { *sv.get_unchecked(state_index(new_up as usize, new_scored as usize)) };
            if val > best_val {
                best_val = val;
                best_cat = c;
                best_score = scr;
            }
        }
    }

    // Lower categories (6-14): do not affect upper_score, so the successor
    // state index uses the unchanged up_score directly.
    for c in 6..CATEGORY_COUNT {
        if !is_category_scored(scored, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            let new_scored = scored | (1 << c);
            // SAFETY: up_score is in 0..64, new_scored < 2^15. So
            // state_index(up_score, new_scored) < NUM_STATES = sv.len().
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

/// Find the best category for the final turn (turn 14, all categories scored after).
///
/// On the last turn there are no successor states, so we cannot look up E(successor).
/// Instead we maximize `score + bonus_delta` directly. The bonus-crossing logic:
/// if scoring an upper category (0-5) pushes upper_score to >= 63 when it was < 63,
/// that triggers the +50 upper bonus. Lower categories (6-14) never affect the bonus.
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

/// Compute Group 6: best category EV for each dice set.
// PERF: intentional duplication of widget_solver Group 6 for lockstep self-containment
#[inline(always)]
pub(crate) fn compute_group6(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    e_ds_0: &mut [f32; 252],
) {
    // Preload lower-category successor EVs: for categories 6-14 the successor
    // state value sv[state_index(up, scored|(1<<c))] depends only on (up, scored),
    // not on the dice roll. Reading these once here avoids 252 redundant sv lookups
    // per lower category in the inner loop below.
    let mut lower_succ_ev = [0.0f32; CATEGORY_COUNT];
    for c in 6..CATEGORY_COUNT {
        if !is_category_scored(scored, c) {
            // SAFETY: c in 6..15, scored < 2^15, up_score in 0..64. So
            // state_index(up_score, scored | (1<<c)) < NUM_STATES = sv.len().
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
                // SAFETY: new_up = update_upper_score(..) in 0..64,
                // new_scored < 2^15. state_index(new_up, new_scored) < NUM_STATES = sv.len().
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
                // SAFETY: c is in 6..CATEGORY_COUNT (15), so c < lower_succ_ev.len() (15).
                let val = scr as f32 + unsafe { *lower_succ_ev.get_unchecked(c) };
                if val > best_val {
                    best_val = val;
                }
            }
        }

        e_ds_0[ds_i] = best_val;
    }
}

/// Simulate N games using lockstep processing (EV mode only, θ=0).
///
/// // PERF: intentional duplication with simulation/engine.rs. Lockstep processes
/// // all N games at each turn together (horizontal), while engine.rs processes
/// // one game at a time (vertical). Lockstep is 4.5x faster due to radix sort
/// // grouping and amortized per-state computation.
///
/// All games advance through each turn together, amortizing the expensive
/// per-state computation across games that share the same state.
///
/// Uses radix sort for O(N) grouping and SplitMix64 for fast PRNG.
pub fn simulate_batch_lockstep(
    ctx: &YatzyContext,
    num_games: usize,
    seed: u64,
) -> SimulationResult {
    let start = Instant::now();
    let sv = ctx.state_values.as_slice();

    // Initialize all games with SplitMix64 PRNG
    let mut games: Vec<GameState> = (0..num_games)
        .map(|i| GameState {
            upper_score: 0,
            scored: 0,
            total_score: 0,
            rng: SplitMix64::new(seed.wrapping_add(i as u64)),
        })
        .collect();

    // Per-turn buffers, hoisted: allocating these fresh every turn costs
    // ~36 MB of page-faulted allocations per turn at 1M games.
    let mut dice_per_game: Vec<[i32; 5]> = vec![[0i32; 5]; num_games];
    let mut keys: Vec<u32> = vec![0u32; num_games];
    let mut sorted = RadixScratch::new();
    let mut caches: Vec<StateCache> = Vec::new();

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;

        // Step 1: Roll initial dice for all games
        games
            .par_iter_mut()
            .zip(dice_per_game.par_iter_mut())
            .for_each(|(g, dice)| *dice = roll_dice(&mut g.rng));

        // Step 2: Build state keys and radix sort to group games by state
        keys.par_iter_mut()
            .zip(games.par_iter())
            .for_each(|(k, g)| {
                *k = state_index(g.upper_score as usize, g.scored as usize) as u32;
            });

        radix_sort_by_state_into_parallel(&keys, num_games, &mut sorted);

        // Step 3: Compute per-state caches in parallel
        // Collect unique state indices (from group headers)
        sorted
            .groups
            .par_iter()
            .map(|&(si, _, count)| {
                let scored = (si as usize) / STATE_STRIDE;
                let up = ((si as usize) % STATE_STRIDE) as i32;
                let mut cache = StateCache {
                    e_ds_0: [0.0f32; 252],
                    e_ds_1: [0.0f32; 252],
                    keep_ev_0: [0.0f32; NUM_KEEP_MULTISETS],
                    keep_ev_1: [0.0f32; NUM_KEEP_MULTISETS],
                    has_keep_ev_1: false,
                    table: None,
                };
                compute_group6(ctx, sv, up, scored as i32, &mut cache.e_ds_0);
                // keep_ev_0 is a free by-product of the e_ds_0 -> e_ds_1 pass
                // (identical FP order to the per-game argmax dot products).
                compute_max_ev_and_keep_ev(
                    ctx,
                    &cache.e_ds_0,
                    &mut cache.e_ds_1,
                    &mut cache.keep_ev_0,
                );
                if count >= KEEP_EV_THRESHOLD {
                    compute_keep_ev(ctx, &cache.e_ds_1, &mut cache.keep_ev_1);
                    cache.has_keep_ev_1 = true;
                }

                // Fat groups: precompute all 252 per-dice-set decisions once
                // instead of recomputing the argmax per game. Mask argmaxes
                // are pure lookups over the persisted keep_ev arrays
                // (identical values and tie-breaking as the per-game path).
                if count >= TABLE_THRESHOLD {
                    let mut t = Box::new(DecisionTable {
                        first_mask: [0u8; 252],
                        second_mask: [0u8; 252],
                        cat: [0u8; 252],
                        score: [0u8; 252],
                    });
                    for ds in 0..NUM_DICE_SETS {
                        let m2 = choose_best_reroll_mask_by_keep_ev(
                            ctx,
                            &cache.keep_ev_1,
                            &cache.e_ds_1,
                            ds,
                        );
                        let m1 = choose_best_reroll_mask_by_keep_ev(
                            ctx,
                            &cache.keep_ev_0,
                            &cache.e_ds_0,
                            ds,
                        );
                        t.first_mask[ds] = m2 as u8;
                        t.second_mask[ds] = m1 as u8;
                        let (cat, scr) = if is_last_turn {
                            find_best_category_final(ctx, up, scored as i32, ds)
                        } else {
                            find_best_category(ctx, sv, up, scored as i32, ds)
                        };
                        t.cat[ds] = cat as u8;
                        t.score[ds] = scr as u8;
                    }
                    cache.table = Some(t);
                }
                cache
            })
            .collect_into_vec(&mut caches);

        // Step 4: process games group-by-group. Each group's StateCache /
        // DecisionTable stays cache-hot for all of its games, and no
        // game→cache inversion pass is needed. Every game index appears in
        // exactly one group (the sort is a partition), so the raw-pointer
        // accesses into games/dice are disjoint — same argument as the
        // AtomicPtr scatter in state_computation.rs. Each game's RNG stream
        // and decisions are independent of processing order, so scores are
        // bit-identical to the game-index-order scan.
        let games_addr = games.as_mut_ptr() as usize;
        let dice_addr = dice_per_game.as_mut_ptr() as usize;
        sorted.groups.par_iter().enumerate().for_each(
            |(group_idx, &(_si, group_start, group_count))| {
                let cache = &caches[group_idx];
                let idx =
                    &sorted.indices[group_start as usize..(group_start + group_count) as usize];
                let process = |gi: usize| {
                    // SAFETY: gi appears in exactly one group; disjoint access.
                    let g = unsafe { &mut *(games_addr as *mut GameState).add(gi) };
                    let dice = unsafe { &mut *(dice_addr as *mut [i32; 5]).add(gi) };
                    process_game_turn(ctx, sv, cache, g, dice, is_last_turn);
                };
                // Fat groups (e.g. turn 0 is a single group of all games)
                // split into parallel chunks to avoid load imbalance.
                if group_count >= 16_384 {
                    idx.par_chunks(4_096).for_each(|chunk| {
                        for &gi in chunk {
                            process(gi as usize);
                        }
                    });
                } else {
                    for &gi in idx {
                        process(gi as usize);
                    }
                }
            },
        );
    }
    // Apply bonus
    let mut scores: Vec<i32> = games
        .iter()
        .map(|g| {
            if g.upper_score >= 63 {
                g.total_score + 50
            } else {
                g.total_score
            }
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

/// Simulate N games using oracle-based lockstep processing (θ=0 only).
///
/// Instead of computing Group 6 + Groups 5/3 per unique state, reads
/// precomputed decisions from the PolicyOracle in O(1). Each turn becomes:
/// 1. Roll initial dice
/// 2. Look up oracle_keep2[si * 252 + ds] → reroll decision
/// 3. If rerolling: apply mask, look up oracle_keep1 → second reroll decision
/// 4. Look up oracle_cat → score
///
/// Eliminates all per-state computation. The radix sort grouping is still
/// used but only for bonus of locality — individual games are now independent.
pub fn simulate_batch_lockstep_oracle(
    ctx: &YatzyContext,
    oracle: &crate::types::PolicyOracle,
    num_games: usize,
    seed: u64,
) -> SimulationResult {
    let start = Instant::now();

    let mut games: Vec<GameState> = (0..num_games)
        .map(|i| GameState {
            upper_score: 0,
            scored: 0,
            total_score: 0,
            rng: SplitMix64::new(seed.wrapping_add(i as u64)),
        })
        .collect();

    let kt = &ctx.keep_table;

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;

        games.par_iter_mut().for_each(|g| {
            let si = state_index(g.upper_score as usize, g.scored as usize);
            let base = si * 252;

            // Roll initial dice
            let mut dice = roll_dice(&mut g.rng);

            // First reroll decision (2 rerolls left)
            let ds0 = find_dice_set_index(ctx, &dice);
            let oracle_keep2 = oracle.keep2();
            let oracle_keep1 = oracle.keep1();
            let oracle_cat = oracle.cat();

            let keep2 = oracle_keep2[base + ds0];
            if keep2 != 0 {
                // Decode oracle keep_id to a reroll bitmask (bit=1 → reroll that die)
                let j = (keep2 - 1) as usize;
                let mask = kt.keep_to_mask[ds0 * 32 + j];
                apply_reroll(&mut dice, mask, &mut g.rng);

                // Second reroll decision (1 reroll left)
                let ds1 = find_dice_set_index(ctx, &dice);
                let keep1 = oracle_keep1[base + ds1];
                if keep1 != 0 {
                    let j1 = (keep1 - 1) as usize;
                    let mask1 = kt.keep_to_mask[ds1 * 32 + j1];
                    apply_reroll(&mut dice, mask1, &mut g.rng);
                }
            } else {
                // Kept all dice from initial roll, check second reroll
                let keep1 = oracle_keep1[base + ds0];
                if keep1 != 0 {
                    let j1 = (keep1 - 1) as usize;
                    let mask1 = kt.keep_to_mask[ds0 * 32 + j1];
                    apply_reroll(&mut dice, mask1, &mut g.rng);
                }
            }

            // Category assignment
            let ds_final = find_dice_set_index(ctx, &dice);
            // Both paths are identical: the oracle already accounts for terminal values
            let cat = if is_last_turn {
                oracle_cat[base + ds_final] as usize
            } else {
                oracle_cat[base + ds_final] as usize
            };

            let scr = ctx.precomputed_scores[ds_final][cat];
            g.upper_score = update_upper_score(g.upper_score, cat, scr);
            g.scored |= 1 << cat;
            g.total_score += scr;
        });
    }

    // Apply bonus
    let mut scores: Vec<i32> = games
        .iter()
        .map(|g| {
            if g.upper_score >= 63 {
                g.total_score + 50
            } else {
                g.total_score
            }
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
    fn test_lockstep_produces_valid_scores() {
        let ctx = make_ctx();
        let result = simulate_batch_lockstep(&ctx, 100, 42);

        for &score in &result.scores {
            assert!(score >= 0 && score <= 374, "Invalid score: {}", score);
        }
    }

    #[test]
    fn test_lockstep_mean_reasonable() {
        let ctx = make_ctx();
        let result = simulate_batch_lockstep(&ctx, 10000, 42);

        // Expected EV ~248.4 with SV loaded; without SV we still expect
        // reasonable scores between 150-300 on average
        assert!(
            result.mean > 100.0 && result.mean < 350.0,
            "Mean {} out of reasonable range",
            result.mean
        );
    }

    /// T7: engine.rs and lockstep.rs carry INTENTIONALLY DUPLICATED copies of
    /// compute_group6 (hot-path rule: no shared helper). The two copies must
    /// stay bit-identical in behavior or the engines silently diverge. This
    /// differential runs both on synthetic state values (no data files) over
    /// random states and requires bitwise-equal outputs.
    #[test]
    fn test_group6_matches_engine_copy() {
        let mut ctx = crate::types::YatzyContext::new_boxed();
        crate::phase0_tables::precompute_lookup_tables(&mut ctx);

        // Synthetic sv: deterministic pseudo-random values; the comparison is
        // a pure function of sv, so reachability does not matter.
        let mut seed = 0x5EEDu64;
        let mut next = move || {
            seed = seed.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            (z >> 40) as f32 / 65536.0 // ~[0, 256)
        };
        let sv: Vec<f32> = (0..crate::constants::NUM_STATES).map(|_| next()).collect();

        let mut state_rng = SplitMix64::new(77);
        for _ in 0..500 {
            let scored = (state_rng.next_u64() % ((1 << 15) - 1)) as i32; // not terminal
            let up_score = (state_rng.next_u64() % 64) as i32;

            let mut a = [0.0f32; 252];
            let mut b = [0.0f32; 252];
            compute_group6(&ctx, &sv, up_score, scored, &mut a);
            crate::simulation::engine::compute_group6(&ctx, &sv, up_score, scored, &mut b);
            for ds in 0..252 {
                assert!(
                    a[ds].to_bits() == b[ds].to_bits(),
                    "group6 copies diverge at (up={up_score}, scored={scored:#x}, \
                     ds={ds}): lockstep {} vs engine {}",
                    a[ds],
                    b[ds]
                );
            }
        }
    }
}
