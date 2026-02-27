//! Per-state transition computation with decision tracking.
//!
//! For a given (upper_score, scored_categories), computes all possible
//! (next_state, points_earned, probability) triples by enumerating:
//! - All 252 initial dice outcomes
//! - Optimal reroll decisions at each step
//! - All reachable dice after rerolling
//! - Optimal category assignment

use std::collections::HashMap;

use crate::constants::*;
use crate::game_mechanics::update_upper_score;
use crate::types::YatzyContext;

/// A single transition: from current state to next_state, earning `points` with probability `prob`.
#[derive(Debug, Clone)]
pub struct StateTransition {
    pub next_state: u32,
    pub points: u16,
    pub prob: f64,
}

/// Sentinel: keep all dice (mask=0), encoded as keep_id.
const KEEP_ALL: u16 = 0xFFFF;

/// Compute the optimal category for each dice set (Group 6 with decision tracking).
///
/// Returns (e_ds_0[252], best_cat[252]) where:
/// - e_ds_0[ds] = best EV achievable by scoring dice set ds
/// - best_cat[ds] = category index that achieves it
fn compute_group6_with_category(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    theta: f32,
    minimize: bool,
) -> ([f32; 252], [u8; 252]) {
    let mut e_ds_0 = [0.0f32; 252];
    let mut best_cat = [0u8; 252];

    let use_risk = theta != 0.0;

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
        let mut bc = 0u8;

        for c in 0..6 {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_i][c];
                let new_up = update_upper_score(up_score, c, scr);
                let new_scored = scored | (1 << c);
                let val = if use_risk {
                    theta * scr as f32
                        + unsafe {
                            *sv.get_unchecked(state_index(new_up as usize, new_scored as usize))
                        }
                } else {
                    scr as f32
                        + unsafe {
                            *sv.get_unchecked(state_index(new_up as usize, new_scored as usize))
                        }
                };
                let better = if minimize {
                    val < best_val
                } else {
                    val > best_val
                };
                if better {
                    best_val = val;
                    bc = c as u8;
                }
            }
        }

        for c in 6..CATEGORY_COUNT {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_i][c];
                let val = if use_risk {
                    theta * scr as f32 + unsafe { *lower_succ_ev.get_unchecked(c) }
                } else {
                    scr as f32 + unsafe { *lower_succ_ev.get_unchecked(c) }
                };
                let better = if minimize {
                    val < best_val
                } else {
                    val > best_val
                };
                if better {
                    best_val = val;
                    bc = c as u8;
                }
            }
        }

        e_ds_0[ds_i] = best_val;
        best_cat[ds_i] = bc;
    }

    (e_ds_0, best_cat)
}

/// Compute reroll EV with decision tracking.
///
/// Returns (e_ds_current[252], best_keep[252]) where:
/// - e_ds_current[ds] = best EV after optimal keep decision
/// - best_keep[ds] = keep_id of the optimal keep, or KEEP_ALL if keeping all dice
fn compute_reroll_with_decisions(
    ctx: &YatzyContext,
    e_ds_prev: &[f32; 252],
    theta: f32,
    minimize: bool,
) -> ([f32; 252], [u16; 252]) {
    let kt = &ctx.keep_table;
    let vals = &kt.vals;
    let cols = &kt.cols;
    let use_risk = theta != 0.0;

    // Step 1: compute EV/LSE for each unique keep-multiset
    let mut keep_ev = [0.0f32; NUM_KEEP_MULTISETS];
    for kid in 0..NUM_KEEP_MULTISETS {
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;
        if start == end {
            keep_ev[kid] = if minimize {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
            continue;
        }

        if use_risk {
            // LSE for risk-sensitive mode
            let mut max_val = f32::NEG_INFINITY;
            for k in start..end {
                let v = unsafe { *e_ds_prev.get_unchecked(*cols.get_unchecked(k) as usize) };
                if v > max_val {
                    max_val = v;
                }
            }
            let mut sum: f32 = 0.0;
            for k in start..end {
                unsafe {
                    let v = *e_ds_prev.get_unchecked(*cols.get_unchecked(k) as usize);
                    sum += *vals.get_unchecked(k) * (v - max_val).exp();
                }
            }
            keep_ev[kid] = max_val + sum.ln();
        } else {
            // Weighted sum for EV mode
            let mut ev: f32 = 0.0;
            for k in start..end {
                unsafe {
                    ev += *vals.get_unchecked(k)
                        * e_ds_prev.get_unchecked(*cols.get_unchecked(k) as usize);
                }
            }
            keep_ev[kid] = ev;
        }
    }

    // Step 2: for each dice set, find optimal keep
    let mut e_ds_current = [0.0f32; 252];
    let mut best_keep = [KEEP_ALL; 252];

    for ds_i in 0..252 {
        let mut best_val = e_ds_prev[ds_i]; // mask=0: keep all

        for j in 0..kt.unique_count[ds_i] as usize {
            let kid = unsafe { *kt.unique_keep_ids[ds_i].get_unchecked(j) } as usize;
            let ev = keep_ev[kid];
            let better = if minimize {
                ev < best_val
            } else {
                ev > best_val
            };
            if better {
                best_val = ev;
                best_keep[ds_i] = kid as u16;
            }
        }
        e_ds_current[ds_i] = best_val;
    }

    (e_ds_current, best_keep)
}

/// Compute all transitions from a given state (upper_score, scored_categories).
///
/// Enumerates all paths through a turn:
/// 1. Initial roll → 252 dice sets with known probabilities
/// 2. Optimal keep decision → reroll
/// 3. All reachable dice after reroll
/// 4. Optimal keep decision → second reroll
/// 5. All reachable dice after second reroll
/// 6. Optimal category assignment → points earned + next state
///
/// Returns deduplicated transitions: (next_state, points, probability).
pub fn compute_transitions(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    theta: f32,
) -> Vec<StateTransition> {
    let minimize = theta < 0.0;
    let kt = &ctx.keep_table;

    // Group 6: best category per dice set
    let (e_ds_0, best_cat) =
        compute_group6_with_category(ctx, sv, up_score, scored, theta, minimize);

    // Group 5: best keep after 1st reroll (uses Group 6 EVs)
    let (e_ds_1, best_keep_1) = compute_reroll_with_decisions(ctx, &e_ds_0, theta, minimize);

    // Group 3: best keep from initial roll (uses Group 5 EVs)
    let (_e_ds_2, best_keep_2) = compute_reroll_with_decisions(ctx, &e_ds_1, theta, minimize);

    // Enumerate all paths through one turn and accumulate transition probabilities.
    // A turn has 3 decision points: keep2 (from initial roll), keep1 (after 1st reroll),
    // and category selection (after final dice). This creates 4 path shapes:
    //
    //   A. keep2=all, keep1=all → score ds0 directly
    //   B. keep2=all, keep1=reroll → score rerolled ds_final
    //   C. keep2=reroll, keep1=all → score ds_mid from 1st reroll
    //   D. keep2=reroll, keep1=reroll → score ds_final from 2nd reroll
    //
    // Each path accumulates P(ds0) * P(reroll outcomes) into (next_state, points).
    let mut accum: HashMap<(u32, u16), f64> = HashMap::new();

    for ds0 in 0..252usize {
        let p_ds0 = ctx.dice_set_probabilities[ds0];
        if p_ds0 == 0.0 {
            continue;
        }

        // Decision: keep2 for initial roll ds0
        let kid2 = best_keep_2[ds0];

        if kid2 == KEEP_ALL {
            // Keep all dice from initial roll → skip both rerolls
            // Decision: keep1 for ds0 after "1st reroll" (which didn't happen)
            let kid1 = best_keep_1[ds0];

            if kid1 == KEEP_ALL {
                // Keep all again → score ds0 directly
                let cat = best_cat[ds0] as usize;
                let pts = ctx.precomputed_scores[ds0][cat];
                let new_up = if cat < 6 {
                    update_upper_score(up_score, cat, pts)
                } else {
                    up_score
                };
                let new_scored = scored | (1 << cat);
                let next_si = state_index(new_up as usize, new_scored as usize) as u32;

                *accum.entry((next_si, pts as u16)).or_insert(0.0) += p_ds0;
            } else {
                // Reroll from ds0 with keep1
                let start1 = kt.row_start[kid1 as usize] as usize;
                let end1 = kt.row_start[kid1 as usize + 1] as usize;

                for k1 in start1..end1 {
                    let p_reroll1 = kt.vals[k1] as f64;
                    let ds_final = kt.cols[k1] as usize;

                    let cat = best_cat[ds_final] as usize;
                    let pts = ctx.precomputed_scores[ds_final][cat];
                    let new_up = if cat < 6 {
                        update_upper_score(up_score, cat, pts)
                    } else {
                        up_score
                    };
                    let new_scored = scored | (1 << cat);
                    let next_si = state_index(new_up as usize, new_scored as usize) as u32;

                    *accum.entry((next_si, pts as u16)).or_insert(0.0) += p_ds0 * p_reroll1;
                }
            }
        } else {
            // Reroll from ds0 with keep2
            let start2 = kt.row_start[kid2 as usize] as usize;
            let end2 = kt.row_start[kid2 as usize + 1] as usize;

            for k2 in start2..end2 {
                let p_reroll2 = kt.vals[k2] as f64;
                let ds_mid = kt.cols[k2] as usize;

                // Decision: keep1 for ds_mid
                let kid1 = best_keep_1[ds_mid];

                if kid1 == KEEP_ALL {
                    // Keep all → score ds_mid
                    let cat = best_cat[ds_mid] as usize;
                    let pts = ctx.precomputed_scores[ds_mid][cat];
                    let new_up = if cat < 6 {
                        update_upper_score(up_score, cat, pts)
                    } else {
                        up_score
                    };
                    let new_scored = scored | (1 << cat);
                    let next_si = state_index(new_up as usize, new_scored as usize) as u32;

                    *accum.entry((next_si, pts as u16)).or_insert(0.0) += p_ds0 * p_reroll2;
                } else {
                    // Second reroll from ds_mid with keep1
                    let start1 = kt.row_start[kid1 as usize] as usize;
                    let end1 = kt.row_start[kid1 as usize + 1] as usize;

                    for k1 in start1..end1 {
                        let p_reroll1 = kt.vals[k1] as f64;
                        let ds_final = kt.cols[k1] as usize;

                        let cat = best_cat[ds_final] as usize;
                        let pts = ctx.precomputed_scores[ds_final][cat];
                        let new_up = if cat < 6 {
                            update_upper_score(up_score, cat, pts)
                        } else {
                            up_score
                        };
                        let new_scored = scored | (1 << cat);
                        let next_si = state_index(new_up as usize, new_scored as usize) as u32;

                        *accum.entry((next_si, pts as u16)).or_insert(0.0) +=
                            p_ds0 * p_reroll2 * p_reroll1;
                    }
                }
            }
        }
    }

    // Convert to Vec<StateTransition>
    accum
        .into_iter()
        .map(|((next_state, points), prob)| StateTransition {
            next_state,
            points,
            prob,
        })
        .collect()
}

/// Compute all transitions using the precomputed oracle (θ=0 only).
///
/// Uses probability-array propagation instead of path-by-path enumeration.
/// Two passes propagate a [f64; 252] probability distribution through the oracle's
/// reroll decisions, then a single scoring pass collects transitions with sort+merge
/// instead of HashMap.
///
/// Working set: 2 × [f64; 252] = 4 KB (L1-resident). Oracle reads are sequential
/// (base+0..base+252) for cache locality.
pub fn compute_transitions_oracle(
    ctx: &YatzyContext,
    oracle: &crate::types::PolicyOracle,
    up_score: i32,
    scored: i32,
) -> Vec<StateTransition> {
    let kt = &ctx.keep_table;
    let si = state_index(up_score as usize, scored as usize);
    let base = si * NUM_DICE_SETS;

    let mut cur_probs = [0.0f64; NUM_DICE_SETS];
    let mut nxt_probs = [0.0f64; NUM_DICE_SETS];

    // Initialize with dice set probabilities
    for ds in 0..NUM_DICE_SETS {
        cur_probs[ds] = ctx.dice_set_probabilities[ds];
    }

    // Pass 1: apply reroll-2 oracle decisions (2 rerolls left → 1 reroll left)
    for ds in 0..NUM_DICE_SETS {
        let p = cur_probs[ds];
        if p == 0.0 {
            continue;
        }
        let keep2 = oracle.oracle_keep2[base + ds];
        if keep2 == 0 {
            // Keep all dice
            nxt_probs[ds] += p;
        } else {
            let kid = kt.unique_keep_ids[ds][(keep2 - 1) as usize] as usize;
            let start = kt.row_start[kid] as usize;
            let end = kt.row_start[kid + 1] as usize;
            for k in start..end {
                nxt_probs[kt.cols[k] as usize] += p * kt.vals[k] as f64;
            }
        }
    }

    // Swap: nxt_probs becomes input for pass 2
    cur_probs = nxt_probs;
    nxt_probs = [0.0f64; NUM_DICE_SETS];

    // Pass 2: apply reroll-1 oracle decisions (1 reroll left → final dice)
    for ds in 0..NUM_DICE_SETS {
        let p = cur_probs[ds];
        if p == 0.0 {
            continue;
        }
        let keep1 = oracle.oracle_keep1[base + ds];
        if keep1 == 0 {
            // Keep all dice
            nxt_probs[ds] += p;
        } else {
            let kid = kt.unique_keep_ids[ds][(keep1 - 1) as usize] as usize;
            let start = kt.row_start[kid] as usize;
            let end = kt.row_start[kid + 1] as usize;
            for k in start..end {
                nxt_probs[kt.cols[k] as usize] += p * kt.vals[k] as f64;
            }
        }
    }

    // Pass 3: score each final dice set and collect transitions
    let mut raw: Vec<((u32, u16), f64)> = Vec::with_capacity(NUM_DICE_SETS);
    for ds in 0..NUM_DICE_SETS {
        let p = nxt_probs[ds];
        if p == 0.0 {
            continue;
        }
        let cat = oracle.oracle_cat[base + ds] as usize;
        let pts = ctx.precomputed_scores[ds][cat];
        let new_up = if cat < 6 {
            update_upper_score(up_score, cat, pts)
        } else {
            up_score
        };
        let new_scored = scored | (1 << cat);
        let next_si = state_index(new_up as usize, new_scored as usize) as u32;
        raw.push(((next_si, pts as u16), p));
    }

    // Sort by key and merge duplicates (replaces HashMap)
    raw.sort_unstable_by_key(|&(k, _)| k);
    let mut result: Vec<StateTransition> = Vec::new();
    for ((ns, pts), prob) in raw {
        if let Some(last) = result.last_mut() {
            if last.next_state == ns && last.points == pts {
                last.prob += prob;
                continue;
            }
        }
        result.push(StateTransition {
            next_state: ns,
            points: pts,
            prob,
        });
    }
    result
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
    fn test_transition_probabilities_sum_to_one() {
        let ctx = make_ctx();
        let sv = ctx.state_values.as_slice();

        // Test several states
        let test_cases = [
            (0, 0),                             // game start
            (0, (1 << CATEGORY_COUNT) - 1 - 1), // all but Ones
            (30, 0b111111),                     // all upper scored, up=30
        ];

        for &(up, scored) in &test_cases {
            let transitions = compute_transitions(&ctx, sv, up, scored, 0.0);
            let total_prob: f64 = transitions.iter().map(|t| t.prob).sum();
            assert!(
                (total_prob - 1.0).abs() < 1e-6,
                "Transitions from (up={}, scored=0x{:x}) sum to {} (should be 1.0)",
                up,
                scored,
                total_prob
            );
        }
    }

    #[test]
    fn test_single_category_remaining() {
        let ctx = make_ctx();
        let sv = ctx.state_values.as_slice();

        // Only Yatzy remaining
        let scored = ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_YATZY);
        let transitions = compute_transitions(&ctx, sv, 0, scored, 0.0);

        let total_prob: f64 = transitions.iter().map(|t| t.prob).sum();
        assert!((total_prob - 1.0).abs() < 1e-6);

        // All transitions should go to the same next scored state (all scored)
        let all_scored = (1 << CATEGORY_COUNT) - 1;
        for t in &transitions {
            let next_scored = (t.next_state as usize) / STATE_STRIDE;
            assert_eq!(next_scored, all_scored);
        }
    }
}
