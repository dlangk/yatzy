//! Forward density evolution — propagate exact score distributions through 15 turns.
//!
//! Starting from state (0, 0) with accumulated score 0 and probability 1.0,
//! push the distribution forward through each turn using the transition
//! probabilities from [`super::transitions::compute_transitions`].
//!
//! Uses dense f64 arrays for score distributions and parallelizes both the
//! transition computation and the merge phase.

use std::collections::HashMap;
use std::time::Instant;

use rayon::prelude::*;

use crate::constants::*;
use crate::types::YatzyContext;

use super::transitions::{compute_transitions, compute_transitions_oracle};

/// Maximum possible score (374) padded to 384 for alignment.
const MAX_SCORE: usize = 384;

/// Result of exact density evolution: PMF + summary statistics.
pub struct DensityResult {
    /// Exact PMF: (score, probability) pairs, sorted by score.
    pub pmf: Vec<(i32, f64)>,
    /// Exact mean score.
    pub mean: f64,
    /// Exact variance.
    pub variance: f64,
    /// Exact standard deviation.
    pub std_dev: f64,
    /// Percentiles: p1, p5, p10, p25, p50, p75, p90, p95, p99.
    pub percentiles: HashMap<String, i32>,
}

/// Dense score distribution: probability for each score bin 0..MAX_SCORE.
type ScoreDist = Vec<f64>;

/// Active state entry: (state_index, dense score distribution).
type StateEntry = (u32, ScoreDist);

/// Run exact density evolution using the precomputed oracle (θ=0 only).
///
/// Same algorithm as `density_evolution` but uses O(1) oracle lookups
/// instead of recomputing Group 6 + Group 5/3 per state.
pub fn density_evolution_oracle(
    ctx: &YatzyContext,
    oracle: &crate::types::PolicyOracle,
) -> DensityResult {
    density_evolution_inner(ctx, 0.0, Some(oracle))
}

/// Run exact density evolution for a given theta.
///
/// Propagates P(state, accumulated_score) through 15 turns, applying the
/// optimal policy from the precomputed strategy table.
///
/// Returns the exact score PMF with zero variance.
pub fn density_evolution(ctx: &YatzyContext, theta: f32) -> DensityResult {
    density_evolution_inner(ctx, theta, None)
}

fn density_evolution_inner(
    ctx: &YatzyContext,
    theta: f32,
    oracle: Option<&crate::types::PolicyOracle>,
) -> DensityResult {
    let sv = ctx.state_values.as_slice();
    let t_total = Instant::now();

    // Initialize: state (0, 0) with accumulated score 0, probability 1.0
    let start_si = state_index(0, 0) as u32;
    let mut start_dist = vec![0.0f64; MAX_SCORE];
    start_dist[0] = 1.0;
    let mut active_states: Vec<StateEntry> = vec![(start_si, start_dist)];

    // 15 turns of forward propagation
    for turn in 0..15 {
        let t_turn = Instant::now();
        let num_scored = turn as u32;

        let total_bins: usize = active_states
            .iter()
            .map(|(_, sd)| sd.iter().filter(|&&p| p > 0.0).count())
            .sum();
        println!(
            "  Turn {:2}: {} active states, {} score bins",
            turn,
            active_states.len(),
            total_bins,
        );

        // Phase 1: Compute transitions for each active state in parallel.
        // Also compute max_nonzero index for each source distribution.
        let transitions_and_bounds: Vec<(Vec<super::transitions::StateTransition>, usize)> =
            active_states
                .par_iter()
                .map(|(si, sd)| {
                    let scored = (*si as usize) / STATE_STRIDE;
                    let up = (*si as usize) % STATE_STRIDE;

                    debug_assert_eq!(
                        (scored as u32).count_ones(),
                        num_scored,
                        "State 0x{:x} has {} scored categories, expected {}",
                        scored,
                        (scored as u32).count_ones(),
                        num_scored
                    );

                    let trans = if let Some(orc) = oracle {
                        compute_transitions_oracle(ctx, orc, up as i32, scored as i32)
                    } else {
                        compute_transitions(ctx, sv, up as i32, scored as i32, theta)
                    };
                    let max_i = sd.iter().rposition(|&p| p > 0.0).unwrap_or(0);
                    (trans, max_i)
                })
                .collect();

        // Phase 2: Build destination index — group (src_idx, trans_idx) by dest state.
        let mut dest_map: HashMap<u32, Vec<(usize, usize)>> = HashMap::new();
        for (src_idx, (trans, _)) in transitions_and_bounds.iter().enumerate() {
            for (t_idx, t) in trans.iter().enumerate() {
                dest_map
                    .entry(t.next_state)
                    .or_default()
                    .push((src_idx, t_idx));
            }
        }

        // Phase 3: Parallel merge — each destination is independent.
        let dest_entries: Vec<(u32, Vec<(usize, usize)>)> = dest_map.into_iter().collect();

        let next_states: Vec<StateEntry> = dest_entries
            .par_iter()
            .map(|(dest_si, contribs)| {
                let mut dense = vec![0.0f64; MAX_SCORE];

                for &(src_idx, t_idx) in contribs {
                    let src_dist = &active_states[src_idx].1;
                    let (ref trans, max_i) = transitions_and_bounds[src_idx];
                    let t = &trans[t_idx];
                    let offset = t.points as usize;
                    let prob = t.prob;

                    // Shift-and-add using only the non-zero range
                    for i in 0..=max_i {
                        let p = unsafe { *src_dist.get_unchecked(i) };
                        if p > 0.0 {
                            unsafe {
                                *dense.get_unchecked_mut(i + offset) += prob * p;
                            }
                        }
                    }
                }

                (*dest_si, dense)
            })
            .collect();

        active_states = next_states;

        println!(
            "          → {} next states in {:.2}s",
            active_states.len(),
            t_turn.elapsed().as_secs_f64()
        );
    }

    // All categories are scored. Apply upper bonus and collect final PMF.
    let all_scored_mask = (1 << CATEGORY_COUNT) - 1;

    let mut final_pmf = vec![0.0f64; MAX_SCORE + 50]; // +50 for bonus
    let mut total_prob = 0.0;

    for (si, score_dist) in &active_states {
        let scored = (*si as usize) / STATE_STRIDE;
        let up = (*si as usize) % STATE_STRIDE;

        debug_assert_eq!(
            scored, all_scored_mask,
            "After 15 turns, scored should be all_scored_mask"
        );

        let bonus: usize = if up >= 63 { 50 } else { 0 };

        for (acc_score, &prob) in score_dist.iter().enumerate() {
            if prob > 0.0 {
                final_pmf[acc_score + bonus] += prob;
                total_prob += prob;
            }
        }
    }

    println!("  Total probability: {:.15} (should be 1.0)", total_prob);

    // Convert to sorted PMF (skip zero entries)
    let mut pmf: Vec<(i32, f64)> = final_pmf
        .iter()
        .enumerate()
        .filter(|(_, &p)| p > 0.0)
        .map(|(s, &p)| (s as i32, p))
        .collect();
    pmf.sort_by_key(|&(score, _)| score);

    // Compute statistics
    let mean: f64 = pmf.iter().map(|&(s, p)| s as f64 * p).sum();
    let variance: f64 = pmf
        .iter()
        .map(|&(s, p)| (s as f64 - mean).powi(2) * p)
        .sum();
    let std_dev = variance.sqrt();

    // Compute percentiles from CDF
    let percentile_keys = [1, 5, 10, 25, 50, 75, 90, 95, 99];
    let mut percentiles = HashMap::new();
    let mut cum_prob = 0.0;
    let mut pct_idx = 0;

    for &(score, prob) in &pmf {
        cum_prob += prob;
        while pct_idx < percentile_keys.len() && cum_prob >= percentile_keys[pct_idx] as f64 / 100.0
        {
            percentiles.insert(format!("p{}", percentile_keys[pct_idx]), score);
            pct_idx += 1;
        }
    }

    let elapsed = t_total.elapsed().as_secs_f64();
    println!(
        "  Density evolution complete: mean={:.6}, std={:.4}, {:.1}s",
        mean, std_dev, elapsed
    );

    DensityResult {
        pmf,
        mean,
        variance,
        std_dev,
        percentiles,
    }
}

/// Run exact density evolution starting from an arbitrary mid-game state (oracle required).
///
/// Given a specific (upper_score, scored_categories, accumulated_score), propagates
/// the distribution forward through the remaining turns. Fewer remaining turns = faster:
/// ~0.5s for 5 turns left, ~3s for full game.
///
/// Returns the same DensityResult as `density_evolution_oracle` but starting from mid-game.
pub fn density_evolution_from_state(
    ctx: &YatzyContext,
    oracle: &crate::types::PolicyOracle,
    upper_score: usize,
    scored_categories: usize,
    accumulated_score: usize,
) -> DensityResult {
    let start_si = state_index(upper_score, scored_categories) as u32;
    let turn = (scored_categories as u32).count_ones() as usize;

    // Initialize: single entry at the given state with the given accumulated score
    let mut start_dist = vec![0.0f64; MAX_SCORE];
    if accumulated_score < MAX_SCORE {
        start_dist[accumulated_score] = 1.0;
    }
    let mut active_states: Vec<StateEntry> = vec![(start_si, start_dist)];

    // Forward propagation for remaining turns
    for t in turn..15 {
        let num_scored = t as u32;

        let transitions_and_bounds: Vec<(Vec<super::transitions::StateTransition>, usize)> =
            active_states
                .par_iter()
                .map(|(si, sd)| {
                    let scored = (*si as usize) / STATE_STRIDE;
                    let up = (*si as usize) % STATE_STRIDE;

                    debug_assert_eq!(
                        (scored as u32).count_ones(),
                        num_scored,
                        "State 0x{:x} has {} scored categories, expected {}",
                        scored,
                        (scored as u32).count_ones(),
                        num_scored
                    );

                    let trans = compute_transitions_oracle(ctx, oracle, up as i32, scored as i32);
                    let max_i = sd.iter().rposition(|&p| p > 0.0).unwrap_or(0);
                    (trans, max_i)
                })
                .collect();

        let mut dest_map: HashMap<u32, Vec<(usize, usize)>> = HashMap::new();
        for (src_idx, (trans, _)) in transitions_and_bounds.iter().enumerate() {
            for (t_idx, t) in trans.iter().enumerate() {
                dest_map
                    .entry(t.next_state)
                    .or_default()
                    .push((src_idx, t_idx));
            }
        }

        let dest_entries: Vec<(u32, Vec<(usize, usize)>)> = dest_map.into_iter().collect();

        let next_states: Vec<StateEntry> = dest_entries
            .par_iter()
            .map(|(dest_si, contribs)| {
                let mut dense = vec![0.0f64; MAX_SCORE];

                for &(src_idx, t_idx) in contribs {
                    let src_dist = &active_states[src_idx].1;
                    let (ref trans, max_i) = transitions_and_bounds[src_idx];
                    let t = &trans[t_idx];
                    let offset = t.points as usize;
                    let prob = t.prob;

                    for i in 0..=max_i {
                        let p = unsafe { *src_dist.get_unchecked(i) };
                        if p > 0.0 {
                            unsafe {
                                *dense.get_unchecked_mut(i + offset) += prob * p;
                            }
                        }
                    }
                }

                (*dest_si, dense)
            })
            .collect();

        active_states = next_states;
    }

    // Apply upper bonus and collect final PMF
    let mut final_pmf = vec![0.0f64; MAX_SCORE + 50];
    let mut total_prob = 0.0;

    for (si, score_dist) in &active_states {
        let up = (*si as usize) % STATE_STRIDE;
        let bonus: usize = if up >= 63 { 50 } else { 0 };

        for (acc_score, &prob) in score_dist.iter().enumerate() {
            if prob > 0.0 {
                final_pmf[acc_score + bonus] += prob;
                total_prob += prob;
            }
        }
    }

    // Convert to sorted PMF
    let mut pmf: Vec<(i32, f64)> = final_pmf
        .iter()
        .enumerate()
        .filter(|(_, &p)| p > 0.0)
        .map(|(s, &p)| (s as i32, p))
        .collect();
    pmf.sort_by_key(|&(score, _)| score);

    // Compute statistics
    let mean: f64 = pmf.iter().map(|&(s, p)| s as f64 * p).sum();
    let variance: f64 = pmf
        .iter()
        .map(|&(s, p)| (s as f64 - mean).powi(2) * p)
        .sum();
    let std_dev = variance.sqrt();

    // Normalize if needed (should be ~1.0)
    let norm = if total_prob > 0.0 { total_prob } else { 1.0 };

    // Compute percentiles from CDF
    let percentile_keys = [1, 5, 10, 25, 50, 75, 90, 95, 99];
    let mut percentiles = HashMap::new();
    let mut cum_prob = 0.0;
    let mut pct_idx = 0;

    for &(score, prob) in &pmf {
        cum_prob += prob / norm;
        while pct_idx < percentile_keys.len() && cum_prob >= percentile_keys[pct_idx] as f64 / 100.0
        {
            percentiles.insert(format!("p{}", percentile_keys[pct_idx]), score);
            pct_idx += 1;
        }
    }

    DensityResult {
        pmf,
        mean,
        variance,
        std_dev,
        percentiles,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0_tables;
    use crate::state_computation::compute_all_state_values;

    fn make_ctx_with_sv() -> Box<YatzyContext> {
        let mut ctx = YatzyContext::new_boxed();
        phase0_tables::precompute_lookup_tables(&mut ctx);
        compute_all_state_values(&mut ctx);
        ctx
    }

    /// Verify exact mean matches sv[state_index(0, 0)].
    /// Run with: cargo test density::forward -- --ignored --nocapture
    #[test]
    #[ignore] // ~6 min: full 15-turn density evolution
    fn test_density_mean_matches_ev() {
        let ctx = make_ctx_with_sv();
        let expected_ev = ctx.state_values.as_slice()[state_index(0, 0)] as f64;

        let result = density_evolution(&ctx, 0.0);

        let diff = (result.mean - expected_ev).abs();
        println!(
            "Density mean: {:.6}, EV table: {:.6}, diff: {:.10}",
            result.mean, expected_ev, diff
        );

        assert!(
            diff < 0.01,
            "Density mean {:.6} should match EV {:.6} (diff={:.10})",
            result.mean,
            expected_ev,
            diff
        );
    }

    /// Verify PMF sums to 1.0.
    /// Run with: cargo test density::forward -- --ignored --nocapture
    #[test]
    #[ignore] // ~6 min: full 15-turn density evolution
    fn test_density_probability_conservation() {
        let ctx = make_ctx_with_sv();
        let result = density_evolution(&ctx, 0.0);

        let total: f64 = result.pmf.iter().map(|&(_, p)| p).sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "PMF should sum to 1.0, got {}",
            total
        );
    }
}
