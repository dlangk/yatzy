//! Backward policy-evaluation: expected final score under the theta-policy.
//!
//! Companion to `state_computation.rs`. Where that maximizes (EV) or computes
//! log-utility, this follows a fixed theta-policy (decisions read from `sv_theta`)
//! and accumulates the expected SCORE, giving V_pi_theta(S) = mean final score if
//! you play the theta-optimal policy from S. At theta=0, V_pi0 == V*.

use std::sync::atomic::{AtomicPtr, Ordering};

use rayon::prelude::*;

use crate::constants::*;
use crate::density::transitions::expected_score_under_policy;
use crate::types::YatzyContext;

/// Build V_πθ(S) for all state slots via backward induction, following the
/// theta-policy encoded by `sv_theta` (the decision table already shipped for
/// this θ) while accumulating the expected final SCORE rather than a utility.
///
/// At θ=0, `sv_theta` is V* and the result reproduces `all_states.bin` exactly
/// (the regression gate `test_policy_mean_theta0_matches_v_star`).
pub fn compute_policy_mean_table(ctx: &YatzyContext, sv_theta: &[f32], theta: f32) -> Vec<f32> {
    let mut mean = vec![0.0f32; NUM_STATES];
    let all_scored = ((1u32 << CATEGORY_COUNT) - 1) as usize;

    // Terminal level (all 15 scored): value = upper bonus only.
    for up in 0..=63usize {
        mean[state_index(up, all_scored)] = if up >= 63 { 50.0 } else { 0.0 };
    }
    // Topological padding (up 64..127) = capped value at up=63.
    {
        let capped = mean[state_index(63, all_scored)];
        let base = state_index(0, all_scored);
        for pad in 64..STATE_STRIDE {
            mean[base + pad] = capped;
        }
    }

    // Levels 14..0: value = expected_score_under_policy reading mean at level+1.
    for num_scored in (0..=14).rev() {
        // Collect reachable scored masks at this level.
        let mut scored_masks: Vec<i32> = Vec::new();
        for scored in 0..(1u32 << CATEGORY_COUNT) {
            if scored.count_ones() == num_scored as u32 {
                let upper_mask = (scored & 0x3F) as usize;
                if (0..=63usize).any(|up| ctx.reachable[upper_mask][up]) {
                    scored_masks.push(scored as i32);
                }
            }
        }

        // Mirror state_computation.rs's aliasing pattern EXACTLY: take the raw write
        // pointer first (mutable borrow ends immediately), then hold ONE immutable
        // read slice over the whole array for the readers. Do NOT reconstruct a
        // slice from the write pointer inside the closure.
        let ptr = AtomicPtr::new(mean.as_mut_ptr());
        let mean_slice: &[f32] = &mean;
        // SAFETY of reads: expected_score_under_policy only reads mean at successor
        // states (level num_scored+1), already written. Writes go to distinct
        // (up, scored) indices per mask; each mask handled by one worker, so no two
        // workers write the same index and no write overlaps another worker's reads.
        scored_masks.par_iter().for_each(|&scored| {
            let upper_mask = (scored & 0x3F) as usize;
            let base = state_index(0, scored as usize);
            let p = ptr.load(Ordering::Relaxed);
            let mut capped = 0.0f32;
            for up in 0..=63usize {
                if ctx.reachable[upper_mask][up] {
                    let v = expected_score_under_policy(
                        ctx, sv_theta, mean_slice, up as i32, scored, theta,
                    );
                    // SAFETY: (up, scored) maps to a unique state_index; each
                    // scored mask in this par_iter is handled by exactly one
                    // worker, and distinct masks write distinct index ranges,
                    // so this write never races another worker's write or read.
                    unsafe {
                        *p.add(state_index(up, scored as usize)) = v;
                    }
                    if up == 63 {
                        capped = v;
                    }
                }
            }
            // Fill padding for this mask with the up=63 value (or the highest reachable up if 63 unreachable).
            if capped == 0.0 {
                // find highest reachable up as the capped value
                for up in (0..=63usize).rev() {
                    if ctx.reachable[upper_mask][up] {
                        // SAFETY: this index was just written above in this same
                        // worker's loop, so the read is to already-initialized,
                        // exclusively-owned (by this worker) memory.
                        capped = unsafe { *p.add(state_index(up, scored as usize)) };
                        break;
                    }
                }
            }
            for pad in 64..STATE_STRIDE {
                // SAFETY: base + pad are the topological padding slots for this
                // scored mask, exclusively owned by this worker (see above).
                unsafe {
                    *p.add(base + pad) = capped;
                }
            }
        });
    }

    mean
}

#[cfg(test)]
mod tests {
    use crate::constants::*;

    #[test]
    #[ignore] // ~minutes: full backward mean induction at theta=0
    fn test_policy_mean_theta0_matches_v_star() {
        // V_pi0 == V*, so building the mean table with the EV table as the decision
        // table must reproduce all_states.bin at every reachable state.
        let mut ctx = crate::types::YatzyContext::new_boxed();
        crate::phase0_tables::precompute_lookup_tables(&mut ctx);
        crate::state_computation::compute_all_state_values(&mut ctx);
        let v_star = ctx.state_values.as_slice().to_vec();

        let mean = super::compute_policy_mean_table(&ctx, &v_star, 0.0);

        let all_scored = (1u32 << CATEGORY_COUNT) - 1;
        let mut max_abs = 0.0f32;
        for scored in 0..(1u32 << CATEGORY_COUNT) {
            let upper_mask = (scored & 0x3F) as usize;
            for up in 0..=63usize {
                if ctx.reachable[upper_mask][up] {
                    let i = state_index(up, scored as usize);
                    max_abs = max_abs.max((mean[i] - v_star[i]).abs());
                }
            }
            let _ = all_scored;
        }
        assert!(
            max_abs < 0.05,
            "max |V_pi0 - V*| = {} over reachable states",
            max_abs
        );
    }
}
