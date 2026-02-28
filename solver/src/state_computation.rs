//! Phase 2: Backward induction — compute E_table[S] for all reachable states.
//!
//! Implements pseudocode `COMPUTE_OPTIMAL_STRATEGY`. Processes states in decreasing
//! order of |C| (number of scored categories), from |C|=14 down to |C|=0. Terminal
//! states (|C|=15) are initialized in Phase 0.
//!
//! Each level is parallelized with rayon `par_iter`. Within each state, SOLVE_WIDGET
//! reads only from successor states (|C|+1), which are already computed — the same
//! guarantee that enables parallelism in the pseudocode.
//!
//! ## Unsafe writes
//!
//! Each (upper_score, scored_categories) maps to a unique `state_index`, so parallel
//! workers never write to the same memory location. We use `AtomicPtr` + unsafe raw
//! pointer writes to avoid the overhead
//! of `collect()` + scatter.

use std::time::Instant;

use rayon::prelude::*;

use crate::batched_solver::{
    build_oracle_for_scored_mask, precompute_exp_scores, solve_widget_batched,
    solve_widget_batched_max, solve_widget_batched_risk, solve_widget_batched_utility,
    BatchedBuffers, ExpScores,
};
use crate::constants::*;
use crate::storage::{load_all_state_values, save_all_state_values, state_file_path};
use crate::types::{PolicyOracle, YatzyContext};

/// Maximum |θ| for utility-domain solver. Beyond this, f32 mantissa erasure
/// causes precision loss (e^(θ×100) > 2^24 when θ > 0.166).
/// For |θ| > 0.15, fall back to the log-domain (LSE) solver.
const UTILITY_THETA_LIMIT: f32 = 0.15;

/// Progress tracker for the Phase 2 computation loop.
struct ComputeProgress {
    total_states: i32,
    completed_states: i32,
    start_time: Instant,
    last_report_time: Instant,
    states_per_level: [i32; 16],
    time_per_level: [f64; 16],
}

impl ComputeProgress {
    fn new(ctx: &YatzyContext) -> Self {
        let now = Instant::now();
        let mut progress = ComputeProgress {
            total_states: 0,
            completed_states: 0,
            start_time: now,
            last_report_time: now,
            states_per_level: [0; 16],
            time_per_level: [0.0; 16],
        };

        for num_scored in 0..=15 {
            let mut states = 0;
            for scored in 0..(1u32 << CATEGORY_COUNT) {
                if scored.count_ones() == num_scored as u32 {
                    let upper_mask = (scored & 0x3F) as usize;
                    for up in 0..=63 {
                        if ctx.reachable[upper_mask][up] {
                            states += 1;
                        }
                    }
                }
            }
            progress.states_per_level[num_scored] = states;
            progress.total_states += states;
        }

        progress
    }

    fn print_progress(&mut self, current_level: i32) {
        let now = Instant::now();
        if self.completed_states < self.total_states
            && now.duration_since(self.last_report_time).as_secs_f64() < 0.5
        {
            return;
        }
        self.last_report_time = now;

        let elapsed = now.duration_since(self.start_time).as_secs_f64();
        let pct = self.completed_states as f64 / self.total_states as f64 * 100.0;
        let rate = self.completed_states as f64 / elapsed;
        let eta = (self.total_states - self.completed_states) as f64 / rate;

        print!(
            "\rProgress: {}/{} states ({:.1}%) | Level: {} | Elapsed: {:.1}s | Rate: {:.0} states/s | ETA: {:.1}s     ",
            self.completed_states, self.total_states, pct, current_level, elapsed, rate, eta
        );
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
}

/// Compute E_table[S] and optionally build the PolicyOracle.
///
/// When `build_oracle` is true and θ=0 (EV mode), also records every argmax
/// decision into three flat Vec<u8> arrays (~3.17 GB). The oracle enables
/// O(1) policy lookups in forward passes (MC simulation, density evolution).
pub fn compute_all_state_values_with_oracle(
    ctx: &mut YatzyContext,
    build_oracle: bool,
) -> Option<PolicyOracle> {
    let should_build = build_oracle && ctx.theta == 0.0 && !ctx.max_policy;
    let mut oracle = if should_build {
        println!("Oracle building enabled (~3.17 GB allocation)...");
        Some(PolicyOracle::new())
    } else {
        None
    };
    compute_all_state_values_inner(ctx, &mut oracle);
    oracle
}

/// Compute E_table[S] for all reachable game states using backward induction.
///
/// Pseudocode: `COMPUTE_OPTIMAL_STRATEGY` — iterates num_filled from 14 down to 0,
/// calling SOLVE_WIDGET for each valid state at that level. Level 15 (terminal) is
/// already initialized.
///
/// Results are saved to `data/all_states.bin`.
pub fn compute_all_state_values(ctx: &mut YatzyContext) {
    compute_all_state_values_inner(ctx, &mut None);
}

/// Like `compute_all_state_values` but always recomputes (skips cache check).
/// Used by benchmarks to measure actual computation time.
pub fn compute_all_state_values_nocache(ctx: &mut YatzyContext) {
    compute_all_state_values_nocache_inner(ctx, &mut None);
}

fn compute_all_state_values_nocache_inner(
    ctx: &mut YatzyContext,
    oracle: &mut Option<PolicyOracle>,
) {
    compute_all_state_values_impl(ctx, oracle, true);
}

fn compute_all_state_values_inner(ctx: &mut YatzyContext, oracle: &mut Option<PolicyOracle>) {
    compute_all_state_values_impl(ctx, oracle, false);
}

fn compute_all_state_values_impl(
    ctx: &mut YatzyContext,
    oracle: &mut Option<PolicyOracle>,
    skip_cache: bool,
) {
    let mut progress = ComputeProgress::new(ctx);

    println!("=== Starting State Value Computation ===");
    println!(
        "Total states to compute: {} (after reachability pruning)",
        progress.total_states
    );
    println!("States per level:");
    for i in 0..=15 {
        println!("  Level {:2}: {:6} states", i, progress.states_per_level[i]);
    }
    println!();

    let total_start = Instant::now();

    // Try to load consolidated file first (skipped for benchmarks)
    if !skip_cache {
        let consolidated_file = if ctx.max_policy {
            "data/all_states_max.bin".to_string()
        } else {
            state_file_path(ctx.theta)
        };
        if load_all_state_values(ctx, &consolidated_file) {
            println!("Loaded pre-computed states from consolidated file");
            return;
        }
    }

    // Process states level by level, from game end (14) to game start (0).
    // Level 15 (terminal) is already initialized by precompute_lookup_tables.
    for num_scored in (0..=14).rev() {
        let level_start = Instant::now();

        progress.print_progress(num_scored);

        println!(
            "\nComputing level {} ({} states)...",
            num_scored, progress.states_per_level[num_scored as usize]
        );

        // Build list of scored_categories masks at this level.
        // The batched solver processes all 64 upper-score variants of each mask
        // simultaneously, converting SpMV to SpMM for cache-friendly access.
        let mut scored_masks: Vec<i32> = Vec::new();
        let mut state_count: usize = 0;
        for scored in 0..(1u32 << CATEGORY_COUNT) {
            if scored.count_ones() == num_scored as u32 {
                let upper_mask = (scored & 0x3F) as usize;
                let has_reachable = (0..=63usize).any(|up| ctx.reachable[upper_mask][up]);
                if has_reachable {
                    scored_masks.push(scored as i32);
                    state_count += (0..=63usize)
                        .filter(|&up| ctx.reachable[upper_mask][up])
                        .count();
                }
            }
        }

        // Parallel computation: each scored mask processes all 64 ups at once.
        // Safety: each (up, scored) pair maps to a unique state_index,
        // so no two threads write to the same location.
        let sv_ptr = ctx.state_values.as_mut_slice().as_mut_ptr();

        // AtomicPtr wrapper to satisfy Send+Sync for rayon's for_each.
        let sv_atomic = std::sync::atomic::AtomicPtr::new(sv_ptr);

        let ctx_ref = &*ctx;
        let use_max = ctx_ref.max_policy;
        let use_utility = ctx_ref.theta != 0.0 && ctx_ref.theta.abs() <= UTILITY_THETA_LIMIT;
        let use_risk = ctx_ref.theta != 0.0 && !use_utility;

        // Precompute e^(θ·score) table for utility-domain solver (shared across threads)
        let exp_scores: Option<Box<ExpScores>> = if use_utility {
            Some(precompute_exp_scores(ctx_ref, ctx_ref.theta))
        } else {
            None
        };
        let exp_scores_ref = exp_scores.as_deref();
        let minimize = ctx_ref.theta < 0.0;

        // Extract sv slice for reading (safe: writes go to different indices than reads,
        // since we only write to states at level num_scored and read from num_scored+1).
        let sv_slice = ctx_ref.state_values.as_slice();

        // Oracle pointers (if building). Each (state_index, ds_index) is unique,
        // so parallel writes are safe (same argument as sv writes).
        let oracle_cat_ptr = oracle
            .as_mut()
            .map(|o| std::sync::atomic::AtomicPtr::new(o.cat_mut().as_mut_ptr()));
        let oracle_keep1_ptr = oracle
            .as_mut()
            .map(|o| std::sync::atomic::AtomicPtr::new(o.keep1_mut().as_mut_ptr()));
        let oracle_keep2_ptr = oracle
            .as_mut()
            .map(|o| std::sync::atomic::AtomicPtr::new(o.keep2_mut().as_mut_ptr()));
        let building_oracle = oracle_cat_ptr.is_some();

        scored_masks
            .par_iter()
            .for_each_init(BatchedBuffers::new, |bufs, &scored| {
                // Build oracle decisions if enabled (runs the solver internally)
                let oracle_slice = if building_oracle {
                    Some(build_oracle_for_scored_mask(
                        ctx_ref, sv_slice, scored, bufs,
                    ))
                } else {
                    None
                };

                let results = if building_oracle {
                    // Oracle builder already computed groups 6/5/3 — recompute group 1 only.
                    // bufs.e0 now holds the final e[r][up] after group 3. Just sum group 1.
                    crate::batched_solver::batched_group1_pub(ctx_ref, &bufs.e0)
                } else if use_max {
                    solve_widget_batched_max(ctx_ref, sv_slice, scored, bufs)
                } else if use_utility {
                    solve_widget_batched_utility(
                        ctx_ref,
                        sv_slice,
                        scored,
                        bufs,
                        exp_scores_ref.unwrap(),
                        minimize,
                    )
                } else if use_risk {
                    solve_widget_batched_risk(ctx_ref, sv_slice, scored, bufs)
                } else {
                    solve_widget_batched(ctx_ref, sv_slice, scored, bufs)
                };

                let ptr = sv_atomic.load(std::sync::atomic::Ordering::Relaxed);
                let upper_mask = (scored & 0x3F) as usize;
                for up in 0..=63 {
                    if ctx_ref.reachable[upper_mask][up] {
                        unsafe {
                            *ptr.add(state_index(up, scored as usize)) = results[up];
                        }
                    }
                }
                // Fill topological padding: indices 64..127 get the capped value (index 63).
                // See constants.rs STATE_STRIDE — padding enables branchless upper-category access:
                // sv[base + up + scr] reads correct values even when up + scr > 63.
                let capped_val = results[63];
                let base = state_index(0, scored as usize);
                for pad in 64..STATE_STRIDE {
                    unsafe {
                        *ptr.add(base + pad) = capped_val;
                    }
                }

                // Scatter oracle slice into the global oracle arrays
                if let Some(ref os) = oracle_slice {
                    let cat_ptr = oracle_cat_ptr
                        .as_ref()
                        .unwrap()
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let k1_ptr = oracle_keep1_ptr
                        .as_ref()
                        .unwrap()
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let k2_ptr = oracle_keep2_ptr
                        .as_ref()
                        .unwrap()
                        .load(std::sync::atomic::Ordering::Relaxed);

                    // Transpose from batched [ds*64+up] layout to oracle's [si*252+ds] layout
                    // for sequential forward-simulation access.
                    for up in 0..64 {
                        let si = state_index(up, scored as usize);
                        for ds in 0..NUM_DICE_SETS {
                            let oracle_idx = si * NUM_DICE_SETS + ds;
                            let slice_idx = ds * 64 + up;
                            unsafe {
                                *cat_ptr.add(oracle_idx) = os.cat[slice_idx];
                                *k1_ptr.add(oracle_idx) = os.keep1[slice_idx];
                                *k2_ptr.add(oracle_idx) = os.keep2[slice_idx];
                            }
                        }
                    }
                }
            });

        let pre_loop_completed = progress.completed_states;
        progress.completed_states = pre_loop_completed + state_count as i32;
        progress.time_per_level[num_scored as usize] = total_start.elapsed().as_secs_f64();

        let level_dur = level_start.elapsed().as_secs_f64();
        println!(
            "\nLevel {} completed in {:.2} seconds ({:.0} states/sec)",
            num_scored,
            level_dur,
            progress.states_per_level[num_scored as usize] as f64 / level_dur
        );

        #[cfg(feature = "timing")]
        {
            crate::widget_solver::print_timing();
            crate::widget_solver::reset_timing();
        }
    }

    println!("\n\n=== Computation Complete ===");

    // Convert utility to log-domain: downstream consumers compute CE = L(S)/theta.
    // U(S) = E[e^(θ·remaining)|S] → L(S) = ln(U(S)), giving consistent format for all θ.
    let was_utility = ctx.theta != 0.0 && ctx.theta.abs() <= UTILITY_THETA_LIMIT && !ctx.max_policy;
    if was_utility {
        let t_conv = Instant::now();
        let sv = ctx.state_values.as_mut_slice();
        for i in 0..NUM_STATES {
            let u = sv[i];
            if u > 0.0 {
                sv[i] = u.ln();
            }
            // u == 0.0 maps to -inf in log domain, which is correct
            // (unreachable states stay as initialized)
        }
        println!(
            "Converted utility→log domain ({} states) in {:.2} ms",
            NUM_STATES,
            t_conv.elapsed().as_secs_f64() * 1000.0
        );
    }

    if !skip_cache {
        let save_file = if ctx.max_policy {
            "data/all_states_max.bin".to_string()
        } else {
            state_file_path(ctx.theta)
        };
        println!("\nSaving consolidated state file...");
        save_all_state_values(ctx, &save_file);
    }

    let total_time = total_start.elapsed().as_secs_f64();
    println!("\nTotal computation time: {:.2} seconds", total_time);
    println!(
        "Average processing rate: {:.0} states/second",
        progress.total_states as f64 / total_time
    );

    println!("\nDetailed timing breakdown by level:");
    println!("Level | States  | Time (s) | Rate (states/s)");
    println!("------|---------|----------|----------------");

    let mut prev_time = 0.0;
    for level in (0..=15).rev() {
        let level_time = progress.time_per_level[level] - prev_time;
        if level_time > 0.0 {
            println!(
                "  {:2}  | {:6}  | {:7.2}  | {:6.0}",
                level,
                progress.states_per_level[level],
                level_time,
                progress.states_per_level[level] as f64 / level_time
            );
        }
        prev_time = progress.time_per_level[level];
    }
}
