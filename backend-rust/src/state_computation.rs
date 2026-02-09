use std::time::Instant;

use rayon::prelude::*;

use crate::constants::*;
use crate::storage::{load_all_state_values, save_all_state_values};
use crate::types::{YatzyContext, YatzyState};
use crate::widget_solver::compute_expected_state_value;

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

/// Compute expected values for all possible game states using backward induction.
pub fn compute_all_state_values(ctx: &mut YatzyContext) {
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

    // Try to load consolidated file first
    let consolidated_file = "data/all_states.bin";
    if load_all_state_values(ctx, consolidated_file) {
        println!("Loaded pre-computed states from consolidated file");
        return;
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

        // Build state list
        let mut state_list: Vec<(usize, i32)> = Vec::new();
        for scored in 0..(1u32 << CATEGORY_COUNT) {
            if scored.count_ones() == num_scored as u32 {
                let upper_mask = (scored & 0x3F) as usize;
                for up in 0..=63usize {
                    if ctx.reachable[upper_mask][up] {
                        state_list.push((up, scored as i32));
                    }
                }
            }
        }

        // Parallel computation: collect results then scatter back
        // We need a read-only reference for the parallel computation
        let ctx_ref = &*ctx;
        let results: Vec<(usize, f32)> = state_list
            .par_iter()
            .map(|&(up, scored)| {
                let state = YatzyState {
                    upper_score: up as i32,
                    scored_categories: scored,
                };
                let ev = compute_expected_state_value(ctx_ref, &state);
                (state_index(up, scored as usize), ev as f32)
            })
            .collect();

        // Scatter results back to state_values
        let state_values = ctx.state_values.as_mut_slice();
        for (idx, val) in &results {
            state_values[*idx] = *val;
        }

        let pre_loop_completed = progress.completed_states;
        progress.completed_states = pre_loop_completed + state_list.len() as i32;
        progress.time_per_level[num_scored as usize] = total_start.elapsed().as_secs_f64();

        let level_dur = level_start.elapsed().as_secs_f64();
        println!(
            "\nLevel {} completed in {:.2} seconds ({:.0} states/sec)",
            num_scored,
            level_dur,
            progress.states_per_level[num_scored as usize] as f64 / level_dur
        );
    }

    println!("\n\n=== Computation Complete ===");

    println!("\nSaving consolidated state file...");
    save_all_state_values(ctx, consolidated_file);

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
