//! Compute the unconstrained clairvoyant upper bound on sequential win rate.
//!
//! Solves the 3D threshold tablebase: for each target score T, find the policy
//! that maximizes P(final_score > T). The state space is the standard
//! (upper_score, scored_categories) × target dimension.
//!
//! Each state stores [f32; T_SIZE] where entry t = P(remaining_score > t | optimal play).
//! At terminal states this is a step function at the bonus value.
//!
//! The backward induction follows the same 6-group structure as the scalar solver
//! but operates on arrays. No SIMD intrinsics; relies on autovectorization.

use std::time::Instant;

use rayon::prelude::*;

use yatzy::constants::{self, *};
use yatzy::game_mechanics::update_upper_score;
use yatzy::phase0_tables;
use yatzy::types::YatzyContext;

/// Target score range. Padded to 384 for SIMD alignment (actual max ~374).
const T_SIZE: usize = 384;

/// Wrapper to send raw pointer across threads (we guarantee no aliased writes).
struct SendPtr(*const f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// Main value table: V[state_index][t] = P(remaining_score > t | state, optimal play for target t).
/// 4,194,304 states × 384 targets × 4 bytes = ~6.1 GB.
struct ValueTable {
    data: Vec<f32>, // flat: data[state_index * T_SIZE + t]
}

impl ValueTable {
    fn new() -> Self {
        let n = NUM_STATES * T_SIZE;
        println!(
            "Allocating value table: {:.1} GB ({} × {} entries)",
            n as f64 * 4.0 / 1e9,
            NUM_STATES,
            T_SIZE
        );
        ValueTable {
            data: vec![0.0f32; n],
        }
    }

    #[inline]
    fn get(&self, si: usize) -> &[f32] {
        let start = si * T_SIZE;
        &self.data[start..start + T_SIZE]
    }

    #[inline]
    fn set(&mut self, si: usize, values: &[f32; T_SIZE]) {
        let start = si * T_SIZE;
        self.data[start..start + T_SIZE].copy_from_slice(values);
    }

    fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }
}

/// Per-thread working buffers for the 3D widget solver.
struct Buffers3D {
    /// Ping-pong: e0[ds][t] and e1[ds][t]
    e0: Vec<[f32; T_SIZE]>, // [252]
    e1: Vec<[f32; T_SIZE]>, // [252]
    /// Keep EVs: keep_ev[kid][t]
    keep_ev: Vec<[f32; T_SIZE]>, // [462]
    /// Output: one [f32; T_SIZE] per upper_score (0..64)
    results: Vec<[f32; T_SIZE]>, // [64]
}

impl Buffers3D {
    fn new() -> Self {
        Buffers3D {
            e0: vec![[0.0; T_SIZE]; 252],
            e1: vec![[0.0; T_SIZE]; 252],
            keep_ev: vec![[0.0; T_SIZE]; 462],
            results: vec![[0.0; T_SIZE]; 64],
        }
    }
}

/// Solve one scored_mask for all 64 upper scores simultaneously.
fn solve_3d_for_mask(
    ctx: &YatzyContext,
    vt_ptr: *const f32, // raw pointer to value table for reads
    scored: i32,
    bufs: &mut Buffers3D,
) {
    let kt = &ctx.keep_table;

    for up in 0..64usize {
        let upper_mask = (scored & 0x3F) as usize;
        if !ctx.reachable[upper_mask][up] {
            bufs.results[up] = [0.0; T_SIZE];
            continue;
        }

        // Group 6: for each dice set, find best category (element-wise max over targets)
        for ds_i in 0..252 {
            let mut best = [0.0f32; T_SIZE];

            for c in 0..CATEGORY_COUNT {
                if is_category_scored(scored, c) {
                    continue;
                }
                let scr = ctx.precomputed_scores[ds_i][c] as usize;
                let new_scored = scored | (1 << c);

                let new_up = if c < 6 {
                    update_upper_score(up as i32, c, scr as i32) as usize
                } else {
                    up
                };

                let si = state_index(new_up, new_scored as usize);
                let succ_start = si * T_SIZE;

                // V_next[t - scr]: shift the successor's win-prob array left by scr
                // If t < scr, we've exceeded target → P(win) = 1.0
                for t in 0..T_SIZE {
                    let val = if t < scr {
                        1.0 // remaining score includes scr which already exceeds target t
                    } else {
                        // SAFETY: si < NUM_STATES, (t-scr) < T_SIZE
                        unsafe { *vt_ptr.add(succ_start + t - scr) }
                    };
                    if val > best[t] {
                        best[t] = val;
                    }
                }
            }
            bufs.e0[ds_i] = best;
        }

        // Group 5: best keep with 1 reroll left
        // e1[ds][t] = max_keep Σ P(keep→ds') · e0[ds'][t]
        compute_max_keep_3d(ctx, &bufs.e0, &mut bufs.e1, &mut bufs.keep_ev);

        // Group 3: best keep with 2 rerolls left (initial roll)
        // e0[ds][t] = max_keep Σ P(keep→ds') · e1[ds'][t]
        compute_max_keep_3d(ctx, &bufs.e1, &mut bufs.e0, &mut bufs.keep_ev);

        // Group 1: E(S)[t] = Σ P(∅→ds) · e0[ds][t]
        let mut result = [0.0f32; T_SIZE];
        for ds_i in 0..252 {
            let p = ctx.dice_set_probabilities[ds_i] as f32;
            for t in 0..T_SIZE {
                result[t] += p * bufs.e0[ds_i][t];
            }
        }

        bufs.results[up] = result;
    }
}

/// Groups 5/3: for each dice set, find the best keep decision (element-wise over targets).
fn compute_max_keep_3d(
    ctx: &YatzyContext,
    e_prev: &Vec<[f32; T_SIZE]>,
    e_next: &mut Vec<[f32; T_SIZE]>,
    keep_ev: &mut Vec<[f32; T_SIZE]>,
) {
    let kt = &ctx.keep_table;

    // Step 1: compute EV for each of the 462 unique keeps
    for kid in 0..constants::NUM_KEEP_MULTISETS {
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;
        let mut ev = [0.0f32; T_SIZE];
        for k in start..end {
            let prob = kt.vals[k];
            let child_ds = kt.cols[k] as usize;
            for t in 0..T_SIZE {
                ev[t] += prob * e_prev[child_ds][t];
            }
        }
        keep_ev[kid] = ev;
    }

    // Step 2: for each dice set, find the best keep (element-wise max)
    for ds_i in 0..252 {
        // Start with keep-all (mask=0): value is just e_prev[ds_i]
        let mut best = e_prev[ds_i];

        for j in 0..kt.unique_count[ds_i] as usize {
            let kid = kt.unique_keep_ids[ds_i][j] as usize;
            for t in 0..T_SIZE {
                if keep_ev[kid][t] > best[t] {
                    best[t] = keep_ev[kid][t];
                }
            }
        }

        e_next[ds_i] = best;
    }
}

fn main() {
    let base_path = yatzy::env_config::init_base_path();
    let num_threads = yatzy::env_config::init_rayon_threads();

    // Phase 0: build context
    let t0 = Instant::now();
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);
    println!("Phase 0: {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0);

    // Allocate 3D value table
    let t1 = Instant::now();
    let mut vt = ValueTable::new();
    println!("Allocation: {:.1} s", t1.elapsed().as_secs_f64());

    // Initialize terminal states (popcount = 15, all categories scored)
    // Terminal: remaining_score = bonus = 50 if upper >= 63, else 0
    // V[terminal][t] = 1.0 if bonus > t, else 0.0
    let t2 = Instant::now();
    for scored in 0..(1u32 << CATEGORY_COUNT) {
        if scored.count_ones() != CATEGORY_COUNT as u32 {
            continue;
        }
        for up in 0..=63usize {
            let bonus = if up >= 63 { 50 } else { 0 };
            let si = state_index(up, scored as usize);
            let mut vals = [0.0f32; T_SIZE];
            for t in 0..T_SIZE {
                vals[t] = if bonus > t { 1.0 } else { 0.0 };
            }
            vt.set(si, &vals);
            // Topological padding
            if up == 63 {
                for pad in 64..STATE_STRIDE {
                    vt.set(scored as usize * STATE_STRIDE + pad, &vals);
                }
            }
        }
    }
    println!(
        "Terminal init: {:.1} ms",
        t2.elapsed().as_secs_f64() * 1000.0
    );

    // Backward induction: popcount 14 down to 0
    let total_start = Instant::now();
    for num_scored in (0..CATEGORY_COUNT as i32).rev() {
        let level_start = Instant::now();

        let mut scored_masks: Vec<i32> = Vec::new();
        for scored in 0..(1u32 << CATEGORY_COUNT) {
            if scored.count_ones() == num_scored as u32 {
                let upper_mask = (scored & 0x3F) as usize;
                if (0..=63usize).any(|up| ctx.reachable[upper_mask][up]) {
                    scored_masks.push(scored as i32);
                }
            }
        }

        let n_masks = scored_masks.len();
        let vt_read_ptr = std::sync::atomic::AtomicPtr::new(vt.data.as_ptr() as *mut f32);
        let vt_write_ptr = std::sync::atomic::AtomicPtr::new(vt.data.as_mut_ptr());
        let ctx_ref = &*ctx;

        scored_masks.par_iter().for_each_init(
            || Buffers3D::new(),
            |bufs, &scored| {
                let rp = vt_read_ptr.load(std::sync::atomic::Ordering::Relaxed) as *const f32;
                solve_3d_for_mask(ctx_ref, rp, scored, bufs);

                let ptr = vt_write_ptr.load(std::sync::atomic::Ordering::Relaxed);
                let upper_mask = (scored & 0x3F) as usize;
                for up in 0..=63usize {
                    if ctx_ref.reachable[upper_mask][up] {
                        let si = state_index(up, scored as usize);
                        let start = si * T_SIZE;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                bufs.results[up].as_ptr(),
                                ptr.add(start),
                                T_SIZE,
                            );
                        }
                    }
                }
                // Topological padding
                let capped = bufs.results[63];
                let base = scored as usize * STATE_STRIDE;
                for pad in 64..STATE_STRIDE {
                    let si = base + pad;
                    let start = si * T_SIZE;
                    unsafe {
                        std::ptr::copy_nonoverlapping(capped.as_ptr(), ptr.add(start), T_SIZE);
                    }
                }
            },
        );

        let elapsed = level_start.elapsed().as_secs_f64();
        println!(
            "Level {:2} ({:5} masks): {:.1}s",
            num_scored, n_masks, elapsed
        );
    }

    let total_elapsed = total_start.elapsed().as_secs_f64();
    println!("\nTotal backward induction: {:.1}s", total_elapsed);

    // Read the initial state value: V(up=0, scored=0, t) for all t
    let start_si = state_index(0, 0);
    let win_probs = vt.get(start_si);

    // Load P1's exact PMF
    let density_path = base_path.join("outputs/density/density_0.json");
    let density_str = std::fs::read_to_string(&density_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", density_path.display(), e));
    let density: serde_json::Value = serde_json::from_str(&density_str)
        .unwrap_or_else(|e| panic!("Failed to parse density JSON: {}", e));
    let pmf = density["pmf"].as_array().expect("Missing pmf field");

    // Compute clairvoyant win rate: Σ P(P1=T) · V(start, T)
    // V(start, T) = P(P2's optimal remaining score > T)
    // Since start state has 0 accumulated score, V(start, T) = P(total_score > T)
    let mut clairvoyant_win = 0.0f64;
    let mut baseline_win = 0.0f64;
    let mut total_weight = 0.0f64;

    for entry in pmf {
        let score = entry[0].as_f64().unwrap() as usize;
        let prob = entry[1].as_f64().unwrap();
        if prob < 1e-15 || score >= T_SIZE {
            continue;
        }
        total_weight += prob;

        // P(P2 > T | optimal for target T) — need score strictly greater than T
        // V[start][T] = P(remaining > T), which means total > T (since accumulated=0)
        // But we want P(total > T), and V stores P(remaining > t).
        // At start state, remaining = total, so V[start][T] = P(total > T). Correct.
        let p_win = if score < T_SIZE {
            win_probs[score] as f64
        } else {
            0.0
        };
        clairvoyant_win += prob * p_win;

        // For reference: P(P2 > T) under theta=0 (from the same density)
        let p_baseline: f64 = pmf
            .iter()
            .filter_map(|e| {
                let s = e[0].as_f64().unwrap() as usize;
                let p = e[1].as_f64().unwrap();
                if s > score {
                    Some(p)
                } else {
                    None
                }
            })
            .sum();
        baseline_win += prob * p_baseline;
    }

    println!("\n=== Clairvoyant Upper Bound (Unconstrained) ===");
    println!("Baseline (θ=0 vs θ=0):      {:.4}%", baseline_win * 100.0);
    println!("Constant-θ clairvoyant:      50.218%  (from exact PMF optimization)");
    println!(
        "Unconstrained clairvoyant:   {:.4}%",
        clairvoyant_win * 100.0
    );
    println!(
        "Advantage over baseline:     +{:.3}pp",
        (clairvoyant_win - baseline_win) * 100.0
    );
    println!("Our best policy (varscaled): +0.86pp");
    println!("Total PMF weight:            {:.6}", total_weight);

    // Print win probability for selected targets
    println!("\nWin probability by target score:");
    println!("{:>6} {:>10} {:>12}", "Target", "P(P1=T)", "P(P2>T|opt)");
    for &t in &[
        100, 150, 200, 220, 240, 248, 250, 260, 270, 280, 300, 320, 340, 360,
    ] {
        if t < T_SIZE {
            let p1_prob: f64 = pmf
                .iter()
                .filter_map(|e| {
                    let s = e[0].as_f64().unwrap() as usize;
                    let p = e[1].as_f64().unwrap();
                    if s == t {
                        Some(p)
                    } else {
                        None
                    }
                })
                .sum();
            println!("{:6} {:10.6} {:12.6}", t, p1_prob, win_probs[t]);
        }
    }
}
