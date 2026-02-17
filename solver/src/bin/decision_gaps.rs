//! Decision gap analysis: measure how close the runner-up action is to optimal
//! across all reachable states, and compute policy compression statistics.
//!
//! Three phases:
//! 1. Gap Distribution: for every reachable (state, dice, decision_type), compute
//!    gap = V_best - V_second. Accumulate into per-stratum histograms.
//! 2. Policy Compression: count policy-distinct states by hashing 252-action vectors.
//! 3. Visit-Weighted Gaps: simulate N games, record visit counts, cross-reference
//!    with recomputed gaps to get frequency-weighted statistics.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::time::Instant;

use yatzy::constants::*;
use yatzy::dice_mechanics::{find_dice_set_index, sort_dice_set};
use yatzy::game_mechanics::update_upper_score;
use yatzy::phase0_tables;
use yatzy::storage::{load_all_state_values, state_file_path};
use yatzy::types::YatzyContext;
use yatzy::widget_solver::{choose_best_reroll_mask, compute_max_ev_for_n_rerolls};

// ── Data structures ─────────────────────────────────────────────────────

/// Decision type at each point in a turn.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum DecisionType {
    Reroll1,
    Reroll2,
    Category,
}

impl DecisionType {
    fn as_str(&self) -> &'static str {
        match self {
            DecisionType::Reroll1 => "reroll1",
            DecisionType::Reroll2 => "reroll2",
            DecisionType::Category => "category",
        }
    }
}

/// Gap histogram with 0.01 resolution up to 50.0 (5001 bins + overflow).
#[derive(Clone)]
struct GapHistogram {
    bins: Vec<u64>, // 5001 bins
    overflow: u64,
    count: u64,
    sum: f64,
    sum_sq: f64,
}

impl GapHistogram {
    fn new() -> Self {
        Self {
            bins: vec![0u64; 5001],
            overflow: 0,
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    fn record(&mut self, gap: f32) {
        let gap = gap.max(0.0);
        self.count += 1;
        self.sum += gap as f64;
        self.sum_sq += (gap as f64) * (gap as f64);

        let bin = (gap * 100.0) as usize;
        if bin < 5001 {
            self.bins[bin] += 1;
        } else {
            self.overflow += 1;
        }
    }

    fn merge(&mut self, other: &GapHistogram) {
        for i in 0..5001 {
            self.bins[i] += other.bins[i];
        }
        self.overflow += other.overflow;
        self.count += other.count;
        self.sum += other.sum;
        self.sum_sq += other.sum_sq;
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    fn std(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f64;
        let var = (self.sum_sq - self.sum * self.sum / n) / (n - 1.0);
        var.max(0.0).sqrt()
    }

    fn percentile(&self, p: f64) -> f64 {
        let target = (p * self.count as f64).ceil() as u64;
        let mut cumulative = 0u64;
        for i in 0..5001 {
            cumulative += self.bins[i];
            if cumulative >= target {
                return i as f64 * 0.01;
            }
        }
        50.0 // overflow
    }

    fn fraction_below(&self, threshold: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let bin_limit = ((threshold * 100.0).ceil() as usize).min(5001);
        let below: u64 = self.bins[..bin_limit].iter().sum();
        below as f64 / self.count as f64
    }
}

/// Stratum key for gap histograms.
#[derive(Clone, PartialEq, Eq, Hash)]
struct StratumKey {
    decision_type: DecisionType,
    turn: usize,
}

/// Thread-local accumulator for Phase 1+2.
struct ThreadAccumulator {
    histograms: HashMap<StratumKey, GapHistogram>,
    // Phase 2b: policy-distinct counting — hash of 252-element action vectors
    category_hashes: HashMap<usize, HashSet<u64>>,
    reroll1_hashes: HashMap<usize, HashSet<u64>>,
    reroll2_hashes: HashMap<usize, HashSet<u64>>,
    total_states_per_turn: HashMap<usize, u64>,
}

impl ThreadAccumulator {
    fn new() -> Self {
        Self {
            histograms: HashMap::new(),
            category_hashes: HashMap::new(),
            reroll1_hashes: HashMap::new(),
            reroll2_hashes: HashMap::new(),
            total_states_per_turn: HashMap::new(),
        }
    }

    fn record_gap(&mut self, dt: DecisionType, turn: usize, gap: f32) {
        let key = StratumKey {
            decision_type: dt,
            turn,
        };
        self.histograms
            .entry(key)
            .or_insert_with(GapHistogram::new)
            .record(gap);
    }

    fn record_policy_hash(&mut self, dt: DecisionType, turn: usize, hash: u64) {
        let map = match dt {
            DecisionType::Category => &mut self.category_hashes,
            DecisionType::Reroll1 => &mut self.reroll1_hashes,
            DecisionType::Reroll2 => &mut self.reroll2_hashes,
        };
        map.entry(turn).or_default().insert(hash);
    }

    fn record_state(&mut self, turn: usize) {
        *self.total_states_per_turn.entry(turn).or_default() += 1;
    }

    fn merge(&mut self, other: ThreadAccumulator) {
        for (key, hist) in other.histograms {
            self.histograms
                .entry(key)
                .or_insert_with(GapHistogram::new)
                .merge(&hist);
        }
        for (turn, hashes) in other.category_hashes {
            self.category_hashes.entry(turn).or_default().extend(hashes);
        }
        for (turn, hashes) in other.reroll1_hashes {
            self.reroll1_hashes.entry(turn).or_default().extend(hashes);
        }
        for (turn, hashes) in other.reroll2_hashes {
            self.reroll2_hashes.entry(turn).or_default().extend(hashes);
        }
        for (turn, count) in other.total_states_per_turn {
            *self.total_states_per_turn.entry(turn).or_default() += count;
        }
    }
}

// ── Core gap computation ────────────────────────────────────────────────

/// Find best and second-best category for a dice set at a game state.
/// Returns (best_val, second_val, best_cat, second_cat).
#[inline(always)]
fn best_two_categories(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    ds_index: usize,
) -> (f32, f32, usize, usize) {
    let mut best = (f32::NEG_INFINITY, 0usize);
    let mut second = (f32::NEG_INFINITY, 0usize);

    for c in 0..6 {
        if is_category_scored(scored, c) {
            continue;
        }
        let scr = ctx.precomputed_scores[ds_index][c];
        let new_up = update_upper_score(up_score, c, scr);
        let new_scored = scored | (1 << c);
        let val = scr as f32
            + unsafe { *sv.get_unchecked(state_index(new_up as usize, new_scored as usize)) };
        if val > best.0 {
            second = best;
            best = (val, c);
        } else if val > second.0 {
            second = (val, c);
        }
    }
    for c in 6..CATEGORY_COUNT {
        if is_category_scored(scored, c) {
            continue;
        }
        let scr = ctx.precomputed_scores[ds_index][c];
        let new_scored = scored | (1 << c);
        let val = scr as f32
            + unsafe { *sv.get_unchecked(state_index(up_score as usize, new_scored as usize)) };
        if val > best.0 {
            second = best;
            best = (val, c);
        } else if val > second.0 {
            second = (val, c);
        }
    }
    (best.0, second.0, best.1, second.1)
}

/// Find best and second-best keep EV for a specific dice set.
/// Returns (best_ev, second_ev, best_action_id, second_action_id).
/// Action ID: keep_multiset index (or -1 for keep-all / no second).
#[inline(always)]
fn best_two_keeps(ctx: &YatzyContext, e_ds: &[f32; 252], ds_index: usize) -> (f32, f32, i32, i32) {
    let kt = &ctx.keep_table;

    // mask=0: keep all
    let keep_all_ev = e_ds[ds_index];
    let mut best = (keep_all_ev, -1i32); // -1 = keep all
    let mut second = (f32::NEG_INFINITY, -2i32);

    for j in 0..kt.unique_count[ds_index] as usize {
        let kid = kt.unique_keep_ids[ds_index][j] as usize;
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;
        let mut ev: f32 = 0.0;
        for k in start..end {
            unsafe {
                ev += (*kt.vals.get_unchecked(k) as f32)
                    * e_ds.get_unchecked(*kt.cols.get_unchecked(k) as usize);
            }
        }

        if ev > best.0 {
            second = best;
            best = (ev, kid as i32);
        } else if ev > second.0 {
            second = (ev, kid as i32);
        }
    }
    (best.0, second.0, best.1, second.1)
}

/// Compute Group 6: best category EV per dice set.
#[inline(always)]
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

/// Hash a 252-element action vector to a u64 for policy-distinct counting.
fn hash_action_vec(actions: &[u16; 252]) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    actions.hash(&mut hasher);
    hasher.finish()
}

/// Build 252-element category action vector (best category index per dice set).
fn category_action_vec(ctx: &YatzyContext, sv: &[f32], up_score: i32, scored: i32) -> [u16; 252] {
    let mut actions = [0u16; 252];
    for ds_i in 0..252 {
        let (_, _, best_cat, _) = best_two_categories(ctx, sv, up_score, scored, ds_i);
        actions[ds_i] = best_cat as u16;
    }
    actions
}

/// Build 252-element reroll action vector (best keep multiset index per dice set).
fn reroll_action_vec(ctx: &YatzyContext, e_ds: &[f32; 252]) -> [u16; 252] {
    let mut actions = [0u16; 252];
    let kt = &ctx.keep_table;
    for ds_i in 0..252 {
        let mut best_val = e_ds[ds_i]; // keep all
        let mut best_id = 0u16; // 0 = keep all sentinel

        for j in 0..kt.unique_count[ds_i] as usize {
            let kid = kt.unique_keep_ids[ds_i][j] as usize;
            let start = kt.row_start[kid] as usize;
            let end = kt.row_start[kid + 1] as usize;
            let mut ev: f32 = 0.0;
            for k in start..end {
                unsafe {
                    ev += (*kt.vals.get_unchecked(k) as f32)
                        * e_ds.get_unchecked(*kt.cols.get_unchecked(k) as usize);
                }
            }
            if ev > best_val {
                best_val = ev;
                best_id = (kid + 1) as u16; // +1 to distinguish from keep-all
            }
        }
        actions[ds_i] = best_id;
    }
    actions
}

// ── Simulation helpers (Phase 3) ────────────────────────────────────────

fn roll_dice(rng: &mut SmallRng) -> [i32; 5] {
    let mut dice = [0i32; 5];
    for d in &mut dice {
        *d = rng.random_range(1..=6);
    }
    sort_dice_set(&mut dice);
    dice
}

fn apply_reroll(dice: &mut [i32; 5], mask: i32, rng: &mut SmallRng) {
    for i in 0..5 {
        if mask & (1 << i) != 0 {
            dice[i] = rng.random_range(1..=6);
        }
    }
    sort_dice_set(dice);
}

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

/// Phase 3: per-visit gap record.
struct VisitGap {
    gap: f32,
    decision_type: DecisionType,
}

/// Simulate one game collecting gaps on the fly.
fn simulate_game_with_gaps(ctx: &YatzyContext, rng: &mut SmallRng) -> Vec<VisitGap> {
    let sv = ctx.state_values.as_slice();
    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut results = Vec::with_capacity(45);

    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;
        let mut dice = roll_dice(rng);

        // Compute e_ds arrays for this state
        compute_group6(ctx, sv, up_score, scored, &mut e_ds_0);
        compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);

        // Reroll1 gap
        {
            let ds_idx = find_dice_set_index(ctx, &dice);
            let (best, second, _, _) = best_two_keeps(ctx, &e_ds_1, ds_idx);
            results.push(VisitGap {
                gap: best - second,
                decision_type: DecisionType::Reroll1,
            });
        }

        // Apply first reroll
        let mut best_ev = 0.0;
        let mask1 = choose_best_reroll_mask(ctx, &e_ds_1, &dice, &mut best_ev);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }

        // Reroll2 gap
        {
            let ds_idx = find_dice_set_index(ctx, &dice);
            let (best, second, _, _) = best_two_keeps(ctx, &e_ds_0, ds_idx);
            results.push(VisitGap {
                gap: best - second,
                decision_type: DecisionType::Reroll2,
            });
        }

        // Apply second reroll
        let mask2 = choose_best_reroll_mask(ctx, &e_ds_0, &dice, &mut best_ev);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }

        // Category gap
        {
            let ds_idx = find_dice_set_index(ctx, &dice);
            if is_last_turn {
                // Last turn: only 1 category left, gap = 0
                results.push(VisitGap {
                    gap: 0.0,
                    decision_type: DecisionType::Category,
                });
            } else {
                let (best, second, _, _) = best_two_categories(ctx, sv, up_score, scored, ds_idx);
                results.push(VisitGap {
                    gap: best - second,
                    decision_type: DecisionType::Category,
                });
            }
        }

        // Execute optimal action
        let ds_index = find_dice_set_index(ctx, &dice);
        let (cat, scr) = if is_last_turn {
            find_best_category_final(ctx, up_score, scored, ds_index)
        } else {
            find_best_category(ctx, sv, up_score, scored, ds_index)
        };
        up_score = update_upper_score(up_score, cat, scr);
        scored |= 1 << cat;
    }

    results
}

// ── Output serialization ────────────────────────────────────────────────

#[derive(Serialize)]
struct PolicySizeJson {
    total_reachable_states: u64,
    total_decisions: u64,
    raw_policy_bytes_category: u64,
    raw_policy_bytes_reroll: u64,
    unique_category_policies_by_turn: HashMap<usize, usize>,
    unique_reroll1_policies_by_turn: HashMap<usize, usize>,
    unique_reroll2_policies_by_turn: HashMap<usize, usize>,
}

#[derive(Serialize)]
struct SummaryJson {
    total_reachable_states: u64,
    total_state_dice_pairs: u64,
    phase1: Phase1Summary,
    phase2: Phase2Summary,
    phase3: Phase3Summary,
}

#[derive(Serialize)]
struct Phase1Summary {
    total_gap_records: u64,
    overall_mean_gap_category: f64,
    overall_mean_gap_reroll1: f64,
    overall_mean_gap_reroll2: f64,
    frac_below_0_1_category: f64,
    frac_below_0_1_reroll1: f64,
    frac_below_0_1_reroll2: f64,
    frac_below_1_0_category: f64,
    frac_below_1_0_reroll1: f64,
    frac_below_1_0_reroll2: f64,
}

#[derive(Serialize)]
struct Phase2Summary {
    total_reachable_states: u64,
    category_distinct_total: usize,
    reroll1_distinct_total: usize,
    reroll2_distinct_total: usize,
}

#[derive(Serialize)]
struct Phase3Summary {
    games_simulated: usize,
    decisions_per_game: f64,
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 100_000usize;
    let mut seed = 42u64;
    let mut output_dir = String::from("outputs/policy_compression");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                i += 1;
                num_games = args[i].parse().expect("Invalid --games");
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("Invalid --seed");
            }
            "--output" => {
                i += 1;
                output_dir = args[i].clone();
            }
            "--help" | "-h" => {
                println!("Usage: yatzy-decision-gaps [OPTIONS]");
                println!("  --games N     Number of games for visit weighting (default: 100000)");
                println!("  --seed S      Random seed (default: 42)");
                println!("  --output DIR  Output directory (default: outputs/policy_compression)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Resolve output path
    let output_dir = if std::path::Path::new(&output_dir).is_absolute() {
        output_dir
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&output_dir).to_string_lossy().to_string())
            .unwrap_or(output_dir)
    };

    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    if std::env::set_current_dir(&base_path).is_err() {
        eprintln!("Failed to change directory to {}", base_path);
        std::process::exit(1);
    }

    let num_threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    let total_start = Instant::now();

    // Phase 0: precompute + load
    println!("Phase 0: Precomputing lookup tables...");
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    let file0 = state_file_path(0.0);
    if !load_all_state_values(&mut ctx, &file0) {
        eprintln!("Failed to load θ=0 state values from {}", file0);
        eprintln!("Run yatzy-precompute first.");
        std::process::exit(1);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Phase 1+2: Iterate all reachable states (parallel over scored groups)
    // ══════════════════════════════════════════════════════════════════════
    println!("\nPhase 1+2: Computing decision gaps for all reachable states...");
    let phase12_start = Instant::now();

    // Collect all scored bitmasks grouped by popcount (turn)
    let mut scored_by_turn: Vec<Vec<i32>> = vec![Vec::new(); CATEGORY_COUNT];
    for scored in 0..(1u32 << CATEGORY_COUNT) {
        let turn = scored.count_ones() as usize;
        if turn < CATEGORY_COUNT {
            scored_by_turn[turn].push(scored as i32);
        }
    }

    // Process all turns, accumulating results
    let sv = ctx.state_values.as_slice();
    let mut global_acc = ThreadAccumulator::new();
    let mut total_reachable: u64 = 0;

    for turn in 0..CATEGORY_COUNT {
        let scored_list = &scored_by_turn[turn];
        let is_last_turn = turn == CATEGORY_COUNT - 1;

        // For each scored bitmask in this turn, collect reachable (scored, up_score) pairs
        let mut work_items: Vec<(i32, i32)> = Vec::new();
        for &scored in scored_list {
            let upper_mask = (scored & 0x3F) as usize;
            for up in 0..=63i32 {
                if ctx.reachable[upper_mask][up as usize] {
                    work_items.push((scored, up));
                }
            }
        }

        total_reachable += work_items.len() as u64;

        // Parallel processing of work items
        let turn_acc: ThreadAccumulator = work_items
            .par_iter()
            .fold(
                || ThreadAccumulator::new(),
                |mut acc, &(scored, up_score)| {
                    let mut e_ds_0 = [0.0f32; 252];
                    let mut e_ds_1 = [0.0f32; 252];

                    // Group 6: best category EV per dice set
                    compute_group6(&ctx, sv, up_score, scored, &mut e_ds_0);
                    // Group 5: propagate through one reroll
                    compute_max_ev_for_n_rerolls(&ctx, &e_ds_0, &mut e_ds_1);

                    // Build action vectors for policy-distinct hashing
                    let cat_actions = category_action_vec(&ctx, sv, up_score, scored);
                    let reroll2_actions = reroll_action_vec(&ctx, &e_ds_0);
                    let reroll1_actions = reroll_action_vec(&ctx, &e_ds_1);

                    acc.record_policy_hash(
                        DecisionType::Category,
                        turn,
                        hash_action_vec(&cat_actions),
                    );
                    acc.record_policy_hash(
                        DecisionType::Reroll2,
                        turn,
                        hash_action_vec(&reroll2_actions),
                    );
                    acc.record_policy_hash(
                        DecisionType::Reroll1,
                        turn,
                        hash_action_vec(&reroll1_actions),
                    );
                    acc.record_state(turn);

                    // Gap computation for each dice set
                    for ds_i in 0..252 {
                        // Category gap
                        if is_last_turn {
                            // Only 1 category available → gap = 0
                            acc.record_gap(DecisionType::Category, turn, 0.0);
                        } else {
                            let (best, second, _, _) =
                                best_two_categories(&ctx, sv, up_score, scored, ds_i);
                            let avail = (0..CATEGORY_COUNT)
                                .filter(|&c| !is_category_scored(scored, c))
                                .count();
                            if avail >= 2 {
                                acc.record_gap(DecisionType::Category, turn, best - second);
                            } else {
                                acc.record_gap(DecisionType::Category, turn, 0.0);
                            }
                        }

                        // Reroll2 gap (uses e_ds_0)
                        let (best, second, _, _) = best_two_keeps(&ctx, &e_ds_0, ds_i);
                        acc.record_gap(DecisionType::Reroll2, turn, best - second);

                        // Reroll1 gap (uses e_ds_1)
                        let (best, second, _, _) = best_two_keeps(&ctx, &e_ds_1, ds_i);
                        acc.record_gap(DecisionType::Reroll1, turn, best - second);
                    }

                    acc
                },
            )
            .reduce(
                || ThreadAccumulator::new(),
                |mut a, b| {
                    a.merge(b);
                    a
                },
            );

        global_acc.merge(turn_acc);

        // Progress
        let elapsed = phase12_start.elapsed().as_secs_f64();
        let total_gaps: u64 = global_acc.histograms.values().map(|h| h.count).sum();
        println!(
            "  Turn {:>2}: {:>7} states, {:>12} gaps total ({:.1}s)",
            turn,
            work_items.len(),
            total_gaps,
            elapsed
        );
    }

    let phase12_time = phase12_start.elapsed().as_secs_f64();
    println!(
        "Phase 1+2 done: {} reachable states in {:.1}s",
        total_reachable, phase12_time
    );

    // ══════════════════════════════════════════════════════════════════════
    // Phase 3: Visit-weighted gaps via simulation
    // ══════════════════════════════════════════════════════════════════════
    println!(
        "\nPhase 3: Simulating {} games for visit weighting...",
        num_games
    );
    let phase3_start = Instant::now();

    let all_visit_gaps: Vec<Vec<VisitGap>> = (0..num_games)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
            simulate_game_with_gaps(&ctx, &mut rng)
        })
        .collect();

    let phase3_time = phase3_start.elapsed().as_secs_f64();
    let total_visit_decisions: usize = all_visit_gaps.iter().map(|v| v.len()).sum();
    println!(
        "  {} decisions in {:.1}s ({:.1} decisions/game)",
        total_visit_decisions,
        phase3_time,
        total_visit_decisions as f64 / num_games as f64,
    );

    // Build visit-weighted gap histograms
    let mut visit_hist_all = GapHistogram::new();
    let mut visit_hist_by_type: HashMap<DecisionType, GapHistogram> = HashMap::new();
    for game in &all_visit_gaps {
        for vg in game {
            visit_hist_all.record(vg.gap);
            visit_hist_by_type
                .entry(vg.decision_type)
                .or_insert_with(GapHistogram::new)
                .record(vg.gap);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Output
    // ══════════════════════════════════════════════════════════════════════
    let _ = std::fs::create_dir_all(&output_dir);

    // 1. gap_summary.csv
    {
        let path = format!("{}/gap_summary.csv", output_dir);
        let mut f = std::fs::File::create(&path).expect("Failed to create gap_summary.csv");
        writeln!(
            f,
            "decision_type,turn,count,mean_gap,std_gap,median_gap,p1,p5,p25,p75,p95,p99,\
             frac_below_0.1,frac_below_0.5,frac_below_1.0,frac_below_5.0"
        )
        .unwrap();

        let mut keys: Vec<_> = global_acc.histograms.keys().cloned().collect();
        keys.sort_by_key(|k| (k.turn, k.decision_type as u8));

        for key in &keys {
            let h = &global_acc.histograms[key];
            writeln!(
                f,
                "{},{},{},{:.4},{:.4},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.6},{:.6},{:.6},{:.6}",
                key.decision_type.as_str(),
                key.turn,
                h.count,
                h.mean(),
                h.std(),
                h.percentile(0.50),
                h.percentile(0.01),
                h.percentile(0.05),
                h.percentile(0.25),
                h.percentile(0.75),
                h.percentile(0.95),
                h.percentile(0.99),
                h.fraction_below(0.1),
                h.fraction_below(0.5),
                h.fraction_below(1.0),
                h.fraction_below(5.0),
            )
            .unwrap();
        }
        println!("Wrote {}", path);
    }

    // 2. gap_cdf.csv — fine-grained CDF for plotting
    {
        let path = format!("{}/gap_cdf.csv", output_dir);
        let mut f = std::fs::File::create(&path).expect("Failed to create gap_cdf.csv");
        writeln!(f, "decision_type,turn,gap_threshold,fraction_below").unwrap();

        // Aggregate histograms by decision type (across all turns)
        let mut agg: HashMap<DecisionType, GapHistogram> = HashMap::new();
        for (key, hist) in &global_acc.histograms {
            agg.entry(key.decision_type)
                .or_insert_with(GapHistogram::new)
                .merge(hist);
        }

        // Write per-type CDF at fine resolution
        let thresholds: Vec<f64> = (0..=5000).map(|i| i as f64 * 0.01).collect();
        for dt in [
            DecisionType::Category,
            DecisionType::Reroll1,
            DecisionType::Reroll2,
        ] {
            if let Some(h) = agg.get(&dt) {
                for &t in &thresholds {
                    writeln!(f, "{},all,{:.2},{:.8}", dt.as_str(), t, h.fraction_below(t),)
                        .unwrap();
                }
            }
        }
        println!("Wrote {}", path);
    }

    // 3. gap_aggregate.csv — overall aggregates
    {
        let path = format!("{}/gap_aggregate.csv", output_dir);
        let mut f = std::fs::File::create(&path).expect("Failed to create gap_aggregate.csv");
        writeln!(
            f,
            "decision_type,count,mean_gap,frac_below_0.1,frac_below_0.5,frac_below_1.0,\
             frac_below_5.0,frac_below_10.0"
        )
        .unwrap();

        let mut agg: HashMap<DecisionType, GapHistogram> = HashMap::new();
        for (key, hist) in &global_acc.histograms {
            agg.entry(key.decision_type)
                .or_insert_with(GapHistogram::new)
                .merge(hist);
        }

        for dt in [
            DecisionType::Category,
            DecisionType::Reroll1,
            DecisionType::Reroll2,
        ] {
            if let Some(h) = agg.get(&dt) {
                writeln!(
                    f,
                    "{},{},{:.4},{:.6},{:.6},{:.6},{:.6},{:.6}",
                    dt.as_str(),
                    h.count,
                    h.mean(),
                    h.fraction_below(0.1),
                    h.fraction_below(0.5),
                    h.fraction_below(1.0),
                    h.fraction_below(5.0),
                    h.fraction_below(10.0),
                )
                .unwrap();
            }
        }
        println!("Wrote {}", path);
    }

    // 4. policy_distinct.csv
    {
        let path = format!("{}/policy_distinct.csv", output_dir);
        let mut f = std::fs::File::create(&path).expect("Failed to create policy_distinct.csv");
        writeln!(
            f,
            "turn,total_states,distinct_category,distinct_reroll1,distinct_reroll2"
        )
        .unwrap();

        for turn in 0..CATEGORY_COUNT {
            let total = global_acc
                .total_states_per_turn
                .get(&turn)
                .copied()
                .unwrap_or(0);
            let cat = global_acc
                .category_hashes
                .get(&turn)
                .map(|s| s.len())
                .unwrap_or(0);
            let r1 = global_acc
                .reroll1_hashes
                .get(&turn)
                .map(|s| s.len())
                .unwrap_or(0);
            let r2 = global_acc
                .reroll2_hashes
                .get(&turn)
                .map(|s| s.len())
                .unwrap_or(0);
            writeln!(f, "{},{},{},{},{}", turn, total, cat, r1, r2).unwrap();
        }
        println!("Wrote {}", path);
    }

    // 5. policy_size.json
    {
        let mut cat_by_turn = HashMap::new();
        let mut r1_by_turn = HashMap::new();
        let mut r2_by_turn = HashMap::new();

        for turn in 0..CATEGORY_COUNT {
            let c = global_acc
                .category_hashes
                .get(&turn)
                .map(|s| s.len())
                .unwrap_or(0);
            let r1 = global_acc
                .reroll1_hashes
                .get(&turn)
                .map(|s| s.len())
                .unwrap_or(0);
            let r2 = global_acc
                .reroll2_hashes
                .get(&turn)
                .map(|s| s.len())
                .unwrap_or(0);
            cat_by_turn.insert(turn, c);
            r1_by_turn.insert(turn, r1);
            r2_by_turn.insert(turn, r2);
        }

        let ps = PolicySizeJson {
            total_reachable_states: total_reachable,
            total_decisions: total_reachable * 252 * 3,
            // 252 dice × 4 bits per category (15 cats fit in 4 bits)
            raw_policy_bytes_category: total_reachable * 252,
            // 252 dice × 2 bytes per keep-multiset index
            raw_policy_bytes_reroll: total_reachable * 252 * 2,
            unique_category_policies_by_turn: cat_by_turn,
            unique_reroll1_policies_by_turn: r1_by_turn,
            unique_reroll2_policies_by_turn: r2_by_turn,
        };

        let path = format!("{}/policy_size.json", output_dir);
        let json = serde_json::to_string_pretty(&ps).unwrap();
        std::fs::write(&path, json).unwrap();
        println!("Wrote {}", path);
    }

    // 6. visit_gaps.csv — visit-weighted threshold analysis
    {
        let path = format!("{}/visit_gaps.csv", output_dir);
        let mut f = std::fs::File::create(&path).expect("Failed to create visit_gaps.csv");
        writeln!(
            f,
            "gap_threshold,decisions_per_game_above,visit_fraction_above"
        )
        .unwrap();

        let decisions_per_game = total_visit_decisions as f64 / num_games as f64;
        let thresholds = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        for &t in &thresholds {
            let frac_above = 1.0 - visit_hist_all.fraction_below(t);
            let dpg_above = frac_above * decisions_per_game;
            writeln!(f, "{:.2},{:.4},{:.8}", t, dpg_above, frac_above).unwrap();
        }
        println!("Wrote {}", path);
    }

    // 7. visit_gap_by_type.csv
    {
        let path = format!("{}/visit_gap_by_type.csv", output_dir);
        let mut f = std::fs::File::create(&path).expect("Failed to create visit_gap_by_type.csv");
        writeln!(
            f,
            "decision_type,gap_threshold,decisions_per_game_above,visit_fraction_above"
        )
        .unwrap();

        let thresholds = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        for dt in [
            DecisionType::Category,
            DecisionType::Reroll1,
            DecisionType::Reroll2,
        ] {
            if let Some(h) = visit_hist_by_type.get(&dt) {
                let dpg = h.count as f64 / num_games as f64;
                for &t in &thresholds {
                    let frac_above = 1.0 - h.fraction_below(t);
                    writeln!(
                        f,
                        "{},{:.2},{:.4},{:.8}",
                        dt.as_str(),
                        t,
                        frac_above * dpg,
                        frac_above,
                    )
                    .unwrap();
                }
            }
        }
        println!("Wrote {}", path);
    }

    // 8. decision_gaps_summary.json
    {
        let mut agg_cat = GapHistogram::new();
        let mut agg_r1 = GapHistogram::new();
        let mut agg_r2 = GapHistogram::new();
        for (key, hist) in &global_acc.histograms {
            match key.decision_type {
                DecisionType::Category => agg_cat.merge(hist),
                DecisionType::Reroll1 => agg_r1.merge(hist),
                DecisionType::Reroll2 => agg_r2.merge(hist),
            }
        }

        let cat_dist_total: usize = global_acc.category_hashes.values().map(|s| s.len()).sum();
        let r1_dist_total: usize = global_acc.reroll1_hashes.values().map(|s| s.len()).sum();
        let r2_dist_total: usize = global_acc.reroll2_hashes.values().map(|s| s.len()).sum();

        let summary = SummaryJson {
            total_reachable_states: total_reachable,
            total_state_dice_pairs: total_reachable * 252,
            phase1: Phase1Summary {
                total_gap_records: agg_cat.count + agg_r1.count + agg_r2.count,
                overall_mean_gap_category: agg_cat.mean(),
                overall_mean_gap_reroll1: agg_r1.mean(),
                overall_mean_gap_reroll2: agg_r2.mean(),
                frac_below_0_1_category: agg_cat.fraction_below(0.1),
                frac_below_0_1_reroll1: agg_r1.fraction_below(0.1),
                frac_below_0_1_reroll2: agg_r2.fraction_below(0.1),
                frac_below_1_0_category: agg_cat.fraction_below(1.0),
                frac_below_1_0_reroll1: agg_r1.fraction_below(1.0),
                frac_below_1_0_reroll2: agg_r2.fraction_below(1.0),
            },
            phase2: Phase2Summary {
                total_reachable_states: total_reachable,
                category_distinct_total: cat_dist_total,
                reroll1_distinct_total: r1_dist_total,
                reroll2_distinct_total: r2_dist_total,
            },
            phase3: Phase3Summary {
                games_simulated: num_games,
                decisions_per_game: total_visit_decisions as f64 / num_games as f64,
            },
        };

        let path = format!("{}/decision_gaps_summary.json", output_dir);
        let json = serde_json::to_string_pretty(&summary).unwrap();
        std::fs::write(&path, json).unwrap();
        println!("Wrote {}", path);
    }

    // Print summary to console
    println!("\n{}", "=".repeat(70));
    println!("=== Decision Gap Summary ===\n");

    let mut agg: HashMap<DecisionType, GapHistogram> = HashMap::new();
    for (key, hist) in &global_acc.histograms {
        agg.entry(key.decision_type)
            .or_insert_with(GapHistogram::new)
            .merge(hist);
    }

    println!(
        "{:>10} | {:>12} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}",
        "Type", "Count", "Mean", "<0.1", "<0.5", "<1.0", "<5.0"
    );
    println!("{}", "-".repeat(78));
    for dt in [
        DecisionType::Category,
        DecisionType::Reroll1,
        DecisionType::Reroll2,
    ] {
        if let Some(h) = agg.get(&dt) {
            println!(
                "{:>10} | {:>12} | {:>8.3} | {:>7.1}% | {:>7.1}% | {:>7.1}% | {:>7.1}%",
                dt.as_str(),
                h.count,
                h.mean(),
                h.fraction_below(0.1) * 100.0,
                h.fraction_below(0.5) * 100.0,
                h.fraction_below(1.0) * 100.0,
                h.fraction_below(5.0) * 100.0,
            );
        }
    }

    println!("\n=== Policy Compression ===\n");
    println!(
        "{:>5} | {:>10} | {:>10} | {:>10} | {:>10}",
        "Turn", "States", "Cat Dist", "R1 Dist", "R2 Dist"
    );
    println!("{}", "-".repeat(55));
    for turn in 0..CATEGORY_COUNT {
        let total = global_acc
            .total_states_per_turn
            .get(&turn)
            .copied()
            .unwrap_or(0);
        let cat = global_acc
            .category_hashes
            .get(&turn)
            .map(|s| s.len())
            .unwrap_or(0);
        let r1 = global_acc
            .reroll1_hashes
            .get(&turn)
            .map(|s| s.len())
            .unwrap_or(0);
        let r2 = global_acc
            .reroll2_hashes
            .get(&turn)
            .map(|s| s.len())
            .unwrap_or(0);
        println!(
            "{:>5} | {:>10} | {:>10} | {:>10} | {:>10}",
            turn, total, cat, r1, r2
        );
    }

    println!("\n=== Visit-Weighted Coverage ===\n");
    println!(
        "{:>12} | {:>10} | {:>10}",
        "Gap Thresh", "Dec/Game>", "Frac Above"
    );
    println!("{}", "-".repeat(40));
    let decisions_per_game = total_visit_decisions as f64 / num_games as f64;
    for &t in &[0.1, 0.5, 1.0, 5.0, 10.0] {
        let frac_above = 1.0 - visit_hist_all.fraction_below(t);
        println!(
            "{:>12.1} | {:>10.2} | {:>9.1}%",
            t,
            frac_above * decisions_per_game,
            frac_above * 100.0,
        );
    }

    println!(
        "\nTotal: {:.1}s (Phase 1+2: {:.1}s, Phase 3: {:.1}s)",
        total_start.elapsed().as_secs_f64(),
        phase12_time,
        phase3_time,
    );
}
