//! Export per-decision training data for surrogate model training.
//!
//! Simulates N games under the EV-optimal (θ=0) policy, collecting feature vectors,
//! optimal actions, and decision gaps for each of the 3 decision types:
//! - Category selection (15 classes)
//! - Reroll 1 — first keep decision (32 classes: 5-bit mask)
//! - Reroll 2 — second keep decision (32 classes: 5-bit mask)
//!
//! Outputs 3 binary files in the specified directory, each with a 32-byte header
//! followed by fixed-size records.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;

use yatzy::constants::*;
use yatzy::dice_mechanics::{count_faces, find_dice_set_index, sort_dice_set};
use yatzy::game_mechanics::update_upper_score;
use yatzy::phase0_tables;
use yatzy::storage::{load_all_state_values, state_file_path};
use yatzy::types::YatzyContext;
use yatzy::widget_solver::{choose_best_reroll_mask, compute_max_ev_for_n_rerolls};

// ── Constants ────────────────────────────────────────────────────────────

const MAGIC: u32 = 0x59545244; // "YTRD"
const VERSION: u32 = 1;

const NUM_CATEGORY_FEATURES: u32 = 29;
const NUM_REROLL_FEATURES: u32 = 30;
const NUM_CATEGORY_ACTIONS: u32 = 15;
const NUM_REROLL_ACTIONS: u32 = 32;

// ── Record types ─────────────────────────────────────────────────────────

/// A single training record for any decision type.
struct TrainingRecord {
    features: Vec<f32>,
    best_action: u16,
    gap: f32,
}

/// Per-game collection of training records, separated by decision type.
struct GameRecords {
    category: Vec<TrainingRecord>,
    reroll1: Vec<TrainingRecord>,
    reroll2: Vec<TrainingRecord>,
}

// ── Feature extraction ───────────────────────────────────────────────────

/// Build shared features (indices 0-28) that are common across all decision types.
/// Returns 29 features for category decisions, 30 for reroll (caller appends rerolls_remaining).
#[inline(always)]
fn build_shared_features(
    turn: usize,
    up_score: i32,
    scored: i32,
    dice: &[i32; 5],
) -> Vec<f32> {
    let face_count = count_faces(dice);
    let dice_sum: i32 = dice.iter().sum();
    let max_face = face_count[1..=6].iter().max().copied().unwrap_or(0);
    let num_distinct = face_count[1..=6].iter().filter(|&&c| c > 0).count();

    // Count upper categories remaining
    let mut upper_cats_left = 0;
    for c in 0..6 {
        if !is_category_scored(scored, c) {
            upper_cats_left += 1;
        }
    }
    let bonus_secured = up_score >= 63;
    let bonus_deficit = if bonus_secured { 0 } else { 63 - up_score };

    let mut features = Vec::with_capacity(30);

    // 0: turn normalized
    features.push(turn as f32 / 14.0);
    // 1: upper_score normalized
    features.push(up_score as f32 / 63.0);
    // 2: upper categories left
    features.push(upper_cats_left as f32 / 6.0);
    // 3: bonus secured
    features.push(if bonus_secured { 1.0 } else { 0.0 });
    // 4: bonus deficit
    features.push(bonus_deficit as f32 / 63.0);
    // 5-10: face counts
    for f in 1..=6 {
        features.push(face_count[f] as f32 / 5.0);
    }
    // 11: dice sum
    features.push(dice_sum as f32 / 30.0);
    // 12: max face count
    features.push(max_face as f32 / 5.0);
    // 13: num distinct faces
    features.push(num_distinct as f32 / 6.0);
    // 14-28: category availability
    for c in 0..CATEGORY_COUNT {
        features.push(if is_category_scored(scored, c) { 0.0 } else { 1.0 });
    }

    features
}

// ── Gap computation ──────────────────────────────────────────────────────

/// Find best and second-best category. Returns (best_cat, gap).
#[inline(always)]
fn best_category_with_gap(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    ds_index: usize,
) -> (usize, f32) {
    let mut best = (f32::NEG_INFINITY, 0usize);
    let mut second = f32::NEG_INFINITY;

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
            second = best.0;
            best = (val, c);
        } else if val > second {
            second = val;
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
            second = best.0;
            best = (val, c);
        } else if val > second {
            second = val;
        }
    }

    let gap = if second == f32::NEG_INFINITY {
        0.0 // only 1 category available
    } else {
        best.0 - second
    };
    (best.1, gap)
}

/// Find best and second-best keep mask. Returns (best_mask, gap).
/// best_mask is the raw 5-bit reroll mask (0-31).
#[inline(always)]
fn best_keep_with_gap(
    ctx: &YatzyContext,
    e_ds: &[f32; 252],
    dice: &[i32; 5],
) -> (i32, f32) {
    let ds_index = find_dice_set_index(ctx, dice);
    let kt = &ctx.keep_table;

    // mask=0: keep all dice
    let keep_all_ev = e_ds[ds_index];
    let mut best = (keep_all_ev, 0i32); // (ev, mask)
    let mut second_ev = f32::NEG_INFINITY;

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

        let mask = kt.keep_to_mask[ds_index * 32 + j];
        if ev > best.0 {
            second_ev = best.0;
            best = (ev, mask);
        } else if ev > second_ev {
            second_ev = ev;
        }
    }

    let gap = if second_ev == f32::NEG_INFINITY {
        0.0
    } else {
        best.0 - second_ev
    };
    (best.1, gap)
}

// ── Simulation helpers ───────────────────────────────────────────────────

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

/// Compute Group 6: best category EV for each of the 252 dice sets.
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

// ── Game simulation with record collection ───────────────────────────────

fn simulate_game_collecting(ctx: &YatzyContext, rng: &mut SmallRng) -> GameRecords {
    let sv = ctx.state_values.as_slice();
    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;

    let mut records = GameRecords {
        category: Vec::with_capacity(15),
        reroll1: Vec::with_capacity(15),
        reroll2: Vec::with_capacity(15),
    };

    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;
        let mut dice = roll_dice(rng);

        // Compute e_ds arrays for this state
        compute_group6(ctx, sv, up_score, scored, &mut e_ds_0);
        compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);

        // ── Reroll 1 decision ──
        {
            let (best_mask, gap) = best_keep_with_gap(ctx, &e_ds_1, &dice);
            let mut features = build_shared_features(turn, up_score, scored, &dice);
            features.push(1.0); // rerolls_remaining = 2, normalized / 2.0
            records.reroll1.push(TrainingRecord {
                features,
                best_action: best_mask as u16,
                gap,
            });
        }

        // Apply first reroll
        let mut best_ev = 0.0;
        let mask1 = choose_best_reroll_mask(ctx, &e_ds_1, &dice, &mut best_ev);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }

        // ── Reroll 2 decision ──
        {
            let (best_mask, gap) = best_keep_with_gap(ctx, &e_ds_0, &dice);
            let mut features = build_shared_features(turn, up_score, scored, &dice);
            features.push(0.5); // rerolls_remaining = 1, normalized / 2.0
            records.reroll2.push(TrainingRecord {
                features,
                best_action: best_mask as u16,
                gap,
            });
        }

        // Apply second reroll
        let mask2 = choose_best_reroll_mask(ctx, &e_ds_0, &dice, &mut best_ev);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }

        // ── Category decision ──
        let ds_index = find_dice_set_index(ctx, &dice);
        if is_last_turn {
            // Last turn: only 1 category, gap = 0
            let (cat, _) = find_best_category_final(ctx, up_score, scored, ds_index);
            let features = build_shared_features(turn, up_score, scored, &dice);
            records.category.push(TrainingRecord {
                features,
                best_action: cat as u16,
                gap: 0.0,
            });
            let (cat, scr) = find_best_category_final(ctx, up_score, scored, ds_index);
            up_score = update_upper_score(up_score, cat, scr);
            scored |= 1 << cat;
        } else {
            let (cat, gap) = best_category_with_gap(ctx, sv, up_score, scored, ds_index);
            let features = build_shared_features(turn, up_score, scored, &dice);
            records.category.push(TrainingRecord {
                features,
                best_action: cat as u16,
                gap,
            });
            let (cat, scr) = find_best_category(ctx, sv, up_score, scored, ds_index);
            up_score = update_upper_score(up_score, cat, scr);
            scored |= 1 << cat;
        }
    }

    records
}

// ── Binary I/O ───────────────────────────────────────────────────────────

fn write_binary_file(
    path: &str,
    records: &[TrainingRecord],
    num_features: u32,
    num_actions: u32,
) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);

    // Header (32 bytes)
    f.write_all(&MAGIC.to_le_bytes())?;
    f.write_all(&VERSION.to_le_bytes())?;
    f.write_all(&(records.len() as u64).to_le_bytes())?;
    f.write_all(&num_features.to_le_bytes())?;
    f.write_all(&num_actions.to_le_bytes())?;
    f.write_all(&0u64.to_le_bytes())?; // reserved

    // Records
    for rec in records {
        for &feat in &rec.features {
            f.write_all(&feat.to_le_bytes())?;
        }
        f.write_all(&rec.best_action.to_le_bytes())?;
        f.write_all(&rec.gap.to_le_bytes())?;
    }

    f.flush()?;
    Ok(())
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 200_000usize;
    let mut seed = 42u64;
    let mut output_dir = String::from("data/surrogate");

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
                println!("Usage: yatzy-export-training-data [OPTIONS]");
                println!("  --games N     Number of games to simulate (default: 200000)");
                println!("  --seed S      Random seed (default: 42)");
                println!("  --output DIR  Output directory (default: data/surrogate)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    if std::env::set_current_dir(&base_path).is_err() {
        eprintln!("Failed to change directory to {}", base_path);
        std::process::exit(1);
    }

    // Resolve output path after chdir
    let output_dir = if std::path::Path::new(&output_dir).is_absolute() {
        output_dir
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&output_dir).to_string_lossy().to_string())
            .unwrap_or(output_dir)
    };

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
    println!("Loading lookup tables and state values...");
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    let file0 = state_file_path(0.0);
    if !load_all_state_values(&mut ctx, &file0) {
        eprintln!("Failed to load θ=0 state values from {}", file0);
        eprintln!("Run yatzy-precompute first.");
        std::process::exit(1);
    }

    // Simulate games and collect training data
    println!(
        "Simulating {} games (seed={}, threads={})...",
        num_games, seed, num_threads
    );
    let sim_start = Instant::now();

    let all_records: Vec<GameRecords> = (0..num_games)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
            simulate_game_collecting(&ctx, &mut rng)
        })
        .collect();

    let sim_time = sim_start.elapsed().as_secs_f64();
    println!(
        "  Simulation done in {:.1}s ({:.0} games/s)",
        sim_time,
        num_games as f64 / sim_time
    );

    // Flatten records
    let mut category_records: Vec<TrainingRecord> = Vec::with_capacity(num_games * 15);
    let mut reroll1_records: Vec<TrainingRecord> = Vec::with_capacity(num_games * 15);
    let mut reroll2_records: Vec<TrainingRecord> = Vec::with_capacity(num_games * 15);

    for game in all_records {
        category_records.extend(game.category);
        reroll1_records.extend(game.reroll1);
        reroll2_records.extend(game.reroll2);
    }

    // Print summary statistics
    println!("\n=== Training Data Summary ===");
    for (name, recs, n_feat, n_act) in [
        (
            "category",
            &category_records,
            NUM_CATEGORY_FEATURES,
            NUM_CATEGORY_ACTIONS,
        ),
        (
            "reroll1",
            &reroll1_records,
            NUM_REROLL_FEATURES,
            NUM_REROLL_ACTIONS,
        ),
        (
            "reroll2",
            &reroll2_records,
            NUM_REROLL_FEATURES,
            NUM_REROLL_ACTIONS,
        ),
    ] {
        let total_gap: f64 = recs.iter().map(|r| r.gap as f64).sum();
        let mean_gap = total_gap / recs.len() as f64;
        let zero_gap_frac =
            recs.iter().filter(|r| r.gap < 0.01).count() as f64 / recs.len() as f64;
        let record_bytes = (n_feat as usize * 4 + 2 + 4) * recs.len() + 32;
        println!(
            "  {:<10}: {:>10} records, {:<2} features, {:<2} actions, mean_gap={:.3}, zero_gap={:.1}%, ~{:.1} MB",
            name,
            recs.len(),
            n_feat,
            n_act,
            mean_gap,
            zero_gap_frac * 100.0,
            record_bytes as f64 / 1e6,
        );
    }

    // Write binary files
    let _ = std::fs::create_dir_all(&output_dir);

    for (name, recs, n_feat, n_act) in [
        (
            "category_decisions.bin",
            &category_records,
            NUM_CATEGORY_FEATURES,
            NUM_CATEGORY_ACTIONS,
        ),
        (
            "reroll1_decisions.bin",
            &reroll1_records,
            NUM_REROLL_FEATURES,
            NUM_REROLL_ACTIONS,
        ),
        (
            "reroll2_decisions.bin",
            &reroll2_records,
            NUM_REROLL_FEATURES,
            NUM_REROLL_ACTIONS,
        ),
    ] {
        let path = format!("{}/{}", output_dir, name);
        write_binary_file(&path, recs, n_feat, n_act).expect(&format!("Failed to write {}", path));
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        println!("Wrote {} ({:.1} MB)", path, size as f64 / 1e6);
    }

    println!(
        "\nTotal time: {:.1}s",
        total_start.elapsed().as_secs_f64()
    );
}
