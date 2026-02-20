//! Export per-decision regret data with semantic features for skill ladder induction.
//!
//! Simulates N games under the EV-optimal (θ=0) policy, collecting:
//! - Semantic features (human-interpretable)
//! - Full Q-value vectors (not just best + gap)
//! - Optimal action + regret vector (q_best - q[a] for each action)
//!
//! Outputs 3 binary files:
//! - regret_category.bin:  features + q_values[15] + best_cat + regret[15]
//! - regret_reroll1.bin:   features + q_values[32] + best_mask + regret[32]
//! - regret_reroll2.bin:   features + q_values[32] + best_mask + regret[32]

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;

use yatzy::constants::*;
use yatzy::dice_mechanics::{find_dice_set_index, sort_dice_set};
use yatzy::game_mechanics::update_upper_score;
use yatzy::phase0_tables;
use yatzy::rosetta::dsl::{
    compute_category_q_values, compute_features, compute_reroll_q_values, features_to_f32,
    NUM_SEMANTIC_FEATURES,
};
use yatzy::storage::{load_all_state_values, state_file_path};
use yatzy::types::YatzyContext;
use yatzy::widget_solver::{choose_best_reroll_mask, compute_max_ev_for_n_rerolls};

// ── Constants ────────────────────────────────────────────────────────────

const MAGIC: u32 = 0x52455052; // "REPR" (regret export)
const VERSION: u32 = 1;

// Binary record layout:
//   Category: features[56] + q_values[15] + best_cat(u8 as f32) + regret[15] = 87 floats
//   Reroll:   features[56] + q_values[32] + best_mask(i32 as f32) + regret[32] = 121 floats
const CATEGORY_RECORD_FLOATS: usize = NUM_SEMANTIC_FEATURES + 15 + 1 + 15;
const REROLL_RECORD_FLOATS: usize = NUM_SEMANTIC_FEATURES + 32 + 1 + 32;

// ── Record types ─────────────────────────────────────────────────────────

struct CategoryRecord {
    features: [f32; NUM_SEMANTIC_FEATURES],
    q_values: [f32; CATEGORY_COUNT],
    best_cat: u8,
    regret: [f32; CATEGORY_COUNT],
}

struct RerollRecord {
    features: [f32; NUM_SEMANTIC_FEATURES],
    q_values: [f32; 32],  // padded to 32 (max possible masks)
    best_mask: i32,
    regret: [f32; 32],
}

struct GameRecords {
    category: Vec<CategoryRecord>,
    reroll1: Vec<RerollRecord>,
    reroll2: Vec<RerollRecord>,
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

// ── Reroll Q-values → padded [f32; 32] ──────────────────────────────────

fn reroll_q_to_padded(
    ctx: &YatzyContext,
    e_ds: &[f32; 252],
    dice: &[i32; 5],
) -> ([f32; 32], [f32; 32], i32) {
    let (actions, best_mask, best_q) = compute_reroll_q_values(ctx, e_ds, dice);

    let mut q_values = [f32::NEG_INFINITY; 32];
    let mut regret = [f32::NEG_INFINITY; 32];

    for &(mask, ev) in &actions {
        let idx = mask as usize;
        if idx < 32 {
            q_values[idx] = ev;
            regret[idx] = best_q - ev;
        }
    }

    (q_values, regret, best_mask)
}

// ── Game simulation with regret collection ───────────────────────────────

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

        // ── Reroll 1 decision (2 rerolls remaining) ──
        {
            let features = compute_features(turn, up_score, scored, &dice, 2);
            let feat_f32 = features_to_f32(&features);
            let (q_values, regret, best_mask) = reroll_q_to_padded(ctx, &e_ds_1, &dice);
            records.reroll1.push(RerollRecord {
                features: feat_f32,
                q_values,
                best_mask,
                regret,
            });
        }

        // Apply first reroll
        let mut best_ev = 0.0;
        let mask1 = choose_best_reroll_mask(ctx, &e_ds_1, &dice, &mut best_ev);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }

        // ── Reroll 2 decision (1 reroll remaining) ──
        {
            let features = compute_features(turn, up_score, scored, &dice, 1);
            let feat_f32 = features_to_f32(&features);
            let (q_values, regret, best_mask) = reroll_q_to_padded(ctx, &e_ds_0, &dice);
            records.reroll2.push(RerollRecord {
                features: feat_f32,
                q_values,
                best_mask,
                regret,
            });
        }

        // Apply second reroll
        let mask2 = choose_best_reroll_mask(ctx, &e_ds_0, &dice, &mut best_ev);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }

        // ── Category decision ──
        let ds_index = find_dice_set_index(ctx, &dice);
        {
            let features = compute_features(turn, up_score, scored, &dice, 0);
            let feat_f32 = features_to_f32(&features);
            let (q_vals, best_cat, best_q) =
                compute_category_q_values(ctx, sv, up_score, scored, ds_index, is_last_turn);
            let mut regret = [f32::NEG_INFINITY; CATEGORY_COUNT];
            for c in 0..CATEGORY_COUNT {
                if q_vals[c] != f32::NEG_INFINITY {
                    regret[c] = best_q - q_vals[c];
                }
            }
            records.category.push(CategoryRecord {
                features: feat_f32,
                q_values: q_vals,
                best_cat: best_cat as u8,
                regret,
            });
        }

        // Advance game state
        if is_last_turn {
            let (cat, scr) = find_best_category_final(ctx, up_score, scored, ds_index);
            up_score = update_upper_score(up_score, cat, scr);
            scored |= 1 << cat;
        } else {
            let (cat, scr) = find_best_category(ctx, sv, up_score, scored, ds_index);
            up_score = update_upper_score(up_score, cat, scr);
            scored |= 1 << cat;
        }
    }

    records
}

// ── Binary I/O ───────────────────────────────────────────────────────────

fn write_category_file(
    path: &str,
    records: &[CategoryRecord],
) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);

    // Header (32 bytes)
    f.write_all(&MAGIC.to_le_bytes())?;
    f.write_all(&VERSION.to_le_bytes())?;
    f.write_all(&(records.len() as u64).to_le_bytes())?;
    f.write_all(&(NUM_SEMANTIC_FEATURES as u32).to_le_bytes())?;
    f.write_all(&(CATEGORY_COUNT as u32).to_le_bytes())?;
    f.write_all(&0u64.to_le_bytes())?; // reserved

    for rec in records {
        for &feat in &rec.features {
            f.write_all(&feat.to_le_bytes())?;
        }
        for &q in &rec.q_values {
            f.write_all(&q.to_le_bytes())?;
        }
        f.write_all(&(rec.best_cat as f32).to_le_bytes())?;
        for &r in &rec.regret {
            f.write_all(&r.to_le_bytes())?;
        }
    }
    f.flush()?;
    Ok(())
}

fn write_reroll_file(
    path: &str,
    records: &[RerollRecord],
) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);

    // Header (32 bytes)
    f.write_all(&MAGIC.to_le_bytes())?;
    f.write_all(&VERSION.to_le_bytes())?;
    f.write_all(&(records.len() as u64).to_le_bytes())?;
    f.write_all(&(NUM_SEMANTIC_FEATURES as u32).to_le_bytes())?;
    f.write_all(&32u32.to_le_bytes())?; // num actions
    f.write_all(&0u64.to_le_bytes())?; // reserved

    for rec in records {
        for &feat in &rec.features {
            f.write_all(&feat.to_le_bytes())?;
        }
        for &q in &rec.q_values {
            f.write_all(&q.to_le_bytes())?;
        }
        f.write_all(&(rec.best_mask as f32).to_le_bytes())?;
        for &r in &rec.regret {
            f.write_all(&r.to_le_bytes())?;
        }
    }
    f.flush()?;
    Ok(())
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 200_000usize;
    let mut seed = 42u64;
    let mut output_dir = String::from("outputs/rosetta");

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
                println!("Usage: yatzy-regret-export [OPTIONS]");
                println!("  --games N     Number of games to simulate (default: 200000)");
                println!("  --seed S      Random seed (default: 42)");
                println!("  --output DIR  Output directory (default: outputs/rosetta)");
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

    // Simulate games and collect regret data
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
    let mut cat_records: Vec<CategoryRecord> = Vec::with_capacity(num_games * 15);
    let mut rr1_records: Vec<RerollRecord> = Vec::with_capacity(num_games * 15);
    let mut rr2_records: Vec<RerollRecord> = Vec::with_capacity(num_games * 15);

    for game in all_records {
        cat_records.extend(game.category);
        rr1_records.extend(game.reroll1);
        rr2_records.extend(game.reroll2);
    }

    // Print summary
    println!("\n=== Regret Export Summary ===");
    println!(
        "  category:  {:>10} records ({} floats/rec, ~{:.1} MB)",
        cat_records.len(),
        CATEGORY_RECORD_FLOATS,
        (cat_records.len() * CATEGORY_RECORD_FLOATS * 4 + 32) as f64 / 1e6,
    );
    println!(
        "  reroll1:   {:>10} records ({} floats/rec, ~{:.1} MB)",
        rr1_records.len(),
        REROLL_RECORD_FLOATS,
        (rr1_records.len() * REROLL_RECORD_FLOATS * 4 + 32) as f64 / 1e6,
    );
    println!(
        "  reroll2:   {:>10} records ({} floats/rec, ~{:.1} MB)",
        rr2_records.len(),
        REROLL_RECORD_FLOATS,
        (rr2_records.len() * REROLL_RECORD_FLOATS * 4 + 32) as f64 / 1e6,
    );

    // Regret statistics
    {
        let total_regret: f64 = cat_records
            .iter()
            .map(|r| {
                r.regret
                    .iter()
                    .filter(|&&v| v != f32::NEG_INFINITY && v > 0.0)
                    .map(|&v| v as f64)
                    .sum::<f64>()
            })
            .sum();
        let zero_regret = cat_records.iter().filter(|r| {
            r.regret.iter().all(|&v| v == f32::NEG_INFINITY || v < 0.01)
        }).count();
        println!(
            "  category: total_regret={:.1}, zero_regret_records={:.1}%",
            total_regret,
            100.0 * zero_regret as f64 / cat_records.len() as f64,
        );
    }

    // Write binary files
    let _ = std::fs::create_dir_all(&output_dir);

    let cat_path = format!("{}/regret_category.bin", output_dir);
    write_category_file(&cat_path, &cat_records).expect("Failed to write category file");
    let cat_size = std::fs::metadata(&cat_path).map(|m| m.len()).unwrap_or(0);
    println!("Wrote {} ({:.1} MB)", cat_path, cat_size as f64 / 1e6);

    let rr1_path = format!("{}/regret_reroll1.bin", output_dir);
    write_reroll_file(&rr1_path, &rr1_records).expect("Failed to write reroll1 file");
    let rr1_size = std::fs::metadata(&rr1_path).map(|m| m.len()).unwrap_or(0);
    println!("Wrote {} ({:.1} MB)", rr1_path, rr1_size as f64 / 1e6);

    let rr2_path = format!("{}/regret_reroll2.bin", output_dir);
    write_reroll_file(&rr2_path, &rr2_records).expect("Failed to write reroll2 file");
    let rr2_size = std::fs::metadata(&rr2_path).map(|m| m.len()).unwrap_or(0);
    println!("Wrote {} ({:.1} MB)", rr2_path, rr2_size as f64 / 1e6);

    println!("\nTotal time: {:.1}s", total_start.elapsed().as_secs_f64());
}
