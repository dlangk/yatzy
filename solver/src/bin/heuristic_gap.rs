//! Heuristic gap analysis: compares heuristic vs EV-optimal decisions per turn.
//!
//! For each of ~45 decisions per game (15 turns × 3 decisions: reroll1, reroll2, category),
//! records where the heuristic disagrees with optimal and by how much EV.
//!
//! Outputs:
//! - heuristic_gap.csv: per-decision disagreements
//! - heuristic_gap_summary.json: top mistake patterns by total EV cost

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

use yatzy::api_computations::compute_roll_response;
use yatzy::constants::*;
use yatzy::dice_mechanics::{count_faces, sort_dice_set};
use yatzy::game_mechanics::{calculate_category_score, update_upper_score};
use yatzy::phase0_tables;
use yatzy::simulation::heuristic::{heuristic_pick_category, heuristic_reroll_mask};
use yatzy::storage::load_all_state_values;
use yatzy::types::YatzyContext;

fn set_working_directory() -> PathBuf {
    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    let path = PathBuf::from(&base_path);
    if std::env::set_current_dir(&base_path).is_err() {
        eprintln!("Failed to change directory to {}", base_path);
        std::process::exit(1);
    }
    path
}

struct Args {
    num_games: u32,
    seed: u64,
    output: String,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut num_games = 100_000u32;
    let mut seed = 42u64;
    let mut output = "outputs/heuristic_gap".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                i += 1;
                if i < args.len() {
                    num_games = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid --games value: {}", args[i]);
                        std::process::exit(1);
                    });
                }
            }
            "--seed" => {
                i += 1;
                if i < args.len() {
                    seed = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid --seed value: {}", args[i]);
                        std::process::exit(1);
                    });
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    output = args[i].clone();
                }
            }
            "--help" | "-h" => {
                println!("Usage: yatzy-heuristic-gap [OPTIONS]");
                println!();
                println!("Compare heuristic vs EV-optimal decisions per turn.");
                println!();
                println!("Options:");
                println!("  --games N    Number of games (default: 100000)");
                println!("  --seed S     RNG seed (default: 42)");
                println!("  --output DIR Output directory (default: outputs/heuristic_gap)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Args {
        num_games,
        seed,
        output,
    }
}

/// A single decision disagreement record.
#[derive(Clone)]
struct Disagreement {
    turn: usize,
    decision_type: &'static str, // "reroll1", "reroll2", "category"
    scored_categories: i32,
    upper_score: i32,
    dice: [i32; 5],
    heuristic_action: i32,
    optimal_action: i32,
    ev_heuristic: f64,
    ev_optimal: f64,
    ev_gap: f64,
}

/// Aggregated pattern for summary.
#[derive(Clone, Default)]
struct MistakePattern {
    decision_type: String,
    description: String,
    count: u64,
    total_ev_gap: f64,
    max_ev_gap: f64,
    example_dice: String,
}

/// Roll 5 random dice and sort them.
fn roll_dice(rng: &mut SmallRng) -> [i32; 5] {
    use rand::Rng;
    let mut dice = [0i32; 5];
    for d in &mut dice {
        *d = rng.random_range(1..=6);
    }
    sort_dice_set(&mut dice);
    dice
}

/// Apply a reroll mask.
fn apply_reroll(dice: &mut [i32; 5], mask: i32, rng: &mut SmallRng) {
    use rand::Rng;
    for (i, d) in dice.iter_mut().enumerate() {
        if mask & (1 << i) != 0 {
            *d = rng.random_range(1..=6);
        }
    }
    sort_dice_set(dice);
}

/// Simulate one game, comparing heuristic vs optimal at every decision point.
fn analyze_game(ctx: &YatzyContext, rng: &mut SmallRng) -> (Vec<Disagreement>, i32, i32) {
    let mut upper_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut total_score: i32 = 0;
    let mut disagreements = Vec::new();
    let mut num_category_mistakes = 0i32;

    for turn in 0..CATEGORY_COUNT {
        let mut dice = roll_dice(rng);

        // === Reroll 1 ===
        let fc1 = count_faces(&dice);
        let h_mask1 = heuristic_reroll_mask(&dice, &fc1, scored, upper_score);
        let resp1 = compute_roll_response(ctx, upper_score, scored, &dice, 2);

        let o_mask1 = resp1.optimal_mask.unwrap_or(0);
        if h_mask1 != o_mask1 {
            let mask_evs = resp1.mask_evs.unwrap();
            let ev_h = mask_evs[h_mask1 as usize];
            let ev_o = resp1.optimal_mask_ev.unwrap();
            let gap = ev_o - ev_h;
            if gap > 0.001 {
                disagreements.push(Disagreement {
                    turn,
                    decision_type: "reroll1",
                    scored_categories: scored,
                    upper_score,
                    dice,
                    heuristic_action: h_mask1,
                    optimal_action: o_mask1,
                    ev_heuristic: ev_h,
                    ev_optimal: ev_o,
                    ev_gap: gap,
                });
            }
        }

        // Apply heuristic's reroll
        if h_mask1 != 0 {
            apply_reroll(&mut dice, h_mask1, rng);
        }

        // === Reroll 2 ===
        let fc2 = count_faces(&dice);
        let h_mask2 = heuristic_reroll_mask(&dice, &fc2, scored, upper_score);
        let resp2 = compute_roll_response(ctx, upper_score, scored, &dice, 1);

        let o_mask2 = resp2.optimal_mask.unwrap_or(0);
        if h_mask2 != o_mask2 {
            let mask_evs = resp2.mask_evs.unwrap();
            let ev_h = mask_evs[h_mask2 as usize];
            let ev_o = resp2.optimal_mask_ev.unwrap();
            let gap = ev_o - ev_h;
            if gap > 0.001 {
                disagreements.push(Disagreement {
                    turn,
                    decision_type: "reroll2",
                    scored_categories: scored,
                    upper_score,
                    dice,
                    heuristic_action: h_mask2,
                    optimal_action: o_mask2,
                    ev_heuristic: ev_h,
                    ev_optimal: ev_o,
                    ev_gap: gap,
                });
            }
        }

        // Apply heuristic's reroll
        if h_mask2 != 0 {
            apply_reroll(&mut dice, h_mask2, rng);
        }

        // === Category selection ===
        let fc_final = count_faces(&dice);
        let h_cat = heuristic_pick_category(&dice, &fc_final, scored, upper_score);
        let resp3 = compute_roll_response(ctx, upper_score, scored, &dice, 0);

        let o_cat = resp3.optimal_category as usize;
        if h_cat != o_cat {
            let ev_h = resp3.categories[h_cat].ev_if_scored;
            let ev_o = resp3.optimal_category_ev;
            let gap = ev_o - ev_h;
            num_category_mistakes += 1;
            if gap > 0.001 {
                disagreements.push(Disagreement {
                    turn,
                    decision_type: "category",
                    scored_categories: scored,
                    upper_score,
                    dice,
                    heuristic_action: h_cat as i32,
                    optimal_action: o_cat as i32,
                    ev_heuristic: ev_h,
                    ev_optimal: ev_o,
                    ev_gap: gap,
                });
            }
        }

        // Apply heuristic's category choice
        let scr = calculate_category_score(&dice, h_cat);
        upper_score = update_upper_score(upper_score, h_cat, scr);
        scored |= 1 << h_cat;
        total_score += scr;
    }

    if upper_score >= 63 {
        total_score += 50;
    }

    (disagreements, total_score, num_category_mistakes)
}

/// Classify a disagreement into a human-readable pattern.
fn classify_disagreement(d: &Disagreement) -> String {
    match d.decision_type {
        "category" => {
            let h_name = if (d.heuristic_action as usize) < CATEGORY_COUNT {
                CATEGORY_NAMES[d.heuristic_action as usize]
            } else {
                "?"
            };
            let o_name = if (d.optimal_action as usize) < CATEGORY_COUNT {
                CATEGORY_NAMES[d.optimal_action as usize]
            } else {
                "?"
            };

            // Classify by pattern
            let h_is_upper = (d.heuristic_action as usize) < 6;
            let o_is_upper = (d.optimal_action as usize) < 6;
            let bonus_dead = d.upper_score >= 63;

            if !h_is_upper && o_is_upper && !bonus_dead {
                format!("cat:missed_upper({} vs {})", h_name, o_name)
            } else if h_is_upper && !o_is_upper {
                format!("cat:wasted_upper({} vs {})", h_name, o_name)
            } else if d.heuristic_action == CATEGORY_YATZY as i32 {
                format!("cat:yatzy_dump(vs {})", o_name)
            } else if d.optimal_action == CATEGORY_CHANCE as i32 {
                format!("cat:should_chance({} vs Chance)", h_name)
            } else if d.heuristic_action == CATEGORY_CHANCE as i32 {
                format!("cat:wasted_chance(vs {})", o_name)
            } else {
                format!("cat:other({} vs {})", h_name, o_name)
            }
        }
        "reroll1" | "reroll2" => {
            let h_kept = 5 - (d.heuristic_action as u32).count_ones();
            let o_kept = 5 - (d.optimal_action as u32).count_ones();

            if d.heuristic_action == 0 && d.optimal_action != 0 {
                format!("{}:should_reroll", d.decision_type)
            } else if d.heuristic_action != 0 && d.optimal_action == 0 {
                format!("{}:should_keep_all", d.decision_type)
            } else {
                format!(
                    "{}:wrong_keep(kept {} vs {})",
                    d.decision_type, h_kept, o_kept
                )
            }
        }
        _ => "unknown".to_string(),
    }
}

fn main() {
    let _base_path = set_working_directory();
    let args = parse_args();

    let num_threads = std::env::var("RAYON_NUM_THREADS")
        .or_else(|_| std::env::var("OMP_NUM_THREADS"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    // Phase 0: build context
    let t0 = Instant::now();
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);
    println!(
        "Phase 0 tables: {:.1} ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Load EV-optimal state values
    let t1 = Instant::now();
    if !load_all_state_values(&mut ctx, "data/strategy_tables/all_states.bin") {
        eprintln!("Failed to load state values. Run yatzy-precompute first.");
        std::process::exit(1);
    }
    println!(
        "State values loaded: {:.1} ms",
        t1.elapsed().as_secs_f64() * 1000.0
    );
    println!();

    let n = args.num_games;

    println!("=== Heuristic Gap Analysis ===");
    println!(
        "Games: {} | Seed: {} | Threads: {}",
        n, args.seed, num_threads
    );
    println!();

    // Run analysis in parallel
    let t2 = Instant::now();
    let results: Vec<_> = (0..n as u64)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(args.seed.wrapping_add(i));
            analyze_game(&ctx, &mut rng)
        })
        .collect();
    let elapsed = t2.elapsed();

    // Aggregate results
    let mut all_disagreements: Vec<Disagreement> = Vec::new();
    let mut total_score_sum = 0i64;
    let mut total_cat_mistakes = 0i64;

    for (disagreements, score, cat_mistakes) in &results {
        all_disagreements.extend_from_slice(disagreements);
        total_score_sum += *score as i64;
        total_cat_mistakes += *cat_mistakes as i64;
    }

    let mean_score = total_score_sum as f64 / n as f64;
    let total_disagreements = all_disagreements.len();
    let disagreements_per_game = total_disagreements as f64 / n as f64;
    let total_ev_loss: f64 = all_disagreements.iter().map(|d| d.ev_gap).sum();
    let ev_loss_per_game = total_ev_loss / n as f64;

    println!("Results:");
    println!("  Heuristic mean score: {:.1}", mean_score);
    println!(
        "  Total disagreements: {} ({:.1} per game)",
        total_disagreements, disagreements_per_game
    );
    println!(
        "  Total EV loss: {:.1} ({:.1} per game)",
        total_ev_loss, ev_loss_per_game
    );
    println!(
        "  Category mistakes: {} ({:.2} per game)",
        total_cat_mistakes,
        total_cat_mistakes as f64 / n as f64
    );
    println!("  Elapsed: {:.1}s", elapsed.as_secs_f64());
    println!();

    // Classify and aggregate by pattern
    let mut pattern_map: HashMap<String, MistakePattern> = HashMap::new();
    for d in &all_disagreements {
        let key = classify_disagreement(d);
        let entry = pattern_map.entry(key.clone()).or_default();
        if entry.description.is_empty() {
            entry.decision_type = d.decision_type.to_string();
            entry.description = key;
        }
        entry.count += 1;
        entry.total_ev_gap += d.ev_gap;
        if d.ev_gap > entry.max_ev_gap {
            entry.max_ev_gap = d.ev_gap;
            entry.example_dice = format!("{:?}", d.dice);
        }
    }

    let mut patterns: Vec<MistakePattern> = pattern_map.into_values().collect();
    patterns.sort_by(|a, b| {
        b.total_ev_gap
            .partial_cmp(&a.total_ev_gap)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Print top patterns
    println!("Top Mistake Patterns (by total EV cost per game):");
    println!("  Rank       Count     TotalEV   EV/game    MaxGap  Pattern");
    for (i, p) in patterns.iter().take(20).enumerate() {
        println!(
            "  {:>4}  {:>10}  {:>10.1}  {:>8.2}  {:>8.2}  {}",
            i + 1,
            p.count,
            p.total_ev_gap,
            p.total_ev_gap / n as f64,
            p.max_ev_gap,
            p.description,
        );
    }
    println!();

    // Breakdown by decision type
    let mut by_type: HashMap<&str, (u64, f64)> = HashMap::new();
    for d in &all_disagreements {
        let entry = by_type.entry(d.decision_type).or_default();
        entry.0 += 1;
        entry.1 += d.ev_gap;
    }
    println!("EV Loss Breakdown by Decision Type:");
    let mut type_entries: Vec<_> = by_type.iter().collect();
    type_entries.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
    for (dtype, (count, ev)) in &type_entries {
        println!(
            "  {:>10}: {:>8} disagreements, {:.1} total EV ({:.2} per game)",
            dtype,
            count,
            ev,
            ev / n as f64,
        );
    }

    // Save outputs
    std::fs::create_dir_all(&args.output).unwrap_or_else(|e| {
        eprintln!("Failed to create output directory: {}", e);
        std::process::exit(1);
    });

    // CSV: sample up to 100K disagreements (to keep file manageable)
    let csv_path = format!("{}/heuristic_gap.csv", args.output);
    let mut csv = String::new();
    csv.push_str(
        "turn,decision_type,scored_categories,upper_score,dice,\
         heuristic_action,optimal_action,ev_heuristic,ev_optimal,ev_gap,pattern\n",
    );
    for d in all_disagreements.iter().take(100_000) {
        let pattern = classify_disagreement(d);
        csv.push_str(&format!(
            "{},{},{},{},\"{:?}\",{},{},{:.4},{:.4},{:.4},{}\n",
            d.turn,
            d.decision_type,
            d.scored_categories,
            d.upper_score,
            d.dice,
            d.heuristic_action,
            d.optimal_action,
            d.ev_heuristic,
            d.ev_optimal,
            d.ev_gap,
            pattern,
        ));
    }
    std::fs::write(&csv_path, csv).unwrap();
    println!();
    println!(
        "Saved {} disagreements to {}",
        all_disagreements.len().min(100_000),
        csv_path
    );

    // JSON summary
    let json_path = format!("{}/heuristic_gap_summary.json", args.output);
    let top_patterns: Vec<serde_json::Value> = patterns
        .iter()
        .take(20)
        .map(|p| {
            serde_json::json!({
                "pattern": p.description,
                "decision_type": p.decision_type,
                "count": p.count,
                "total_ev_gap": (p.total_ev_gap * 100.0).round() / 100.0,
                "ev_per_game": ((p.total_ev_gap / n as f64) * 100.0).round() / 100.0,
                "max_ev_gap": (p.max_ev_gap * 100.0).round() / 100.0,
                "example_dice": p.example_dice,
            })
        })
        .collect();

    let type_breakdown: Vec<serde_json::Value> = type_entries
        .iter()
        .map(|(dtype, (count, ev))| {
            serde_json::json!({
                "decision_type": dtype,
                "disagreements": count,
                "total_ev_loss": (ev * 100.0).round() / 100.0,
                "ev_loss_per_game": ((ev / n as f64) * 100.0).round() / 100.0,
            })
        })
        .collect();

    let summary = serde_json::json!({
        "num_games": n,
        "seed": args.seed,
        "heuristic_mean_score": (mean_score * 10.0).round() / 10.0,
        "total_disagreements": total_disagreements,
        "disagreements_per_game": (disagreements_per_game * 100.0).round() / 100.0,
        "total_ev_loss": (total_ev_loss * 10.0).round() / 10.0,
        "ev_loss_per_game": (ev_loss_per_game * 100.0).round() / 100.0,
        "category_mistakes_per_game": ((total_cat_mistakes as f64 / n as f64) * 100.0).round() / 100.0,
        "top_patterns": top_patterns,
        "by_decision_type": type_breakdown,
    });
    std::fs::write(&json_path, serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    println!("Saved summary to {}", json_path);
}
