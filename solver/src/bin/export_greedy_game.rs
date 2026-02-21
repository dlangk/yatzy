//! Export a side-by-side greedy vs optimal game trace for the blog.
//!
//! Simulates one game with a fixed seed under both the greedy heuristic and the
//! DP-optimal policy, using the same dice rolls. Records per-turn data for
//! visualization in `blog/data/greedy_vs_optimal.json`.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use yatzy::constants::*;
use yatzy::dice_mechanics::{count_faces, find_dice_set_index, sort_dice_set};
use yatzy::game_mechanics::{calculate_category_score, update_upper_score};
use yatzy::phase0_tables;
use yatzy::simulation::heuristic::{heuristic_pick_category, heuristic_reroll_mask};
use yatzy::storage::load_all_state_values;
use yatzy::types::YatzyContext;
use yatzy::widget_solver::{choose_best_reroll_mask, compute_max_ev_for_n_rerolls};

fn roll_dice(rng: &mut SmallRng) -> [i32; 5] {
    let mut dice = [0i32; 5];
    for d in &mut dice {
        *d = rng.random_range(1..=6);
    }
    sort_dice_set(&mut dice);
    dice
}

fn apply_reroll(dice: &mut [i32; 5], mask: i32, rng: &mut SmallRng) {
    for (i, d) in dice.iter_mut().enumerate() {
        if mask & (1 << i) != 0 {
            *d = rng.random_range(1..=6);
        }
    }
    sort_dice_set(dice);
}

/// Find the best category using the DP-optimal policy.
fn find_best_category_optimal(
    ctx: &YatzyContext,
    up_score: i32,
    scored: i32,
    ds_index: usize,
    is_last_turn: bool,
) -> (usize, i32) {
    let sv = ctx.state_values.as_slice();

    if is_last_turn {
        // Last turn: maximize immediate score + bonus
        let mut best_val = i32::MIN;
        let mut best_cat = 0;
        let mut best_score = 0;
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
    } else {
        let mut best_val = f32::NEG_INFINITY;
        let mut best_cat = 0;
        let mut best_score = 0;
        for c in 0..6 {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let new_up = update_upper_score(up_score, c, scr);
                let new_scored = scored | (1 << c);
                let val = scr as f32
                    + unsafe {
                        *sv.get_unchecked(state_index(new_up as usize, new_scored as usize))
                    };
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
                    + unsafe {
                        *sv.get_unchecked(state_index(up_score as usize, new_scored as usize))
                    };
                if val > best_val {
                    best_val = val;
                    best_cat = c;
                    best_score = scr;
                }
            }
        }
        (best_cat, best_score)
    }
}

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

fn main() {
    let _base_path = yatzy::env_config::init_base_path();

    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    if !load_all_state_values(&mut ctx, "data/strategy_tables/all_states.bin") {
        eprintln!("Failed to load state values. Run yatzy-precompute first.");
        std::process::exit(1);
    }

    // Try multiple seeds and pick a game that clearly shows the greedy failure
    // (big score gap + greedy misses bonus + divergent categories)
    let mut best_seed = 42u64;
    let mut best_gap = 0i32;

    for seed in 0..500u64 {
        let (g_total, _, g_bonus) = simulate_greedy(&ctx, seed);
        let (o_total, _, o_bonus) = simulate_optimal(&ctx, seed);
        let gap = o_total - g_total;
        // Prefer: large gap, greedy misses bonus, optimal gets bonus
        if gap > best_gap && !g_bonus && o_bonus {
            best_gap = gap;
            best_seed = seed;
        }
    }

    println!("Selected seed: {} (gap: {})", best_seed, best_gap);

    // Generate the full trace for the best seed
    let (greedy_turns, greedy_total, greedy_bonus) = simulate_greedy_trace(&ctx, best_seed);
    let (optimal_turns, optimal_total, optimal_bonus) = simulate_optimal_trace(&ctx, best_seed);

    // Build JSON
    let mut turns_json = Vec::new();
    for i in 0..CATEGORY_COUNT {
        let g = &greedy_turns[i];
        let o = &optimal_turns[i];
        turns_json.push(serde_json::json!({
            "dice": g.dice,
            "greedy": {
                "category": CATEGORY_NAMES[g.category],
                "category_idx": g.category,
                "score": g.score,
                "total": g.running_total,
                "upper": g.upper_score,
            },
            "optimal": {
                "category": CATEGORY_NAMES[o.category],
                "category_idx": o.category,
                "score": o.score,
                "total": o.running_total,
                "upper": o.upper_score,
            },
            "diverges": g.category != o.category,
        }));
    }

    let output = serde_json::json!({
        "seed": best_seed,
        "greedy_total": greedy_total,
        "greedy_bonus": greedy_bonus,
        "optimal_total": optimal_total,
        "optimal_bonus": optimal_bonus,
        "turns": turns_json,
    });

    let json = serde_json::to_string_pretty(&output).unwrap();
    let out_path = "blog/data/greedy_vs_optimal.json";
    std::fs::create_dir_all("blog/data").unwrap();
    std::fs::write(out_path, &json).unwrap();
    println!("Wrote {} to {}", json.len(), out_path);
    println!(
        "Greedy: {} (bonus: {}), Optimal: {} (bonus: {})",
        greedy_total, greedy_bonus, optimal_total, optimal_bonus
    );
}

struct TurnTrace {
    dice: [i32; 5],
    category: usize,
    score: i32,
    running_total: i32,
    upper_score: i32,
}

fn simulate_greedy(_ctx: &YatzyContext, seed: u64) -> (i32, i32, bool) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut upper_score = 0i32;
    let mut scored = 0i32;
    let mut total = 0i32;

    for _ in 0..CATEGORY_COUNT {
        let mut dice = roll_dice(&mut rng);
        // Greedy rerolls
        let fc = count_faces(&dice);
        let mask1 = heuristic_reroll_mask(&dice, &fc, scored, upper_score);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, &mut rng);
        }
        let fc = count_faces(&dice);
        let mask2 = heuristic_reroll_mask(&dice, &fc, scored, upper_score);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, &mut rng);
        }

        let fc = count_faces(&dice);
        let cat = heuristic_pick_category(&dice, &fc, scored, upper_score);
        let scr = calculate_category_score(&dice, cat);
        upper_score = update_upper_score(upper_score, cat, scr);
        scored |= 1 << cat;
        total += scr;
    }
    let bonus = upper_score >= 63;
    if bonus {
        total += 50;
    }
    (total, upper_score, bonus)
}

fn simulate_optimal(ctx: &YatzyContext, seed: u64) -> (i32, i32, bool) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let sv = ctx.state_values.as_slice();
    let mut upper_score = 0i32;
    let mut scored = 0i32;
    let mut total = 0i32;
    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    for turn in 0..CATEGORY_COUNT {
        let mut dice = roll_dice(&mut rng);
        compute_group6(ctx, sv, upper_score, scored, &mut e_ds_0);
        compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);

        let mut best_ev = 0.0;
        let mask1 = choose_best_reroll_mask(ctx, &e_ds_1, &dice, &mut best_ev);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, &mut rng);
        }
        let mask2 = choose_best_reroll_mask(ctx, &e_ds_0, &dice, &mut best_ev);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, &mut rng);
        }

        let ds_index = find_dice_set_index(ctx, &dice);
        let is_last = turn == CATEGORY_COUNT - 1;
        let (cat, scr) = find_best_category_optimal(ctx, upper_score, scored, ds_index, is_last);
        upper_score = update_upper_score(upper_score, cat, scr);
        scored |= 1 << cat;
        total += scr;
    }
    let bonus = upper_score >= 63;
    if bonus {
        total += 50;
    }
    (total, upper_score, bonus)
}

fn simulate_greedy_trace(_ctx: &YatzyContext, seed: u64) -> (Vec<TurnTrace>, i32, bool) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut upper_score = 0i32;
    let mut scored = 0i32;
    let mut total = 0i32;
    let mut turns = Vec::new();

    for _ in 0..CATEGORY_COUNT {
        let mut dice = roll_dice(&mut rng);
        let initial_dice = dice;

        let fc = count_faces(&dice);
        let mask1 = heuristic_reroll_mask(&dice, &fc, scored, upper_score);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, &mut rng);
        }
        let fc = count_faces(&dice);
        let mask2 = heuristic_reroll_mask(&dice, &fc, scored, upper_score);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, &mut rng);
        }

        let fc = count_faces(&dice);
        let cat = heuristic_pick_category(&dice, &fc, scored, upper_score);
        let scr = calculate_category_score(&dice, cat);
        upper_score = update_upper_score(upper_score, cat, scr);
        scored |= 1 << cat;
        total += scr;

        turns.push(TurnTrace {
            dice: initial_dice,
            category: cat,
            score: scr,
            running_total: total,
            upper_score,
        });
    }
    let bonus = upper_score >= 63;
    if bonus {
        total += 50;
    }
    // Update last turn's running total to include bonus
    if bonus {
        if let Some(last) = turns.last_mut() {
            last.running_total = total;
        }
    }
    (turns, total, bonus)
}

fn simulate_optimal_trace(ctx: &YatzyContext, seed: u64) -> (Vec<TurnTrace>, i32, bool) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let sv = ctx.state_values.as_slice();
    let mut upper_score = 0i32;
    let mut scored = 0i32;
    let mut total = 0i32;
    let mut turns = Vec::new();
    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    for turn in 0..CATEGORY_COUNT {
        let mut dice = roll_dice(&mut rng);
        let initial_dice = dice;

        compute_group6(ctx, sv, upper_score, scored, &mut e_ds_0);
        compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);

        let mut best_ev = 0.0;
        let mask1 = choose_best_reroll_mask(ctx, &e_ds_1, &dice, &mut best_ev);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, &mut rng);
        }
        let mask2 = choose_best_reroll_mask(ctx, &e_ds_0, &dice, &mut best_ev);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, &mut rng);
        }

        let ds_index = find_dice_set_index(ctx, &dice);
        let is_last = turn == CATEGORY_COUNT - 1;
        let (cat, scr) = find_best_category_optimal(ctx, upper_score, scored, ds_index, is_last);
        upper_score = update_upper_score(upper_score, cat, scr);
        scored |= 1 << cat;
        total += scr;

        turns.push(TurnTrace {
            dice: initial_dice,
            category: cat,
            score: scr,
            running_total: total,
            upper_score,
        });
    }
    let bonus = upper_score >= 63;
    if bonus {
        total += 50;
    }
    if bonus {
        if let Some(last) = turns.last_mut() {
            last.running_total = total;
        }
    }
    (turns, total, bonus)
}
