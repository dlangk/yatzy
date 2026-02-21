//! Unified candidate collection from simulated games.
//!
//! Always tracks `category_scores: [i32; 15]` during play.
//! Supports both optimal and noisy (profiling) simulation modes.

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::constants::*;
use crate::dice_mechanics::find_dice_set_index;
use crate::game_mechanics::update_upper_score;
use crate::profiling::qvalues::{
    compute_group6_profiling, compute_q_categories, compute_q_rerolls, softmax_sample,
};
use crate::scenarios::actions::*;
use crate::scenarios::classify::{is_realistic, validate_candidate};
use crate::scenarios::types::*;
use crate::types::YatzyContext;
use crate::widget_solver::{choose_best_reroll_mask, compute_max_ev_for_n_rerolls};

/// Simulate one game under θ=0 optimal play, collecting all 3 decision types per turn.
/// Always tracks category_scores.
fn simulate_game_optimal(ctx: &YatzyContext, rng: &mut SmallRng) -> Vec<RawDecision> {
    let sv = ctx.state_values.as_slice();
    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut category_scores = [-1i32; 15];
    let mut decisions = Vec::with_capacity(45);

    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;
        let mut dice = roll_dice(rng);

        compute_group6(ctx, sv, up_score, scored, &mut e_ds_0);
        compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);

        // Decision 1: Reroll1
        decisions.push(RawDecision {
            upper_score: up_score,
            scored,
            dice,
            turn,
            decision_type: DecisionType::Reroll1,
            category_scores,
        });

        let mut best_ev = 0.0;
        let mask1 = choose_best_reroll_mask(ctx, &e_ds_1, &dice, &mut best_ev);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }

        // Decision 2: Reroll2
        decisions.push(RawDecision {
            upper_score: up_score,
            scored,
            dice,
            turn,
            decision_type: DecisionType::Reroll2,
            category_scores,
        });

        let mask2 = choose_best_reroll_mask(ctx, &e_ds_0, &dice, &mut best_ev);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }

        // Decision 3: Category
        decisions.push(RawDecision {
            upper_score: up_score,
            scored,
            dice,
            turn,
            decision_type: DecisionType::Category,
            category_scores,
        });

        let ds_index = find_dice_set_index(ctx, &dice);
        let (cat, scr) = if is_last_turn {
            find_best_category_final(ctx, up_score, scored, ds_index)
        } else {
            find_best_category(ctx, sv, up_score, scored, ds_index)
        };

        category_scores[cat] = scr;
        up_score = update_upper_score(up_score, cat, scr);
        scored |= 1 << cat;
    }

    decisions
}

/// Simulate one game using a noisy "competent human" agent.
/// Always tracks category_scores.
fn simulate_game_noisy(
    ctx: &YatzyContext,
    rng: &mut SmallRng,
    beta_sim: f32,
    gamma_sim: f32,
    sigma_d: f32,
) -> Vec<RawDecision> {
    let sv = ctx.state_values.as_slice();
    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut category_scores = [-1i32; 15];
    let mut decisions = Vec::with_capacity(45);

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;
        let mut dice = roll_dice(rng);

        // Record decision 1: Reroll1
        decisions.push(RawDecision {
            upper_score: up_score,
            scored,
            dice,
            turn,
            decision_type: DecisionType::Reroll1,
            category_scores,
        });

        let mut e_ds_0 = [0.0f32; 252];
        let mut e_ds_1 = [0.0f32; 252];
        compute_group6_profiling(ctx, sv, up_score, scored, gamma_sim, sigma_d, &mut e_ds_0);
        compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);

        let reroll1_qs = compute_q_rerolls(ctx, &e_ds_1, &dice, false);
        let mask1 = softmax_sample(&reroll1_qs, beta_sim, rng);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }

        // Record decision 2: Reroll2
        decisions.push(RawDecision {
            upper_score: up_score,
            scored,
            dice,
            turn,
            decision_type: DecisionType::Reroll2,
            category_scores,
        });

        let reroll2_qs = compute_q_rerolls(ctx, &e_ds_0, &dice, false);
        let mask2 = softmax_sample(&reroll2_qs, beta_sim, rng);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }

        // Record decision 3: Category
        decisions.push(RawDecision {
            upper_score: up_score,
            scored,
            dice,
            turn,
            decision_type: DecisionType::Category,
            category_scores,
        });

        let ds_index = find_dice_set_index(ctx, &dice);
        let (cat, scr) = if is_last_turn {
            find_best_category_final(ctx, up_score, scored, ds_index)
        } else {
            let cat_qs = compute_q_categories(
                ctx, sv, up_score, scored, ds_index, gamma_sim, sigma_d, false,
            );
            let chosen_cat = softmax_sample(
                &cat_qs
                    .iter()
                    .map(|(c, v)| (*c as i32, *v))
                    .collect::<Vec<_>>(),
                beta_sim,
                rng,
            ) as usize;
            let scr = ctx.precomputed_scores[ds_index][chosen_cat];
            (chosen_cat, scr)
        };

        category_scores[cat] = scr;
        up_score = update_upper_score(up_score, cat, scr);
        scored |= 1 << cat;
    }

    decisions
}

/// Noisy simulation parameters.
pub struct NoisyParams {
    pub beta: f32,
    pub gamma: f32,
    pub sigma_d: f32,
}

/// Collect candidate scenarios from simulated games.
///
/// If `noisy` is Some, uses noisy simulation; otherwise optimal play.
/// Always tracks category_scores. Returns deduped decisions with visit counts.
pub fn collect_candidates(
    ctx: &YatzyContext,
    num_games: usize,
    seed: u64,
    noisy: Option<&NoisyParams>,
) -> HashMap<DecisionKey, (RawDecision, usize)> {
    let batch_size = 100_000usize;
    let num_batches = num_games.div_ceil(batch_size);

    let mut visit_counts: HashMap<DecisionKey, (RawDecision, usize)> = HashMap::new();
    let mut games_done = 0usize;

    for batch in 0..num_batches {
        let batch_start = batch * batch_size;
        let batch_end = (batch_start + batch_size).min(num_games);
        let batch_games = batch_end - batch_start;

        let batch_decisions: Vec<RawDecision> = (batch_start..batch_end)
            .into_par_iter()
            .flat_map_iter(|i| {
                let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
                match noisy {
                    Some(p) => simulate_game_noisy(ctx, &mut rng, p.beta, p.gamma, p.sigma_d),
                    None => simulate_game_optimal(ctx, &mut rng),
                }
            })
            .collect();

        for d in &batch_decisions {
            let key = decision_key(d);
            visit_counts
                .entry(key)
                .and_modify(|e| e.1 += 1)
                .or_insert_with(|| (d.clone(), 1));
        }

        games_done += batch_games;
        if (batch + 1) % 2 == 0 || batch == num_batches - 1 {
            println!(
                "  Batch {}/{}: {} games, {} unique scenarios",
                batch + 1,
                num_batches,
                games_done,
                visit_counts.len(),
            );
        }
    }

    visit_counts
}

/// Filter and validate candidates, returning only viable ones.
///
/// Filters by min_visits, is_realistic, and >= 2 legal actions.
/// Logs validation failures.
pub fn filter_and_validate(
    ctx: &YatzyContext,
    candidates: HashMap<DecisionKey, (RawDecision, usize)>,
    min_visits: usize,
    require_realistic: bool,
) -> Vec<(RawDecision, usize)> {
    let sv = ctx.state_values.as_slice();
    let mut valid = Vec::new();
    let mut validation_failures = 0usize;
    let mut filtered_visits = 0usize;
    let mut filtered_realistic = 0usize;
    let mut filtered_actions = 0usize;

    for (d, count) in candidates.into_values() {
        if count < min_visits {
            filtered_visits += 1;
            continue;
        }

        if require_realistic && !is_realistic(d.upper_score, d.scored, d.turn, &d.category_scores) {
            filtered_realistic += 1;
            continue;
        }

        // Validate
        if let Some(failure) = validate_candidate(&d) {
            validation_failures += 1;
            if validation_failures <= 5 {
                eprintln!(
                    "  VALIDATION FAILURE: turn={}, scored=0x{:04X}, up={}: {}",
                    d.turn, d.scored, d.upper_score, failure
                );
            }
            continue;
        }

        // Check >= 2 legal actions
        let actions = get_actions(ctx, sv, &d, false);
        if actions.len() < 2 {
            filtered_actions += 1;
            continue;
        }

        valid.push((d, count));
    }

    if validation_failures > 0 {
        eprintln!(
            "  {} validation failures (BUG — investigate)",
            validation_failures
        );
    }
    if filtered_visits > 0 || filtered_realistic > 0 || filtered_actions > 0 {
        println!(
            "  Filtered: {} visits, {} realistic, {} actions, {} validation",
            filtered_visits, filtered_realistic, filtered_actions, validation_failures
        );
    }

    valid
}
