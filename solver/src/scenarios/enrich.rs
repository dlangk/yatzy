//! Enrichment: θ-sensitivity and Q-grid computation.

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;

use crate::constants::*;
use crate::dice_mechanics::find_dice_set_index;
use crate::game_mechanics::update_upper_score;
use crate::profiling::qvalues::{
    compute_eds_for_scenario, compute_q_categories, compute_q_rerolls, sigma_for_depth,
};
use crate::scenarios::actions::{compute_group6, compute_group6_risk, format_mask};
use crate::scenarios::io::ThetaEntry;
use crate::scenarios::types::*;
use crate::types::YatzyContext;
use crate::widget_solver::{compute_max_ev_for_n_rerolls, compute_opt_lse_for_n_rerolls};

// ── θ-sensitivity enrichment (replaces scenario_sensitivity.rs logic) ──

/// Find best and runner-up reroll mask at a given θ.
fn find_best_and_runner_up_mask(
    ctx: &YatzyContext,
    e_ds: &[f32; 252],
    dice: &[i32; 5],
    is_risk: bool,
) -> (i32, f32, i32, f32) {
    let ds_index = find_dice_set_index(ctx, dice);
    let kt = &ctx.keep_table;
    let vals = &kt.vals;
    let cols = &kt.cols;

    let mut options: Vec<(i32, f32)> = Vec::with_capacity(32);
    options.push((0, e_ds[ds_index]));

    for j in 0..kt.unique_count[ds_index] as usize {
        let kid = kt.unique_keep_ids[ds_index][j] as usize;
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;

        let ev = if is_risk {
            let mut max_x = f32::NEG_INFINITY;
            for k in start..end {
                let v = unsafe { *e_ds.get_unchecked(*cols.get_unchecked(k) as usize) };
                if v > max_x {
                    max_x = v;
                }
            }
            let mut sum: f32 = 0.0;
            for k in start..end {
                unsafe {
                    let v = *e_ds.get_unchecked(*cols.get_unchecked(k) as usize);
                    sum += *vals.get_unchecked(k) * (v - max_x).exp();
                }
            }
            max_x + sum.ln()
        } else {
            let mut ev: f32 = 0.0;
            for k in start..end {
                unsafe {
                    ev += *vals.get_unchecked(k)
                        * e_ds.get_unchecked(*cols.get_unchecked(k) as usize);
                }
            }
            ev
        };

        let mask = kt.keep_to_mask[ds_index * 32 + j];
        options.push((mask, ev));
    }

    options.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let (best_mask, best_ev) = options[0];
    let (ru_mask, ru_ev) = if options.len() > 1 {
        options[1]
    } else {
        (0, best_ev)
    };

    (best_mask, best_ev, ru_mask, ru_ev)
}

/// Find best and runner-up category at a given θ.
fn find_best_and_runner_up_category(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    ds_index: usize,
    theta: f32,
    is_last_turn: bool,
) -> (usize, f32, usize, f32) {
    let mut options: Vec<(usize, f32)> = Vec::with_capacity(CATEGORY_COUNT);

    if is_last_turn {
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
                let val = if theta == 0.0 {
                    (scr + bonus) as f32
                } else {
                    theta * (scr + bonus) as f32
                };
                options.push((c, val));
            }
        }
    } else if theta == 0.0 {
        for c in 0..6 {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let new_up = update_upper_score(up_score, c, scr);
                let new_scored = scored | (1 << c);
                let val = scr as f32
                    + unsafe {
                        *sv.get_unchecked(state_index(new_up as usize, new_scored as usize))
                    };
                options.push((c, val));
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
                options.push((c, val));
            }
        }
    } else {
        for c in 0..6 {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let new_up = update_upper_score(up_score, c, scr);
                let new_scored = scored | (1 << c);
                let val = theta * scr as f32
                    + unsafe {
                        *sv.get_unchecked(state_index(new_up as usize, new_scored as usize))
                    };
                options.push((c, val));
            }
        }
        for c in 6..CATEGORY_COUNT {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let new_scored = scored | (1 << c);
                let val = theta * scr as f32
                    + unsafe {
                        *sv.get_unchecked(state_index(up_score as usize, new_scored as usize))
                    };
                options.push((c, val));
            }
        }
    }

    options.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let (best_cat, best_val) = options[0];
    let (ru_cat, ru_val) = if options.len() > 1 {
        options[1]
    } else {
        (best_cat, best_val)
    };

    (best_cat, best_val, ru_cat, ru_val)
}

/// Parameters for evaluating a scenario at a given θ.
pub struct ScenarioParams<'a> {
    pub upper_score: i32,
    pub scored_categories: i32,
    pub dice: &'a [i32; 5],
    pub turn: usize,
    pub decision_type: &'a str,
}

/// Evaluate a single scenario at a single θ, returning best/runner-up.
pub fn evaluate_scenario_at_theta(
    ctx: &YatzyContext,
    sv: &[f32],
    theta: f32,
    params: &ScenarioParams,
) -> ThetaResult {
    let ds_index = find_dice_set_index(ctx, params.dice);
    let is_last = params.turn == CATEGORY_COUNT - 1;
    let is_risk = theta != 0.0;

    let (action_str, action_id, value, ru_str, ru_id, ru_value) = match params.decision_type {
        "reroll1" => {
            let mut e_ds_0 = [0.0f32; 252];
            let mut e_ds_1 = [0.0f32; 252];
            if is_risk {
                compute_group6_risk(
                    ctx,
                    sv,
                    params.upper_score,
                    params.scored_categories,
                    theta,
                    &mut e_ds_0,
                );
                compute_opt_lse_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1, false);
            } else {
                compute_group6(
                    ctx,
                    sv,
                    params.upper_score,
                    params.scored_categories,
                    &mut e_ds_0,
                );
                compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);
            }
            let (bm, bv, rm, rv) =
                find_best_and_runner_up_mask(ctx, &e_ds_1, params.dice, is_risk);
            (
                format_mask(bm, params.dice),
                bm,
                bv,
                format_mask(rm, params.dice),
                rm,
                rv,
            )
        }
        "reroll2" => {
            let mut e_ds_0 = [0.0f32; 252];
            if is_risk {
                compute_group6_risk(
                    ctx,
                    sv,
                    params.upper_score,
                    params.scored_categories,
                    theta,
                    &mut e_ds_0,
                );
            } else {
                compute_group6(
                    ctx,
                    sv,
                    params.upper_score,
                    params.scored_categories,
                    &mut e_ds_0,
                );
            }
            let (bm, bv, rm, rv) =
                find_best_and_runner_up_mask(ctx, &e_ds_0, params.dice, is_risk);
            (
                format_mask(bm, params.dice),
                bm,
                bv,
                format_mask(rm, params.dice),
                rm,
                rv,
            )
        }
        "category" => {
            let (bc, bv, rc, rv) = find_best_and_runner_up_category(
                ctx,
                sv,
                params.upper_score,
                params.scored_categories,
                ds_index,
                theta,
                is_last,
            );
            (
                CATEGORY_NAMES[bc].to_string(),
                bc as i32,
                bv,
                CATEGORY_NAMES[rc].to_string(),
                rc as i32,
                rv,
            )
        }
        _ => panic!("Unknown decision type: {}", params.decision_type),
    };

    ThetaResult {
        theta,
        action: action_str,
        action_id,
        value,
        runner_up: ru_str,
        runner_up_id: ru_id,
        runner_up_value: ru_value,
        gap: value - ru_value,
    }
}

/// Evaluate a scenario across all θ entries and detect flips.
pub fn evaluate_scenario_sensitivity(
    ctx: &YatzyContext,
    theta_entries: &[ThetaEntry],
    params: &ScenarioParams,
) -> (Vec<ThetaResult>, bool, f32, String, i32, f32, f32) {
    let mut theta_results: Vec<ThetaResult> = Vec::with_capacity(theta_entries.len());

    for entry in theta_entries {
        let sv = entry.sv.as_slice();
        let tr = evaluate_scenario_at_theta(ctx, sv, entry.theta, params);
        theta_results.push(tr);
    }

    // Find θ=0 baseline
    let theta0_result = theta_results
        .iter()
        .find(|r| r.theta == 0.0)
        .expect("θ=0 missing from results");
    let theta0_action_id = theta0_result.action_id;
    let theta0_gap = theta0_result.gap;

    // Detect first flip
    let mut has_flip = false;
    let mut flip_theta: f32 = 0.0;
    let mut flip_action = String::new();
    let mut flip_action_id: i32 = 0;
    let mut gap_at_flip: f32 = 0.0;

    for tr in &theta_results {
        if tr.theta != 0.0 && !has_flip && tr.action_id != theta0_action_id {
            has_flip = true;
            flip_theta = tr.theta;
            flip_action = tr.action.clone();
            flip_action_id = tr.action_id;
            gap_at_flip = tr.gap;
        }
    }

    (
        theta_results,
        has_flip,
        flip_theta,
        flip_action,
        flip_action_id,
        gap_at_flip,
        theta0_gap,
    )
}

// ── Q-grid enrichment (replaces profiling q_grid computation) ──

/// Compute Q-value grid for a single scenario across parameter combinations.
pub fn compute_q_grid(
    ctx: &YatzyContext,
    scenario: &ProfilingScenario,
    theta_tables: &[(f32, crate::types::StateValues)],
    gamma_values: &[f32],
    d_values: &[u32],
) -> HashMap<String, Vec<f32>> {
    let mut grid: HashMap<String, Vec<f32>> = HashMap::new();

    for &gamma in gamma_values {
        for &d in d_values {
            let sigma = sigma_for_depth(d);

            for (theta_val, theta_sv) in theta_tables {
                let sv = theta_sv.as_slice();
                let is_risk = *theta_val != 0.0;
                let ds_index = find_dice_set_index(ctx, &scenario.dice);
                let is_last = scenario.turn == CATEGORY_COUNT - 1;

                let q_values: Vec<f32> = match scenario.decision_type {
                    DecisionType::Category => {
                        let cats = compute_q_categories(
                            ctx,
                            sv,
                            scenario.upper_score,
                            scenario.scored_categories,
                            ds_index,
                            gamma,
                            sigma,
                            is_last,
                        );
                        scenario
                            .actions
                            .iter()
                            .map(|a| {
                                cats.iter()
                                    .find(|(c, _)| *c as i32 == a.id)
                                    .map(|(_, v)| *v)
                                    .unwrap_or(f32::NEG_INFINITY)
                            })
                            .collect()
                    }
                    DecisionType::Reroll1 | DecisionType::Reroll2 => {
                        let (e_ds_0, e_ds_1) = compute_eds_for_scenario(
                            ctx,
                            sv,
                            scenario.upper_score,
                            scenario.scored_categories,
                            *theta_val,
                            gamma,
                            sigma,
                        );
                        let e_ds = match scenario.decision_type {
                            DecisionType::Reroll1 => &e_ds_1,
                            _ => &e_ds_0,
                        };
                        let rerolls = compute_q_rerolls(ctx, e_ds, &scenario.dice, is_risk);

                        scenario
                            .actions
                            .iter()
                            .map(|a| {
                                rerolls
                                    .iter()
                                    .find(|(m, _)| *m == a.id)
                                    .map(|(_, v)| *v)
                                    .unwrap_or(f32::NEG_INFINITY)
                            })
                            .collect()
                    }
                };

                let key = format!("{},{},{}", theta_val, gamma, d);
                grid.insert(key, q_values);
            }
        }
    }

    grid
}
