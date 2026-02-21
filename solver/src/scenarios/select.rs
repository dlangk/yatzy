//! Scenario selection: difficulty-based and diagnostic-based modes.

use std::collections::HashMap;

use crate::constants::*;
use crate::dice_mechanics::find_dice_set_index;
use crate::scenarios::actions::*;
use crate::scenarios::classify::*;
use crate::scenarios::types::*;
use crate::types::{StateValues, YatzyContext};

// ── Difficulty selection (replaces difficult_scenarios.rs ranking) ──

/// Analyze candidates and select top N by difficulty score, stratified by turn.
pub fn select_by_difficulty(
    ctx: &YatzyContext,
    candidates: &[(RawDecision, usize)],
    top_n: usize,
) -> Vec<Candidate> {
    let sv = ctx.state_values.as_slice();

    // Score all candidates
    let mut scored: Vec<Candidate> = candidates
        .iter()
        .filter_map(|(d, count)| {
            let actions = get_actions(ctx, sv, d, false);
            if actions.len() < 2 {
                return None;
            }

            let gap = actions[0].ev - actions[1].ev;
            let difficulty_score = *count as f64 / (gap as f64).max(0.01);
            let phase = classify_phase(d.turn);
            let tension = classify_tension(d.scored, d.upper_score, &d.dice);
            let fingerprint = compute_fingerprint(ctx, sv, d, &actions);
            let top_labels: Vec<String> = actions.iter().take(2).map(|a| a.label.clone()).collect();

            Some(Candidate {
                decision: d.clone(),
                visit_count: *count,
                actions,
                ev_gap: gap,
                difficulty_score,
                game_phase: phase,
                board_tension: tension,
                diagnostic_scores: None,
                quadrant: None,
                fingerprint,
                top_action_labels: top_labels,
            })
        })
        .collect();

    // Stratified selection: equal allocation per turn, hardest within each
    let mut by_turn: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, c) in scored.iter().enumerate() {
        by_turn.entry(c.decision.turn).or_default().push(i);
    }
    for indices in by_turn.values_mut() {
        indices.sort_by(|&a, &b| {
            scored[b]
                .difficulty_score
                .partial_cmp(&scored[a].difficulty_score)
                .unwrap()
        });
    }

    let active_turns = by_turn.len();
    let base_per_turn = top_n / active_turns.max(1);
    let mut remainder = top_n - base_per_turn * active_turns;

    let mut selected_indices: Vec<usize> = Vec::with_capacity(top_n);
    let mut leftover_indices: Vec<usize> = Vec::new();
    let mut turn_keys: Vec<usize> = by_turn.keys().copied().collect();
    turn_keys.sort();

    for turn in &turn_keys {
        let indices = by_turn.remove(turn).unwrap();
        let take = if indices.len() < base_per_turn {
            remainder += base_per_turn - indices.len();
            indices.len()
        } else {
            base_per_turn
        };
        selected_indices.extend_from_slice(&indices[..take]);
        leftover_indices.extend_from_slice(&indices[take..]);
    }

    // Fill remainder from leftover pool
    leftover_indices.sort_by(|&a, &b| {
        scored[b]
            .difficulty_score
            .partial_cmp(&scored[a].difficulty_score)
            .unwrap()
    });
    selected_indices.extend(leftover_indices.iter().take(remainder));

    // Final sort and truncate
    selected_indices.sort_by(|&a, &b| {
        scored[b]
            .difficulty_score
            .partial_cmp(&scored[a].difficulty_score)
            .unwrap()
    });
    selected_indices.truncate(top_n);

    // Extract candidates in order
    // We need to take ownership, so swap_remove in reverse order
    selected_indices.sort();
    let mut result = Vec::with_capacity(selected_indices.len());
    for &i in selected_indices.iter().rev() {
        result.push(scored.swap_remove(i));
    }
    result.reverse();

    // Re-sort by difficulty score desc and assign ranks
    result.sort_by(|a, b| b.difficulty_score.partial_cmp(&a.difficulty_score).unwrap());
    result
}

// ── Diagnostic selection (replaces profiling/scenarios.rs assembly) ──

/// Build master pool from candidates with diagnostic scoring.
pub fn build_master_pool(
    ctx: &YatzyContext,
    candidates: Vec<(RawDecision, usize)>,
    theta_tables: &[(f32, StateValues)],
    max_per_bucket: usize,
) -> Vec<ScoredCandidate> {
    let sv = ctx.state_values.as_slice();

    let min_visits_early_late = 5usize;
    let min_visits_mid = 2usize;

    let viable: Vec<(RawDecision, usize)> = candidates
        .into_iter()
        .filter(|(d, count)| {
            let min_visits = match classify_phase(d.turn) {
                GamePhase::Mid => min_visits_mid,
                _ => min_visits_early_late,
            };
            *count >= min_visits && d.turn > 0 && d.turn < CATEGORY_COUNT - 1
        })
        .collect();

    println!(
        "  {} viable candidates (visits >= {}/{})",
        viable.len(),
        min_visits_early_late,
        min_visits_mid,
    );

    let theta_slices: Vec<(f32, &[f32])> = theta_tables
        .iter()
        .map(|(t, sv)| (*t, sv.as_slice()))
        .collect();

    let mut scored_candidates: Vec<ScoredCandidate> = Vec::new();
    let mut skipped_few_actions = 0usize;

    for (d, count) in &viable {
        let ds_index = find_dice_set_index(ctx, &d.dice);
        let is_last = d.turn == CATEGORY_COUNT - 1;

        let actions = match d.decision_type {
            DecisionType::Reroll1 => {
                let mut e_ds_0 = [0.0f32; 252];
                let mut e_ds_1 = [0.0f32; 252];
                compute_group6(ctx, sv, d.upper_score, d.scored, &mut e_ds_0);
                compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);
                enumerate_reroll_actions(ctx, &e_ds_1, &d.dice, true)
            }
            DecisionType::Reroll2 => {
                let mut e_ds_0 = [0.0f32; 252];
                compute_group6(ctx, sv, d.upper_score, d.scored, &mut e_ds_0);
                enumerate_reroll_actions(ctx, &e_ds_0, &d.dice, true)
            }
            DecisionType::Category => {
                enumerate_category_actions(ctx, sv, d.upper_score, d.scored, ds_index, is_last)
            }
        };

        if actions.len() < 2 {
            skipped_few_actions += 1;
            continue;
        }

        let gap = actions[0].ev - actions[1].ev;
        if gap < 0.1 {
            continue;
        }

        let bucket = classify_bucket(d);
        let diag_scores = compute_diagnostic_scores(ctx, sv, d, gap, &theta_slices);

        if diag_scores.max_score() < 0.01 && gap < 0.5 {
            continue;
        }

        let fingerprint = compute_fingerprint(ctx, sv, d, &actions);
        let top_action_labels: Vec<String> =
            actions.iter().take(2).map(|a| a.label.clone()).collect();

        scored_candidates.push(ScoredCandidate {
            decision: d.clone(),
            visit_count: *count,
            bucket,
            scores: diag_scores,
            ev_gap: gap,
            fingerprint,
            top_action_labels,
        });
    }

    if skipped_few_actions > 0 {
        println!(
            "  Skipped {} candidates with < 2 actions",
            skipped_few_actions
        );
    }
    println!(
        "  {} candidates with diagnostic scores",
        scored_candidates.len()
    );

    // Group by bucket, keep top N per bucket
    let mut buckets: HashMap<SemanticBucket, Vec<usize>> = HashMap::new();
    for (i, sc) in scored_candidates.iter().enumerate() {
        buckets.entry(sc.bucket).or_default().push(i);
    }

    let mut pool_indices: Vec<usize> = Vec::new();
    for (bucket, mut indices) in buckets {
        indices.sort_by(|&a, &b| {
            let sa = &scored_candidates[a];
            let sb = &scored_candidates[b];
            sb.scores
                .max_score()
                .partial_cmp(&sa.scores.max_score())
                .unwrap()
                .then(sb.visit_count.cmp(&sa.visit_count))
        });
        let take = indices.len().min(max_per_bucket);
        pool_indices.extend_from_slice(&indices[..take]);
        if !indices.is_empty() {
            println!(
                "    {:?}: {} candidates, keeping {}",
                bucket,
                indices.len(),
                take
            );
        }
    }

    pool_indices.sort();
    pool_indices.dedup();

    let mut pool: Vec<ScoredCandidate> = Vec::with_capacity(pool_indices.len());
    for i in pool_indices.into_iter().rev() {
        pool.push(scored_candidates.swap_remove(i));
    }

    println!("  Master pool: {} candidates", pool.len());
    pool
}

/// Assemble quiz of N scenarios from the master pool with diversity constraints.
pub fn assemble_quiz(pool: &[ScoredCandidate], n_scenarios: usize) -> Vec<usize> {
    if pool.len() <= n_scenarios {
        return (0..pool.len()).collect();
    }

    let mut selected: Vec<usize> = Vec::with_capacity(n_scenarios);
    let mut used: Vec<bool> = vec![false; pool.len()];

    let mut phase_count = [0usize; 3];
    let mut dtype_count = [0usize; 3];
    let mut theta_count = 0usize;
    let mut gamma_count = 0usize;
    let mut depth_count = 0usize;
    let mut beta_count = 0usize;
    let mut turn_set: Vec<usize> = Vec::new();

    let phase_idx = |p: GamePhase| -> usize {
        match p {
            GamePhase::Early => 0,
            GamePhase::Mid => 1,
            GamePhase::Late => 2,
        }
    };
    let dtype_idx = |d: DecisionType| -> usize {
        match d {
            DecisionType::Reroll1 => 0,
            DecisionType::Reroll2 => 1,
            DecisionType::Category => 2,
        }
    };

    let min_per_phase = 4usize;
    let min_per_dtype = 6usize;
    let min_theta = 5usize;
    let min_gamma = 5usize;
    let min_depth = 4usize;
    let min_beta = 4usize;
    let max_theta = 12usize;
    let max_gamma = 12usize;
    let max_action_fatigue = 4usize;

    let mut action_label_count: HashMap<String, usize> = HashMap::new();

    let theta_thresh = 0.5;
    let gamma_thresh = 0.5;
    let beta_thresh = 0.15;

    let find_best = |used: &[bool],
                     turn_set: &[usize],
                     theta_count: usize,
                     gamma_count: usize,
                     action_label_count: &HashMap<String, usize>,
                     pred: &dyn Fn(usize) -> bool|
     -> Option<usize> {
        let mut best_idx: Option<usize> = None;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..pool.len() {
            if used[i] || !pred(i) {
                continue;
            }
            let is_theta_dominant = pool[i].scores.s_theta >= theta_thresh;
            let is_gamma_dominant =
                pool[i].scores.s_gamma >= gamma_thresh && pool[i].scores.s_theta < theta_thresh;
            if is_theta_dominant && theta_count >= max_theta {
                continue;
            }
            if is_gamma_dominant && gamma_count >= max_gamma {
                continue;
            }
            let fatigued = pool[i]
                .top_action_labels
                .iter()
                .any(|lbl| *action_label_count.get(lbl).unwrap_or(&0) >= max_action_fatigue);
            if fatigued {
                continue;
            }
            let mut score = pool[i].scores.total_score();
            if !turn_set.contains(&pool[i].decision.turn) {
                score += 3.0;
            }
            if is_theta_dominant && theta_count >= max_theta.saturating_sub(2) {
                score -= 2.0;
            }
            if is_gamma_dominant && gamma_count >= max_gamma.saturating_sub(2) {
                score -= 2.0;
            }
            score += 0.001 * pool[i].visit_count as f32;
            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }
        best_idx
    };

    let select_one = |idx: usize,
                      selected: &mut Vec<usize>,
                      used: &mut Vec<bool>,
                      phase_count: &mut [usize; 3],
                      dtype_count: &mut [usize; 3],
                      theta_count: &mut usize,
                      gamma_count: &mut usize,
                      depth_count: &mut usize,
                      beta_count: &mut usize,
                      turn_set: &mut Vec<usize>,
                      action_label_count: &mut HashMap<String, usize>| {
        let sc = &pool[idx];
        selected.push(idx);
        used[idx] = true;

        let fp = &sc.fingerprint;
        for j in 0..pool.len() {
            if !used[j] && pool[j].fingerprint == *fp {
                used[j] = true;
            }
        }

        phase_count[phase_idx(sc.bucket.phase)] += 1;
        dtype_count[dtype_idx(sc.bucket.dtype)] += 1;
        if sc.scores.s_theta >= theta_thresh {
            *theta_count += 1;
        }
        if sc.scores.s_gamma >= gamma_thresh {
            *gamma_count += 1;
        }
        if sc.scores.s_d > 0.0 {
            *depth_count += 1;
        }
        if sc.scores.s_beta >= beta_thresh {
            *beta_count += 1;
        }
        if !turn_set.contains(&sc.decision.turn) {
            turn_set.push(sc.decision.turn);
        }
        for lbl in &sc.top_action_labels {
            *action_label_count.entry(lbl.clone()).or_insert(0) += 1;
        }
    };

    // Phase 1: Fill must-fill quotas
    loop {
        if selected.len() >= n_scenarios {
            break;
        }

        let mut worst_deficit = 0i32;
        let mut worst_type = 0u8;
        let mut worst_param = 0usize;

        for p in 0..3 {
            let deficit = min_per_phase as i32 - phase_count[p] as i32;
            if deficit > worst_deficit {
                worst_deficit = deficit;
                worst_type = 1;
                worst_param = p;
            }
        }
        for d in 0..3 {
            let deficit = min_per_dtype as i32 - dtype_count[d] as i32;
            if deficit > worst_deficit {
                worst_deficit = deficit;
                worst_type = 2;
                worst_param = d;
            }
        }
        {
            let deficit = min_theta as i32 - theta_count as i32;
            if deficit > worst_deficit {
                worst_deficit = deficit;
                worst_type = 3;
            }
        }
        {
            let deficit = min_gamma as i32 - gamma_count as i32;
            if deficit > worst_deficit {
                worst_deficit = deficit;
                worst_type = 4;
            }
        }
        {
            let deficit = min_depth as i32 - depth_count as i32;
            if deficit > worst_deficit {
                worst_deficit = deficit;
                worst_type = 5;
            }
        }
        {
            let deficit = min_beta as i32 - beta_count as i32;
            if deficit > worst_deficit {
                worst_deficit = deficit;
                worst_type = 6;
            }
        }

        if worst_deficit <= 0 {
            break;
        }

        let idx = match worst_type {
            1 => {
                let target_phase = match worst_param {
                    0 => GamePhase::Early,
                    1 => GamePhase::Mid,
                    _ => GamePhase::Late,
                };
                find_best(
                    &used,
                    &turn_set,
                    theta_count,
                    gamma_count,
                    &action_label_count,
                    &|i| pool[i].bucket.phase == target_phase,
                )
            }
            2 => {
                let target_dtype = match worst_param {
                    0 => DecisionType::Reroll1,
                    1 => DecisionType::Reroll2,
                    _ => DecisionType::Category,
                };
                find_best(
                    &used,
                    &turn_set,
                    theta_count,
                    gamma_count,
                    &action_label_count,
                    &|i| pool[i].bucket.dtype == target_dtype,
                )
            }
            3 => find_best(
                &used,
                &turn_set,
                theta_count,
                gamma_count,
                &action_label_count,
                &|i| pool[i].scores.s_theta >= theta_thresh,
            ),
            4 => find_best(
                &used,
                &turn_set,
                theta_count,
                gamma_count,
                &action_label_count,
                &|i| pool[i].scores.s_gamma >= gamma_thresh,
            ),
            5 => find_best(
                &used,
                &turn_set,
                theta_count,
                gamma_count,
                &action_label_count,
                &|i| pool[i].scores.s_d > 0.0,
            ),
            6 => find_best(
                &used,
                &turn_set,
                theta_count,
                gamma_count,
                &action_label_count,
                &|i| pool[i].scores.s_beta >= beta_thresh,
            ),
            _ => None,
        };

        match idx {
            Some(i) => select_one(
                i,
                &mut selected,
                &mut used,
                &mut phase_count,
                &mut dtype_count,
                &mut theta_count,
                &mut gamma_count,
                &mut depth_count,
                &mut beta_count,
                &mut turn_set,
                &mut action_label_count,
            ),
            None => break,
        }
    }

    // Phase 2: Fill remaining slots
    while selected.len() < n_scenarios {
        let mut best_idx: Option<usize> = None;
        let mut best_score = f32::NEG_INFINITY;

        for i in 0..pool.len() {
            if used[i] {
                continue;
            }
            let sc = &pool[i];
            let is_theta_dominant = sc.scores.s_theta >= theta_thresh;
            let is_gamma_dominant =
                sc.scores.s_gamma >= gamma_thresh && sc.scores.s_theta < theta_thresh;
            if is_theta_dominant && theta_count >= max_theta {
                continue;
            }
            if is_gamma_dominant && gamma_count >= max_gamma {
                continue;
            }
            let fatigued = sc
                .top_action_labels
                .iter()
                .any(|lbl| *action_label_count.get(lbl).unwrap_or(&0) >= max_action_fatigue);
            if fatigued {
                continue;
            }

            let mut score = sc.scores.total_score();
            if !turn_set.contains(&sc.decision.turn) {
                score += 5.0;
            }
            let pi = phase_idx(sc.bucket.phase);
            let di = dtype_idx(sc.bucket.dtype);
            if phase_count[pi] < min_per_phase {
                score += 3.0;
            }
            if dtype_count[di] < min_per_dtype {
                score += 3.0;
            }
            if is_theta_dominant && theta_count >= max_theta.saturating_sub(2) {
                score -= 2.0;
            }
            if is_gamma_dominant && gamma_count >= max_gamma.saturating_sub(2) {
                score -= 2.0;
            }
            if sc.scores.s_gamma >= gamma_thresh && gamma_count < min_gamma {
                score += 3.0;
            }
            if sc.scores.s_d > 0.0 && depth_count < min_depth {
                score += 3.0;
            }
            if sc.scores.s_beta >= beta_thresh && beta_count < min_beta {
                score += 3.0;
            }
            score += 0.001 * sc.visit_count as f32;

            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }

        match best_idx {
            Some(i) => select_one(
                i,
                &mut selected,
                &mut used,
                &mut phase_count,
                &mut dtype_count,
                &mut theta_count,
                &mut gamma_count,
                &mut depth_count,
                &mut beta_count,
                &mut turn_set,
                &mut action_label_count,
            ),
            None => break,
        }
    }

    selected.sort_by_key(|&i| pool[i].decision.turn);
    selected
}

/// Build ProfilingScenarios from selected pool indices.
pub fn build_profiling_scenarios(
    ctx: &YatzyContext,
    pool: &[ScoredCandidate],
    selected: &[usize],
) -> Vec<ProfilingScenario> {
    let sv = ctx.state_values.as_slice();
    let mut result: Vec<ProfilingScenario> = Vec::with_capacity(selected.len());

    for (id, &pool_idx) in selected.iter().enumerate() {
        let sc = &pool[pool_idx];
        let d = &sc.decision;
        let actions = get_actions(ctx, sv, d, true);
        let optimal_id = actions.first().map(|a| a.id).unwrap_or(0);
        let desc = make_description(d, &actions, sc.ev_gap);

        result.push(ProfilingScenario {
            id,
            upper_score: d.upper_score,
            scored_categories: d.scored,
            dice: d.dice,
            turn: d.turn,
            decision_type: d.decision_type,
            quadrant: primary_quadrant(sc),
            visit_count: sc.visit_count,
            ev_gap: sc.ev_gap,
            optimal_action_id: optimal_id,
            actions,
            description: desc,
        });
    }

    result
}

fn make_description(d: &RawDecision, actions: &[ActionInfo], gap: f32) -> String {
    let remaining = CATEGORY_COUNT - (d.scored.count_ones() as usize);
    let best = actions.first().map(|a| a.label.as_str()).unwrap_or("?");
    let runner = actions.get(1).map(|a| a.label.as_str()).unwrap_or("?");
    format!(
        "Turn {}, {}: {:?}, {} open. {} vs {}, gap {:.2}",
        d.turn + 1,
        d.decision_type.phase_label(),
        d.dice,
        remaining,
        best,
        runner,
        gap,
    )
}

use crate::widget_solver::compute_max_ev_for_n_rerolls;
