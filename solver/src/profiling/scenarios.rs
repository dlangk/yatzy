//! Scenario generation for cognitive profiling.
//!
//! Generates candidate scenarios from noisy-simulated games, classifies them
//! into a 3D semantic grid (phase × dtype × tension), scores diagnostic value,
//! and assembles a 30-scenario quiz with diversity constraints.

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::constants::*;
use crate::dice_mechanics::{count_faces, find_dice_set_index, sort_dice_set};
use crate::game_mechanics::update_upper_score;
use crate::profiling::qvalues::{
    compute_eds_for_scenario, compute_group6_profiling, compute_group6_risk_profiling,
    compute_q_categories, compute_q_rerolls, sigma_for_depth, softmax_sample,
};
use crate::simulation::heuristic::{heuristic_pick_category, heuristic_reroll_mask};
use crate::types::{StateValues, YatzyContext};
use crate::widget_solver::{
    choose_best_reroll_mask, compute_max_ev_for_n_rerolls, compute_opt_lse_for_n_rerolls,
};

// ── Types ──

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DecisionType {
    Reroll1,
    Reroll2,
    Category,
}

impl DecisionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DecisionType::Reroll1 => "reroll1",
            DecisionType::Reroll2 => "reroll2",
            DecisionType::Category => "category",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GamePhase {
    Early,
    Mid,
    Late,
}

impl GamePhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            GamePhase::Early => "early",
            GamePhase::Mid => "mid",
            GamePhase::Late => "late",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BoardTension {
    BonusChase,
    PatternHunt,
    Cleanup,
    Open,
}

impl BoardTension {
    pub fn as_str(&self) -> &'static str {
        match self {
            BoardTension::BonusChase => "bonus_chase",
            BoardTension::PatternHunt => "pattern_hunt",
            BoardTension::Cleanup => "cleanup",
            BoardTension::Open => "open",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SemanticBucket {
    pub phase: GamePhase,
    pub dtype: DecisionType,
    pub tension: BoardTension,
}

#[derive(Clone, Debug)]
pub struct DiagnosticScores {
    pub s_theta: f32,
    pub s_gamma: f32,
    pub s_d: f32,
    pub s_beta: f32,
}

impl DiagnosticScores {
    pub fn max_score(&self) -> f32 {
        self.s_theta
            .max(self.s_gamma)
            .max(self.s_d)
            .max(self.s_beta)
    }

    pub fn total_score(&self) -> f32 {
        self.s_theta + self.s_gamma + self.s_d + self.s_beta
    }
}

/// Which diagnostic quadrant a scenario belongs to (primary).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quadrant {
    Theta,
    Gamma,
    Depth,
    Beta,
}

impl Quadrant {
    pub fn as_str(&self) -> &'static str {
        match self {
            Quadrant::Theta => "theta",
            Quadrant::Gamma => "gamma",
            Quadrant::Depth => "depth",
            Quadrant::Beta => "beta",
        }
    }
}

#[derive(Clone)]
pub struct RawDecision {
    pub upper_score: i32,
    pub scored: i32,
    pub dice: [i32; 5],
    pub turn: usize,
    pub decision_type: DecisionType,
}

pub type DecisionKey = (i32, i32, [i32; 5], DecisionType);

pub fn decision_key(d: &RawDecision) -> DecisionKey {
    (d.upper_score, d.scored, d.dice, d.decision_type)
}

/// A scored candidate with diagnostic metadata.
pub struct ScoredCandidate {
    pub decision: RawDecision,
    pub visit_count: usize,
    pub bucket: SemanticBucket,
    pub scores: DiagnosticScores,
    pub ev_gap: f32,
    /// Functional fingerprint: identifies scenarios that are decision-equivalent.
    /// Two scenarios with the same fingerprint present identical choices to the player.
    pub fingerprint: String,
    /// Top-2 action labels for action-fatigue tracking.
    pub top_action_labels: Vec<String>,
}

/// A selected profiling scenario with all metadata.
pub struct ProfilingScenario {
    pub id: usize,
    pub upper_score: i32,
    pub scored_categories: i32,
    pub dice: [i32; 5],
    pub turn: usize,
    pub decision_type: DecisionType,
    pub quadrant: Quadrant,
    pub visit_count: usize,
    pub ev_gap: f32,
    pub optimal_action_id: i32,
    pub actions: Vec<ActionInfo>,
    pub description: String,
}

pub struct ActionInfo {
    pub id: i32,
    pub label: String,
    pub ev_theta0: f32,
}

/// Compute a functional fingerprint for a decision scenario.
///
/// Two scenarios with the same fingerprint present effectively the same
/// dilemma to the player. The fingerprint captures the decision-relevant
/// features: the top action EVs (rounded), board state, and decision type.
///
/// For category decisions: different dice can yield different raw scores
/// on irrelevant categories while the actual choice (top 2-3 actions) is
/// identical. So we fingerprint on rounded top-3 EVs rather than all scores.
///
/// For reroll decisions: the dice directly determine outcomes, so we include them.
fn compute_fingerprint(
    _ctx: &YatzyContext,
    _sv: &[f32],
    d: &RawDecision,
    actions: &[ActionInfo],
) -> String {
    // Top-3 action EVs rounded to 0.5 — captures the decision structure
    let top_evs: Vec<i32> = actions
        .iter()
        .take(3)
        .map(|a| (a.ev_theta0 * 2.0).round() as i32) // round to nearest 0.5
        .collect();
    let top_ids: Vec<i32> = actions.iter().take(3).map(|a| a.id).collect();

    match d.decision_type {
        DecisionType::Category => {
            // Board state + top action IDs + rounded EVs
            format!(
                "cat:s{},u{},ids{:?},evs{:?}",
                d.scored, d.upper_score, top_ids, top_evs
            )
        }
        _ => {
            // For rerolls, include dice (different dice → different reroll outcomes)
            format!(
                "{:?}:s{},u{},{:?},ids{:?},evs{:?}",
                d.decision_type, d.scored, d.upper_score, d.dice, top_ids, top_evs
            )
        }
    }
}

// ── Simulation helpers ──

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

// ── Noisy simulation (competent human agent) ──

/// Simulate one game using a noisy "competent human" agent.
///
/// Uses softmax(β·Q) for action selection instead of argmax, with γ-discounted
/// future values and depth-noise perturbation. This produces realistic mid-game
/// board states that humans would encounter.
fn simulate_game_collecting_noisy(
    ctx: &YatzyContext,
    rng: &mut SmallRng,
    beta_sim: f32,
    gamma_sim: f32,
    sigma_d: f32,
) -> Vec<RawDecision> {
    let sv = ctx.state_values.as_slice();
    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
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
        });

        // Compute Q-values for reroll1 with profiling parameters
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
        });

        let ds_index = find_dice_set_index(ctx, &dice);
        let (cat, scr) = if is_last_turn {
            find_best_category_final(ctx, up_score, scored, ds_index)
        } else {
            // Use softmax for category selection with γ-discounting
            let cat_qs =
                compute_q_categories(ctx, sv, up_score, scored, ds_index, gamma_sim, sigma_d, false);
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

        up_score = update_upper_score(up_score, cat, scr);
        scored |= 1 << cat;
    }

    decisions
}

// ── Classification ──

/// Classify turn number into game phase.
pub fn classify_phase(turn: usize) -> GamePhase {
    match turn {
        0..=4 => GamePhase::Early,
        5..=9 => GamePhase::Mid,
        _ => GamePhase::Late,
    }
}

/// Classify board state into tension type.
pub fn classify_tension(scored: i32, upper_score: i32, dice: &[i32; 5]) -> BoardTension {
    let num_scored = scored.count_ones() as usize;
    let num_open = CATEGORY_COUNT - num_scored;

    // Cleanup: few categories left (≤3)
    if num_open <= 3 {
        return BoardTension::Cleanup;
    }

    // BonusChase: upper score near threshold with upper cats still open
    let upper_open = (0..6).filter(|&c| !is_category_scored(scored, c)).count();
    if upper_open > 0 && upper_score >= 30 && upper_score < 63 {
        return BoardTension::BonusChase;
    }

    // PatternHunt: straights/full house/yatzy open with partial dice match
    let face_count = count_faces(dice);
    let has_pattern_potential = {
        let straight_open = !is_category_scored(scored, CATEGORY_SMALL_STRAIGHT)
            || !is_category_scored(scored, CATEGORY_LARGE_STRAIGHT);
        let full_house_open = !is_category_scored(scored, CATEGORY_FULL_HOUSE);
        let yatzy_open = !is_category_scored(scored, CATEGORY_YATZY);

        let has_run = {
            // Check for 3+ consecutive values
            let mut max_run = 0;
            let mut current_run = 0;
            for f in 1..=6 {
                if face_count[f] >= 1 {
                    current_run += 1;
                    max_run = max_run.max(current_run);
                } else {
                    current_run = 0;
                }
            }
            max_run >= 3
        };
        let has_trips = face_count.iter().any(|&c| c >= 3);
        let has_pair_trips = {
            let pairs = face_count.iter().filter(|&&c| c >= 2).count();
            pairs >= 2
        };

        (straight_open && has_run)
            || (full_house_open && has_pair_trips)
            || (yatzy_open && has_trips)
    };

    if has_pattern_potential {
        return BoardTension::PatternHunt;
    }

    BoardTension::Open
}

/// Classify a candidate into its semantic bucket.
fn classify_bucket(d: &RawDecision) -> SemanticBucket {
    SemanticBucket {
        phase: classify_phase(d.turn),
        dtype: d.decision_type,
        tension: classify_tension(d.scored, d.upper_score, &d.dice),
    }
}

// ── Format helpers ──

/// Format a reroll mask as human-readable string.
fn format_mask(mask: i32, dice: &[i32; 5]) -> String {
    if mask == 0 {
        return "Keep all".to_string();
    }
    if mask == 31 {
        return "Reroll all".to_string();
    }
    let mut kept: Vec<i32> = Vec::new();
    for i in 0..5 {
        if mask & (1 << i) == 0 {
            kept.push(dice[i]);
        }
    }
    if kept.is_empty() {
        "Reroll all".to_string()
    } else {
        format!("Keep {:?}", kept)
    }
}

/// Enumerate all reroll actions with EVs.
fn enumerate_reroll_actions(
    ctx: &YatzyContext,
    e_ds: &[f32; 252],
    dice: &[i32; 5],
) -> Vec<ActionInfo> {
    let ds_index = find_dice_set_index(ctx, dice);
    let kt = &ctx.keep_table;
    let mut actions: Vec<ActionInfo> = Vec::with_capacity(32);

    actions.push(ActionInfo {
        id: 0,
        label: "Keep all".to_string(),
        ev_theta0: e_ds[ds_index],
    });

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
        actions.push(ActionInfo {
            id: mask,
            label: format_mask(mask, dice),
            ev_theta0: ev,
        });
    }

    actions.sort_by(|a, b| b.ev_theta0.partial_cmp(&a.ev_theta0).unwrap());
    actions
}

/// Enumerate all category actions with EVs.
fn enumerate_category_actions(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    ds_index: usize,
    is_last_turn: bool,
) -> Vec<ActionInfo> {
    let mut actions: Vec<ActionInfo> = Vec::with_capacity(CATEGORY_COUNT);

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
                actions.push(ActionInfo {
                    id: c as i32,
                    label: CATEGORY_NAMES[c].to_string(),
                    ev_theta0: (scr + bonus) as f32,
                });
            }
        }
    } else {
        for c in 0..6 {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let new_up = update_upper_score(up_score, c, scr);
                let new_scored = scored | (1 << c);
                let val = scr as f32
                    + unsafe {
                        *sv.get_unchecked(state_index(new_up as usize, new_scored as usize))
                    };
                actions.push(ActionInfo {
                    id: c as i32,
                    label: CATEGORY_NAMES[c].to_string(),
                    ev_theta0: val,
                });
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
                actions.push(ActionInfo {
                    id: c as i32,
                    label: CATEGORY_NAMES[c].to_string(),
                    ev_theta0: val,
                });
            }
        }
    }

    actions.sort_by(|a, b| b.ev_theta0.partial_cmp(&a.ev_theta0).unwrap());
    actions
}

// ── Diagnostic scoring ──

/// Compute S_θ: how much the optimal action changes across θ values.
///
/// For reroll decisions, compute Q-values at θ=−0.05 and θ=+0.05.
/// Score = rank-change magnitude of top actions.
fn compute_s_theta(
    ctx: &YatzyContext,
    d: &RawDecision,
    theta_tables: &[(f32, &[f32])],
) -> f32 {
    if !matches!(d.decision_type, DecisionType::Reroll1 | DecisionType::Reroll2) {
        return 0.0;
    }

    let mut rankings: Vec<(f32, Vec<i32>)> = Vec::new();

    for &(theta, sv) in theta_tables {
        let is_risk = theta != 0.0;
        let mut e_ds_0 = [0.0f32; 252];
        let mut e_ds_1 = [0.0f32; 252];

        if is_risk {
            compute_group6_risk_profiling(
                ctx, sv, d.upper_score, d.scored, theta, 1.0, 0.0, &mut e_ds_0,
            );
            compute_opt_lse_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1, false);
        } else {
            compute_group6(ctx, sv, d.upper_score, d.scored, &mut e_ds_0);
            compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);
        }

        let e_ds = match d.decision_type {
            DecisionType::Reroll1 => &e_ds_1,
            DecisionType::Reroll2 => &e_ds_0,
            _ => unreachable!(),
        };

        let mut rerolls = compute_q_rerolls(ctx, e_ds, &d.dice, is_risk);
        rerolls.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let ranked: Vec<i32> = rerolls.iter().map(|(m, _)| *m).collect();
        rankings.push((theta, ranked));
    }

    // Compare: find negative and positive θ rankings
    let neg = rankings.iter().find(|(t, _)| *t < -0.01);
    let pos = rankings.iter().find(|(t, _)| *t > 0.01);

    match (neg, pos) {
        (Some((_, rank_neg)), Some((_, rank_pos))) => {
            if rank_neg.is_empty() || rank_pos.is_empty() {
                return 0.0;
            }
            // Score = 1.0 if top action flips, 0.5 if top-2 swap, 0.0 if same
            if rank_neg[0] != rank_pos[0] {
                1.0
            } else if rank_neg.len() >= 2
                && rank_pos.len() >= 2
                && rank_neg[1] != rank_pos[1]
            {
                0.5
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

/// Compute S_γ: how much greedy differs from optimal.
///
/// For category decisions, compare greedy ranking vs DP-optimal ranking.
fn compute_s_gamma(
    ctx: &YatzyContext,
    sv: &[f32],
    d: &RawDecision,
) -> f32 {
    if d.decision_type != DecisionType::Category {
        // Reroll decisions get reduced S_γ based on how many categories
        // are affected by γ at this state. Simple proxy: 0.2 if mid-game.
        return match classify_phase(d.turn) {
            GamePhase::Mid => 0.2,
            _ => 0.0,
        };
    }
    if d.turn == CATEGORY_COUNT - 1 {
        return 0.0; // Last turn has no future
    }

    let ds_index = find_dice_set_index(ctx, &d.dice);

    // Greedy: pick category with highest immediate score
    let mut greedy_ranking: Vec<(usize, i32)> = Vec::new();
    for c in 0..CATEGORY_COUNT {
        if !is_category_scored(d.scored, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            greedy_ranking.push((c, scr));
        }
    }
    greedy_ranking.sort_by(|a, b| b.1.cmp(&a.1));

    // Optimal: use full state values
    let (optimal_cat, _) = find_best_category(ctx, sv, d.upper_score, d.scored, ds_index);

    // Score based on where optimal cat ranks in greedy ordering
    if let Some(greedy_top) = greedy_ranking.first() {
        if greedy_top.0 == optimal_cat {
            0.0
        } else {
            // Position of optimal in greedy ranking
            let pos = greedy_ranking
                .iter()
                .position(|(c, _)| *c == optimal_cat)
                .unwrap_or(greedy_ranking.len());
            // Normalize: pos 1 → 0.5, pos 2+ → 1.0
            if pos <= 1 {
                0.5
            } else {
                1.0
            }
        }
    } else {
        0.0
    }
}

/// Compute S_d: does heuristic disagree with optimal? (0 or 1)
fn compute_s_d(
    ctx: &YatzyContext,
    sv: &[f32],
    d: &RawDecision,
) -> f32 {
    let ds_index = find_dice_set_index(ctx, &d.dice);
    let face_count = count_faces(&d.dice);

    match d.decision_type {
        DecisionType::Reroll1 | DecisionType::Reroll2 => {
            let heuristic_mask =
                heuristic_reroll_mask(&d.dice, &face_count, d.scored, d.upper_score);

            let mut e_ds_0 = [0.0f32; 252];
            let mut e_ds_1 = [0.0f32; 252];
            compute_group6(ctx, sv, d.upper_score, d.scored, &mut e_ds_0);

            let mut best_ev = 0.0;
            let optimal_mask = match d.decision_type {
                DecisionType::Reroll1 => {
                    compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);
                    choose_best_reroll_mask(ctx, &e_ds_1, &d.dice, &mut best_ev)
                }
                DecisionType::Reroll2 => {
                    choose_best_reroll_mask(ctx, &e_ds_0, &d.dice, &mut best_ev)
                }
                _ => unreachable!(),
            };

            if heuristic_mask != optimal_mask {
                1.0
            } else {
                0.0
            }
        }
        DecisionType::Category => {
            if d.turn == CATEGORY_COUNT - 1 {
                return 0.0;
            }
            let heuristic_cat =
                heuristic_pick_category(&d.dice, &face_count, d.scored, d.upper_score);
            let (optimal_cat, _) =
                find_best_category(ctx, sv, d.upper_score, d.scored, ds_index);
            if heuristic_cat != optimal_cat {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Compute all diagnostic scores for a candidate.
fn compute_diagnostic_scores(
    ctx: &YatzyContext,
    sv: &[f32],
    d: &RawDecision,
    ev_gap: f32,
    theta_tables: &[(f32, &[f32])],
) -> DiagnosticScores {
    DiagnosticScores {
        s_theta: compute_s_theta(ctx, d, theta_tables),
        s_gamma: compute_s_gamma(ctx, sv, d),
        s_d: compute_s_d(ctx, sv, d),
        // S_β: normalize gap to [0, 1] range. Gaps > 20 all map to 1.0.
        s_beta: (ev_gap / 20.0).min(1.0),
    }
}

// ── Candidate collection ──

/// Collect candidate scenarios from noisy-simulated games.
pub fn collect_candidates_noisy(
    ctx: &YatzyContext,
    num_games: usize,
    seed: u64,
    beta_sim: f32,
    gamma_sim: f32,
    sigma_d: f32,
) -> HashMap<DecisionKey, (RawDecision, usize)> {
    let batch_size = 100_000usize;
    let num_batches = num_games.div_ceil(batch_size);

    let mut visit_counts: HashMap<DecisionKey, (RawDecision, usize)> = HashMap::new();

    for batch in 0..num_batches {
        let batch_start = batch * batch_size;
        let batch_end = (batch_start + batch_size).min(num_games);

        let batch_decisions: Vec<RawDecision> = (batch_start..batch_end)
            .into_par_iter()
            .flat_map_iter(|i| {
                let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
                simulate_game_collecting_noisy(ctx, &mut rng, beta_sim, gamma_sim, sigma_d)
            })
            .collect();

        for d in &batch_decisions {
            let key = decision_key(d);
            visit_counts
                .entry(key)
                .and_modify(|e| e.1 += 1)
                .or_insert_with(|| (d.clone(), 1));
        }

        if (batch + 1) % 5 == 0 || batch == num_batches - 1 {
            println!(
                "  Batch {}/{}: {} unique scenarios",
                batch + 1,
                num_batches,
                visit_counts.len(),
            );
        }
    }

    visit_counts
}

/// Collect candidate scenarios from optimal-play simulated games (original method).
pub fn collect_candidates(
    ctx: &YatzyContext,
    num_games: usize,
    seed: u64,
) -> HashMap<DecisionKey, (RawDecision, usize)> {
    let batch_size = 100_000usize;
    let num_batches = num_games.div_ceil(batch_size);

    let mut visit_counts: HashMap<DecisionKey, (RawDecision, usize)> = HashMap::new();

    for batch in 0..num_batches {
        let batch_start = batch * batch_size;
        let batch_end = (batch_start + batch_size).min(num_games);

        let batch_decisions: Vec<RawDecision> = (batch_start..batch_end)
            .into_par_iter()
            .flat_map_iter(|i| {
                let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
                simulate_game_collecting_optimal(ctx, &mut rng)
            })
            .collect();

        for d in &batch_decisions {
            let key = decision_key(d);
            visit_counts
                .entry(key)
                .and_modify(|e| e.1 += 1)
                .or_insert_with(|| (d.clone(), 1));
        }

        if (batch + 1) % 5 == 0 || batch == num_batches - 1 {
            println!(
                "  Batch {}/{}: {} unique scenarios",
                batch + 1,
                num_batches,
                visit_counts.len(),
            );
        }
    }

    visit_counts
}

/// Simulate one game with optimal play, collecting all decision points.
fn simulate_game_collecting_optimal(ctx: &YatzyContext, rng: &mut SmallRng) -> Vec<RawDecision> {
    let sv = ctx.state_values.as_slice();
    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut decisions = Vec::with_capacity(45);

    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;
        let mut dice = roll_dice(rng);

        compute_group6(ctx, sv, up_score, scored, &mut e_ds_0);
        compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);

        decisions.push(RawDecision {
            upper_score: up_score,
            scored,
            dice,
            turn,
            decision_type: DecisionType::Reroll1,
        });

        let mut best_ev = 0.0;
        let mask1 = choose_best_reroll_mask(ctx, &e_ds_1, &dice, &mut best_ev);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }

        decisions.push(RawDecision {
            upper_score: up_score,
            scored,
            dice,
            turn,
            decision_type: DecisionType::Reroll2,
        });

        let mask2 = choose_best_reroll_mask(ctx, &e_ds_0, &dice, &mut best_ev);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }

        decisions.push(RawDecision {
            upper_score: up_score,
            scored,
            dice,
            turn,
            decision_type: DecisionType::Category,
        });

        let ds_index = find_dice_set_index(ctx, &dice);
        let (cat, scr) = if is_last_turn {
            find_best_category_final(ctx, up_score, scored, ds_index)
        } else {
            find_best_category(ctx, sv, up_score, scored, ds_index)
        };

        up_score = update_upper_score(up_score, cat, scr);
        scored |= 1 << cat;
    }

    decisions
}

// ── Master pool + quiz assembly ──

/// Build the master pool from candidates: score diagnostics, bucket, keep top N per bucket.
pub fn build_master_pool(
    ctx: &YatzyContext,
    candidates: HashMap<DecisionKey, (RawDecision, usize)>,
    theta_tables: &[(f32, StateValues)],
    max_per_bucket: usize,
) -> Vec<ScoredCandidate> {
    let sv = ctx.state_values.as_slice();

    // Phase-dependent visit threshold: mid-game states are so diverse under
    // noisy simulation that requiring high visit counts eliminates them all.
    let min_visits_early_late = 5usize;
    let min_visits_mid = 2usize;

    // Filter to viable candidates
    let viable: Vec<(RawDecision, usize)> = candidates
        .into_values()
        .filter(|(d, count)| {
            let min_visits = match classify_phase(d.turn) {
                GamePhase::Mid => min_visits_mid,
                _ => min_visits_early_late,
            };
            *count >= min_visits
                && d.turn > 0
                && d.turn < CATEGORY_COUNT - 1
        })
        .collect();

    println!(
        "  {} viable candidates (visits >= {}/{})",
        viable.len(), min_visits_early_late, min_visits_mid,
    );

    // Prepare theta table slices
    let theta_slices: Vec<(f32, &[f32])> = theta_tables
        .iter()
        .map(|(t, sv)| (*t, sv.as_slice()))
        .collect();

    // Score all candidates
    let mut scored_candidates: Vec<ScoredCandidate> = Vec::new();
    let mut skipped_few_actions = 0usize;

    for (d, count) in &viable {
        let ds_index = find_dice_set_index(ctx, &d.dice);
        let is_last = d.turn == CATEGORY_COUNT - 1;

        // Compute EV gap
        let actions = match d.decision_type {
            DecisionType::Reroll1 => {
                let mut e_ds_0 = [0.0f32; 252];
                let mut e_ds_1 = [0.0f32; 252];
                compute_group6(ctx, sv, d.upper_score, d.scored, &mut e_ds_0);
                compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);
                enumerate_reroll_actions(ctx, &e_ds_1, &d.dice)
            }
            DecisionType::Reroll2 => {
                let mut e_ds_0 = [0.0f32; 252];
                compute_group6(ctx, sv, d.upper_score, d.scored, &mut e_ds_0);
                enumerate_reroll_actions(ctx, &e_ds_0, &d.dice)
            }
            DecisionType::Category => {
                enumerate_category_actions(ctx, sv, d.upper_score, d.scored, ds_index, is_last)
            }
        };

        if actions.len() < 2 {
            skipped_few_actions += 1;
            continue;
        }

        let gap = actions[0].ev_theta0 - actions[1].ev_theta0;
        if gap < 0.1 {
            continue; // Skip near-ties that are too close
        }

        let bucket = classify_bucket(d);
        let diag_scores = compute_diagnostic_scores(ctx, sv, d, gap, &theta_slices);

        // Only keep candidates with at least some diagnostic value
        if diag_scores.max_score() < 0.01 && gap < 0.5 {
            continue;
        }

        let fingerprint = compute_fingerprint(ctx, sv, d, &actions);
        let top_action_labels: Vec<String> = actions.iter().take(2).map(|a| a.label.clone()).collect();

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
        println!("  Skipped {} candidates with < 2 actions", skipped_few_actions);
    }
    println!("  {} candidates with diagnostic scores", scored_candidates.len());

    // Group by bucket, keep top N per bucket
    let mut buckets: HashMap<SemanticBucket, Vec<usize>> = HashMap::new();
    for (i, sc) in scored_candidates.iter().enumerate() {
        buckets.entry(sc.bucket).or_default().push(i);
    }

    let mut pool_indices: Vec<usize> = Vec::new();
    for (bucket, mut indices) in buckets {
        // Sort by max diagnostic score descending, then visit count
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
        if indices.len() > 0 {
            println!(
                "    {:?}: {} candidates, keeping {}",
                bucket, indices.len(), take
            );
        }
    }

    // Extract from scored_candidates
    pool_indices.sort();
    pool_indices.dedup();

    // Build final pool
    let mut pool: Vec<ScoredCandidate> = Vec::with_capacity(pool_indices.len());
    for i in pool_indices.into_iter().rev() {
        pool.push(scored_candidates.swap_remove(i));
    }

    println!("  Master pool: {} candidates", pool.len());
    pool
}

/// Assemble quiz of N scenarios from the master pool with diversity constraints.
pub fn assemble_quiz(
    pool: &[ScoredCandidate],
    n_scenarios: usize,
) -> Vec<usize> {
    if pool.len() <= n_scenarios {
        return (0..pool.len()).collect();
    }

    let mut selected: Vec<usize> = Vec::with_capacity(n_scenarios);
    let mut used: Vec<bool> = vec![false; pool.len()];

    // Track coverage
    let mut phase_count = [0usize; 3]; // Early, Mid, Late
    let mut dtype_count = [0usize; 3]; // Reroll1, Reroll2, Category
    let mut theta_count = 0usize;
    let mut gamma_count = 0usize;
    let mut depth_count = 0usize;
    let mut beta_count = 0usize;
    let mut turn_set: Vec<usize> = Vec::new(); // track which turns are covered

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

    // Minimum quotas (scaled for 30 scenarios)
    let min_per_phase = 4usize;
    let min_per_dtype = 6usize;
    let min_theta = 5usize;
    let min_gamma = 5usize;
    let min_depth = 4usize;
    let min_beta = 4usize;
    let max_theta = 12usize;
    let max_gamma = 12usize;

    // Action-fatigue: no action label in top-2 more than max_action_fatigue times
    let max_action_fatigue = 4usize;
    let mut action_label_count: HashMap<String, usize> = HashMap::new();

    // Threshold for "diagnostic" classification
    let theta_thresh = 0.5;
    let gamma_thresh = 0.5;
    let beta_thresh = 0.15; // ev_gap > 3 pts — lowered to get more β candidates

    // Helper: find best candidate satisfying a predicate, with turn-diversity bonus,
    // quadrant cap penalties, and action-fatigue check
    let find_best =
        |used: &[bool], turn_set: &[usize], theta_count: usize, gamma_count: usize,
         action_label_count: &HashMap<String, usize>,
         pred: &dyn Fn(usize) -> bool| -> Option<usize> {
            let mut best_idx: Option<usize> = None;
            let mut best_score = f32::NEG_INFINITY;
            for i in 0..pool.len() {
                if used[i] || !pred(i) {
                    continue;
                }
                let is_theta_dominant = pool[i].scores.s_theta >= theta_thresh;
                let is_gamma_dominant = pool[i].scores.s_gamma >= gamma_thresh
                    && pool[i].scores.s_theta < theta_thresh;
                // Hard skip when over cap
                if is_theta_dominant && theta_count >= max_theta {
                    continue;
                }
                if is_gamma_dominant && gamma_count >= max_gamma {
                    continue;
                }
                // Action-fatigue: skip if any top-2 action label is over the limit
                let fatigued = pool[i].top_action_labels.iter().any(|lbl| {
                    *action_label_count.get(lbl).unwrap_or(&0) >= max_action_fatigue
                });
                if fatigued {
                    continue;
                }
                let mut score = pool[i].scores.total_score();
                // Bonus for covering a new turn number
                if !turn_set.contains(&pool[i].decision.turn) {
                    score += 3.0;
                }
                // Penalty when approaching caps
                if is_theta_dominant && theta_count >= max_theta.saturating_sub(2) {
                    score -= 2.0;
                }
                if is_gamma_dominant && gamma_count >= max_gamma.saturating_sub(2) {
                    score -= 2.0;
                }
                // Small tiebreak by visit count
                score += 0.001 * pool[i].visit_count as f32;
                if score > best_score {
                    best_score = score;
                    best_idx = Some(i);
                }
            }
            best_idx
        };

    let select = |idx: usize, selected: &mut Vec<usize>, used: &mut Vec<bool>,
                      phase_count: &mut [usize; 3], dtype_count: &mut [usize; 3],
                      theta_count: &mut usize, gamma_count: &mut usize,
                      depth_count: &mut usize, beta_count: &mut usize,
                      turn_set: &mut Vec<usize>,
                      action_label_count: &mut HashMap<String, usize>| {
        let sc = &pool[idx];
        selected.push(idx);
        used[idx] = true;

        // Block all functionally-equivalent candidates (same fingerprint)
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
        // Track action-fatigue
        for lbl in &sc.top_action_labels {
            *action_label_count.entry(lbl.clone()).or_insert(0) += 1;
        }
    };

    // Phase 1: Fill must-fill quotas
    loop {
        if selected.len() >= n_scenarios {
            break;
        }

        // Find the most under-served constraint
        let mut worst_deficit = 0i32;
        let mut worst_type = 0u8; // 0=none, 1=phase, 2=dtype, 3=theta, 4=gamma, 5=depth, 6=beta
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
            break; // All quotas met
        }

        let idx = match worst_type {
            1 => {
                let target_phase = match worst_param {
                    0 => GamePhase::Early,
                    1 => GamePhase::Mid,
                    _ => GamePhase::Late,
                };
                find_best(&used, &turn_set, theta_count, gamma_count, &action_label_count, &|i| pool[i].bucket.phase == target_phase)
            }
            2 => {
                let target_dtype = match worst_param {
                    0 => DecisionType::Reroll1,
                    1 => DecisionType::Reroll2,
                    _ => DecisionType::Category,
                };
                find_best(&used, &turn_set, theta_count, gamma_count, &action_label_count, &|i| pool[i].bucket.dtype == target_dtype)
            }
            3 => find_best(&used, &turn_set, theta_count, gamma_count, &action_label_count, &|i| pool[i].scores.s_theta >= theta_thresh),
            4 => find_best(&used, &turn_set, theta_count, gamma_count, &action_label_count, &|i| pool[i].scores.s_gamma >= gamma_thresh),
            5 => find_best(&used, &turn_set, theta_count, gamma_count, &action_label_count, &|i| pool[i].scores.s_d > 0.0),
            6 => find_best(&used, &turn_set, theta_count, gamma_count, &action_label_count, &|i| pool[i].scores.s_beta >= beta_thresh),
            _ => None,
        };

        match idx {
            Some(i) => select(
                i, &mut selected, &mut used,
                &mut phase_count, &mut dtype_count,
                &mut theta_count, &mut gamma_count, &mut depth_count,
                &mut beta_count, &mut turn_set,
                &mut action_label_count,
            ),
            None => break, // Can't fill this constraint, move on
        }
    }

    // Phase 2: Fill remaining slots with best total diagnostic score,
    // strongly preferring turn diversity and under-represented buckets
    while selected.len() < n_scenarios {
        let mut best_idx: Option<usize> = None;
        let mut best_score = f32::NEG_INFINITY;

        for i in 0..pool.len() {
            if used[i] {
                continue;
            }
            let sc = &pool[i];

            // Hard skip: over-represented quadrants
            let is_theta_dominant = sc.scores.s_theta >= theta_thresh;
            let is_gamma_dominant = sc.scores.s_gamma >= gamma_thresh
                && sc.scores.s_theta < theta_thresh;
            if is_theta_dominant && theta_count >= max_theta {
                continue;
            }
            if is_gamma_dominant && gamma_count >= max_gamma {
                continue;
            }
            // Action-fatigue: skip if any top-2 action label is over the limit
            let fatigued = sc.top_action_labels.iter().any(|lbl| {
                *action_label_count.get(lbl).unwrap_or(&0) >= max_action_fatigue
            });
            if fatigued {
                continue;
            }

            let mut score = sc.scores.total_score();

            // Strong bonus for covering a new turn number
            if !turn_set.contains(&sc.decision.turn) {
                score += 5.0;
            }

            // Bonus for under-represented phases/dtypes
            let pi = phase_idx(sc.bucket.phase);
            let di = dtype_idx(sc.bucket.dtype);
            if phase_count[pi] < min_per_phase {
                score += 3.0;
            }
            if dtype_count[di] < min_per_dtype {
                score += 3.0;
            }

            // Penalty when approaching caps
            if is_theta_dominant && theta_count >= max_theta.saturating_sub(2) {
                score -= 2.0;
            }
            if is_gamma_dominant && gamma_count >= max_gamma.saturating_sub(2) {
                score -= 2.0;
            }

            // Bonus for under-represented diagnostic types
            if sc.scores.s_gamma >= gamma_thresh && gamma_count < min_gamma {
                score += 3.0;
            }
            if sc.scores.s_d > 0.0 && depth_count < min_depth {
                score += 3.0;
            }
            if sc.scores.s_beta >= beta_thresh && beta_count < min_beta {
                score += 3.0;
            }

            // Small tiebreak by visit count
            score += 0.001 * sc.visit_count as f32;

            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }

        match best_idx {
            Some(i) => select(
                i, &mut selected, &mut used,
                &mut phase_count, &mut dtype_count,
                &mut theta_count, &mut gamma_count, &mut depth_count,
                &mut beta_count, &mut turn_set,
                &mut action_label_count,
            ),
            None => break,
        }
    }

    // Sort by turn number for narrative flow
    selected.sort_by_key(|&i| pool[i].decision.turn);

    selected
}

/// Determine primary quadrant for a scored candidate.
/// Uses a priority scheme: theta and gamma are more specific diagnostics,
/// so they take priority when their scores are high.
fn primary_quadrant(sc: &ScoredCandidate) -> Quadrant {
    let s = &sc.scores;
    // θ-diagnostic: reroll action flips across θ values
    if s.s_theta >= 0.5 {
        return Quadrant::Theta;
    }
    // γ-diagnostic: greedy ≠ optimal for category decisions
    if s.s_gamma >= 0.5 {
        return Quadrant::Gamma;
    }
    // d-diagnostic: heuristic ≠ optimal
    if s.s_d > 0.0 {
        return Quadrant::Depth;
    }
    // β-diagnostic: everything else (gap-based discrimination)
    Quadrant::Beta
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
        let actions = get_actions(ctx, sv, d);
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

fn get_actions(ctx: &YatzyContext, sv: &[f32], d: &RawDecision) -> Vec<ActionInfo> {
    let ds_index = find_dice_set_index(ctx, &d.dice);
    let is_last = d.turn == CATEGORY_COUNT - 1;
    match d.decision_type {
        DecisionType::Reroll1 => {
            let mut e_ds_0 = [0.0f32; 252];
            let mut e_ds_1 = [0.0f32; 252];
            compute_group6(ctx, sv, d.upper_score, d.scored, &mut e_ds_0);
            compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);
            enumerate_reroll_actions(ctx, &e_ds_1, &d.dice)
        }
        DecisionType::Reroll2 => {
            let mut e_ds_0 = [0.0f32; 252];
            compute_group6(ctx, sv, d.upper_score, d.scored, &mut e_ds_0);
            enumerate_reroll_actions(ctx, &e_ds_0, &d.dice)
        }
        DecisionType::Category => {
            enumerate_category_actions(ctx, sv, d.upper_score, d.scored, ds_index, is_last)
        }
    }
}

fn make_description(d: &RawDecision, actions: &[ActionInfo], gap: f32) -> String {
    let phase = match d.decision_type {
        DecisionType::Reroll1 => "1st reroll",
        DecisionType::Reroll2 => "2nd reroll",
        DecisionType::Category => "category choice",
    };
    let remaining = CATEGORY_COUNT - (d.scored.count_ones() as usize);
    let best = actions.first().map(|a| a.label.as_str()).unwrap_or("?");
    let runner = actions.get(1).map(|a| a.label.as_str()).unwrap_or("?");
    format!(
        "Turn {}, {}: {:?}, {} open. {} vs {}, gap {:.2}",
        d.turn + 1,
        phase,
        d.dice,
        remaining,
        best,
        runner,
        gap,
    )
}

/// Compute Q-value grid for a single scenario across parameter combinations.
pub fn compute_q_grid(
    ctx: &YatzyContext,
    scenario: &ProfilingScenario,
    theta_tables: &[(f32, StateValues)],
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
