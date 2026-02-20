//! Classification and validation for scenario candidates.
//!
//! Game phase, board tension, diagnostic scoring, fingerprinting, and
//! realistic-state validation.

#![allow(clippy::needless_range_loop)]

use crate::constants::*;
use crate::dice_mechanics::{count_faces, find_dice_set_index};
use crate::profiling::qvalues::{compute_group6_risk_profiling, compute_q_rerolls};
use crate::scenarios::actions::{compute_group6, find_best_category};
use crate::scenarios::types::*;
use crate::simulation::heuristic::{heuristic_pick_category, heuristic_reroll_mask};
use crate::types::YatzyContext;
use crate::widget_solver::{
    choose_best_reroll_mask, compute_max_ev_for_n_rerolls, compute_opt_lse_for_n_rerolls,
};

// ── Phase classification ──

pub fn classify_phase(turn: usize) -> GamePhase {
    match turn {
        0..=4 => GamePhase::Early,
        5..=9 => GamePhase::Mid,
        _ => GamePhase::Late,
    }
}

// ── Board tension classification ──

pub fn classify_tension(scored: i32, upper_score: i32, dice: &[i32; 5]) -> BoardTension {
    let num_scored = scored.count_ones() as usize;
    let num_open = CATEGORY_COUNT - num_scored;

    if num_open <= 3 {
        return BoardTension::Cleanup;
    }

    let upper_open = (0..6).filter(|&c| !is_category_scored(scored, c)).count();
    if upper_open > 0 && (30..63).contains(&upper_score) {
        return BoardTension::BonusChase;
    }

    let face_count = count_faces(dice);
    let has_pattern_potential = {
        let straight_open = !is_category_scored(scored, CATEGORY_SMALL_STRAIGHT)
            || !is_category_scored(scored, CATEGORY_LARGE_STRAIGHT);
        let full_house_open = !is_category_scored(scored, CATEGORY_FULL_HOUSE);
        let yatzy_open = !is_category_scored(scored, CATEGORY_YATZY);

        let has_run = {
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
pub fn classify_bucket(d: &RawDecision) -> SemanticBucket {
    SemanticBucket {
        phase: classify_phase(d.turn),
        dtype: d.decision_type,
        tension: classify_tension(d.scored, d.upper_score, &d.dice),
    }
}

// ── Realistic state validation ──

pub fn is_realistic(
    upper_score: i32,
    scored: i32,
    turn: usize,
    category_scores: &[i32; 15],
) -> bool {
    let mut zero_count = 0;
    let mut scored_sum = 0i32;
    let mut upper_scored_count = 0;
    for c in 0..CATEGORY_COUNT {
        if is_category_scored(scored, c) {
            let scr = category_scores[c];
            if scr == 0 {
                zero_count += 1;
            }
            scored_sum += scr;
            if c < 6 {
                upper_scored_count += 1;
            }
        }
    }
    let max_zeros = (turn + 3) / 3;
    if zero_count > max_zeros {
        return false;
    }
    if turn <= 5 && upper_scored_count >= 3 && upper_score == 0 {
        return false;
    }
    if turn >= 3 && scored_sum < (turn as i32) * 5 {
        return false;
    }
    true
}

/// Full candidate validation — returns None if valid, or a description of the failure.
pub fn validate_candidate(d: &RawDecision) -> Option<String> {
    // 1. scored_count == turn
    let scored_count = d.scored.count_ones() as usize;
    if scored_count != d.turn {
        return Some(format!(
            "scored_count ({}) != turn ({})",
            scored_count, d.turn
        ));
    }

    // 2. upper_score consistency
    let mut computed_upper: i32 = 0;
    for c in 0..6 {
        if is_category_scored(d.scored, c) {
            let scr = d.category_scores[c];
            if scr < 0 {
                return Some(format!("scored category {} has score {}", c, scr));
            }
            computed_upper += scr;
        }
    }
    let expected_upper = computed_upper.min(63);
    if d.upper_score != expected_upper {
        return Some(format!(
            "upper_score ({}) != min(sum(upper_scores), 63) ({})",
            d.upper_score, expected_upper
        ));
    }

    // 3. Category score consistency
    for c in 0..CATEGORY_COUNT {
        if is_category_scored(d.scored, c) {
            if d.category_scores[c] < 0 {
                return Some(format!(
                    "scored category {} has negative score {}",
                    c, d.category_scores[c]
                ));
            }
        } else if d.category_scores[c] != -1 {
            return Some(format!(
                "unscored category {} has score {} (expected -1)",
                c, d.category_scores[c]
            ));
        }
    }

    // 4. Dice valid
    for (i, &die) in d.dice.iter().enumerate() {
        if !(1..=6).contains(&die) {
            return Some(format!("dice[{}] = {} (out of range)", i, die));
        }
    }
    for i in 0..4 {
        if d.dice[i] > d.dice[i + 1] {
            return Some(format!(
                "dice not sorted: dice[{}]={} > dice[{}]={}",
                i,
                d.dice[i],
                i + 1,
                d.dice[i + 1]
            ));
        }
    }

    // 5. Turn valid
    if d.turn > 14 {
        return Some(format!("turn {} out of range", d.turn));
    }

    None
}

// ── Fingerprinting ──

pub fn compute_fingerprint(
    _ctx: &YatzyContext,
    _sv: &[f32],
    d: &RawDecision,
    actions: &[ActionInfo],
) -> String {
    let top_evs: Vec<i32> = actions
        .iter()
        .take(3)
        .map(|a| (a.ev * 2.0).round() as i32)
        .collect();
    let top_ids: Vec<i32> = actions.iter().take(3).map(|a| a.id).collect();

    match d.decision_type {
        DecisionType::Category => {
            format!(
                "cat:s{},u{},ids{:?},evs{:?}",
                d.scored, d.upper_score, top_ids, top_evs
            )
        }
        _ => {
            format!(
                "{:?}:s{},u{},{:?},ids{:?},evs{:?}",
                d.decision_type, d.scored, d.upper_score, d.dice, top_ids, top_evs
            )
        }
    }
}

// ── Diagnostic scoring ──

/// Compute S_θ: how much the optimal action changes across θ values.
pub fn compute_s_theta(
    ctx: &YatzyContext,
    d: &RawDecision,
    theta_tables: &[(f32, &[f32])],
) -> f32 {
    if !matches!(
        d.decision_type,
        DecisionType::Reroll1 | DecisionType::Reroll2
    ) {
        return 0.0;
    }

    let mut rankings: Vec<(f32, Vec<i32>)> = Vec::new();

    for &(theta, sv) in theta_tables {
        let is_risk = theta != 0.0;
        let mut e_ds_0 = [0.0f32; 252];
        let mut e_ds_1 = [0.0f32; 252];

        if is_risk {
            compute_group6_risk_profiling(
                ctx,
                sv,
                d.upper_score,
                d.scored,
                theta,
                1.0,
                0.0,
                &mut e_ds_0,
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

    let neg = rankings.iter().find(|(t, _)| *t < -0.01);
    let pos = rankings.iter().find(|(t, _)| *t > 0.01);

    match (neg, pos) {
        (Some((_, rank_neg)), Some((_, rank_pos))) => {
            if rank_neg.is_empty() || rank_pos.is_empty() {
                return 0.0;
            }
            if rank_neg[0] != rank_pos[0] {
                1.0
            } else if rank_neg.len() >= 2 && rank_pos.len() >= 2 && rank_neg[1] != rank_pos[1] {
                0.5
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

/// Compute S_γ: how much greedy differs from optimal.
pub fn compute_s_gamma(ctx: &YatzyContext, sv: &[f32], d: &RawDecision) -> f32 {
    if d.decision_type != DecisionType::Category {
        return match classify_phase(d.turn) {
            GamePhase::Mid => 0.2,
            _ => 0.0,
        };
    }
    if d.turn == CATEGORY_COUNT - 1 {
        return 0.0;
    }

    let ds_index = find_dice_set_index(ctx, &d.dice);

    let mut greedy_ranking: Vec<(usize, i32)> = Vec::new();
    for c in 0..CATEGORY_COUNT {
        if !is_category_scored(d.scored, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            greedy_ranking.push((c, scr));
        }
    }
    greedy_ranking.sort_by(|a, b| b.1.cmp(&a.1));

    let (optimal_cat, _) = find_best_category(ctx, sv, d.upper_score, d.scored, ds_index);

    if let Some(greedy_top) = greedy_ranking.first() {
        if greedy_top.0 == optimal_cat {
            0.0
        } else {
            let pos = greedy_ranking
                .iter()
                .position(|(c, _)| *c == optimal_cat)
                .unwrap_or(greedy_ranking.len());
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
pub fn compute_s_d(ctx: &YatzyContext, sv: &[f32], d: &RawDecision) -> f32 {
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
            let (optimal_cat, _) = find_best_category(ctx, sv, d.upper_score, d.scored, ds_index);
            if heuristic_cat != optimal_cat {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Compute all diagnostic scores for a candidate.
pub fn compute_diagnostic_scores(
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
        s_beta: (ev_gap / 20.0).min(1.0),
    }
}

/// Determine primary quadrant for a scored candidate.
pub fn primary_quadrant(sc: &ScoredCandidate) -> Quadrant {
    let s = &sc.scores;
    if s.s_theta >= 0.5 {
        return Quadrant::Theta;
    }
    if s.s_gamma >= 0.5 {
        return Quadrant::Gamma;
    }
    if s.s_d > 0.0 {
        return Quadrant::Depth;
    }
    Quadrant::Beta
}

