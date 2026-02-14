//! Multiplayer Yatzy simulation.
//!
//! N-player round-robin games where each player uses a named [`Strategy`].
//! All game state is public: every player can see every other player's
//! `(upper_score, scored_categories, total_score)` at every decision point.
//!
//! Strategies can be opponent-aware (e.g., play riskier when trailing).

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::Serialize;

use crate::constants::*;
use crate::dice_mechanics::{find_dice_set_index, sort_dice_set};
use crate::game_mechanics::update_upper_score;
use crate::types::YatzyContext;
use crate::widget_solver::{
    choose_best_reroll_mask, choose_best_reroll_mask_risk, compute_max_ev_for_n_rerolls,
    compute_opt_lse_for_n_rerolls,
};

use super::adaptive::TurnConfig;
use super::strategy::{GameView, PlayerState, Strategy};

// ── Inline helpers (duplicated for perf, same as engine.rs / adaptive.rs) ─

#[inline(always)]
fn roll_dice(rng: &mut SmallRng) -> [i32; 5] {
    let mut dice = [0i32; 5];
    for d in &mut dice {
        *d = rng.random_range(1..=6);
    }
    sort_dice_set(&mut dice);
    dice
}

#[inline(always)]
fn apply_reroll(dice: &mut [i32; 5], mask: i32, rng: &mut SmallRng) {
    for i in 0..5 {
        if mask & (1 << i) != 0 {
            dice[i] = rng.random_range(1..=6);
        }
    }
    sort_dice_set(dice);
}

#[inline(always)]
fn compute_group6_ev(
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

#[inline(always)]
fn compute_group6_risk(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    theta: f32,
    minimize: bool,
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
        let mut best_val = if minimize {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };
        for c in 0..6 {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_i][c];
                let new_up = update_upper_score(up_score, c, scr);
                let new_scored = scored | (1 << c);
                let val = theta * scr as f32
                    + unsafe {
                        *sv.get_unchecked(state_index(new_up as usize, new_scored as usize))
                    };
                let better = if minimize {
                    val < best_val
                } else {
                    val > best_val
                };
                if better {
                    best_val = val;
                }
            }
        }
        for c in 6..CATEGORY_COUNT {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_i][c];
                let val = theta * scr as f32 + unsafe { *lower_succ_ev.get_unchecked(c) };
                let better = if minimize {
                    val < best_val
                } else {
                    val > best_val
                };
                if better {
                    best_val = val;
                }
            }
        }
        e_ds_0[ds_i] = best_val;
    }
}

#[inline(always)]
fn find_best_category_ev(
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

#[inline(always)]
fn find_best_category_risk(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    ds_index: usize,
    theta: f32,
    minimize: bool,
) -> (usize, i32) {
    let mut best_val = if minimize {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };
    let mut best_cat = 0usize;
    let mut best_score = 0i32;
    for c in 0..6 {
        if !is_category_scored(scored, c) {
            let scr = ctx.precomputed_scores[ds_index][c];
            let new_up = update_upper_score(up_score, c, scr);
            let new_scored = scored | (1 << c);
            let val = theta * scr as f32
                + unsafe { *sv.get_unchecked(state_index(new_up as usize, new_scored as usize)) };
            let better = if minimize {
                val < best_val
            } else {
                val > best_val
            };
            if better {
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
            let val = theta * scr as f32
                + unsafe { *sv.get_unchecked(state_index(up_score as usize, new_scored as usize)) };
            let better = if minimize {
                val < best_val
            } else {
                val > best_val
            };
            if better {
                best_val = val;
                best_cat = c;
                best_score = scr;
            }
        }
    }
    (best_cat, best_score)
}

#[inline(always)]
fn find_best_category_final_ev(
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

#[inline(always)]
fn find_best_category_final_risk(
    ctx: &YatzyContext,
    up_score: i32,
    scored: i32,
    ds_index: usize,
    theta: f32,
    minimize: bool,
) -> (usize, i32) {
    let mut best_val = if minimize {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };
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
            let val = theta * (scr + bonus) as f32;
            let better = if minimize {
                val < best_val
            } else {
                val > best_val
            };
            if better {
                best_val = val;
                best_cat = c;
                best_score = scr;
            }
        }
    }
    (best_cat, best_score)
}

// ── Single-turn play ──────────────────────────────────────────────────────

/// Play one turn for a player: roll → reroll → reroll → score.
/// Returns (category, score, new_upper_score).
#[inline]
fn play_single_turn(
    ctx: &YatzyContext,
    config: &TurnConfig,
    state: &PlayerState,
    turn: usize,
    rng: &mut SmallRng,
) -> (usize, i32, i32) {
    let sv = config.sv;
    let theta = config.theta;
    let minimize = config.minimize;
    let use_risk = theta != 0.0;
    let is_last_turn = turn == CATEGORY_COUNT - 1;

    let up_score = state.upper_score;
    let scored = state.scored_categories;

    let mut dice = roll_dice(rng);
    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    if use_risk {
        compute_group6_risk(ctx, sv, up_score, scored, theta, minimize, &mut e_ds_0);
        compute_opt_lse_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1, minimize);

        let mut best_ev = 0.0;
        let mask1 = choose_best_reroll_mask_risk(ctx, &e_ds_1, &dice, &mut best_ev, minimize);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }

        let mask2 = choose_best_reroll_mask_risk(ctx, &e_ds_0, &dice, &mut best_ev, minimize);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }

        let ds_index = find_dice_set_index(ctx, &dice);
        let (cat, scr) = if is_last_turn {
            find_best_category_final_risk(ctx, up_score, scored, ds_index, theta, minimize)
        } else {
            find_best_category_risk(ctx, sv, up_score, scored, ds_index, theta, minimize)
        };

        let new_upper = update_upper_score(up_score, cat, scr);
        (cat, scr, new_upper)
    } else {
        compute_group6_ev(ctx, sv, up_score, scored, &mut e_ds_0);
        compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);

        let mut best_ev = 0.0;
        let mask1 = choose_best_reroll_mask(ctx, &e_ds_1, &dice, &mut best_ev);
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }

        let mask2 = choose_best_reroll_mask(ctx, &e_ds_0, &dice, &mut best_ev);
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }

        let ds_index = find_dice_set_index(ctx, &dice);
        let (cat, scr) = if is_last_turn {
            find_best_category_final_ev(ctx, up_score, scored, ds_index)
        } else {
            find_best_category_ev(ctx, sv, up_score, scored, ds_index)
        };

        let new_upper = update_upper_score(up_score, cat, scr);
        (cat, scr, new_upper)
    }
}

// ── Recording ────────────────────────────────────────────────────────────

/// Packed per-game record for binary storage (64 bytes for 2 players).
///
/// - `scores`: final scores including upper bonus
/// - `turn_totals`: running total (excl. bonus) after each turn
#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
pub struct MultiplayerGameRecord {
    pub scores: [i16; 2],            // 4 bytes
    pub turn_totals: [[i16; 15]; 2], // 60 bytes
}

const _RECORD_SIZE: () = assert!(std::mem::size_of::<MultiplayerGameRecord>() == 64);

// ── Game outcome ──────────────────────────────────────────────────────────

/// Per-game result for all players.
pub struct GameOutcome {
    pub scores: Vec<i32>,
    pub winner: Option<usize>,
}

/// Aggregate results across all games.
#[derive(Serialize)]
pub struct MultiplayerResult {
    pub strategies: Vec<String>,
    pub games: u32,
    pub wins: Vec<u32>,
    pub win_rates: Vec<f64>,
    pub draws: u32,
    pub score_means: Vec<f64>,
    pub score_stds: Vec<f64>,
    pub head_to_head: Vec<Vec<u32>>,
    pub avg_margin_when_winning: Vec<f64>,
}

// ── Game loop ─────────────────────────────────────────────────────────────

/// Play one multiplayer game (15 rounds, round-robin).
fn play_one_game(ctx: &YatzyContext, strategies: &[Strategy], rng: &mut SmallRng) -> GameOutcome {
    let n = strategies.len();
    let mut states = vec![PlayerState::default(); n];

    for turn in 0..CATEGORY_COUNT {
        for player_idx in 0..n {
            let view = GameView {
                my_index: player_idx,
                players: &states,
                turn,
            };
            let config = strategies[player_idx].resolve_turn(&view);

            let (cat, scr, new_upper) =
                play_single_turn(ctx, &config, &states[player_idx], turn, rng);

            states[player_idx].upper_score = new_upper;
            states[player_idx].scored_categories |= 1 << cat;
            states[player_idx].total_score += scr;
        }
    }

    // Apply upper bonuses
    let final_scores: Vec<i32> = states
        .iter()
        .map(|s| s.total_score + if s.upper_score >= 63 { 50 } else { 0 })
        .collect();

    // Determine winner (None if draw)
    let max_score = *final_scores.iter().max().unwrap();
    let winners: Vec<usize> = final_scores
        .iter()
        .enumerate()
        .filter(|(_, &s)| s == max_score)
        .map(|(i, _)| i)
        .collect();

    let winner = if winners.len() == 1 {
        Some(winners[0])
    } else {
        None
    };

    GameOutcome {
        scores: final_scores,
        winner,
    }
}

/// Batch multiplayer simulation with rayon parallelism.
pub fn simulate_multiplayer(
    ctx: &YatzyContext,
    strategies: &[Strategy],
    games: u32,
    seed: u64,
) -> MultiplayerResult {
    let n = strategies.len();

    // Run all games in parallel, collect outcomes
    let outcomes: Vec<GameOutcome> = (0..games as u64)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i));
            play_one_game(ctx, strategies, &mut rng)
        })
        .collect();

    // Aggregate
    let mut wins = vec![0u32; n];
    let mut draws = 0u32;
    let mut score_sums = vec![0.0f64; n];
    let mut score_sq_sums = vec![0.0f64; n];
    let mut head_to_head = vec![vec![0u32; n]; n];
    let mut margin_sums = vec![0.0f64; n];
    let mut win_counts_for_margin = vec![0u32; n];

    for outcome in &outcomes {
        // Score stats
        for (i, &s) in outcome.scores.iter().enumerate() {
            score_sums[i] += s as f64;
            score_sq_sums[i] += (s as f64) * (s as f64);
        }

        // Wins/draws
        match outcome.winner {
            Some(w) => {
                wins[w] += 1;
                let winner_score = outcome.scores[w];
                // Head-to-head: winner beats all others
                for j in 0..n {
                    if j != w {
                        head_to_head[w][j] += 1;
                    }
                }
                // Margin: winner score minus second-best
                let second_best = outcome
                    .scores
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != w)
                    .map(|(_, &s)| s)
                    .max()
                    .unwrap_or(0);
                margin_sums[w] += (winner_score - second_best) as f64;
                win_counts_for_margin[w] += 1;
            }
            None => {
                draws += 1;
                // For head-to-head in draws: each player beats those with lower scores
                for i in 0..n {
                    for j in 0..n {
                        if i != j && outcome.scores[i] > outcome.scores[j] {
                            head_to_head[i][j] += 1;
                        }
                    }
                }
            }
        }
    }

    let g = games as f64;
    let score_means: Vec<f64> = score_sums.iter().map(|&s| s / g).collect();
    let score_stds: Vec<f64> = score_sq_sums
        .iter()
        .zip(score_means.iter())
        .map(|(&sq, &m)| (sq / g - m * m).max(0.0).sqrt())
        .collect();
    let win_rates: Vec<f64> = wins.iter().map(|&w| w as f64 / g * 100.0).collect();
    let avg_margin_when_winning: Vec<f64> = margin_sums
        .iter()
        .zip(win_counts_for_margin.iter())
        .map(|(&m, &c)| if c > 0 { m / c as f64 } else { 0.0 })
        .collect();

    MultiplayerResult {
        strategies: strategies.iter().map(|s| s.name.clone()).collect(),
        games,
        wins,
        win_rates,
        draws,
        score_means,
        score_stds,
        head_to_head,
        avg_margin_when_winning,
    }
}

// ── Recording game loop ──────────────────────────────────────────────────

/// Play one multiplayer game recording per-turn running totals.
/// Only supports 2-player games for the packed record format.
fn play_one_game_with_recording(
    ctx: &YatzyContext,
    strategies: &[Strategy],
    rng: &mut SmallRng,
) -> MultiplayerGameRecord {
    assert!(strategies.len() == 2, "Recording only supports 2 players");
    let mut states = vec![PlayerState::default(); 2];
    let mut record = MultiplayerGameRecord::default();

    for turn in 0..CATEGORY_COUNT {
        for player_idx in 0..2 {
            let view = GameView {
                my_index: player_idx,
                players: &states,
                turn,
            };
            let config = strategies[player_idx].resolve_turn(&view);

            let (cat, scr, new_upper) =
                play_single_turn(ctx, &config, &states[player_idx], turn, rng);

            states[player_idx].upper_score = new_upper;
            states[player_idx].scored_categories |= 1 << cat;
            states[player_idx].total_score += scr;

            record.turn_totals[player_idx][turn] = states[player_idx].total_score as i16;
        }
    }

    // Final scores with bonus
    for i in 0..2 {
        let bonus = if states[i].upper_score >= 63 { 50 } else { 0 };
        record.scores[i] = (states[i].total_score + bonus) as i16;
    }

    record
}

/// Batch multiplayer simulation with recording (rayon parallel).
/// Returns Vec of per-game records (only supports 2 players).
pub fn simulate_multiplayer_with_recording(
    ctx: &YatzyContext,
    strategies: &[Strategy],
    games: u32,
    seed: u64,
) -> Vec<MultiplayerGameRecord> {
    assert!(strategies.len() == 2, "Recording only supports 2 players");

    (0..games as u64)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i));
            play_one_game_with_recording(ctx, strategies, &mut rng)
        })
        .collect()
}

/// Derive MultiplayerResult from recorded games (no re-simulation needed).
pub fn aggregate_from_records(
    records: &[MultiplayerGameRecord],
    strategy_names: &[String],
) -> MultiplayerResult {
    let n = 2usize;
    let games = records.len() as u32;

    let mut wins = vec![0u32; n];
    let mut draws = 0u32;
    let mut score_sums = vec![0.0f64; n];
    let mut score_sq_sums = vec![0.0f64; n];
    let mut head_to_head = vec![vec![0u32; n]; n];
    let mut margin_sums = vec![0.0f64; n];
    let mut win_counts_for_margin = vec![0u32; n];

    for rec in records {
        let s0 = rec.scores[0] as i32;
        let s1 = rec.scores[1] as i32;
        let scores = [s0, s1];

        for i in 0..n {
            score_sums[i] += scores[i] as f64;
            score_sq_sums[i] += (scores[i] as f64) * (scores[i] as f64);
        }

        if s0 > s1 {
            wins[0] += 1;
            head_to_head[0][1] += 1;
            margin_sums[0] += (s0 - s1) as f64;
            win_counts_for_margin[0] += 1;
        } else if s1 > s0 {
            wins[1] += 1;
            head_to_head[1][0] += 1;
            margin_sums[1] += (s1 - s0) as f64;
            win_counts_for_margin[1] += 1;
        } else {
            draws += 1;
        }
    }

    let g = games as f64;
    let score_means: Vec<f64> = score_sums.iter().map(|&s| s / g).collect();
    let score_stds: Vec<f64> = score_sq_sums
        .iter()
        .zip(score_means.iter())
        .map(|(&sq, &m)| (sq / g - m * m).max(0.0).sqrt())
        .collect();
    let win_rates: Vec<f64> = wins.iter().map(|&w| w as f64 / g * 100.0).collect();
    let avg_margin_when_winning: Vec<f64> = margin_sums
        .iter()
        .zip(win_counts_for_margin.iter())
        .map(|(&m, &c)| if c > 0 { m / c as f64 } else { 0.0 })
        .collect();

    MultiplayerResult {
        strategies: strategy_names.to_vec(),
        games,
        wins,
        win_rates,
        draws,
        score_means,
        score_stds,
        head_to_head,
        avg_margin_when_winning,
    }
}
