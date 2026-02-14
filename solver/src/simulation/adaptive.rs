//! Adaptive θ policies — switch precomputed tables per-turn based on game state.
//!
//! Fixed θ applies a uniform risk preference across all states. Adaptive policies
//! vary θ per turn based on features like upper bonus status, game phase, and
//! remaining high-variance categories. This can beat the Pareto frontier of fixed-θ
//! policies because it avoids paying the mean penalty in states where risk is expensive.
//!
//! ## Architecture
//!
//! Each policy holds references to multiple precomputed state-value tables (one per θ).
//! At the start of each turn, `select_theta_index` examines the current game state and
//! returns which table to use. All decisions within that turn use the selected table.
//!
//! The successor state values were computed assuming the SAME θ for all future turns.
//! Switching tables between turns violates this assumption, but for small |Δθ| the
//! approximation is excellent (policies differ on <5% of decisions).

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::constants::*;
use crate::dice_mechanics::{find_dice_set_index, sort_dice_set};
use crate::game_mechanics::update_upper_score;
use crate::types::{StateValues, YatzyContext};
use crate::widget_solver::{
    choose_best_reroll_mask, choose_best_reroll_mask_risk, compute_max_ev_for_n_rerolls,
    compute_opt_lse_for_n_rerolls,
};

use super::engine::{GameRecord, TurnRecord};

/// A precomputed state-value table for one fixed θ.
pub struct ThetaTable {
    pub theta: f32,
    pub sv: StateValues,
    pub minimize: bool,
}

/// Configuration for one turn: which θ/table to use.
pub struct TurnConfig<'a> {
    pub theta: f32,
    pub sv: &'a [f32],
    pub minimize: bool,
}

// ── State feature helpers ───────────────────────────────────────────────────

/// Number of categories remaining to score.
#[inline(always)]
fn categories_left(scored: i32) -> usize {
    CATEGORY_COUNT - (scored.count_ones() as usize)
}

/// Number of upper-section categories (0-5) remaining to score.
#[inline(always)]
fn upper_categories_left(scored: i32) -> usize {
    6 - ((scored & 0x3F).count_ones() as usize)
}

/// Whether the upper bonus has been secured (upper_score >= 63).
#[inline(always)]
fn bonus_secured(upper_score: i32) -> bool {
    upper_score >= 63
}

/// Maximum remaining upper-section score: sum of face*5 for each unscored upper category.
#[inline(always)]
fn max_remaining_upper(scored: i32) -> i32 {
    let mut total = 0;
    for c in 0..6 {
        if !is_category_scored(scored, c) {
            total += (c as i32 + 1) * 5;
        }
    }
    total
}

/// Whether reaching the upper bonus (63) is still possible.
#[inline(always)]
fn bonus_reachable(upper_score: i32, scored: i32) -> bool {
    upper_score + max_remaining_upper(scored) >= 63
}

/// Whether any high-variance lower categories remain (Straight, Yatzy, Full House).
#[inline(always)]
fn has_high_variance_left(scored: i32) -> bool {
    !is_category_scored(scored, CATEGORY_SMALL_STRAIGHT)
        || !is_category_scored(scored, CATEGORY_LARGE_STRAIGHT)
        || !is_category_scored(scored, CATEGORY_YATZY)
        || !is_category_scored(scored, CATEGORY_FULL_HOUSE)
}

// ── Adaptive policy trait ───────────────────────────────────────────────────

/// An adaptive policy selects which θ table to use at the start of each turn.
pub trait AdaptivePolicy: Send + Sync {
    /// Human-readable name for this policy.
    fn name(&self) -> &str;

    /// Select which θ table to use for this turn.
    /// Returns the index into the `tables` array.
    fn select_theta_index(&self, upper_score: i32, scored: i32, turn: usize) -> usize;
}

// ── Policy 1: Bonus-Adaptive ────────────────────────────────────────────────

/// Formalizes the "Dad's strategy": protect the bonus, go for tail when safe.
///
/// - Bonus secured → use θ_high (risk is cheap)
/// - Bonus reachable → use θ=0 (protect the bonus)
/// - Bonus unreachable → use θ_high (nothing to protect)
pub struct BonusAdaptive {
    /// Index of θ=0 table
    pub ev_idx: usize,
    /// Index of θ_high table
    pub high_idx: usize,
}

impl AdaptivePolicy for BonusAdaptive {
    fn name(&self) -> &str {
        "bonus-adaptive"
    }

    fn select_theta_index(&self, upper_score: i32, scored: i32, _turn: usize) -> usize {
        if bonus_secured(upper_score) {
            self.high_idx
        } else if bonus_reachable(upper_score, scored) {
            self.ev_idx
        } else {
            self.high_idx
        }
    }
}

// ── Policy 2: Phase-Based ───────────────────────────────────────────────────

/// Risk tolerance varies by game phase.
///
/// - Early (>=10 left): mild risk (many turns to recover)
/// - Mid (5-9 left): θ=0 (lock in value)
/// - Late (<=4 left): θ_high if bonus secured, else θ=0
pub struct PhaseBased {
    /// Index of θ=0 table
    pub ev_idx: usize,
    /// Index of θ_mild table
    pub mild_idx: usize,
    /// Index of θ_high table
    pub high_idx: usize,
}

impl AdaptivePolicy for PhaseBased {
    fn name(&self) -> &str {
        "phase-based"
    }

    fn select_theta_index(&self, upper_score: i32, scored: i32, _turn: usize) -> usize {
        let left = categories_left(scored);
        if left >= 10 {
            self.mild_idx
        } else if left >= 5 {
            self.ev_idx
        } else if bonus_secured(upper_score) {
            self.high_idx
        } else {
            self.ev_idx
        }
    }
}

// ── Policy 3: Combined ─────────────────────────────────────────────────────

/// Most nuanced policy: bonus health + phase + high-variance awareness.
///
/// Combines multiple signals to select θ:
/// - Bonus secured + high-variance left → aggressive
/// - Bonus secured + only low-variance left → mild
/// - Bonus looks hard + late game → mild risk
/// - Bonus on track → EV-optimal
pub struct Combined {
    /// Index of θ=0 table
    pub ev_idx: usize,
    /// Index of θ_mild table (0.03)
    pub mild_idx: usize,
    /// Index of θ_moderate table (0.05)
    pub moderate_idx: usize,
    /// Index of θ_high table (0.08)
    pub high_idx: usize,
}

impl AdaptivePolicy for Combined {
    fn name(&self) -> &str {
        "combined"
    }

    fn select_theta_index(&self, upper_score: i32, scored: i32, _turn: usize) -> usize {
        let left = categories_left(scored);

        if bonus_secured(upper_score) {
            if has_high_variance_left(scored) && left >= 3 {
                self.high_idx
            } else {
                self.mild_idx
            }
        } else {
            let upper_left = upper_categories_left(scored);
            let max_remaining = max_remaining_upper(scored);
            let deficit = 63 - upper_score;
            // bonus_health: >1.0 means we need more than average per remaining category
            let bonus_health = if upper_left > 0 && max_remaining > 0 {
                deficit as f32 / (max_remaining as f32 * 0.6) // 0.6 ≈ avg scoring fraction
            } else {
                0.0 // no upper categories left, bonus decided
            };

            if bonus_health > 1.5 {
                // Bonus looks hard to get
                if left <= 5 {
                    self.moderate_idx
                } else {
                    self.ev_idx
                }
            } else {
                // Bonus on track — protect it
                self.ev_idx
            }
        }
    }
}

// ── Policy 4: Upper-Deficit ────────────────────────────────────────────────

/// State-dependent θ based on upper-section bonus proximity.
///
/// Concentrates variance-inflation on turns where it's cheap:
/// - Bonus secured → θ_high (risk is free — bonus already locked)
/// - Bonus unreachable → θ_moderate (nothing to protect)
/// - Bonus in play → measure "health" = deficit / max_remaining
///   - health < 0.6: on track → θ=0 (protect the 50-point bonus)
///   - health 0.6-0.8: marginal → θ_mild
///   - health > 0.8: struggling → θ=0 (don't gamble away the slim chance)
pub struct UpperDeficit {
    /// Index of θ=0 table
    pub ev_idx: usize,
    /// Index of θ_mild table (0.03)
    pub mild_idx: usize,
    /// Index of θ_moderate table (0.05)
    pub moderate_idx: usize,
    /// Index of θ_high table (0.10)
    pub high_idx: usize,
}

impl AdaptivePolicy for UpperDeficit {
    fn name(&self) -> &str {
        "upper-deficit"
    }

    fn select_theta_index(&self, upper_score: i32, scored: i32, _turn: usize) -> usize {
        if bonus_secured(upper_score) {
            self.high_idx
        } else if !bonus_reachable(upper_score, scored) {
            self.moderate_idx
        } else {
            let deficit = 63 - upper_score;
            let max_rem = max_remaining_upper(scored);
            let health = if max_rem > 0 {
                deficit as f32 / max_rem as f32
            } else {
                0.0
            };
            if health < 0.6 {
                self.ev_idx
            } else if health < 0.8 {
                self.mild_idx
            } else {
                self.ev_idx // paradoxically: if bonus looks hard, don't gamble it away
            }
        }
    }
}

// ── Policy 5: Always-EV (verification baseline) ────────────────────────────

/// Always selects θ=0. Used to verify adaptive simulation matches standard simulation.
pub struct AlwaysEv {
    pub ev_idx: usize,
}

impl AdaptivePolicy for AlwaysEv {
    fn name(&self) -> &str {
        "always-ev"
    }

    fn select_theta_index(&self, _upper_score: i32, _scored: i32, _turn: usize) -> usize {
        self.ev_idx
    }
}

// ── Simulation engine ───────────────────────────────────────────────────────

/// Roll 5 random dice and sort them.
#[inline(always)]
fn roll_dice(rng: &mut SmallRng) -> [i32; 5] {
    let mut dice = [0i32; 5];
    for d in &mut dice {
        *d = rng.random_range(1..=6);
    }
    sort_dice_set(&mut dice);
    dice
}

/// Apply a reroll mask: bits set in `mask` indicate dice positions to reroll.
#[inline(always)]
fn apply_reroll(dice: &mut [i32; 5], mask: i32, rng: &mut SmallRng) {
    for i in 0..5 {
        if mask & (1 << i) != 0 {
            dice[i] = rng.random_range(1..=6);
        }
    }
    sort_dice_set(dice);
}

/// Group 6: best category EV for each of the 252 dice sets.
/// Standard EV path (θ=0).
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

/// Group 6 for risk-sensitive mode.
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

/// Find best category (EV path).
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

/// Find best category (risk path).
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

/// Find best category for the last turn (EV path — just maximize score+bonus).
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

/// Find best category for the last turn (risk path).
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

/// Convert i32 dice to u8 array for recording.
#[inline(always)]
fn dice_to_u8(dice: &[i32; 5]) -> [u8; 5] {
    [
        dice[0] as u8,
        dice[1] as u8,
        dice[2] as u8,
        dice[3] as u8,
        dice[4] as u8,
    ]
}

/// Simulate one game with an adaptive policy.
///
/// At each turn, the policy selects which θ table to use. When θ=0, the standard
/// EV code path is used (no LSE overhead).
pub fn simulate_game_adaptive(
    ctx: &YatzyContext,
    tables: &[ThetaTable],
    policy: &dyn AdaptivePolicy,
    rng: &mut SmallRng,
) -> i32 {
    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut total_score: i32 = 0;

    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;
        let ti = policy.select_theta_index(up_score, scored, turn);
        let table = &tables[ti];
        let theta = table.theta;
        let sv = table.sv.as_slice();
        let minimize = table.minimize;
        let use_risk = theta != 0.0;

        let mut dice = roll_dice(rng);

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

            up_score = update_upper_score(up_score, cat, scr);
            scored |= 1 << cat;
            total_score += scr;
        } else {
            // Standard EV fast path
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

            up_score = update_upper_score(up_score, cat, scr);
            scored |= 1 << cat;
            total_score += scr;
        }
    }

    if up_score >= 63 {
        total_score += 50;
    }

    total_score
}

/// Simulate one game with recording, using an adaptive policy.
pub fn simulate_game_adaptive_with_recording(
    ctx: &YatzyContext,
    tables: &[ThetaTable],
    policy: &dyn AdaptivePolicy,
    rng: &mut SmallRng,
) -> GameRecord {
    let mut up_score: i32 = 0;
    let mut scored: i32 = 0;
    let mut total_score: i32 = 0;
    let mut record = GameRecord::default();

    let mut e_ds_0 = [0.0f32; 252];
    let mut e_ds_1 = [0.0f32; 252];

    for turn in 0..CATEGORY_COUNT {
        let is_last_turn = turn == CATEGORY_COUNT - 1;
        let ti = policy.select_theta_index(up_score, scored, turn);
        let table = &tables[ti];
        let theta = table.theta;
        let sv = table.sv.as_slice();
        let minimize = table.minimize;
        let use_risk = theta != 0.0;

        let mut dice = roll_dice(rng);
        let dice_initial = dice_to_u8(&dice);

        if use_risk {
            compute_group6_risk(ctx, sv, up_score, scored, theta, minimize, &mut e_ds_0);
            compute_opt_lse_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1, minimize);
        } else {
            compute_group6_ev(ctx, sv, up_score, scored, &mut e_ds_0);
            compute_max_ev_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1);
        }

        let mut best_ev = 0.0;
        let mask1 = if use_risk {
            choose_best_reroll_mask_risk(ctx, &e_ds_1, &dice, &mut best_ev, minimize)
        } else {
            choose_best_reroll_mask(ctx, &e_ds_1, &dice, &mut best_ev)
        };
        if mask1 != 0 {
            apply_reroll(&mut dice, mask1, rng);
        }
        let dice_after_reroll1 = dice_to_u8(&dice);

        let mask2 = if use_risk {
            choose_best_reroll_mask_risk(ctx, &e_ds_0, &dice, &mut best_ev, minimize)
        } else {
            choose_best_reroll_mask(ctx, &e_ds_0, &dice, &mut best_ev)
        };
        if mask2 != 0 {
            apply_reroll(&mut dice, mask2, rng);
        }
        let dice_final = dice_to_u8(&dice);

        let ds_index = find_dice_set_index(ctx, &dice);
        let (cat, scr) = if use_risk {
            if is_last_turn {
                find_best_category_final_risk(ctx, up_score, scored, ds_index, theta, minimize)
            } else {
                find_best_category_risk(ctx, sv, up_score, scored, ds_index, theta, minimize)
            }
        } else if is_last_turn {
            find_best_category_final_ev(ctx, up_score, scored, ds_index)
        } else {
            find_best_category_ev(ctx, sv, up_score, scored, ds_index)
        };

        up_score = update_upper_score(up_score, cat, scr);
        scored |= 1 << cat;
        total_score += scr;

        record.turns[turn] = TurnRecord {
            dice_initial,
            mask1: mask1 as u8,
            dice_after_reroll1,
            mask2: mask2 as u8,
            dice_final,
            category: cat as u8,
            score: scr as u8,
        };
    }

    let got_bonus = up_score >= 63;
    if got_bonus {
        total_score += 50;
    }

    record.total_score = total_score as u16;
    record.upper_total = up_score.min(63) as u8;
    record.got_bonus = got_bonus as u8;

    record
}

/// Simulate N games in parallel with recording, using an adaptive policy.
pub fn simulate_batch_adaptive_with_recording(
    ctx: &YatzyContext,
    tables: &[ThetaTable],
    policy: &dyn AdaptivePolicy,
    num_games: usize,
    seed: u64,
) -> Vec<GameRecord> {
    (0..num_games)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
            simulate_game_adaptive_with_recording(ctx, tables, policy, &mut rng)
        })
        .collect()
}

/// Simulate N games in parallel (no recording), using an adaptive policy.
/// Returns sorted scores vector.
pub fn simulate_batch_adaptive(
    ctx: &YatzyContext,
    tables: &[ThetaTable],
    policy: &dyn AdaptivePolicy,
    num_games: usize,
    seed: u64,
) -> Vec<i32> {
    let mut scores: Vec<i32> = (0..num_games)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
            simulate_game_adaptive(ctx, tables, policy, &mut rng)
        })
        .collect();
    scores.sort_unstable();
    scores
}

/// Default θ values used by each policy.
pub struct PolicyThetas {
    pub name: &'static str,
    pub thetas: &'static [f32],
}

pub const POLICY_CONFIGS: &[PolicyThetas] = &[
    PolicyThetas {
        name: "always-ev",
        thetas: &[0.0],
    },
    PolicyThetas {
        name: "bonus-adaptive",
        thetas: &[0.0, 0.08],
    },
    PolicyThetas {
        name: "phase-based",
        thetas: &[0.0, 0.03, 0.08],
    },
    PolicyThetas {
        name: "combined",
        thetas: &[0.0, 0.03, 0.05, 0.08],
    },
    PolicyThetas {
        name: "upper-deficit",
        thetas: &[0.0, 0.03, 0.05, 0.10],
    },
];

/// Look up the θ values needed for a given policy name.
pub fn policy_thetas(name: &str) -> Option<&'static [f32]> {
    POLICY_CONFIGS
        .iter()
        .find(|p| p.name == name)
        .map(|p| p.thetas)
}

/// Construct a policy from its name and the loaded tables.
/// Returns None if the name is unknown.
pub fn make_policy(name: &str, tables: &[ThetaTable]) -> Option<Box<dyn AdaptivePolicy>> {
    // Build a theta-to-index map
    let find_idx = |theta: f32| -> usize {
        tables
            .iter()
            .position(|t| (t.theta - theta).abs() < 1e-6)
            .unwrap_or_else(|| panic!("Table for θ={} not loaded", theta))
    };

    match name {
        "always-ev" => Some(Box::new(AlwaysEv {
            ev_idx: find_idx(0.0),
        })),
        "bonus-adaptive" => Some(Box::new(BonusAdaptive {
            ev_idx: find_idx(0.0),
            high_idx: find_idx(0.08),
        })),
        "phase-based" => Some(Box::new(PhaseBased {
            ev_idx: find_idx(0.0),
            mild_idx: find_idx(0.03),
            high_idx: find_idx(0.08),
        })),
        "combined" => Some(Box::new(Combined {
            ev_idx: find_idx(0.0),
            mild_idx: find_idx(0.03),
            moderate_idx: find_idx(0.05),
            high_idx: find_idx(0.08),
        })),
        "upper-deficit" => Some(Box::new(UpperDeficit {
            ev_idx: find_idx(0.0),
            mild_idx: find_idx(0.03),
            moderate_idx: find_idx(0.05),
            high_idx: find_idx(0.10),
        })),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_categories_left() {
        assert_eq!(categories_left(0), 15);
        assert_eq!(categories_left(0b111_1111_1111_1111), 0);
        assert_eq!(categories_left(0b1), 14);
    }

    #[test]
    fn test_upper_categories_left() {
        assert_eq!(upper_categories_left(0), 6);
        assert_eq!(upper_categories_left(0x3F), 0); // all 6 upper scored
        assert_eq!(upper_categories_left(0b1), 5); // only Ones scored
    }

    #[test]
    fn test_bonus_secured() {
        assert!(!bonus_secured(62));
        assert!(bonus_secured(63));
    }

    #[test]
    fn test_max_remaining_upper() {
        // No categories scored: 1*5 + 2*5 + 3*5 + 4*5 + 5*5 + 6*5 = 105
        assert_eq!(max_remaining_upper(0), 105);
        // All upper scored
        assert_eq!(max_remaining_upper(0x3F), 0);
        // Only Sixes (bit 5) unscored: 6*5 = 30
        assert_eq!(max_remaining_upper(0x1F), 30);
    }

    #[test]
    fn test_bonus_reachable() {
        // Fresh game: 0 + 105 >= 63
        assert!(bonus_reachable(0, 0));
        // Only Ones left (bit 0 unscored, rest scored): max remaining = 5
        assert!(!bonus_reachable(50, 0x3E)); // 50 + 5 < 63
        assert!(bonus_reachable(58, 0x3E)); // 58 + 5 >= 63
    }

    #[test]
    fn test_has_high_variance_left() {
        assert!(has_high_variance_left(0));
        // Score all high-variance: bits 10, 11, 12, 14
        let scored = (1 << 10) | (1 << 11) | (1 << 12) | (1 << 14);
        assert!(!has_high_variance_left(scored));
    }

    #[test]
    fn test_policy_thetas_lookup() {
        assert_eq!(policy_thetas("bonus-adaptive"), Some(&[0.0, 0.08][..]));
        assert_eq!(
            policy_thetas("combined"),
            Some(&[0.0, 0.03, 0.05, 0.08][..])
        );
        assert_eq!(
            policy_thetas("upper-deficit"),
            Some(&[0.0, 0.03, 0.05, 0.10][..])
        );
        assert_eq!(policy_thetas("nonexistent"), None);
    }
}
