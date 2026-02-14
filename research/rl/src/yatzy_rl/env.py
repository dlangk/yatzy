"""Yatzy environments for RL training.

Three environment variants:
- YatzyEnv: Base class with shared game mechanics
- ThetaSwitchEnv: Approach A — agent picks theta per turn, table handles decisions
- DirectActionEnv: Approaches B/C — agent makes each reroll/category decision
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dice import (
    NUM_KEEP_MULTISETS,
    KeepTable,
    build_all_dice_sets,
    build_keep_table,
    compute_dice_set_probabilities,
    find_dice_set_index,
)
from .features import compute_base_obs, compute_full_obs
from .scoring import (
    CATEGORY_COUNT,
    UPPER_BONUS,
    calculate_category_score,
    count_faces,
    is_category_scored,
    precompute_all_scores,
    state_index,
    update_upper_score,
)


@dataclass
class GameState:
    """Mutable game state for one Yatzy game."""
    upper_score: int = 0
    scored_categories: int = 0
    total_score: int = 0
    turn: int = 0


def _sort_dice(dice: np.ndarray) -> None:
    """Sort dice in-place (ascending)."""
    dice.sort()


def _roll_dice(rng: np.random.Generator) -> np.ndarray:
    """Roll 5 random dice, sorted."""
    dice = rng.integers(1, 7, size=5, dtype=np.int32)
    dice.sort()
    return dice


def _apply_reroll(dice: np.ndarray, mask: int, rng: np.random.Generator) -> np.ndarray:
    """Apply reroll mask: bits set indicate positions to reroll. Returns new sorted dice."""
    new_dice = dice.copy()
    for i in range(5):
        if mask & (1 << i):
            new_dice[i] = rng.integers(1, 7)
    new_dice.sort()
    return new_dice


class TableContext:
    """Precomputed lookup tables shared across environments."""

    def __init__(self) -> None:
        self.all_dice_sets, self.index_lookup = build_all_dice_sets()
        self.scores = precompute_all_scores(self.all_dice_sets)
        self.keep_table = build_keep_table(self.all_dice_sets, self.index_lookup)
        self.dice_probs = compute_dice_set_probabilities(self.all_dice_sets)

    def find_index(self, dice: np.ndarray) -> int:
        return find_dice_set_index(self.index_lookup, dice)


def choose_best_reroll_mask(
    ctx: TableContext,
    e_ds: np.ndarray,
    dice: np.ndarray,
) -> tuple[int, float]:
    """Find the best reroll mask for given dice using precomputed e_ds values.

    Uses keep-multiset dedup with vectorized sparse dot products.

    Returns (best_mask, best_ev). mask=0 means keep all dice.
    """
    ds_index = ctx.find_index(dice)
    kt = ctx.keep_table

    # Start with keep-all (mask=0)
    best_ev = float(e_ds[ds_index])
    best_mask = 0

    n_unique = kt.unique_count[ds_index]
    for j in range(n_unique):
        ki = kt.unique_keep_ids[ds_index, j]
        mask = kt.keep_to_mask[ds_index * 32 + j]

        start = kt.row_start[ki]
        end = kt.row_start[ki + 1]
        ev = float(np.dot(kt.vals[start:end], e_ds[kt.cols[start:end]]))

        if ev > best_ev:
            best_ev = ev
            best_mask = mask

    return best_mask, best_ev


def choose_best_reroll_mask_risk(
    ctx: TableContext,
    e_ds: np.ndarray,
    dice: np.ndarray,
    minimize: bool,
) -> tuple[int, float]:
    """Risk-sensitive version: uses LSE for stochastic nodes, min/max for decision."""
    ds_index = ctx.find_index(dice)
    kt = ctx.keep_table

    best_ev = float(e_ds[ds_index])
    best_mask = 0

    n_unique = kt.unique_count[ds_index]
    for j in range(n_unique):
        ki = kt.unique_keep_ids[ds_index, j]
        mask = kt.keep_to_mask[ds_index * 32 + j]

        start = kt.row_start[ki]
        end = kt.row_start[ki + 1]

        # LSE: log-sum-exp for stochastic node
        vals_slice = e_ds[kt.cols[start:end]]
        probs_slice = kt.vals[start:end]
        m = float(np.max(vals_slice))
        lse = m + np.log(np.sum(probs_slice * np.exp(vals_slice - m)))

        better = lse < best_ev if minimize else lse > best_ev
        if better:
            best_ev = lse
            best_mask = mask

    return best_mask, best_ev


def compute_group6(
    ctx: TableContext,
    sv: np.ndarray,
    up_score: int,
    scored: int,
    e_ds_0: np.ndarray,
) -> None:
    """Group 6: best category EV for each of 252 dice sets (vectorized).

    e_ds_0[ds] = max_{c not scored} [score(ds, c) + sv[successor_state]]
    """
    e_ds_0[:] = -np.inf

    # Lower categories: successor EV is independent of dice set (same upper score)
    for c in range(6, CATEGORY_COUNT):
        if not is_category_scored(scored, c):
            succ_ev = sv[state_index(up_score, scored | (1 << c))]
            vals = ctx.scores[:, c].astype(np.float32) + succ_ev  # (252,)
            np.maximum(e_ds_0, vals, out=e_ds_0)

    # Upper categories: successor depends on score (affects upper_score)
    for c in range(6):
        if not is_category_scored(scored, c):
            cat_scores = ctx.scores[:, c]  # (252,)
            new_scored = scored | (1 << c)
            # Vectorize update_upper_score: min(upper_score + score, 63)
            new_ups = np.minimum(up_score + cat_scores, 63)  # (252,)
            # Vectorize state_index lookup
            succ_indices = new_scored * 64 + new_ups  # (252,)
            succ_evs = sv[succ_indices]  # (252,)
            vals = cat_scores.astype(np.float32) + succ_evs
            np.maximum(e_ds_0, vals, out=e_ds_0)


def compute_group6_risk(
    ctx: TableContext,
    sv: np.ndarray,
    up_score: int,
    scored: int,
    theta: float,
    minimize: bool,
    e_ds_0: np.ndarray,
) -> None:
    """Group 6 risk-sensitive: val = theta * score + sv[successor]."""
    lower_succ = np.zeros(CATEGORY_COUNT, dtype=np.float32)
    for c in range(6, CATEGORY_COUNT):
        if not is_category_scored(scored, c):
            lower_succ[c] = sv[state_index(up_score, scored | (1 << c))]

    init_val = np.inf if minimize else -np.inf

    for ds_i in range(252):
        best_val = init_val

        for c in range(6):
            if not is_category_scored(scored, c):
                scr = int(ctx.scores[ds_i, c])
                new_up = update_upper_score(up_score, c, scr)
                new_scored = scored | (1 << c)
                val = theta * scr + sv[state_index(new_up, new_scored)]
                better = val < best_val if minimize else val > best_val
                if better:
                    best_val = val

        for c in range(6, CATEGORY_COUNT):
            if not is_category_scored(scored, c):
                scr = int(ctx.scores[ds_i, c])
                val = theta * scr + lower_succ[c]
                better = val < best_val if minimize else val > best_val
                if better:
                    best_val = val

        e_ds_0[ds_i] = best_val


def compute_max_ev_for_n_rerolls(
    ctx: TableContext,
    e_ds_in: np.ndarray,
    e_ds_out: np.ndarray,
) -> None:
    """Propagate through one reroll level: e_ds_out[ds] = max_mask E[e_ds_in | keep(mask)].

    This is Groups 5/3 from the widget solver. Uses vectorized sparse dot products.
    """
    kt = ctx.keep_table

    # Precompute all keep EVs at once: for each keep ki, dot(probs, e_ds_in[targets])
    # This avoids recomputing the same keep for multiple dice sets.
    keep_evs = np.full(NUM_KEEP_MULTISETS, -np.inf, dtype=np.float64)
    for ki in range(NUM_KEEP_MULTISETS):
        start = kt.row_start[ki]
        end = kt.row_start[ki + 1]
        if start < end:
            keep_evs[ki] = np.dot(kt.vals[start:end], e_ds_in[kt.cols[start:end]])

    for ds_i in range(252):
        best_ev = float(e_ds_in[ds_i])  # keep-all baseline

        n_unique = kt.unique_count[ds_i]
        for j in range(n_unique):
            ki = kt.unique_keep_ids[ds_i, j]
            ev = keep_evs[ki]
            if ev > best_ev:
                best_ev = ev

        e_ds_out[ds_i] = best_ev


def compute_opt_lse_for_n_rerolls(
    ctx: TableContext,
    e_ds_in: np.ndarray,
    e_ds_out: np.ndarray,
    minimize: bool,
) -> None:
    """Risk-sensitive reroll propagation: LSE for stochastic, min/max for decision."""
    kt = ctx.keep_table

    # Precompute LSE for all keeps
    keep_lse = np.full(NUM_KEEP_MULTISETS, np.inf if minimize else -np.inf, dtype=np.float64)
    for ki in range(NUM_KEEP_MULTISETS):
        start = kt.row_start[ki]
        end = kt.row_start[ki + 1]
        if start < end:
            vals_slice = e_ds_in[kt.cols[start:end]]
            probs_slice = kt.vals[start:end]
            m = float(np.max(vals_slice))
            keep_lse[ki] = m + np.log(np.sum(probs_slice * np.exp(vals_slice - m)))

    for ds_i in range(252):
        best_ev = float(e_ds_in[ds_i])

        n_unique = kt.unique_count[ds_i]
        for j in range(n_unique):
            ki = kt.unique_keep_ids[ds_i, j]
            ev = keep_lse[ki]
            better = ev < best_ev if minimize else ev > best_ev
            if better:
                best_ev = ev

        e_ds_out[ds_i] = best_ev


def find_best_category(
    ctx: TableContext,
    sv: np.ndarray,
    up_score: int,
    scored: int,
    ds_index: int,
) -> tuple[int, int]:
    """Find best category for given dice set. Returns (category, score)."""
    best_val = -np.inf
    best_cat = 0
    best_score = 0

    for c in range(6):
        if not is_category_scored(scored, c):
            scr = int(ctx.scores[ds_index, c])
            new_up = update_upper_score(up_score, c, scr)
            new_scored = scored | (1 << c)
            val = scr + sv[state_index(new_up, new_scored)]
            if val > best_val:
                best_val = val
                best_cat = c
                best_score = scr

    for c in range(6, CATEGORY_COUNT):
        if not is_category_scored(scored, c):
            scr = int(ctx.scores[ds_index, c])
            new_scored = scored | (1 << c)
            val = scr + sv[state_index(up_score, new_scored)]
            if val > best_val:
                best_val = val
                best_cat = c
                best_score = scr

    return best_cat, best_score


def find_best_category_final(
    ctx: TableContext,
    up_score: int,
    scored: int,
    ds_index: int,
) -> tuple[int, int]:
    """Find best category for the last turn (no successor state)."""
    best_val = -1_000_000
    best_cat = 0
    best_score = 0

    for c in range(CATEGORY_COUNT):
        if not is_category_scored(scored, c):
            scr = int(ctx.scores[ds_index, c])
            bonus = 0
            if c < 6:
                new_up = update_upper_score(up_score, c, scr)
                if new_up >= 63 and up_score < 63:
                    bonus = 50
            val = scr + bonus
            if val > best_val:
                best_val = val
                best_cat = c
                best_score = scr

    return best_cat, best_score


def find_best_category_risk(
    ctx: TableContext,
    sv: np.ndarray,
    up_score: int,
    scored: int,
    ds_index: int,
    theta: float,
    minimize: bool,
) -> tuple[int, int]:
    """Risk-sensitive category selection."""
    best_val = np.inf if minimize else -np.inf
    best_cat = 0
    best_score = 0

    for c in range(6):
        if not is_category_scored(scored, c):
            scr = int(ctx.scores[ds_index, c])
            new_up = update_upper_score(up_score, c, scr)
            new_scored = scored | (1 << c)
            val = theta * scr + sv[state_index(new_up, new_scored)]
            better = val < best_val if minimize else val > best_val
            if better:
                best_val = val
                best_cat = c
                best_score = scr

    for c in range(6, CATEGORY_COUNT):
        if not is_category_scored(scored, c):
            scr = int(ctx.scores[ds_index, c])
            new_scored = scored | (1 << c)
            val = theta * scr + sv[state_index(up_score, new_scored)]
            better = val < best_val if minimize else val > best_val
            if better:
                best_val = val
                best_cat = c
                best_score = scr

    return best_cat, best_score


def find_best_category_final_risk(
    ctx: TableContext,
    up_score: int,
    scored: int,
    ds_index: int,
    theta: float,
    minimize: bool,
) -> tuple[int, int]:
    """Risk-sensitive category for last turn."""
    best_val = np.inf if minimize else -np.inf
    best_cat = 0
    best_score = 0

    for c in range(CATEGORY_COUNT):
        if not is_category_scored(scored, c):
            scr = int(ctx.scores[ds_index, c])
            bonus = 0
            if c < 6:
                new_up = update_upper_score(up_score, c, scr)
                if new_up >= 63 and up_score < 63:
                    bonus = 50
            val = theta * (scr + bonus)
            better = val < best_val if minimize else val > best_val
            if better:
                best_val = val
                best_cat = c
                best_score = scr

    return best_cat, best_score


def simulate_turn_with_table(
    ctx: TableContext,
    sv: np.ndarray,
    up_score: int,
    scored: int,
    turn: int,
    rng: np.random.Generator,
    theta: float = 0.0,
) -> tuple[int, int, int]:
    """Simulate one full turn using precomputed table.

    Returns (category, score, new_upper_score).
    """
    is_last = turn == CATEGORY_COUNT - 1
    use_risk = theta != 0.0
    minimize = theta < 0.0

    dice = _roll_dice(rng)

    e_ds_0 = np.zeros(252, dtype=np.float32)
    e_ds_1 = np.zeros(252, dtype=np.float32)

    if use_risk:
        compute_group6_risk(ctx, sv, up_score, scored, theta, minimize, e_ds_0)
        compute_opt_lse_for_n_rerolls(ctx, e_ds_0, e_ds_1, minimize)
    else:
        compute_group6(ctx, sv, up_score, scored, e_ds_0)
        compute_max_ev_for_n_rerolls(ctx, e_ds_0, e_ds_1)

    # First reroll
    if use_risk:
        mask1, _ = choose_best_reroll_mask_risk(ctx, e_ds_1, dice, minimize)
    else:
        mask1, _ = choose_best_reroll_mask(ctx, e_ds_1, dice)
    if mask1 != 0:
        dice = _apply_reroll(dice, mask1, rng)

    # Second reroll
    if use_risk:
        mask2, _ = choose_best_reroll_mask_risk(ctx, e_ds_0, dice, minimize)
    else:
        mask2, _ = choose_best_reroll_mask(ctx, e_ds_0, dice)
    if mask2 != 0:
        dice = _apply_reroll(dice, mask2, rng)

    # Choose category
    ds_index = ctx.find_index(dice)
    if use_risk:
        if is_last:
            cat, scr = find_best_category_final_risk(ctx, up_score, scored, ds_index, theta, minimize)
        else:
            cat, scr = find_best_category_risk(ctx, sv, up_score, scored, ds_index, theta, minimize)
    else:
        if is_last:
            cat, scr = find_best_category_final(ctx, up_score, scored, ds_index)
        else:
            cat, scr = find_best_category(ctx, sv, up_score, scored, ds_index)

    new_up = update_upper_score(up_score, cat, scr)
    return cat, scr, new_up


class ThetaSwitchEnv:
    """Approach A environment: agent picks which theta table to use per turn.

    Action space: discrete K (number of theta tables).
    Observation space: 10-dim base observation.
    Episode: 15 steps (one per turn).
    """

    def __init__(
        self,
        theta_tables: dict[float, np.ndarray],
        ctx: TableContext,
        seed: int = 42,
    ):
        self.theta_values = sorted(theta_tables.keys())
        self.tables = theta_tables
        self.ctx = ctx
        self.rng = np.random.default_rng(seed)
        self.n_actions = len(self.theta_values)

        # Use theta=0 table for observation features
        self.base_sv = self.tables[0.0]
        self.state = GameState()

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset to new game. Returns initial observation."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = GameState()
        return compute_base_obs(
            self.state.upper_score,
            self.state.scored_categories,
            self.state.total_score,
            self.base_sv,
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Execute one turn using the selected theta table.

        Args:
            action: index into theta_values list

        Returns:
            (observation, reward, done, info)
        """
        theta = self.theta_values[action]
        sv = self.tables[theta]

        cat, scr, new_up = simulate_turn_with_table(
            self.ctx,
            sv,
            self.state.upper_score,
            self.state.scored_categories,
            self.state.turn,
            self.rng,
            theta=theta,
        )

        self.state.upper_score = new_up
        self.state.scored_categories |= 1 << cat
        self.state.total_score += scr
        self.state.turn += 1

        done = self.state.turn >= CATEGORY_COUNT
        reward = 0.0

        if done:
            if self.state.upper_score >= 63:
                self.state.total_score += int(UPPER_BONUS)
            reward = float(self.state.total_score)

        obs = compute_base_obs(
            self.state.upper_score,
            self.state.scored_categories,
            self.state.total_score,
            self.base_sv,
        )

        info = {
            "turn": self.state.turn,
            "category": cat,
            "score": scr,
            "total_score": self.state.total_score,
            "theta": theta,
        }

        return obs, reward, done, info


class DirectActionEnv:
    """Approaches B/C environment: agent makes each reroll/category decision.

    Action space:
        - During reroll phase: 32 masks (0=keep all, 1-31=reroll patterns)
        - During scoring phase: 15 categories (only unscored valid)

    Observation space: 18-dim (base + dice features).
    Episode: ~45 steps (3 decisions per turn x 15 turns).
    """

    def __init__(
        self,
        state_values: np.ndarray,
        ctx: TableContext,
        seed: int = 42,
    ):
        self.sv = state_values
        self.ctx = ctx
        self.rng = np.random.default_rng(seed)

        self.state = GameState()
        self.dice = np.zeros(5, dtype=np.int32)
        self.rerolls_remaining = 2
        self.phase = "roll"  # "roll", "reroll", "score"

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = GameState()
        self._new_turn()
        return self._get_obs()

    def _new_turn(self) -> None:
        """Start a new turn: roll dice."""
        self.dice = _roll_dice(self.rng)
        self.rerolls_remaining = 2
        self.phase = "reroll"

    def _get_obs(self) -> np.ndarray:
        fc = count_faces(self.dice)
        ds_index = self.ctx.find_index(self.dice)
        return compute_full_obs(
            self.state.upper_score,
            self.state.scored_categories,
            self.state.total_score,
            self.sv,
            fc,
            self.rerolls_remaining,
            ds_index,
        )

    def get_valid_actions(self) -> np.ndarray:
        """Return mask of valid actions."""
        if self.phase == "reroll":
            # All 32 masks valid (0 = keep all)
            return np.ones(32, dtype=bool)
        else:
            # Only unscored categories
            valid = np.zeros(CATEGORY_COUNT, dtype=bool)
            for c in range(CATEGORY_COUNT):
                if not is_category_scored(self.state.scored_categories, c):
                    valid[c] = True
            return valid

    def get_table_action(self) -> int:
        """Get the action that the theta=0 table would take (for imitation learning)."""
        if self.phase == "reroll":
            e_ds = np.zeros(252, dtype=np.float32)
            compute_group6(self.ctx, self.sv, self.state.upper_score,
                          self.state.scored_categories, e_ds)
            if self.rerolls_remaining == 2:
                e_ds_next = np.zeros(252, dtype=np.float32)
                compute_max_ev_for_n_rerolls(self.ctx, e_ds, e_ds_next)
                mask, _ = choose_best_reroll_mask(self.ctx, e_ds_next, self.dice)
            else:
                mask, _ = choose_best_reroll_mask(self.ctx, e_ds, self.dice)
            return mask
        else:
            ds_index = self.ctx.find_index(self.dice)
            is_last = self.state.turn == CATEGORY_COUNT - 1
            if is_last:
                cat, _ = find_best_category_final(
                    self.ctx, self.state.upper_score, self.state.scored_categories, ds_index
                )
            else:
                cat, _ = find_best_category(
                    self.ctx, self.sv, self.state.upper_score,
                    self.state.scored_categories, ds_index
                )
            return cat

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Execute one decision step.

        For reroll: action is mask (0-31). After processing, transitions to next
        reroll or score phase.
        For score: action is category (0-14). Transitions to next turn.
        """
        info: dict[str, Any] = {"phase": self.phase, "turn": self.state.turn}

        if self.phase == "reroll":
            mask = action
            if mask != 0:
                self.dice = _apply_reroll(self.dice, mask, self.rng)
            self.rerolls_remaining -= 1

            if self.rerolls_remaining > 0:
                self.phase = "reroll"
            else:
                self.phase = "score"

            info["mask"] = mask
            return self._get_obs(), 0.0, False, info

        else:  # score phase
            cat = action
            ds_index = self.ctx.find_index(self.dice)
            scr = int(self.ctx.scores[ds_index, cat])

            new_up = update_upper_score(self.state.upper_score, cat, scr)
            self.state.upper_score = new_up
            self.state.scored_categories |= 1 << cat
            self.state.total_score += scr
            self.state.turn += 1

            info["category"] = cat
            info["score"] = scr

            done = self.state.turn >= CATEGORY_COUNT
            reward = 0.0

            if done:
                if self.state.upper_score >= 63:
                    self.state.total_score += int(UPPER_BONUS)
                reward = float(self.state.total_score)
                info["total_score"] = self.state.total_score
            else:
                self._new_turn()

            obs = self._get_obs() if not done else np.zeros(18, dtype=np.float32)
            return obs, reward, done, info
