"""State -> observation tensor conversion.

Converts game state into normalized feature vectors for RL agents.
"""

from __future__ import annotations

import numpy as np

from .scoring import (
    CATEGORY_COUNT,
    UPPER_SCORE_CAP,
    is_category_scored,
    state_index,
)


def compute_base_obs(
    upper_score: int,
    scored_categories: int,
    total_score: int,
    state_values: np.ndarray,
) -> np.ndarray:
    """Compute base observation vector (10 features) shared by all approaches.

    Features:
        0: upper_score / 63
        1: categories_left / 15
        2: upper_categories_left / 6
        3: bonus_secured (0 or 1)
        4: bonus_reachable (heuristic)
        5: bonus_health (deficit / max_remaining, clamped)
        6: has_high_variance_left (0 or 1)
        7: total_score / 374
        8: ev_remaining / 200
        9: score_z = (total - expected) / sigma
    """
    obs = np.zeros(10, dtype=np.float32)

    # Basic state
    cats_left = CATEGORY_COUNT - bin(scored_categories).count("1")
    upper_cats_left = sum(
        1 for c in range(6) if not is_category_scored(scored_categories, c)
    )

    obs[0] = upper_score / 63.0
    obs[1] = cats_left / 15.0
    obs[2] = upper_cats_left / 6.0
    obs[3] = 1.0 if upper_score >= 63 else 0.0

    # Bonus reachable heuristic: need 63 points from remaining upper categories
    # Max possible from remaining upper: sum of 5*face for unscored upper cats
    max_remaining_upper = sum(
        5 * (c + 1) for c in range(6) if not is_category_scored(scored_categories, c)
    )
    deficit = max(0, 63 - upper_score)
    obs[4] = 1.0 if max_remaining_upper >= deficit else 0.0

    # Bonus health: how close to securing bonus (0=impossible, 1=secured)
    if max_remaining_upper > 0:
        expected_upper = max_remaining_upper * 0.6  # rough heuristic
        obs[5] = min(1.0, max(0.0, 1.0 - deficit / max(expected_upper, 1.0)))
    else:
        obs[5] = 1.0 if upper_score >= 63 else 0.0

    # High-variance categories left (straights, yatzy, full house)
    high_var_cats = [10, 11, 12, 14]  # small straight, large straight, full house, yatzy
    obs[6] = 1.0 if any(
        not is_category_scored(scored_categories, c) for c in high_var_cats
    ) else 0.0

    # Score features
    obs[7] = total_score / 374.0

    # EV remaining from current state
    si = state_index(upper_score, scored_categories)
    ev_at_state = float(state_values[si])
    obs[8] = ev_at_state / 200.0

    # Score z-score: how far ahead/behind expected trajectory
    # Expected total at this point ≈ 248 * (15 - cats_left) / 15
    turns_played = 15 - cats_left
    if turns_played > 0:
        expected_score = 248.0 * turns_played / 15.0
        sigma = 40.0 * (turns_played / 15.0) ** 0.5  # rough estimate
        obs[9] = (total_score - expected_score) / max(sigma, 1.0)
    else:
        obs[9] = 0.0

    return obs


def compute_dice_obs(
    face_counts: np.ndarray,
    rerolls_remaining: int,
    dice_set_index: int,
) -> np.ndarray:
    """Compute dice observation vector (8 features) for Approaches B/C.

    Features:
        0-5: face_counts[1..6] / 5.0
        6: rerolls_remaining / 2.0
        7: dice_set_index / 252.0
    """
    obs = np.zeros(8, dtype=np.float32)
    obs[0:6] = face_counts[1:7] / 5.0
    obs[6] = rerolls_remaining / 2.0
    obs[7] = dice_set_index / 252.0
    return obs


def compute_full_obs(
    upper_score: int,
    scored_categories: int,
    total_score: int,
    state_values: np.ndarray,
    face_counts: np.ndarray,
    rerolls_remaining: int,
    dice_set_index: int,
) -> np.ndarray:
    """Compute full observation (18 features) = base + dice."""
    base = compute_base_obs(upper_score, scored_categories, total_score, state_values)
    dice = compute_dice_obs(face_counts, rerolls_remaining, dice_set_index)
    return np.concatenate([base, dice])
