"""Feature engineering for surrogate models.

Computes derived features from the base feature set exported by the Rust solver.
All features are derivable from existing features (no Rust changes needed):
un-normalize face counts, apply Yatzy scoring rules, compute pattern indicators.
"""
from __future__ import annotations

import numpy as np

from .surrogate_eval import CATEGORY_COUNT, calculate_score, is_scored


# ── Feature index constants (must match build_features in surrogate_eval.py) ──

IDX_TURN = 0
IDX_UPPER_SCORE = 1
IDX_UPPER_CATS_LEFT = 2
IDX_BONUS_SECURED = 3
IDX_BONUS_DEFICIT = 4
IDX_FACE1 = 5
IDX_FACE6 = 10
IDX_DICE_SUM = 11
IDX_MAX_FACE_COUNT = 12
IDX_NUM_DISTINCT = 13
IDX_CAT0_AVAIL = 14
IDX_CAT14_AVAIL = 28
IDX_REROLLS_REM = 29

# Feature group definitions for ablation
FEATURE_GROUPS = {
    "turn": [IDX_TURN],
    "upper_state": [IDX_UPPER_SCORE, IDX_UPPER_CATS_LEFT, IDX_BONUS_SECURED, IDX_BONUS_DEFICIT],
    "dice_counts": list(range(IDX_FACE1, IDX_FACE6 + 1)),
    "dice_summary": [IDX_DICE_SUM, IDX_MAX_FACE_COUNT, IDX_NUM_DISTINCT],
    "category_avail": list(range(IDX_CAT0_AVAIL, IDX_CAT14_AVAIL + 1)),
    "rerolls_rem": [IDX_REROLLS_REM],
}


def _reconstruct_dice(features: np.ndarray) -> np.ndarray:
    """Reconstruct sorted dice array from normalized face counts.

    features[IDX_FACE1..IDX_FACE6] are fc[face] / 5.0.
    Returns array of 5 dice values (sorted), one per sample.
    """
    n = len(features)
    dice = np.zeros((n, 5), dtype=np.int32)
    for i in range(n):
        idx = 0
        for face in range(1, 7):
            count = int(round(features[i, IDX_FACE1 + face - 1] * 5))
            for _ in range(count):
                if idx < 5:
                    dice[i, idx] = face
                    idx += 1
    return dice


def _reconstruct_scored(features: np.ndarray) -> np.ndarray:
    """Reconstruct scored bitmask from category availability features."""
    n = len(features)
    scored = np.zeros(n, dtype=np.int32)
    for i in range(n):
        for c in range(CATEGORY_COUNT):
            if features[i, IDX_CAT0_AVAIL + c] < 0.5:  # 0.0 means scored
                scored[i] |= 1 << c
    return scored


def compute_category_scores(features: np.ndarray) -> np.ndarray:
    """Compute actual score for each available category given current dice.

    Returns (n_samples, 15) array of normalized scores.
    Scored categories get 0.0.
    """
    n = len(features)
    dice = _reconstruct_dice(features)
    scored = _reconstruct_scored(features)
    result = np.zeros((n, CATEGORY_COUNT), dtype=np.float32)

    # Normalization constants per category (approximate max scores)
    max_scores = np.array([
        5, 10, 15, 20, 25, 30,  # upper: face * 5
        12, 22, 18, 24,  # pair, 2pair, 3kind, 4kind
        15, 20, 30, 30, 50,  # sm_str, lg_str, full_house, chance, yatzy
    ], dtype=np.float32)

    for i in range(n):
        d = dice[i]
        for c in range(CATEGORY_COUNT):
            if not is_scored(scored[i], c):
                s = calculate_score(d, c)
                result[i, c] = s / max_scores[c]
    return result


def compute_pattern_indicators(features: np.ndarray) -> np.ndarray:
    """Compute 8 boolean pattern indicators from dice.

    Returns (n_samples, 8) array: has_pair, has_two_pair, has_three_kind,
    has_four_kind, has_full_house, has_small_straight, has_large_straight, has_yatzy.
    """
    n = len(features)
    dice = _reconstruct_dice(features)
    result = np.zeros((n, 8), dtype=np.float32)

    for i in range(n):
        d = dice[i]
        result[i, 0] = 1.0 if calculate_score(d, 6) > 0 else 0.0   # pair
        result[i, 1] = 1.0 if calculate_score(d, 7) > 0 else 0.0   # two pair
        result[i, 2] = 1.0 if calculate_score(d, 8) > 0 else 0.0   # three kind
        result[i, 3] = 1.0 if calculate_score(d, 9) > 0 else 0.0   # four kind
        result[i, 4] = 1.0 if calculate_score(d, 12) > 0 else 0.0  # full house
        result[i, 5] = 1.0 if calculate_score(d, 10) > 0 else 0.0  # small straight
        result[i, 6] = 1.0 if calculate_score(d, 11) > 0 else 0.0  # large straight
        result[i, 7] = 1.0 if calculate_score(d, 14) > 0 else 0.0  # yatzy

    return result


def compute_best_available_score(features: np.ndarray) -> np.ndarray:
    """Max score achievable across all available categories. Returns (n, 1)."""
    n = len(features)
    dice = _reconstruct_dice(features)
    scored = _reconstruct_scored(features)
    result = np.zeros((n, 1), dtype=np.float32)

    for i in range(n):
        best = 0
        for c in range(CATEGORY_COUNT):
            if not is_scored(scored[i], c):
                s = calculate_score(dice[i], c)
                if s > best:
                    best = s
        result[i, 0] = best / 50.0  # normalize by max possible (Yatzy=50)
    return result


def compute_upper_opportunity(features: np.ndarray) -> np.ndarray:
    """Sum of face*count for remaining upper categories. Returns (n, 1)."""
    n = len(features)
    result = np.zeros((n, 1), dtype=np.float32)

    for i in range(n):
        total = 0
        for c in range(6):  # upper categories only
            if features[i, IDX_CAT0_AVAIL + c] > 0.5:  # available
                face = c + 1
                count = int(round(features[i, IDX_FACE1 + c] * 5))
                total += face * count
        result[i, 0] = total / 30.0  # normalize by max dice sum
    return result


def compute_n_categories_remaining(features: np.ndarray) -> np.ndarray:
    """Count of available categories. Returns (n, 1)."""
    n = len(features)
    result = np.zeros((n, 1), dtype=np.float32)
    for i in range(n):
        count = 0
        for c in range(CATEGORY_COUNT):
            if features[i, IDX_CAT0_AVAIL + c] > 0.5:
                count += 1
        result[i, 0] = count / 15.0
    return result


def augment_features(
    features: np.ndarray,
    *,
    category_scores: bool = True,
    pattern_indicators: bool = True,
    best_available: bool = True,
    upper_opportunity: bool = True,
    n_remaining: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Augment feature matrix with engineered features.

    Returns (augmented_features, new_feature_names).
    """
    parts = [features]
    names: list[str] = []

    if category_scores:
        cs = compute_category_scores(features)
        parts.append(cs)
        names.extend([f"cat{i}_score" for i in range(CATEGORY_COUNT)])

    if pattern_indicators:
        pi = compute_pattern_indicators(features)
        parts.append(pi)
        names.extend([
            "has_pair", "has_two_pair", "has_three_kind", "has_four_kind",
            "has_full_house", "has_small_straight", "has_large_straight", "has_yatzy",
        ])

    if best_available:
        ba = compute_best_available_score(features)
        parts.append(ba)
        names.append("best_available_score")

    if upper_opportunity:
        uo = compute_upper_opportunity(features)
        parts.append(uo)
        names.append("upper_opportunity")

    if n_remaining:
        nr = compute_n_categories_remaining(features)
        parts.append(nr)
        names.append("n_categories_remaining")

    augmented = np.hstack(parts)
    return augmented, names
