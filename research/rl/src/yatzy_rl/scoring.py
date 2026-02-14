"""Scandinavian Yatzy scoring rules: 15 categories.

Port of backend/src/game_mechanics.rs. All functions operate on face_counts[7]
(index 0 unused, indices 1-6 are counts) or raw dice arrays.
"""

from __future__ import annotations

import numpy as np

# Category indices (same as Rust constants.rs)
CATEGORY_ONES = 0
CATEGORY_TWOS = 1
CATEGORY_THREES = 2
CATEGORY_FOURS = 3
CATEGORY_FIVES = 4
CATEGORY_SIXES = 5
CATEGORY_ONE_PAIR = 6
CATEGORY_TWO_PAIRS = 7
CATEGORY_THREE_OF_A_KIND = 8
CATEGORY_FOUR_OF_A_KIND = 9
CATEGORY_SMALL_STRAIGHT = 10
CATEGORY_LARGE_STRAIGHT = 11
CATEGORY_FULL_HOUSE = 12
CATEGORY_CHANCE = 13
CATEGORY_YATZY = 14

CATEGORY_COUNT = 15
NUM_STATES = 64 * (1 << 15)  # 2,097,152
UPPER_BONUS = 50
UPPER_SCORE_CAP = 63

CATEGORY_NAMES = [
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "One Pair", "Two Pairs", "Three of a Kind", "Four of a Kind",
    "Small Straight", "Large Straight", "Full House", "Chance", "Yatzy",
]


def count_faces(dice: np.ndarray) -> np.ndarray:
    """Count occurrences of each face value (1-6). Returns array of length 7 (index 0 unused)."""
    fc = np.zeros(7, dtype=np.int32)
    for d in dice:
        fc[d] += 1
    return fc


def calculate_category_score(dice: np.ndarray, category: int) -> int:
    """Compute s(S, r, c): score for placing 5-dice roll in the given category."""
    fc = count_faces(dice)
    sum_all = int(dice.sum())

    if category <= CATEGORY_SIXES:
        face = category + 1
        return int(fc[face] * face)

    if category == CATEGORY_ONE_PAIR:
        for f in range(6, 0, -1):
            if fc[f] >= 2:
                return 2 * f
        return 0

    if category == CATEGORY_TWO_PAIRS:
        pairs = []
        for f in range(6, 0, -1):
            if fc[f] >= 2:
                pairs.append(f)
                if len(pairs) == 2:
                    return 2 * pairs[0] + 2 * pairs[1]
        return 0

    if category == CATEGORY_THREE_OF_A_KIND:
        for f in range(6, 0, -1):
            if fc[f] >= 3:
                return 3 * f
        return 0

    if category == CATEGORY_FOUR_OF_A_KIND:
        for f in range(6, 0, -1):
            if fc[f] >= 4:
                return 4 * f
        return 0

    if category == CATEGORY_SMALL_STRAIGHT:
        if fc[1] == 1 and fc[2] == 1 and fc[3] == 1 and fc[4] == 1 and fc[5] == 1:
            return 15
        return 0

    if category == CATEGORY_LARGE_STRAIGHT:
        if fc[2] == 1 and fc[3] == 1 and fc[4] == 1 and fc[5] == 1 and fc[6] == 1:
            return 20
        return 0

    if category == CATEGORY_FULL_HOUSE:
        three_face = 0
        pair_face = 0
        for f in range(1, 7):
            if fc[f] == 3:
                three_face = f
            elif fc[f] == 2:
                pair_face = f
        if three_face != 0 and pair_face != 0:
            return sum_all
        return 0

    if category == CATEGORY_CHANCE:
        return sum_all

    if category == CATEGORY_YATZY:
        for f in range(1, 7):
            if fc[f] == 5:
                return 50
        return 0

    return 0


def update_upper_score(upper_score: int, category: int, score: int) -> int:
    """Compute successor upper score: min(upper_score + score, 63) for upper categories."""
    if category < 6:
        return min(upper_score + score, 63)
    return upper_score


def state_index(upper_score: int, scored_categories: int) -> int:
    """Map state S = (upper_score, scored_categories) to flat array index."""
    return scored_categories * 64 + upper_score


def is_category_scored(scored: int, cat: int) -> bool:
    """Test whether category cat has been scored (bit cat is set)."""
    return (scored & (1 << cat)) != 0


def precompute_all_scores(all_dice_sets: np.ndarray) -> np.ndarray:
    """Precompute scores[252][15] for all dice sets and categories."""
    scores = np.zeros((252, CATEGORY_COUNT), dtype=np.int32)
    for i in range(252):
        for c in range(CATEGORY_COUNT):
            scores[i, c] = calculate_category_score(all_dice_sets[i], c)
    return scores
