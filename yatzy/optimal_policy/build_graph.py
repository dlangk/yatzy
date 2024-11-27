import os
import time
from itertools import combinations
from typing import List

import numpy as np

from yatzy.mechanics.categories import Category
from yatzy.mechanics.yatzy_scorer import YatzyScorer
from yatzy.mechanics.yatzy_state import YatzyState
from yatzy.probabilities.probability_calculator import ProbabilityCalculator

DATA_DIR = os.path.join(os.path.dirname(__file__), "../probabilities/data")


def calculate_category_scores(
        dice_set: np.ndarray,
        scored_categories: int,
        upper_score: int,
        combinations: dict
) -> np.ndarray:
    """
    Calculates the additional score for each Yatzy category based on the current dice_set.

    Parameters:
    - dice_set (np.ndarray): The rolled dice set (length 5).
    - scored_categories (int): A binary representation of which categories have been scored.
    - upper_score (int): The current score for the upper section.
    - combinations (dict): The category metadata for validation and scoring.

    Returns:
    - np.ndarray: An array of length 13, where each entry represents the score for a category (0 if already scored).
    """
    if len(dice_set) != 5:
        raise ValueError("dice_set must contain exactly 5 dice.")

    # Initialize scores for all 13 categories
    scores = np.zeros(len(combinations), dtype=int)

    for idx, (category, meta) in enumerate(combinations.items()):
        if scored_categories & (1 << idx):  # Skip already scored categories
            continue

        # Extract validator, scorer, and required parameters
        validator = meta["validator"]
        scorer = meta["scorer"]
        required_die_value = meta.get("required_die_value")
        required_die_count = meta.get("required_die_count")
        is_upper_section = meta["upper_section"]

        # Validate and score
        if validator(dice_set, die_value=required_die_value, die_count=required_die_count):
            scores[idx] = scorer(
                dice_set, die_value=required_die_value, die_count=required_die_count
            )

    # Add the bonus for upper section if applicable
    if upper_score + scores[:6].sum() >= 63:
        scores[:6] += 35  # Apply bonus points to upper section categories

    return scores


if __name__ == "__main__":
    calculator = ProbabilityCalculator()
    scored_categories = (1 << len(Category)) - 1 - (1 << Category.ONES.value)  # 32767 - 16384 = 16383
    state = YatzyState(upper_score=0, scored_categories=16383)

    if state.is_game_over():
        bonus = state.get_upper_bonus()
        print(bonus)
    else:

        rewards = state.get_best_scores_array(calculator.unique_dice_combinations)

        # Corrected nested loop
        for ix, dice_set_1 in enumerate(calculator.unique_dice_combinations):
            prob_1 = calculator.get_dice_set_probability(dice_set_1)
            best_reroll_vector_1, max_ev_1 = calculator.get_best_reroll_vector(dice_set_1, rewards)
            p_dict_1 = calculator.get_dice_set_probabilities(dice_set_1, best_reroll_vector_1)
            print(f"{ix}:-> ds={dice_set_1} p={prob_1} brv={best_reroll_vector_1} rv={max_ev_1}")

            for dice_set_2, prob_2 in p_dict_1.items():
                best_reroll_vector_2, max_ev_2 = calculator.get_best_reroll_vector(dice_set_2, rewards)
                print(f"{ix}:->-> ds={dice_set_2} p={prob_2} brv={best_reroll_vector_2} ev={max_ev_2}")

                # Corrected to use dice_set_2 and best_reroll_vector_2
                p_dict_2 = calculator.get_dice_set_probabilities(dice_set_2, best_reroll_vector_2)

                for dice_set_3, prob_3 in p_dict_2.items():
                    combination_key = tuple(sorted(map(int, dice_set_3)))
                    index = calculator.combination_to_index.get(combination_key, None)
                    if index is not None:
                        reward = rewards[index]
                        print(
                            f"{ix}:->->-> ds={dice_set_3} p={prob_3} brv={best_reroll_vector_2} reward={reward}")
                    else:
                        print(f"Combination {combination_key} not found in combination_to_index.")