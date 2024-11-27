import os
import time
from collections import Counter
from itertools import combinations

import numpy as np
from math import factorial

from yatzy.mechanics import const
from yatzy.probabilities.probability_calculator import ProbabilityCalculator

DATA_DIR = os.path.join(os.path.dirname(__file__), "../probabilities/data")


def calculate_rewards(dice_combinations, scored_categories, upper_score):
    """
    Calculate the reward for each `dice_set_3` based on the scoring logic provided.
    Args:
        dice_combinations: List of all unique `dice_set_3` configurations.
        scored_categories: Bitmask of already scored categories.
        upper_score: Current score in the upper section.

    Returns:
        rewards: NumPy array of rewards for each `dice_set_3`.
    """
    rewards = np.zeros(len(dice_combinations))

    for dice_index, exit_dice in enumerate(dice_combinations):
        best_additional_score = 0
        new_upper_score = upper_score
        keys = list(const.combinations.keys())  # List of scoring categories
        best_category_name = None

        for ix, category_name in enumerate(keys):
            # Check if the bit at position `ix` is 0
            if (scored_categories & (1 << (12 - ix))) == 0:
                valid = is_dice_set_valid(category_name, exit_dice)
                if valid:
                    score = get_score(category_name, exit_dice)
                    if score >= best_additional_score:
                        best_additional_score = score
                        best_category_name = category_name
                else:
                    score = 0
                    if score >= best_additional_score:
                        best_additional_score = score
                        best_category_name = category_name

        # If the best category is in the upper section, add `best_additional_score` to `new_upper_score`
        if is_upper_section(best_category_name):
            new_upper_score += best_additional_score
        best_category_index = keys.index(best_category_name)
        new_scored_categories = scored_categories | (1 << (12 - best_category_index))
        rewards[dice_index] = best_additional_score

    return rewards


def is_upper_section(category_name):
    return const.combinations[category_name]["upper_section"]


def get_score(category_name, dice_set):
    return const.combinations[category_name]["scorer"] \
        (dices=dice_set,
         die_value=const.combinations[category_name]["required_die_value"],
         die_count=const.combinations[category_name]["required_die_count"])


def is_dice_set_valid(category_name, dice_set):
    return const.combinations[category_name]["validator"] \
        (dices=dice_set,
         die_value=const.combinations[category_name]["required_die_value"],
         die_count=const.combinations[category_name]["required_die_count"])





def compute_expectation(dice_set,
                        scored_categories,
                        upper_score,
                        calculator,
                        rewards,
                        n):
    p = calculator.dice_probability(dice_set)
    best_reroll_vector, max_expectation = calculator.get_best_reroll_vector(dice_set, rewards)


if __name__ == "__main__":
    calculator = ProbabilityCalculator()
    E = {}
    count = 0
    print("Iterating over all states...")
    for num_ones in range(13, -1, -1):  # From 13 ones to 0 ones
        for positions in combinations(range(13), num_ones):  # Choose positions for the ones
            scored_categories = sum(1 << pos for pos in positions)
            # Iterate over all possible values of upper_score
            for upper_score in range(64):
                if count > 0:
                    break
                S = (scored_categories << 6) | upper_score

                game_over = bin(scored_categories).count('1') == 13
                if game_over:
                    E[S] = const.UPPER_BONUS if upper_score >= const.UPPER_BONUS_THRESHOLD else 0
                else:
                    print(f"US: {upper_score} C: {scored_categories:013b}")
                    # Calculate rewards for this state
                    start_time = time.time()
                    rewards = calculate_rewards(calculator.unique_dice_combinations, scored_categories, upper_score)

                    for dice_set_1 in calculator.unique_dice_combinations:
                        p = calculator.get_dice_set_probability(dice_set_1)
                        reroll_vector, max_ev = calculator.get_best_reroll_vector(dice_set_1, rewards)
                        print("DS: ", dice_set_1, "RV: ", reroll_vector, "MEV: ", max_ev)

                    stop_time = time.time()
                    print("Time:", stop_time - start_time)
                    count += 1
