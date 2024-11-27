import os

import numpy as np

from yatzy.mechanics.categories import Category
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
    E = {}
    calculator = ProbabilityCalculator()
    scored_categories = (1 << len(Category)) - 1 - (1 << Category.ONES.value)
    state = YatzyState(upper_score=5, scored_categories=16383)
    if state.is_game_over():
        bonus = state.get_upper_bonus()
    else:


        def precompute_best_options(calculator, state, rewards):
            num_combinations = len(calculator.unique_dice_combinations)

            # Initialize arrays
            best_reroll_vectors = np.zeros((num_combinations, 5), dtype=np.uint8)
            best_categories = np.empty(num_combinations, dtype=object)

            for idx, dice_set in enumerate(calculator.unique_dice_combinations):
                best_reroll_vector, _ = calculator.get_best_reroll_vector(dice_set, rewards)
                best_reroll_vectors[idx] = best_reroll_vector
                best_categories[idx] = state.get_best_category(list(dice_set))

            return best_reroll_vectors, best_categories


        rewards = state.get_best_scores_array(calculator.unique_dice_combinations)
        best_reroll_vectors, best_categories = precompute_best_options(calculator, state, rewards)
        unique_combinations = calculator.unique_dice_combinations
        num_combinations = len(unique_combinations)

        for ix1 in range(num_combinations):
            dice_set_1 = unique_combinations[ix1]
            print(dice_set_1)
            best_reroll_vector_1 = best_reroll_vectors[ix1]
            p_dict_1 = calculator.get_dice_set_probabilities(dice_set_1, best_reroll_vector_1)

            # Convert p_dict_1 to arrays for vectorization
            dice_sets_2 = np.array(list(p_dict_1.keys()))
            probs_2 = np.array(list(p_dict_1.values()))

            # Precompute best reroll vectors and categories for dice_sets_2
            indices_2 = [calculator.combination_to_index[tuple(sorted(dice_set_2))] for dice_set_2 in dice_sets_2]
            best_reroll_vectors_2 = best_reroll_vectors[indices_2]
            best_categories_2 = best_categories[indices_2]

            # print(f"{ix1}:- > ds={dice_set_1} brv={best_reroll_vector_1}")

            for ix2, dice_set_2 in enumerate(dice_sets_2):
                prob_2 = probs_2[ix2]
                best_reroll_vector_2 = best_reroll_vectors_2[ix2]
                p_dict_2 = calculator.get_dice_set_probabilities(dice_set_2, best_reroll_vector_2)

                # Vectorize the innermost loop
                dice_sets_3 = np.array(list(p_dict_2.keys()))
                probs_3 = np.array(list(p_dict_2.values()))
                combination_keys = [tuple(sorted(map(int, dice_set_3))) for dice_set_3 in dice_sets_3]
                indices_3 = [calculator.combination_to_index.get(key, None) for key in combination_keys]

                for ix3, index in enumerate(indices_3):
                    if index is not None:
                        reward = rewards[index]
                        best_category = best_categories[index]

                        if best_category is not None:
                            new_state_id = state.get_next_state_id(best_category, dice_sets_3[ix3])
                            # print(new_state_id)
