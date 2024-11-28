from itertools import combinations_with_replacement
from typing import List, Dict, Optional, Tuple

import numpy as np
from numba import njit

from yatzy.mechanics import const
from yatzy.mechanics.const import CATEGORY_COUNT
from yatzy.mechanics.yatzy_scorer import YatzyScorer
from yatzy.probabilities.probability_calculator import ProbabilityCalculator


# Numba-accelerated function to get allowed categories
@njit
def get_allowed_categories_numba(scored_categories: int, all_categories: np.ndarray) -> np.ndarray:
    allowed = []
    for i in range(all_categories.size):
        category = all_categories[i]
        if (scored_categories & (1 << category)) == 0:
            allowed.append(category)
    return np.array(allowed, dtype=np.int32)


@njit
def get_next_state_id_numba(state_upper_score, state_scored_categories, category, dice_set,
                            precomputed_scores,
                            combination_lookup, upper_section_bitmask):
    # Check if category is already scored
    if (state_scored_categories & (1 << category)) != 0:
        return -1  # Sentinel for already scored

    # Convert sorted dice_set to unique key
    key = 0
    for i in range(5):
        die = dice_set[i]
        # Validate die value
        if die < 1 or die > 6:
            return -2  # Sentinel value for invalid die value
        key = key * 6 + (die - 1)

    combination_index = combination_lookup[key]
    if combination_index == -1:
        return -2  # Sentinel for invalid dice set

    score = precomputed_scores[category, combination_index]

    # Update upper score
    if ((1 << category) & upper_section_bitmask) != 0:
        new_upper_score = state_upper_score + score
        if new_upper_score > 63:
            new_upper_score = 63
    else:
        new_upper_score = state_upper_score

    # Update scored categories
    new_scored_categories = state_scored_categories | (1 << category)

    # Compute new state_id directly
    new_state_id = (new_upper_score << 15) | (new_scored_categories & 0x7FFF)

    return new_state_id


class YatzyState:
    def __init__(self, upper_score: int, scored_categories: int, calculator: ProbabilityCalculator):
        """
        Initializes the YatzyState with integer representations.

        :param upper_score: An integer (0-63) representing the upper score.
        :param scored_categories: An integer (bitmask) representing scored categories.
        """
        self.upper_score = min(max(upper_score, 0), 63)
        self.scored_categories = scored_categories

        precomputed_scores = np.zeros((CATEGORY_COUNT, len(calculator.unique_dice_combinations)), dtype=np.int32)

        for category in range(CATEGORY_COUNT):
            for idx, dice_set in enumerate(calculator.unique_dice_combinations):
                precomputed_scores[category, idx] = YatzyScorer.calculate_score(category, list(dice_set))

        self.precomputed_scores = precomputed_scores

    def get_state_id(self) -> int:
        """
        Generates a unique state ID by combining upper_score and scored_categories.

        :return: An integer representing the unique state ID.
        """
        # Shift upper_score by 15 bits to the left and combine with scored_categories
        return (self.upper_score << 15) | (self.scored_categories & 0x7FFF)

    def is_category_scored(self, category: int) -> bool:
        """
        Checks if a category has already been scored.

        :param category: The category to check.
        :return: True if scored, False otherwise.
        """
        return (self.scored_categories & (1 << category)) != 0

    def get_scores_for_allowed_categories(self, dice_set: List[int]) -> Dict[int, int]:
        """
        Calculates the potential score for each allowed category based on the dice set.

        :param dice_set: List of five integers representing the dice.
        :return: Dictionary mapping Category to its potential score.
        """
        return {category: self.calculate_score(category, dice_set) for category in get_allowed_categories_numba()}

    def get_allowed_categories(self) -> List[int]:
        """
        Determines which categories are still available for scoring.

        :return: List of allowed category integers.
        """
        allowed_array = get_allowed_categories_numba(self.scored_categories, const.ALL_CATEGORIES_ARRAY)
        return allowed_array.tolist()

    @staticmethod
    def combination_to_index(dice_set: List[int], combination_lookup: np.ndarray) -> np.ndarray:
        """
        Converts a sorted dice set to its corresponding index using the lookup table.

        :param dice_set: List of five integers representing the dice.
        :param combination_lookup: NumPy array serving as the lookup table.
        :return: Index corresponding to the dice set, or -1 if invalid.
        """
        key = 0
        for die in sorted(dice_set):
            key = key * 6 + (die - 1)
        return combination_lookup[key]

    def get_best_category(self, dice_set: List[int], precomputed_scores: np.ndarray, combination_lookup: np.ndarray) -> int:
        """
        Identifies the allowed category that yields the highest score.

        :param dice_set: List of five integers representing the dice.
        :param precomputed_scores: Precomputed scores array.
        :param combination_lookup: Lookup table to convert dice_set to index.
        :return: The category integer with the highest potential score, or -1 if no categories left.
        """
        best_score = -1
        best_category = -1

        # Convert dice_set to index
        key = 0
        for die in sorted(dice_set):
            key = key * 6 + (die - 1)
        combination_index = combination_lookup[key]

        if combination_index == -1:
            return -1  # Invalid dice set

        for category in const.ALL_CATEGORIES:
            if (self.scored_categories & (1 << category)) == 0:
                score = precomputed_scores[category, combination_index]
                if score > best_score:
                    best_score = score
                    best_category = category

        return best_category if best_category != -1 else -1

    @staticmethod
    def calculate_score(category: int, dice_set: List[int]) -> int:
        """
        Calculates the score for a specific category based on the dice set.

        :param category: The category to score.
        :param dice_set: List of five integers representing the dice.
        :return: The score for the category.
        """
        return YatzyScorer.calculate_score(category, dice_set)

    def apply_scoring(self, category: int, dice_set: List[int]) -> None:
        """
        Applies the scoring for a selected category, updating the game state accordingly.

        :param category: The category to score.
        :param dice_set: List of five integers representing the dice.
        """
        if self.is_category_scored(category):
            raise ValueError(f"Category {category} has already been scored.")

        score = self.calculate_score(category, dice_set)

        # Update upper_score if category is in the upper section
        if category in {const.ONES, const.TWOS, const.THREES,
                        const.FOURS, const.FIVES, const.SIXES}:
            new_upper_score = self.upper_score + score
            self.upper_score = min(new_upper_score, 63)

        # Mark the category as scored
        self.scored_categories |= (1 << category)

    def get_upper_bonus(self) -> int:
        """
        Determines if the upper score qualifies for a bonus.

        :return: 35 if upper_score >= 63, otherwise 0.
        """
        return 35 if self.upper_score >= 63 else 0

    def is_game_over(self) -> bool:
        """
        Checks if the game is over (i.e., all categories have been scored).

        :return: True if the game is over, False otherwise.
        """
        return self.scored_categories == (1 << const.CATEGORY_COUNT) - 1  # 0b111111111111111

    def get_best_scores_array(
            self,
            unique_dice_combinations: List[Tuple[int, ...]]
    ) -> np.ndarray:
        """
        For each unique dice set, determine the best possible score based on the current state.
        Returns a NumPy array where the index corresponds to the position in unique_dice_combinations.

        :param unique_dice_combinations: List of all unique dice combinations.
        :return: NumPy array of best scores indexed by dice set.
        """
        best_scores = np.zeros(len(unique_dice_combinations), dtype=int)
        allowed_categories = self.get_allowed_categories()

        for idx, dice_set in enumerate(unique_dice_combinations):
            scores = [
                self.calculate_score(category, list(dice_set))
                for category in allowed_categories
            ]
            best_scores[idx] = max(scores) if scores else 0

        return best_scores

    def get_best_scores_for_all_dice_sets(self) -> Dict[Tuple[int, ...], Tuple[int, int]]:
        """
        Evaluates all 252 unique dice sets and determines the best score and corresponding category for each.

        :return: Dictionary mapping each unique dice set tuple to a tuple of (best_score, best_category).
        """
        best_scores = {}
        # Generate all unique dice sets using combinations with replacement
        for dice_set in combinations_with_replacement(range(1, 7), 5):
            scores = self.get_scores_for_allowed_categories(list(dice_set))
            if scores:
                best_category, best_score = max(scores.items(), key=lambda item: item[1])
                best_scores[dice_set] = (best_score, best_category)
            else:
                best_scores[dice_set] = (0, None)
        return best_scores

    def get_upper_score_binary(self) -> str:
        """
        Returns the upper_score as a 6-bit binary string.

        :return: 6-bit binary string.
        """
        return format(self.upper_score, '06b')

    def get_scored_categories_binary(self) -> str:
        """
        Returns the scored_categories as a 15-bit binary string.

        :return: 15-bit binary string.
        """
        return format(self.scored_categories, '015b')

    def __str__(self):
        """
        Returns a string representation of the YatzyState.

        :return: String showing upper_score and scored_categories in binary.
        """
        return (
            f"upper_score: {self.get_upper_score_binary()} ({self.upper_score}) "
            f"scored_categories: {self.get_scored_categories_binary()}"
        )
