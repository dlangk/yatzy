from itertools import combinations_with_replacement
from typing import List, Dict, Optional, Tuple

import numpy as np

from yatzy.mechanics.categories import Category
from yatzy.mechanics.yatzy_scorer import YatzyScorer


class YatzyState:
    def __init__(self, upper_score: int, scored_categories: int):
        """
        Initializes the YatzyState with integer representations.

        :param upper_score: An integer (0-63) representing the upper score.
        :param scored_categories: An integer (bitmask) representing scored categories.
        """
        self.upper_score = min(max(upper_score, 0), 63)
        self.scored_categories = scored_categories

    def is_category_scored(self, category: Category) -> bool:
        """
        Checks if a category has already been scored.

        :param category: The category to check.
        :return: True if scored, False otherwise.
        """
        return (self.scored_categories & (1 << category.value)) != 0

    def get_allowed_categories(self) -> List[Category]:
        """
        Determines which categories are still available for scoring.

        :return: List of allowed Category enums.
        """
        return [category for category in Category if not self.is_category_scored(category)]

    def get_scores_for_allowed_categories(self, dice_set: List[int]) -> Dict[Category, int]:
        """
        Calculates the potential score for each allowed category based on the dice set.

        :param dice_set: List of five integers representing the dice.
        :return: Dictionary mapping Category to its potential score.
        """
        return {category: self.calculate_score(category, dice_set) for category in self.get_allowed_categories()}

    def get_best_category(self, dice_set: List[int]) -> Optional[Category]:
        """
        Identifies the allowed category that yields the highest score.

        :param dice_set: List of five integers representing the dice.
        :return: The Category enum with the highest potential score, or None if no categories left.
        """
        scores = self.get_scores_for_allowed_categories(dice_set)
        if not scores:
            return None  # No categories left to score
        return max(scores.items(), key=lambda item: item[1])[0]

    def calculate_score(self, category: Category, dice_set: List[int]) -> int:
        """
        Calculates the score for a specific category based on the dice set.

        :param category: The category to score.
        :param dice_set: List of five integers representing the dice.
        :return: The score for the category.
        """
        return YatzyScorer.calculate_score(category, dice_set)

    def apply_scoring(self, category: Category, dice_set: List[int]) -> None:
        """
        Applies the scoring for a selected category, updating the game state accordingly.

        :param category: The category to score.
        :param dice_set: List of five integers representing the dice.
        """
        if self.is_category_scored(category):
            raise ValueError(f"Category {category.name} has already been scored.")

        score = self.calculate_score(category, dice_set)

        # Update upper_score if category is in the upper section
        if category in {Category.ONES, Category.TWOS, Category.THREES,
                        Category.FOURS, Category.FIVES, Category.SIXES}:
            new_upper_score = self.upper_score + score
            self.upper_score = min(new_upper_score, 63)

        # Mark the category as scored
        self.scored_categories |= (1 << category.value)

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
        return self.scored_categories == (1 << len(Category)) - 1  # 0b111111111111111

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

    def get_best_scores_for_all_dice_sets(self) -> Dict[Tuple[int, ...], Tuple[int, Category]]:
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