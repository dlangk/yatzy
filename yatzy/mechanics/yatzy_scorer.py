from collections import Counter
from typing import List

from yatzy.mechanics.categories import Category


class YatzyScorer:
    @staticmethod
    def score_ones(dice: List[int]) -> int:
        return dice.count(1) * 1

    @staticmethod
    def score_twos(dice: List[int]) -> int:
        return dice.count(2) * 2

    @staticmethod
    def score_threes(dice: List[int]) -> int:
        return dice.count(3) * 3

    @staticmethod
    def score_fours(dice: List[int]) -> int:
        return dice.count(4) * 4

    @staticmethod
    def score_fives(dice: List[int]) -> int:
        return dice.count(5) * 5

    @staticmethod
    def score_sixes(dice: List[int]) -> int:
        return dice.count(6) * 6

    @staticmethod
    def score_one_pair(dice: List[int]) -> int:
        counts = Counter(dice)
        pairs = [num for num, cnt in counts.items() if cnt >= 2]
        if pairs:
            return max(pairs) * 2
        return 0

    @staticmethod
    def score_two_pairs(dice: List[int]) -> int:
        counts = Counter(dice)
        pairs = [num for num, cnt in counts.items() if cnt >= 2]
        if len(pairs) >= 2:
            top_two = sorted(pairs, reverse=True)[:2]
            return sum([num * 2 for num in top_two])
        return 0

    @staticmethod
    def score_three_of_a_kind(dice: List[int]) -> int:
        counts = Counter(dice)
        triples = [num for num, cnt in counts.items() if cnt >= 3]
        if triples:
            return max(triples) * 3
        return 0

    @staticmethod
    def score_four_of_a_kind(dice: List[int]) -> int:
        counts = Counter(dice)
        quads = [num for num, cnt in counts.items() if cnt >= 4]
        if quads:
            return max(quads) * 4
        return 0

    @staticmethod
    def score_small_straight(dice: List[int]) -> int:
        if sorted(dice) == [1, 2, 3, 4, 5]:
            return 15
        return 0

    @staticmethod
    def score_large_straight(dice: List[int]) -> int:
        if sorted(dice) == [2, 3, 4, 5, 6]:
            return 20
        return 0

    @staticmethod
    def score_full_house(dice: List[int]) -> int:
        counts = Counter(dice)
        has_three = None
        has_two = None
        for num, cnt in counts.items():
            if cnt >= 3:
                if has_three is None or num > has_three:
                    has_three = num
        if has_three is not None:
            for num, cnt in counts.items():
                if num != has_three and cnt >= 2:
                    if has_two is None or num > has_two:
                        has_two = num
        if has_three is not None and has_two is not None:
            return has_three * 3 + has_two * 2
        return 0

    @staticmethod
    def score_chance(dice: List[int]) -> int:
        return sum(dice)

    @staticmethod
    def score_yatzy(dice: List[int]) -> int:
        if len(set(dice)) == 1:
            return 50
        return 0

    @staticmethod
    def calculate_score(category: Category, dice_set: List[int]) -> int:
        """
        Calculates the score for a specific category based on the dice set.

        :param category: The category to score.
        :param dice_set: List of five integers representing the dice.
        :return: The score for the category.
        """
        if category == Category.ONES:
            return YatzyScorer.score_ones(dice_set)
        elif category == Category.TWOS:
            return YatzyScorer.score_twos(dice_set)
        elif category == Category.THREES:
            return YatzyScorer.score_threes(dice_set)
        elif category == Category.FOURS:
            return YatzyScorer.score_fours(dice_set)
        elif category == Category.FIVES:
            return YatzyScorer.score_fives(dice_set)
        elif category == Category.SIXES:
            return YatzyScorer.score_sixes(dice_set)
        elif category == Category.ONE_PAIR:
            return YatzyScorer.score_one_pair(dice_set)
        elif category == Category.TWO_PAIRS:
            return YatzyScorer.score_two_pairs(dice_set)
        elif category == Category.THREE_OF_A_KIND:
            return YatzyScorer.score_three_of_a_kind(dice_set)
        elif category == Category.FOUR_OF_A_KIND:
            return YatzyScorer.score_four_of_a_kind(dice_set)
        elif category == Category.SMALL_STRAIGHT:
            return YatzyScorer.score_small_straight(dice_set)
        elif category == Category.LARGE_STRAIGHT:
            return YatzyScorer.score_large_straight(dice_set)
        elif category == Category.FULL_HOUSE:
            return YatzyScorer.score_full_house(dice_set)
        elif category == Category.CHANCE:
            return YatzyScorer.score_chance(dice_set)
        elif category == Category.YATZY:
            return YatzyScorer.score_yatzy(dice_set)
        else:
            raise ValueError(f"Unknown category: {category}")
