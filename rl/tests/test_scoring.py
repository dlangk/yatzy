"""Tests for scoring.py — validates against Rust test vectors from game_mechanics.rs."""

import numpy as np
import pytest

from yatzy_rl.scoring import (
    CATEGORY_CHANCE,
    CATEGORY_COUNT,
    CATEGORY_FIVES,
    CATEGORY_FOURS,
    CATEGORY_FOUR_OF_A_KIND,
    CATEGORY_FULL_HOUSE,
    CATEGORY_LARGE_STRAIGHT,
    CATEGORY_ONES,
    CATEGORY_ONE_PAIR,
    CATEGORY_SIXES,
    CATEGORY_SMALL_STRAIGHT,
    CATEGORY_THREES,
    CATEGORY_THREE_OF_A_KIND,
    CATEGORY_TWOS,
    CATEGORY_TWO_PAIRS,
    CATEGORY_YATZY,
    calculate_category_score,
    count_faces,
    is_category_scored,
    state_index,
    update_upper_score,
)


def d(vals: list[int]) -> np.ndarray:
    return np.array(vals, dtype=np.int32)


class TestUpperSection:
    def test_ones(self):
        assert calculate_category_score(d([1, 1, 1, 1, 1]), CATEGORY_ONES) == 5
        assert calculate_category_score(d([1, 2, 3, 4, 5]), CATEGORY_ONES) == 1

    def test_twos(self):
        assert calculate_category_score(d([1, 2, 3, 4, 5]), CATEGORY_TWOS) == 2

    def test_threes(self):
        assert calculate_category_score(d([3, 3, 4, 5, 6]), CATEGORY_THREES) == 6

    def test_fours(self):
        assert calculate_category_score(d([4, 4, 4, 4, 4]), CATEGORY_FOURS) == 20

    def test_fives(self):
        assert calculate_category_score(d([5, 5, 5, 1, 2]), CATEGORY_FIVES) == 15

    def test_sixes(self):
        assert calculate_category_score(d([6, 6, 6, 6, 6]), CATEGORY_SIXES) == 30


class TestOnePair:
    def test_pair(self):
        assert calculate_category_score(d([3, 3, 4, 5, 6]), CATEGORY_ONE_PAIR) == 6

    def test_highest_pair(self):
        assert calculate_category_score(d([5, 5, 6, 6, 1]), CATEGORY_ONE_PAIR) == 12

    def test_no_pair(self):
        assert calculate_category_score(d([1, 2, 3, 4, 6]), CATEGORY_ONE_PAIR) == 0


class TestTwoPairs:
    def test_two_pairs(self):
        assert calculate_category_score(d([3, 3, 5, 5, 6]), CATEGORY_TWO_PAIRS) == 16

    def test_one_pair_only(self):
        assert calculate_category_score(d([1, 1, 2, 3, 4]), CATEGORY_TWO_PAIRS) == 0


class TestNOfAKind:
    def test_three_of_a_kind(self):
        assert calculate_category_score(d([2, 2, 2, 4, 5]), CATEGORY_THREE_OF_A_KIND) == 6

    def test_four_of_a_kind(self):
        assert calculate_category_score(d([4, 4, 4, 4, 2]), CATEGORY_FOUR_OF_A_KIND) == 16

    def test_no_three(self):
        assert calculate_category_score(d([1, 2, 3, 4, 5]), CATEGORY_THREE_OF_A_KIND) == 0

    def test_no_four(self):
        assert calculate_category_score(d([3, 3, 3, 4, 5]), CATEGORY_FOUR_OF_A_KIND) == 0


class TestStraights:
    def test_small_straight(self):
        assert calculate_category_score(d([1, 2, 3, 4, 5]), CATEGORY_SMALL_STRAIGHT) == 15

    def test_large_straight(self):
        assert calculate_category_score(d([2, 3, 4, 5, 6]), CATEGORY_LARGE_STRAIGHT) == 20

    def test_not_small(self):
        assert calculate_category_score(d([1, 2, 3, 4, 6]), CATEGORY_SMALL_STRAIGHT) == 0
        assert calculate_category_score(d([2, 3, 4, 5, 6]), CATEGORY_SMALL_STRAIGHT) == 0

    def test_not_large(self):
        assert calculate_category_score(d([1, 2, 3, 4, 5]), CATEGORY_LARGE_STRAIGHT) == 0


class TestFullHouse:
    def test_full_house(self):
        assert calculate_category_score(d([2, 2, 3, 3, 3]), CATEGORY_FULL_HOUSE) == 13

    def test_no_full_house(self):
        assert calculate_category_score(d([1, 2, 3, 4, 6]), CATEGORY_FULL_HOUSE) == 0

    def test_five_of_a_kind_not_full_house(self):
        assert calculate_category_score(d([5, 5, 5, 5, 5]), CATEGORY_FULL_HOUSE) == 0


class TestChance:
    def test_chance(self):
        assert calculate_category_score(d([1, 3, 4, 5, 6]), CATEGORY_CHANCE) == 19
        assert calculate_category_score(d([1, 1, 1, 1, 1]), CATEGORY_CHANCE) == 5


class TestYatzy:
    def test_yatzy(self):
        assert calculate_category_score(d([6, 6, 6, 6, 6]), CATEGORY_YATZY) == 50
        assert calculate_category_score(d([1, 1, 1, 1, 1]), CATEGORY_YATZY) == 50

    def test_not_yatzy(self):
        assert calculate_category_score(d([5, 6, 6, 6, 6]), CATEGORY_YATZY) == 0


class TestUpdateUpperScore:
    def test_upper_category(self):
        assert update_upper_score(0, CATEGORY_ONES, 5) == 5
        assert update_upper_score(10, CATEGORY_SIXES, 30) == 40

    def test_capped_at_63(self):
        assert update_upper_score(60, CATEGORY_FIVES, 30) == 63
        assert update_upper_score(63, CATEGORY_ONES, 5) == 63

    def test_lower_category_no_change(self):
        assert update_upper_score(10, CATEGORY_ONE_PAIR, 12) == 10
        assert update_upper_score(50, CATEGORY_YATZY, 50) == 50


class TestStateIndex:
    def test_index(self):
        assert state_index(0, 0) == 0
        assert state_index(1, 0) == 1
        assert state_index(0, 1) == 64
        assert state_index(63, (1 << 15) - 1) == ((1 << 15) - 1) * 64 + 63


class TestCategoryScored:
    def test_scored(self):
        assert is_category_scored(0b1, 0)
        assert not is_category_scored(0b1, 1)
        assert is_category_scored(0b101, 2)


class TestCountFaces:
    def test_count(self):
        fc = count_faces(d([1, 1, 2, 3, 3]))
        assert fc[1] == 2
        assert fc[2] == 1
        assert fc[3] == 2
        assert fc[4] == 0

    def test_all_same(self):
        fc = count_faces(d([6, 6, 6, 6, 6]))
        assert fc[6] == 5
        assert fc[1] == 0
