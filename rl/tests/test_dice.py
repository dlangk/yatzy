"""Tests for dice.py — validates dice set enumeration and keep table."""

import numpy as np
import pytest

from yatzy_rl.dice import (
    NUM_DICE_SETS,
    NUM_KEEP_MULTISETS,
    build_all_dice_sets,
    build_keep_table,
    compute_dice_set_probabilities,
    find_dice_set_index,
)


@pytest.fixture(scope="module")
def dice_data():
    all_dice_sets, index_lookup = build_all_dice_sets()
    return all_dice_sets, index_lookup


class TestDiceSets:
    def test_252_dice_sets(self, dice_data):
        all_dice_sets, _ = dice_data
        assert all_dice_sets.shape == (252, 5)

    def test_first_and_last(self, dice_data):
        all_dice_sets, _ = dice_data
        np.testing.assert_array_equal(all_dice_sets[0], [1, 1, 1, 1, 1])
        np.testing.assert_array_equal(all_dice_sets[251], [6, 6, 6, 6, 6])

    def test_all_sorted(self, dice_data):
        all_dice_sets, _ = dice_data
        for i in range(252):
            for j in range(4):
                assert all_dice_sets[i, j] <= all_dice_sets[i, j + 1]

    def test_all_unique(self, dice_data):
        all_dice_sets, _ = dice_data
        seen = set()
        for i in range(252):
            key = tuple(all_dice_sets[i])
            assert key not in seen, f"Duplicate dice set at index {i}"
            seen.add(key)

    def test_round_trip(self, dice_data):
        all_dice_sets, index_lookup = dice_data
        for i in range(252):
            idx = find_dice_set_index(index_lookup, all_dice_sets[i])
            assert idx == i, f"Round-trip failed for index {i}"


class TestDiceProbabilities:
    def test_sum_to_one(self, dice_data):
        all_dice_sets, _ = dice_data
        probs = compute_dice_set_probabilities(all_dice_sets)
        assert abs(probs.sum() - 1.0) < 1e-9

    def test_all_ones_prob(self, dice_data):
        all_dice_sets, _ = dice_data
        probs = compute_dice_set_probabilities(all_dice_sets)
        # P(1,1,1,1,1) = 1/7776
        assert abs(probs[0] - 1.0 / 7776.0) < 1e-12

    def test_four_ones_plus_two(self, dice_data):
        all_dice_sets, _ = dice_data
        probs = compute_dice_set_probabilities(all_dice_sets)
        # P(1,1,1,1,2) = 5/7776
        assert abs(probs[1] - 5.0 / 7776.0) < 1e-12


class TestKeepTable:
    @pytest.fixture(scope="class")
    def keep_table(self, dice_data):
        all_dice_sets, index_lookup = dice_data
        return build_keep_table(all_dice_sets, index_lookup)

    def test_row_sums(self, keep_table):
        """Each keep-multiset's transition probabilities should sum to 1."""
        kt = keep_table
        for ki in range(NUM_KEEP_MULTISETS):
            start = kt.row_start[ki]
            end = kt.row_start[ki + 1]
            if start == end:
                continue
            row_sum = kt.vals[start:end].sum()
            assert abs(row_sum - 1.0) < 1e-9, f"Row {ki} sums to {row_sum}"

    def test_unique_count_range(self, keep_table):
        """Each dice set should have 1-31 unique keeps."""
        kt = keep_table
        for ds in range(252):
            n = kt.unique_count[ds]
            assert 1 <= n <= 31, f"dice set {ds} has {n} unique keeps"

    def test_all_ones_unique_keeps(self, keep_table):
        """[1,1,1,1,1] should have 5 unique keeps."""
        assert keep_table.unique_count[0] == 5

    def test_all_distinct_has_31_keeps(self, dice_data, keep_table):
        """[1,2,3,4,5] should have 31 unique keeps (all masks distinct)."""
        _, index_lookup = dice_data
        ds_12345 = find_dice_set_index(index_lookup, np.array([1, 2, 3, 4, 5], dtype=np.int32))
        assert keep_table.unique_count[ds_12345] == 31

    def test_reroll_all_maps_to_same_keep(self, keep_table):
        """All dice sets with mask=31 should map to the same keep index (empty keep)."""
        empty_kid = keep_table.mask_to_keep[0 * 32 + 31]
        for ds in range(1, 252):
            assert keep_table.mask_to_keep[ds * 32 + 31] == empty_kid
