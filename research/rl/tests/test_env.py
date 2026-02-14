"""Tests for env.py — validates game simulation logic."""

import numpy as np
import pytest

from yatzy_rl.env import (
    DirectActionEnv,
    GameState,
    TableContext,
    _apply_reroll,
    _roll_dice,
)
from yatzy_rl.scoring import CATEGORY_COUNT


@pytest.fixture(scope="module")
def ctx():
    return TableContext()


class TestRollDice:
    def test_range(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            dice = _roll_dice(rng)
            assert all(1 <= d <= 6 for d in dice)
            assert len(dice) == 5

    def test_sorted(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            dice = _roll_dice(rng)
            for i in range(4):
                assert dice[i] <= dice[i + 1]


class TestApplyReroll:
    def test_mask_zero_keeps_all(self):
        rng = np.random.default_rng(42)
        dice = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        new_dice = _apply_reroll(dice, 0, rng)
        np.testing.assert_array_equal(new_dice, [1, 2, 3, 4, 5])

    def test_mask_31_rerolls_all(self):
        rng = np.random.default_rng(42)
        dice = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        new_dice = _apply_reroll(dice, 31, rng)
        assert all(1 <= d <= 6 for d in new_dice)
        for i in range(4):
            assert new_dice[i] <= new_dice[i + 1]


class TestTableContext:
    def test_scores_shape(self, ctx):
        assert ctx.scores.shape == (252, CATEGORY_COUNT)

    def test_first_dice_set(self, ctx):
        np.testing.assert_array_equal(ctx.all_dice_sets[0], [1, 1, 1, 1, 1])

    def test_find_index_round_trip(self, ctx):
        for i in range(252):
            idx = ctx.find_index(ctx.all_dice_sets[i])
            assert idx == i


class TestDirectActionEnv:
    def test_episode_completes(self, ctx):
        """A full episode should score all 15 categories."""
        sv = np.zeros(64 * (1 << 15), dtype=np.float32)
        env = DirectActionEnv(sv, ctx, seed=42)
        obs = env.reset()

        done = False
        steps = 0
        while not done:
            if env.phase == "reroll":
                action = 0  # Always keep all
            else:
                valid = env.get_valid_actions()
                action = int(np.where(valid)[0][0])  # First valid category

            obs, reward, done, info = env.step(action)
            steps += 1
            assert steps < 100, "Episode did not terminate"

        assert done
        assert info["total_score"] >= 0

    def test_valid_actions_reroll(self, ctx):
        sv = np.zeros(64 * (1 << 15), dtype=np.float32)
        env = DirectActionEnv(sv, ctx, seed=42)
        env.reset()
        assert env.phase == "reroll"
        valid = env.get_valid_actions()
        assert valid.sum() == 32  # All masks valid

    def test_valid_actions_score_decreases(self, ctx):
        """After scoring, fewer categories should be available."""
        sv = np.zeros(64 * (1 << 15), dtype=np.float32)
        env = DirectActionEnv(sv, ctx, seed=42)
        env.reset()

        # Skip both rerolls
        env.step(0)  # Keep all, reroll 1
        env.step(0)  # Keep all, reroll 2
        assert env.phase == "score"

        valid_before = env.get_valid_actions()
        assert valid_before.sum() == CATEGORY_COUNT  # All 15 categories available

        # Score first valid category
        action = int(np.where(valid_before)[0][0])
        env.step(action)

        # Skip rerolls for next turn
        env.step(0)
        env.step(0)
        assert env.phase == "score"

        valid_after = env.get_valid_actions()
        assert valid_after.sum() == CATEGORY_COUNT - 1
