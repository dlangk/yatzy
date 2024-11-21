import unittest

from yatzy.optimal_policy.build_graph import get_transition_probability


class TestTransitionProbabilityMethods(unittest.TestCase):

    def test_transition_probability(self):
        dice_set = (1, 1, 1, 1, 1)
        reroll_mask = 0b00000
        target_outcome = (1, 1, 1, 1, 1)
        self.assertEqual(get_transition_probability(dice_set, reroll_mask, target_outcome), (1 / 6) ** 5)

        dice_set = (1, 1, 1, 1, 1)
        reroll_mask = 0b00001
        target_outcome = (1, 1, 1, 1, 1)
        self.assertEqual(get_transition_probability(dice_set, reroll_mask, target_outcome), (1 / 6) ** 4)

        dice_set = (1, 1, 1, 1, 1)
        reroll_mask = 0b00011
        target_outcome = (1, 1, 1, 1, 1)
        self.assertEqual(get_transition_probability(dice_set, reroll_mask, target_outcome), (1 / 6) ** 3)

        dice_set = (1, 1, 1, 1, 1)
        reroll_mask = 0b00111
        target_outcome = (1, 1, 1, 1, 1)
        self.assertEqual(get_transition_probability(dice_set, reroll_mask, target_outcome), (1 / 6) ** 2)

        dice_set = (1, 1, 1, 1, 1)
        reroll_mask = 0b01111
        target_outcome = (1, 1, 1, 1, 1)
        self.assertEqual(get_transition_probability(dice_set, reroll_mask, target_outcome), (1 / 6) ** 1)

        dice_set = (1, 1, 1, 1, 1)
        reroll_mask = 0b11111
        target_outcome = (1, 1, 1, 1, 1)
        self.assertEqual(get_transition_probability(dice_set, reroll_mask, target_outcome), (1 / 6) ** 0)

        dice_set = (1, 2, 3, 4, 5)
        reroll_mask = 0b11111
        target_outcome = (1, 1, 1, 1, 1)
        self.assertEqual(get_transition_probability(dice_set, reroll_mask, target_outcome), 0)

        dice_set = (1, 2, 3, 4, 5)
        reroll_mask = 0b11000
        target_outcome = (1, 2, 1, 1, 1)
        self.assertEqual(get_transition_probability(dice_set, reroll_mask, target_outcome), (1 / 6) ** 3)
