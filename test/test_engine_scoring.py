import unittest

import yatzy.mechanics.scorers as scorers


class TestScoringMethods(unittest.TestCase):

    def test_upper_section_scoring(self):
        dices = [1, 1, 1, 3, 3]

        self.assertEqual(scorers.upper_section_scorer(dices, die_value=1), 3)
        self.assertEqual(scorers.upper_section_scorer(dices, die_value=3), 6)
        self.assertEqual(scorers.upper_section_scorer(dices, die_value=2), 0)
        self.assertEqual(scorers.upper_section_scorer(dices, die_value=6), 0)

    def test_n_kind_scoring(self):
        dices = [1, 1, 1, 2, 3]
        self.assertEqual(scorers.n_kind_scorer(dices, 3, 1), 3)

    def test_pair_scoring(self):
        dices = [1, 1, 2, 6, 5]
        self.assertEqual(scorers.n_kind_scorer(dices, die_value=1, die_count=2), 2)
        dices = [1, 1, 1, 6, 5]
        self.assertEqual(scorers.n_kind_scorer(dices, die_value=1, die_count=2), 2)
        dices = [3, 1, 1, 5, 5]
        self.assertEqual(scorers.n_kind_scorer(dices, die_value=1, die_count=2), 2)
        self.assertEqual(scorers.n_kind_scorer(dices, die_value=5, die_count=2), 10)

    def test_two_pair_scoring(self):
        dices = [3, 3, 2, 2, 3]
        self.assertEqual(scorers.two_pairs_scorer(dices), 10)
        dices = [3, 3, 1, 1, 5]
        self.assertEqual(scorers.two_pairs_scorer(dices), 8)
        dices = [1, 1, 1, 5, 5]
        self.assertEqual(scorers.two_pairs_scorer(dices), 12)
        dices = [6, 6, 5, 5, 1]
        self.assertEqual(scorers.two_pairs_scorer(dices), 22)
        dices = [6, 1, 3, 6, 3]
        self.assertEqual(scorers.two_pairs_scorer(dices), 18)

    def test_all_dice_scoring(self):
        dices = [1, 1, 1, 2, 2]
        self.assertEqual(scorers.all_dice_scorer(dices), 7)
