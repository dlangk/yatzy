from yatzy import engine
import unittest


class TestScoringMethods(unittest.TestCase):

    def test_upper_section_scoring(self):
        dices = [1, 1, 1, 3, 3]
        self.assertEqual(engine.upper_section_scoring(dices, 1, 3), 3)
        self.assertEqual(engine.upper_section_scoring(dices, 3, 2), 6)
        self.assertEqual(engine.upper_section_scoring(dices, 2, 0), 0)
        self.assertEqual(engine.upper_section_scoring(dices, 6, 0), 0)

    def test_n_kind_scoring(self):
        dices = [1, 1, 1, 2, 3]
        self.assertEqual(engine.n_kind_scoring(dices, 3, 1), 3)

    def test_two_pair_scoring(self):
        dices = [3, 3, 2, 2, 3]
        self.assertEqual(engine.two_pair_scoring(dices, None, None), 10)
        dices = [3, 3, 1, 1, 5]
        self.assertEqual(engine.two_pair_scoring(dices, None, None), 8)
        dices = [6, 6, 5, 5, 1]
        self.assertEqual(engine.two_pair_scoring(dices, None, None), 22)

    def test_all_dice_scoring(self):
        dices = [1, 1, 1, 2, 2]
        self.assertEqual(engine.all_dice_scoring(dices), 7)
