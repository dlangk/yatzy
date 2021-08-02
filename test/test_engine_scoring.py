from yatzy import engine
import unittest


class TestScoringMethods(unittest.TestCase):

    def test_upper_section_scoring(self):
        dices = [1, 1, 1, 3, 3]

        self.assertEqual(engine.Engine.upper_section_scoring(dices, die_value=1), 3)
        self.assertEqual(engine.Engine.upper_section_scoring(dices, die_value=3), 6)
        self.assertEqual(engine.Engine.upper_section_scoring(dices, die_value=2), 0)
        self.assertEqual(engine.Engine.upper_section_scoring(dices, die_value=6), 0)

    def test_n_kind_scoring(self):
        dices = [1, 1, 1, 2, 3]
        self.assertEqual(engine.Engine.n_kind_scoring(dices, 3, 1), 3)

    def test_pair_scoring(self):
        dices = [1, 1, 2, 6, 5]
        self.assertEqual(engine.Engine.n_kind_scoring(dices, die_value=1, n=2), 2)
        dices = [1, 1, 1, 6, 5]
        self.assertEqual(engine.Engine.n_kind_scoring(dices, die_value=1, n=2), 2)
        dices = [3, 1, 1, 5, 5]
        self.assertEqual(engine.Engine.n_kind_scoring(dices, die_value=1, n=2), 2)
        self.assertEqual(engine.Engine.n_kind_scoring(dices, die_value=5, n=2), 10)

    def test_two_pair_scoring(self):
        dices = [3, 3, 2, 2, 3]
        self.assertEqual(engine.Engine.two_pair_scoring(dices), 10)
        dices = [3, 3, 1, 1, 5]
        self.assertEqual(engine.Engine.two_pair_scoring(dices), 8)
        dices = [1, 1, 1, 5, 5]
        self.assertEqual(engine.Engine.two_pair_scoring(dices), 12)
        dices = [6, 6, 5, 5, 1]
        self.assertEqual(engine.Engine.two_pair_scoring(dices), 22)
        dices = [6, 1, 3, 6, 3]
        self.assertEqual(engine.Engine.two_pair_scoring(dices, n=2), 18)

    def test_all_dice_scoring(self):
        dices = [1, 1, 1, 2, 2]
        self.assertEqual(engine.Engine.all_dice_scoring(dices), 7)
