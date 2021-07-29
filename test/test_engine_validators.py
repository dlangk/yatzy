from yatzy import engine
import unittest


class TestValidatorMethods(unittest.TestCase):

    def test_upper_section_validator(self):
        self.assertTrue(engine.upper_section_validator(engine.roll_dice(5), 1, 1))

    def test_n_kind_validator(self):
        dices = [1, 1, 1, 2, 5]
        self.assertTrue(engine.n_kind_validator(dices, None, 2)[0])
        self.assertTrue(engine.n_kind_validator(dices, None, 3)[0])
        self.assertFalse(engine.n_kind_validator(dices, None, 4)[0])
        dices = [1, 1, 1, 1, 1]
        self.assertTrue(engine.n_kind_validator(dices, None, 5)[0])
        self.assertFalse(engine.n_kind_validator(dices, None, 6)[0])
        dices = [1, 1, 2, 2, 3]
        self.assertEqual(engine.n_kind_validator(dices, None, 2), (True, [1, 2]))

    def test_two_pair_validator(self):
        dices = [1, 1, 2, 2, 3]
        self.assertTrue(engine.two_pair_validator(dices, None, 2)[0])
        dices = [1, 1, 1, 1, 2]
        self.assertFalse(engine.two_pair_validator(dices, None, 2)[0])

    def test_straights_validator(self):
        dices = [1, 2, 3, 4, 5]
        self.assertTrue(engine.small_straight_validator(dices)[0])
        self.assertFalse(engine.large_straight_validator(dices)[0])
        dices = [2, 3, 4, 5, 6]
        self.assertFalse(engine.small_straight_validator(dices)[0])
        self.assertTrue(engine.large_straight_validator(dices)[0])

    def test_full_house_validator(self):
        dices = [6, 6, 6, 5, 5]
        self.assertTrue(engine.full_house_validator(dices)[0])
        dices = [5, 5, 5, 6, 6]
        self.assertTrue(engine.full_house_validator(dices)[0])
        dices = [1, 2, 3, 3, 3]
        self.assertFalse(engine.full_house_validator(dices)[0])
        dices = [2, 2, 3, 4, 5]
        self.assertFalse(engine.full_house_validator(dices)[0])
        dices = [1, 1, 1, 1, 1]
        self.assertFalse(engine.full_house_validator(dices)[0])

    def test_chance_validator(self):
        dices = [1, 5, 3, 4, 5]
        self.assertTrue(engine.chance_validator(dices)[0])
        dices = [1, 2, 2, 3, 4]
        self.assertTrue(engine.chance_validator(dices)[0])


if __name__ == '__main__':
    unittest.main()
