import unittest


class TestValidatorMethods(unittest.TestCase):

    def test_upper_section_validator(self):
        dices = [1, 2, 3, 4, 5]
        self.assertTrue(engine.GameEngine.upper_section_validator(dices, 1, 1))

    def test_n_kind_validator(self):
        dices = [1, 1, 1, 2, 5]
        self.assertTrue(engine.GameEngine.n_kind_validator(dices, n=2)[0])
        self.assertTrue(engine.GameEngine.n_kind_validator(dices, n=3)[0])
        self.assertFalse(engine.GameEngine.n_kind_validator(dices, n=4)[0])
        dices = [1, 1, 1, 1, 1]
        self.assertTrue(engine.GameEngine.n_kind_validator(dices, n=5)[0])
        self.assertFalse(engine.GameEngine.n_kind_validator(dices, n=6)[0])

        dices = [2, 1, 4, 4, 5]
        self.assertFalse(engine.GameEngine.n_kind_validator(dices, n=5)[0])

        dices = [1, 5, 5, 1, 5]
        self.assertTrue(engine.GameEngine.n_kind_validator(dices, n=3)[0])

        dices = [1, 1, 2, 2, 3]
        self.assertEqual(engine.GameEngine.n_kind_validator(dices, n=2), (True, [1, 2]))

    def test_two_pair_validator(self):
        dices = [1, 1, 2, 2, 3]
        self.assertTrue(engine.GameEngine.two_pairs_validator(dices, n=12)[0])
        dices = [1, 1, 1, 1, 2]
        self.assertFalse(engine.GameEngine.two_pairs_validator(dices, n=1)[0])

    def test_straights_validator(self):
        dices = [1, 2, 3, 4, 5]
        self.assertTrue(engine.GameEngine.small_straight_validator(dices)[0])
        self.assertFalse(engine.GameEngine.large_straight_validator(dices)[0])
        dices = [2, 3, 4, 5, 6]
        self.assertFalse(engine.GameEngine.small_straight_validator(dices)[0])
        self.assertTrue(engine.GameEngine.large_straight_validator(dices)[0])

    def test_full_house_validator(self):
        dices = [6, 6, 6, 5, 5]
        self.assertTrue(engine.GameEngine.full_house_validator(dices)[0])
        dices = [5, 5, 5, 6, 6]
        self.assertTrue(engine.GameEngine.full_house_validator(dices)[0])
        dices = [1, 2, 3, 3, 3]
        self.assertFalse(engine.GameEngine.full_house_validator(dices)[0])
        dices = [2, 2, 3, 4, 5]
        self.assertFalse(engine.GameEngine.full_house_validator(dices)[0])
        dices = [1, 1, 1, 1, 1]
        self.assertFalse(engine.GameEngine.full_house_validator(dices)[0])

    def test_chance_validator(self):
        dices = [1, 5, 3, 4, 5]
        self.assertTrue(engine.GameEngine.chance_validator(dices)[0])
        dices = [1, 2, 2, 3, 4]
        self.assertTrue(engine.GameEngine.chance_validator(dices)[0])


if __name__ == '__main__':
    unittest.main()
