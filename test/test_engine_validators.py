import unittest

import yatzy.mechanics.validators as validators


class TestValidatorMethods(unittest.TestCase):

    def test_upper_section_validator(self):
        dices = [1, 2, 3, 4, 5]
        self.assertTrue(validators.upper_section_validator(dices, 1, 1))

    def test_n_kind_validator(self):
        dices = [1, 1, 1, 2, 5]
        self.assertTrue(validators.n_kind_validator(dices, die_count=2)["playable"])
        self.assertTrue(validators.n_kind_validator(dices, die_value=1, die_count=3)["playable"])
        self.assertFalse(validators.n_kind_validator(dices, die_count=4)["playable"])
        dices = [1, 1, 1, 1, 1]
        self.assertTrue(validators.n_kind_validator(dices, die_count=5)["playable"])
        self.assertFalse(validators.n_kind_validator(dices, die_count=6)["playable"])

        dices = [2, 1, 4, 4, 5]
        self.assertFalse(validators.n_kind_validator(dices, die_count=5)["playable"])

        dices = [1, 5, 5, 1, 5]
        self.assertTrue(validators.n_kind_validator(dices, die_count=3)["playable"])

        dices = [1, 1, 2, 2, 3]
        self.assertEqual(validators.n_kind_validator(dices, die_count=2), {'playable': True, 'die': [1, 2]})

    def test_two_pair_validator(self):
        dices = [1, 1, 2, 2, 3]
        self.assertTrue(validators.two_pairs_validator(dices, die_count=12)["playable"])
        dices = [1, 1, 1, 1, 2]
        self.assertFalse(validators.two_pairs_validator(dices, die_count=1)["playable"])

    def test_straights_validator(self):
        dices = [1, 2, 3, 4, 5]
        self.assertTrue(validators.small_straight_validator(dices)["playable"])
        self.assertFalse(validators.large_straight_validator(dices)["playable"])
        dices = [2, 3, 4, 5, 6]
        self.assertFalse(validators.small_straight_validator(dices)["playable"])
        self.assertTrue(validators.large_straight_validator(dices)["playable"])

    def test_full_house_validator(self):
        dices = [6, 6, 6, 5, 5]
        self.assertTrue(validators.full_house_validator(dices)["playable"])
        dices = [5, 5, 5, 6, 6]
        self.assertTrue(validators.full_house_validator(dices)["playable"])
        dices = [1, 2, 3, 3, 3]
        self.assertFalse(validators.full_house_validator(dices)["playable"])
        dices = [2, 2, 3, 4, 5]
        self.assertFalse(validators.full_house_validator(dices)["playable"])
        dices = [1, 1, 1, 1, 1]
        self.assertFalse(validators.full_house_validator(dices)["playable"])

    def test_chance_validator(self):
        dices = [1, 5, 3, 4, 5]
        self.assertTrue(validators.chance_validator(dices)["playable"])
        dices = [1, 2, 2, 3, 4]
        self.assertTrue(validators.chance_validator(dices)["playable"])

    if __name__ == '__main__':
        unittest.main()
