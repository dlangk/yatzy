import unittest

import yatzy.graph.build_graph as build_graph


class TestRerollMasks(unittest.TestCase):
    def test_reroll_masks_aabcd(self):
        dices = [1, 1, 2, 3, 4]
        reroll_filter = build_graph.get_reroll_filter(dices)
        self.assertTrue(reroll_filter == "aabcd")
        reroll_masks = [list(mask) for mask in build_graph.generate_reroll_mask(5, reroll_filter)]
        test_1 = [1, 0, 0, 0, 0]
        test_2 = [0, 1, 0, 0, 0]
        both = (test_1 in reroll_masks) and (test_2 in reroll_masks)
        # Make sure we only have one way to keep the 1s (either [1, 0, ...] or [0, 1, ...])
        self.assertFalse(both)

    def test_reroll_masks_aaabc(self):
        dices = [1, 1, 1, 2, 3]
        reroll_filter = build_graph.get_reroll_filter(dices)
        self.assertFalse(reroll_filter == "aabcd")
        self.assertTrue(reroll_filter == "aaabc")

        reroll_masks = [list(mask) for mask in build_graph.generate_reroll_mask(5, reroll_filter)]

        test_1 = [1, 0, 0, 0, 0]
        test_2 = [0, 0, 1, 0, 0]
        test_3 = [0, 1, 0, 0, 0]
        test_4 = [1, 0, 0, 0, 0]
        both_1 = (test_1 in reroll_masks) and (test_2 in reroll_masks)
        both_2 = (test_1 in reroll_masks) and (test_3 in reroll_masks)
        both_3 = (test_3 in reroll_masks) and (test_4 in reroll_masks)

        self.assertFalse(both_1)
        self.assertFalse(both_2)
        self.assertFalse(both_3)


if __name__ == '__main__':
    unittest.main()
