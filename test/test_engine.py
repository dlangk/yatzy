from yatzy import engine
import unittest


class TestDiceMethods(unittest.TestCase):

    def test_roll(self):
        self.assertEqual(len(engine.roll_dice(5)), 5)


if __name__ == '__main__':
    unittest.main()
