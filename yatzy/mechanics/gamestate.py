from yatzy.mechanics.scorecard import Scorecard
from copy import deepcopy


class GameState:
    def __init__(self, rolls, dices, scorecard: Scorecard = None):
        self.rolls = rolls
        self.dices = dices
        self.scorecard = scorecard if scorecard else Scorecard()

    def set_dices(self, dices):
        self.dices = dices

    def copy(self):
        return deepcopy(self)
