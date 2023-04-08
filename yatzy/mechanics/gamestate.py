from yatzy.mechanics.scorecard import Scorecard


class GameState:
    def __init__(self, rolls, dices, scorecard: Scorecard = None):
        self.rolls = rolls
        self.dices = dices
        self.scorecard = scorecard if scorecard else Scorecard()
