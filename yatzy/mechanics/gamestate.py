from yatzy.mechanics.const import combinations


class GameState:
    def __init__(self, rolls, dices, scorecard=None):
        self.rolls = rolls
        self.dices = dices

        # Either build a scorecard or accept one as input
        if not scorecard:
            self.scorecard = {}
            for c in combinations:
                self.scorecard[c] = {
                    "played": False,
                    "score": None
                }
        else:
            self.scorecard = scorecard
