from logger import YatzyLogger


class State:
    def __init__(self, rolls, dices, combinations):
        self.logger = YatzyLogger(__name__).get_logger()
        self.logger.debug("creating state")
        self.rolls = rolls
        self.dices = dices
        self.scorecard = {}
        for c in combinations:
            self.scorecard[c] = {
                "allowed": True,
                "score": None
            }

    def __str__(self):
        return f"rolls: {self.rolls} dices: {self.dices} score: {self.get_score()}"

    def get_simple_scorecard(self):
        return [(c, self.scorecard[c]["score"]) for c in self.scorecard.keys()]

    def get_index_of_remaining_options(self):
        remaining = []
        for ix, c in enumerate(self.scorecard):
            if self.scorecard[c]["allowed"]:
                remaining.append(ix)
        return remaining

    def get_score(self):
        return sum([self.scorecard[c]["score"] if self.scorecard[c]["score"] else 0 for c in self.scorecard])

    def enter_score(self, combination, score):
        self.scorecard[combination]["allowed"] = False
        self.scorecard[combination]["score"] = score

    def get_allowed_vector(self):
        return [int(self.scorecard[c]["allowed"] == True) for c in self.scorecard]

    def reset_rolls(self):
        self.rolls = 1
