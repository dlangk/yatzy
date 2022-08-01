from logger import YatzyLogger


class PlayerAction:
    def __init__(self, reroll, locked_dices, scoring_vector, zero_score=None):
        self.logger = YatzyLogger(__name__).get_logger()
        self.reroll = reroll
        self.locked_dices = locked_dices
        self.zero_score = zero_score
        self.scoring_vector = scoring_vector

    def __str__(self):
        return f"reroll: {self.reroll} locked_dices: {self.locked_dices} scoring_vector: {self.scoring_vector}"
