class Action:

    def __init__(self,
                 score: bool,
                 locked_dices=None,
                 scored_combination=None):
        self.score = score
        self.locked_dices = locked_dices
        self.scored_combination = scored_combination
