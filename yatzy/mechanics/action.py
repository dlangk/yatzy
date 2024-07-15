class Action:

    def __init__(self,
                 score: bool,
                 forced: bool = False,
                 locked_dices=None,
                 scored_combination=None):
        self.score = score
        self.forced = forced
        self.locked_dices = locked_dices
        self.scored_combination = scored_combination
