from yatzy.mechanics import const
from yatzy.mechanics.const import combinations


class Scorecard:
    def __init__(self):
        self.scorecard = None
        self.create_scorecard()

    def get_unplayed_combinations(self):
        playable_combinations = []
        for c in const.combinations:
            if not self.scorecard[c]["played"]:
                playable_combinations.append(c)
        return playable_combinations

    def get_upper_score(self):
        upper_score = 0
        for c in const.combinations:
            if const.combinations[c]["upper_section"]:
                upper_score += self.scorecard[c]["score"]
        return upper_score

    def get_bonus(self):
        upper_score = self.get_upper_score()
        return 50 if upper_score >= 63 else 0

    def get_final_score(self):
        final_score = 0
        for c in const.combinations:
            final_score += self.get_score(c)
        final_score += self.get_bonus()
        return final_score

    def get_score(self, combination) -> int:
        return self.scorecard[combination]["score"]

    def is_played(self, combination) -> bool:
        return self.scorecard[combination]["played"]

    def score(self, combination, score):
        if "pair_" in combination:
            for c in const.combinations:
                if "pair_" in c:
                    if c == combination:
                        self.scorecard[c]["played"] = True
                        self.scorecard[c]["score"] = score
                    else:
                        self.scorecard[c]["played"] = True
                        self.scorecard[c]["score"] = 0
        self.scorecard[combination]["played"] = True
        self.scorecard[combination]["score"] = score

    def create_scorecard(self):
        self.scorecard = {}
        for c in combinations:
            self.scorecard[c] = {
                "played": False,
                "score": None
            }

    def game_over(self):
        for c in self.scorecard:
            if not self.scorecard[c]["played"]:
                return False
        return True
