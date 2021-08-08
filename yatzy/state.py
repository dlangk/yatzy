import torch

import numpy as np

from logger import YatzyLogger
from utils import upper_section


class State:
    def __init__(self, rolls, dices, combinations):
        self.logger = YatzyLogger(__name__).get_logger()
        self.logger.debug("creating state")
        self.rolls = rolls
        self.dices = dices
        self.scorecard = {}
        self.use_cuda = torch.cuda.is_available()
        for c in combinations:
            self.scorecard[c] = {
                "allowed": True,
                "score": None
            }

    def __str__(self):
        return f"rolls: {self.rolls} dices: {self.dices} score: {self.get_score()}"

    def get_tensor(self):
        rolls = round(int(self.rolls))
        allowed = [int(self.scorecard[c]["allowed"] == True) for c in self.scorecard]
        scored = [int(self.scorecard[c]["score"]) if self.scorecard[c]["score"] else int(0) for c in self.scorecard]

        state_tensor_np = np.append(np.append(np.array(rolls), np.array(allowed)), np.array(scored))
        state_tensor_torch = torch.Tensor(state_tensor_np)

        if self.use_cuda:
            state_tensor = torch.FloatTensor(state_tensor_torch).cuda()
        else:
            state_tensor = torch.FloatTensor(state_tensor_torch)

        return state_tensor

    def get_simple_scorecard(self):
        return [(c, self.scorecard[c]["score"]) for c in self.scorecard.keys()]

    def get_uppersection_score(self):
        score = 0
        for c in self.scorecard:
            if c in upper_section:
                score += self.scorecard[c]["score"]
        return score

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
