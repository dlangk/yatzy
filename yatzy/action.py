from logger import YatzyLogger
import torch

import numpy as np


class Action:
    def __init__(self, reroll, locked_dices, scoring_vector, zero_score=None):
        self.logger = YatzyLogger(__name__).get_logger()
        self.reroll = reroll
        self.locked_dices = locked_dices
        self.zero_score = zero_score
        self.scoring_vector = scoring_vector
        self.use_cuda = torch.cuda.is_available()

    def get_tensor(self) -> torch.Tensor:
        rerolls = [int(self.reroll)]
        if self.reroll > 3:
            self.logger.error(f"rerolls too large: {rerolls}")
        locked_dices = self.locked_dices
        zero_score = [int(self.zero_score == True)]
        scoring_vector = self.scoring_vector

        action_tensor_np = np.append(np.append(np.append(np.array(rerolls),
                                                         np.array(locked_dices)),
                                               np.array(zero_score)),
                                     np.array(scoring_vector))

        action_tensor_torch = torch.Tensor(action_tensor_np).type(torch.LongTensor)

        if self.use_cuda:
            action_tensor = torch.LongTensor(action_tensor_torch).cuda()
        else:
            action_tensor = torch.LongTensor(action_tensor_torch)

        return action_tensor

    def __str__(self):
        return f"reroll: {self.reroll} locked_dices: {self.locked_dices} scoring_vector: {self.scoring_vector}"
