import random

import const
from action import Action
from engine import Engine
from engine import State
from logger import YatzyLogger


class Agent:
    def __init__(self):
        self.logger = YatzyLogger(__name__).get_logger()

    def act(self, engine: Engine, state: State):
        self.logger.debug("agent acting")
        reroll = round(random.uniform(0, 1), 2)
        locked_dices = [round(random.uniform(0, 1), 2) for n in range(const.DICES_COUNT)]
        scoring_vector = [round(random.uniform(0, 1), 2) for n in range(const.COMBINATIONS_COUNT)]
        suggested_action = Action(reroll, locked_dices, scoring_vector)
        legal_action = engine.make_action_legal(suggested_action, state)
        return legal_action

    def cache(self, state, next_state, action, reward, done):
        pass

    def learn(self):
        pass
