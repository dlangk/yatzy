import random

from yatzy.mechanics.action import Action
from yatzy.mechanics.gamestate import GameState


class ActorRandom:

    def __init__(self):
        pass

    @staticmethod
    def get_action(state: GameState, playable_combinations) -> Action:
        playable_combinations = list(playable_combinations.keys())
        score = random.randint(0, 1)
        if state.rolls > 2:
            score = True
        locked_dices = [random.randint(0, 1) for x in range(0, 5)]
        action = Action(score, locked_dices, random.choice(playable_combinations))
        return action
