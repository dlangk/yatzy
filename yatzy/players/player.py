from abc import abstractmethod

from yatzy.mechanics.action import Action
from yatzy.mechanics.gamestate import GameState


class Player:

    @staticmethod
    @abstractmethod
    def get_action(state: GameState, playable_combinations) -> Action:
        pass
