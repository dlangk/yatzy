import random
import yatzy.mechanics.gameengine as Engine

from yatzy.mechanics.action import Action
from yatzy.mechanics.gamestate import GameState
from yatzy.players.player import Player


class Random(Player):

    @staticmethod
    def get_action(state: GameState) -> Action:
        unplayed_combinations = state.scorecard.get_unplayed_combinations()
        legal_combinations = Engine.get_legal_combinations(state)
        playable_combinations = [c for c in unplayed_combinations if c in legal_combinations]

        locked_dices = [random.randint(0, 1) for x in range(0, 5)]
        score = random.choice([True, False])
        if state.rolls > 2:
            score = True

        if len(playable_combinations) > 0:
            scored_combination = random.choice(playable_combinations)
            forced = False
            return Action(score, forced, locked_dices, scored_combination)
        else:
            scored_combination = random.choice(unplayed_combinations)
            forced = True
            return Action(score, forced, locked_dices, scored_combination)
