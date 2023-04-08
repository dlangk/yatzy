from yatzy.players.player import Player
from yatzy.mechanics.action import Action
from yatzy.mechanics.gamestate import GameState

import yatzy.mechanics.const as const


class Daniel(Player):

    def __init__(self):
        pass

    def get_action(self, state: GameState, playable_combinations) -> Action:
        locked = None
        dices = state.dices
        completed = upper_completed(state)
        if not completed:
            best_upper = find_best_upper(dices)
            locked = lock_die_value(const.combinations[best_upper]['required_die_value'], dices)
        return Action(False, locked)


def lock_die_value(die_value, dices):
    locked = []
    for d in dices:
        if d == die_value:
            locked.append(1)
        else:
            locked.append(0)
    return locked


def find_best_upper(dices):
    max_dices = 0
    best_combination = None
    for c in const.upper_combinations:
        count = dices.count(const.combinations[c]['required_die_value'])
        if count >= max_dices:
            max_dices = count
            best_combination = c
    return best_combination


def upper_completed(state: GameState):
    for c in const.upper_combinations:
        if not state.scorecard[c]['played']:
            return False
    return True
