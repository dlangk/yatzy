from math import floor
from random import random

import yatzy.mechanics.const as const
from yatzy.mechanics.action import Action
from yatzy.mechanics.gamestate import GameState


def dice_roll():
    return floor(6 * random()) + 1


def create_initial_state() -> GameState:
    dices = [dice_roll() for n in range(const.DICES_COUNT)]
    return GameState(rolls=1, dices=dices)


def roll_dices():
    return [dice_roll() for n in range(const.DICES_COUNT)]


def game_over(state: GameState) -> bool:
    return state.scorecard.game_over()


def get_legal_combinations(state: GameState):
    legal_combinations = {}
    for combination in const.combinations:
        combs = const.combinations[combination]
        required_die_value = combs["required_die_value"]
        required_die_count = combs["required_die_count"]
        legal_combinations[combination] = combs["validator"](dices=state.dices,
                                                             die_value=required_die_value,
                                                             die_count=required_die_count)
    # filter list based on valid = true
    legal_combinations = {k: v for k, v in legal_combinations.items() if v}
    return legal_combinations


def validate_action(state: GameState, action: Action):
    if state.rolls > 2:
        if not action.score:
            print("Too many rolls! You have to score.")
            return False
    if action.score:
        playable_combinations = state.scorecard.get_unplayed_combinations()
        if action.scored_combination not in playable_combinations:
            print(action.scored_combination + "has already been played!")
            return False
    return True


def reroll_dices(dices, locked_dices):
    new_dices = dices.copy()
    for ix, dice_lock in enumerate(locked_dices):
        if dice_lock == 0:
            new_dices[ix] = dice_roll()
    return new_dices


def step(state: GameState, action: Action) -> GameState:
    rolls = state.rolls
    dices = state.dices

    if not action.score:  # Either we roll some dices without scoring
        dices = reroll_dices(dices, action.locked_dices)
        return GameState(rolls=rolls + 1, dices=dices, scorecard=state.scorecard)
    else:  # or we score, in which case we roll all dices and reset rolls counter
        new_scorecard = update_scorecard(state, action)
        dices = [dice_roll() for n in range(const.DICES_COUNT)]
        return GameState(rolls=1, dices=dices, scorecard=new_scorecard)


def get_score(state, scored_combination):
    scorer_function_ = const.combinations[scored_combination]["scorer"]

    return scorer_function_(dices=state.dices,
                            die_value=const.combinations[scored_combination]["required_die_value"],
                            die_count=const.combinations[scored_combination]["required_die_count"])


# this method returns a new state, not a modified one
def update_scorecard(state: GameState, action: Action):
    old_state = state
    new_state = state.copy()
    if action.scored_combination in old_state.scorecard.get_unplayed_combinations():
        if action.forced:
            new_state.scorecard.score(action.scored_combination, 0)
            return new_state
        else:
            legal_combinations = get_legal_combinations(state)
            if action.scored_combination not in legal_combinations:
                new_state.scorecard.score(action.scored_combination, 0)
                return new_state
            score = get_score(state, action.scored_combination)
            new_state.scorecard.score(action.scored_combination, score)
            return new_state
    else:
        raise Exception("You tried to score an invalid combination")


def final_score(state: GameState):
    return state.scorecard.get_final_score()
