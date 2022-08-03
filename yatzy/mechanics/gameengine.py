from random import random
from math import floor

from yatzy.mechanics.gamestate import GameState
from yatzy.mechanics.action import Action

import yatzy.mechanics.const as const


def dice_roll():
    return floor(6 * random()) + 1


def create_initial_state() -> GameState:
    dices = [dice_roll() for n in range(const.DICES_COUNT)]
    return GameState(rolls=1, dices=dices)


def game_over(state: GameState) -> bool:
    for c in state.scorecard:
        if not state.scorecard[c]["played"]:
            return False
    return True


def get_legal_combinations(state: GameState):
    legal_combinations = {}
    for combination in const.combinations:
        combs = const.combinations[combination]
        required_die_value = combs["required_die_value"]
        required_die_count = combs["required_die_count"]
        legal_combinations[combination] = combs["validator"](dices=state.dices,
                                                             die_value=required_die_value,
                                                             die_count=required_die_count)

    return legal_combinations


def get_non_zero_combinations(state: GameState, legal_combinations):
    playable = {}
    for combination in state.scorecard:
        if legal_combinations[combination]["playable"] and not state.scorecard[combination]['played']:
            if len(legal_combinations[combination]["die"]) > 0:
                for die in legal_combinations[combination]["die"]:
                    playable[combination] = die
            else:
                playable[combination] = None
    return playable


def score_combinations(state: GameState):
    legal_combinations = get_legal_combinations(state)
    non_zero_combinations = get_non_zero_combinations(state, legal_combinations)
    scored_combinations = {}
    for combination in const.combinations:
        if combination in non_zero_combinations:
            score = const.combinations[combination]["scorer"](dices=state.dices,
                                                              die_value=non_zero_combinations[combination],
                                                              die_count=const.combinations[combination][
                                                                  "required_die_count"])
            scored_combinations[combination] = score
        else:
            if not state.scorecard[combination]['played']:
                scored_combinations[combination] = 0
    return scored_combinations


def validate_action(state: GameState, action: Action, playable_combinations):
    if state.rolls > 2:
        if not action.score:
            print("Too many rolls! You have to score.")
            return False
    if action.score:
        if action.scored_combination not in playable_combinations:
            print("<< " + action.scored_combination + " >> is not playable")
            return False
    return True


def step(state: GameState, action: Action, playable_combinations) -> GameState:
    rolls = state.rolls
    dices = state.dices

    if not action.score:  # Either we roll some dices without scoring
        for ix, dice_lock in enumerate(action.locked_dices):
            if dice_lock == 0:
                dices[ix] = dice_roll()
        return GameState(rolls=rolls + 1, dices=dices, scorecard=state.scorecard)
    else:  # or we score, in which case we roll all dices and reset rolls counter
        new_scorecard = apply_score(state.scorecard,
                                    action.scored_combination,
                                    playable_combinations)
        dices = [dice_roll() for n in range(const.DICES_COUNT)]
        return GameState(rolls=1, dices=dices, scorecard=new_scorecard)


def apply_score(scorecard, scored_combination, scored_combinations):
    scorecard[scored_combination]["played"] = True

    # It sucks, but we have to treat pairs special...
    if "pair_" in scored_combination:
        for combination in scorecard:
            if "pair_" in combination:
                scorecard[combination]["played"] = True
                scorecard[combination]["score"] = 0  # next operation will override this

    scorecard[scored_combination]["score"] = scored_combinations[scored_combination]
    return scorecard


def final_score(state: GameState):
    final_score = 0
    upper_score = 0
    for c in state.scorecard:
        final_score += state.scorecard[c]["score"]
        if const.combinations[c]["upper_section"]:
            upper_score += state.scorecard[c]["score"]
    if upper_score > 63:
        final_score += 50
    return final_score
