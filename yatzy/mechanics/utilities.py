from yatzy.mechanics import const
from yatzy.mechanics.action import Action
from yatzy.mechanics.gamestate import GameState
import yatzy.mechanics.gameengine as Engine


def serialize_action(action: Action):
    vector = [int(action.score)]
    for lock in action.locked_dices:
        vector.append(lock)
    for combination in const.combinations:
        if action.scored_combination == combination:
            vector.append(1)
        else:
            vector.append(0)
    return vector


def serialize_state(state: GameState):
    vector = [state.rolls]
    for die in state.dices:
        vector.append(die)
    for combination in const.combinations:
        vector.append(int(state.scorecard.scorecard[combination]["played"]))
        vector.append(state.scorecard.scorecard[combination]["score"]
                      if state.scorecard.scorecard[combination]["score"] else 0)
    return vector


def print_game_state(state: GameState):
    unplayed_combinations = state.scorecard.get_unplayed_combinations()
    legal_combinations = Engine.get_legal_combinations(state)
    playable_combinations = [c for c in unplayed_combinations if c in legal_combinations]

    print("###########################")
    total_score = 0
    for c in const.combinations:
        total_score += state.scorecard.scorecard[c]["score"] if state.scorecard.scorecard[c]["score"] else 0

        if state.scorecard.scorecard[c]["played"]:
            score = state.scorecard.scorecard[c]["score"]
            if score > 0:
                print("{:<17}".format(c) + str(score))
        else:
            if c in playable_combinations:
                print("{:<17}".format(c) + "[] -> Playable")
            else:
                print("{:<17}".format(c) + "[]")
    print("---------------------------")
    print("{:<17}".format("total_score") + str(total_score))
    print("---------------------------")
    print("rolls: ", state.rolls)
    print("dices: ", state.dices)
    print("###########################")
