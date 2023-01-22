from yatzy.mechanics import const
from yatzy.mechanics.action import Action
from yatzy.mechanics.gamestate import GameState


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
    for combination in state.scorecard:
        vector.append(int(state.scorecard[combination]["played"]))
        vector.append(state.scorecard[combination]["score"]
                      if state.scorecard[combination]["score"] else 0)
    return vector


def print_game_state(state: GameState, playable_combinations):
    print("###########################")
    total_score = 0

    for c in state.scorecard:
        total_score += state.scorecard[c]["score"] if state.scorecard[c]["score"] else 0

        if state.scorecard[c]["played"]:
            print("{:<17}".format(c) + str(state.scorecard[c]["score"]))
        else:
            if c in playable_combinations:
                print("{:<17}".format(c) + "[] -> " + str(playable_combinations[c]))
            else:
                print("{:<17}".format(c) + "[]")
    print("---------------------------")
    print("{:<17}".format("total_score") + str(total_score))
    print("---------------------------")
    print("rolls: ", state.rolls)
    print("dices: ", state.dices)
    print("###########################")
