import json
import time

from typing import Type

from yatzy.players.player import Player
from yatzy.mechanics import gameengine as Engine
from yatzy.mechanics.action import Action
from yatzy.mechanics.gamestate import GameState

start_time = time.time()


def evaluate(player: Type[Player], eval_runs: int):
    result = {
        "final_scores": [],
        "times": [],
        "player": player.__name__,
        "eval_runs": eval_runs,
    }
    runs = 0

    while runs < eval_runs:
        runs += 1
        state: GameState = Engine.create_initial_state()
        action: Action = Action(False)

        while not Engine.game_over(state):
            valid_action = False
            while not valid_action:
                action: Action = player.get_action(state)
                valid_action = Engine.validate_action(state, action)

            state = Engine.step(state, action)

        result["final_scores"].append(Engine.final_score(state))
        result["times"].append(time.time() - start_time)

    total_time = (time.time() - start_time)
    result["time_total"] = round(total_time * 1000, 4)
    result["time_average"] = round((total_time / runs) * 1000, 4)

    result["score_max"] = max(result["final_scores"])
    result["score_min"] = min(result["final_scores"])
    result["score_average"] = sum_all_values_in_key(result) / len(result["final_scores"])
    result["score_median"] = sorted(result["final_scores"])[len(result["final_scores"]) // 2]
    result["standard_deviation"] = standard_deviation_of_values_in_key(result)

    # remove times and final_scores from result
    del result["times"]
    del result["final_scores"]

    print(json.dumps(result, indent=4))


def standard_deviation_of_values_in_key(result):
    average = average_all_values_in_key(result)
    sum = 0
    for value in result["final_scores"]:
        sum += (value - average) ** 2
    return (sum / len(result["final_scores"])) ** 0.5


def sum_all_values_in_key(result):
    sum = 0
    for value in result["final_scores"]:
        sum += value
    return sum


def average_all_values_in_key(result):
    return sum_all_values_in_key(result) / len(result["final_scores"])
