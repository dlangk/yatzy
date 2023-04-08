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
        "average_score": None,
        "average_time": None,
    }
    runs = 0

    while runs < eval_runs:
        runs += 1
        state: GameState = Engine.create_initial_state()
        action: Action = Action(False)

        while not Engine.game_over(state):
            playable_combinations = Engine.score_combinations(state)

            valid_action = False
            while not valid_action:
                action: Action = player.get_action(state, playable_combinations)
                valid_action = Engine.validate_action(state, action, playable_combinations)

            state = Engine.step(state, action, playable_combinations)

        result["final_scores"].append(Engine.final_score(state))
        result["times"].append(time.time() - start_time)

    total_time = (time.time() - start_time)
    result["average_time"] = (total_time / runs) * 1000
    result["average_score"] = sum_all_values_in_key(result) / len(result["final_scores"])
    result["standard_deviation"] = standard_deviation_of_values_in_key(result)
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
