import random
import json
from collections import Counter

def roll_dice(dices):
    return [random.randint(1,6) for n in range(dices)]

def upper_section_validator(dices, die_value, n):
    # Any set of dices are valid for the upper section
    return (True, [die_value])

def n_kind_validator(dices, die_value, n):
    # check if there are n of any number
    # returns Boolean, [list of options]
    valid = False
    options = []
    die_count = Counter(dices)
    for die in die_count:
        if die_count[die] >= n:
            valid = True
            options.append(die)
    return (valid, options)

def two_pair_validator(dices, die_value, n):
    pairs = 0
    dies = []
    die_count = Counter(dices)
    for die in die_count:
        if die_count[die] >= 2:
            pairs += 1
            dies.append(die)
    if pairs > 1:
        return (True, [])
    return (False, [])

def two_pair_scoring(dices, die_value = None, n = None):
    score = 0
    die_count = Counter(dices)
    for die in die_count:
        if die_count[die] >= 2:
            score += (die * 2)
    return score

def small_straight_validator(dices, die_value = None, n = None):
    return (set([1,2,3,4,5]).issubset(dices), [])

def large_straight_validator(dices, die_value = None, n = None):
    return (set([2,3,4,5,6]).issubset(dices), [])

def full_house_validator(dices, die_value = None, n = None):
    die_count = sorted(Counter(dices).items(), key=lambda item: item[1])
    if len(die_count) > 1:
        if die_count[-1][1] == 3 and die_count[-2][1] == 2:
            return (True, [])
    return (False, [])

def chance_validator(dices, die_value = None, n = None):
    return (True, [])

def upper_section_scoring(dices, die_value, n):
    # only die of the target value are counted
    score = 0
    for die in dices:
        if die == die_value:
            score += die
    return score

def n_kind_scoring(dices, die_value = None, n = None):
    score = 0
    for die in dices:
       if die == die_value:
           score += die
    return score

def yatzy_scoring(dices, die_value = None, n = None):
    return 50

def all_dice_scoring(dices, die_value = None, n = None):
    return sum(dices)

combinations = {
        "aces": {
            "validator": upper_section_validator,
            "die_value": 1,
            "scoring": upper_section_scoring,
            "n": None
        },
        "twos": {
            "validator": upper_section_validator,
            "die_value": 2,
            "scoring": upper_section_scoring,
            "n": None
        },
        "threes": {
            "validator": upper_section_validator,
            "die_value": 3,
            "scoring": upper_section_scoring,
            "n": None
        },
        "fours": {
            "validator": upper_section_validator,
            "die_value": 4,
            "scoring": upper_section_scoring,
            "n": None
            },
        "fives": {
            "validator": upper_section_validator,
            "die_value": 5,
            "scoring": upper_section_scoring,
            "n": None
            },
        "sixes": {
            "validator": upper_section_validator,
            "die_value": 6,
            "scoring": upper_section_scoring,
            "n": None
            },
        "pair": {
            "validator": n_kind_validator,
            "die_value": None,
            "scoring": n_kind_scoring,
            "n": 2
            },
        "two_pair": {
            "validator": two_pair_validator,
            "die_value": None,
            "scoring": two_pair_scoring,
            "n": None
            },
        "three_of_a_kind": {
            "validator": n_kind_validator,
            "die_value": None,
            "scoring": n_kind_scoring,
            "n": 3
            },
        "four_of_a_kind": {
                "validator": n_kind_validator,
                "die_value": None,
                "scoring": n_kind_scoring,
                "n": 4
            },
        "small_straight": {
                "validator": small_straight_validator,
                "die_value": None,
                "scoring": all_dice_scoring,
                "n": None
                },
        "large_straight": {
                "validator": large_straight_validator,
                "die_value": None,
                "scoring": all_dice_scoring,
                "n": None
                },
        "full_house": {
                "validator": full_house_validator,
                "die_value": None,
                "scoring": all_dice_scoring,
                "n": None,
                },
        "chance": {
                "validator": chance_validator,
                "die_value": None,
                "scoring": all_dice_scoring,
                "n": None,
                },
        "yatzy": {
                "validator": n_kind_validator,
                "die_value": None,
                "scoring": yatzy_scoring,
                "n": 5
                }
        }

def get_options(dices):
    options = []
    for c in combinations:
        comb = combinations[c]
        comb_name = c
        valid = comb["validator"](dices, comb["die_value"], comb["n"])
        if valid[0]:
            if len(valid[1]) == 0:
                score = comb["scoring"](dices, None, comb["n"])
                option = {
                        "name": comb_name,
                        "die_value": comb["die_value"],
                        "score": score
                        }
                options.append(option)
            for die_value in valid[1]:
                score = comb["scoring"](dices, die_value, comb["n"])
                option = {
                        "name": comb_name,
                        "die_value": die_value,
                        "score": score
                        }
                options.append(option)
    return options

def initialize_game(players):
    state = {}
    state["players"] = players
    state["active_player"] = players[0]
    state["active_player_roll"] = 0
    state["dice_state"] = None
    for player in players:
        state["scorecard"] = {}
        state["scorecard"][player] = {}
        if player == "players":
            raise Exception("name 'players' not allowed")
        for c in combinations:
            state["scorecard"][player][c] = {
                    "played": False,
                    "score": None
                    }
    return state

def game_finished(state):
    for player in state["players"]:
        for c in combinations:
            if not state["scorecard"][player][c]["played"]:
                return False
    return True

def get_dice_state(state):
    return state["dice_state"]


def get_active_player(state):
    return state["active_player"]

def get_active_player_roll(state):
    return state["active_player_roll"]

def start_turn(state):
    dices = roll_dice(5)
    state["dice_state"] = dices
    state["active_player_roll"] = 1
    return state

def print_state(state):
    print("\n")
    print("game finished?", game_finished(state))
    print("active player", get_active_player(state))
    print("dice state", get_dice_state(state))
    print("active_player_roll", get_active_player_roll(state))
