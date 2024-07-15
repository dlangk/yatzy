import math
from random import random
from fractions import Fraction

import yatzy.mechanics.const as const
import yatzy.mechanics.gameengine as Engine


def combinations(n, r):
    # C(n,r) = n!/(r!(n-r)!)
    # C(n,r) is the number of combinations;
    # n is the total number of elements in the set; and
    # r is the number of elements you choose from this set.
    return math.factorial(n) / (math.factorial(r) * (math.factorial(n - r)))


def binomial(num_dice, num_successes, p_success):
    # P(X=r) = nCr * pʳ * (1-p)ⁿ⁻ʳ
    # r = number of successes
    # nCr = number of combinations (n choose r)
    # p = number of sides on the die (e.g. 1/6)
    return combinations(num_dice, num_successes) * \
           math.pow(p_success, num_successes) * \
           math.pow(1 - p_success, num_dice - num_successes)


def get_score_distribution(target_combination, dices, locked):
    scores = []
    valid_dices = []
    invalid_dices = []
    total_score = 0
    tries = 100000

    for i in range(0, tries):
        dices = Engine.reroll_dices(dices, locked)
        result = const.combinations[target_combination]["validator"](
            dices=dices,
            die_value=const.combinations[target_combination]["required_die_value"],
            die_count=const.combinations[target_combination]["required_die_count"])

        if result["playable"]:
            score = const.combinations[target_combination]["scorer"](
                dices=dices,
                die_value=const.combinations[target_combination]["required_die_value"],
                die_count=const.combinations[target_combination]["required_die_count"])
            total_score += score
            scores.append(score)
            dices_string = [','.join(str(x) for x in dices)]
            valid_dices += dices_string
        else:
            dices_string = [','.join(str(x) for x in dices)]
            invalid_dices += dices_string
            total_score += 0
            scores.append(0)

    maximum_score = max(scores)
    minimum_score = min(scores)
    return maximum_score, minimum_score


def get_all_possible_dices():
    dices = []
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            for d3 in range(1, 7):
                for d4 in range(1, 7):
                    for d5 in range(1, 7):
                        array_of_numbers = [d1, d2, d3, d4, d5]
                        dices.append(sorted(array_of_numbers, key=abs, reverse=True))
    return dices


def get_bag_of_dices():
    set_of_dices = set(tuple(i) for i in get_all_possible_dices())
    list_of_dices = [list(elem) for elem in set_of_dices]
    return list_of_dices


def get_all_possible_locks():
    return [[1, 0, 1, 1, 0], [0, 0, 0, 0, 1], [1, 0, 1, 0, 0], [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [1, 1, 1, 0, 1], [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0], [0, 1, 1, 0, 1], [1, 0, 1, 1, 1], [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 1],
            [1, 1, 1, 0, 0], [1, 0, 0, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 0, 0],
            [1, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 1, 1, 1], [1, 0, 0, 0, 1],
            [1, 1, 0, 1, 1], [0, 0, 1, 1, 0], [1, 0, 1, 0, 1], [0, 1, 1, 1, 1],
            [1, 1, 0, 0, 1], [0, 1, 0, 0, 1], [0, 1, 0, 1, 1], [0, 0, 0, 1, 1]]


def get_expected_value(target_combination, dices, locked):
    valid_dices = []
    invalid_dices = []
    total_score = 0
    max_score = 0
    max_count = 0
    tries = 10000

    for i in range(0, tries):
        new_dices = Engine.reroll_dices(dices, locked)
        result = const.combinations[target_combination]["validator"](
            dices=new_dices,
            die_value=const.combinations[target_combination]["required_die_value"],
            die_count=const.combinations[target_combination]["required_die_count"])

        if result["playable"]:
            score = const.combinations[target_combination]["scorer"](
                dices=new_dices,
                die_value=const.combinations[target_combination]["required_die_value"],
                die_count=const.combinations[target_combination]["required_die_count"])
            if score >= max_score:
                if score > max_score:
                    max_count = 0
                max_score = score
                max_count += 1
            total_score += score
            dices_string = [','.join(str(x) for x in new_dices)]
            valid_dices += dices_string
        else:
            dices_string = [','.join(str(x) for x in new_dices)]
            invalid_dices += dices_string
            total_score += 0

    unique_valid_rolls = len(set(valid_dices))
    unique_invalid_rolls = len(set(invalid_dices))
    probability = round(unique_valid_rolls / (unique_valid_rolls + unique_invalid_rolls), 3)
    p_max_score = round(max_count / tries, int(math.log10(tries)) + 1)
    expected_value = round(total_score / tries, 3)
    # print("{:<20}".format(str(target_combination) + ": ") +
    #       "{:<7}".format("P=" + str(probability) + " ") +
    #       "{:<7}".format("E=" + str(expected_value) + " "))
    return probability, expected_value, max_score, p_max_score


def print_expected_value(combination, expected_value):
    print("{:<20}".format("E[" + combination + "]:") + "{:<8}".format(str(round(expected_value, 4))))


def print_probability(combination, probability, unique_valid_rolls, unique_invalid_rolls):
    frac = str(Fraction(unique_valid_rolls, unique_valid_rolls + unique_invalid_rolls))
    print("{:<20}".format("P(" + combination + "):") + "{:<8}".format(str(round(probability, 4))) + str(frac))


def get_probability(target_combination, dices, locked):
    valid_dices = []
    invalid_dices = []

    tries = 100000
    for i in range(0, tries):
        dices = Engine.reroll_dices(dices, locked)
        result = const.combinations[target_combination]["validator"](
            dices=dices,
            die_value=const.combinations[target_combination]["required_die_value"],
            die_count=const.combinations[target_combination]["required_die_count"])

        if result["playable"]:
            dices_string = [','.join(str(x) for x in dices)]
            valid_dices += dices_string
        else:
            dices_string = [','.join(str(x) for x in dices)]
            invalid_dices += dices_string

    unique_valid_rolls = len(set(valid_dices))
    unique_invalid_rolls = len(set(invalid_dices))
    probability = unique_valid_rolls / (unique_valid_rolls + unique_invalid_rolls)
    print_probability(target_combination, probability, unique_valid_rolls, unique_invalid_rolls)
    return probability


def dice_stats():
    hashes = []
    tries = 10000000
    for i in range(0, tries):
        dices = sorted([Engine.dice_roll() for n in range(5)])
        hash = str(''.join(str(d) for d in dices))
        hashes.append(hash)
    print(hashes)
    print(set(hashes))
    print(len(set(hashes)))


def search_valid_dices(combination):
    valid_dices = []
    invalid_dices = []
    tries = 100000
    for i in range(0, tries):
        dices = [Engine.dice_roll() for n in range(5)]
        result = const.combinations[combination]["validator"] \
            (dices=dices,
             die_value=const.combinations[combination]["required_die_value"],
             die_count=const.combinations[combination]["required_die_count"])

        if result["playable"]:
            dices_string = [','.join(str(x) for x in dices)]
            valid_dices += dices_string
        else:
            dices_string = [','.join(str(x) for x in dices)]
            invalid_dices += dices_string

    print_probability(combination, valid_dices, invalid_dices)


def estimate_probability(combination, dices, locked):
    success = 0
    tries = 1000000

    for i in range(0, tries):
        for ix, dice_lock in enumerate(locked):
            if dice_lock == 0:
                dices[ix] = Engine.dice_roll()

        result = const.combinations[combination]["validator"] \
            (dices=dices,
             die_value=const.combinations[combination]["required_die_value"],
             die_count=const.combinations[combination]["required_die_count"])

        if result["playable"]:
            success += 1

    print(success)
