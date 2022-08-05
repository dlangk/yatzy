import math

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
        if i % 1000000 == 0:
            print("tested " + str(i) + " dice rolls")

    print(valid_dices)
    print(set(valid_dices))

    print("valid rolls")
    print(len(valid_dices))
    print("unique valid dices")
    print(len(set(valid_dices)))
    print("invalid dices")
    print(len(invalid_dices))
    print("unique invalid dices")
    print(len(set(invalid_dices)))


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
