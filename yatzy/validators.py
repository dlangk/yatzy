from collections import Counter


def upper_section_validator(dices, die_value=None, die_count=None):
    # Any set of dices are valid for the upper section
    return {"playable": True, "die": [die_value]}


def n_kind_validator(dices: list, die_value=None, die_count=None):
    # check if there are die_count number of die_value
    # or die_count of any if die_value is None
    # returns Boolean, [list of options]

    valid = False
    options = []

    if not die_value:
        number_1 = dices.count(1)
        number_2 = dices.count(2)
        number_3 = dices.count(3)
        number_4 = dices.count(4)
        number_5 = dices.count(5)
        number_6 = dices.count(6)
        if number_1 > die_count:
            options.append(1)
        if number_2 > die_count:
            options.append(2)
        if number_3 > die_count:
            options.append(3)
        if number_4 > die_count:
            options.append(4)
        if number_5 > die_count:
            options.append(5)
        if number_6 > die_count:
            options.append(6)

    if die_value:
        number_of_die = dices.count(die_value)
        if number_of_die > die_count:
            options.append(die_value)

    if len(options) > 0:
        valid = True

    return {"playable": valid, "die": options}


def two_pairs_validator(dices, die_value=None, die_count=None):
    pairs = 0
    dies = []
    die_count = Counter(dices)
    for die in die_count:
        if die_count[die] >= 2:
            pairs += 1
            dies.append(die)
    if pairs > 1:
        return {"playable": True, "die": []}
    return {"playable": False, "die": []}


def small_straight_validator(dices, die_value=None, die_count=None):
    return {"playable": {1, 2, 3, 4, 5}.issubset(dices), "die": []}


def large_straight_validator(dices, die_value=None, die_count=None):
    return {"playable": {2, 3, 4, 5, 6}.issubset(dices), "die": []}


def full_house_validator(dices, die_value=None, die_count=None):
    die_count = sorted(Counter(dices).items(), key=lambda item: item[1])
    if len(die_count) > 1:
        if die_count[-1][1] == 3 and die_count[-2][1] == 2:
            return {"playable": True, "die": []}
    return {"playable": False, "die": []}


def chance_validator(dices, die_value=None, die_count=None):
    return {"playable": True, "die": []}
