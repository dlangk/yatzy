def fast_counter(dices):
    count = {1: dices.count(1),
             2: dices.count(2),
             3: dices.count(3),
             4: dices.count(4),
             5: dices.count(5),
             6: dices.count(6)}
    return count


def two_pairs_scorer(dices, die_value=None, die_count=None):
    score = 0
    die_count = fast_counter(dices)
    for die in die_count:
        if die_count[die] >= 2:
            score += (die * 2)
    return score


def upper_section_scorer(dices, die_value=None, die_count=None):
    # only die of the target value are counted
    score = 0
    for die in dices:
        if die == die_value:
            score += die
    return score


def n_kind_scorer(dices, die_value=None, die_count=None):
    times_scored = 0
    score = 0
    if die_value is None:
        counted_die = fast_counter(dices)
        for die in counted_die:
            if counted_die[die] >= die_count:
                score = (die_count * die)
    else:
        for die in dices:
            if die == die_value:
                if times_scored < die_count:
                    score += die
                    times_scored += 1
    return score


def yatzy_scorer(dices, die_value=None, die_count=None):
    return 50


def all_dice_scorer(dices, die_value=None, die_count=None):
    return sum(dices)
