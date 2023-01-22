import yatzy.mechanics.validators as validators
import yatzy.mechanics.scorers as scorers

DICES_COUNT = 5

upper_combinations = ["aces", "twos", "threes", "fours", "fives", "sixes"]

combinations = {
    "aces": {
        "validator": validators.upper_section_validator,
        "required_die_value": 1,
        "scorer": scorers.upper_section_scorer,
        "required_die_count": None,
        "upper_section": True,
        "probability": 1
    },
    "twos": {
        "validator": validators.upper_section_validator,
        "required_die_value": 2,
        "scorer": scorers.upper_section_scorer,
        "required_die_count": None,
        "upper_section": True,
        "probability": 1
    },
    "threes": {
        "validator": validators.upper_section_validator,
        "required_die_value": 3,
        "scorer": scorers.upper_section_scorer,
        "required_die_count": None,
        "upper_section": True,
        "probability": 1
    },
    "fours": {
        "validator": validators.upper_section_validator,
        "required_die_value": 4,
        "scorer": scorers.upper_section_scorer,
        "required_die_count": None,
        "upper_section": True,
        "probability": 1
    },
    "fives": {
        "validator": validators.upper_section_validator,
        "required_die_value": 5,
        "scorer": scorers.upper_section_scorer,
        "required_die_count": None,
        "upper_section": True,
        "probability": 1
    },
    "sixes": {
        "validator": validators.upper_section_validator,
        "required_die_value": 6,
        "scorer": scorers.upper_section_scorer,
        "required_die_count": None,
        "upper_section": True,
        "probability": 1
    },
    "pair_ones": {
        "validator": validators.n_kind_validator,
        "required_die_value": 1,
        "scorer": scorers.n_kind_scorer,
        "required_die_count": 2,
        "upper_section": False,
        "probability": 763/3888
    },
    "pair_twos": {
        "validator": validators.n_kind_validator,
        "required_die_value": 2,
        "scorer": scorers.n_kind_scorer,
        "required_die_count": 2,
        "upper_section": False,
        "probability": 763/3888
    },
    "pair_threes": {
        "validator": validators.n_kind_validator,
        "required_die_value": 3,
        "scorer": scorers.n_kind_scorer,
        "required_die_count": 2,
        "upper_section": False,
        "probability": 763/3888
    },
    "pair_fours": {
        "validator": validators.n_kind_validator,
        "required_die_value": 4,
        "scorer": scorers.n_kind_scorer,
        "required_die_count": 2,
        "upper_section": False,
        "probability": 763/3888
    },
    "pair_fives": {
        "validator": validators.n_kind_validator,
        "required_die_value": 5,
        "scorer": scorers.n_kind_scorer,
        "required_die_count": 2,
        "upper_section": False,
        "probability": 763/3888
    },
    "pair_sixes": {
        "validator": validators.n_kind_validator,
        "required_die_value": 6,
        "scorer": scorers.n_kind_scorer,
        "required_die_count": 2,
        "upper_section": False,
        "probability": 763/3888
    },
    "two_pairs": {
        "validator": validators.two_pairs_validator,
        "required_die_value": None,
        "scorer": scorers.two_pairs_scorer,
        "required_die_count": None,
        "upper_section": False,
        "probability": 175/648
    },
    "three_of_a_kind": {
        "validator": validators.n_kind_validator,
        "required_die_value": None,
        "scorer": scorers.n_kind_scorer,
        "required_die_count": 3,
        "upper_section": False,
        "probability": 23/108
    },
    "four_of_a_kind": {
        "validator": validators.n_kind_validator,
        "required_die_value": None,
        "scorer": scorers.n_kind_scorer,
        "required_die_count": 4,
        "upper_section": False,
        "probability": 13/648
    },
    "small_straight": {
        "validator": validators.small_straight_validator,
        "required_die_value": None,
        "scorer": scorers.all_dice_scorer,
        "required_die_count": None,
        "upper_section": False,
        "probability": 5/324
    },
    "large_straight": {
        "validator": validators.large_straight_validator,
        "required_die_value": None,
        "scorer": scorers.all_dice_scorer,
        "required_die_count": None,
        "upper_section": False,
        "probability": 5/324
    },
    "full_house": {
        "validator": validators.full_house_validator,
        "required_die_value": None,
        "scorer": scorers.all_dice_scorer,
        "required_die_count": None,
        "upper_section": False,
        "probability": 25/648
    },
    "chance": {
        "validator": validators.chance_validator,
        "required_die_value": None,
        "scorer": scorers.all_dice_scorer,
        "required_die_count": None,
        "upper_section": False,
        "probability": 1
    },
    "yatzy": {
        "validator": validators.n_kind_validator,
        "required_die_value": None,
        "scorer": scorers.yatzy_scorer,
        "required_die_count": 5,
        "upper_section": False,
        "probability": 1/1296
    }
}
