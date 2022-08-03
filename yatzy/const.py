DICES_COUNT = 5

import validators
import scorers

combinations = {
    "aces": {
        "validator": validators.upper_section_validator,
        "required_die_value": 1,
        "scorer": scorers.upper_section_scorer,
        "required_die_count": None,
        "upper_section": True
    },
    "twos": {
        "validator": validators.upper_section_validator,
        "required_die_value": 2,
        "scorer": scorers.upper_section_scorer,
        "required_die_count": None,
        "upper_section": True
    },
    "threes": {
        "validator": validators.upper_section_validator,
        "required_die_value": 3,
        "scorer": scorers.upper_section_scorer,
        "required_die_count": None,
        "upper_section": True
    },
    "fours": {
        "validator": validators.upper_section_validator,
        "required_die_value": 4,
        "scorer": scorers.upper_section_scorer,
        "required_die_count": None,
        "upper_section": True
    },
    "fives": {
        "validator": validators.upper_section_validator,
        "required_die_value": 5,
        "scorer": scorers.upper_section_scorer,
        "required_die_count": None,
        "upper_section": True
    },
    "sixes": {
        "validator": validators.upper_section_validator,
        "required_die_value": 6,
        "scorer": scorers.upper_section_scorer,
        "required_die_count": None,
        "upper_section": True
    },
    "pair_ones": {
        "validator": validators.n_kind_validator,
        "required_die_value": 1,
        "scorer": scorers.n_kind_scorer,
        "required_die_count": 2,
        "upper_section": False
    },
    "pair_twos": {
        "validator": validators.n_kind_validator,
        "required_die_value": 2,
        "scorer": scorers.n_kind_scorer,
        "required_die_count": 2,
        "upper_section": False
    },
    "pair_threes": {
        "validator": validators.n_kind_validator,
        "required_die_value": 3,
        "scorer": scorers.n_kind_scorer,
        "required_die_count": 2,
        "upper_section": False
    },
    "pair_fours": {
        "validator": validators.n_kind_validator,
        "required_die_value": 4,
        "scorer": scorers.n_kind_scorer,
        "required_die_count": 2,
        "upper_section": False
    },
    "pair_fives": {
        "validator": validators.n_kind_validator,
        "required_die_value": 5,
        "scorer": scorers.n_kind_scorer,
        "required_die_count": 2,
        "upper_section": False
    },
    "pair_sixes": {
        "validator": validators.n_kind_validator,
        "required_die_value": 6,
        "scorer": scorers.n_kind_scorer,
        "required_die_count": 2,
        "upper_section": False
    },
    "two_pairs": {
        "validator": validators.two_pairs_validator,
        "required_die_value": None,
        "scorer": scorers.two_pairs_scorer,
        "required_die_count": None,
        "upper_section": False
    },
    "three_of_a_kind": {
        "validator": validators.n_kind_validator,
        "required_die_value": None,
        "scorer": scorers.n_kind_scorer,
        "required_die_count": 3,
        "upper_section": False
    },
    "four_of_a_kind": {
        "validator": validators.n_kind_validator,
        "required_die_value": None,
        "scorer": scorers.n_kind_scorer,
        "required_die_count": 4,
        "upper_section": False
    },
    "small_straight": {
        "validator": validators.small_straight_validator,
        "required_die_value": None,
        "scorer": scorers.all_dice_scorer,
        "required_die_count": None,
        "upper_section": False
    },
    "large_straight": {
        "validator": validators.large_straight_validator,
        "required_die_value": None,
        "scorer": scorers.all_dice_scorer,
        "required_die_count": None,
        "upper_section": False
    },
    "full_house": {
        "validator": validators.full_house_validator,
        "required_die_value": None,
        "scorer": scorers.all_dice_scorer,
        "required_die_count": None,
        "upper_section": False
    },
    "chance": {
        "validator": validators.chance_validator,
        "required_die_value": None,
        "scorer": scorers.all_dice_scorer,
        "required_die_count": None,
        "upper_section": False
    },
    "yatzy": {
        "validator": validators.n_kind_validator,
        "required_die_value": None,
        "scorer": scorers.yatzy_scorer,
        "required_die_count": 5,
        "upper_section": False
    }
}
