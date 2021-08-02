import json

combinations = ["aces", "twos", "threes", "fours", "fives", "sixes",
                "pair_ones", "pair_twos", "pair_threes", "pair_fours", "pair_fives", "pair_sixes",
                "two_pair", "three_of_a_kind", "four_of_a_kind",
                "small_straight", "large_straight",
                "full_house",
                "chance",
                "yatzy"]

upper_section = ["aces", "twos", "threes", "fours", "fives", "sixes"]


def ppj(parsed_json):
    print(json.dumps(parsed_json, indent=4, sort_keys=True))
