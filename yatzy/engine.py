import random
from collections import Counter

import const
from action import Action
from logger import YatzyLogger
from state import State
from utils import combinations


class Engine:

    def __init__(self):
        self.logger = YatzyLogger(__name__).get_logger()
        self.combinations = {
            "aces": {
                "validator": self.upper_section_validator,
                "die_value": 1,
                "scoring": self.upper_section_scoring,
                "n": None
            },
            "twos": {
                "validator": self.upper_section_validator,
                "die_value": 2,
                "scoring": self.upper_section_scoring,
                "n": None
            },
            "threes": {
                "validator": self.upper_section_validator,
                "die_value": 3,
                "scoring": self.upper_section_scoring,
                "n": None
            },
            "fours": {
                "validator": self.upper_section_validator,
                "die_value": 4,
                "scoring": self.upper_section_scoring,
                "n": None
            },
            "fives": {
                "validator": self.upper_section_validator,
                "die_value": 5,
                "scoring": self.upper_section_scoring,
                "n": None
            },
            "sixes": {
                "validator": self.upper_section_validator,
                "die_value": 6,
                "scoring": self.upper_section_scoring,
                "n": None
            },
            "pair_ones": {
                "validator": self.n_kind_validator,
                "die_value": 1,
                "scoring": self.n_kind_scoring,
                "n": 2
            },
            "pair_twos": {
                "validator": self.n_kind_validator,
                "die_value": 2,
                "scoring": self.n_kind_scoring,
                "n": 2
            },
            "pair_threes": {
                "validator": self.n_kind_validator,
                "die_value": 3,
                "scoring": self.n_kind_scoring,
                "n": 2
            },
            "pair_fours": {
                "validator": self.n_kind_validator,
                "die_value": 4,
                "scoring": self.n_kind_scoring,
                "n": 2
            },
            "pair_fives": {
                "validator": self.n_kind_validator,
                "die_value": 5,
                "scoring": self.n_kind_scoring,
                "n": 2
            },
            "pair_sixes": {
                "validator": self.n_kind_validator,
                "die_value": 6,
                "scoring": self.n_kind_scoring,
                "n": 2
            },
            "two_pair": {
                "validator": self.two_pair_validator,
                "die_value": None,
                "scoring": self.two_pair_scoring,
                "n": None
            },
            "three_of_a_kind": {
                "validator": self.n_kind_validator,
                "die_value": None,
                "scoring": self.n_kind_scoring,
                "n": 3
            },
            "four_of_a_kind": {
                "validator": self.n_kind_validator,
                "die_value": None,
                "scoring": self.n_kind_scoring,
                "n": 4
            },
            "small_straight": {
                "validator": self.small_straight_validator,
                "die_value": None,
                "scoring": self.all_dice_scoring,
                "n": None
            },
            "large_straight": {
                "validator": self.large_straight_validator,
                "die_value": None,
                "scoring": self.all_dice_scoring,
                "n": None
            },
            "full_house": {
                "validator": self.full_house_validator,
                "die_value": None,
                "scoring": self.all_dice_scoring,
                "n": None,
            },
            "chance": {
                "validator": self.chance_validator,
                "die_value": None,
                "scoring": self.all_dice_scoring,
                "n": None,
            },
            "yatzy": {
                "validator": self.n_kind_validator,
                "die_value": None,
                "scoring": self.yatzy_scoring,
                "n": 5
            }
        }

    @staticmethod
    def dice_roll():
        return random.randint(1, 6)

    @staticmethod
    def upper_section_validator(dices, die_value=None, n=None):
        # Any set of dices are valid for the upper section
        return True, [die_value]

    @staticmethod
    def n_kind_validator(dices, die_value=None, n=None):
        # check if there are n of any number
        # returns Boolean, [list of options]
        valid = False
        options = []
        die_count = Counter(dices)
        for die in die_count:
            if not die_value:
                if die_count[die] >= n:
                    valid = True
                    options.append(die)
            elif die_value == die:
                if die_count[die] >= n:
                    valid = True
                    options.append(die)

        return valid, options

    @staticmethod
    def two_pair_validator(dices, die_value=None, n=None):
        pairs = 0
        dies = []
        die_count = Counter(dices)
        for die in die_count:
            if die_count[die] >= 2:
                pairs += 1
                dies.append(die)
        if pairs > 1:
            return True, []
        return False, []

    @staticmethod
    def two_pair_scoring(dices, die_value=None, n=None):
        score = 0
        die_count = Counter(dices)
        for die in die_count:
            if die_count[die] >= 2:
                score += (die * 2)
        return score

    @staticmethod
    def small_straight_validator(dices, die_value=None, n=None):
        return {1, 2, 3, 4, 5}.issubset(dices), []

    @staticmethod
    def large_straight_validator(dices, die_value=None, n=None):
        return {2, 3, 4, 5, 6}.issubset(dices), []

    @staticmethod
    def full_house_validator(dices, die_value=None, n=None):
        die_count = sorted(Counter(dices).items(), key=lambda item: item[1])
        if len(die_count) > 1:
            if die_count[-1][1] == 3 and die_count[-2][1] == 2:
                return True, []
        return False, []

    @staticmethod
    def chance_validator(dices, die_value=None, n=None):
        return True, []

    @staticmethod
    def upper_section_scoring(dices, die_value=None, n=None):
        # only die of the target value are counted
        score = 0
        for die in dices:
            if die == die_value:
                score += die
        return score

    @staticmethod
    def n_kind_scoring(dices, die_value=None, n=None):
        times_scored = 0
        score = 0
        for die in dices:
            if die == die_value:
                if times_scored < n:
                    score += die
                    times_scored += 1
        return score

    @staticmethod
    def yatzy_scoring(dices, die_value=None, n=None):
        return 50

    @staticmethod
    def all_dice_scoring(dices, die_value=None, n=None):
        return sum(dices)

    def get_score(self, combination, dices, die_value=None):
        if not die_value:
            return self.combinations[combination]["scoring"](dices,
                                                             self.combinations[combination]["die_value"],
                                                             self.combinations[combination]["n"])
        else:
            return self.combinations[combination]["scoring"](dices,
                                                             die_value,
                                                             self.combinations[combination]["n"])

    @staticmethod
    def game_over(state: State) -> bool:
        for c in state.scorecard:
            if state.scorecard[c]["allowed"]:
                return False
        return True

    @staticmethod
    def get_bonus(state: State) -> int:
        upper_section_score = state.get_uppersection_score()
        if upper_section_score >= 63:
            return 50
        return 0

    @staticmethod
    def get_final_score(state):
        return sum(state[const.STATE_SCORED_RANGE_START:const.STATE_SCORED_RANGE_END])

    def step(self, state: State, action: Action):
        score_before = state.get_score()

        self.logger.debug("stepping")

        if action.reroll == 1:
            self.logger.debug(f"rerolling some dices")
            self.logger.debug(f"locked dices    {action.locked_dices}")
            self.logger.debug(f"dices before    {state.dices}")
            new_state = self.reroll(state, action.locked_dices)
            self.logger.debug(f"dices after     {new_state.dices}")
        else:
            new_state = self.score(state, action)
            new_state = self.reroll(new_state, action.locked_dices)
            state.reset_rolls()

        game_over = self.game_over(state)
        score_after = new_state.get_score()
        if game_over:
            bonus = self.get_bonus(new_state)
            score_after += bonus

        reward = score_before - score_after

        return new_state, reward, game_over

    def get_initial_state(self) -> State:
        dices = [self.dice_roll() for n in range(const.DICES_COUNT)]
        return State(rolls=1, dices=dices, combinations=combinations)

    def get_options(self, dices):
        options = [0] * const.COMBINATIONS_COUNT
        for ix, c in enumerate(self.combinations):
            comb = self.combinations[c]
            comb_name = c
            valid = comb["validator"](dices, comb["die_value"], comb["n"])
            if valid[0]:
                if len(valid[1]) == 0:
                    options[ix] = 1
                for die_value in valid[1]:
                    options[ix] = 1
        return options

    def make_action_legal(self, action: Action, state: State):

        # First we check if it is allowed to reroll, otherwise we force scoring
        reroll = round(action.reroll)
        if state.rolls > 3:
            reroll = 0

        # Next we turn our "keep dice" decision into a one-hot encoded vector
        locked_dices = [round(n) for n in action.locked_dices]

        # Next we check that we are not trying to score something
        # that has already been scored
        scoring_vector = action.scoring_vector
        allowed_vector = state.get_allowed_vector()
        not_played_yet = [a * b for a, b in zip(allowed_vector, scoring_vector)]
        legal_options = self.get_options(state.dices)
        legal_scoring_vector = [a * b for a, b in zip(not_played_yet, legal_options)]

        # If there are no legal scoring options left, we are forced to zero score
        if sum(legal_scoring_vector) > 0:
            scoring_index = legal_scoring_vector.index(max(legal_scoring_vector))
            zero_score = False
        else:
            self.logger.debug("no legal option, forced to zero score")
            remaining_options_indices = state.get_index_of_remaining_options()
            zero_score = True
            scoring_index = random.choice(remaining_options_indices)

        # zero out everything but the thing we will score
        # if a forced zero score occurs, we send a boolean flag for that along
        legal_scoring_vector = [1 if i == scoring_index else 0 for i in range(len(legal_scoring_vector))]

        return Action(reroll, locked_dices, legal_scoring_vector, zero_score)

    def score(self, state: State, action: Action):
        self.logger.debug(f"scoring")
        for i in range(len(action.scoring_vector)):
            if action.scoring_vector[i] == 1:
                combination_name = combinations[i]
                if combination_name == 'yatzy':
                    self.logger.info(f"yatzy dices: {state.dices}")
                score = self.get_score(combination_name, state.dices)
                if "pair_" in combination_name:
                    for c in combinations:
                        if "pair_" in c:
                            state.scorecard[c]["allowed"] = False
                if action.zero_score:
                    score = 0
                state.enter_score(combination_name, score)
                self.logger.debug(f"force_zero  : {action.zero_score}")
                self.logger.debug(f"combination : {combination_name}")
                self.logger.debug(f"dices       : {state.dices}")
                self.logger.debug(f"score       : {score}")
        return state

    def reroll(self, state: State, locked_dice) -> State:
        self.logger.debug(f"new roll with dices")
        for i in range(len(locked_dice)):
            if locked_dice[i] == 0:
                state.dices[i] = self.dice_roll()
            else:
                pass
        state.rolls += 1
        return state
