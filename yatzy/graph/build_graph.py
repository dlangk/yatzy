import itertools
import yatzy.mechanics.const as const
import os
import time
import pickle
import gzip
import numpy as np

from functools import lru_cache
from collections import Counter
from math import factorial, comb
from itertools import product
from tqdm import tqdm


def build_graph():
    print("\n******* building graph *******\n")

    print("Generating all dice sets")
    all_dice_sets = generate_all_dice_sets()

    print("Generating all start states")
    all_start_states = reduce_start_states(generate_start_states())

    print("Generating all reroll probabilities")
    transition_probabilities = compute_transition_probabilities(all_dice_sets)

    print("Generating all dice set probabilities")
    all_dice_set_probabilities = generate_all_dice_set_probabilities(all_dice_sets)

    print("Generate all state integers")
    all_state_integers = generate_all_state_integers()

    print("Generating all reroll edges")
    all_reroll_masks_by_filter = generate_reroll_masks(all_dice_sets)

    state_score_file = "/Users/langkilde/IdeaProjects/yatzy-old/data/state_scores.pkl.gz"
    expected_state_scores = load_or_create_state_score_file(state_score_file)

    for i in range(0, 13):
        start_states = filter_states(all_start_states, i)
        for state in start_states:
            depth = 0

            start_time = time.time()  # Start timing
            time_checkpoints = [start_time]

            upper_score, scored_categories = decode_state_integer(state)
            print("Processing state: ", state)
            print(f"{upper_score:06b} = ", upper_score)
            print(f"{scored_categories:013b}")

            score = get_expected_score(state,
                                       upper_score,
                                       scored_categories,
                                       all_dice_sets,
                                       all_dice_set_probabilities,
                                       expected_state_scores,
                                       transition_probabilities,
                                       all_reroll_masks_by_filter,
                                       all_state_integers,
                                       time_checkpoints,
                                       depth)

            print("Expected score = ", score)
            print("Time taken to get expected score: {:.8f} seconds.\n".format(time.time() - start_time))
            expected_state_scores[state] = score


def get_expected_score(state,
                       upper_score,
                       scored_categories,
                       all_dice_sets,
                       all_dice_set_probabilities,
                       expected_state_scores,
                       transition_probabilities,
                       all_reroll_masks,
                       all_state_integers,
                       time_checkpoints,
                       depth):
    # We start by checking if the game is already over
    # This should only happen for the 63 * 13 cases when all categories are scored.
    # It should return 0 for all cases except when the upper score is at or above the threshold
    # Then it returns the upper bonus

    # This method checks if the last 13 bits of the state binary number are all 1
    game_over = (state & 0x1FFF) == 0x1FFF

    if game_over:
        new_upper_score, expected_score, new_scored_categories = \
            get_max_exit_score(state, upper_score, scored_categories, dice_set=None, game_over=game_over)
        expected_state_scores[state] = expected_score
        return expected_score

    # We will now proceed to calculate all the internal states to this node
    all_exit_scores = get_all_exit_scores(state, upper_score, scored_categories, all_dice_sets, game_over)

    # We then start by calculating the expected score of each of the possible initial dice sets we can roll
    expected_score = 0
    rerolls_remaining = 2

    for dice_set in all_dice_sets:
        reroll_filter = get_reroll_filter(dice_set)

        # The expected score of a state is the sum of the probability of the dice set
        # times the expected score of the state that dice set takes us to
        expected_score += \
            all_dice_set_probabilities[dice_set] * \
            get_expected_score_reroll(state, dice_set, rerolls_remaining, upper_score, scored_categories, reroll_filter,
                                      all_dice_sets, all_reroll_masks, transition_probabilities, all_exit_scores,
                                      expected_state_scores, game_over, all_state_integers, time_checkpoints, depth)
    return expected_score


def no_more_rerolls_get_expected_score(dice_set,
                                       all_exit_scores,
                                       all_state_integers,
                                       expected_state_scores):
    (new_upper_score, best_additional_score, new_scored_categories) = retrieve_scores(all_exit_scores, dice_set)
    new_upper_score = upper_score_set_to_threshold_if_over(new_upper_score)
    next_state = (new_upper_score << 13) | new_scored_categories
    next_state_potential = expected_state_scores[next_state]
    reroll_state_expected_score = best_additional_score + next_state_potential
    return reroll_state_expected_score


def upper_score_set_to_threshold_if_over(new_upper_score):
    if new_upper_score >= const.UPPER_BONUS_THRESHOLD:
        new_upper_score = const.UPPER_BONUS_THRESHOLD
    return new_upper_score


def get_next_state(new_upper_score, new_scored_categories, all_state_integers):
    next_state_key = (new_upper_score << 13) | new_scored_categories
    next_state = all_state_integers[next_state_key]
    return next_state


def retrieve_scores(all_exit_scores, dice_set_str):
    return all_exit_scores[dice_set_str]


def get_expected_score_reroll(state, dice_set, rerolls_remaining, upper_score, scored_categories, reroll_filter,
                              all_dice_sets, all_reroll_masks, transition_probabilities, all_exit_scores,
                              expected_state_scores, game_over, all_state_integers, time_checkpoints, depth):
    # The reroll expected score is the maximum expected score of all possible reroll options

    # If we do not have any rerolls remaining, we stop
    if rerolls_remaining == 0:
        return no_more_rerolls_get_expected_score(dice_set,
                                                  all_exit_scores,
                                                  all_state_integers,
                                                  expected_state_scores)

    # Otherwise, we deduct one reroll and calculate the expected score for all reroll options
    max_expected_score = 0
    reroll_masks = all_reroll_masks[reroll_filter]
    for reroll_mask in reroll_masks:
        expected_reroll_state_score = get_expected_score_reroll_mask(state, dice_set, reroll_mask,
                                                                     rerolls_remaining - 1, upper_score,
                                                                     scored_categories, reroll_filter, all_dice_sets,
                                                                     all_reroll_masks, transition_probabilities,
                                                                     all_exit_scores, expected_state_scores, game_over,
                                                                     all_state_integers, time_checkpoints, depth)
        if expected_reroll_state_score >= max_expected_score:
            max_expected_score = expected_reroll_state_score

    return max_expected_score


@lru_cache(maxsize=None)
def calculate_match_and_reroll_counts(dice_set_1, dice_set_2, reroll_mask):
    match_count = 0
    reroll_count = 0

    # Iterate through each position (0 to 4)
    for i in range(5):
        # Check if the die is locked (1 in reroll_mask)
        if (reroll_mask & (1 << (4 - i))) != 0:
            # Locked position: increment match_count if dice match
            if dice_set_1[i] == dice_set_2[i]:
                match_count += 1
        else:
            # Rerolled position: increment reroll_count
            reroll_count += 1

    return match_count, reroll_count


def get_expected_score_reroll_mask(state, dice_set, reroll_mask, rerolls_remaining, upper_score, scored_categories,
                                   reroll_filter, all_dice_sets, all_reroll_masks, transition_probabilities,
                                   all_exit_scores, expected_state_scores, game_over, all_state_integers,
                                   time_checkpoints, depth):
    # The expected score of a reroll mask is the sum of the probability of getting a dice set from the reroll mask
    # times the expected score of the state that dice set takes us to
    expected_score = 0
    remaining_remaining_decreased = rerolls_remaining - 1

    for dice_set_target in all_dice_sets:
        # This is equivalent to P(r' -> r)

        target_probability = transition_probabilities.get((reroll_mask, match_count, reroll_count), 0)

        if target_probability > 0:
            # This is E(S, r, n-1)
            expected_score_target = get_expected_score_reroll(state, dice_set, remaining_remaining_decreased,
                                                              upper_score,
                                                              scored_categories, reroll_filter, all_dice_sets,
                                                              all_reroll_masks, transition_probabilities,
                                                              all_exit_scores,
                                                              expected_state_scores, game_over, all_state_integers,
                                                              time_checkpoints, depth)
            # This is Sum(P(r' -> r) * E(S, r, n-1))
            expected_score += (target_probability * expected_score_target)

    return expected_score


def load_or_create_state_score_file(score_filename):
    # Check if the file already exists
    if os.path.isfile(score_filename):
        print("Loading existing state scores")
        # Load the scores from the existing file
        with gzip.open(score_filename, 'rb') as f:
            expected_state_scores = pickle.load(f)
    else:
        # Initialize a dictionary to store scores
        expected_state_scores = {}

    return expected_state_scores


def generate_all_state_integers():
    state_integers = {}
    for upper_score in range(0, 64):
        for scored_categories in range(8192):
            # Combine the two numbers into a single 19-bit integer key directly
            start_state_integer = (upper_score << 13) | scored_categories
            state_integers[(upper_score, scored_categories)] = start_state_integer
    return state_integers


def generate_state_integer(upper_score, scored_categories):
    if upper_score >= 63:
        upper_score = 63
    return encode_state_integer(upper_score, scored_categories)


def generate_all_dice_set_probabilities(all_dice_sets):
    all_dice_set_probabilities = {}
    for dice_set in all_dice_sets:
        probability_of_dice_set = get_probability_of_dice_set(dice_set)
        all_dice_set_probabilities[dice_set] = probability_of_dice_set
    return all_dice_set_probabilities


def get_probability_of_dice_set(dice_set):
    counts = count_dice_faces(dice_set)
    probability = probability_of_dice_set_by_counts(counts)
    return probability


def generate_all_dice_sets():
    dice_faces = [1, 2, 3, 4, 5, 6]
    return [dice_face for dice_face in product(dice_faces, repeat=5)]


def generate_dice_states(dices, num_dices_reroll):
    dice_faces = [1, 2, 3, 4, 5, 6]
    if num_dices_reroll == 0:
        return [dices]
    outcome = [dices[:-num_dices_reroll] + list(comb) for comb in
               itertools.combinations_with_replacement(dice_faces, num_dices_reroll)]
    return outcome


def get_all_exit_scores(state, upper_score, scored_categories, all_dice_sets, game_over):
    exit_scores = {}
    for dice_set in all_dice_sets:
        new_upper_score, best_additional_score, new_scored_categories = get_max_exit_score(state,
                                                                                           upper_score,
                                                                                           scored_categories,
                                                                                           dice_set,
                                                                                           game_over)
        exit_scores[dice_set] = new_upper_score, best_additional_score, new_scored_categories
    return exit_scores


def encode_state_integer(new_upper, new_cat):
    upper_score_binary = bin(new_upper)[2:].zfill(6)
    score_category_int = int(new_cat, 2)
    scored_categories_bin = bin(score_category_int)[2:].zfill(13)
    return int(upper_score_binary + scored_categories_bin, 2)


@lru_cache(maxsize=None)  # Cache results to optimize repeated calls
def get_transition_probability(dice_set_1, dice_set_2, reroll_mask):
    match_count = 0
    reroll_count = 0

    for i in range(5):
        if (reroll_mask & (1 << (4 - i))) != 0:  # Locked position
            if dice_set_1[i] == dice_set_2[i]:
                match_count += 1
        else:  # Rerolled position
            reroll_count += 1

    # Calculate probability based on reroll count
    return (1 / 6) ** reroll_count if reroll_count > 0 else (1.0 if match_count == len(dice_set_1) else 0.0)


def is_outcome_possible(dice_set, reroll_mask, target_outcome):
    for i in range(5):
        # Check if the bit at position `i` is 1 (locked)
        if (reroll_mask & (1 << (4 - i))) != 0:  # `4 - i` because we want to start from the leftmost bit
            # If the die is locked, it must match the target
            if dice_set[i] != target_outcome[i]:
                return False
    return True


def remove_list_from_list(list_to_remove, original_list):
    # Create a copy of the original list so we don't modify the original
    result = original_list.copy()

    # Remove each element in list_to_remove from the result
    for element in list_to_remove:
        if element in result:
            result.remove(element)

    return result


def combinations(n, r):
    return factorial(n) / (factorial(r) * factorial(n - r))


def get_probability_of_specific_roll(num_faces, num_dice, face_counts):
    total_outcomes = num_faces ** num_dice
    ways_to_get_outcome = 1
    for count in face_counts:
        ways_to_get_outcome *= combinations(num_dice, count)
        num_dice -= count
    return ways_to_get_outcome / total_outcomes


def probability_of_specific_outcome_unordered(dice_outcome):
    # We assume all dice have the same number of faces
    num_faces = 6
    num_dice = len(dice_outcome)

    # Count the number of each face in the outcome
    face_counts = [dice_outcome.count(face) for face in set(dice_outcome)]

    return get_probability_of_specific_roll(num_faces, num_dice, face_counts)


def possible_reroll_mask(kept_dices, outcome):
    # We create counter objects for both kept dices and resulting dices
    counter1 = Counter(kept_dices)
    counter2 = Counter(outcome)

    # We then check if every element in counter1 is in counter2 with a count that's at least as high
    for element, count in counter1.items():
        if count > counter2[element]:
            return False

    # If we haven't returned False by now, the resulting dice set is possible
    return True


def all_scored_categories():
    return "1111111111111"


def get_max_exit_score(state, upper_score, scored_categories, dice_set, game_over):
    # At the end of a turn, the expected score is given as the maximum of the unused categories
    # plus the potential of the next state
    # If all categories are already scored, we check for bonus and return the total
    if game_over:
        # Check if upper_score meets the threshold
        # Remember, Python allows you to compare integers regardless of how they are stored
        # In this case, one is in binary and the other as int
        if upper_score >= const.UPPER_BONUS_THRESHOLD:
            best_additional_score = const.UPPER_BONUS
        else:
            best_additional_score = 0
        # Return upper_score directly, as it's already stored as a binary integer
        return upper_score, best_additional_score, scored_categories

    # If the game is not over, we iterate over all possible categories and check if it is still open for scoring
    # If it is, we check if the available dices are valid for that category, and what the score would be.
    # If it is not valid, we can still score 0, so we return zero for all invalid categories.
    best_additional_score = 0
    new_upper_score = upper_score

    keys = list(const.combinations.keys())
    best_category_name = None

    for ix, category_name in enumerate(keys):
        # Check if the bit at position `ix` is 0 by examining the bit at position `12 - ix`
        if (scored_categories & (1 << (12 - ix))) == 0:
            valid = is_dice_set_valid(category_name, dice_set)
            if valid:
                score = get_score(category_name, dice_set)
                if score >= best_additional_score:
                    best_additional_score = score
                    best_category_name = category_name
            else:
                score = 0
                if score >= best_additional_score:
                    best_additional_score = score
                    best_category_name = category_name

    # Update `scored_categories` by setting the bit for `best_category_name`
    new_scored_categories = update_scored_categories(scored_categories, best_category_name)

    # If the best category is in the upper section, add `best_additional_score` to `new_upper_score`
    if is_upper_section(best_category_name):
        new_upper_score += best_additional_score

    return new_upper_score, best_additional_score, new_scored_categories


def update_scored_categories(scored_categories, scored_category):
    # Find the index of the scored category
    index_of_scored_category = list(const.combinations.keys()).index(scored_category)
    # Set the bit at the index position to 1
    return scored_categories | (1 << (12 - index_of_scored_category))


def is_upper_section(category_name):
    return const.combinations[category_name]["upper_section"]


def get_score(category_name, dice_set):
    return const.combinations[category_name]["scorer"] \
        (dices=dice_set,
         die_value=const.combinations[category_name]["required_die_value"],
         die_count=const.combinations[category_name]["required_die_count"])


def is_dice_set_valid(category_name, dice_set):
    return const.combinations[category_name]["validator"] \
        (dices=dice_set,
         die_value=const.combinations[category_name]["required_die_value"],
         die_count=const.combinations[category_name]["required_die_count"])


def check_all_ones(string):
    return all(char == '1' for char in string)


def permutations(counts):
    permutations = factorial(sum(counts))  # we will always have 5 dices but for clarity we do this
    # This is equivalent to 5! / (c1! * c2! * c3! * c4! * c5!)
    for c in counts:
        permutations //= factorial(c)  # floor division + assignment
    return permutations


def count_dice_faces(dice_set):
    counts = {}
    # Iterate over the dice in the set
    for die in dice_set:
        # Count how many times this die appears in the set and store the count in the dictionary
        counts[die] = counts.get(die, 0) + 1
    # Return the counts of unique dice faces as a list
    return list(counts.values())


def probability_of_dice_set_by_counts(counts):
    total_outcomes = 6 ** 5  # There are 7776 possible outcomes for 5 dices
    dice_set_outcomes = permutations(counts)  # There is 5! / (c1! * c2! * c3! * c4! * c5!) outcomes for this dice set
    return dice_set_outcomes / total_outcomes  # Probability of this dice set


def count_ones(s):
    # Count the number of '1' characters
    num_ones = s.count('1')
    return num_ones


def filter_states(states, remaining_to_score):
    filtered_states = []
    for start_state in states:
        upper_score, scored_categories = decode_state_integer(start_state)
        if is_remaining_to_score(scored_categories, remaining_to_score):
            filtered_states.append(start_state)
    return filtered_states


def is_remaining_to_score(scored_categories, num_categories):
    return num_categories == (13 - scored_categories.bit_count())


def generate_reroll_states(dices):
    # There are 462 distinct outcomes from to rerolling a specific set of 5 6-sided dices.
    # Five dices have 252 unique outcomes => None of the original dices are kept
    # Four dices have 126 unique outcomes => One of the original dices are kept
    # Three dices have 56 unique outcomes => Two of the original dices are kept
    # Two dices have 21 unique outcomes => Three of the original dices are kept
    # One die has 6 unique outcomes => Four of the original dices are kept
    reroll_states = []
    for i in range(6):
        states = generate_dice_states(dices, i)
        reroll_states.extend(states)
    if len(reroll_states) == 462:
        return reroll_states
    else:
        raise Exception("There should be 462 distinct outcomes from to rerolling a specific set of 5 6-sided dices.")


def generate_reroll_masks(dice_sets):
    # There are 32 ways to partially reroll the first dice set.
    # If the dice set is [1,1,2,3,4] re-rolling the first and second dice is the same. Therefor we can reduce edges.
    # If each dice face is different on the first roll, then there are 32 ways to reroll the dice.
    # If two of the dices faces are the same but the rest are different, then there are 24 ways to reroll the dice.
    # If three of the dices faces are the same but the rest are different, then there are 18 ways to reroll the dice.
    # If four of the dices faces are the same but the rest are different, then there are 16 ways to reroll the dice.
    # If all the dices faces are the same, then there are 6 ways to keep the dice.
    abcde = "abcde"  # case 5 unique faces => 32 ways to reroll
    aabcd = "aabcd"  # case 4 unique faces => 24 ways to reroll
    aabbc = "aabbc"  # case 3 unique faces and max 2 of same face => 18 ways to reroll
    aaabc = "aaabc"  # case 3 unique faces and max 3 of same face => 16 ways to reroll
    aaabb = "aaabb"  # case 2 unique faces and max 3 of same face => 12 ways to reroll
    aaaab = "aaaab"  # case 2 unique faces and max 4 of same face => 10 ways to reroll
    aaaaa = "aaaaa"  # case 1 unique face => 6 ways to reroll
    edges = {abcde: [], aabcd: [], aabbc: [], aaabc: [], aaabb: [], aaaab: [], aaaaa: []}

    for dice_set in dice_sets:
        reroll_filter = get_reroll_filter(dice_set)
        if len(edges[reroll_filter]) == 0:
            dice_set_edges = generate_reroll_edges_from_dice_set(dice_set)
            edges[reroll_filter].extend(dice_set_edges[reroll_filter])

    for reroll_key in edges.keys():
        edges[reroll_key] = [(tuple_to_binary_number(mask)) for mask in edges[reroll_key]]

    return edges


def generate_reroll_edges_from_dice_set(dice_set):
    abcde = "abcde"  # case 5 unique faces => 32 ways to reroll
    aabcd = "aabcd"  # case 4 unique faces => 24 ways to reroll
    aabbc = "aabbc"  # case 3 unique faces and max 2 of same face => 18 ways to reroll
    aaabc = "aaabc"  # case 3 unique faces and max 3 of same face => 16 ways to reroll
    aaabb = "aaabb"  # case 2 unique faces and max 3 of same face => 12 ways to reroll
    aaaab = "aaaab"  # case 2 unique faces and max 4 of same face => 10 ways to reroll
    aaaaa = "aaaaa"  # case 1 unique face => 6 ways to reroll
    edges = {abcde: [], aabcd: [], aabbc: [], aaabc: [], aaabb: [], aaaab: [], aaaaa: []}

    unique_faces_count = count_unique_dice(dice_set)
    max_dice_face_count = get_max_dice_face_count(dice_set)
    min_dice_face_count = get_min_dice_face_count(dice_set)

    if unique_faces_count == 5:
        # There are 32 ways to reroll five different dices.
        edges[abcde].extend(generate_reroll_mask(5, abcde))

    elif unique_faces_count == 4:
        # There are 24 ways to reroll four different dices since rerolling dice 1 or 2 is equivalent.
        edges[aabcd].extend(generate_reroll_mask(5, aabcd))

    elif unique_faces_count == 3:
        if max_dice_face_count == 2:
            # There are 18 ways to reroll if two dice faces occur two times.
            edges[aabbc].extend(generate_reroll_mask(5, aabbc))
        elif max_dice_face_count == 3:
            # There are 16 ways to reroll if one die faces occur three times and the remaining two are different.
            edges[aaabc].extend(generate_reroll_mask(5, aaabc))

    elif unique_faces_count == 2:
        if min_dice_face_count == 2:
            # There are 12 ways to reroll if one die face occurs three times and the remaining two are the same.
            edges[aaabb].extend(generate_reroll_mask(5, aaabb))
        else:
            # There are 10 ways to reroll if one die face occurs four times and the remaining one is different.
            edges[aaaab].extend(generate_reroll_mask(5, aaaab))
    else:
        # There are 6 ways to keep the dice if all the dice faces are the same.
        edges[aaaaa].extend(generate_reroll_mask(5, aaaaa))

    return edges


def get_reroll_filter(dices):
    unique_faces_count = count_unique_dice(dices)
    max_dice_face_count = get_max_dice_face_count(dices)
    min_dice_face_count = get_min_dice_face_count(dices)

    if unique_faces_count == 5:
        return "abcde"
    elif unique_faces_count == 4:
        return "aabcd"
    elif unique_faces_count == 3:
        if max_dice_face_count == 2:
            return "aabbc"
        elif max_dice_face_count == 3:
            return "aaabc"
    elif unique_faces_count == 2:
        if min_dice_face_count == 2:
            return "aaabb"
        else:
            return "aaaab"
    else:
        return "aaaaa"


def generate_reroll_mask(num_dices, filter=None):
    # Generate all possible keep/roll combinations for 5 dice
    masks = list(itertools.product([0, 1], repeat=num_dices))

    if filter == "abcde":
        if len(masks) == 32:
            return masks
        else:
            raise Exception("Unexpected number of masks after filtering: " + str(len(masks)))

    if filter == "aabcd":
        # Remove all masks where the first dice is 1 and the second is 0 but both dices are not the same
        filtered_masks = [mask for mask in masks if not (mask[0] == 1 and mask[1] == 0 and mask[0] != mask[1])]
        if len(filtered_masks) == 24:
            return filtered_masks
        else:
            raise Exception("Unexpected number of masks after filtering: " + str(len(filtered_masks)))

    if filter == "aabbc":
        # Remove all masks where the first dice is 1 and the second is 0 but both dices are not the same
        filtered_masks_1 = [mask for mask in masks if
                            not (mask[0] == 1 and mask[1] == 0 and mask[0] != mask[1])]
        # Remove all masks where the third dice is 1 and the fourth is 0 but both dices are not the same
        filtered_masks_2 = [mask for mask in filtered_masks_1 if
                            not (mask[2] == 1 and mask[3] == 0 and mask[2] != mask[3])]

        if len(filtered_masks_2) == 18:
            return filtered_masks_2
        else:
            raise Exception("Unexpected number of masks after filtering: " + str(len(filtered_masks_2)))

    if filter == "aaabc":
        masks = [[1, 1, 1, 0, 0], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1],
                 [0, 1, 1, 0, 0], [0, 1, 1, 0, 1], [0, 1, 1, 1, 0], [0, 1, 1, 1, 1],
                 [0, 0, 1, 0, 0], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0], [0, 0, 1, 1, 1],
                 [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1]]

        if len(masks) == 16:
            return masks
        else:
            raise Exception("Unexpected number of masks after filtering: " + str(len(masks)))

    if filter == "aaabb":
        masks = [[1, 1, 1, 0, 0], [1, 1, 1, 0, 1], [1, 1, 1, 1, 1],
                 [0, 1, 1, 0, 0], [0, 1, 1, 0, 1], [0, 1, 1, 1, 1],
                 [0, 0, 1, 0, 0], [0, 0, 1, 0, 1], [0, 0, 1, 1, 1],
                 [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 1]]

        if len(masks) == 12:
            return masks
        else:
            raise Exception("Unexpected number of masks after filtering: " + str(len(masks)))

    if filter == "aaaab":
        masks = [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 0], [1, 1, 1, 1, 1],
                 [0, 0, 1, 1, 0], [1, 1, 1, 1, 1],
                 [0, 0, 0, 1, 0], [1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]

        if len(masks) == 10:
            return masks
        else:
            raise Exception("Unexpected number of masks after filtering: " + str(len(masks)))

    if filter == "aaaaa":
        masks = [[1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1],
                 [0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0]]
        if len(masks) == 6:
            return masks
        else:
            raise Exception("Unexpected number of masks after filtering: " + str(len(masks)))

    raise Exception("Unknown filter: " + filter)


def get_max_dice_face_count(dice_set):
    dice_faces_count = Counter(dice_set)
    return max(dice_faces_count.values())


def get_min_dice_face_count(dice_set):
    dice_faces_count = Counter(dice_set)
    return min(dice_faces_count.values())


def count_unique_dice(dice_set):
    return len(set(dice_set))


def reduce_start_states(start_states):
    """Reduce the number of start states by removing states that are not possible."""
    reduced_start_states = []
    for start_state in start_states:
        # Extract the upper_score (first 6 bits)
        upper_score = start_state >> 13
        # Extract the scored_categories (last 13 bits)
        scored_categories = start_state & 0x1FFF  # Mask with 13 bits of 1s (0x1FFF)

        # Compact conditional check for filtering
        if (upper_score > 0 and (scored_categories & 0x1F80) != 0) or \
                (upper_score == 0 and (scored_categories & 0x1F80) == 0):
            reduced_start_states.append(start_state)

    return reduced_start_states


def check_all_zeros(binary_string):
    # Check if all characters in the string are '0'
    return all(char == '0' for char in binary_string)


def generate_start_states():
    start_states = []

    # Generate all possible states as binary numbers (integers)
    for upper_score in range(64):  # 2^6 possible states for upper_score
        for scored_categories in range(8192):  # 2^13 possible states for scored_categories
            # Combine upper_score and scored_categories into a single 19-bit integer
            start_state_integer = (upper_score << 13) | scored_categories
            start_states.append(start_state_integer)

    return start_states


def decode_state_integer(start_state):
    # Extract the upper_score (first 6 bits)
    upper_score = start_state >> 13
    # Extract the scored_categories (last 13 bits)
    scored_categories = start_state & 0x1FFF  # Mask with 13 bits of 1s (0x1FFF)
    return upper_score, scored_categories


def tuple_to_binary_number(tup):
    binary_number = (tup[0] << 4) | (tup[1] << 3) | (tup[2] << 2) | (tup[3] << 1) | tup[4]
    return binary_number


def generate_all_reroll_masks(num_dices):
    reroll_masks = list(product([0, 1], repeat=num_dices))
    reroll_masks = [(tuple_to_binary_number(mask), "".join(map(str, mask))) for mask in reroll_masks]
    return reroll_masks


def calculate_probability(dice_set_1, dice_set_2, reroll_mask):
    reroll_count = 0

    for i in range(5):
        if (reroll_mask & (1 << (4 - i))) != 0:  # Locked position
            if dice_set_1[i] != dice_set_2[i]:  # Locked dice must match for transition
                return 0.0
        else:
            reroll_count += 1  # Count rerolled positions

    # Calculate probability based on reroll count
    return (1 / 6) ** reroll_count if reroll_count > 0 else 1.0


def compute_transition_probabilities(dice_sets):
    filename = '/Users/langkilde/IdeaProjects/yatzy-old/data/transition_probabilities.pkl.gz'

    # Check if the file already exists
    if os.path.isfile(filename):
        print("Loading probabilities from file: " + filename)
        # Load the probabilities from the existing file
        with gzip.open(filename, 'rb') as f:
            probabilities = pickle.load(f)
    else:
        print("Computing probabilities and saving to file: " + filename)
        reroll_masks = [int("".join(map(str, mask)), 2) for mask in product([0, 1], repeat=5)]

        probabilities = {}
        total_iterations = len(dice_sets) * len(dice_sets) * len(reroll_masks)

        with tqdm(total=total_iterations) as tq:
            for dice_set_1 in dice_sets:
                for dice_set_2 in dice_sets:
                    for reroll_mask in reroll_masks:
                        probability = calculate_probability(dice_set_1, dice_set_2, reroll_mask)
                        if probability > 0:  # Store only non-zero probabilities to save space
                            probabilities[(dice_set_1, dice_set_2, reroll_mask)] = probability
                        tq.update(1)

        # Save probabilities to a compressed pickle file
        with gzip.open(filename, 'wb') as f:
            pickle.dump(probabilities, f, protocol=pickle.HIGHEST_PROTOCOL)

    return probabilities


if __name__ == "__main__":
    build_graph()
