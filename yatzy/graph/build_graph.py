import itertools
from collections import Counter
from math import factorial, comb

import yatzy.mechanics.const as const

EXPECTED_STATE_SCORE = {}


def build_graph():
    print("\n******* building graph *******\n")
    edges = {}

    start_states = generate_start_states()

    for i in range(0, 13):
        print("checking states for round " + str(13 - i))
        states = get_states(start_states, i)

        # We start by building out the end of the graph
        for state in states:
            upper_score, scored_categories = decode_state_integer(state)
            print("checking state " + str(state))
            print("upper_score: " + str(upper_score))
            print("scored_categories: " + str(scored_categories))
            score = get_expected_score(state)
            print("expected score: " + str(score))


def get_expected_score(state):
    # We are building a cache of all expected state scores, and first check if this state is already in there
    if EXPECTED_STATE_SCORE.get(state) is not None:
        return EXPECTED_STATE_SCORE.get(state)

    upper_score, scored_categories = decode_state_integer(state)
    # If all categories are already scored, we check for bonus and return the total
    game_over = check_all_ones(scored_categories)
    if game_over:
        return get_expected_exit_score(state, None)

    # If not, then we start examining the most likely state score
    expected_score = 0

    # First of all, there are 252 possible dice sets after the first roll
    dice_sets = generate_dice_states(dices=[1, 1, 1, 1, 1], num_dices_reroll=5)

    # We want to start by determining the expected value of all 252 possible end-states
    # I.e. assume we are ready to score, and have dice_set, what do we get?
    for dice_set in dice_sets:
        exit_score = get_expected_exit_score(state, dice_set)
        print("dice_set: " + str(dice_set) + ", exit_score: " + str(exit_score))

    counter = 0
    for dice_set in dice_sets:

        # Each of these have a probability of occurring
        p_dice_set = get_probability_of_dice_set(dice_set)

        reroll_filter = get_reroll_filter(dice_set)
        reroll_masks = generate_reroll_edges_from_dice_set(dice_set)[reroll_filter]

        for reroll_mask in reroll_masks:
            for dice_set_inner in dice_sets:
                p_r_rp = get_reroll_probability(dice_set, reroll_mask, dice_set_inner)
                if p_r_rp > 0:
                    print("d1: " + str(dice_set) +
                          ", m: " + str(reroll_mask) +
                          ", d2: " + str(dice_set_inner) +
                          ", p: " + str(p_r_rp))
                counter += 1
            pass
        print(counter)
        print("checking dice set " + str(dice_set) + " for state " + str(state))
        print("p_dice_set: " + str(p_dice_set))
        print(len(reroll_masks))
        print(len(dice_sets))
        print(len(dice_sets) * len(reroll_masks))

    # There are 462 ways of rerolling 5 dices (keep either 5,4,3,2,1)
    # Keep 5 gives 1 possible outcome, keep 4 gives 6 possible outcomes, etc.
    # We need to know what the probability for each of these outcomes is given
    # An initial dice set (r), a reroll mask (m) and a target outcome (r')

    # for reroll_edge in reroll_edges:
    #     # For every edge we need to determine what possible dice sets we can get from it
    #     print("checking reroll edge " + str(reroll_edge))
    #     mask = reroll_edges[reroll_edge]
    #     pass
    # expected_score += p_dice_set * get_expected_reroll_score(state, dice_set, 2)

    return expected_score


def get_reroll_probability(dice_set, reroll_mask, target_outcome):
    kept_dices = [dice_set[i] for i in range(0, 5) if reroll_mask[i] == 0]

    # First of all, we should check if the resulting dice set is even possible
    # If not, then the probability is 0
    if not possible_reroll_mask(kept_dices, target_outcome):
        return 0

    required_dices = remove_list_from_list(kept_dices, target_outcome)
    probability = probability_of_specific_outcome_unordered(required_dices)

    return probability


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


def get_expected_exit_score(state, dice_set):
    # At the end of a turn, the expected score is given as the maximum of the unused categories
    # plus the potential of the next state
    upper_score, scored_categories = decode_state_integer(state)
    # If all categories are already scored, we check for bonus and return the total
    game_over = check_all_ones(scored_categories)
    if game_over:
        if upper_score >= const.UPPER_BONUS_THRESHOLD:
            return const.UPPER_BONUS
        else:
            return 0

    max_score = 0
    max_score_category = None
    keys = list(const.combinations.keys())

    for ix, category in enumerate(scored_categories):
        category_name = keys[ix]
        if category == '0':
            valid = const.combinations[category_name]["validator"](dice_set)
            if valid:
                score = const.combinations[category_name]["scorer"] \
                    (dices=dice_set,
                     die_value=const.combinations[category_name]["required_die_value"],
                     die_count=const.combinations[category_name]["required_die_count"])
                if score >= max_score:
                    max_score = score
                    max_score_category = category_name
            else:
                score = 0
                if score >= max_score:
                    max_score = score
                    max_score_category = category_name

    index_of_max_score_category = list(const.combinations.keys()).index(max_score_category)
    new_scored_categories = scored_categories[:index_of_max_score_category] + \
                            "1" + scored_categories[index_of_max_score_category + 1:]
    if const.combinations[max_score_category]["upper_section"]:
        upper_score += max_score

    return upper_score, max_score, new_scored_categories


def check_all_ones(string):
    return all(char == '1' for char in string)


def get_probability_of_dice_set(dice_set):
    counts = count_dice_faces(dice_set)
    return probability_of_dice_set_by_counts(counts)


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


def get_states(states, remaining_to_score):
    filtered_states = []
    for start_state in states:
        upper_score, scored_categories = decode_state_integer(start_state)
        if is_remaining_to_score(scored_categories, remaining_to_score):
            filtered_states.append(start_state)
    return filtered_states


def is_remaining_to_score(s, num_categories):
    return num_categories == len(s) - s.count('1')


def size_graph():
    # First we will generate all start states. A start state consists of:
    # - scored categories (13 bits)
    # - upper score (0-63 or 64+, 6 bits)
    # We store each state as a 19-bit integer for later decoding.
    start_states = generate_start_states()
    # We can reduce the number of start states since for example
    # you cannot have a total score of 4 if you have only scored 3s.
    reduced_start_states = reduce_start_states(start_states)
    print("There are " + str(len(reduced_start_states)) + " states at the beginning.")

    print("There are 252 edges from the entry state to the first state of the dices.")

    # There are 252 ways you can roll five dices with six faces assuming order does not matter.
    first_dice_states = generate_dice_states(dices=[1, 1, 1, 1, 1], num_dices_reroll=5)
    print("There are " + str(len(first_dice_states)) + " states since we roll five 6-sided dices.")

    # There are 4368 edges out from the set of dices
    first_reroll_edges = generate_reroll_edges(first_dice_states)
    total_first_edges = sum(len(lst) for lst in first_reroll_edges.values())
    print("There are " + str(total_first_edges) + " edges out from the first set of dices.")

    # There are 462 distinct outcomes from to rerolling a specific set of 5 6-sided dices.
    reroll_states = generate_reroll_states(first_dice_states[0])
    print("There are " + str(len(reroll_states)) + " states possible when rerolling a specific set of 5 6-sided dices.")

    # There are 4368 edges out from the second set of dices
    first_reroll_edges = generate_reroll_edges(first_dice_states)
    total_first_edges = sum(len(lst) for lst in first_reroll_edges.values())
    print("There are " + str(total_first_edges) + " edges out from the second set of dices.")

    # There are 252 new ways the dices can be configured after the first reroll
    second_dice_states = generate_dice_states(dices=[1, 1, 1, 1, 1], num_dices_reroll=5)
    print("There are " + str(len(second_dice_states)) + " states since we roll five 6-sided dices.")

    # There are 4368 edges out from the set of dices
    first_reroll_edges = generate_reroll_edges(first_dice_states)
    total_first_edges = sum(len(lst) for lst in first_reroll_edges.values())
    print("There are " + str(total_first_edges) + " edges out from the first set of dices.")

    # There are 462 distinct ways to choose how many dices to keep, i.e. 5,4,3,2,1 and 0 dices.
    print("There are 462 states representing possible choices of which dices to keep.")

    # There are 4368 edges out from the first roll
    second_reroll_edges = generate_reroll_edges(second_dice_states)
    total_second_edges = sum(len(lst) for lst in second_reroll_edges.values())
    print("There are " + str(total_second_edges) + " edges out from the second set of dices.")

    # Finally, there are 252 ways the dices can be configured when the player is done
    third_dice_states = generate_dice_states(dices=[1, 1, 1, 1, 1], num_dices_reroll=5)
    print("There are " + str(len(third_dice_states)) + " states since we roll five 6-sided dices.")

    print("There are 3276 edges since there are 13 ways to score each of the 252 dice sets.")

    print("")
    print("Nodes: There are " + str(len(reduced_start_states)) + " nodes in total.")
    print("States: There are 1 + 3 * 252 + 2 * 462 = 1681 states in total for each node.")
    print("Edges: There are 252 + 4 * 4368 + 3276 = 21000 edges in total for each node.")
    total_edges = len(reduced_start_states) * 21000
    print("There are " + str(total_edges) + " edges in total.")


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


def generate_reroll_edges(dice_sets):
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
        dice_set_edges = generate_reroll_edges_from_dice_set(dice_set)
        edges[reroll_filter].extend(dice_set_edges[reroll_filter])
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
        upper_score, scored_categories = decode_state_integer(start_state)
        if upper_score > 0:
            if not check_all_zeros(scored_categories[0:6]):
                reduced_start_states.append(start_state)

    return reduced_start_states


def check_all_zeros(binary_string):
    # Check if all characters in the string are '0'
    return all(char == '0' for char in binary_string)


def generate_dice_states(dices, num_dices_reroll):
    dice_faces = [1, 2, 3, 4, 5, 6]
    if num_dices_reroll == 0:
        return [dices]
    return [dices[:-num_dices_reroll] + list(comb) for comb in
            itertools.combinations_with_replacement(dice_faces, num_dices_reroll)]


def generate_start_states():
    """Generate all possible start states."""
    start_states = []

    # There are 2^6 = 64 possible states for the upper score
    for upper_score in range(0, 64):
        upper_score_binary = bin(upper_score)[2:].zfill(6)

        # There are 2^13 = 8192 possible states for the scored categories
        for scored_categories in range(8192):
            scored_categories_binary = bin(scored_categories)[2:].zfill(13)

            # We can now store each start-state as a 19-bit integer for later decoding.
            # Note that the fact that Yatzy is tri-state means not all integers will be used.
            start_state_integer = int(upper_score_binary + scored_categories_binary, 2)
            start_states.append(start_state_integer)

    return start_states


def decode_state_integer(start_state_integer):
    binary = bin(start_state_integer)[2:].zfill(19)
    upper_score = int(binary[:6], 2)
    scored_categories = binary[6:]
    return upper_score, scored_categories


if __name__ == "__main__":
    # size_graph()

    build_graph()
