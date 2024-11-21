import gzip
import os
import pickle
import time
from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement
from itertools import product
from math import factorial

import numpy as np

import yatzy.mechanics.const as const

# Constants
NUM_DICE_COMBINATIONS = 252  # Total unique dice combinations
NUM_REROLL_VECTORS = 6  # Total unique reroll vectors


# Precompute unique dice combinations and reroll vectors with their indices
def precompute_indices():
    dice_combinations = list(combinations_with_replacement(range(1, 7), 5))  # All unique dice combinations
    reroll_vectors = [(r, 5 - r) for r in range(6)]  # (rerolled, locked)

    # Create dictionaries that map each combination to a unique index
    dice_to_index = {comb: i for i, comb in enumerate(dice_combinations)}
    reroll_to_index = {rv: i for i, rv in enumerate(reroll_vectors)}

    return dice_combinations, dice_to_index, reroll_to_index


# Calculate the probability based on reroll vector
def calculate_probability(initial_dice, target_dice, reroll_vector):
    DICE_PROB = 1 / 6
    reroll_prob = 1.0
    locked_prob = 1.0

    for i in range(5):
        if reroll_vector[i] == 1:  # Rerolled dice
            reroll_prob *= DICE_PROB
        else:  # Locked dice
            locked_prob *= 1 if initial_dice[i] == target_dice[i] else 0

    return reroll_prob * locked_prob


# Precompute transition probabilities
def precompute_transition_probabilities(dice_to_index, reroll_to_index, V):
    P1 = np.zeros((NUM_DICE_COMBINATIONS, NUM_DICE_COMBINATIONS))
    P2 = np.zeros((NUM_DICE_COMBINATIONS, NUM_DICE_COMBINATIONS))

    for initial_dice in combinations_with_replacement(range(1, 7), 5):
        for target_dice in combinations_with_replacement(range(1, 7), 5):
            for rerolled_count, locked_count in reroll_to_index.keys():
                # Simulate probabilities
                reroll_vector = [1] * rerolled_count + [0] * locked_count
                prob = calculate_probability(initial_dice, target_dice, reroll_vector)

                initial_index = dice_to_index[tuple(sorted(initial_dice))]
                target_index = dice_to_index[tuple(sorted(target_dice))]

                # Update P1 for the first transition
                P1[initial_index, target_index] += prob

                # Update P2 for the second transition
                P2[initial_index, target_index] += prob

    # Apply the fix to P2: Zero-out transitions leading to invalid `dice_set_3` states
    P2[:, V == 0] = 0

    # Normalize rows while handling zero rows correctly for P2
    row_sums_p2 = P2.sum(axis=1, keepdims=True)
    non_zero_rows_p2 = row_sums_p2.ravel() != 0
    P2[non_zero_rows_p2] /= row_sums_p2[non_zero_rows_p2, :]
    P2[~non_zero_rows_p2] = 0

    # Identify invalid dice_set_2 states in P2
    invalid_states = row_sums_p2.ravel() == 0

    # Zero-out invalid rows in P1
    P1[invalid_states, :] = 0

    # Normalize rows for valid states in P1 only
    row_sums_p1 = P1.sum(axis=1, keepdims=True)
    valid_rows_p1 = ~invalid_states  # Mask for valid rows
    P1[valid_rows_p1] /= row_sums_p1[valid_rows_p1, :]
    P1[~valid_rows_p1] = 0  # Ensure invalid rows remain zero

    return P1, P2


def calculate_expected_value(P1, P2, V):
    """
    Compute the expected value for all dice_set_1 configurations.
    """
    # Zero out invalid transitions in P2
    P2_fixed = P2.copy()
    P2_fixed[:, V == 0] = 0  # Remove transitions to invalid dice_set_3 states

    # Normalize P2 rows
    row_sums = P2_fixed.sum(axis=1, keepdims=True)
    non_zero_rows = row_sums.ravel() != 0
    P2_fixed[non_zero_rows] /= row_sums[non_zero_rows, :]
    P2_fixed[~non_zero_rows] = 0  # Ensure rows with zero sums remain zero

    # Compute intermediate weighted values for dice_set_2
    weighted_values = P2_fixed @ V

    # Zero out invalid transitions in P1
    P1_fixed = P1.copy()
    invalid_dice_set_2 = (weighted_values == 0)  # Mark invalid dice_set_2 states
    P1_fixed[:, invalid_dice_set_2] = 0  # Remove transitions to invalid dice_set_2 states

    # Normalize P1 rows
    row_sums_p1 = P1_fixed.sum(axis=1, keepdims=True)
    non_zero_rows_p1 = row_sums_p1.ravel() != 0
    P1_fixed[non_zero_rows_p1] /= row_sums_p1[non_zero_rows_p1, :]
    P1_fixed[~non_zero_rows_p1] = 0  # Ensure rows with zero sums remain zero

    # Compute final expected values for dice_set_1
    expected_values = P1_fixed @ weighted_values

    # Ensure expected values for invalid dice_set_3 states are zero
    expected_values[V == 0] = 0

    return expected_values


def build_graph():
    print("\n******* building optimal_policy *******\n")

    print("Generating all start states")
    all_states = generate_all_states()

    expected_state_scores = {}

    for i in range(0, 13):
        print(f"Calculating expected scores for {i} remaining categories")
        current_states = filter_states(all_states, i)
        print("Number of states: ", len(current_states))
        if i > 1:
            break
        for state in current_states:
            upper_score, scored_categories = decode_state_integer(state)
            print("\nProcessing state: ", state)
            print(f"{upper_score:06b} = ", upper_score)
            print(f"{scored_categories:013b}")
            # First we check if this state means the game is over
            game_over = (state & 0x1FFF) == 0x1FFF
            if game_over:
                additional_score = const.UPPER_BONUS if upper_score >= const.UPPER_BONUS_THRESHOLD else 0
                expected_state_scores[state] = additional_score
            else:
                start_time = time.time()

                # Decode state integer
                step_start = time.time()
                upper_score, scored_categories = decode_state_integer(state)
                print(f"Time to decode state integer: {time.time() - step_start:.4f} seconds")

                # Precompute indices
                step_start = time.time()
                dice_combinations, dice_to_index, reroll_to_index = precompute_indices()
                print(f"Time to precompute indices: {time.time() - step_start:.4f} seconds")

                # Calculate rewards
                step_start = time.time()
                V = calculate_rewards(dice_combinations, scored_categories, upper_score)
                print(f"Time to calculate rewards: {time.time() - step_start:.4f} seconds")

                # Precompute transition probabilities
                step_start = time.time()
                P1, P2 = precompute_transition_probabilities(dice_to_index, reroll_to_index, V)
                print(f"Time to precompute transition probabilities: {time.time() - step_start:.4f} seconds")

                # Calculate expected values
                step_start = time.time()
                expected_values = calculate_expected_value(P1, P2, V)
                print(f"Time to calculate expected values: {time.time() - step_start:.4f} seconds")

                # Determine the best expected value
                step_start = time.time()
                best_expected_value = np.max(expected_values)
                print(f"Time to find best expected value: {time.time() - step_start:.4f} seconds")

                # Output the best expected value
                print("Best expected value: ", best_expected_value)

                # Total elapsed time
                print(f"Total time elapsed: {time.time() - start_time:.4f} seconds")


def calculate_rewards(dice_combinations, scored_categories, upper_score):
    """
    Calculate the reward for each `dice_set_3` based on the scoring logic provided.
    Args:
        dice_combinations: List of all unique `dice_set_3` configurations.
        scored_categories: Bitmask of already scored categories.
        upper_score: Current score in the upper section.

    Returns:
        rewards: NumPy array of rewards for each `dice_set_3`.
    """
    rewards = np.zeros(len(dice_combinations), dtype=np.float32)

    for dice_index, exit_dice in enumerate(dice_combinations):
        best_additional_score = 0
        new_upper_score = upper_score
        keys = list(const.combinations.keys())  # List of scoring categories
        best_category_name = None

        for ix, category_name in enumerate(keys):
            # Check if the bit at position `ix` is 0
            if (scored_categories & (1 << (12 - ix))) == 0:
                valid = is_dice_set_valid(category_name, exit_dice)
                if valid:
                    score = get_score(category_name, exit_dice)
                    if score >= best_additional_score:
                        best_additional_score = score
                        best_category_name = category_name
                else:
                    score = 0
                    if score >= best_additional_score:
                        best_additional_score = score
                        best_category_name = category_name

        # If the best category is in the upper section, add `best_additional_score` to `new_upper_score`
        if is_upper_section(best_category_name):
            new_upper_score += best_additional_score

        # Store the reward for this `dice_set_3`
        rewards[dice_index] = best_additional_score

    return rewards


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


def filter_states(states, remaining_to_score):
    filtered_states = []
    for start_state in states:
        upper_score, scored_categories = decode_state_integer(start_state)
        if is_remaining_to_score(scored_categories, remaining_to_score):
            filtered_states.append(start_state)
    return filtered_states


def is_remaining_to_score(scored_categories, num_categories):
    return num_categories == (13 - scored_categories.bit_count())


def generate_all_states():
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


if __name__ == "__main__":
    build_graph()
