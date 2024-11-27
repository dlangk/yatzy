import numpy as np
from itertools import combinations_with_replacement, product, permutations
from collections import Counter

from math import factorial
from tqdm import tqdm
import multiprocessing as mp
import os
import time
from typing import Tuple, Dict, Any, Optional


def generate_unique_dice_combinations(num_dice: int, num_sides: int) -> np.ndarray:
    """
    Generates all unique combinations of dice rolls (order doesn't matter).

    Parameters:
    - num_dice: Number of dice.
    - num_sides: Number of sides on each die.

    Returns:
    - unique_combinations: An array of unique dice combinations.
    """
    dice_values = np.arange(1, num_sides + 1, dtype=np.uint8)
    combinations = combinations_with_replacement(dice_values, num_dice)
    unique_combinations = np.array(list(combinations), dtype=np.uint8)
    return unique_combinations


class ProbabilityCalculator:
    def __init__(
            self,
            num_dice: int = 5,
            num_sides: int = 6,
            probabilities_filename: str = 'probabilities_reduced.npy',
    ):
        """
        Initializes the ProbabilityCalculator.

        Parameters:
        - num_dice: Number of dice in the game.
        - num_sides: Number of sides on each die.
        - probabilities_filename: Filename to store or load the reduced probabilities.
        """
        self.num_dice = num_dice
        self.num_sides = num_sides
        self.probabilities_filename = probabilities_filename

        # Generate the 252 unique dice combinations (order doesn't matter)
        self.unique_dice_combinations = generate_unique_dice_combinations(num_dice, num_sides)
        self.num_unique_combinations = len(self.unique_dice_combinations)

        # Map each unique combination to an index
        self.combination_to_index = {
            tuple(sorted(comb)): idx for idx, comb in enumerate(self.unique_dice_combinations)
        }
        self.index_to_combination = {
            idx: comb for idx, comb in enumerate(self.unique_dice_combinations)
        }

        # Generate all possible reroll vectors (32 possibilities for 5 dice)
        self.unique_reroll_vectors = np.array(
            list(product([0, 1], repeat=num_dice)), dtype=np.uint8
        )
        self.num_reroll_vectors = self.unique_reroll_vectors.shape[0]

        # Mapping from full dice sets (7776) to unique combinations (252)
        self.full_dice_values = np.arange(1, num_sides + 1, dtype=np.uint8)
        self.all_full_dice_sets = np.array(
            list(product(self.full_dice_values, repeat=num_dice)), dtype=np.uint8
        )
        self.full_to_combination_index = np.array([
            self.combination_to_index[tuple(sorted(dice_set))]
            for dice_set in self.all_full_dice_sets
        ], dtype=np.int32)

        self.num_full_dice_sets = len(self.all_full_dice_sets)

        # Initialize probabilities array
        self.probabilities_shape = (
            self.num_unique_combinations,
            self.num_reroll_vectors,
            self.num_unique_combinations,
        )
        self.probabilities_dtype = np.float32

        # Load or compute probabilities
        self.probabilities = self.load_or_compute_probabilities()

        # Precompute combination probabilities
        self.combination_probabilities = self.calculate_combination_probabilities()

    def calculate_combination_probabilities(self) -> Dict[Tuple[int, ...], float]:
        """
        Calculates and stores the probability of each unique dice combination.

        Returns:
        - combination_probabilities: A dictionary mapping combinations to their probabilities.
        """
        total_permutations = self.num_sides ** self.num_dice  # Total number of possible permutations (6^5)
        combination_probabilities = {}

        for combination in self.unique_dice_combinations:
            counts = Counter(combination)
            numerator = factorial(self.num_dice)
            denominator = 1
            for count in counts.values():
                denominator *= factorial(count)
            num_permutations = numerator // denominator
            probability = num_permutations / total_permutations
            combination_probabilities[tuple(sorted(combination))] = probability

        return combination_probabilities

    def get_dice_set_probability(self, dice_set: np.ndarray) -> float:
        """
        Returns the probability of rolling the given dice set.

        Parameters:
        - dice_set: A NumPy array representing the dice set.

        Returns:
        - probability: The probability of rolling the dice set.
        """
        combination_key = tuple(sorted(dice_set))
        return self.combination_probabilities.get(combination_key, 0.0)

    def load_or_compute_probabilities(self) -> np.memmap:
        """
        Loads precomputed reduced probabilities from disk or computes them if not found.

        Returns:
        - probabilities: A memory-mapped NumPy array containing reduced transition probabilities.
        """
        if os.path.exists(self.probabilities_filename):
            print("Loading precomputed reduced probabilities from disk...")
            probabilities = np.memmap(
                self.probabilities_filename,
                dtype=self.probabilities_dtype,
                mode='r',
                shape=self.probabilities_shape,
            )
        else:
            print("Reduced probabilities file not found. Computing reduced probabilities...")
            probabilities = np.memmap(
                self.probabilities_filename,
                dtype=self.probabilities_dtype,
                mode='w+',
                shape=self.probabilities_shape,
            )
            # Initialize to zeros
            probabilities[:] = 0.0

            # Compute reduced probabilities using multiprocessing
            self.compute_reduced_probabilities()

            # Reload the memmap in read-only mode
            del probabilities  # Close the memmap
            probabilities = np.memmap(
                self.probabilities_filename,
                dtype=self.probabilities_dtype,
                mode='r',
                shape=self.probabilities_shape,
            )

        return probabilities

    def compute_reduced_probabilities(self):
        """
        Computes the reduced transition probabilities using multiprocessing.
        """
        num_processes = mp.cpu_count()
        chunk_size = self.num_unique_combinations // num_processes
        processes = []

        # Create a shared counter and a lock
        manager = mp.Manager()
        progress_counter = manager.Value('i', 0)
        lock = manager.Lock()

        total_iterations = self.num_unique_combinations

        # Start the worker processes
        for i in range(num_processes):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_processes - 1 else self.num_unique_combinations
            p = mp.Process(
                target=compute_reduced_probabilities_chunk,
                args=(
                    start_idx,
                    end_idx,
                    self.probabilities_filename,
                    self.num_unique_combinations,
                    self.num_reroll_vectors,
                    self.combination_to_index,
                    self.index_to_combination,
                    self.unique_reroll_vectors,
                    self.num_dice,
                    self.num_sides,
                    self.full_dice_values,
                    progress_counter,
                    lock,
                ),
            )
            processes.append(p)
            p.start()

        # Start the progress updater process
        progress_process = mp.Process(
            target=update_progress,
            args=(progress_counter, total_iterations, lock),
        )
        progress_process.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()
        progress_process.join()

    def calculate_expected_values(
            self,
            dice_set: np.ndarray,
            rewards_vector: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the expected values for all reroll vectors given a rewards vector.

        Parameters:
        - dice_set: The initial dice set as a NumPy array.
        - rewards_vector: The rewards vector corresponding to each unique dice combination.

        Returns:
        - reroll_exp: Array of expected values for each reroll vector.
        """
        combination_key = tuple(sorted(dice_set))
        dice_combination_index = self.combination_to_index[combination_key]
        reroll_exp = np.zeros(self.num_reroll_vectors)
        for j in range(self.num_reroll_vectors):
            prob_distribution = self.probabilities[dice_combination_index, j, :]
            expected_value = np.dot(prob_distribution, rewards_vector)
            reroll_exp[j] = expected_value
        return reroll_exp

    def get_probability(
            self,
            dice_set_1: np.ndarray,
            reroll_vector: np.ndarray,
            dice_set_2: np.ndarray,
    ) -> float:
        """
        Retrieves the probability of transitioning from dice_set_1 to dice_set_2 given a reroll_vector.

        Parameters:
        - dice_set_1: Initial dice set as a NumPy array.
        - reroll_vector: Reroll vector as a NumPy array.
        - dice_set_2: Target dice set as a NumPy array.

        Returns:
        - probability: The transition probability.
        """
        combination_key_1 = tuple(sorted(dice_set_1))
        combination_key_2 = tuple(sorted(dice_set_2))
        dice_combination_index_1 = self.combination_to_index[combination_key_1]
        dice_combination_index_2 = self.combination_to_index[combination_key_2]
        reroll_vector_index = np.where(
            (self.unique_reroll_vectors == reroll_vector).all(axis=1)
        )[0][0]
        probability = self.probabilities[
            dice_combination_index_1, reroll_vector_index, dice_combination_index_2
        ]
        return probability

    def get_dice_set_probabilities(
            self,
            dice_set_1: np.ndarray,
            reroll_vector: np.ndarray,
    ) -> Dict[Tuple[int, ...], float]:
        """
        Returns the probabilities of reaching any of the 252 unique dice combinations
        starting from dice_set_1 and using the given reroll_vector.

        Parameters:
        - dice_set_1: The initial dice set as a NumPy array.
        - reroll_vector: The reroll vector as a NumPy array.

        Returns:
        - probabilities_dict: A dictionary mapping each reachable dice combination (as a tuple) to its probability.
        """
        # Convert the initial dice set to a sorted tuple to match the combination keys
        combination_key_1 = tuple(sorted(dice_set_1))
        dice_combination_index_1 = self.combination_to_index[combination_key_1]

        # Find the index of the reroll vector
        reroll_vector_index = np.where(
            (self.unique_reroll_vectors == reroll_vector).all(axis=1)
        )[0][0]

        # Retrieve the probability distribution for the given initial combination and reroll vector
        probabilities = self.probabilities[dice_combination_index_1, reroll_vector_index, :]

        # Create a dictionary mapping combinations to their probabilities
        probabilities_dict = {}
        for idx, prob in enumerate(probabilities):
            if prob > 0:
                combination = self.index_to_combination[idx]
                probabilities_dict[tuple(map(int, combination))] = prob

        return probabilities_dict

    def get_best_reroll_vector(
            self,
            dice_set: np.ndarray,
            rewards_vector: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Determines the best reroll vector for a given initial dice set.

        Parameters:
        - dice_set: The initial dice set as a NumPy array.
        - rewards_vector: The rewards vector corresponding to each unique dice combination.

        Returns:
        - best_reroll_vector: The reroll vector with the highest expected value.
        - max_expectation: The maximum expected value.
        """
        reroll_exp = self.calculate_expected_values(dice_set, rewards_vector)
        best_reroll_idx = np.argmax(reroll_exp)
        best_reroll_vector = self.unique_reroll_vectors[best_reroll_idx]
        max_expectation = reroll_exp[best_reroll_idx]
        return best_reroll_vector, max_expectation

    def calculate_rewards_vector(self, calculate_reward_func) -> np.ndarray:
        """
        Calculates the rewards vector using the provided scoring function.

        Parameters:
        - calculate_reward_func: A function that calculates the reward for a dice combination.

        Returns:
        - rewards_vector: Array of rewards for each unique dice combination.
        """
        rewards_vector = np.array(
            [calculate_reward_func(comb) for comb in self.unique_dice_combinations],
            dtype=np.float32,
        )
        return rewards_vector


def compute_reduced_probabilities_chunk(
        start_idx: int,
        end_idx: int,
        probabilities_filename: str,
        num_unique_combinations: int,
        num_reroll_vectors: int,
        combination_to_index: Dict[Tuple[int, ...], int],
        index_to_combination: Dict[int, np.ndarray],
        unique_reroll_vectors: np.ndarray,
        num_dice: int,
        num_sides: int,
        dice_values: np.ndarray,
        progress_counter: mp.Value,
        lock: mp.Lock,
):
    """
    Computes reduced probabilities for a chunk of unique dice combinations.

    Parameters:
    - start_idx: Start index for the chunk.
    - end_idx: End index for the chunk.
    - probabilities_filename: Filename of the shared probabilities array.
    - num_unique_combinations: Total number of unique dice combinations.
    - num_reroll_vectors: Total number of reroll vectors.
    - combination_to_index: Dictionary mapping combinations to indices.
    - index_to_combination: Dictionary mapping indices to combinations.
    - unique_reroll_vectors: Array of unique reroll vectors.
    - num_dice: Number of dice.
    - num_sides: Number of sides on each die.
    - dice_values: Array of possible dice values.
    - progress_counter: A shared counter for progress tracking.
    - lock: A lock for synchronizing access to the progress counter.
    """
    # Re-open the memmap in each process
    probabilities = np.memmap(
        probabilities_filename,
        dtype=np.float32,
        mode='r+',
        shape=(num_unique_combinations, num_reroll_vectors, num_unique_combinations),
    )

    local_count = 0
    for comb_idx in range(start_idx, end_idx):
        initial_combination = index_to_combination[comb_idx]
        for reroll_vector_idx, reroll_vector in enumerate(unique_reroll_vectors):
            num_rerolled = reroll_vector.sum()
            if num_rerolled == 0:
                # No reroll; stay in the same combination
                probabilities[comb_idx, reroll_vector_idx, comb_idx] = 1.0
                continue
            kept_positions = np.where(reroll_vector == 0)[0]
            kept_values = initial_combination[kept_positions]
            rerolled_positions = np.where(reroll_vector == 1)[0]
            num_combinations = num_sides ** num_rerolled

            # Generate all possible combinations (with replacement) of the rerolled dice
            possible_rerolls = np.array(list(product(dice_values, repeat=num_rerolled)), dtype=np.uint8)

            for rerolled_values in possible_rerolls:
                new_dice = np.concatenate((kept_values, rerolled_values))
                new_combination = tuple(sorted(new_dice))
                new_comb_idx = combination_to_index[new_combination]
                prob = 1.0 / num_combinations
                probabilities[comb_idx, reroll_vector_idx, new_comb_idx] += prob

        # Update the progress counter
        local_count += 1
        if local_count % 10 == 0:  # Update every 10 iterations to reduce overhead
            with lock:
                progress_counter.value += 10
            local_count = 0

    # Update any remaining counts
    if local_count > 0:
        with lock:
            progress_counter.value += local_count

    # Flush changes to disk
    probabilities.flush()
    del probabilities  # Close the memmap


def update_progress(
        progress_counter: mp.Value, total_iterations: int, lock: mp.Lock
):
    """
    Updates the progress bar.

    Parameters:
    - progress_counter: A shared counter for progress tracking.
    - total_iterations: Total number of iterations.
    - lock: A lock for synchronizing access to the progress counter.
    """
    pbar = tqdm(total=total_iterations, desc='Computing probabilities', position=0)
    last_value = 0
    while True:
        with lock:
            current_value = progress_counter.value
        delta = current_value - last_value
        if delta > 0:
            pbar.update(delta)
            last_value = current_value
        if current_value >= total_iterations:
            break
        time.sleep(0.1)  # Adjust sleep time as needed
    pbar.close()
