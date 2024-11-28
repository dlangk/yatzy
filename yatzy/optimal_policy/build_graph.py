from itertools import combinations_with_replacement, product, combinations
from numba import njit, prange
import numpy as np
import os
import time
from typing import Tuple

# ==============================
# Constants and Category Definitions
# ==============================

# Define category constants as integers (no Enum)
YATZY = 0
CHANCE = 1
FULL_HOUSE = 2
LARGE_STRAIGHT = 3
SMALL_STRAIGHT = 4
FOUR_OF_A_KIND = 5
THREE_OF_A_KIND = 6
TWO_PAIRS = 7
ONE_PAIR = 8
SIXES = 9
FIVES = 10
FOURS = 11
THREES = 12
TWOS = 13
ONES = 14

CATEGORY_COUNT = 15

# Upper section categories bitmask
UPPER_CATEGORIES = [ONES, TWOS, THREES, FOURS, FIVES, SIXES]
UPPER_SECTION_BITMASK = 0
for category in UPPER_CATEGORIES:
    UPPER_SECTION_BITMASK |= (1 << category)

# All categories array for Numba compatibility
ALL_CATEGORIES = np.arange(CATEGORY_COUNT, dtype=np.int32)


# ==============================
# Numba-Accelerated Scoring Function
# ==============================

@njit
def calculate_score_numba(category, dice_set):
    """
    Calculates the score for a specific category based on the dice set.

    :param category: Integer representing the category.
    :param dice_set: NumPy array of five integers representing the dice.
    :return: Score for the category.
    """
    counts = np.zeros(7, dtype=np.int32)  # Index 0 unused for simplicity
    for die in dice_set:
        counts[die] += 1

    if category == YATZY:
        return 50 if np.max(counts) == 5 else 0
    elif category == CHANCE:
        return np.sum(dice_set)
    elif category == FULL_HOUSE:
        has_three = False
        has_two = False
        for cnt in counts[1:]:
            if cnt == 3:
                has_three = True
            elif cnt == 2:
                has_two = True
        return np.sum(dice_set) if has_three and has_two else 0
    elif category == THREE_OF_A_KIND:
        for num in range(6, 0, -1):
            if counts[num] >= 3:
                return num * 3
        return 0
    elif category == FOUR_OF_A_KIND:
        for num in range(6, 0, -1):
            if counts[num] >= 4:
                return num * 4
        return 0
    elif category == ONE_PAIR:
        for num in range(6, 0, -1):
            if counts[num] >= 2:
                return num * 2
        return 0
    elif category == TWO_PAIRS:
        pairs = []
        for num in range(6, 0, -1):
            if counts[num] >= 2:
                pairs.append(num)
                if len(pairs) == 2:
                    break
        if len(pairs) == 2:
            return pairs[0] * 2 + pairs[1] * 2
        return 0
    elif category == SMALL_STRAIGHT:
        return 15 if np.array_equal(dice_set, np.array([1, 2, 3, 4, 5])) else 0
    elif category == LARGE_STRAIGHT:
        return 20 if np.array_equal(dice_set, np.array([2, 3, 4, 5, 6])) else 0
    elif category in [ONES, TWOS, THREES, FOURS, FIVES, SIXES]:
        face = 15 - category  # Correct face mapping
        return counts[face] * face
    else:
        return 0


# ==============================
# Numba-Accelerated Probability Computation
# ==============================

@njit(parallel=True)
def compute_reduced_probabilities_numba(
        unique_combinations,
        combination_lookup,
        index_to_combination,
        unique_reroll_vectors,
        num_unique_combinations,
        num_reroll_vectors,
        num_sides
):
    """
    Computes the transition probabilities between dice combinations given reroll vectors.

    :param unique_combinations: 2D NumPy array of unique dice combinations.
    :param combination_lookup: 1D NumPy array mapping dice set keys to combination indices.
    :param index_to_combination: 2D NumPy array mapping indices to dice combinations.
    :param unique_reroll_vectors: 2D NumPy array of all possible reroll vectors.
    :param num_unique_combinations: Total number of unique dice combinations.
    :param num_reroll_vectors: Total number of reroll vectors.
    :param num_sides: Number of sides on each die.
    :return: 3D NumPy array of probabilities.
    """
    probabilities = np.zeros((num_unique_combinations, num_reroll_vectors, num_unique_combinations), dtype=np.float32)

    for comb_idx in prange(num_unique_combinations):
        initial_combination = unique_combinations[comb_idx]
        for reroll_vector_idx in range(num_reroll_vectors):
            reroll_vector = unique_reroll_vectors[reroll_vector_idx]
            rerolled_count = 0
            # Count rerolled dice
            for die_idx in range(5):  # Assuming 5 dice
                if reroll_vector[die_idx] == 1:
                    rerolled_count += 1
            if rerolled_count == 0:
                probabilities[comb_idx, reroll_vector_idx, comb_idx] = 1.0
                continue
            num_combinations = num_sides ** rerolled_count
            for outcome in range(num_combinations):
                rerolled_values = np.empty(5, dtype=np.int32)  # Max 5 dice
                temp = outcome
                for i in range(rerolled_count):
                    rerolled_values[i] = (temp % num_sides) + 1
                    temp = temp // num_sides
                # Combine kept and rerolled values into new_dice
                new_dice = np.empty(5, dtype=np.int32)
                rerolled_idx = 0
                for die_idx in range(5):
                    if reroll_vector[die_idx] == 0:
                        new_dice[die_idx] = initial_combination[die_idx]
                    else:
                        new_dice[die_idx] = rerolled_values[rerolled_idx]
                        rerolled_idx += 1
                # Sort the new_dice
                new_dice_sorted = np.sort(new_dice)
                # Compute key
                key = 0
                for die in new_dice_sorted:
                    key = key * 6 + (die - 1)
                new_comb_idx = combination_lookup[key]
                probabilities[comb_idx, reroll_vector_idx, new_comb_idx] += 1.0 / num_combinations
    return probabilities


# ==============================
# Numba-Accelerated State Operations
# ==============================

@njit
def is_game_over_numba(scored_categories):
    return scored_categories == 32767  # (1 << 15) - 1


@njit
def get_upper_bonus_numba(upper_score):
    return 35 if upper_score >= 63 else 0


@njit
def apply_scoring_numba(state_upper_score, state_scored_categories, category, score, upper_section_bitmask):
    """
    Applies scoring to the current state and returns the new state ID.

    :param state_upper_score: Current upper score (0-63).
    :param state_scored_categories: Bitmask of scored categories.
    :param category: Category to score.
    :param score: Score obtained from the category.
    :param upper_section_bitmask: Bitmask indicating upper section categories.
    :return: New state ID as an integer.
    """
    # Update upper score if necessary
    if (1 << category) & upper_section_bitmask:
        new_upper_score = state_upper_score + score
        if new_upper_score > 63:
            new_upper_score = 63
    else:
        new_upper_score = state_upper_score
    # Update scored categories
    new_scored_categories = state_scored_categories | (1 << category)
    # Compute new state ID
    new_state_id = (new_upper_score << 15) | (new_scored_categories & 0x7FFF)
    return new_state_id


# ==============================
# Helper Functions
# ==============================

def generate_unique_dice_combinations(num_dice: int, num_sides: int) -> np.ndarray:
    """
    Generates all unique combinations of dice rolls (order doesn't matter).

    :param num_dice: Number of dice.
    :param num_sides: Number of sides on each die.
    :return: 2D NumPy array of unique dice combinations.
    """
    return np.array(list(combinations_with_replacement(range(1, num_sides + 1), num_dice)), dtype=np.int32)


def create_combination_lookup(unique_combinations):
    """
    Creates a lookup table mapping each possible dice set key to its combination index.

    :param unique_combinations: 2D NumPy array of unique dice combinations.
    :return: 1D NumPy array serving as the lookup table.
    """
    lookup_size = 6 ** 5  # 7776
    combination_lookup = np.full((lookup_size,), -1, dtype=np.int32)
    for idx in range(unique_combinations.shape[0]):
        comb = unique_combinations[idx]
        key = 0
        for die in comb:
            key = key * 6 + (die - 1)
        combination_lookup[key] = idx
    return combination_lookup


def create_index_to_combination(unique_combinations):
    """
    Creates a mapping from combination indices to dice combinations.

    :param unique_combinations: 2D NumPy array of unique dice combinations.
    :return: 2D NumPy array mapping indices to dice combinations.
    """
    return unique_combinations.copy()


# ==============================
# Probability Calculator Class
# ==============================

class ProbabilityCalculator:
    def __init__(
            self,
            num_dice: int = 5,
            num_sides: int = 6,
            probabilities_filename: str = 'probabilities_reduced.npy',
    ):
        """
        Initializes the ProbabilityCalculator.

        :param num_dice: Number of dice in the game.
        :param num_sides: Number of sides on each die.
        :param probabilities_filename: Filename to store or load the reduced probabilities.
        """
        self.num_dice = num_dice
        self.num_sides = num_sides
        self.probabilities_filename = probabilities_filename

        # Generate unique dice combinations
        self.unique_dice_combinations = generate_unique_dice_combinations(num_dice, num_sides)
        self.num_unique_combinations = self.unique_dice_combinations.shape[0]

        # Create lookup tables
        self.combination_lookup = create_combination_lookup(self.unique_dice_combinations)
        self.index_to_combination = create_index_to_combination(self.unique_dice_combinations)

        # Generate all possible reroll vectors (32 for 5 dice)
        self.unique_reroll_vectors = np.array(list(product([0, 1], repeat=num_dice)), dtype=np.int32)
        self.n_reroll_vec = self.unique_reroll_vectors.shape[0]

        # Precompute scores
        self.precomputed_scores = precompute_scores_numba(
            self.unique_dice_combinations,
            self.combination_lookup,
            self.index_to_combination
        )

        # Load or compute probabilities
        self.probabilities = self.load_or_compute_probabilities()

    def load_or_compute_probabilities(self):
        """
        Loads precomputed reduced probabilities from disk or computes them if not found.

        :return: 3D NumPy array containing reduced transition probabilities.
        """
        if os.path.exists(self.probabilities_filename):
            print("Loading precomputed reduced probabilities from disk...")
            try:
                probabilities = np.load(self.probabilities_filename, mmap_mode='r')
                # Validate shape
                expected_shape = (self.num_unique_combinations, self.n_reroll_vec, self.num_unique_combinations)
                if probabilities.shape != expected_shape:
                    print(
                        f"Existing probabilities file has shape {probabilities.shape}, expected {expected_shape}. Recomputing...")
                    os.remove(self.probabilities_filename)
                    probabilities = self.compute_and_save_probabilities()
                else:
                    print("Probabilities loaded successfully.")
            except Exception as e:
                print(f"Failed to load probabilities: {e}. Recomputing...")
                os.remove(self.probabilities_filename)
                probabilities = self.compute_and_save_probabilities()
        else:
            print("Probabilities file not found. Computing probabilities...")
            probabilities = self.compute_and_save_probabilities()
        return probabilities

    def compute_and_save_probabilities(self):
        """
        Computes the reduced transition probabilities and saves them to disk.

        :return: 3D NumPy array of computed probabilities.
        """
        probabilities = compute_reduced_probabilities_numba(
            self.unique_dice_combinations,
            self.combination_lookup,
            self.index_to_combination,
            self.unique_reroll_vectors,
            self.num_unique_combinations,
            self.n_reroll_vec,
            self.num_sides
        )
        np.save(self.probabilities_filename, probabilities)
        print(f"Probabilities computed and saved to {self.probabilities_filename}.")
        return probabilities

    def calculate_expected_values(self, dice_set_idx: int, rewards_vector: np.ndarray) -> np.ndarray:
        """
        Calculates the expected values for all reroll vectors given a rewards vector.

        :param dice_set_idx: Index of the current dice set.
        :param rewards_vector: 1D NumPy array of rewards for each unique dice combination.
        :return: 1D NumPy array of expected values for each reroll vector.
        """
        return np.dot(self.probabilities[dice_set_idx], rewards_vector)

    def get_best_reroll_vector(self, dice_set_idx: int, rewards_vector: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Determines the best reroll vector for a given dice set based on expected rewards.

        :param dice_set_idx: Index of the current dice set.
        :param rewards_vector: 1D NumPy array of rewards for each unique dice combination.
        :return: Tuple containing the best reroll vector and its expected value.
        """
        expected_values = self.calculate_expected_values(dice_set_idx, rewards_vector)
        best_reroll_idx = np.argmax(expected_values)
        return self.unique_reroll_vectors[best_reroll_idx], expected_values[best_reroll_idx]


# ==============================
# Yatzy State Class
# ==============================

class YatzyState:
    def __init__(self, upper_score: int, scored_categories: int):
        """
        Initializes the YatzyState with integer representations.

        :param upper_score: An integer (0-63) representing the upper score.
        :param scored_categories: An integer (bitmask) representing scored categories.
        """
        # Clamp upper_score between 0 and 63
        self.upper_score = upper_score
        self.upper_score_bin = bin(upper_score)[2:].zfill(6)
        self.scored_categories = scored_categories
        self.scored_categories_bin = bin(scored_categories)[2:].zfill(15)
        self.state_id = (min(max(upper_score, 0), 63) << 15) | (scored_categories & 0x7FFF)

    def is_game_over(self) -> bool:
        """
        Checks if the game is over (i.e., all categories have been scored).

        :return: True if the game is over, False otherwise.
        """
        return is_game_over_numba(self.state_id & 0x7FFF)

    def apply_scoring(self, category: int, score: int) -> 'YatzyState':
        """
        Applies the scoring for a selected category, updating the game state accordingly.

        :param category: The category to score.
        :param score: The score obtained from the category.
        :return: A new YatzyState instance representing the updated state.
        """
        new_state_id = apply_scoring_numba(
            (self.state_id >> 15) & 0x3F,  # upper_score (6 bits)
            self.state_id & 0x7FFF,  # scored_categories (15 bits)
            category,
            score,
            UPPER_SECTION_BITMASK
        )
        return YatzyState(
            upper_score=(new_state_id >> 15) & 0x3F,
            scored_categories=new_state_id & 0x7FFF
        )

    def get_upper_bonus(self) -> int:
        """
        Determines if the upper score qualifies for a bonus.

        :return: 35 if upper_score >= 63, otherwise 0.
        """
        upper_score = (self.state_id >> 15) & 0x3F
        return get_upper_bonus_numba(upper_score)

    def get_current_upper_score(self) -> int:
        """
        Retrieves the current upper score.

        :return: Current upper score as an integer.
        """
        return (self.state_id >> 15) & 0x3F

    def get_scored_categories(self) -> int:
        """
        Retrieves the current scored categories bitmask.

        :return: Scored categories as an integer bitmask.
        """
        return self.state_id & 0x7FFF

    def __str__(self):
        """
        Returns a string representation of the YatzyState.

        :return: String showing state_id, upper_score, and scored_categories.
        """
        return (
            f"upper_score: {self.upper_score_bin} ({self.upper_score}) "
            f"scored_categories: {self.scored_categories_bin} ({self.scored_categories})"
        )


# ==============================
# Precompute Scores Function
# ==============================

@njit(parallel=True)
def precompute_scores_numba(unique_combinations, combination_lookup, index_to_combination):
    """
    Precomputes the scores for all categories and dice combinations.

    :param unique_combinations: 2D NumPy array of unique dice combinations.
    :param combination_lookup: 1D NumPy array mapping dice set keys to combination indices.
    :param index_to_combination: 2D NumPy array mapping indices to dice combinations.
    :return: 2D NumPy array of precomputed scores.
    """
    precomputed_scores = np.zeros((CATEGORY_COUNT, unique_combinations.shape[0]), dtype=np.int32)
    for category in prange(CATEGORY_COUNT):
        for comb_idx in prange(unique_combinations.shape[0]):
            dice_set = index_to_combination[comb_idx]
            precomputed_scores[category, comb_idx] = calculate_score_numba(category, dice_set)
    return precomputed_scores


# ==============================
# Numba-Accelerated Best Scores Function
# ==============================

@njit(parallel=True)
def get_best_scores_numba(precomputed_scores, scored_categories, state_upper_score, upper_section_bitmask,
                          category_count, num_unique_combinations):
    """
    Determines the best score, category, and new state for each dice set.

    :param precomputed_scores: 2D NumPy array of precomputed scores (categories x dice_sets).
    :param scored_categories: Integer bitmask representing scored categories.
    :param state_upper_score: Current upper score (0-63).
    :param upper_section_bitmask: Bitmask indicating upper section categories.
    :param category_count: Total number of categories.
    :param num_unique_combinations: Total number of unique dice combinations.
    :return: Tuple of three NumPy arrays:
             - best_scores: Best score for each dice set.
             - best_categories: Best category for each dice set.
             - new_state_ids: New state ID after scoring for each dice set.
    """
    best_scores = np.zeros(num_unique_combinations, dtype=np.int32)
    best_categories = np.full(num_unique_combinations, -1, dtype=np.int32)
    new_state_ids = np.zeros(num_unique_combinations, dtype=np.int32)
    for dice_set_idx in prange(num_unique_combinations):
        max_score = -1
        best_cat = -1
        for category in range(category_count):
            if (scored_categories & (1 << category)) == 0:
                score = precomputed_scores[category, dice_set_idx]
                if score > max_score:
                    max_score = score
                    best_cat = category
        if best_cat != -1:
            best_scores[dice_set_idx] = max_score
            best_categories[dice_set_idx] = best_cat
            # Compute new_state_id
            if (1 << best_cat) & upper_section_bitmask:
                new_upper_score = state_upper_score + max_score
                if new_upper_score > 63:
                    new_upper_score = 63
            else:
                new_upper_score = state_upper_score
            new_scored_categories = scored_categories | (1 << best_cat)
            new_state_id = (new_upper_score << 15) | (new_scored_categories & 0x7FFF)
            new_state_ids[dice_set_idx] = new_state_id
        else:
            # No category to score, keep the state unchanged
            best_scores[dice_set_idx] = 0
            best_categories[dice_set_idx] = -1
            new_state_ids[dice_set_idx] = (state_upper_score << 15) | (scored_categories & 0x7FFF)
    return best_scores, best_categories, new_state_ids


# ==============================
# Numba-Accelerated Factorial Lookup
# ==============================

@njit
def precompute_factorial_lookup_numba():
    """
    Precomputes factorial values from 0! to 5! for quick access.

    :return: 1D NumPy array containing factorial values.
    """
    lookup = np.zeros(6, dtype=np.int32)
    lookup[0] = 1
    for i in range(1, 6):
        lookup[i] = lookup[i - 1] * i
    return lookup


# ==============================
# Precompute Dice Set Probabilities
# ==============================

@njit(parallel=True)
def compute_dice_set_probabilities_numba(unique_combinations, factorial_lookup):
    """
    Computes the probability of each unique dice set occurring in an initial roll.

    :param unique_combinations: 2D NumPy array of unique dice combinations.
    :param factorial_lookup: 1D NumPy array of precomputed factorial values.
    :return: 1D NumPy array of probabilities for each dice set.
    """
    probabilities = np.zeros(unique_combinations.shape[0], dtype=np.float32)
    total_permutations = 6 ** 5  # 7776

    for idx in prange(unique_combinations.shape[0]):
        counts = np.zeros(6, dtype=np.int32)  # Counts for die faces 1-6
        for die in unique_combinations[idx]:
            counts[die - 1] += 1

        # Compute multinomial coefficient: 5! / (n1! * n2! * ... * n6!)
        multinomial = 120  # 5!
        for count in counts:
            if count > 0:
                multinomial //= factorial_lookup[count]

        probabilities[idx] = multinomial / total_permutations

    return probabilities

def get_best_reroll_vector(
        E_array,
        probabilities,
        unique_reroll_vectors,
        unique_combinations,
        num_unique_combinations,
        best_scores,
        new_state_ids,
        dice_set,
        n
):
    """
    Optimized recursive function to get the best reroll vector for a given state and level (n).
    """

    @njit
    def compute_reroll_vector_scores(
            dice_set_idx,
            reroll_vectors,
            probabilities,
            num_combinations,
            best_scores,
            new_state_ids,
            E_array,
            level,
            memo
    ):
        """
        Compute scores for all reroll vectors at a given level with memoization.
        """
        reroll_vector_scores = np.zeros(len(reroll_vectors), dtype=np.float32)
        for reroll_vector_idx in prange(len(reroll_vectors)):
            dice_set_next_p_dist = probabilities[dice_set_idx, reroll_vector_idx, :]
            dice_set_next_p_non_zero_idx = np.nonzero(dice_set_next_p_dist)[0]
            dice_set_next_scores = np.zeros(num_combinations, dtype=np.float32)

            for dice_set_next_idx in dice_set_next_p_non_zero_idx:
                if level == 0:
                    dice_set_next_scores[dice_set_next_idx] = (
                            dice_set_next_p_dist[dice_set_next_idx]
                            * (
                                    best_scores[dice_set_next_idx]
                                    + E_array[new_state_ids[dice_set_next_idx]]
                            )
                    )
                else:
                    # Check memoized results to avoid recomputation
                    if memo[level - 1, dice_set_next_idx] != -1:
                        best_score = memo[level - 1, dice_set_next_idx]
                    else:
                        _, best_score = compute_reroll_vector_scores(
                            dice_set_next_idx,
                            reroll_vectors,
                            probabilities,
                            num_combinations,
                            best_scores,
                            new_state_ids,
                            E_array,
                            level - 1,
                            memo
                        )
                        memo[level - 1, dice_set_next_idx] = best_score
                    dice_set_next_scores[dice_set_next_idx] = (
                            dice_set_next_p_dist[dice_set_next_idx] * best_score
                    )

            reroll_vector_scores[reroll_vector_idx] = np.sum(dice_set_next_scores)
        return reroll_vector_scores, np.max(reroll_vector_scores)

    # Locate dice_set index
    dice_set_idx = np.where(np.all(unique_combinations == dice_set, axis=1))[0][0]

    if n == 0:
        # Base case: return best score and state value
        return best_scores[dice_set_idx] + E_array[new_state_ids[dice_set_idx]]

    # Initialize memoization array with -1 (indicating uncomputed values)
    memo = np.full((n, num_unique_combinations), -1, dtype=np.float32)

    # Compute reroll vector scores for level `n`
    reroll_vectors = unique_reroll_vectors
    reroll_vector_scores, best_score = compute_reroll_vector_scores(
        dice_set_idx, reroll_vectors, probabilities, num_unique_combinations,
        best_scores, new_state_ids, E_array, n - 1, memo
    )

    # Get the best reroll vector index
    best_reroll_vector_idx = np.argmax(reroll_vector_scores)
    return best_reroll_vector_idx, best_score


def main():
    start_time = time.time()
    calc = ProbabilityCalculator()
    dice_set_p = compute_dice_set_probabilities_numba(calc.unique_dice_combinations,
                                                      precompute_factorial_lookup_numba())

    # Pre-allocate E_array instead of using a dict
    num_states = (1 << CATEGORY_COUNT) * 64  # 2^15 * 64 = 2,097,152
    E_array = np.zeros(num_states, dtype=np.float32)

    # Populate end states
    for num_scored in range(CATEGORY_COUNT, -1, -1):  # 15 down to 0
        for scored_combination in combinations(range(CATEGORY_COUNT), num_scored):
            scored_categories_bitmask = sum(1 << cat for cat in scored_combination)
            for upper_score in range(64):
                state = YatzyState(upper_score=upper_score, scored_categories=scored_categories_bitmask)
                if state.is_game_over():
                    E_array[state.state_id] = state.get_upper_bonus()

    print(f"Initialization time: {time.time() - start_time:.2f} seconds")

    # Test for a specific state and dice set
    scored_categories_bitmask = ((1 << CATEGORY_COUNT) - 1) ^ (1 << FIVES)
    upper_score = 0
    state = YatzyState(upper_score=upper_score, scored_categories=scored_categories_bitmask)

    best_scores, best_categories, new_state_ids = get_best_scores_numba(
        calc.precomputed_scores,
        state.get_scored_categories(),
        state.get_current_upper_score(),
        UPPER_SECTION_BITMASK,
        CATEGORY_COUNT,
        calc.num_unique_combinations
    )

    expected_value = 0
    for dice_set_1 in calc.unique_dice_combinations:
        dice_set_1_idx = np.where(np.all(calc.unique_dice_combinations == dice_set_1, axis=1))[0][0]
        reroll_vector_1_best_idx, reroll_vector_1_best_score = get_best_reroll_vector(
            E_array, calc.probabilities, calc.unique_reroll_vectors, calc.unique_dice_combinations,
            calc.num_unique_combinations, best_scores, new_state_ids, dice_set_1, 2
        )
        dice_set_value = dice_set_p[dice_set_1_idx] * reroll_vector_1_best_score
        expected_value += dice_set_value
        print(f"Dice Set: {dice_set_1} | probability: {dice_set_p[dice_set_1_idx]:.5f} | "
              f"best score: {reroll_vector_1_best_score:.5f} | value: {dice_set_value:.5f}")
    print(f"Expected Value: {expected_value:.5f}")

# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    main()