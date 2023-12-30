import gzip
import json
import os
import pickle

from collections import Counter
from itertools import combinations, product, combinations_with_replacement
from math import factorial

from pyspark import SparkConf
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, concat_ws, udf, row_number, desc
from pyspark.sql.types import BooleanType

import yatzy.mechanics.const as const


def build_graph():
    print("\n******* building graph *******\n")

    print("Generating all dice sets")
    all_dice_sets = generate_all_dice_sets()

    print("Generating all start states")
    all_start_states = reduce_start_states(generate_start_states())

    print("Generating all reroll masks")
    all_reroll_masks = generate_all_reroll_masks(num_dices=5)

    print("Generating all reroll probabilities")
    all_reroll_probs = compute_all_reroll_probabilities(all_dice_sets, all_reroll_masks)

    print("Generating all dice set probabilities")
    all_dice_set_probabilities = generate_all_dice_set_probabilities(all_dice_sets)

    print("Generate all state integers")
    all_state_integers = generate_all_state_integers()

    print("Generating all reroll edges")
    all_reroll_masks_by_filter = generate_reroll_masks(all_dice_sets)

    print("Generate all possible reroll masks")
    all_possible_reroll_masks = generate_all_possible_reroll_masks(all_dice_sets, all_reroll_masks_by_filter)

    spark = setup_spark()
    PARTITION_NUM = 60

    expected_state_scores = {}
    expected_state_scores = set_exit_state_values(spark, all_start_states, expected_state_scores)

    print("Depth 0 state scores: " + str(list(expected_state_scores.keys())))
    print("Number of Depth 0 state scores: " + str(len(expected_state_scores.keys())))

    DATA_DIR_PATH = "/Users/langkilde/IdeaProjects/yatzy/data/"
    DATA_FILE = "data.json"
    FULL_PATH = DATA_DIR_PATH + DATA_FILE
    if os.path.exists(FULL_PATH):
        # Open the file and load the JSON data
        with open(FULL_PATH, 'r') as file:
            data = json.load(file)
            for state in data:
                expected_state_scores[state] = data[state]
        print("Data loaded successfully")

    print("Updated state scores: " + str(list(expected_state_scores.keys())))
    print("Number of updated state scores: " + str(len(expected_state_scores.keys())))

    all_unfinished_start_states = filter_states_by_unfinished(all_start_states, list(expected_state_scores.keys()))
    print(len(all_unfinished_start_states))


    # We then prepare all dice transition probabilities
    dice_transitions_rdd = spark.sparkContext.parallelize(all_dice_sets) \
        .map(lambda dice_set: (dice_set, get_reroll_filter(dice_set))) \
        .flatMap(lambda row: expand_by_reroll_masks(row[0], row[1], all_reroll_masks_by_filter)) \
        .flatMap(lambda row: expand_by_target_dice_set(row[0], row[1], all_dice_sets)) \
        .map(lambda row: (row[0], row[1], row[2], get_reroll_prob_from_map(all_reroll_probs, row[0], row[1], row[2]))) \
        .filter(lambda row: row[3] > 0).repartition(PARTITION_NUM)

    # Next, we start working our way through all possible game states, starting from the end
    for i in range(1, 13):
        print("Starting with depth: " + str(i))
        states_with_same_depth = filter_states_by_depth(all_unfinished_start_states, i)
        print("Number of states with depth " + str(i) + ": " + str(len(states_with_same_depth)))

        BATCH_SIZE = 1000

        for j in range(0, len(states_with_same_depth), BATCH_SIZE):
            print("Starting batch: " + str(j))
            filtered_states = states_with_same_depth[j: j + BATCH_SIZE]
            print("Number of batched states:" + str(len(filtered_states)))
            states_rdd = spark.sparkContext.parallelize(filtered_states)

            enriched_states_rdd = states_rdd.map(lambda state: (state, decode_state_integer(state))) \
                .map(lambda row: (row[0], row[1][0], row[1][1])) \
                .flatMap(lambda row: expand_by_all_dice_sets(row[0], row[1], row[2], all_dice_sets)) \
                .map(lambda row:
                     (row[0], row[1], row[2], row[3], get_max_exit_score(row[0], row[1], row[2], row[3], False))) \
                .map(lambda row: (row[0], row[1], row[2], row[3], row[4][0], row[4][1], row[4][2])) \
                .map(lambda row:
                     (row[0], row[1], row[2], row[3], row[4], row[5], row[6], encode_state_integer(row[4], row[6]))) \
                .map(lambda row:
                     (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], expected_state_scores[row[7]])) \
                .map(lambda row:
                     (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[5] + row[8]))

            enriched_states_df = enriched_states_rdd \
                .toDF(['state',
                       'upper_score',
                       'scored_categories',
                       'exit_dice_set',
                       'new_upper_score',
                       'max_additional_score',
                       'new_scored_categories',
                       'next_state',
                       'next_state_potential',
                       'exit_dice_set_value']) \
                .withColumn('exit_dice_set', concat_ws('', 'exit_dice_set')).repartition(PARTITION_NUM)

            dice_transitions_df = dice_transitions_rdd.toDF(['r_start', 'r_prim', 'r_target', 'prob']) \
                .withColumn('r_start', concat_ws('', 'r_start')) \
                .withColumn('r_prim', concat_ws('', 'r_prim')) \
                .withColumn('r_target', concat_ws('', 'r_target'))

            joined_df = dice_transitions_df \
                .join(enriched_states_df, col('exit_dice_set') == col('r_target'), 'left') \
                .repartition(PARTITION_NUM)

            second_reroll_df = joined_df.repartition(PARTITION_NUM) \
                .withColumn('target_dice_set_potential', (col('prob') * col('exit_dice_set_value')).cast('float')) \
                .select('state', 'r_start', 'r_prim', 'target_dice_set_potential') \
                .groupBy("state", "r_start", "r_prim") \
                .sum("target_dice_set_potential")
            second_reroll_df = second_reroll_df.repartition(PARTITION_NUM)

            windowSpec = Window.partitionBy("state", "r_start") \
                .orderBy(desc("sum(target_dice_set_potential)"))
            df_with_rownum = second_reroll_df.repartition(PARTITION_NUM) \
                .withColumn("row_num", row_number().over(windowSpec))
            second_reroll_dice_set_values_df = df_with_rownum.filter(df_with_rownum["row_num"] == 1).drop("row_num")
            second_reroll_dice_set_values_df = second_reroll_dice_set_values_df \
                .select('state', 'r_start', 'sum(target_dice_set_potential)') \
                .withColumnRenamed("r_start", "dice_set") \
                .withColumnRenamed("sum(target_dice_set_potential)", "dice_set_value")
            second_reroll_dice_set_values_df = second_reroll_dice_set_values_df.repartition(PARTITION_NUM)

            joined_df = dice_transitions_df \
                .join(second_reroll_dice_set_values_df,
                      dice_transitions_df['r_target'] == second_reroll_dice_set_values_df['dice_set'],
                      how='inner') \
                .withColumn('target_dice_set_potential', (col('prob') * col('dice_set_value')).cast('float')) \
                .groupBy("state", "r_start", "r_prim") \
                .sum("target_dice_set_potential") \
                .alias("reroll_mask_potential")

            windowSpec = Window.partitionBy("state", "r_start") \
                .orderBy(desc("sum(target_dice_set_potential)"))
            df_with_rownum = joined_df.repartition(PARTITION_NUM) \
                .withColumn("row_num", row_number().over(windowSpec))
            joined_df = df_with_rownum.filter(df_with_rownum["row_num"] == 1).drop("row_num")

            final_df = joined_df.rdd \
                .map(lambda row: (row['state'],
                                  decode_state_integer(row['state']),
                                  row['r_start'],
                                  row['sum(target_dice_set_potential)'] *
                                  all_dice_set_probabilities["".join(map(str, row['r_start']))])) \
                .toDF(["state", "upper+cat", "dice_set", "value"])

            state_values_df = final_df.groupBy("state").sum("value").sort("sum(value)", ascending=False)
            state_values_list = state_values_df.rdd.collect()

            for state_value in state_values_list:
                print(state_value)
                expected_state_scores[state_value[0]] = state_value[1]

            print("writing file to disk")
            with open(DATA_DIR_PATH + 'data.json', 'w') as file:
                json.dump(expected_state_scores, file)


def setup_spark():
    master = 'local[8]'
    appName = 'Yatzy'
    conf = SparkConf()
    # Set the executor memory
    conf.setMaster("local[8]")  # Adjust as needed
    conf.set("spark.executor.memory", "50g")
    conf.set("spark.driver.memory", "4g")
    spark = SparkSession.builder.config(conf=conf).appName(appName).master(master).getOrCreate()
    return spark


def calculate_exit_state_values(spark, start_states):
    states = spark.sparkContext.parallelize(start_states)
    # Then we first calculate the expected value of each possible end-state for the game
    exit_states_df = ((states
                       .map(lambda state: (state, decode_state_integer(state)))
                       .map(lambda state: (state[0], state[1][0], state[1][1]))
                       .map(lambda state: (state[0], state[1], state[2], check_all_ones(state[2]))))
                      .map(lambda state: game_over_check(state[0], state[1], state[2], state[3])))
    return exit_states_df


def set_exit_state_values(spark, all_start_states, expected_state_scores):
    start_states = filter_states_by_depth(all_start_states, 0)
    exit_state_values = calculate_exit_state_values(spark, start_states).collect()

    for r in exit_state_values:
        state_int = r[0]
        state_value = r[2]
        expected_state_scores[state_int] = state_value
    return expected_state_scores


def is_string(value):
    return isinstance(value, str)


def number_to_list(num):
    return [int(digit) for digit in str(num)]


is_string_udf = udf(is_string, BooleanType())


def expand_by_reroll_masks(dice_set, reroll_filter, all_reroll_masks_by_filter):
    # For each reroll_filter in all_reroll_masks_by_filter, create a new row combined with the current row's dice_set
    return [(dice_set, reroll_mask) for reroll_mask in all_reroll_masks_by_filter[reroll_filter]]


def expand_by_target_dice_set(dice_set, reroll_mask, all_dice_sets):
    # For each dice_set in all_dice_sets, create a new row combined with the current row's reroll_mask
    return [(dice_set, reroll_mask, target_dice_set) for target_dice_set in all_dice_sets]


def expand_by_all_dice_sets(state, upper_score, scored_categories, all_dice_sets):
    # For each dice_set in all_dice_sets, create a new row combined with the current row's reroll_mask
    return [(state, upper_score, scored_categories, dice_set) for dice_set in all_dice_sets]


def game_over_check(state, upper_score, scored_categories, game_over):
    if game_over:
        new_upper_score, expected_score, new_scored_categories = \
            get_max_exit_score(state, upper_score, scored_categories, dice_set=None, game_over=game_over)
        return state, new_upper_score, expected_score, new_scored_categories


def get_possible_reroll_outcomes(all_dice_sets, dice_set, reroll_mask):
    possible_outcomes = []
    for check_dice_set in all_dice_sets:
        for i in range(0, 6):  # we will check every dice position
            if reroll_mask[0] == 0:  # if we are not rerolling this dice
                if dice_set[0] == check_dice_set[0]:  # the it needs to be the same in this position to be possible
                    possible_outcomes.append(check_dice_set)
    return possible_outcomes


def get_reroll_prob_from_map(all_reroll_probabilities, dice_set, reroll_mask, target_dice_set):
    key = str(dice_set) + str(list(reroll_mask)) + str(target_dice_set)
    if key not in all_reroll_probabilities:
        return 0
    else:
        return all_reroll_probabilities[key]


def get_expected_score(state,
                       upper_score,
                       scored_categories,
                       all_dice_sets,
                       all_dice_set_probabilities,
                       expected_state_scores,
                       all_reroll_probabilities,
                       all_reroll_masks,
                       all_state_integers):
    # We start by checking if the game is already over
    # This should only happen for the 63 * 13 cases when all categories are scored.
    # It should return 0 for all cases except when the upper score is at or above the threshold
    # Then it returns the upper bonus

    # We will now proceed to calculate all the internal states to this node

    # We then start by calculating the expected score of each of the possible initial dice sets we can roll
    expected_score = 0
    rerolls_remaining = 2

    for dice_set in all_dice_sets:
        dice_str = "".join(map(str, dice_set))
        reroll_filter = get_reroll_filter(dice_set)
        # The expected score of a state is the sum of the probability of the dice set
        # times the expected score of the state that dice set takes us to
        expected_score += \
            all_dice_set_probabilities[dice_str] * \
            get_expected_score_reroll(state, dice_set, rerolls_remaining, upper_score, scored_categories, reroll_filter,
                                      all_dice_sets, all_reroll_masks, all_reroll_probabilities, all_exit_scores,
                                      expected_state_scores, game_over, all_state_integers)
    return expected_score


def get_expected_score_reroll(state,
                              dice_set,
                              rerolls_remaining,
                              upper_score,
                              scored_categories,
                              reroll_filter,
                              all_dice_sets,
                              all_reroll_masks,
                              all_reroll_probabilities,
                              all_exit_scores,
                              expected_state_scores,
                              game_over,
                              all_state_integers):
    dice_str = "".join(map(str, dice_set))
    # The reroll expected score is the maximum expected score of all possible reroll options

    # If we do not have any rerolls remaining, we stop
    if rerolls_remaining == 0:
        new_upper_score, best_additional_score, new_scored_categories = all_exit_scores[dice_str]
        if new_upper_score >= const.UPPER_BONUS_THRESHOLD:
            new_upper_score = const.UPPER_BONUS_THRESHOLD
        next_state = all_state_integers[str(new_upper_score) + str(new_scored_categories)]
        next_state_potential = expected_state_scores[next_state]
        reroll_state_expected_score = best_additional_score + next_state_potential
        return reroll_state_expected_score

    # Otherwise, we deduct one reroll and calculate the expected score for all reroll options
    max_expected_score = 0
    reroll_masks = all_reroll_masks[reroll_filter]
    for reroll_mask in reroll_masks:
        expected_reroll_state_score = get_expected_score_reroll_mask(state,
                                                                     dice_set,
                                                                     reroll_mask,
                                                                     rerolls_remaining - 1,
                                                                     upper_score,
                                                                     scored_categories,
                                                                     reroll_filter,
                                                                     all_dice_sets,
                                                                     all_reroll_masks,
                                                                     all_reroll_probabilities,
                                                                     all_exit_scores,
                                                                     expected_state_scores,
                                                                     game_over,
                                                                     all_state_integers)
        if expected_reroll_state_score >= max_expected_score:
            max_expected_score = expected_reroll_state_score

    return max_expected_score


def get_expected_score_reroll_mask(state,
                                   dice_set,
                                   reroll_mask,
                                   rerolls_remaining,
                                   upper_score,
                                   scored_categories,
                                   reroll_filter,
                                   all_dice_sets,
                                   all_reroll_masks,
                                   all_reroll_probabilities,
                                   all_exit_scores,
                                   expected_state_scores,
                                   game_over,
                                   all_state_integers):
    # The expected score of a reroll mask is the sum of the probability of getting a dice set from the reroll mask
    # times the expected score of the state that dice set takes us to
    expected_score = 0
    remaining_remaining_decreased = rerolls_remaining - 1

    for dice_set_target in all_dice_sets:
        # This is equivalent to P(r' -> r)
        reroll_probability_key = str(dice_set) + str(list(reroll_mask)) + str(dice_set_target)
        reroll_mask_target_prob = all_reroll_probabilities[reroll_probability_key]

        # This is E(S, r, n-1)
        expected_score_target = get_expected_score_reroll(state,
                                                          dice_set,
                                                          remaining_remaining_decreased,
                                                          upper_score,
                                                          scored_categories,
                                                          reroll_filter,
                                                          all_dice_sets,
                                                          all_reroll_masks,
                                                          all_reroll_probabilities,
                                                          all_exit_scores,
                                                          expected_state_scores,
                                                          game_over,
                                                          all_state_integers)

        # This is Sum(P(r' -> r) * E(S, r, n-1))
        expected_score += (reroll_mask_target_prob * expected_score_target)

    return expected_score


def generate_all_possible_reroll_masks(all_dice_sets, all_reroll_masks_by_filter):
    all_possible_reroll_masks = []
    for dice in all_dice_sets:
        reroll_filter = get_reroll_filter(dice)
        masks = all_reroll_masks_by_filter[reroll_filter]
        all_possible_reroll_masks.extend(masks)
    return all_possible_reroll_masks


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
        upper_score_binary = bin(upper_score)[2:].zfill(6)

        # There are 2^13 = 8192 possible states for the scored categories
        for scored_categories in range(8192):
            scored_categories_binary = bin(scored_categories)[2:].zfill(13)

            # We can now store each start-state as a 19-bit integer for later decoding.
            # Note that the fact that Yatzy is tri-state means not all integers will be used.
            start_state_integer = int(upper_score_binary + scored_categories_binary, 2)
            key = str(str(upper_score) + str(scored_categories_binary))
            state_integers[key] = start_state_integer
    return state_integers


def generate_state_integer(upper_score, scored_categories):
    if upper_score >= 63:
        upper_score = 63
    return encode_state_integer(upper_score, scored_categories)


def generate_all_dice_set_probabilities(all_dice_sets):
    all_dice_set_probabilities = {}
    for dice_set in all_dice_sets:
        dice_str = "".join(map(str, dice_set))
        probability_of_dice_set = get_probability_of_dice_set(dice_set)
        all_dice_set_probabilities[dice_str] = probability_of_dice_set
    return all_dice_set_probabilities


def get_probability_of_dice_set(dice_set):
    counts = count_dice_faces(dice_set)
    probability = probability_of_dice_set_by_counts(counts)
    return probability


def generate_all_dice_sets():
    dices = [1, 1, 1, 1, 1]
    dice_faces = [1, 2, 3, 4, 5, 6]
    num_dices_reroll = 5
    outcome = [dices[:-num_dices_reroll] + list(comb) for comb in
               combinations_with_replacement(dice_faces, num_dices_reroll)]
    return outcome


def generate_dice_states(dices, num_dices_reroll):
    dice_faces = [1, 2, 3, 4, 5, 6]
    if num_dices_reroll == 0:
        return [dices]
    outcome = [dices[:-num_dices_reroll] + list(comb) for comb in
               combinations_with_replacement(dice_faces, num_dices_reroll)]
    return outcome


def get_all_exit_scores(state, upper_score, scored_categories, all_dice_sets, game_over, expected_state_scores):
    exit_scores = {}
    for dice_set in all_dice_sets:
        dice_str = "".join(map(str, dice_set))
        new_upper_score, best_additional_score, new_scored_categories = get_max_exit_score(state,
                                                                                           upper_score,
                                                                                           scored_categories,
                                                                                           dice_set,
                                                                                           game_over)
        next_state_integer = encode_state_integer(new_upper_score, new_scored_categories)
        next_state_score = expected_state_scores[next_state_integer]
        exit_scores[dice_str] = best_additional_score + next_state_score
    return exit_scores


def encode_state_integer(new_upper, new_cat):
    if new_upper > const.UPPER_BONUS_THRESHOLD:
        new_upper = const.UPPER_BONUS_THRESHOLD
    upper_score_binary = bin(new_upper)[2:].zfill(6)
    score_category_int = int(new_cat, 2)
    scored_categories_bin = bin(score_category_int)[2:].zfill(13)
    return int(upper_score_binary + scored_categories_bin, 2)


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
        if upper_score >= const.UPPER_BONUS_THRESHOLD:
            new_upper_score = upper_score
            best_additional_score = const.UPPER_BONUS
            return new_upper_score, best_additional_score, scored_categories
        else:
            new_upper_score = upper_score
            best_additional_score = 0
            return new_upper_score, best_additional_score, scored_categories

    # If the game is not over, we iterate over all possible categories and check if it is still open for scoring
    # If it is, we check if the available dices are valid for that category, and what the score would be.
    # If it is not valid, we can still score 0, so we return zero for all invalid categories.
    best_additional_score = 0
    new_upper_score = upper_score
    best_category_name = None
    not_scored_yet = '0'

    keys = list(const.combinations.keys())

    for ix, category in enumerate(scored_categories):
        category_name = keys[ix]
        if category == not_scored_yet:
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

    new_scored_categories = update_scored_categories(scored_categories, best_category_name)

    if is_upper_section(best_category_name):
        new_upper_score += best_additional_score

    return new_upper_score, best_additional_score, new_scored_categories


def is_upper_section(category_name):
    return const.combinations[category_name]["upper_section"]


def update_scored_categories(scored_categories, scored_category):
    index_of_max_score_category = list(const.combinations.keys()).index(scored_category)
    return scored_categories[:index_of_max_score_category] + \
        "1" + scored_categories[index_of_max_score_category + 1:]


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


def filter_states_by_unfinished(states, finished):
    unfinished_states = []
    for state in states:
        if state not in finished:
            unfinished_states.append(state)
    return unfinished_states


def filter_states_by_depth(states, remaining_to_score):
    filtered_states = []
    for start_state in states:
        upper_score, scored_categories = decode_state_integer(start_state)
        if is_remaining_to_score(scored_categories, remaining_to_score):
            filtered_states.append(start_state)
    return filtered_states


def is_remaining_to_score(s, num_categories):
    return num_categories == len(s) - s.count('1')


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
    return edges


def generate_reroll_edges_from_dice_set(dice_set):
    dice_str = "".join(map(str, dice_set))

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
    masks = list(product([0, 1], repeat=num_dices))

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


def generate_start_states():
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


def generate_all_reroll_masks(num_dices):
    return list(product([0, 1], repeat=num_dices))


def compute_all_reroll_probabilities(dice_sets, reroll_masks):
    filename = '/Users/langkilde/IdeaProjects/yatzy/data/reroll_probabilities.pkl.gz'

    # Check if the file already exists
    if os.path.isfile(filename):
        print("Loading probabilities from file: " + filename)
        # Load the probabilities from the existing file
        with gzip.open(filename, 'rb') as f:
            probabilities = pickle.load(f)
    else:
        print("Computing probabilities and saving to file: " + filename)
        # Initialize a map to store probabilities
        probabilities = {}

        # Loop over all dice sets, reroll masks, and target outcomes
        for dice_set in dice_sets:
            for reroll_mask in reroll_masks:
                for target_outcome in dice_sets:
                    # Compute the probability
                    probability = get_reroll_probability(dice_set, reroll_mask, target_outcome)
                    rounded_probability = round(probability, 6)
                    # Store the probability in the map
                    if probability > 0:
                        probabilities[str(dice_set) + str(list(reroll_mask)) + str(target_outcome)] = probability

        # Save probabilities to a compressed pickle file
        with gzip.open(filename, 'wb') as f:
            pickle.dump(probabilities, f)

    return probabilities


if __name__ == "__main__":
    build_graph()
