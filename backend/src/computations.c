#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <dice_mechanics.h>
#include <computations.h>
#include "context.h"
#include "storage.h"

/**
 * @brief Precomputes the scores for all dice combinations across all categories.
 *
 * This function iterates over all unique sorted dice combinations (252 in total)
 * and computes the score for each category, storing the results in the
 * `precomputed_scores` array in the YatzyContext. The scores are computed using
 * the `CalculateCategoryScore` function.
 *
 * Parallelization with OpenMP is used to distribute the computation across
 * multiple threads, improving performance for large calculations.
 *
 * The precomputed scores allow for fast lookup during gameplay or analysis,
 * avoiding redundant score calculations.
 *
 * @param ctx Pointer to the YatzyContext where the precomputed scores will be stored.
 */
void PrecomputeCategoryScores(YatzyContext *ctx) {
    for (int i = 0; i < 252; i++) {
        int *dice = ctx->all_dice_sets[i];
        for (int cat = 0; cat < CATEGORY_COUNT; cat++) {
            ctx->precomputed_scores[i][cat] = CalculateCategoryScore(dice, cat);
        }
    }
}

/**
 * @brief Precomputes factorial values from 0! to 5! and stores them in the YatzyContext.
 *
 * This function calculates factorials iteratively and stores the results in the
 * `factorial` array of the provided YatzyContext. The precomputed values can be
 * used for various probability calculations during the game, avoiding redundant
 * computations.
 *
 * @param ctx Pointer to the YatzyContext where the factorial values will be stored.
 */
void PrecomputeFactorials(YatzyContext *ctx) {
    ctx->factorial[0] = 1;
    for (int i = 1; i <= 5; i++) ctx->factorial[i] = ctx->factorial[i - 1] * i;
}

/**
 * @brief Generates all unique combinations of sorted dice rolls and stores them in the YatzyContext.
 *
 * This function systematically builds all possible combinations of five dice rolls, where the values
 * are sorted in non-decreasing order. It populates the `all_dice_sets` array in the YatzyContext with
 * each unique combination and maintains a mapping from dice values to their corresponding index in
 * the `index_lookup` array for quick access during calculations.
 *
 * The total number of unique combinations is stored in `num_combinations`, which will always be 252
 * (the number of unique sorted combinations of 5 dice with 6 faces each).
 *
 * @param ctx Pointer to the YatzyContext where the dice combinations and lookup table will be stored.
 */
void BuildAllDiceCombinations(YatzyContext *ctx) {
    ctx->num_combinations = 0;
    for (int a = 1; a <= 6; a++) {
        for (int b = a; b <= 6; b++) {
            for (int c = b; c <= 6; c++) {
                for (int d = c; d <= 6; d++) {
                    for (int e = d; e <= 6; e++) {
                        int idx = ctx->num_combinations++;
                        ctx->all_dice_sets[idx][0] = a;
                        ctx->all_dice_sets[idx][1] = b;
                        ctx->all_dice_sets[idx][2] = c;
                        ctx->all_dice_sets[idx][3] = d;
                        ctx->all_dice_sets[idx][4] = e;
                        ctx->index_lookup[a - 1][b - 1][c - 1][d - 1][e - 1] = idx;
                    }
                }
            }
        }
    }
}

/**
 * @brief Extract locked and reroll indices based on the reroll mask.
 */
static void ExtractLockedAndRerollIndices(int reroll_mask, int *locked_indices, int *count_locked, int *count_reroll) {
    *count_locked = 0;
    *count_reroll = 0;
    for (int i = 0; i < 5; i++) {
        if (reroll_mask & (1 << i)) {
            (*count_reroll)++;
        } else {
            locked_indices[(*count_locked)++] = i;
        }
    }
}

/**
 * @brief Generate all possible rerolled dice values.
 */
static void GenerateRerolledValues(int outcome_id, int count_reroll, int *rerolled_values) {
    for (int p = 0; p < count_reroll; p++) {
        rerolled_values[p] = (outcome_id % 6) + 1;
        outcome_id /= 6;
    }
}

/**
 * @brief Combine locked and rerolled dice into a final dice set.
 */
static void CombineDiceSets(const int *ds1, const int *locked_indices, int count_locked, const int *rerolled_values,
                            int count_reroll, int *final_dice) {
    for (int idx = 0; idx < count_locked; idx++) {
        final_dice[idx] = ds1[locked_indices[idx]];
    }
    for (int idx = 0; idx < count_reroll; idx++) {
        final_dice[count_locked + idx] = rerolled_values[idx];
    }
}

/**
 * @brief Precomputes transition probabilities for rerolling dice in Yatzy.
 *
 * This function calculates the probability of transitioning from one sorted dice set (`ds1`)
 * to another (`ds2`) for all possible reroll masks. It populates the `transition_table`
 * in the provided YatzyContext, which is used for determining expected values and
 * optimal strategies during gameplay.
 *
 * The process involves:
 * 1. Iterating through all unique sorted dice combinations (`ds1`).
 * 2. For each reroll mask (32 possible masks), identifying locked and rerolled dice indices.
 * 3. Simulating all possible outcomes for rerolled dice and determining the resulting sorted dice set (`ds2`).
 * 4. Counting occurrences of each resulting `ds2` to compute probabilities, normalized by the total number of outcomes.
 * 5. Storing the normalized probabilities in the `transition_table`.
 *
 * OpenMP parallelization is used to accelerate the computation by processing multiple dice sets in parallel.
 *
 * @param ctx Pointer to the YatzyContext where the precomputed probabilities will be stored.
 */
void PrecomputeRerollTransitionProbabilities(YatzyContext *ctx) {
    for (int ds1_i = 0; ds1_i < 252; ds1_i++) {
        const int *ds1 = ctx->all_dice_sets[ds1_i];

        for (int reroll_mask = 0; reroll_mask < 32; reroll_mask++) {
            int locked_indices[5];
            int count_locked, count_reroll;

            // Extract locked and reroll indices
            ExtractLockedAndRerollIndices(reroll_mask, locked_indices, &count_locked, &count_reroll);

            // If no reroll is performed, transition is deterministic
            if (count_reroll == 0) {
                for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
                    ctx->transition_table[ds1_i][reroll_mask][ds2_i] = (ds1_i == ds2_i) ? 1.0 : 0.0;
                }
                continue;
            }

            // Compute the total number of outcomes for rerolling
            int total_outcomes = 1;
            for (int p = 0; p < count_reroll; p++) {
                total_outcomes *= 6;
            }

            // Initialize counts for transition probabilities
            double counts[252] = {0.0};

            // Simulate all possible reroll outcomes
            for (int outcome_id = 0; outcome_id < total_outcomes; outcome_id++) {
                int final_dice[5];
                int rerolled_values[5];
                GenerateRerolledValues(outcome_id, count_reroll, rerolled_values);
                CombineDiceSets(ds1, locked_indices, count_locked, rerolled_values, count_reroll, final_dice);

                SortDiceSet(final_dice);
                const int ds2_index = ctx->index_lookup[final_dice[0] - 1][final_dice[1] - 1][final_dice[2] - 1][
                    final_dice[3] - 1][final_dice[4] - 1];
                counts[ds2_index] += 1.0;
            }

            // Normalize counts to probabilities
            const double inv_total = 1.0 / total_outcomes;
            for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
                ctx->transition_table[ds1_i][reroll_mask][ds2_i] = counts[ds2_i] * inv_total;
            }
        }
    }
}

void PrecomputeScoredCategoryCounts(YatzyContext *ctx) {
    for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
        int count = 0;
        int temp = scored;
        for (int i = 0; i < CATEGORY_COUNT; i++) {
            count += (temp & 1);
            temp >>= 1;
        }
        ctx->scored_category_count_cache[scored] = count;
    }
}

void PrecomputeDiceSetProbabilities(YatzyContext *ctx) {
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        ctx->dice_set_probabilities[ds_i] = ComputeProbabilityOfDiceSet(ctx, ctx->all_dice_sets[ds_i]);
    }
}

void InitializeFinalStates(YatzyContext *ctx) {
    int all_scored_mask = (1 << CATEGORY_COUNT) - 1;
    for (int up = 0; up <= 63; up++) {
        double final_val = (up >= 63) ? 50.0 : 0.0;
        ctx->state_values[STATE_INDEX(up, all_scored_mask)] = final_val;
    }
}

void ComputeDistributionForRerollMask(const YatzyContext *ctx,
                                      int ds_index,
                                      const double E_ds_for_masks[252],
                                      int mask,
                                      EVProbabilityPair out_distribution[252]) {
    // Probability array for going from ds_index -> ds2_i under 'mask'
    const double *prob_row = ctx->transition_table[ds_index][mask];

    // Fill out the distribution
    for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
        out_distribution[ds2_i].ev = E_ds_for_masks[ds2_i];
        out_distribution[ds2_i].probability = prob_row[ds2_i];
        out_distribution[ds2_i].ds2_index = ds2_i;
    }
}

double ComputeBestScoringValueForDiceSet(const YatzyContext *ctx,
                                         const YatzyState *S,
                                         const int dice[5]) {
    int ds_index = FindDiceSetIndex(ctx, dice);
    double best_val = -INFINITY;
    int scored = S->scored_categories;
    int up_score = S->upper_score;

    for (int c = 0; c < CATEGORY_COUNT; c++) {
        if (!IS_CATEGORY_SCORED(scored, c)) {
            int scr = ctx->precomputed_scores[ds_index][c];
            int new_up = UpdateUpperScore(up_score, c, scr);
            int new_scored = (scored | (1 << c));
            double val = scr + GetStateValue(ctx, new_up, new_scored);
            if (val > best_val) best_val = val;
        }
    }

    return best_val;
}

void ComputeEDs0ForState(const YatzyContext *ctx,
                         int upper_score,
                         int scored_categories,
                         double E_ds_0[252]) {
    YatzyState state = {upper_score, scored_categories};
    #pragma omp parallel for
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        E_ds_0[ds_i] = ComputeBestScoringValueForDiceSet(ctx, &state, ctx->all_dice_sets[ds_i]);
    }
}

double ComputeExpectedValueForRerollMask(const YatzyContext *ctx,
                                         int ds_index,
                                         const double E_ds_for_masks[252],
                                         int mask) {
    const double *probs = ctx->transition_table[ds_index][mask];
    double ev = 0.0;
    for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
        ev += probs[ds2_i] * E_ds_for_masks[ds2_i];
    }
    return ev;
}

int ChooseBestRerollMask(const YatzyContext *ctx,
                         const double E_ds_for_masks[252],
                         const int dice[5],
                         double *best_ev) {
    int sorted_dice[5] = {dice[0], dice[1], dice[2], dice[3], dice[4]};
    SortDiceSet(sorted_dice);
    int ds_index = FindDiceSetIndex(ctx, sorted_dice);

    double best_val = -INFINITY;
    int best_mask = 0;

    for (int mask = 0; mask < 32; mask++) {
        double ev = ComputeExpectedValueForRerollMask(ctx, ds_index, E_ds_for_masks, mask);
        if (ev > best_val) {
            best_val = ev;
            best_mask = mask;
        }
    }

    *best_ev = best_val;
    return best_mask;
}

void ComputeBestRerollStrategy(const YatzyContext *ctx,
                               int upper_score,
                               int scored_categories,
                               const int dice[5],
                               int rerolls_remaining,
                               int *best_mask,
                               double *best_ev) {
    double E_ds_0[252];
    ComputeEDs0ForState(ctx, upper_score, scored_categories, E_ds_0);

    if (rerolls_remaining == 1) {
        *best_mask = ChooseBestRerollMask(ctx, E_ds_0, dice, best_ev);
        return;
    }

    double E_ds_1[252];
    int dummy_mask[252];
    ComputeExpectedValuesForNRerolls(ctx, 1, E_ds_0, E_ds_1, dummy_mask);
    *best_mask = ChooseBestRerollMask(ctx, E_ds_1, dice, best_ev);
}


void ComputeExpectedValuesForNRerolls(const YatzyContext *ctx,
                                      int n,
                                      double E_ds_prev[252],
                                      double E_ds_current[252],
                                      int best_mask_for_n[252]) {
    #pragma omp parallel for schedule(static)
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        double best_val = -INFINITY;
        int best_mask = 0;
        const double (*row)[252] = ctx->transition_table[ds_i];
        for (int mask = 0; mask < 32; mask++) {
            const double *probs = row[mask];
            double ev = 0.0;
            for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
                ev += probs[ds2_i] * E_ds_prev[ds2_i];
            }
            if (ev > best_val) {
                best_val = ev;
                best_mask = mask;
            }
        }
        E_ds_current[ds_i] = best_val;
        best_mask_for_n[ds_i] = best_mask;
    }
}

double ComputeExpectedStateValue(const YatzyContext *ctx,
                                 const YatzyState *S) {
    double E_ds_0[252], E_ds_1[252], E_ds_2[252];
    int best_mask_1[252], best_mask_2[252];

    #pragma omp parallel for schedule(static)
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        E_ds_0[ds_i] = ComputeBestScoringValueForDiceSet(ctx, S, ctx->all_dice_sets[ds_i]);
    }

    ComputeExpectedValuesForNRerolls(ctx, 1, E_ds_0, E_ds_1, best_mask_1);
    ComputeExpectedValuesForNRerolls(ctx, 2, E_ds_1, E_ds_2, best_mask_2);

    double E_S = 0.0;
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        E_S += ctx->dice_set_probabilities[ds_i] * E_ds_2[ds_i];
    }

    return E_S;
}

int ChooseBestCategoryNoRerolls(const YatzyContext *ctx,
                                int upper_score,
                                int scored_categories,
                                const int dice[5],
                                double *best_ev) {
    int ds_index = FindDiceSetIndex(ctx, dice);
    double best_val = -INFINITY;
    int best_category = -1;

    for (int c = 0; c < CATEGORY_COUNT; c++) {
        if (IS_CATEGORY_SCORED(scored_categories, c)) {
            continue;
        }

        int scr = ctx->precomputed_scores[ds_index][c];

        // If this category yields no points, it's effectively useless now
        // Treat it as -INFINITY so we never pick it.
        if (scr == 0) {
            continue; // or explicitly val = -INFINITY
        }

        int new_up = UpdateUpperScore(upper_score, c, scr);
        int new_scored = scored_categories | (1 << c);
        double val = scr + GetStateValue(ctx, new_up, new_scored);

        if (val > best_val) {
            best_val = val;
            best_category = c;
        }
    }

    *best_ev = best_val;
    return best_category;
}

double EvaluateChosenRerollMask(const YatzyContext *ctx,
                                int upper_score,
                                int scored_categories,
                                const int dice[5],
                                int chosen_mask,
                                int rerolls_remaining) {
    double E_ds_0[252];
    ComputeEDs0ForState(ctx, upper_score, scored_categories, E_ds_0);

    double E_ds_1[252];
    int dummy_mask[252];
    if (rerolls_remaining == 2) {
        ComputeExpectedValuesForNRerolls(ctx, 1, E_ds_0, E_ds_1, dummy_mask);
    }

    const double *E_ds_for_masks = (rerolls_remaining == 1) ? E_ds_0 : E_ds_1;

    int sorted_dice[5];
    for (int i = 0; i < 5; i++) sorted_dice[i] = dice[i];
    SortDiceSet(sorted_dice);
    int ds_index = FindDiceSetIndex(ctx, sorted_dice);

    double ev = ComputeExpectedValueForRerollMask(ctx, ds_index, E_ds_for_masks, chosen_mask);
    return ev;
}

double EvaluateChosenCategory(const YatzyContext *ctx,
                              int upper_score,
                              int scored_categories,
                              const int dice[5],
                              int chosen_category) {
    if (IS_CATEGORY_SCORED(scored_categories, chosen_category)) {
        return -INFINITY;
    }
    int ds_index = FindDiceSetIndex(ctx, dice);
    int score = ctx->precomputed_scores[ds_index][chosen_category];

    // print relevant debugging information on the next row
    printf("Score: %d\n", score);
    printf("Category: %d\n", chosen_category);
    printf("Scored categories: %d\n", scored_categories);
    printf("Upper score: %d\n", upper_score);
    printf("Dice: %d %d %d %d %d\n", dice[0], dice[1], dice[2], dice[3], dice[4]);

    if (score == 0) return -INFINITY;
    int new_up = UpdateUpperScore(upper_score, chosen_category, score);
    int new_scored = scored_categories | (1 << chosen_category);
    double future_val = GetStateValue(ctx, new_up, new_scored);
    return score + future_val;
}

void ComputeAllStateValues(YatzyContext *ctx) {
    for (int scored_count = 15; scored_count >= 0; scored_count--) {
        char filename[64];
        snprintf(filename, sizeof(filename), "data/states_%d.bin", scored_count);

        if (scored_count == 15) {
            if (!LoadStateValuesForCount(ctx, scored_count, filename)) {
                SaveStateValuesForCount(ctx, scored_count, filename);
            }
            continue;
        }

        if (LoadStateValuesForCount(ctx, scored_count, filename)) {
            continue;
        }

        int needed_count = 0;
        for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
            if (ctx->scored_category_count_cache[scored] == scored_count) {
                needed_count += 64;
            }
        }

        int (*states)[2] = malloc(sizeof(int) * 2 * needed_count);
        int idx = 0;
        for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
            if (ctx->scored_category_count_cache[scored] == scored_count) {
                for (int up = 0; up <= 63; up++) {
                    states[idx][0] = up;
                    states[idx][1] = scored;
                    idx++;
                }
            }
        }

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < idx; i++) {
            int up = states[i][0];
            int scored = states[i][1];
            YatzyState S = {up, scored};
            double val = ComputeExpectedStateValue(ctx, &S);
            ctx->state_values[STATE_INDEX(up, scored)] = val;
        }

        free(states);
        SaveStateValuesForCount(ctx, scored_count, filename);
    }
}

const double *ComputeExpectedValues(YatzyContext *ctx, int upper_score, int scored_categories, int rerolls) {
    static double E_ds[3][252];
    YatzyState state = {upper_score, scored_categories};

    // Compute E_ds_0 (no rerolls scenario)
    #pragma omp parallel for
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        double best_val = -INFINITY;
        for (int c = 0; c < CATEGORY_COUNT; c++) {
            if (!IS_CATEGORY_SCORED(scored_categories, c)) {
                int scr = ctx->precomputed_scores[ds_i][c];
                int new_up = UpdateUpperScore(upper_score, c, scr);
                int new_scored = scored_categories | (1 << c);
                double val = scr + GetStateValue(ctx, new_up, new_scored);
                if (val > best_val) best_val = val;
            }
        }
        E_ds[0][ds_i] = best_val;
    }

    // If rerolls > 0, compute E_ds_1 and optionally E_ds_2
    for (int n = 1; n <= rerolls; n++) {
        #pragma omp parallel for
        for (int ds_i = 0; ds_i < 252; ds_i++) {
            double best_val = -INFINITY;
            const double (*row)[252] = ctx->transition_table[ds_i];
            for (int mask = 0; mask < 32; mask++) {
                double ev = 0.0;
                for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
                    ev += row[mask][ds2_i] * E_ds[n - 1][ds2_i];
                }
                if (ev > best_val) best_val = ev;
            }
            E_ds[n][ds_i] = best_val;
        }
    }

    // Return the array corresponding to the number of rerolls
    return E_ds[rerolls];
}

double ComputeEVFromDistribution(const EVProbabilityPair distribution[], int size) {
    double ev = 0.0;
    for (int i = 0; i < size; i++) {
        ev += distribution[i].ev * distribution[i].probability;
    }
    return ev;
}
