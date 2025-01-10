#include <dice_mechanics.h>
#include <computations.h>
#include <math.h>

#include "context.h"

void PrecomputeCategoryScores(YatzyContext *ctx) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < 252; i++) {
        int *dice = ctx->all_dice_sets[i];
        for (int cat = 0; cat < CATEGORY_COUNT; cat++) {
            ctx->precomputed_scores[i][cat] = CalculateCategoryScore(dice, cat);
        }
    }
}

void PrecomputeFactorials(YatzyContext *ctx) {
    ctx->factorial[0] = 1;
    for (int i = 1; i <= 5; i++) ctx->factorial[i] = ctx->factorial[i - 1] * i;
}

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

void PrecomputeRerollTransitionProbabilities(YatzyContext *ctx) {
#pragma omp parallel for schedule(dynamic)
    for (int ds1_i = 0; ds1_i < 252; ds1_i++) {
        int *ds1 = ctx->all_dice_sets[ds1_i];
        for (int reroll_mask = 0; reroll_mask < 32; reroll_mask++) {
            int locked_indices[5];
            int count_locked = 0, count_reroll = 0;
            for (int i = 0; i < 5; i++) {
                if (reroll_mask & (1 << i)) count_reroll++;
                else locked_indices[count_locked++] = i;
            }

            if (count_reroll == 0) {
                for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
                    ctx->transition_table[ds1_i][reroll_mask][ds2_i] = (ds1_i == ds2_i) ? 1.0 : 0.0;
                }
                continue;
            }

            int total_outcomes = 1;
            for (int p = 0; p < count_reroll; p++) total_outcomes *= 6;
            double counts[252];
            for (int k = 0; k < 252; k++) counts[k] = 0.0;

            for (int outcome_id = 0; outcome_id < total_outcomes; outcome_id++) {
                int temp = outcome_id;
                int rerolled_values[5];
                for (int p = 0; p < count_reroll; p++) {
                    rerolled_values[p] = (temp % 6) + 1;
                    temp /= 6;
                }

                int final_dice[5];
                for (int idx = 0; idx < count_locked; idx++) final_dice[idx] = ds1[locked_indices[idx]];
                for (int idx = 0; idx < count_reroll; idx++) final_dice[count_locked + idx] = rerolled_values[idx];

                SortDiceSet(final_dice);
                int ds2_index = ctx->index_lookup[final_dice[0] - 1][final_dice[1] - 1][final_dice[2] - 1][
                    final_dice[3] - 1][final_dice[4] - 1];
                counts[ds2_index] += 1.0;
            }

            double inv_total = 1.0 / total_outcomes;
            for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
                ctx->transition_table[ds1_i][reroll_mask][ds2_i] = counts[ds2_i] * inv_total;
            }
        }
    }
}

void PrecomputeDiceSetProbabilities(YatzyContext *ctx) {
#pragma omp parallel for schedule(static)
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        ctx->dice_set_probabilities[ds_i] = ComputeProbabilityOfDiceSetOnce(ctx, ctx->all_dice_sets[ds_i]);
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


void ComputeExpectedValuesForNRerolls(
    const YatzyContext *ctx,
    int n,
    double E_ds_prev[252],
    double E_ds_current[252],
    int best_mask_for_n[252]
) {
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
    int scr = ctx->precomputed_scores[ds_index][chosen_category];
    if (scr == 0) return -INFINITY;
    int new_up = UpdateUpperScore(upper_score, chosen_category, scr);
    int new_scored = scored_categories | (1 << chosen_category);
    double future_val = GetStateValue(ctx, new_up, new_scored);
    return scr + future_val;
}
