#include <yatzy.h>
#include <file_utilities.h>
#include <webserver.h>
#include <stdio.h>
#include <unistd.h>
#include <limits.h>

void PrecomputeCategoryScores(YatzyContext *ctx) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < 252; i++) {
        int *dice = ctx->all_dice_sets[i];
        for (int cat = 0; cat < CATEGORY_COUNT; cat++) {
            ctx->precomputed_scores[i][cat] = CalculateCategoryScore(dice, cat);
        }
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

void PrecomputeFactorials(YatzyContext *ctx) {
    ctx->factorial[0] = 1;
    for (int i = 1; i <= 5; i++) ctx->factorial[i] = ctx->factorial[i - 1] * i;
}

double ComputeProbabilityOfDiceSetOnce(const YatzyContext *ctx,
                                              const int dice[5]) {
    int face_count[7];
    for (int i = 1; i <= 6; i++) face_count[i] = 0;
    for (int i = 0; i < 5; i++) face_count[dice[i]]++;

    int numerator = ctx->factorial[5];
    int denominator = 1;
    for (int f = 1; f <= 6; f++) {
        if (face_count[f] > 1) {
            int fc = face_count[f];
            int sub_fact = 1;
            for (int x = 2; x <= fc; x++) sub_fact *= x;
            denominator *= sub_fact;
        }
    }

    double permutations = (double) numerator / (double) denominator;
    double total_outcomes = pow(6, 5);
    return permutations / total_outcomes;
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

double GetStateValue(const YatzyContext *ctx,
                            int up,
                            int scored) {
    return ctx->state_values[STATE_INDEX(up, scored)];
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

typedef struct {
    double ev; // The expected value if the resulting dice set is ds2_i
    double probability; // Probability that ds2_i occurs given ds_index + reroll mask
    int ds2_index; // (Optional) the index of ds2_i, in case you need it
} EVProbabilityPair;

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

void check_file_existence(int scored_count) {
    char filepath[128];
    snprintf(filepath, sizeof(filepath), "data/states_%d.bin", scored_count);

    FILE *file = fopen(filepath, "r");
    if (file) {
        printf("File found: %s\n", filepath);
        fclose(file);
    } else {
        printf("File not found: %s\n", filepath);
    }
}

char cwd[PATH_MAX];

void ComputeAllStateValues(YatzyContext *ctx) {
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("Current working directory: %s\n", cwd);
    } else {
        perror("getcwd failed");
    }
    for (int scored_count = 15; scored_count >= 0; scored_count--) {
        if (scored_count == 15) {
            char filepath[128]; // Adjust size based on your file path requirements

            snprintf(filepath, sizeof(filepath), "data/states_%d.bin", scored_count);

            if (LoadStateValuesForCount(ctx, scored_count, filepath)) {
                printf("Successfully loaded state values from: %s\n", filepath);
            } else {
                printf("Failed to load state values or file not found: %s\n", filepath);
                SaveStateValuesForCount(ctx, scored_count, filepath); // Compute and save if loading fails
            }

            continue;
        }
        char filename[64];
        snprintf(filename, sizeof(filename), "data/states_%d.bin", scored_count);

        if (LoadStateValuesForCount(ctx, scored_count, filename)) {
            printf("Successfully loaded state values from: %s\n", filename);
            continue;
        }
        printf("Computing state values for %d scored categories...\n", scored_count);

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