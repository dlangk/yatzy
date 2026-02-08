/*
 * phase0_tables.c — Phase 0: Precompute lookup tables
 *
 * Builds all static lookup tables before the DP pass:
 *   - s(S, r, c) scores for all (dice_set, category) pairs
 *   - Factorials for multinomial coefficient calculations
 *   - R_{5,6} enumeration (252 sorted 5-dice multisets)
 *   - P(r' → r) reroll transition probabilities
 *   - P(⊥ → r) initial roll probabilities
 *   - Popcount cache for scored-category bitmasks
 *   - Terminal state values (Phase 2 base case)
 *
 * See: theory/optimal_yahtzee_pseudocode.md, Phase 0.
 */
#include <stdio.h>
#include <stdlib.h>

#include "phase0_tables.h"
#include "dice_mechanics.h"
#include "game_mechanics.h"

/* Precompute s(S, r, c) for all r ∈ R_{5,6} and all 15 categories.
 * Stored in ctx->precomputed_scores[dice_set_index][category]. */
void PrecomputeCategoryScores(YatzyContext *ctx) {
    for (int i = 0; i < 252; i++) {
        int *dice = ctx->all_dice_sets[i];
        for (int cat = 0; cat < CATEGORY_COUNT; cat++) {
            ctx->precomputed_scores[i][cat] = CalculateCategoryScore(dice, cat);
        }
    }
}

/* Precompute factorials 0!..5! for multinomial coefficient calculations.
 * Used by ComputeProbabilityOfDiceSet to compute P(⊥ → r). */
void PrecomputeFactorials(YatzyContext *ctx) {
    ctx->factorial[0] = 1;
    for (int i = 1; i <= 5; i++) ctx->factorial[i] = ctx->factorial[i - 1] * i;
}

/* Enumerate all 252 sorted 5-dice multisets R_{5,6} and build reverse lookup.
 * Populates ctx->all_dice_sets[252][5] and ctx->index_lookup[6][6][6][6][6].
 *
 * See pseudocode Phase 0: "Enumerate all 252 distinct 5-dice multisets -> R_5,6" */
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

/* Helper: extract which dice are locked vs rerolled for a given mask. */
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

/* Helper: generate dice values for one reroll outcome (base-6 decode). */
static void GenerateRerolledValues(int outcome_id, int count_reroll, int *rerolled_values) {
    for (int p = 0; p < count_reroll; p++) {
        rerolled_values[p] = (outcome_id % 6) + 1;
        outcome_id /= 6;
    }
}

/* Helper: merge locked and rerolled dice into a single dice set. */
static void CombineDiceSets(const int *ds1, const int *locked_indices, int count_locked, const int *rerolled_values,
                            int count_reroll, int *final_dice) {
    for (int idx = 0; idx < count_locked; idx++) {
        final_dice[idx] = ds1[locked_indices[idx]];
    }
    for (int idx = 0; idx < count_reroll; idx++) {
        final_dice[count_locked + idx] = rerolled_values[idx];
    }
}

/* Precompute P(r' → r) transition probabilities and build sparse CSR table.
 *
 * Builds transitions into a temporary dense buffer, then compresses to CSR
 * format in ctx->sparse_transitions. The dense buffer is freed after.
 *
 * See pseudocode Phase 0: "For each keep r' in R_k and each full roll r in R_5,6:
 *   Compute P(r' -> r)" */
void PrecomputeRerollTransitionProbabilities(YatzyContext *ctx) {
    /* Allocate temporary dense buffer (252 × 32 × 252 doubles = 16.3 MB) */
    double (*dense)[32][252] = malloc(252 * 32 * 252 * sizeof(double));
    if (!dense) {
        fprintf(stderr, "Failed to allocate temp transition buffer\n");
        return;
    }

    /* Pass 1: compute dense transition probabilities */
    for (int ds1_i = 0; ds1_i < 252; ds1_i++) {
        const int *ds1 = ctx->all_dice_sets[ds1_i];

        for (int reroll_mask = 0; reroll_mask < 32; reroll_mask++) {
            int locked_indices[5];
            int count_locked, count_reroll;

            ExtractLockedAndRerollIndices(reroll_mask, locked_indices, &count_locked, &count_reroll);

            /* mask=0: keep all dice, deterministic transition to self */
            if (count_reroll == 0) {
                for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
                    dense[ds1_i][reroll_mask][ds2_i] = (ds1_i == ds2_i) ? 1.0 : 0.0;
                }
                continue;
            }

            int total_outcomes = 1;
            for (int p = 0; p < count_reroll; p++) {
                total_outcomes *= 6;
            }

            double counts[252] = {0.0};

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

            const double inv_total = 1.0 / total_outcomes;
            for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
                dense[ds1_i][reroll_mask][ds2_i] = counts[ds2_i] * inv_total;
            }
        }
    }

    /* Pass 2: count non-zeros and build CSR row_offsets */
    SparseTransitionTable *sp = &ctx->sparse_transitions;
    int total_nz = 0;
    for (int ds = 0; ds < 252; ds++) {
        for (int mask = 0; mask < 32; mask++) {
            sp->row_offsets[ds * 32 + mask] = total_nz;
            for (int ds2 = 0; ds2 < 252; ds2++) {
                if (dense[ds][mask][ds2] > 0.0) total_nz++;
            }
        }
    }
    sp->row_offsets[252 * 32] = total_nz;
    sp->total_nonzero = total_nz;

    /* Allocate CSR arrays */
    sp->values = malloc(total_nz * sizeof(double));
    sp->col_indices = malloc(total_nz * sizeof(int));
    if (!sp->values || !sp->col_indices) {
        fprintf(stderr, "Failed to allocate sparse transition table (%d entries)\n", total_nz);
        free(dense);
        return;
    }

    /* Pass 3: fill CSR arrays */
    int k = 0;
    for (int ds = 0; ds < 252; ds++) {
        for (int mask = 0; mask < 32; mask++) {
            for (int ds2 = 0; ds2 < 252; ds2++) {
                if (dense[ds][mask][ds2] > 0.0) {
                    sp->values[k] = dense[ds][mask][ds2];
                    sp->col_indices[k] = ds2;
                    k++;
                }
            }
        }
    }

    free(dense);

    printf("    Sparse transition table: %d non-zero / %d total (%.1f%% sparse), %.1f MB\n",
           total_nz, 252 * 32 * 252,
           100.0 * (1.0 - (double)total_nz / (252 * 32 * 252)),
           (total_nz * (sizeof(double) + sizeof(int))) / 1e6);
}

/* Popcount cache: maps scored-category bitmask → |C| (number of categories scored).
 * Avoids repeated __builtin_popcount calls during the DP. */
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

/* Precompute P(⊥ → r) for all r ∈ R_{5,6}.
 * Stored in ctx->dice_set_probabilities[252]. */
void PrecomputeDiceSetProbabilities(YatzyContext *ctx) {
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        ctx->dice_set_probabilities[ds_i] = ComputeProbabilityOfDiceSet(ctx, ctx->all_dice_sets[ds_i]);
    }
}

/* Phase 2 base case (terminal states, |C| = 15).
 * E(S) = 50 if m >= 63 (upper bonus), else 0.
 *
 * Note: Scandinavian Yatzy awards a 50-point upper bonus (not 35 as in
 * standard Yahtzee). See pseudocode Phase 2, Step 1. */
void InitializeFinalStates(YatzyContext *ctx) {
    int all_scored_mask = (1 << CATEGORY_COUNT) - 1;
    for (int up = 0; up <= 63; up++) {
        double final_val = (up >= 63) ? 50.0 : 0.0;
        ctx->state_values[STATE_INDEX(up, all_scored_mask)] = final_val;
    }
}
