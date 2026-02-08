/*
 * api_computations.c — API-facing computation wrappers
 *
 * Composes Phase 2a SOLVE_WIDGET primitives for specific user situations.
 * These functions are called by the webserver HTTP handlers.
 */
#include <math.h>
#include <stdio.h>

#include "api_computations.h"
#include "widget_solver.h"
#include "dice_mechanics.h"
#include "game_mechanics.h"

/* Compute best reroll mask for a specific game situation.
 * Builds the reroll-level arrays then picks the best mask for the given dice. */
void ComputeBestRerollStrategy(const YatzyContext *ctx,
                               int upper_score,
                               int scored_categories,
                               const int dice[5],
                               int rerolls_remaining,
                               int *best_mask,
                               double *best_ev) {
    /* Group 6: E(S, r, 0) for all r */
    double E_ds_0[252];
    YatzyState state = {upper_score, scored_categories};
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        E_ds_0[ds_i] = ComputeBestScoringValueForDiceSet(ctx, &state, ctx->all_dice_sets[ds_i]);
    }

    if (rerolls_remaining == 1) {
        *best_mask = ChooseBestRerollMask(ctx, E_ds_0, dice, best_ev);
        return;
    }

    double E_ds_1[252];
    int dummy_mask[252];
    ComputeExpectedValuesForNRerolls(ctx, E_ds_0, E_ds_1, dummy_mask);
    *best_mask = ChooseBestRerollMask(ctx, E_ds_1, dice, best_ev);
}

/* Choose best category when no rerolls remain (non-zero score only). */
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

        if (scr == 0) {
            continue;
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

/* Evaluate the EV of a user-chosen reroll mask. */
double EvaluateChosenRerollMask(const YatzyContext *ctx,
                                int upper_score,
                                int scored_categories,
                                const int dice[5],
                                int chosen_mask,
                                int rerolls_remaining) {
    /* Group 6: E(S, r, 0) for all r */
    double E_ds_0[252];
    YatzyState state = {upper_score, scored_categories};
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        E_ds_0[ds_i] = ComputeBestScoringValueForDiceSet(ctx, &state, ctx->all_dice_sets[ds_i]);
    }

    double E_ds_1[252];
    int dummy_mask[252];
    if (rerolls_remaining == 2) {
        ComputeExpectedValuesForNRerolls(ctx, E_ds_0, E_ds_1, dummy_mask);
    }

    const double *E_ds_for_masks = (rerolls_remaining == 1) ? E_ds_0 : E_ds_1;

    int sorted_dice[5];
    for (int i = 0; i < 5; i++) sorted_dice[i] = dice[i];
    SortDiceSet(sorted_dice);
    int ds_index = FindDiceSetIndex(ctx, sorted_dice);

    double ev = ComputeExpectedValueForRerollMask(ctx, ds_index, E_ds_for_masks, chosen_mask);
    return ev;
}

/* Evaluate the EV of a user-chosen category assignment. */
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

    if (score == 0) return -INFINITY;
    int new_up = UpdateUpperScore(upper_score, chosen_category, score);
    int new_scored = scored_categories | (1 << chosen_category);
    double future_val = GetStateValue(ctx, new_up, new_scored);
    return score + future_val;
}

/* Build outcome distribution for a specific reroll mask.
 * Pairs each possible outcome ds2 with its EV and transition probability. */
void ComputeDistributionForRerollMask(const YatzyContext *ctx,
                                      int ds_index,
                                      const double E_ds_for_masks[252],
                                      int mask,
                                      EVProbabilityPair out_distribution[252]) {
    /* Initialize all entries with zero probability */
    for (int ds2_i = 0; ds2_i < 252; ds2_i++) {
        out_distribution[ds2_i].ev = E_ds_for_masks[ds2_i];
        out_distribution[ds2_i].probability = 0.0;
        out_distribution[ds2_i].ds2_index = ds2_i;
    }

    /* Fill non-zero probabilities from sparse table */
    const SparseTransitionTable *sp = &ctx->sparse_transitions;
    int row = ds_index * 32 + mask;
    int start = sp->row_offsets[row];
    int end = sp->row_offsets[row + 1];
    for (int k = start; k < end; k++) {
        out_distribution[sp->col_indices[k]].probability = sp->values[k];
    }
}

/* Compute E(S, r, n) for all r, for a given number of rerolls.
 * Copies result into caller-provided buffer (thread-safe). */
void ComputeExpectedValues(const YatzyContext *ctx, int upper_score,
                           int scored_categories, int rerolls,
                           double out_E_ds[252]) {
    double E_ds[3][252];

    /* Level 0: E(S, r, 0) = max_c [s(S,r,c) + E(n(S,r,c))] */
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

    /* Levels 1..rerolls: E(S, r, n) = max_{mask} Σ P(r'→r'') · E_ds[n-1][r''] */
    const double * restrict vals = ctx->sparse_transitions.values;
    const int * restrict cols = ctx->sparse_transitions.col_indices;
    const int *offsets = ctx->sparse_transitions.row_offsets;

    for (int n = 1; n <= rerolls; n++) {
        #pragma omp parallel for
        for (int ds_i = 0; ds_i < 252; ds_i++) {
            double best_val = E_ds[n - 1][ds_i]; /* mask=0: keep all */
            int row_base = ds_i * 32;
            for (int mask = 1; mask < 32; mask++) {
                int start = offsets[row_base + mask];
                int end = offsets[row_base + mask + 1];
                double ev = 0.0;
                for (int k = start; k < end; k++) {
                    ev += vals[k] * E_ds[n - 1][cols[k]];
                }
                if (ev > best_val) best_val = ev;
            }
            E_ds[n][ds_i] = best_val;
        }
    }

    /* Copy result to caller's buffer */
    for (int i = 0; i < 252; i++) {
        out_E_ds[i] = E_ds[rerolls][i];
    }
}

/* Weighted sum of EV over a probability distribution. */
double ComputeEVFromDistribution(const EVProbabilityPair distribution[], int size) {
    double ev = 0.0;
    for (int i = 0; i < size; i++) {
        ev += distribution[i].ev * distribution[i].probability;
    }
    return ev;
}
