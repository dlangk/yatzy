/*
 * widget_solver.c — Phase 2a: SOLVE_WIDGET
 *
 * Computes E(S) for a single turn-start state via backward induction
 * through the widget's 6 groups:
 *   Group 6: score the final roll     → E(S, r, 0)
 *   Group 5: second keep choice       → E(S, r, 1)
 *   Group 3: first keep choice        → E(S, r, 2)
 *   Group 1: expected over initial roll → E(S)
 *
 * See: theory/optimal_yahtzee_pseudocode.md, Phase 2a: SOLVE_WIDGET.
 */
#include <math.h>

#include "widget_solver.h"
#include "dice_mechanics.h"
#include "game_mechanics.h"

/* Group 6, inner loop: best category for one roll r.
 * E(S, r, 0) = max_{c ∉ C} [s(S,r,c) + E(n(S,r,c))]
 *
 * For each unused category c, computes the immediate score plus the
 * precomputed future value of the successor state n(S,r,c).
 *
 * Upper categories (0-5) are handled separately since only they affect
 * upper score. Lower categories (6-14) skip the UpdateUpperScore call. */
double ComputeBestScoringValueForDiceSet(const YatzyContext *ctx,
                                         const YatzyState *S,
                                         const int dice[5]) {
    int ds_index = FindDiceSetIndex(ctx, dice);
    double best_val = -INFINITY;
    int scored = S->scored_categories;
    int up_score = S->upper_score;

    /* Upper categories (0-5): affect upper score */
    for (int c = 0; c < 6; c++) {
        if (!IS_CATEGORY_SCORED(scored, c)) {
            int scr = ctx->precomputed_scores[ds_index][c];
            int new_up = UpdateUpperScore(up_score, c, scr);
            int new_scored = (scored | (1 << c));
            double val = scr + GetStateValue(ctx, new_up, new_scored);
            if (val > best_val) best_val = val;
        }
    }

    /* Lower categories (6-14): upper score unchanged */
    for (int c = 6; c < CATEGORY_COUNT; c++) {
        if (!IS_CATEGORY_SCORED(scored, c)) {
            int scr = ctx->precomputed_scores[ds_index][c];
            int new_scored = (scored | (1 << c));
            double val = scr + GetStateValue(ctx, up_score, new_scored);
            if (val > best_val) best_val = val;
        }
    }

    return best_val;
}

/* Group 6, inner loop variant that takes ds_index directly.
 * Used in the DP hot path where the index is already known. */
static inline double ComputeBestScoringValueForDiceSetByIndex(
        const YatzyContext *ctx,
        int up_score, int scored,
        int ds_index) {
    double best_val = -INFINITY;

    /* Upper categories (0-5): affect upper score */
    for (int c = 0; c < 6; c++) {
        if (!IS_CATEGORY_SCORED(scored, c)) {
            int scr = ctx->precomputed_scores[ds_index][c];
            int new_up = UpdateUpperScore(up_score, c, scr);
            int new_scored = (scored | (1 << c));
            double val = scr + GetStateValue(ctx, new_up, new_scored);
            if (val > best_val) best_val = val;
        }
    }

    /* Lower categories (6-14): upper score unchanged */
    for (int c = 6; c < CATEGORY_COUNT; c++) {
        if (!IS_CATEGORY_SCORED(scored, c)) {
            int scr = ctx->precomputed_scores[ds_index][c];
            int new_scored = (scored | (1 << c));
            double val = scr + GetStateValue(ctx, up_score, new_scored);
            if (val > best_val) best_val = val;
        }
    }

    return best_val;
}

/* Single mask evaluation: Σ_{r''} P(r'→r'') · E_prev[r''].
 * Computes the expected value of keeping the dice indicated by ~mask
 * and rerolling the rest, given E_prev values for the next stage. */
double ComputeExpectedValueForRerollMask(const YatzyContext *ctx,
                                         int ds_index,
                                         const double E_ds_for_masks[252],
                                         int mask) {
    if (mask == 0) return E_ds_for_masks[ds_index]; /* keep all */
    const KeepTable *kt = &ctx->keep_table;
    int kid = kt->mask_to_keep[ds_index * 32 + mask];
    int start = kt->row_start[kid];
    int end = kt->row_start[kid + 1];
    double ev = 0.0;
    for (int k = start; k < end; k++) {
        ev += kt->vals[k] * E_ds_for_masks[kt->cols[k]];
    }
    return ev;
}

/* Find argmax mask: best reroll decision for a specific dice set.
 * Iterates unique keep-multisets and returns the best mask. */
int ChooseBestRerollMask(const YatzyContext *ctx,
                         const double E_ds_for_masks[252],
                         const int dice[5],
                         double *best_ev) {
    int sorted_dice[5] = {dice[0], dice[1], dice[2], dice[3], dice[4]};
    SortDiceSet(sorted_dice);
    int ds_index = FindDiceSetIndex(ctx, sorted_dice);

    const KeepTable *kt = &ctx->keep_table;

    /* mask=0: keep all */
    double best_val = E_ds_for_masks[ds_index];
    int best_mask = 0;

    for (int j = 0; j < kt->unique_count[ds_index]; j++) {
        int kid = kt->unique_keep_ids[ds_index][j];
        int start = kt->row_start[kid];
        int end = kt->row_start[kid + 1];
        double ev = 0.0;
        for (int k = start; k < end; k++) {
            ev += kt->vals[k] * E_ds_for_masks[kt->cols[k]];
        }
        if (ev > best_val) {
            best_val = ev;
            best_mask = kt->keep_to_mask[ds_index * 32 + j];
        }
    }

    *best_ev = best_val;
    return best_mask;
}

/* Groups 5 & 3 (API path): propagate expected values with mask tracking.
 * E(S, r, n) = max_{mask} Σ P(r'→r'') · E_prev[r'']
 *
 * Iterates unique keep-multisets per dice set (avg ~16 vs 31 masks).
 * Sparse per-row storage avoids wasted zero-multiply overhead. */
void ComputeExpectedValuesForNRerolls(const YatzyContext *ctx,
                                      double E_ds_prev[252],
                                      double E_ds_current[252],
                                      int best_mask_for_n[252]) {
    const KeepTable *kt = &ctx->keep_table;

    #pragma omp parallel for schedule(static)
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        /* mask=0 is identity: keeping all dice yields E_ds_prev[ds_i] */
        double best_val = E_ds_prev[ds_i];
        int best_mask = 0;
        for (int j = 0; j < kt->unique_count[ds_i]; j++) {
            int kid = kt->unique_keep_ids[ds_i][j];
            int start = kt->row_start[kid];
            int end = kt->row_start[kid + 1];
            double ev = 0.0;
            for (int k = start; k < end; k++) {
                ev += kt->vals[k] * E_ds_prev[kt->cols[k]];
            }
            if (ev > best_val) {
                best_val = ev;
                best_mask = kt->keep_to_mask[ds_i * 32 + j];
            }
        }
        E_ds_current[ds_i] = best_val;
        best_mask_for_n[ds_i] = best_mask;
    }
}

/* DP-only variant: computes E_ds_current without tracking optimal masks.
 * Iterates unique keep-multisets per dice set for dedup + sparse dot products.
 * Static so the compiler can inline into ComputeExpectedStateValue. */
static void ComputeMaxEVForNRerolls(const YatzyContext *ctx,
                                     const double E_ds_prev[252],
                                     double E_ds_current[252]) {
    const KeepTable *kt = &ctx->keep_table;
    const double * restrict v = kt->vals;
    const int * restrict c = kt->cols;

    for (int ds_i = 0; ds_i < 252; ds_i++) {
        double best_val = E_ds_prev[ds_i]; /* mask=0: keep all */
        for (int j = 0; j < kt->unique_count[ds_i]; j++) {
            int kid = kt->unique_keep_ids[ds_i][j];
            int start = kt->row_start[kid];
            int end = kt->row_start[kid + 1];
            double ev = 0.0;
            for (int k = start; k < end; k++) {
                ev += v[k] * E_ds_prev[c[k]];
            }
            if (ev > best_val) best_val = ev;
        }
        E_ds_current[ds_i] = best_val;
    }
}

/* SOLVE_WIDGET(S): compute E(S) for one turn-start state.
 *
 * Evaluates the widget bottom-up:
 *   1. Group 6: E[0][r] = best category score for each final roll
 *   2. Group 5: E[1][r] = best reroll from E[0] (1 reroll remaining)
 *   3. Group 3: E[0][r] = best reroll from E[1] (2 rerolls remaining, reuses buf)
 *   4. Group 1: E(S) = Σ P(⊥→r) · E[0][r]
 *
 * Uses ping-pong buffers to halve stack footprint (4 KB instead of 6 KB).
 *
 * See pseudocode Phase 2a: SOLVE_WIDGET. */
double ComputeExpectedStateValue(const YatzyContext *ctx,
                                 const YatzyState *S) {
    double E[2][252]; /* ping-pong buffers */

    int up_score = S->upper_score;
    int scored = S->scored_categories;

    /* Group 6: E(S, r, 0) for all r — uses ds_index variant */
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        E[0][ds_i] = ComputeBestScoringValueForDiceSetByIndex(ctx, up_score, scored, ds_i);
    }

    /* Group 5: E(S, r, 1) = max_{mask} Σ P(r'→r'') · E[0][r''] */
    ComputeMaxEVForNRerolls(ctx, E[0], E[1]);

    /* Group 3: E(S, r, 2) = max_{mask} Σ P(r'→r'') · E[1][r''] */
    ComputeMaxEVForNRerolls(ctx, E[1], E[0]);

    /* Group 1: E(S) = Σ P(⊥→r) · E[0][r] */
    double E_S = 0.0;
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        E_S += ctx->dice_set_probabilities[ds_i] * E[0][ds_i];
    }

    return E_S;
}
