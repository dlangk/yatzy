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
 * precomputed future value of the successor state n(S,r,c). */
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

/* Group 6 for all rolls: compute E(S, r, 0) for every r ∈ R_{5,6}.
 * Fills E_ds_0[252] — the "exit values" of the widget. */
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

/* Single mask evaluation: Σ_{r''} P(r'→r'') · E_prev[r''].
 * Computes the expected value of keeping the dice indicated by ~mask
 * and rerolling the rest, given E_prev values for the next stage. */
double ComputeExpectedValueForRerollMask(const YatzyContext *ctx,
                                         int ds_index,
                                         const double E_ds_for_masks[252],
                                         int mask) {
    const SparseTransitionTable *sp = &ctx->sparse_transitions;
    int row = ds_index * 32 + mask;
    int start = sp->row_offsets[row];
    int end = sp->row_offsets[row + 1];
    double ev = 0.0;
    for (int k = start; k < end; k++) {
        ev += sp->values[k] * E_ds_for_masks[sp->col_indices[k]];
    }
    return ev;
}

/* Find argmax mask: best reroll decision for a specific dice set.
 * Tries all 32 masks and returns the one maximizing expected value. */
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

/* Groups 5 & 3: propagate expected values from reroll level n-1 to n.
 * E(S, r, n) = max_{mask} Σ P(r'→r'') · E_prev[r'']
 *
 * For each dice set ds_i (= roll r), tries all 32 reroll masks and picks
 * the one maximizing the weighted sum over transition outcomes.
 *
 * Uses the sparse CSR transition table for ~5× fewer memory accesses.
 *
 * See pseudocode Groups 5 and 3 in SOLVE_WIDGET. */
void ComputeExpectedValuesForNRerolls(const YatzyContext *ctx,
                                      int n,
                                      double E_ds_prev[252],
                                      double E_ds_current[252],
                                      int best_mask_for_n[252]) {
    const double *vals = ctx->sparse_transitions.values;
    const int *cols = ctx->sparse_transitions.col_indices;
    const int *offsets = ctx->sparse_transitions.row_offsets;

    #pragma omp parallel for schedule(static)
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        double best_val = -INFINITY;
        int best_mask = 0;
        int row_base = ds_i * 32;
        for (int mask = 0; mask < 32; mask++) {
            int start = offsets[row_base + mask];
            int end = offsets[row_base + mask + 1];
            double ev = 0.0;
            for (int k = start; k < end; k++) {
                ev += vals[k] * E_ds_prev[cols[k]];
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

/* SOLVE_WIDGET(S): compute E(S) for one turn-start state.
 *
 * Evaluates the widget bottom-up:
 *   1. Group 6: E_ds_0[r] = best category score for each final roll
 *   2. Group 5: E_ds_1[r] = best reroll from E_ds_0 (1 reroll remaining)
 *   3. Group 3: E_ds_2[r] = best reroll from E_ds_1 (2 rerolls remaining)
 *   4. Group 1: E(S) = Σ P(⊥→r) · E_ds_2[r]
 *
 * See pseudocode Phase 2a: SOLVE_WIDGET. */
double ComputeExpectedStateValue(const YatzyContext *ctx,
                                 const YatzyState *S) {
    double E_ds_0[252], E_ds_1[252], E_ds_2[252];
    int best_mask_1[252], best_mask_2[252];

    /* Group 6: E(S, r, 0) for all r */
    #pragma omp parallel for schedule(static)
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        E_ds_0[ds_i] = ComputeBestScoringValueForDiceSet(ctx, S, ctx->all_dice_sets[ds_i]);
    }

    /* Group 5: E(S, r, 1) = max_{mask} Σ P(r'→r'') · E_ds_0[r''] */
    ComputeExpectedValuesForNRerolls(ctx, 1, E_ds_0, E_ds_1, best_mask_1);

    /* Group 3: E(S, r, 2) = max_{mask} Σ P(r'→r'') · E_ds_1[r''] */
    ComputeExpectedValuesForNRerolls(ctx, 2, E_ds_1, E_ds_2, best_mask_2);

    /* Group 1: E(S) = Σ P(⊥→r) · E_ds_2[r] */
    double E_S = 0.0;
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        E_S += ctx->dice_set_probabilities[ds_i] * E_ds_2[ds_i];
    }

    return E_S;
}
