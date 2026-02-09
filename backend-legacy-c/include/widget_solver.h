/*
 * widget_solver.h — Phase 2a: SOLVE_WIDGET
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
#ifndef WIDGET_SOLVER_H
#define WIDGET_SOLVER_H

#include "context.h"

/* SOLVE_WIDGET(S): compute E(S) for one turn-start state.
 * Evaluates Groups 6→5→3→1 within the widget. */
double ComputeExpectedStateValue(const YatzyContext *ctx, const YatzyState *S);

/* Group 6, inner loop: best category for one roll r.
 * E(S, r, 0) = max_{c ∉ C} [s(S,r,c) + E(n(S,r,c))] */
double ComputeBestScoringValueForDiceSet(const YatzyContext *ctx,
                                         const YatzyState *S,
                                         const int dice[5]);

/* Groups 5 & 3 (API path): E(S, r, n) = max_{mask} Σ P(r'→r'') · E_prev[r''].
 * Propagates expected values and tracks the best mask per dice set. */
void ComputeExpectedValuesForNRerolls(const YatzyContext *ctx,
                                      double E_ds_prev[252],
                                      double E_ds_current[252],
                                      int best_mask_for_n[252]);

/* Single mask evaluation: Σ P(r'→r'') · E_prev[r''] for one (ds_index, mask). */
double ComputeExpectedValueForRerollMask(const YatzyContext *ctx,
                                         int ds_index,
                                         const double E_ds_for_masks[252],
                                         int mask);

/* Find argmax mask over all 32 reroll masks for a given dice set. */
int ChooseBestRerollMask(const YatzyContext *ctx,
                         const double E_ds_for_masks[252],
                         const int dice[5],
                         double *best_ev);

#endif // WIDGET_SOLVER_H
