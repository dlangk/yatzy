/*
 * api_computations.h — API-facing computation wrappers
 *
 * Composes Phase 2a SOLVE_WIDGET primitives for specific user situations.
 * These functions are called by the webserver HTTP handlers.
 */
#ifndef API_COMPUTATIONS_H
#define API_COMPUTATIONS_H

#include "context.h"

typedef struct {
    double ev;
    double probability;
    int ds2_index;
} EVProbabilityPair;

/* Compute best reroll mask for a specific game situation. */
void ComputeBestRerollStrategy(const YatzyContext *ctx,
                               int upper_score,
                               int scored_categories,
                               const int dice[5],
                               int rerolls_remaining,
                               int *best_mask,
                               double *best_ev);

/* Choose best category when no rerolls remain (score > 0 only). */
int ChooseBestCategoryNoRerolls(const YatzyContext *ctx,
                                int upper_score,
                                int scored_categories,
                                const int dice[5],
                                double *best_ev);

/* Evaluate the EV of a user-chosen reroll mask. */
double EvaluateChosenRerollMask(const YatzyContext *ctx,
                                int upper_score,
                                int scored_categories,
                                const int dice[5],
                                int chosen_mask,
                                int rerolls_remaining);

/* Evaluate the EV of a user-chosen category assignment. */
double EvaluateChosenCategory(const YatzyContext *ctx,
                              int upper_score,
                              int scored_categories,
                              const int dice[5],
                              int chosen_category);

/* Compute E(S, r, n) for all r, for a given number of rerolls.
 * Thread-safe: writes result into caller-provided buffer. */
void ComputeExpectedValues(const YatzyContext *ctx, int upper_score,
                           int scored_categories, int rerolls,
                           double out_E_ds[252]);

/* Build outcome distribution for a specific reroll mask. */
void ComputeDistributionForRerollMask(const YatzyContext *ctx,
                                      int ds_index,
                                      const double E_ds_for_masks[252],
                                      int mask,
                                      EVProbabilityPair out_distribution[252]);

/* Weighted sum of EV over a probability distribution. */
double ComputeEVFromDistribution(const EVProbabilityPair distribution[], int size);

#endif // API_COMPUTATIONS_H
