#ifndef PRECOMPUTE_SCORES_H
#define PRECOMPUTE_SCORES_H

#include "yatzy.h"

// Struct for expected value and probability distribution
typedef struct {
    double ev;           // Expected value for a dice set
    double probability;  // Probability of this dice set
    int ds2_index;       // Index of the resulting dice set
} EVProbabilityPair;

// Precomputation functions
void PrecomputeCategoryScores(YatzyContext *ctx); // Ensure this function is included here
void PrecomputeRerollTransitionProbabilities(YatzyContext *ctx);
void PrecomputeFactorials(YatzyContext *ctx);
void PrecomputeDiceSetProbabilities(YatzyContext *ctx);
void InitializeFinalStates(YatzyContext *ctx);

// Utility functions for state computation
double GetStateValue(const YatzyContext *ctx, int up, int scored);
double ComputeBestScoringValueForDiceSet(const YatzyContext *ctx, const YatzyState *S, const int dice[5]);
void ComputeExpectedValuesForNRerolls(const YatzyContext *ctx, int n, double E_ds_prev[252], 
                                      double E_ds_current[252], int best_mask_for_n[252]);
double ComputeExpectedStateValue(const YatzyContext *ctx, const YatzyState *S);

// Functions for reroll mask distribution
void ComputeDistributionForRerollMask(const YatzyContext *ctx, int ds_index, 
                                      const double E_ds_for_masks[252], int mask, 
                                      EVProbabilityPair out_distribution[252]);

// Comprehensive state computation
void ComputeAllStateValues(YatzyContext *ctx);

// State-specific calculations
void ComputeEDs0ForState(const YatzyContext *ctx, int upper_score, int scored_categories, double E_ds_0[252]);
double ComputeExpectedValueForRerollMask(const YatzyContext *ctx, int ds_index, 
                                         const double E_ds_for_masks[252], int mask);

#endif // PRECOMPUTE_SCORES_H