#include "context.h"
#include "game_mechanics.h"

#ifndef PRECOMPUTATIONS_H
#define PRECOMPUTATIONS_H

typedef struct {
    double ev;
    double probability;
    int ds2_index;
} EVProbabilityPair;

// Function declarations
void PrecomputeCategoryScores(YatzyContext *ctx);
void PrecomputeFactorials(YatzyContext *ctx);
void BuildAllDiceCombinations(YatzyContext *ctx);
void PrecomputeScoredCategoryCounts(YatzyContext *ctx);
void PrecomputeRerollTransitionProbabilities(YatzyContext *ctx);
void PrecomputeDiceSetProbabilities(YatzyContext *ctx);
void InitializeFinalStates(YatzyContext *ctx);

void ComputeDistributionForRerollMask(const YatzyContext *ctx,
                                      int ds_index,
                                      const double E_ds_for_masks[252],
                                      int mask,
                                      EVProbabilityPair out_distribution[252]);

double ComputeBestScoringValueForDiceSet(const YatzyContext *ctx,
                                         const YatzyState *S,
                                         const int dice[5]);

void ComputeEDs0ForState(const YatzyContext *ctx,
                         int upper_score,
                         int scored_categories,
                         double E_ds_0[252]);

int ChooseBestRerollMask(const YatzyContext *ctx,
                         const double E_ds_for_masks[252],
                         const int dice[5],
                         double *best_ev);

void ComputeBestRerollStrategy(const YatzyContext *ctx,
                               int upper_score,
                               int scored_categories,
                               const int dice[5],
                               int rerolls_remaining,
                               int *best_mask,
                               double *best_ev);

void ComputeExpectedValuesForNRerolls(const YatzyContext *ctx,
                                      int n,
                                      double E_ds_prev[252],
                                      double E_ds_current[252],
                                      int best_mask_for_n[252]);

double ComputeExpectedStateValue(const YatzyContext *ctx,
                                 const YatzyState *S);

double ComputeExpectedValueForRerollMask(const YatzyContext *ctx,
                                                int ds_index,
                                                const double E_ds_for_masks[252],
                                                int mask);

int ChooseBestCategoryNoRerolls(const YatzyContext *ctx,
                                       int upper_score,
                                       int scored_categories,
                                       const int dice[5],
                                       double *best_ev);

double EvaluateChosenRerollMask(const YatzyContext *ctx,
                                       int upper_score,
                                       int scored_categories,
                                       const int dice[5],
                                       int chosen_mask,
                                       int rerolls_remaining);

double EvaluateChosenCategory(const YatzyContext *ctx,
                                     int upper_score,
                                     int scored_categories,
                                     const int dice[5],
                                     int chosen_category);

#endif