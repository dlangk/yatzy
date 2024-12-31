#ifndef YATZY_H
#define YATZY_H

#include <math.h>
#include <string.h>
#include <stdbool.h>
#include "constants.h"

// --------------------------- Structures ---------------------------

typedef struct {
    int upper_score;         // Upper section score
    int scored_categories;   // Bitmask of scored categories
} YatzyState;

typedef struct YatzyContext {
    char *category_names[CATEGORY_COUNT];
    double *state_values;
    int scored_category_count_cache[1 << CATEGORY_COUNT];
    int all_dice_sets[252][5];
    int index_lookup[6][6][6][6][6];
    int precomputed_scores[252][CATEGORY_COUNT];
    double transition_table[252][32][252];
    double dice_set_probabilities[252]; // Add this member
    int factorial[6];
    int num_combinations;
} YatzyContext;

// --------------------------- Function Declarations ---------------------------

void PrecomputeCategoryScores(YatzyContext *ctx);
void PrecomputeRerollTransitionProbabilities(YatzyContext *ctx);
void PrecomputeDiceSetProbabilities(YatzyContext *ctx);
void InitializeFinalStates(YatzyContext *ctx);

// Category and Scoring Functions
int categoryToIndex(const char *category_name);
int CalculateCategoryScore(const int dice[5], int category);

void SortDiceSet(int arr[5]);
void CountFaces(const int dice[5], int face_count[7]);
int NOfAKindScore(const int face_count[7], int n);

int FindDiceSetIndex(const YatzyContext *ctx, const int dice[5]);
int UpdateUpperScore(int upper_score, int category, int score);

double GetStateValue(const YatzyContext *ctx, int up, int scored);

// Precomputation Functions
void PrecomputeScoredCategoryCounts(YatzyContext *ctx);
void BuildAllDiceCombinations(YatzyContext *ctx);

// Dice Rolling Functions
void RollDice(int dice[5]);
void RerollDice(int dice[5], int mask);

// Context Management
YatzyContext *CreateYatzyContext();
void FreeYatzyContext(YatzyContext *ctx);

// Decision-Making Functions
int ChooseBestCategoryNoRerolls(const YatzyContext *ctx, int upper_score, int scored_categories,
                                const int dice[5], double *best_ev);
int ChooseBestRerollMask(const YatzyContext *ctx, const double E_ds_for_masks[252],
                         const int dice[5], double *best_ev);
void ComputeBestRerollStrategy(const YatzyContext *ctx, int upper_score, int scored_categories,
                               const int dice[5], int rerolls_remaining, int *best_mask, double *best_ev);

// Evaluation Functions
double SuggestOptimalAction(const YatzyContext *ctx, int upper_score, int scored_categories,
                            const int dice[5], int rerolls_remaining);
double EvaluateChosenCategory(const YatzyContext *ctx, int upper_score, int scored_categories,
                              const int dice[5], int chosen_category);
double EvaluateChosenRerollMask(const YatzyContext *ctx, int upper_score, int scored_categories,
                                 const int dice[5], int chosen_mask, int rerolls_remaining);
double EvaluateAction(const YatzyContext *ctx, int upper_score, int scored_categories,
                      const int dice[5], int action, int rerolls_remaining);

// Simulation and Utility Functions
void WriteResultsToCSV(const char *filename, const int *scores, int num_scores);
int SimulateSingleGame(YatzyContext *ctx);
int MaskFromBinaryString(const char *action_str);

#endif // YATZY_H