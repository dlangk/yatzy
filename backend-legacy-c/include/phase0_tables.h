/*
 * phase0_tables.h — Phase 0: Precompute lookup tables
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
#ifndef PHASE0_TABLES_H
#define PHASE0_TABLES_H

#include "context.h"

/* Precompute s(S, r, c) for all r ∈ R_{5,6} and all 15 categories. */
void PrecomputeCategoryScores(YatzyContext *ctx);

/* Precompute factorials 0!..5! for multinomial coefficient calculations. */
void PrecomputeFactorials(YatzyContext *ctx);

/* Enumerate all 252 sorted 5-dice multisets R_{5,6} and build reverse lookup. */
void BuildAllDiceCombinations(YatzyContext *ctx);

/* Popcount cache: maps each scored-category bitmask to |C|. */
void PrecomputeScoredCategoryCounts(YatzyContext *ctx);

/* Precompute dense keep-multiset transition table. */
void PrecomputeKeepTable(YatzyContext *ctx);

/* Precompute P(⊥ → r) for all r ∈ R_{5,6}. */
void PrecomputeDiceSetProbabilities(YatzyContext *ctx);

/* Phase 2 base case: E(S) = 50 if m >= 63, else 0 for terminal states (|C|=15). */
void InitializeFinalStates(YatzyContext *ctx);

/* Phase 1: mark reachable (upper_mask, upper_score) pairs via DP.
 * Populates ctx->reachable[64][64]. */
void PrecomputeReachability(YatzyContext *ctx);

#endif // PHASE0_TABLES_H
