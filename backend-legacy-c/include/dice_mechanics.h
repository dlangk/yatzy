/*
 * dice_mechanics.h — Dice multiset operations and probability (Phase 0)
 *
 * Operations on R_{5,6} (the 252 distinct sorted 5-dice multisets) and
 * probability calculations P(⊥ → r).
 * See: theory/optimal_yahtzee_pseudocode.md, Phase 0.
 */
#pragma once
#include "context.h"

/* Count occurrences of each face (1–6) in a 5-dice set. */
void CountFaces(const int dice[5], int face_count[7]);

/* Scoring helper: highest face with >= n occurrences, times n. */
int NOfAKindScore(const int face_count[7], int n);

/* Compute P(⊥ → r): probability of rolling dice[] from 5 fresh dice.
 * Uses multinomial coefficient: 5! / prod(face_count[f]!) / 6^5. */
double ComputeProbabilityOfDiceSet(const YatzyContext *ctx, const int dice[5]);

/* Normalize dice to canonical sorted form (ascending). */
void SortDiceSet(int arr[5]);

/* Map a sorted dice set to its index in R_{5,6} (0–251). */
int FindDiceSetIndex(const YatzyContext *ctx, const int dice[5]);

/* Simulation helpers: roll/reroll dice (uses rand()). */
void RollDice(int dice[5]);
void RerollDice(int dice[5], int mask);
