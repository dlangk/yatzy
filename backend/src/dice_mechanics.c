/*
 * dice_mechanics.c — Phase 0: Dice enumeration & probability
 *
 * Implements dice multiset operations on R_{5,6} (the 252 sorted 5-dice
 * multisets) and probability calculations P(⊥ → r).
 *
 * See: theory/optimal_yahtzee_pseudocode.md, Phase 0: PRECOMPUTE_ROLLS_AND_PROBABILITIES.
 */
#include <stdlib.h>
#include <math.h>

#include <context.h>

/* Simulation helper: roll all 5 dice uniformly at random. */
void RollDice(int dice[5]) {
    for (int i = 0; i < 5; i++) {
        dice[i] = rand() % 6 + 1;
    }
}

/* Simulation helper: reroll dice indicated by mask (bit i = reroll die i). */
void RerollDice(int dice[5], const int mask) {
    for (int i = 0; i < 5; i++) {
        if (mask & (1 << i)) {
            dice[i] = rand() % 6 + 1;
        }
    }
}

/* Normalize dice to canonical sorted form (ascending).
 * All dice sets in R_{5,6} are stored sorted for deduplication. */
void SortDiceSet(int arr[5]) {
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 5; j++) {
            if (arr[j] < arr[i]) {
                const int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }
}

/* Map a sorted dice set to its index in R_{5,6} (0–251).
 * Uses the precomputed 5D index_lookup table for O(1) access. */
int FindDiceSetIndex(const YatzyContext *ctx, const int dice[5]) {
    return ctx->index_lookup[dice[0] - 1][dice[1] - 1][dice[2] - 1][dice[3] - 1][dice[4] - 1];
}

/* Compute P(⊥ → r): probability of rolling this exact sorted dice set
 * from 5 fresh dice.
 *
 * Formula: multinomial(5; face_counts) / 6^5
 *        = 5! / (n1! · n2! · ... · n6!) / 6^5
 *
 * See pseudocode Phase 0: "For each full roll r in R_5,6:
 *   Compute P(⊥ → r) = multinomial(5; counts of each face in r) / 6^5" */
double ComputeProbabilityOfDiceSet(const YatzyContext *ctx, const int dice[5]) {
    int face_count[7];
    for (int i = 1; i <= 6; i++) face_count[i] = 0;
    for (int i = 0; i < 5; i++) face_count[dice[i]]++;

    const int numerator = ctx->factorial[5];
    int denominator = 1;
    for (int f = 1; f <= 6; f++) {
        if (face_count[f] > 1) {
            const int fc = face_count[f];
            int sub_fact = 1;
            for (int x = 2; x <= fc; x++) sub_fact *= x;
            denominator *= sub_fact;
        }
    }

    const double permutations = (double) numerator / (double) denominator;
    const double total_outcomes = pow(6, 5);
    return permutations / total_outcomes;
}

/* Count occurrences of each face (1–6) in a 5-dice set.
 * face_count[0] is unused; face_count[f] = count of face f. */
void CountFaces(const int dice[5], int face_count[7]) {
    for (int i = 1; i <= 6; i++) face_count[i] = 0;
    for (int i = 0; i < 5; i++) face_count[dice[i]]++;
}

/* Scoring helper for N-of-a-kind categories.
 * Returns highest_face * n if any face appears >= n times, else 0. */
int NOfAKindScore(const int face_count[7], int n) {
    for (int face = 6; face >= 1; face--) {
        if (face_count[face] >= n) return face * n;
    }
    return 0;
}
