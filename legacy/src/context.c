/*
 * context.c — Context lifecycle and Phase 0 orchestration
 *
 * Allocates YatzyContext and orchestrates Phase 0 table construction
 * (all precomputed lookup tables needed before the DP pass).
 *
 * See: theory/optimal_yahtzee_pseudocode.md, Phase 0.
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/mman.h>

#include "context.h"
#include "phase0_tables.h"
#include "timing.h"

/* Look up E_table[S] for state (m, C). */
double GetStateValue(const YatzyContext *ctx, int up, int scored) {
    return ctx->state_values[STATE_INDEX(up, scored)];
}

/* Allocate and zero-initialize a YatzyContext.
 * Sets up category names and allocates E_table (state_values). */
YatzyContext *CreateYatzyContext() {
    YatzyContext *ctx = malloc(sizeof(YatzyContext));
    if (!ctx) return NULL;

    memset(ctx, 0, sizeof(YatzyContext));

    ctx->category_names[0] = "Ones";
    ctx->category_names[1] = "Twos";
    ctx->category_names[2] = "Threes";
    ctx->category_names[3] = "Fours";
    ctx->category_names[4] = "Fives";
    ctx->category_names[5] = "Sixes";
    ctx->category_names[6] = "One Pair";
    ctx->category_names[7] = "Two Pairs";
    ctx->category_names[8] = "Three of a Kind";
    ctx->category_names[9] = "Four of a Kind";
    ctx->category_names[10] = "Small Straight";
    ctx->category_names[11] = "Large Straight";
    ctx->category_names[12] = "Full House";
    ctx->category_names[13] = "Chance";
    ctx->category_names[14] = "Yatzy";

    ctx->state_values = (float *) malloc(NUM_STATES * sizeof(float));
    if (!ctx->state_values) {
        free(ctx);
        return NULL;
    }
    for (int i = 0; i < NUM_STATES; i++) ctx->state_values[i] = 0.0f;

    return ctx;
}

/*
 * Phase 0 orchestrator: build all static lookup tables in dependency order.
 *
 * Order matters:
 *   1. Factorials         — needed by multinomial P(⊥ → r) calculations
 *   2. Dice combinations  — enumerates R_{5,6} (252 sorted 5-dice multisets)
 *   3. Category scores    — precomputes s(S, r, c) for all (r, c)
 *   4. Reroll transitions — P(r' → r) for all (dice_set, reroll_mask) pairs
 *   5. Dice set probs     — P(⊥ → r) for all r ∈ R_{5,6}
 *   6. Scored cat counts  — popcount cache for 2^15 bitmasks
 *   7. Final states       — Phase 2 base case: E(S) for |C| = 15
 *   8. Reachability       — Phase 1: mark reachable (upper_mask, upper_score) pairs
 *
 * See pseudocode Phase 0: PRECOMPUTE_ROLLS_AND_PROBABILITIES.
 */
void PrecomputeLookupTables(YatzyContext *ctx) {
    printf("=== Phase 0: Precompute Lookup Tables ===\n");
    double phase0_start = timer_now();

    TIMER_BLOCK("Factorials",              PrecomputeFactorials(ctx));
    TIMER_BLOCK("Dice combinations (252)", BuildAllDiceCombinations(ctx));
    TIMER_BLOCK("Category scores",         PrecomputeCategoryScores(ctx));
    TIMER_BLOCK("Keep-multiset table",     PrecomputeKeepTable(ctx));
    TIMER_BLOCK("Dice set probabilities",  PrecomputeDiceSetProbabilities(ctx));
    TIMER_BLOCK("Scored category counts",  PrecomputeScoredCategoryCounts(ctx));
    TIMER_BLOCK("Terminal states",         InitializeFinalStates(ctx));
    TIMER_BLOCK("Reachability pruning",    PrecomputeReachability(ctx));

    printf("  %-42s %8.3f ms\n", "TOTAL Phase 0", timer_elapsed_ms(phase0_start));
    printf("\n");
}

/* Free all memory owned by the context. Safe to call with NULL. */
void FreeYatzyContext(YatzyContext *ctx) {
    if (!ctx) return;
    if (ctx->mmap_base) {
        munmap(ctx->mmap_base, ctx->mmap_size);
    } else {
        free(ctx->state_values);
    }
    /* KeepTable is inline in YatzyContext — no heap to free */
    free(ctx);
}
