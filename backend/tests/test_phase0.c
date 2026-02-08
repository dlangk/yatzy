/*
 * test_phase0.c — Lookup table structural invariants
 *
 * Requires full Phase 0 context. Tests invariants that must hold for
 * correctness of the DP solver.
 */
#include <string.h>
#include "test_helpers.h"
#include "context.h"
#include "dice_mechanics.h"
#include "game_mechanics.h"
#include "phase0_tables.h"

static void test_252_dice_sets(YatzyContext *ctx) {
    ASSERT_EQ_INT(ctx->num_combinations, 252,
                  "252 dice combinations enumerated");

    /* All sorted */
    int all_sorted = 1;
    for (int i = 0; i < 252; i++) {
        for (int j = 0; j < 4; j++) {
            if (ctx->all_dice_sets[i][j] > ctx->all_dice_sets[i][j+1]) {
                all_sorted = 0;
                break;
            }
        }
        if (!all_sorted) break;
    }
    ASSERT(all_sorted, "All 252 dice sets are sorted");

    /* All unique */
    int all_unique = 1;
    for (int i = 0; i < 252 && all_unique; i++) {
        for (int j = i+1; j < 252 && all_unique; j++) {
            if (memcmp(ctx->all_dice_sets[i], ctx->all_dice_sets[j], 5*sizeof(int)) == 0) {
                all_unique = 0;
            }
        }
    }
    ASSERT(all_unique, "All 252 dice sets are unique");
}

static void test_transition_row_sums(YatzyContext *ctx) {
    /* For every (ds, mask), sparse transition probabilities sum to 1.0 */
    const SparseTransitionTable *sp = &ctx->sparse_transitions;
    int all_ok = 1;
    for (int ds = 0; ds < 252; ds++) {
        for (int mask = 0; mask < 32; mask++) {
            int row = ds * 32 + mask;
            int start = sp->row_offsets[row];
            int end = sp->row_offsets[row + 1];
            double sum = 0.0;
            for (int k = start; k < end; k++) {
                sum += sp->values[k];
            }
            if (fabs(sum - 1.0) > 1e-9) {
                all_ok = 0;
                fprintf(stderr, "  sparse_transitions[%d][%d] sums to %.12f\n", ds, mask, sum);
                break;
            }
        }
        if (!all_ok) break;
    }
    ASSERT(all_ok, "Transition rows sum to 1.0 for all (ds, mask)");
}

static void test_transition_identity(YatzyContext *ctx) {
    /* mask=0 (keep all) → exactly 1 non-zero entry: P(ds→ds)=1.0 */
    const SparseTransitionTable *sp = &ctx->sparse_transitions;
    int identity_ok = 1;
    for (int ds = 0; ds < 252; ds++) {
        int row = ds * 32 + 0; /* mask=0 */
        int start = sp->row_offsets[row];
        int end = sp->row_offsets[row + 1];
        /* Should have exactly 1 entry: self-transition */
        if (end - start != 1) {
            identity_ok = 0;
            fprintf(stderr, "  mask=0 ds=%d: expected 1 entry, got %d\n", ds, end - start);
            break;
        }
        if (sp->col_indices[start] != ds || fabs(sp->values[start] - 1.0) > 1e-9) {
            identity_ok = 0;
            break;
        }
    }
    ASSERT(identity_ok, "Transition identity: mask=0 → single entry P(ds→ds)=1");
}

static void test_transition_reroll_all(YatzyContext *ctx) {
    /* mask=31 (reroll all 5) gives same distribution regardless of starting ds.
     * Reconstruct dense rows for ds=0 and ds=N, compare. */
    const SparseTransitionTable *sp = &ctx->sparse_transitions;
    int symmetry_ok = 1;

    /* Build dense reference row for ds=0, mask=31 */
    double ref[252] = {0.0};
    int row0 = 0 * 32 + 31;
    for (int k = sp->row_offsets[row0]; k < sp->row_offsets[row0 + 1]; k++) {
        ref[sp->col_indices[k]] = sp->values[k];
    }

    for (int ds = 1; ds < 252 && symmetry_ok; ds++) {
        double row_vals[252] = {0.0};
        int row = ds * 32 + 31;
        for (int k = sp->row_offsets[row]; k < sp->row_offsets[row + 1]; k++) {
            row_vals[sp->col_indices[k]] = sp->values[k];
        }
        for (int ds2 = 0; ds2 < 252; ds2++) {
            if (fabs(row_vals[ds2] - ref[ds2]) > 1e-9) {
                symmetry_ok = 0;
                break;
            }
        }
    }
    ASSERT(symmetry_ok, "Transition symmetry: mask=31 gives same dist for all starting ds");
}

static void test_precomputed_scores_match(YatzyContext *ctx) {
    /* Spot-check: precomputed_scores[i][c] == CalculateCategoryScore(ds_i, c) */
    int spot_checks[][2] = {
        {0, CATEGORY_ONES},      /* [1,1,1,1,1] Ones */
        {0, CATEGORY_YATZY},     /* [1,1,1,1,1] Yatzy */
        {251, CATEGORY_SIXES},   /* [6,6,6,6,6] Sixes */
        {251, CATEGORY_YATZY},   /* [6,6,6,6,6] Yatzy */
    };
    int n_checks = sizeof(spot_checks) / sizeof(spot_checks[0]);

    int all_match = 1;
    for (int c = 0; c < n_checks; c++) {
        int ds_idx = spot_checks[c][0];
        int cat = spot_checks[c][1];
        int precomp = ctx->precomputed_scores[ds_idx][cat];
        int direct = CalculateCategoryScore(ctx->all_dice_sets[ds_idx], cat);
        if (precomp != direct) {
            all_match = 0;
            fprintf(stderr, "  precomputed_scores[%d][%d]=%d != direct=%d\n",
                    ds_idx, cat, precomp, direct);
        }
    }
    ASSERT(all_match, "Precomputed scores match direct CalculateCategoryScore");

    /* Full check: all 252 × 15 entries */
    int full_match = 1;
    for (int i = 0; i < 252 && full_match; i++) {
        for (int c = 0; c < CATEGORY_COUNT; c++) {
            int precomp = ctx->precomputed_scores[i][c];
            int direct = CalculateCategoryScore(ctx->all_dice_sets[i], c);
            if (precomp != direct) {
                full_match = 0;
                fprintf(stderr, "  Mismatch at ds=%d cat=%d: %d != %d\n",
                        i, c, precomp, direct);
            }
        }
    }
    ASSERT(full_match, "All 252×15 precomputed scores match direct computation");
}

static void test_popcount_cache(YatzyContext *ctx) {
    int all_match = 1;
    for (int i = 0; i < (1 << 15); i++) {
        if (ctx->scored_category_count_cache[i] != __builtin_popcount(i)) {
            all_match = 0;
            fprintf(stderr, "  popcount_cache[%d]=%d != %d\n",
                    i, ctx->scored_category_count_cache[i], __builtin_popcount(i));
            break;
        }
    }
    ASSERT(all_match, "Popcount cache matches __builtin_popcount for all 2^15 values");
}

static void test_initial_roll_probs(YatzyContext *ctx) {
    double sum = 0.0;
    for (int i = 0; i < 252; i++) {
        sum += ctx->dice_set_probabilities[i];
    }
    ASSERT_NEAR(sum, 1.0, 1e-9, "Initial roll probabilities sum to 1.0");
}

static void test_index_round_trip(YatzyContext *ctx) {
    int all_ok = 1;
    for (int i = 0; i < 252; i++) {
        if (FindDiceSetIndex(ctx, ctx->all_dice_sets[i]) != i) {
            all_ok = 0;
            break;
        }
    }
    ASSERT(all_ok, "Index round-trip: FindDiceSetIndex(all_dice_sets[i]) == i for all 252");
}

TEST_MAIN_BEGIN("test_phase0")
    YatzyContext *ctx = CreateYatzyContext();
    ASSERT(ctx != NULL, "CreateYatzyContext succeeds");
    PrecomputeLookupTables(ctx);

    test_252_dice_sets(ctx);
    test_transition_row_sums(ctx);
    test_transition_identity(ctx);
    test_transition_reroll_all(ctx);
    test_precomputed_scores_match(ctx);
    test_popcount_cache(ctx);
    test_initial_roll_probs(ctx);
    test_index_round_trip(ctx);

    FreeYatzyContext(ctx);
TEST_MAIN_END
