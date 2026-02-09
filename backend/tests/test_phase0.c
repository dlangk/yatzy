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

static void test_keep_table_row_sums(YatzyContext *ctx) {
    /* Every probability row in the keep table sums to 1.0 */
    const KeepTable *kt = &ctx->keep_table;
    int all_ok = 1;
    for (int ki = 0; ki < NUM_KEEP_MULTISETS; ki++) {
        int start = kt->row_start[ki];
        int end = kt->row_start[ki + 1];
        if (start == end) continue; /* empty row (unused keep) */
        double sum = 0.0;
        for (int k = start; k < end; k++) {
            sum += kt->vals[k];
        }
        if (fabs(sum - 1.0) > 1e-9) {
            all_ok = 0;
            fprintf(stderr, "  keep_table row %d sums to %.12f\n", ki, sum);
            break;
        }
    }
    ASSERT(all_ok, "Keep table: all non-zero probability rows sum to 1.0");
}

static void test_keep_table_enumeration(YatzyContext *ctx) {
    /* Check unique_count bounds for known dice sets */
    const KeepTable *kt = &ctx->keep_table;

    /* ds=0 is [1,1,1,1,1] (all same): only 5 unique keeps
     * (keep 4, keep 3, keep 2, keep 1, keep 0) */
    ASSERT(kt->unique_count[0] == 5,
           "Keep table: [1,1,1,1,1] has 5 unique keeps");

    /* ds=251 is [6,6,6,6,6] (all same): also 5 unique keeps */
    ASSERT(kt->unique_count[251] == 5,
           "Keep table: [6,6,6,6,6] has 5 unique keeps");

    /* Find [1,2,3,4,5] (all distinct): should have 31 unique keeps
     * (every mask 1-31 produces a distinct keep multiset) */
    int ds_12345 = ctx->index_lookup[0][1][2][3][4]; /* [1,2,3,4,5] */
    ASSERT(kt->unique_count[ds_12345] == 31,
           "Keep table: [1,2,3,4,5] has 31 unique keeps (all distinct)");

    /* Average should be roughly 18-20 */
    int total = 0;
    for (int ds = 0; ds < 252; ds++) total += kt->unique_count[ds];
    double avg = (double)total / 252;
    ASSERT(avg > 15.0 && avg < 25.0,
           "Keep table: average unique keeps per ds is reasonable (15-25)");
}

static void test_keep_table_reroll_all(YatzyContext *ctx) {
    /* mask=31 (reroll all 5) maps to the same keep (empty set) for all dice sets.
     * The empty keep's probabilities should match dice_set_probabilities. */
    const KeepTable *kt = &ctx->keep_table;

    /* All dice sets with mask=31 should map to the same keep index */
    int empty_kid = kt->mask_to_keep[0 * 32 + 31]; /* ds=0, mask=31 */
    int all_same = 1;
    for (int ds = 1; ds < 252; ds++) {
        if (kt->mask_to_keep[ds * 32 + 31] != empty_kid) {
            all_same = 0;
            fprintf(stderr, "  ds=%d mask=31 keep=%d != expected %d\n",
                    ds, kt->mask_to_keep[ds * 32 + 31], empty_kid);
            break;
        }
    }
    ASSERT(all_same, "Keep table: mask=31 maps to same keep (empty) for all ds");

    /* Empty keep probs should match dice_set_probabilities.
     * Build dense row from sparse storage for comparison. */
    double empty_probs[252] = {0.0};
    int start = kt->row_start[empty_kid];
    int end = kt->row_start[empty_kid + 1];
    for (int k = start; k < end; k++) {
        empty_probs[kt->cols[k]] = kt->vals[k];
    }

    int probs_match = 1;
    for (int t = 0; t < 252; t++) {
        if (fabs(empty_probs[t] - ctx->dice_set_probabilities[t]) > 1e-9) {
            probs_match = 0;
            fprintf(stderr, "  empty keep prob[%d]=%.12f != dice_set_prob=%.12f\n",
                    t, empty_probs[t], ctx->dice_set_probabilities[t]);
            break;
        }
    }
    ASSERT(probs_match, "Keep table: empty keep probs match dice_set_probabilities");
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
    test_keep_table_row_sums(ctx);
    test_keep_table_enumeration(ctx);
    test_keep_table_reroll_all(ctx);
    test_precomputed_scores_match(ctx);
    test_popcount_cache(ctx);
    test_initial_roll_probs(ctx);
    test_index_round_trip(ctx);

    FreeYatzyContext(ctx);
TEST_MAIN_END
