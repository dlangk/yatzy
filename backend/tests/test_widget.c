/*
 * test_widget.c — Widget solver correctness + timing benchmarks
 *
 * Requires full Phase 0 context. Tests late-game states where we can
 * verify expected values by hand.
 */
#include "test_helpers.h"
#include "context.h"
#include "phase0_tables.h"
#include "widget_solver.h"
#include "dice_mechanics.h"
#include "timing.h"

/*
 * Late-game correctness: only Yatzy (cat 14) remaining, upper=0.
 *
 * With 14 categories scored and only Yatzy left:
 *   - For [6,6,6,6,6]: score Yatzy = 50, successor has all scored,
 *     upper stays 0 → E(successor) = 0 → total = 50
 *   - For [1,2,3,4,5]: score Yatzy = 0, E(successor) = 0 → total = 0
 */
static void test_late_game_scoring(YatzyContext *ctx) {
    int all_but_yatzy = ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_YATZY);

    /* ComputeBestScoringValueForDiceSet with [6,6,6,6,6] */
    YatzyState state = {0, all_but_yatzy};
    int d1[] = {6,6,6,6,6};
    double ev1 = ComputeBestScoringValueForDiceSet(ctx, &state, d1);
    ASSERT_EQ_DBL(ev1, 50.0,
                  "Late game [6,6,6,6,6] only Yatzy left → EV = 50");

    /* ComputeBestScoringValueForDiceSet with [1,2,3,4,5] → score 0 */
    int d2[] = {1,2,3,4,5};
    double ev2 = ComputeBestScoringValueForDiceSet(ctx, &state, d2);
    ASSERT_EQ_DBL(ev2, 0.0,
                  "Late game [1,2,3,4,5] only Yatzy left → EV = 0");
}

/*
 * Only Chance remaining, upper=0:
 *   - For [6,6,6,6,6]: score Chance = 30, successor E=0 → total = 30
 *   - For [1,1,1,1,1]: score Chance = 5, successor E=0 → total = 5
 */
static void test_late_game_chance_only(YatzyContext *ctx) {
    int all_but_chance = ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_CHANCE);

    YatzyState state = {0, all_but_chance};
    int d1[] = {6,6,6,6,6};
    double ev1 = ComputeBestScoringValueForDiceSet(ctx, &state, d1);
    ASSERT_EQ_DBL(ev1, 30.0,
                  "Late game [6,6,6,6,6] only Chance left → EV = 30");

    int d2[] = {1,1,1,1,1};
    double ev2 = ComputeBestScoringValueForDiceSet(ctx, &state, d2);
    ASSERT_EQ_DBL(ev2, 5.0,
                  "Late game [1,1,1,1,1] only Chance left → EV = 5");
}

/*
 * Reroll mask=0 (keep all) should equal E_ds_0[ds_index].
 */
static void test_keep_all_mask(YatzyContext *ctx) {
    int all_but_yatzy = ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_YATZY);

    /* Compute E_ds_0 inline (Group 6 for all rolls) */
    double E_ds_0[252];
    YatzyState state = {0, all_but_yatzy};
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        E_ds_0[ds_i] = ComputeBestScoringValueForDiceSet(ctx, &state, ctx->all_dice_sets[ds_i]);
    }

    /* For ds_index=251 ([6,6,6,6,6]), mask=0 should give E_ds_0[251] */
    double ev_keep = ComputeExpectedValueForRerollMask(ctx, 251, E_ds_0, 0);
    ASSERT_EQ_DBL(ev_keep, E_ds_0[251],
                  "mask=0 (keep all) equals E_ds_0[ds_index]");

    /* For ds_index=0 ([1,1,1,1,1]), mask=0 should give E_ds_0[0] */
    double ev_keep0 = ComputeExpectedValueForRerollMask(ctx, 0, E_ds_0, 0);
    ASSERT_EQ_DBL(ev_keep0, E_ds_0[0],
                  "mask=0 (keep all) equals E_ds_0[0] for [1,1,1,1,1]");
}

/*
 * ChooseBestRerollMask: mask ∈ [0,31] and ev >= E_ds for mask=0.
 */
static void test_choose_best_reroll(YatzyContext *ctx) {
    int all_but_yatzy = ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_YATZY);

    /* Compute E_ds_0 inline */
    double E_ds_0[252];
    YatzyState state = {0, all_but_yatzy};
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        E_ds_0[ds_i] = ComputeBestScoringValueForDiceSet(ctx, &state, ctx->all_dice_sets[ds_i]);
    }

    for (int ds = 0; ds < 252; ds += 50) { /* spot-check every 50th */
        double best_ev;
        int mask = ChooseBestRerollMask(ctx, E_ds_0, ctx->all_dice_sets[ds], &best_ev);
        ASSERT(mask >= 0 && mask < 32, "ChooseBestRerollMask returns mask in [0,31]");

        double ev_keep = ComputeExpectedValueForRerollMask(ctx, ds, E_ds_0, 0);
        ASSERT(best_ev >= ev_keep - 1e-9,
               "Best reroll EV >= keep-all EV");
    }
}

/*
 * Timing benchmarks — print performance data for widget solver components.
 */
static void bench_widget(YatzyContext *ctx) {
    printf("\n--- Widget Solver Benchmarks ---\n");

    /* Benchmark Group 6 (252 scoring evaluations) */
    int all_but_yatzy = ((1 << CATEGORY_COUNT) - 1) ^ (1 << CATEGORY_YATZY);
    double E_ds_0[252];
    double E_ds_1[252];
    int best_masks[252];

    YatzyState state = {0, all_but_yatzy};
    double t0 = timer_now();
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        E_ds_0[ds_i] = ComputeBestScoringValueForDiceSet(ctx, &state, ctx->all_dice_sets[ds_i]);
    }
    double dt_group6 = timer_elapsed_ms(t0);
    printf("  %-42s %8.3f ms\n", "Group 6 (inline loop)", dt_group6);

    /* Benchmark ComputeExpectedValuesForNRerolls (Group 5: 252×32×252) */
    t0 = timer_now();
    ComputeExpectedValuesForNRerolls(ctx, E_ds_0, E_ds_1, best_masks);
    double dt_group5 = timer_elapsed_ms(t0);
    printf("  %-42s %8.3f ms\n", "Group 5 (N-rerolls, 1 level)", dt_group5);

    /* Benchmark full ComputeExpectedStateValue (early game, |C|=1) */
    YatzyState early = {0, 1}; /* only Ones scored */
    t0 = timer_now();
    double ev = ComputeExpectedStateValue(ctx, &early);
    double dt_full = timer_elapsed_ms(t0);
    printf("  %-42s %8.3f ms  (EV=%.4f)\n",
           "Full widget solve (|C|=1)", dt_full, ev);

    /* Benchmark full widget for empty state */
    YatzyState empty = {0, 0};
    t0 = timer_now();
    double ev0 = ComputeExpectedStateValue(ctx, &empty);
    double dt_empty = timer_elapsed_ms(t0);
    printf("  %-42s %8.3f ms  (EV=%.4f)\n",
           "Full widget solve (|C|=0)", dt_empty, ev0);
}

TEST_MAIN_BEGIN("test_widget")
    YatzyContext *ctx = CreateYatzyContext();
    ASSERT(ctx != NULL, "CreateYatzyContext succeeds");
    PrecomputeLookupTables(ctx);

    /* Set up terminal states for late-game tests */
    InitializeFinalStates(ctx);

    test_late_game_scoring(ctx);
    test_late_game_chance_only(ctx);
    test_keep_all_mask(ctx);
    test_choose_best_reroll(ctx);
    bench_widget(ctx);

    FreeYatzyContext(ctx);
TEST_MAIN_END
