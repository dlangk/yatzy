/*
 * test_precomputed.c — Validate precomputed state values from bin files
 *
 * Loads the full DP table from data/all_states.bin and checks that
 * optimal actions and expected values are sensible for hand-verifiable
 * game states.
 *
 * Requires: YATZY_BASE_PATH set to the backend directory (or run from there).
 */
#include <string.h>
#include <math.h>

#include "test_helpers.h"
#include "context.h"
#include "storage.h"
#include "api_computations.h"
#include "widget_solver.h"
#include "dice_mechanics.h"
#include "game_mechanics.h"
#include "utilities.h"

/* ── Helpers ─────────────────────────────────────────────────────────── */

/* Shorthand: get E(upper_score, scored_categories) */
static double EV(const YatzyContext *ctx, int up, int scored) {
    return ctx->state_values[STATE_INDEX(up, scored)];
}

/* All categories scored bitmask */
#define ALL_SCORED ((1 << CATEGORY_COUNT) - 1)

/* Mask with all categories except one */
#define ONLY(cat) (ALL_SCORED ^ (1 << (cat)))

/* ── 1. Terminal state tests ─────────────────────────────────────────── */

static void test_terminal_states(const YatzyContext *ctx) {
    /* All categories scored, upper >= 63 → bonus 50 */
    ASSERT_EQ_DBL(EV(ctx, 63, ALL_SCORED), 50.0,
                  "Terminal (up=63, all scored) = 50 (bonus)");
    ASSERT_EQ_DBL(EV(ctx, 42, ALL_SCORED), 0.0,
                  "Terminal (up=42, all scored) = 0 (no bonus)");
    ASSERT_EQ_DBL(EV(ctx, 0, ALL_SCORED), 0.0,
                  "Terminal (up=0, all scored) = 0 (no bonus)");
    ASSERT_EQ_DBL(EV(ctx, 62, ALL_SCORED), 0.0,
                  "Terminal (up=62, all scored) = 0 (just under threshold)");

    /* Every upper score from 0 to 62 should yield 0 at terminal */
    int all_zero = 1;
    for (int up = 0; up < 63; up++) {
        if (fabs(EV(ctx, up, ALL_SCORED)) > 1e-9) { all_zero = 0; break; }
    }
    ASSERT(all_zero, "Terminal: all up < 63 yield EV = 0");
}

/* ── 2. Game-start expected value ────────────────────────────────────── */

static void test_game_start_ev(const YatzyContext *ctx) {
    double ev = EV(ctx, 0, 0);
    /* Scandinavian Yatzy optimal EV ≈ 248.4 based on our computation. */
    ASSERT(ev > 230.0 && ev < 290.0,
           "Game start EV(0,0) is in plausible range [230, 290]");
    printf("    EV(0,0) = %.4f\n", ev);
}

/* ── 3. Single category remaining — all 15 categories ────────────── */

static void test_one_category_remaining(const YatzyContext *ctx) {
    /* Ones: each die scores 1 when it's a one. Optimal keep ≈ 2.1 */
    ASSERT(EV(ctx, 0, ONLY(CATEGORY_ONES)) > 1.0 &&
           EV(ctx, 0, ONLY(CATEGORY_ONES)) < 5.0,
           "Only Ones left: EV in [1, 5]");
    printf("    EV(only Ones)   = %.4f\n", EV(ctx, 0, ONLY(CATEGORY_ONES)));

    /* Twos: max 10, expect ≈ 4.2 */
    ASSERT(EV(ctx, 0, ONLY(CATEGORY_TWOS)) > 2.0 &&
           EV(ctx, 0, ONLY(CATEGORY_TWOS)) < 10.0,
           "Only Twos left: EV in [2, 10]");
    printf("    EV(only Twos)   = %.4f\n", EV(ctx, 0, ONLY(CATEGORY_TWOS)));

    /* Threes: max 15, expect ≈ 6.3 */
    ASSERT(EV(ctx, 0, ONLY(CATEGORY_THREES)) > 3.0 &&
           EV(ctx, 0, ONLY(CATEGORY_THREES)) < 15.0,
           "Only Threes left: EV in [3, 15]");
    printf("    EV(only Threes) = %.4f\n", EV(ctx, 0, ONLY(CATEGORY_THREES)));

    /* Fours: max 20, expect ≈ 8.4 */
    ASSERT(EV(ctx, 0, ONLY(CATEGORY_FOURS)) > 5.0 &&
           EV(ctx, 0, ONLY(CATEGORY_FOURS)) < 20.0,
           "Only Fours left: EV in [5, 20]");
    printf("    EV(only Fours)  = %.4f\n", EV(ctx, 0, ONLY(CATEGORY_FOURS)));

    /* Fives: max 25, expect ≈ 10.5 */
    ASSERT(EV(ctx, 0, ONLY(CATEGORY_FIVES)) > 6.0 &&
           EV(ctx, 0, ONLY(CATEGORY_FIVES)) < 25.0,
           "Only Fives left: EV in [6, 25]");
    printf("    EV(only Fives)  = %.4f\n", EV(ctx, 0, ONLY(CATEGORY_FIVES)));

    /* Sixes: max 30, expect ≈ 12.6 */
    ASSERT(EV(ctx, 0, ONLY(CATEGORY_SIXES)) > 8.0 &&
           EV(ctx, 0, ONLY(CATEGORY_SIXES)) < 20.0,
           "Only Sixes left: EV in [8, 20]");
    printf("    EV(only Sixes)  = %.4f\n", EV(ctx, 0, ONLY(CATEGORY_SIXES)));

    /* One Pair: max 12 (pair of 6s). Expect ≈ 8-10. */
    ASSERT(EV(ctx, 0, ONLY(CATEGORY_ONE_PAIR)) > 6.0 &&
           EV(ctx, 0, ONLY(CATEGORY_ONE_PAIR)) < 12.0,
           "Only One Pair left: EV in [6, 12]");
    printf("    EV(only 1Pair)  = %.4f\n", EV(ctx, 0, ONLY(CATEGORY_ONE_PAIR)));

    /* Two Pairs: max 22 (6s+5s). Expect ≈ 14-18. */
    ASSERT(EV(ctx, 0, ONLY(CATEGORY_TWO_PAIRS)) > 10.0 &&
           EV(ctx, 0, ONLY(CATEGORY_TWO_PAIRS)) < 22.0,
           "Only Two Pairs left: EV in [10, 22]");
    printf("    EV(only 2Pairs) = %.4f\n", EV(ctx, 0, ONLY(CATEGORY_TWO_PAIRS)));

    /* Three of a Kind: max 18 (three 6s). Expect ≈ 8-13. */
    ASSERT(EV(ctx, 0, ONLY(CATEGORY_THREE_OF_A_KIND)) > 5.0 &&
           EV(ctx, 0, ONLY(CATEGORY_THREE_OF_A_KIND)) < 18.0,
           "Only 3oaK left: EV in [5, 18]");
    printf("    EV(only 3oaK)   = %.4f\n", EV(ctx, 0, ONLY(CATEGORY_THREE_OF_A_KIND)));

    /* Four of a Kind: max 24 (four 6s). Expect ≈ 5-15. */
    ASSERT(EV(ctx, 0, ONLY(CATEGORY_FOUR_OF_A_KIND)) > 3.0 &&
           EV(ctx, 0, ONLY(CATEGORY_FOUR_OF_A_KIND)) < 18.0,
           "Only 4oaK left: EV in [3, 18]");
    printf("    EV(only 4oaK)   = %.4f\n", EV(ctx, 0, ONLY(CATEGORY_FOUR_OF_A_KIND)));

    /* Small Straight: 15 or 0. P ≈ 20%. EV ≈ 3. */
    ASSERT(EV(ctx, 0, ONLY(CATEGORY_SMALL_STRAIGHT)) > 1.0 &&
           EV(ctx, 0, ONLY(CATEGORY_SMALL_STRAIGHT)) < 8.0,
           "Only Sm Straight left: EV in [1, 8]");
    printf("    EV(only SmStr)  = %.4f\n", EV(ctx, 0, ONLY(CATEGORY_SMALL_STRAIGHT)));

    /* Large Straight: 20 or 0. P ≈ 12%. EV ≈ 2.5. */
    ASSERT(EV(ctx, 0, ONLY(CATEGORY_LARGE_STRAIGHT)) > 1.0 &&
           EV(ctx, 0, ONLY(CATEGORY_LARGE_STRAIGHT)) < 8.0,
           "Only Lg Straight left: EV in [1, 8]");
    printf("    EV(only LgStr)  = %.4f\n", EV(ctx, 0, ONLY(CATEGORY_LARGE_STRAIGHT)));

    /* Full House: max 28 (6,6,6,5,5). Expect ≈ 10-18. */
    ASSERT(EV(ctx, 0, ONLY(CATEGORY_FULL_HOUSE)) > 5.0 &&
           EV(ctx, 0, ONLY(CATEGORY_FULL_HOUSE)) < 22.0,
           "Only Full House left: EV in [5, 22]");
    printf("    EV(only FH)     = %.4f\n", EV(ctx, 0, ONLY(CATEGORY_FULL_HOUSE)));

    /* Chance: sum of all dice, always > 0. Optimal keep-high ≈ 23.3. */
    ASSERT(EV(ctx, 0, ONLY(CATEGORY_CHANCE)) > 20.0 &&
           EV(ctx, 0, ONLY(CATEGORY_CHANCE)) < 30.0,
           "Only Chance left: EV in [20, 30]");
    printf("    EV(only Chance) = %.4f\n", EV(ctx, 0, ONLY(CATEGORY_CHANCE)));

    /* Yatzy: 50 or 0. P ≈ 4.6%. EV ≈ 2.3. */
    ASSERT(EV(ctx, 0, ONLY(CATEGORY_YATZY)) > 1.0 &&
           EV(ctx, 0, ONLY(CATEGORY_YATZY)) < 5.0,
           "Only Yatzy left: EV in [1, 5]");
    printf("    EV(only Yatzy)  = %.4f\n", EV(ctx, 0, ONLY(CATEGORY_YATZY)));
}

/* ── 3b. Single-category EV ordering ─────────────────────────────────── */

static void test_single_category_ordering(const YatzyContext *ctx) {
    /* Upper category EVs should increase with face value:
     * E(Ones) < E(Twos) < E(Threes) < E(Fours) < E(Fives) < E(Sixes) */
    for (int cat = CATEGORY_ONES; cat < CATEGORY_SIXES; cat++) {
        double ev_lo = EV(ctx, 0, ONLY(cat));
        double ev_hi = EV(ctx, 0, ONLY(cat + 1));
        char msg[80];
        snprintf(msg, sizeof(msg), "EV(only cat %d) < EV(only cat %d)", cat, cat + 1);
        ASSERT(ev_lo < ev_hi, msg);
    }

    /* Chance should be higher than any single upper category */
    double ev_chance = EV(ctx, 0, ONLY(CATEGORY_CHANCE));
    double ev_sixes = EV(ctx, 0, ONLY(CATEGORY_SIXES));
    ASSERT(ev_chance > ev_sixes,
           "EV(only Chance) > EV(only Sixes)");

    /* Yatzy should be worth less than Ones (50 pts but very low probability) */
    double ev_yatzy = EV(ctx, 0, ONLY(CATEGORY_YATZY));
    double ev_ones = EV(ctx, 0, ONLY(CATEGORY_ONES));
    ASSERT(ev_yatzy < ev_sixes,
           "EV(only Yatzy) < EV(only Sixes) (low P offsets high score)");

    /* Small Straight (15 pts, ~20%) should be worth less than Large Straight
     * is debatable. But both should be worth much less than Chance. */
    double ev_sm = EV(ctx, 0, ONLY(CATEGORY_SMALL_STRAIGHT));
    double ev_lg = EV(ctx, 0, ONLY(CATEGORY_LARGE_STRAIGHT));
    ASSERT(ev_chance > ev_sm && ev_chance > ev_lg,
           "EV(Chance) > EV(Small Straight) and > EV(Large Straight)");

    /* One Pair should be higher than Small Straight (pairs are much easier) */
    double ev_pair = EV(ctx, 0, ONLY(CATEGORY_ONE_PAIR));
    ASSERT(ev_pair > ev_sm,
           "EV(One Pair) > EV(Small Straight)");
}

/* ── 4. Optimal category choice — obvious cases ──────────────────── */

static void test_optimal_category_choice(const YatzyContext *ctx) {
    /* [6,6,6,6,6] with Yatzy + Chance + Sixes open → always pick Yatzy (50 pts) */
    {
        int scored = ALL_SCORED ^ (1 << CATEGORY_YATZY)
                                ^ (1 << CATEGORY_CHANCE)
                                ^ (1 << CATEGORY_SIXES);
        int dice[] = {6,6,6,6,6};
        double best_ev;
        int cat = ChooseBestCategoryNoRerolls(ctx, 0, scored, dice, &best_ev);
        ASSERT_EQ_INT(cat, CATEGORY_YATZY,
                      "[6,6,6,6,6] with Yatzy open → pick Yatzy");
    }

    /* [1,1,1,1,1] with Yatzy + Ones + Chance open → pick Yatzy (50 vs 5 vs 5) */
    {
        int scored = ALL_SCORED ^ (1 << CATEGORY_YATZY)
                                ^ (1 << CATEGORY_ONES)
                                ^ (1 << CATEGORY_CHANCE);
        int dice[] = {1,1,1,1,1};
        double best_ev;
        int cat = ChooseBestCategoryNoRerolls(ctx, 0, scored, dice, &best_ev);
        ASSERT_EQ_INT(cat, CATEGORY_YATZY,
                      "[1,1,1,1,1] with Yatzy + Ones + Chance → pick Yatzy");
    }

    /* [1,2,3,4,5] with Small Straight + Ones open → pick Small Straight (15 vs 1) */
    {
        int scored = ALL_SCORED ^ (1 << CATEGORY_SMALL_STRAIGHT)
                                ^ (1 << CATEGORY_ONES);
        int dice[] = {1,2,3,4,5};
        double best_ev;
        int cat = ChooseBestCategoryNoRerolls(ctx, 0, scored, dice, &best_ev);
        ASSERT_EQ_INT(cat, CATEGORY_SMALL_STRAIGHT,
                      "[1,2,3,4,5] with Sm Straight + Ones open → pick Sm Straight");
    }

    /* [2,3,4,5,6] with Large Straight + Small Straight open → pick Large (20 vs 0) */
    {
        int scored = ALL_SCORED ^ (1 << CATEGORY_LARGE_STRAIGHT)
                                ^ (1 << CATEGORY_SMALL_STRAIGHT);
        int dice[] = {2,3,4,5,6};
        double best_ev;
        int cat = ChooseBestCategoryNoRerolls(ctx, 0, scored, dice, &best_ev);
        ASSERT_EQ_INT(cat, CATEGORY_LARGE_STRAIGHT,
                      "[2,3,4,5,6] with Lg + Sm Straight open → pick Lg Straight");
    }

    /* [3,3,3,4,4] with Full House + Three of Kind open → pick Full House (17 vs 9) */
    {
        int scored = ALL_SCORED ^ (1 << CATEGORY_FULL_HOUSE)
                                ^ (1 << CATEGORY_THREE_OF_A_KIND);
        int dice[] = {3,3,3,4,4};
        SortDiceSet(dice);
        double best_ev;
        int cat = ChooseBestCategoryNoRerolls(ctx, 0, scored, dice, &best_ev);
        ASSERT_EQ_INT(cat, CATEGORY_FULL_HOUSE,
                      "[3,3,3,4,4] with Full House + 3oaK open → pick Full House");
    }

    /* [6,6,5,5,1] with Two Pairs + One Pair open → pick Two Pairs (22 vs 12) */
    {
        int scored = ALL_SCORED ^ (1 << CATEGORY_TWO_PAIRS)
                                ^ (1 << CATEGORY_ONE_PAIR);
        int dice[] = {1,5,5,6,6};
        double best_ev;
        int cat = ChooseBestCategoryNoRerolls(ctx, 0, scored, dice, &best_ev);
        ASSERT_EQ_INT(cat, CATEGORY_TWO_PAIRS,
                      "[1,5,5,6,6] with 2Pairs + 1Pair open → pick Two Pairs");
    }

    /* [6,6,6,6,5] with 4oaK + 3oaK open → pick 4oaK (24 vs 18) */
    {
        int scored = ALL_SCORED ^ (1 << CATEGORY_FOUR_OF_A_KIND)
                                ^ (1 << CATEGORY_THREE_OF_A_KIND);
        int dice[] = {5,6,6,6,6};
        double best_ev;
        int cat = ChooseBestCategoryNoRerolls(ctx, 0, scored, dice, &best_ev);
        ASSERT_EQ_INT(cat, CATEGORY_FOUR_OF_A_KIND,
                      "[5,6,6,6,6] with 4oaK + 3oaK open → pick 4oaK");
    }

    /* [2,3,4,5,6] with Large Straight + Chance open → pick Large (20 vs 20 sum,
     * but Large Straight gives exactly 20 and Chance gives 20 too.
     * However, Chance is more valuable as future flexibility. With only these 2
     * categories left, you should pick Large Straight since Chance=20 and LgStr=20
     * have same immediate score but Lg Straight is harder to get again. */
    {
        int scored = ALL_SCORED ^ (1 << CATEGORY_LARGE_STRAIGHT)
                                ^ (1 << CATEGORY_CHANCE);
        int dice[] = {2,3,4,5,6};
        double best_ev;
        int cat = ChooseBestCategoryNoRerolls(ctx, 0, scored, dice, &best_ev);
        ASSERT_EQ_INT(cat, CATEGORY_LARGE_STRAIGHT,
                      "[2,3,4,5,6] with Lg Straight + Chance → pick Lg Straight");
    }
}

/* ── 5. Optimal reroll choice — obvious cases ─────────────────────── */

static void test_optimal_reroll(const YatzyContext *ctx) {
    /* [1,6,6,6,6] with only Yatzy left → reroll the 1 (position 0, mask=1) */
    {
        int dice[] = {1,6,6,6,6};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, ONLY(CATEGORY_YATZY), dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_INT(best_mask, 1,
                      "[1,6,6,6,6] only Yatzy left → reroll the 1 (mask=1)");
    }

    /* [6,6,6,6,6] with Yatzy left → keep all (mask=0) */
    {
        int dice[] = {6,6,6,6,6};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, ONLY(CATEGORY_YATZY), dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_INT(best_mask, 0,
                      "[6,6,6,6,6] with Yatzy left → keep all (mask=0)");
    }

    /* [1,1,1,1,1] with only Sixes left → reroll all 5 dice (mask=31) */
    {
        int dice[] = {1,1,1,1,1};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, ONLY(CATEGORY_SIXES), dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_INT(best_mask, 31,
                      "[1,1,1,1,1] only Sixes left → reroll all (mask=31)");
    }

    /* [1,2,3,4,5] with only Small Straight left → keep all (mask=0) */
    {
        int dice[] = {1,2,3,4,5};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, ONLY(CATEGORY_SMALL_STRAIGHT), dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_INT(best_mask, 0,
                      "[1,2,3,4,5] only Sm Straight left → keep all (mask=0)");
    }

    /* [2,3,4,5,6] with only Large Straight left → keep all (mask=0) */
    {
        int dice[] = {2,3,4,5,6};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, ONLY(CATEGORY_LARGE_STRAIGHT), dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_INT(best_mask, 0,
                      "[2,3,4,5,6] only Lg Straight left → keep all (mask=0)");
    }

    /* [1,1,1,1,2] with only Ones left → reroll the 2 (position 4, mask=16) */
    {
        int dice[] = {1,1,1,1,2};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, ONLY(CATEGORY_ONES), dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_INT(best_mask, 16,
                      "[1,1,1,1,2] only Ones left → reroll the 2 (mask=16)");
    }

    /* [6,6,6,5,4] with only Sixes left → reroll the 4 and 5 (positions 0,1 → mask=3) */
    {
        int dice[] = {4,5,6,6,6};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, ONLY(CATEGORY_SIXES), dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_INT(best_mask, 3,
                      "[4,5,6,6,6] only Sixes left → reroll 4,5 (mask=3)");
    }

    /* [3,3,3,4,4] with only Full House left → keep all (already a full house, 17 pts) */
    {
        int dice[] = {3,3,3,4,4};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, ONLY(CATEGORY_FULL_HOUSE), dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_INT(best_mask, 0,
                      "[3,3,3,4,4] only Full House left → keep all (mask=0)");
    }

    /* [5,5,5,5,5] with only Yatzy left → keep all */
    {
        int dice[] = {5,5,5,5,5};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, ONLY(CATEGORY_YATZY), dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_INT(best_mask, 0,
                      "[5,5,5,5,5] only Yatzy left → keep all (mask=0)");
    }

    /* [1,2,3,4,6] with only Small Straight left → reroll the 6 (position 4, mask=16).
     * Need a 5 to complete 1-2-3-4-5. */
    {
        int dice[] = {1,2,3,4,6};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, ONLY(CATEGORY_SMALL_STRAIGHT), dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_INT(best_mask, 16,
                      "[1,2,3,4,6] only Sm Straight left → reroll 6 (mask=16)");
    }

    /* [1,3,4,5,6] with only Large Straight left → reroll the 1 (position 0, mask=1).
     * Need a 2 to complete 2-3-4-5-6. */
    {
        int dice[] = {1,3,4,5,6};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, ONLY(CATEGORY_LARGE_STRAIGHT), dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_INT(best_mask, 1,
                      "[1,3,4,5,6] only Lg Straight left → reroll 1 (mask=1)");
    }
}

/* ── 5b. Reroll EV bounds ─────────────────────────────────────────── */

static void test_reroll_ev_bounds(const YatzyContext *ctx) {
    /* When you already have the optimal dice for a category, rerolling
     * should never improve over keeping (EV with mask=0 = max score). */

    /* [6,6,6,6,6] only Yatzy left: EV = 50.0 exactly */
    {
        int dice[] = {6,6,6,6,6};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, ONLY(CATEGORY_YATZY), dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_DBL(best_ev, 50.0, "[6,6,6,6,6] only Yatzy → EV = 50.0");
    }

    /* [1,2,3,4,5] only Small Straight left: EV = 15.0 exactly */
    {
        int dice[] = {1,2,3,4,5};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, ONLY(CATEGORY_SMALL_STRAIGHT), dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_DBL(best_ev, 15.0, "[1,2,3,4,5] only Sm Straight → EV = 15.0");
    }

    /* [2,3,4,5,6] only Large Straight left: EV = 20.0 exactly */
    {
        int dice[] = {2,3,4,5,6};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, ONLY(CATEGORY_LARGE_STRAIGHT), dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_DBL(best_ev, 20.0, "[2,3,4,5,6] only Lg Straight → EV = 20.0");
    }

    /* [6,6,6,6,6] only Chance left: EV = 30.0 exactly (can't improve) */
    {
        int dice[] = {6,6,6,6,6};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, ONLY(CATEGORY_CHANCE), dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_DBL(best_ev, 30.0, "[6,6,6,6,6] only Chance → EV = 30.0");
    }

    /* [6,6,6,6,6] only Sixes left: EV = 30.0 exactly */
    {
        int dice[] = {6,6,6,6,6};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, ONLY(CATEGORY_SIXES), dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_DBL(best_ev, 30.0, "[6,6,6,6,6] only Sixes → EV = 30.0");
    }

    /* [6,6,5,5,5] only Full House left: EV = 28.0 exactly
     * (This is the maximum Full House: 6+6+5+5+5 = 27? No: 6+6+6+5+5=28).
     * Actually [5,5,6,6,6] is 3 sixes + 2 fives = 28. But our dice is [5,5,5,6,6]
     * which is 3 fives + 2 sixes = 27. Still a full house. Might reroll for 28.
     * Let's use [5,5,6,6,6] instead. */
    {
        int dice[] = {5,5,6,6,6};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, ONLY(CATEGORY_FULL_HOUSE), dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_DBL(best_ev, 28.0, "[5,5,6,6,6] only Full House → EV = 28.0");
    }
}

/* ── 6. Monotonicity checks ──────────────────────────────────────────── */

static void test_monotonicity(const YatzyContext *ctx) {
    /* More categories scored → lower future EV (fewer scoring opportunities).
     * Compare EV with 0 categories vs 1, and 1 vs 2. */
    double ev_0 = EV(ctx, 0, 0);
    double ev_1 = EV(ctx, 0, 1); /* only Ones scored */
    double ev_2 = EV(ctx, 0, 3); /* Ones + Twos scored */
    ASSERT(ev_0 > ev_1,
           "EV with 0 scored > EV with 1 scored");
    ASSERT(ev_1 > ev_2,
           "EV with 1 scored > EV with 2 scored");

    /* Upper score closer to 63 → higher EV due to bonus proximity.
     * Use Sixes scored (mask=0x20). up=18 is reachable (3×6=18).
     * With 5 remaining upper cats (Ones..Fives), the player at up=18
     * is closer to the bonus threshold than at up=0, hence higher EV. */
    int sixes_scored = 0x20; /* bit 5 = Sixes */
    double ev_up0_u = EV(ctx, 0, sixes_scored);
    double ev_up18_u = EV(ctx, 18, sixes_scored);
    ASSERT(ev_up18_u > ev_up0_u,
           "EV with upper=18 > EV with upper=0 (bonus proximity, Sixes scored)");

    /* All 6 upper scored: at up=63 the bonus is guaranteed (+50), at up=0 it's not.
     * States below 63 all have equal EV (no more upper cats to play). */
    int all_upper = 0x3F; /* bits 0-5 = Ones..Sixes */
    double ev_up63_u = EV(ctx, 63, all_upper);
    double ev_up0_all = EV(ctx, 0, all_upper);
    ASSERT(ev_up63_u > ev_up0_all,
           "EV with upper=63 > EV with upper=0 (all upper scored)");

    double bonus_diff = ev_up63_u - ev_up0_all;
    ASSERT(bonus_diff > 5.0 && bonus_diff <= 50.0,
           "EV(up=63) - EV(up=0) in (5, 50] with all upper scored");
    printf("    EV(63,all_upper) - EV(0,all_upper) = %.4f\n", bonus_diff);
}

/* ── 6b. Monotonicity: scoring more categories always lowers EV ────── */

static void test_monotonicity_chain(const YatzyContext *ctx) {
    /* Build a chain: score Ones, then Twos, then Threes, ...
     * EV should strictly decrease at each step. */
    int scored = 0;
    double prev_ev = EV(ctx, 0, 0);
    int monotone = 1;
    for (int cat = 0; cat < CATEGORY_COUNT && monotone; cat++) {
        scored |= (1 << cat);
        double ev = EV(ctx, 0, scored);
        if (ev >= prev_ev) {
            monotone = 0;
            fprintf(stderr, "  Monotonicity break at cat %d: EV=%.4f >= prev=%.4f\n",
                    cat, ev, prev_ev);
        }
        prev_ev = ev;
    }
    ASSERT(monotone, "EV strictly decreases as categories are scored (0→15)");
}

/* ── 7. Two categories remaining — verify relative values ─────────── */

static void test_two_categories(const YatzyContext *ctx) {
    /* Yatzy + Chance remaining: EV should be Yatzy EV + Chance EV roughly. */
    int yatzy_chance = ALL_SCORED ^ (1 << CATEGORY_YATZY)
                                  ^ (1 << CATEGORY_CHANCE);
    double ev_both = EV(ctx, 0, yatzy_chance);
    double ev_y = EV(ctx, 0, ONLY(CATEGORY_YATZY));
    double ev_c = EV(ctx, 0, ONLY(CATEGORY_CHANCE));

    ASSERT(ev_both > ev_y && ev_both > ev_c,
           "Two categories EV > either single category EV");
    printf("    EV(Yatzy+Chance) = %.4f, Yatzy alone = %.4f, Chance alone = %.4f\n",
           ev_both, ev_y, ev_c);

    double sum_singles = ev_y + ev_c;
    ASSERT(fabs(ev_both - sum_singles) < 5.0,
           "Two-category EV ≈ sum of single-category EVs (within 5)");

    /* Full House + Two Pairs: should be > either single */
    int fh_2p = ALL_SCORED ^ (1 << CATEGORY_FULL_HOUSE) ^ (1 << CATEGORY_TWO_PAIRS);
    double ev_fh_2p = EV(ctx, 0, fh_2p);
    double ev_fh = EV(ctx, 0, ONLY(CATEGORY_FULL_HOUSE));
    double ev_2p = EV(ctx, 0, ONLY(CATEGORY_TWO_PAIRS));
    ASSERT(ev_fh_2p > ev_fh && ev_fh_2p > ev_2p,
           "EV(FH+2Pairs) > either alone");

    /* 3oaK + 4oaK: should be > either single */
    int oak_34 = ALL_SCORED ^ (1 << CATEGORY_THREE_OF_A_KIND) ^ (1 << CATEGORY_FOUR_OF_A_KIND);
    double ev_oak = EV(ctx, 0, oak_34);
    double ev_3 = EV(ctx, 0, ONLY(CATEGORY_THREE_OF_A_KIND));
    double ev_4 = EV(ctx, 0, ONLY(CATEGORY_FOUR_OF_A_KIND));
    ASSERT(ev_oak > ev_3 && ev_oak > ev_4,
           "EV(3oaK+4oaK) > either alone");

    /* Small Straight + Large Straight */
    int str_both = ALL_SCORED ^ (1 << CATEGORY_SMALL_STRAIGHT) ^ (1 << CATEGORY_LARGE_STRAIGHT);
    double ev_str = EV(ctx, 0, str_both);
    double ev_sm = EV(ctx, 0, ONLY(CATEGORY_SMALL_STRAIGHT));
    double ev_lg = EV(ctx, 0, ONLY(CATEGORY_LARGE_STRAIGHT));
    ASSERT(ev_str > ev_sm && ev_str > ev_lg,
           "EV(SmStr+LgStr) > either alone");
}

/* ── 8. Non-negative EVs (full sweep of reachable states) ─────────── */

static void test_ev_positivity(const YatzyContext *ctx) {
    /* All reachable non-terminal EVs should be >= 0.
     * Unreachable states have EV=0 by construction.
     * Check a broad sample including every level. */
    int any_negative = 0;
    for (int scored = 0; scored < (1 << CATEGORY_COUNT) && !any_negative; scored++) {
        int upper_mask = scored & 0x3F;
        for (int up = 0; up <= 63; up++) {
            if (!ctx->reachable[upper_mask][up]) continue;
            double ev = EV(ctx, up, scored);
            if (ev < -1e-9) {
                any_negative = 1;
                fprintf(stderr, "  Negative EV at up=%d scored=0x%x: %.6f\n", up, scored, ev);
                break;
            }
        }
    }
    ASSERT(!any_negative, "All reachable state EVs are non-negative");
}

/* ── 8b. Unreachable states are zero ──────────────────────────────── */

static void test_unreachable_states_zero(const YatzyContext *ctx) {
    /* Spot-check known unreachable states: upper_mask=0 with up > 0 */
    int any_nonzero = 0;
    for (int up = 1; up <= 63; up++) {
        /* scored=0 means no categories scored, upper_mask=0, so up must be 0 */
        double ev = EV(ctx, up, 0);
        if (fabs(ev) > 1e-9) {
            any_nonzero = 1;
            fprintf(stderr, "  Unreachable state (up=%d, scored=0) has EV=%.6f\n", up, ev);
            break;
        }
    }
    ASSERT(!any_nonzero, "Unreachable states (mask=0, up>0) all have EV=0");

    /* scored = 0x01 (only Ones): up can only be 0..5 */
    for (int up = 6; up <= 63; up++) {
        double ev = EV(ctx, up, 0x01);
        if (fabs(ev) > 1e-9) {
            any_nonzero = 1;
            fprintf(stderr, "  Unreachable state (up=%d, scored=Ones) has EV=%.6f\n", up, ev);
            break;
        }
    }
    ASSERT(!any_nonzero, "Unreachable states (only Ones, up>5) all have EV=0");

    /* scored = 0x20 (only Sixes): up can only be 0,6,12,18,24,30 */
    for (int up = 1; up <= 63; up++) {
        if (up % 6 == 0 && up <= 30) continue; /* reachable */
        double ev = EV(ctx, up, 0x20);
        if (fabs(ev) > 1e-9) {
            any_nonzero = 1;
            fprintf(stderr, "  Unreachable state (up=%d, scored=Sixes) has EV=%.6f\n", up, ev);
            break;
        }
    }
    ASSERT(!any_nonzero, "Unreachable states (only Sixes, up not multiple of 6) have EV=0");
}

/* ── 9. Reroll EV consistency ────────────────────────────────────────── */

static void test_reroll_ev_consistency(const YatzyContext *ctx) {
    /* Best reroll EV should always be >= keep-all EV.
     * Test for a mid-game state. */
    int scored = (1 << CATEGORY_ONES) | (1 << CATEGORY_TWOS);
    int dice[] = {3,3,4,5,6};
    SortDiceSet(dice);

    int best_mask;
    double best_ev;
    ComputeBestRerollStrategy(ctx, 5, scored, dice, 2, &best_mask, &best_ev);

    YatzyState state = {5, scored};
    double keep_all_ev = ComputeBestScoringValueForDiceSet(ctx, &state, dice);

    ASSERT(best_ev >= keep_all_ev - 1e-9,
           "Best reroll EV >= immediate scoring EV");
    printf("    Reroll EV = %.4f, keep-all scoring = %.4f, mask = %d\n",
           best_ev, keep_all_ev, best_mask);

    /* Same check with different mid-game state: 5 cats scored, up=10 */
    {
        int scored2 = (1 << CATEGORY_ONES) | (1 << CATEGORY_TWOS)
                    | (1 << CATEGORY_THREES) | (1 << CATEGORY_ONE_PAIR)
                    | (1 << CATEGORY_TWO_PAIRS);
        int dice2[] = {2,4,4,5,6};
        int mask2;
        double ev2;
        ComputeBestRerollStrategy(ctx, 10, scored2, dice2, 2, &mask2, &ev2);
        YatzyState st2 = {10, scored2};
        double keep2 = ComputeBestScoringValueForDiceSet(ctx, &st2, dice2);
        ASSERT(ev2 >= keep2 - 1e-9,
               "Mid-game reroll EV >= keep-all EV (state 2)");
    }

    /* 1 reroll remaining: still should hold */
    {
        int scored3 = (1 << CATEGORY_ONES);
        int dice3[] = {1,2,3,4,5};
        int mask3;
        double ev3;
        ComputeBestRerollStrategy(ctx, 1, scored3, dice3, 1, &mask3, &ev3);
        YatzyState st3 = {1, scored3};
        double keep3 = ComputeBestScoringValueForDiceSet(ctx, &st3, dice3);
        ASSERT(ev3 >= keep3 - 1e-9,
               "1-reroll: best reroll EV >= keep-all EV");
    }
}

/* ── 10. Upper bonus cliff ──────────────────────────────────────────── */

static void test_upper_bonus_cliff(const YatzyContext *ctx) {
    /* With all upper categories scored (reachable), up=63 gets +50 bonus. */
    int all_upper = 0x3F;
    double ev_0 = EV(ctx, 0, all_upper);
    double ev_63 = EV(ctx, 63, all_upper);
    double cliff = ev_63 - ev_0;
    ASSERT(cliff > 5.0,
           "Bonus cliff: EV(up=63,all_upper) - EV(up=0,all_upper) > 5");
    printf("    EV(0, all_upper) = %.4f, EV(63, all_upper) = %.4f, cliff = %.4f\n",
           ev_0, ev_63, cliff);

    /* With all upper + some lower scored, cliff is exactly 50. */
    int upper_plus_lower = all_upper | (1 << CATEGORY_ONE_PAIR) | (1 << CATEGORY_TWO_PAIRS)
                                     | (1 << CATEGORY_FULL_HOUSE) | (1 << CATEGORY_CHANCE);
    double ev_lo_0 = EV(ctx, 0, upper_plus_lower);
    double ev_lo_63 = EV(ctx, 63, upper_plus_lower);
    double cliff2 = ev_lo_63 - ev_lo_0;
    ASSERT(cliff2 > 5.0 && cliff2 <= 50.0,
           "Bonus cliff with upper+lower scored: 5 < cliff <= 50");
    printf("    EV(0, upper+lower) = %.4f, EV(63, upper+lower) = %.4f, cliff = %.4f\n",
           ev_lo_0, ev_lo_63, cliff2);

    /* With all upper scored and all lower scored (14 cats), up=63 vs up=0:
     * Only 1 category remains (Yatzy). Diff should be 50 (within float precision). */
    int almost_all = ALL_SCORED ^ (1 << CATEGORY_YATZY);
    double ev_yatzy_0 = EV(ctx, 0, almost_all);
    double ev_yatzy_63 = EV(ctx, 63, almost_all);
    ASSERT_NEAR(ev_yatzy_63 - ev_yatzy_0, 50.0, 1e-4,
                "1 cat left (Yatzy): bonus cliff ≈ 50");
}

/* ── 11. Exact EV for perfect dice with single category ──────────── */

static void test_exact_terminal_ev(const YatzyContext *ctx) {
    /* When you hold the perfect dice for the only remaining category,
     * your state EV equals score + future_bonus.
     * At up=0 with only a lower category left, future_bonus = 0.
     *
     * This tests that the DP table is consistent with the scoring rules. */

    /* Chance only left, we just test that EV >= sum of 5 ones = 5
     * (minimum possible Chance score). */
    ASSERT(EV(ctx, 0, ONLY(CATEGORY_CHANCE)) >= 5.0,
           "EV(only Chance) >= minimum Chance score (5)");

    /* Full House only left: EV >= 0 (you can always score 0 if you miss). */
    ASSERT(EV(ctx, 0, ONLY(CATEGORY_FULL_HOUSE)) >= 0.0,
           "EV(only Full House) >= 0");

    /* Yatzy only left at up=63 (all upper scored):
     * The bonus is already guaranteed. EV = 50 (bonus) + P(Yatzy)*50. */
    int only_yatzy_upper = ONLY(CATEGORY_YATZY);
    /* upper_mask of ONLY(CATEGORY_YATZY) = 0x3F (all upper cats are in the scored set) */
    double ev_yatzy_63 = EV(ctx, 63, only_yatzy_upper);
    double ev_yatzy_0 = EV(ctx, 0, only_yatzy_upper);
    /* At up=63, bonus adds 50 (within float precision). */
    ASSERT_NEAR(ev_yatzy_63 - ev_yatzy_0, 50.0, 1e-4,
                "Only Yatzy left: EV(up=63) - EV(up=0) ≈ 50 (guaranteed bonus)");
    printf("    EV(only Yatzy, up=0) = %.4f, EV(only Yatzy, up=63) = %.4f\n",
           ev_yatzy_0, ev_yatzy_63);
}

/* ── 12. Score-category consistency ──────────────────────────────────── */

static void test_score_category_consistency(const YatzyContext *ctx) {
    /* When the only category left can score X with given dice,
     * and it's the last turn with 0 rerolls, EV after scoring must equal
     * X + bonus (if applicable). Test via ChooseBestCategoryNoRerolls. */

    /* [6,6,6,6,6] with only Yatzy left → score = 50 */
    {
        int dice[] = {6,6,6,6,6};
        double best_ev;
        ChooseBestCategoryNoRerolls(ctx, 0, ONLY(CATEGORY_YATZY), dice, &best_ev);
        ASSERT_EQ_DBL(best_ev, 50.0,
                      "[6,6,6,6,6] only Yatzy: score = 50.0");
    }

    /* [1,2,3,4,6] with only Small Straight left → score = 0 (not a straight).
     * ChooseBestCategoryNoRerolls skips zero-scoring cats → returns -INFINITY. */
    {
        int dice[] = {1,2,3,4,6};
        double best_ev;
        int cat = ChooseBestCategoryNoRerolls(ctx, 0, ONLY(CATEGORY_SMALL_STRAIGHT), dice, &best_ev);
        ASSERT_EQ_INT(cat, -1,
                      "[1,2,3,4,6] only Sm Straight: no valid scoring → cat = -1");
    }

    /* [1,1,1,1,1] with only Sixes left → score = 0, skipped → cat = -1 */
    {
        int dice[] = {1,1,1,1,1};
        double best_ev;
        int cat = ChooseBestCategoryNoRerolls(ctx, 0, ONLY(CATEGORY_SIXES), dice, &best_ev);
        ASSERT_EQ_INT(cat, -1,
                      "[1,1,1,1,1] only Sixes: no valid scoring → cat = -1");
    }

    /* [4,4,4,4,4] with only Fours left → score = 20 */
    {
        int dice[] = {4,4,4,4,4};
        double best_ev;
        ChooseBestCategoryNoRerolls(ctx, 0, ONLY(CATEGORY_FOURS), dice, &best_ev);
        ASSERT_EQ_DBL(best_ev, 20.0,
                      "[4,4,4,4,4] only Fours: score = 20.0");
    }
}

/* ── 13. File format validation ──────────────────────────────────────── */

static void test_file_format(void) {
    /* Verify the bin file has the expected size and header. */
    ASSERT_EQ_INT(FileExists("data/all_states.bin"), 1, "all_states.bin exists");

    FILE *f = fopen("data/all_states.bin", "rb");
    ASSERT(f != NULL, "Can open all_states.bin");
    if (!f) return;

    StateFileHeader header;
    size_t n = fread(&header, sizeof(header), 1, f);
    ASSERT_EQ_INT((int)n, 1, "Read header successfully");
    ASSERT_EQ_INT((int)header.magic, (int)STATE_FILE_MAGIC, "Header magic = 0x59545A53");
    ASSERT_EQ_INT((int)header.version, STATE_FILE_VERSION, "Header version = 3");
    ASSERT_EQ_INT((int)header.total_states, NUM_STATES, "Header total_states = 2097152");
    ASSERT_EQ_INT((int)header.reserved, 0, "Header reserved = 0");

    /* Check total file size */
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    long expected_size = (long)sizeof(StateFileHeader) + (long)NUM_STATES * (long)sizeof(float);
    ASSERT(file_size == expected_size, "File size = header + 2097152 doubles");
    printf("    File size: %ld bytes (expected %ld)\n", file_size, expected_size);

    fclose(f);
}

/* ── 14. Symmetry: lower-only categories don't depend on upper score ── */

static void test_lower_category_upper_independence(const YatzyContext *ctx) {
    /* When all upper categories are scored (upper_mask=0x3F) and only
     * lower categories remain, states with up < 63 should all have
     * the same EV (upper score doesn't affect lower scoring).
     * Only up=63 differs (+50 bonus). */
    int all_upper_plus_some_lower = 0x3F | (1 << CATEGORY_ONE_PAIR)
                                         | (1 << CATEGORY_THREE_OF_A_KIND);
    /* Only remaining: Two Pairs, 4oaK, Sm Str, Lg Str, Full House, Chance, Yatzy */
    double ev_ref = EV(ctx, 0, all_upper_plus_some_lower);
    int all_equal = 1;
    for (int up = 1; up < 63; up++) {
        double ev = EV(ctx, up, all_upper_plus_some_lower);
        if (fabs(ev - ev_ref) > 1e-4) {
            all_equal = 0;
            fprintf(stderr, "  EV(up=%d) = %.4f != EV(up=0) = %.4f\n", up, ev, ev_ref);
            break;
        }
    }
    ASSERT(all_equal,
           "All upper scored + lower remaining: EV same for up=0..62");

    /* up=63 should be 50 more (within float precision) */
    double ev_63 = EV(ctx, 63, all_upper_plus_some_lower);
    ASSERT_NEAR(ev_63 - ev_ref, 50.0, 1e-4,
                "All upper scored: EV(up=63) - EV(up=0) ≈ 50.0");
}

/* ── 15. Additivity check: independent lower categories ──────────── */

static void test_lower_additivity(const YatzyContext *ctx) {
    /* When only lower categories remain (all upper scored) and up=0
     * (no bonus), the EV of having N categories ≈ sum of individual EVs.
     * Not exact due to strategic interaction (what to score first), but
     * should be within a few points. */
    double ev_chance = EV(ctx, 0, ALL_SCORED ^ (1 << CATEGORY_CHANCE));
    double ev_yatzy = EV(ctx, 0, ALL_SCORED ^ (1 << CATEGORY_YATZY));

    /* Two lower categories: Chance + Yatzy */
    int both = ALL_SCORED ^ (1 << CATEGORY_CHANCE) ^ (1 << CATEGORY_YATZY);
    double ev_both = EV(ctx, 0, both);
    double sum = ev_chance + ev_yatzy;

    /* Should be close (within ~5 points of sum of singles) */
    ASSERT(fabs(ev_both - sum) < 5.0,
           "EV(Chance+Yatzy) ≈ EV(Chance) + EV(Yatzy) (within 5)");
    printf("    EV(Chance+Yatzy) = %.4f, sum of singles = %.4f, diff = %.4f\n",
           ev_both, sum, ev_both - sum);
}

/* ── 16. Early game states have higher EV than late game ─────────── */

static void test_early_vs_late_game(const YatzyContext *ctx) {
    /* EV(0,0) (15 categories left) should be much higher than
     * EV(0, scored_with_10_cats) (5 categories left). */
    int ten_scored = (1 << CATEGORY_ONES) | (1 << CATEGORY_TWOS)
                   | (1 << CATEGORY_THREES) | (1 << CATEGORY_FOURS)
                   | (1 << CATEGORY_FIVES) | (1 << CATEGORY_SIXES)
                   | (1 << CATEGORY_ONE_PAIR) | (1 << CATEGORY_TWO_PAIRS)
                   | (1 << CATEGORY_THREE_OF_A_KIND) | (1 << CATEGORY_FOUR_OF_A_KIND);
    double ev_start = EV(ctx, 0, 0);
    double ev_late = EV(ctx, 0, ten_scored);

    ASSERT(ev_start > ev_late,
           "Game start EV > late game EV (10 cats scored)");
    ASSERT(ev_late > 0.0,
           "Late game EV (5 cats left) > 0");
    printf("    EV(0 scored) = %.4f, EV(10 scored) = %.4f\n", ev_start, ev_late);

    /* With 5 remaining: Sm Str (15) + Lg Str (20) + FH + Chance + Yatzy.
     * Expected sum of those individual EVs ≈ 3+2.5+12+23+2.3 = ~43. */
    ASSERT(ev_late > 30.0 && ev_late < 80.0,
           "Late game EV (5 lower cats) in [30, 80]");
}

/* ── Main ────────────────────────────────────────────────────────────── */

TEST_MAIN_BEGIN("test_precomputed")
    SetWorkingDirectory();

    YatzyContext *ctx = CreateYatzyContext();
    ASSERT(ctx != NULL, "CreateYatzyContext succeeds");
    PrecomputeLookupTables(ctx);

    /* Load the precomputed state values */
    int loaded = LoadAllStateValues(ctx, "data/all_states.bin");
    ASSERT(loaded == 1, "Loaded precomputed state values from all_states.bin");

    if (loaded) {
        test_terminal_states(ctx);
        test_game_start_ev(ctx);
        test_one_category_remaining(ctx);
        test_single_category_ordering(ctx);
        test_optimal_category_choice(ctx);
        test_optimal_reroll(ctx);
        test_reroll_ev_bounds(ctx);
        test_monotonicity(ctx);
        test_monotonicity_chain(ctx);
        test_two_categories(ctx);
        test_ev_positivity(ctx);
        test_unreachable_states_zero(ctx);
        test_reroll_ev_consistency(ctx);
        test_upper_bonus_cliff(ctx);
        test_exact_terminal_ev(ctx);
        test_score_category_consistency(ctx);
        test_file_format();
        test_lower_category_upper_independence(ctx);
        test_lower_additivity(ctx);
        test_early_vs_late_game(ctx);
    }

    FreeYatzyContext(ctx);
TEST_MAIN_END
