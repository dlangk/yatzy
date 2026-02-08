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
    /* Scandinavian Yatzy optimal EV is approximately 250-270.
     * Literature value (Woodward 2003, adjusted for Scandi rules) ~ 254-260. */
    ASSERT(ev > 230.0 && ev < 290.0,
           "Game start EV(0,0) is in plausible range [230, 290]");
    printf("    EV(0,0) = %.4f\n", ev);
}

/* ── 3. Single category remaining — hand-calculable EVs ──────────── */

static void test_one_category_remaining(const YatzyContext *ctx) {
    /* Only Yatzy left: EV = P(rolling Yatzy in 3 rolls) * 50
     * With 2 rerolls: P ≈ 4.6%. EV ≈ 2.3 */
    int only_yatzy = ALL_SCORED ^ (1 << CATEGORY_YATZY);
    double ev_yatzy = EV(ctx, 0, only_yatzy);
    ASSERT(ev_yatzy > 1.0 && ev_yatzy < 5.0,
           "Only Yatzy left: EV in [1, 5] (probability-weighted 50 pts)");
    printf("    EV(only Yatzy left) = %.4f\n", ev_yatzy);

    /* Only Chance left: EV = expected sum of 5 dice with optimal keep strategy.
     * Always keep high dice. Expected ≈ 23-25. */
    int only_chance = ALL_SCORED ^ (1 << CATEGORY_CHANCE);
    double ev_chance = EV(ctx, 0, only_chance);
    ASSERT(ev_chance > 20.0 && ev_chance < 30.0,
           "Only Chance left: EV in [20, 30]");
    printf("    EV(only Chance left) = %.4f\n", ev_chance);

    /* Only Sixes left: EV = expected sixes score with optimal strategy.
     * Max possible = 30, but each die has 1/6 chance of being a 6.
     * Optimal keep strategy yields ≈ 12-13. */
    int only_sixes = ALL_SCORED ^ (1 << CATEGORY_SIXES);
    double ev_sixes = EV(ctx, 0, only_sixes);
    ASSERT(ev_sixes > 8.0 && ev_sixes < 20.0,
           "Only Sixes left: EV in [8, 20]");
    printf("    EV(only Sixes left) = %.4f\n", ev_sixes);

    /* Only Ones left: each die scores only 1 point when it's a one.
     * With optimal rerolls, expected ≈ 2 (keep ones, reroll rest). */
    int only_ones = ALL_SCORED ^ (1 << CATEGORY_ONES);
    double ev_ones = EV(ctx, 0, only_ones);
    ASSERT(ev_ones > 1.0 && ev_ones < 5.0,
           "Only Ones left: EV in [1, 5]");
    printf("    EV(only Ones left) = %.4f\n", ev_ones);

    /* Only Small Straight left: 15 points or 0.
     * P(getting small straight in 3 rolls) ≈ 20%. EV ≈ 3. */
    int only_sm_str = ALL_SCORED ^ (1 << CATEGORY_SMALL_STRAIGHT);
    double ev_sm = EV(ctx, 0, only_sm_str);
    ASSERT(ev_sm > 1.0 && ev_sm < 8.0,
           "Only Small Straight left: EV in [1, 8]");
    printf("    EV(only Small Straight left) = %.4f\n", ev_sm);
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
}

/* ── 5. Optimal reroll choice — obvious cases ─────────────────────── */

static void test_optimal_reroll(const YatzyContext *ctx) {
    /* [6,6,6,6,1] with only Yatzy left → reroll the 1 (mask bit 0 = die 1).
     * mask = 0b00001 = 1 means reroll die at position 0.
     * But dice are sorted [1,6,6,6,6], so the 1 is at position 0.
     * mask=1 means reroll position 0 (the 1). */
    {
        int only_yatzy = ALL_SCORED ^ (1 << CATEGORY_YATZY);
        int dice[] = {1,6,6,6,6}; /* already sorted */
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, only_yatzy, dice, 2, &best_mask, &best_ev);
        /* Should reroll the 1 (position 0), keeping the four 6s.
         * mask=1 = 0b00001: reroll die 0 */
        ASSERT_EQ_INT(best_mask, 1,
                      "[1,6,6,6,6] only Yatzy left → reroll the 1 (mask=1)");
    }

    /* [6,6,6,6,6] with Yatzy left → keep all (mask=0) */
    {
        int only_yatzy = ALL_SCORED ^ (1 << CATEGORY_YATZY);
        int dice[] = {6,6,6,6,6};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, only_yatzy, dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_INT(best_mask, 0,
                      "[6,6,6,6,6] with Yatzy left → keep all (mask=0)");
    }

    /* [1,1,1,1,1] with only Sixes left → reroll all 5 dice (mask=31=0b11111) */
    {
        int only_sixes = ALL_SCORED ^ (1 << CATEGORY_SIXES);
        int dice[] = {1,1,1,1,1};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, only_sixes, dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_INT(best_mask, 31,
                      "[1,1,1,1,1] only Sixes left → reroll all (mask=31)");
    }

    /* [1,2,3,4,5] with only Small Straight left → keep all (mask=0) */
    {
        int only_sm = ALL_SCORED ^ (1 << CATEGORY_SMALL_STRAIGHT);
        int dice[] = {1,2,3,4,5};
        int best_mask;
        double best_ev;
        ComputeBestRerollStrategy(ctx, 0, only_sm, dice, 2, &best_mask, &best_ev);
        ASSERT_EQ_INT(best_mask, 0,
                      "[1,2,3,4,5] only Sm Straight left → keep all (mask=0)");
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
     * Compare upper=0 vs upper=50 with same scored set. */
    double ev_up0 = EV(ctx, 0, 0);
    double ev_up50 = EV(ctx, 50, 0);
    ASSERT(ev_up50 > ev_up0,
           "EV with upper=50 > EV with upper=0 (bonus proximity)");

    /* At the bonus threshold exactly (63), even more valuable */
    double ev_up63 = EV(ctx, 63, 0);
    ASSERT(ev_up63 > ev_up50,
           "EV with upper=63 > EV with upper=50");

    /* Upper score 63 with nothing scored means bonus is guaranteed (50 pts).
     * But EV(0,0) already includes the *probability* of reaching 63,
     * so the gap is less than 50 — it's 50 minus the expected bonus
     * contribution already baked into EV(0,0). Typically 10-20. */
    double bonus_diff = ev_up63 - ev_up0;
    ASSERT(bonus_diff > 5.0 && bonus_diff < 50.0,
           "EV(up=63) - EV(up=0) in [5, 50] (partial bonus already priced in)");
    printf("    EV(63,0) - EV(0,0) = %.4f\n", bonus_diff);
}

/* ── 7. Two categories remaining — verify relative values ─────────── */

static void test_two_categories(const YatzyContext *ctx) {
    /* Yatzy + Chance remaining: EV should be Yatzy EV + Chance EV roughly.
     * (Not exact due to interaction, but should be close.) */
    int yatzy_chance = ALL_SCORED ^ (1 << CATEGORY_YATZY)
                                  ^ (1 << CATEGORY_CHANCE);
    double ev_both = EV(ctx, 0, yatzy_chance);
    int only_yatzy = ALL_SCORED ^ (1 << CATEGORY_YATZY);
    int only_chance = ALL_SCORED ^ (1 << CATEGORY_CHANCE);
    double ev_y = EV(ctx, 0, only_yatzy);
    double ev_c = EV(ctx, 0, only_chance);

    /* With two turns, you get to score both. Should be > max(single). */
    ASSERT(ev_both > ev_y && ev_both > ev_c,
           "Two categories EV > either single category EV");
    printf("    EV(Yatzy+Chance) = %.4f, Yatzy alone = %.4f, Chance alone = %.4f\n",
           ev_both, ev_y, ev_c);

    /* The two-category EV should be reasonably close to the sum of singles
     * (slight interaction effects, but within ~5 points). */
    double sum_singles = ev_y + ev_c;
    ASSERT(fabs(ev_both - sum_singles) < 5.0,
           "Two-category EV ≈ sum of single-category EVs (within 5)");
}

/* ── 8. Symmetry: same scored count, different sets ──────────────────── */

static void test_ev_positivity(const YatzyContext *ctx) {
    /* All non-terminal EVs should be >= 0.
     * (You can always score 0 in a category, so future value is never negative.) */
    int any_negative = 0;
    for (int up = 0; up <= 63; up++) {
        /* Spot check: empty state */
        if (EV(ctx, up, 0) < -1e-9) { any_negative = 1; break; }
        /* Spot check: one category scored */
        if (EV(ctx, up, 1) < -1e-9) { any_negative = 1; break; }
    }
    ASSERT(!any_negative,
           "All spot-checked EVs are non-negative");
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

    /* Also compute the "keep all" EV: just score the dice as-is after 0 rerolls.
     * The reroll strategy EV should be >= the best immediate scoring. */
    YatzyState state = {5, scored};
    double keep_all_ev = ComputeBestScoringValueForDiceSet(ctx, &state, dice);

    ASSERT(best_ev >= keep_all_ev - 1e-9,
           "Best reroll EV >= immediate scoring EV");
    printf("    Reroll EV = %.4f, keep-all scoring = %.4f, mask = %d\n",
           best_ev, keep_all_ev, best_mask);
}

/* ── 10. Upper bonus cliff ──────────────────────────────────────────── */

static void test_upper_bonus_cliff(const YatzyContext *ctx) {
    /* Test with no categories scored: upper=0 vs upper=63.
     * At upper=63 the bonus is guaranteed. The EV difference should be large. */
    double ev_0 = EV(ctx, 0, 0);
    double ev_63 = EV(ctx, 63, 0);
    double cliff = ev_63 - ev_0;
    ASSERT(cliff > 5.0,
           "Bonus cliff: EV(up=63,0) - EV(up=0,0) > 5");
    printf("    EV(0, 0) = %.4f, EV(63, 0) = %.4f, cliff = %.4f\n",
           ev_0, ev_63, cliff);

    /* With only lower categories scored (no upper cats used), upper still = 0.
     * Compare upper=0 vs upper=63 with same lower-only scored set.
     * Since no upper cats are scored, there are still 6 upper cats to play.
     * The player at upper=63 has guaranteed bonus = +50 over the player
     * at upper=0 who hasn't yet scored any upper cats. */
    int lower_only = (1 << CATEGORY_ONE_PAIR) | (1 << CATEGORY_TWO_PAIRS)
                   | (1 << CATEGORY_FULL_HOUSE) | (1 << CATEGORY_CHANCE);
    double ev_lo_0 = EV(ctx, 0, lower_only);
    double ev_lo_63 = EV(ctx, 63, lower_only);
    double cliff2 = ev_lo_63 - ev_lo_0;
    /* With 6 upper cats remaining, the player at 0 has some chance of reaching
     * 63 naturally, so the gap is less than 50 but should still be large. */
    ASSERT(cliff2 > 10.0 && cliff2 < 50.0,
           "Bonus cliff with lower-only scored: 10 < cliff < 50");
    printf("    EV(0, lower_only) = %.4f, EV(63, lower_only) = %.4f, cliff = %.4f\n",
           ev_lo_0, ev_lo_63, cliff2);
}

/* ── Main ────────────────────────────────────────────────────────────── */

TEST_MAIN_BEGIN("test_precomputed")
    SetWorkingDirectory();

    YatzyContext *ctx = CreateYatzyContext();
    ASSERT(ctx != NULL, "CreateYatzyContext succeeds");
    PrecomputeLookupTables(ctx);

    /* Load the precomputed state values */
    int loaded = LoadAllStateValuesMmap(ctx, "data/all_states.bin");
    ASSERT(loaded == 1, "Loaded precomputed state values from all_states.bin");

    if (loaded) {
        test_terminal_states(ctx);
        test_game_start_ev(ctx);
        test_one_category_remaining(ctx);
        test_optimal_category_choice(ctx);
        test_optimal_reroll(ctx);
        test_monotonicity(ctx);
        test_two_categories(ctx);
        test_ev_positivity(ctx);
        test_reroll_ev_consistency(ctx);
        test_upper_bonus_cliff(ctx);
    }

    FreeYatzyContext(ctx);
TEST_MAIN_END
