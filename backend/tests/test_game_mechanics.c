/*
 * test_game_mechanics.c — Scoring rules for all 15 Scandinavian Yatzy categories
 *
 * Tests CalculateCategoryScore() and UpdateUpperScore(). No context needed.
 */
#include "test_helpers.h"
#include "game_mechanics.h"
#include "context.h"

static void test_upper_section(void) {
    /* Ones: [1,1,1,1,1] → 5 */
    int d1[] = {1,1,1,1,1};
    ASSERT_EQ_INT(CalculateCategoryScore(d1, CATEGORY_ONES), 5,
                  "[1,1,1,1,1] Ones = 5");

    /* Sixes: [6,6,6,6,6] → 30 */
    int d2[] = {6,6,6,6,6};
    ASSERT_EQ_INT(CalculateCategoryScore(d2, CATEGORY_SIXES), 30,
                  "[6,6,6,6,6] Sixes = 30");

    /* Ones: [1,2,3,4,5] → 1 */
    int d3[] = {1,2,3,4,5};
    ASSERT_EQ_INT(CalculateCategoryScore(d3, CATEGORY_ONES), 1,
                  "[1,2,3,4,5] Ones = 1");

    /* Threes: [3,3,4,5,6] → 6 */
    int d4[] = {3,3,4,5,6};
    ASSERT_EQ_INT(CalculateCategoryScore(d4, CATEGORY_THREES), 6,
                  "[3,3,4,5,6] Threes = 6");

    /* Twos: [1,2,3,4,5] → 2 */
    ASSERT_EQ_INT(CalculateCategoryScore(d3, CATEGORY_TWOS), 2,
                  "[1,2,3,4,5] Twos = 2");

    /* Fours: [4,4,4,4,4] → 20 */
    int d5[] = {4,4,4,4,4};
    ASSERT_EQ_INT(CalculateCategoryScore(d5, CATEGORY_FOURS), 20,
                  "[4,4,4,4,4] Fours = 20");

    /* Fives: [5,5,5,1,2] → 15 */
    int d6[] = {5,5,5,1,2};
    ASSERT_EQ_INT(CalculateCategoryScore(d6, CATEGORY_FIVES), 15,
                  "[5,5,5,1,2] Fives = 15");
}

static void test_one_pair(void) {
    /* [3,3,4,5,6] → pair of 3s = 6 */
    int d1[] = {3,3,4,5,6};
    ASSERT_EQ_INT(CalculateCategoryScore(d1, CATEGORY_ONE_PAIR), 6,
                  "[3,3,4,5,6] One Pair = 6");

    /* [6,6,5,5,1] → highest pair is 6s = 12 */
    int d2[] = {6,6,5,5,1};
    ASSERT_EQ_INT(CalculateCategoryScore(d2, CATEGORY_ONE_PAIR), 12,
                  "[6,6,5,5,1] One Pair = 12 (highest pair)");

    /* [1,2,3,4,6] → no pair = 0 */
    int d3[] = {1,2,3,4,6};
    ASSERT_EQ_INT(CalculateCategoryScore(d3, CATEGORY_ONE_PAIR), 0,
                  "[1,2,3,4,6] One Pair = 0");
}

static void test_two_pairs(void) {
    /* [3,3,5,5,6] → 2×5 + 2×3 = 16 */
    int d1[] = {3,3,5,5,6};
    ASSERT_EQ_INT(CalculateCategoryScore(d1, CATEGORY_TWO_PAIRS), 16,
                  "[3,3,5,5,6] Two Pairs = 16");

    /* [1,1,2,3,4] → only one pair = 0 */
    int d2[] = {1,1,2,3,4};
    ASSERT_EQ_INT(CalculateCategoryScore(d2, CATEGORY_TWO_PAIRS), 0,
                  "[1,1,2,3,4] Two Pairs = 0 (only one pair)");
}

static void test_n_of_a_kind(void) {
    /* Three of a Kind: [2,2,2,4,5] → 3×2 = 6 */
    int d1[] = {2,2,2,4,5};
    ASSERT_EQ_INT(CalculateCategoryScore(d1, CATEGORY_THREE_OF_A_KIND), 6,
                  "[2,2,2,4,5] Three of a Kind = 6");

    /* Four of a Kind: [4,4,4,4,2] → 4×4 = 16 */
    int d2[] = {4,4,4,4,2};
    ASSERT_EQ_INT(CalculateCategoryScore(d2, CATEGORY_FOUR_OF_A_KIND), 16,
                  "[4,4,4,4,2] Four of a Kind = 16");

    /* Three of a Kind: [1,2,3,4,5] → 0 */
    int d3[] = {1,2,3,4,5};
    ASSERT_EQ_INT(CalculateCategoryScore(d3, CATEGORY_THREE_OF_A_KIND), 0,
                  "[1,2,3,4,5] Three of a Kind = 0");

    /* Four of a Kind: [3,3,3,4,5] → 0 (only three) */
    int d4[] = {3,3,3,4,5};
    ASSERT_EQ_INT(CalculateCategoryScore(d4, CATEGORY_FOUR_OF_A_KIND), 0,
                  "[3,3,3,4,5] Four of a Kind = 0");
}

static void test_straights(void) {
    /* Small Straight: [1,2,3,4,5] → 15 */
    int d1[] = {1,2,3,4,5};
    ASSERT_EQ_INT(CalculateCategoryScore(d1, CATEGORY_SMALL_STRAIGHT), 15,
                  "[1,2,3,4,5] Small Straight = 15");

    /* Large Straight: [2,3,4,5,6] → 20 */
    int d2[] = {2,3,4,5,6};
    ASSERT_EQ_INT(CalculateCategoryScore(d2, CATEGORY_LARGE_STRAIGHT), 20,
                  "[2,3,4,5,6] Large Straight = 20");

    /* Small Straight: [1,2,3,4,6] → 0 (not 1-5) */
    int d3[] = {1,2,3,4,6};
    ASSERT_EQ_INT(CalculateCategoryScore(d3, CATEGORY_SMALL_STRAIGHT), 0,
                  "[1,2,3,4,6] Small Straight = 0");

    /* Large Straight: [1,2,3,4,5] → 0 (not 2-6) */
    ASSERT_EQ_INT(CalculateCategoryScore(d1, CATEGORY_LARGE_STRAIGHT), 0,
                  "[1,2,3,4,5] Large Straight = 0");

    /* Small Straight: [2,3,4,5,6] → 0 (that's large) */
    ASSERT_EQ_INT(CalculateCategoryScore(d2, CATEGORY_SMALL_STRAIGHT), 0,
                  "[2,3,4,5,6] Small Straight = 0");
}

static void test_full_house(void) {
    /* [2,2,3,3,3] → 2+2+3+3+3 = 13 */
    int d1[] = {2,2,3,3,3};
    ASSERT_EQ_INT(CalculateCategoryScore(d1, CATEGORY_FULL_HOUSE), 13,
                  "[2,2,3,3,3] Full House = 13");

    /* [1,2,3,4,6] → 0 (no FH) */
    int d2[] = {1,2,3,4,6};
    ASSERT_EQ_INT(CalculateCategoryScore(d2, CATEGORY_FULL_HOUSE), 0,
                  "[1,2,3,4,6] Full House = 0");

    /* [5,5,5,5,5] → 0 (five of a kind is not FH: need exactly 3+2) */
    int d3[] = {5,5,5,5,5};
    ASSERT_EQ_INT(CalculateCategoryScore(d3, CATEGORY_FULL_HOUSE), 0,
                  "[5,5,5,5,5] Full House = 0 (five of a kind)");
}

static void test_chance(void) {
    /* [3,4,1,5,6] → 19 */
    int d1[] = {3,4,1,5,6};
    ASSERT_EQ_INT(CalculateCategoryScore(d1, CATEGORY_CHANCE), 19,
                  "[3,4,1,5,6] Chance = 19");

    /* [1,1,1,1,1] → 5 */
    int d2[] = {1,1,1,1,1};
    ASSERT_EQ_INT(CalculateCategoryScore(d2, CATEGORY_CHANCE), 5,
                  "[1,1,1,1,1] Chance = 5");
}

static void test_yatzy(void) {
    /* [6,6,6,6,6] → 50 */
    int d1[] = {6,6,6,6,6};
    ASSERT_EQ_INT(CalculateCategoryScore(d1, CATEGORY_YATZY), 50,
                  "[6,6,6,6,6] Yatzy = 50");

    /* [6,6,6,6,5] → 0 */
    int d2[] = {6,6,6,6,5};
    ASSERT_EQ_INT(CalculateCategoryScore(d2, CATEGORY_YATZY), 0,
                  "[6,6,6,6,5] Yatzy = 0");

    /* [1,1,1,1,1] → 50 */
    int d3[] = {1,1,1,1,1};
    ASSERT_EQ_INT(CalculateCategoryScore(d3, CATEGORY_YATZY), 50,
                  "[1,1,1,1,1] Yatzy = 50");
}

static void test_update_upper_score(void) {
    /* Upper category adds score, capped at 63 */
    ASSERT_EQ_INT(UpdateUpperScore(0, CATEGORY_ONES, 5), 5,
                  "UpdateUpperScore(0, Ones, 5) = 5");

    ASSERT_EQ_INT(UpdateUpperScore(10, CATEGORY_SIXES, 30), 40,
                  "UpdateUpperScore(10, Sixes, 30) = 40");

    /* Cap at 63 */
    ASSERT_EQ_INT(UpdateUpperScore(60, CATEGORY_FIVES, 30), 63,
                  "UpdateUpperScore(60, Fives, 30) = 63 (capped)");

    /* Lower category leaves upper unchanged */
    ASSERT_EQ_INT(UpdateUpperScore(10, CATEGORY_ONE_PAIR, 12), 10,
                  "UpdateUpperScore(10, One Pair, 12) = 10 (unchanged)");

    ASSERT_EQ_INT(UpdateUpperScore(50, CATEGORY_YATZY, 50), 50,
                  "UpdateUpperScore(50, Yatzy, 50) = 50 (unchanged)");

    ASSERT_EQ_INT(UpdateUpperScore(63, CATEGORY_ONES, 5), 63,
                  "UpdateUpperScore(63, Ones, 5) = 63 (already capped)");
}

TEST_MAIN_BEGIN("test_game_mechanics")
    test_upper_section();
    test_one_pair();
    test_two_pairs();
    test_n_of_a_kind();
    test_straights();
    test_full_house();
    test_chance();
    test_yatzy();
    test_update_upper_score();
TEST_MAIN_END
