/*
 * test_dice_mechanics.c — Dice operations: sort, index, count, probability
 *
 * Requires context for index lookup and factorial tables (Phase 0).
 */
#include "test_helpers.h"
#include "context.h"
#include "dice_mechanics.h"

static void test_sort_dice_set(YatzyContext *ctx) {
    (void)ctx;

    /* Unsorted → sorted */
    int d1[] = {5,3,1,4,2};
    SortDiceSet(d1);
    ASSERT(d1[0]==1 && d1[1]==2 && d1[2]==3 && d1[3]==4 && d1[4]==5,
           "SortDiceSet [5,3,1,4,2] → [1,2,3,4,5]");

    /* Already sorted stays sorted */
    int d2[] = {1,2,3,4,5};
    SortDiceSet(d2);
    ASSERT(d2[0]==1 && d2[1]==2 && d2[2]==3 && d2[3]==4 && d2[4]==5,
           "SortDiceSet [1,2,3,4,5] unchanged");

    /* Reverse order */
    int d3[] = {6,5,4,3,2};
    SortDiceSet(d3);
    ASSERT(d3[0]==2 && d3[1]==3 && d3[2]==4 && d3[3]==5 && d3[4]==6,
           "SortDiceSet [6,5,4,3,2] → [2,3,4,5,6]");

    /* All same */
    int d4[] = {3,3,3,3,3};
    SortDiceSet(d4);
    ASSERT(d4[0]==3 && d4[1]==3 && d4[2]==3 && d4[3]==3 && d4[4]==3,
           "SortDiceSet [3,3,3,3,3] unchanged");
}

static void test_find_dice_set_index(YatzyContext *ctx) {
    /* [1,1,1,1,1] should be index 0 */
    int d1[] = {1,1,1,1,1};
    ASSERT_EQ_INT(FindDiceSetIndex(ctx, d1), 0,
                  "FindDiceSetIndex [1,1,1,1,1] = 0");

    /* [6,6,6,6,6] should be index 251 */
    int d2[] = {6,6,6,6,6};
    ASSERT_EQ_INT(FindDiceSetIndex(ctx, d2), 251,
                  "FindDiceSetIndex [6,6,6,6,6] = 251");

    /* Round-trip: for all 252 sets, FindDiceSetIndex(all_dice_sets[i]) == i */
    int round_trip_ok = 1;
    for (int i = 0; i < 252; i++) {
        if (FindDiceSetIndex(ctx, ctx->all_dice_sets[i]) != i) {
            round_trip_ok = 0;
            break;
        }
    }
    ASSERT(round_trip_ok, "Index round-trip: FindDiceSetIndex(all_dice_sets[i]) == i for all 252");
}

static void test_count_faces(YatzyContext *ctx) {
    (void)ctx;

    int d1[] = {1,1,2,3,3};
    int fc[7];
    CountFaces(d1, fc);
    ASSERT_EQ_INT(fc[1], 2, "CountFaces [1,1,2,3,3]: face 1 count = 2");
    ASSERT_EQ_INT(fc[2], 1, "CountFaces [1,1,2,3,3]: face 2 count = 1");
    ASSERT_EQ_INT(fc[3], 2, "CountFaces [1,1,2,3,3]: face 3 count = 2");
    ASSERT_EQ_INT(fc[4], 0, "CountFaces [1,1,2,3,3]: face 4 count = 0");
    ASSERT_EQ_INT(fc[5], 0, "CountFaces [1,1,2,3,3]: face 5 count = 0");
    ASSERT_EQ_INT(fc[6], 0, "CountFaces [1,1,2,3,3]: face 6 count = 0");

    int d2[] = {6,6,6,6,6};
    CountFaces(d2, fc);
    ASSERT_EQ_INT(fc[6], 5, "CountFaces [6,6,6,6,6]: face 6 count = 5");
    ASSERT_EQ_INT(fc[1], 0, "CountFaces [6,6,6,6,6]: face 1 count = 0");
}

static void test_probability(YatzyContext *ctx) {
    /* [1,1,1,1,1]: only 1 way → 1/7776 */
    int d1[] = {1,1,1,1,1};
    double p1 = ComputeProbabilityOfDiceSet(ctx, d1);
    ASSERT_NEAR(p1, 1.0/7776.0, 1e-12,
                "P([1,1,1,1,1]) = 1/7776");

    /* [1,1,1,1,2]: 5 ways (5 positions for the 2) → 5/7776 */
    int d2[] = {1,1,1,1,2};
    double p2 = ComputeProbabilityOfDiceSet(ctx, d2);
    ASSERT_NEAR(p2, 5.0/7776.0, 1e-12,
                "P([1,1,1,1,2]) = 5/7776");

    /* All 252 probabilities sum to 1.0 */
    double sum = 0.0;
    for (int i = 0; i < 252; i++) {
        sum += ctx->dice_set_probabilities[i];
    }
    ASSERT_NEAR(sum, 1.0, 1e-9,
                "All 252 dice set probabilities sum to 1.0");
}

TEST_MAIN_BEGIN("test_dice_mechanics")
    YatzyContext *ctx = CreateYatzyContext();
    ASSERT(ctx != NULL, "CreateYatzyContext succeeds");
    PrecomputeLookupTables(ctx);

    test_sort_dice_set(ctx);
    test_find_dice_set_index(ctx);
    test_count_faces(ctx);
    test_probability(ctx);

    FreeYatzyContext(ctx);
TEST_MAIN_END
