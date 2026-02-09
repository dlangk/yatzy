/*
 * test_context.c — Basic invariant tests for context creation and lookup tables.
 *
 * Verifies:
 *   - Context allocation succeeds
 *   - PrecomputeLookupTables produces correct known values
 *   - InitializeFinalStates sets the bonus correctly
 *   - Known category scores are correct
 */
#include "test_helpers.h"
#include "context.h"
#include "game_mechanics.h"

TEST_MAIN_BEGIN("test_context")

    /* --- Context creation --- */
    YatzyContext *ctx = CreateYatzyContext();
    ASSERT(ctx != NULL, "CreateYatzyContext returns non-NULL");
    ASSERT(ctx->state_values != NULL, "state_values allocated");
    ASSERT_EQ_INT(ctx->num_combinations, 0, "num_combinations is 0 before precomputation");

    /* --- Precompute lookup tables --- */
    PrecomputeLookupTables(ctx);

    /* Factorials */
    ASSERT_EQ_INT(ctx->factorial[0], 1, "factorial[0] == 1");
    ASSERT_EQ_INT(ctx->factorial[1], 1, "factorial[1] == 1");
    ASSERT_EQ_INT(ctx->factorial[5], 120, "factorial[5] == 120");

    /* Dice combinations: C(6+5-1, 5) = 252 */
    ASSERT_EQ_INT(ctx->num_combinations, 252, "num_combinations == 252");

    /* First dice set should be [1,1,1,1,1] (all ones) */
    ASSERT_EQ_INT(ctx->all_dice_sets[0][0], 1, "first dice set starts with 1");
    ASSERT_EQ_INT(ctx->all_dice_sets[0][4], 1, "first dice set ends with 1");

    /* Last dice set should be [6,6,6,6,6] */
    ASSERT_EQ_INT(ctx->all_dice_sets[251][0], 6, "last dice set starts with 6");
    ASSERT_EQ_INT(ctx->all_dice_sets[251][4], 6, "last dice set ends with 6");

    /* Category scores for [1,1,1,1,1] (index 0) */
    ASSERT_EQ_INT(ctx->precomputed_scores[0][CATEGORY_ONES], 5, "[1,1,1,1,1] scores 5 for Ones");
    ASSERT_EQ_INT(ctx->precomputed_scores[0][CATEGORY_TWOS], 0, "[1,1,1,1,1] scores 0 for Twos");
    ASSERT_EQ_INT(ctx->precomputed_scores[0][CATEGORY_YATZY], 50, "[1,1,1,1,1] scores 50 for Yatzy");

    /* Category scores for [6,6,6,6,6] (index 251) */
    ASSERT_EQ_INT(ctx->precomputed_scores[251][CATEGORY_SIXES], 30, "[6,6,6,6,6] scores 30 for Sixes");
    ASSERT_EQ_INT(ctx->precomputed_scores[251][CATEGORY_YATZY], 50, "[6,6,6,6,6] scores 50 for Yatzy");

    /* Scored category counts */
    ASSERT_EQ_INT(ctx->scored_category_count_cache[0], 0, "0 categories scored in bitmask 0");
    ASSERT_EQ_INT(ctx->scored_category_count_cache[(1 << CATEGORY_COUNT) - 1], 15, "15 categories scored in full bitmask");
    ASSERT_EQ_INT(ctx->scored_category_count_cache[1], 1, "1 category scored in bitmask 1");

    /* Dice set probabilities should sum to 1.0 */
    double prob_sum = 0.0;
    for (int i = 0; i < 252; i++) {
        prob_sum += ctx->dice_set_probabilities[i];
    }
    ASSERT(fabs(prob_sum - 1.0) < 1e-9, "dice_set_probabilities sum to 1.0");

    /* --- Final states (bonus check) --- */
    int all_scored = (1 << CATEGORY_COUNT) - 1;
    /* Upper score >= 63 gets 50 point bonus */
    ASSERT_EQ_DBL(GetStateValue(ctx, 63, all_scored), 50.0, "final state with upper=63 gets 50 bonus");
    /* Upper score < 63 gets no bonus */
    ASSERT_EQ_DBL(GetStateValue(ctx, 0, all_scored), 0.0, "final state with upper=0 gets no bonus");
    ASSERT_EQ_DBL(GetStateValue(ctx, 62, all_scored), 0.0, "final state with upper=62 gets no bonus");

    /* --- Cleanup --- */
    FreeYatzyContext(ctx);

TEST_MAIN_END
