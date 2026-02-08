/*
 * test_storage.c — Round-trip I/O tests for v3 consolidated state files
 *
 * Creates a context, sets known state values, saves/loads, and compares.
 */
#include <string.h>
#include <unistd.h>
#include "test_helpers.h"
#include "context.h"
#include "storage.h"

#define TEST_FILE "/tmp/yatzy_test_all_states_v3.bin"

static void cleanup_files(void) {
    unlink(TEST_FILE);
}

static void test_file_exists(void) {
    ASSERT_EQ_INT(FileExists("/tmp"), 1, "FileExists('/tmp') = 1");
    ASSERT_EQ_INT(FileExists("/tmp/nonexistent_yatzy_test_xyz"), 0,
                  "FileExists(nonexistent) = 0");
}

/*
 * V3 round-trip: save all states, load via zero-copy mmap, compare.
 */
static void test_v3_round_trip(void) {
    YatzyContext *ctx1 = CreateYatzyContext();
    ASSERT(ctx1 != NULL, "V3: create ctx1");
    PrecomputeLookupTables(ctx1);

    /* Set some known values: terminal states + a few others */
    int all_scored = (1 << CATEGORY_COUNT) - 1;
    for (int up = 0; up <= 63; up++) {
        ctx1->state_values[STATE_INDEX(up, all_scored)] = (up >= 63) ? 50.0f : 0.0f;
    }
    /* Set recognizable values (float-exact) */
    ctx1->state_values[STATE_INDEX(0, 0)] = 123.5f;
    ctx1->state_values[STATE_INDEX(63, 1)] = 789.0f;

    /* Save via v3 format */
    SaveAllStateValues(ctx1, TEST_FILE);
    ASSERT_EQ_INT(FileExists(TEST_FILE), 1, "V3 file created");

    /* Load into fresh context */
    YatzyContext *ctx2 = CreateYatzyContext();
    ASSERT(ctx2 != NULL, "V3: create ctx2");

    int loaded = LoadAllStateValues(ctx2, TEST_FILE);
    ASSERT_EQ_INT(loaded, 1, "V3: load succeeds");

    /* Compare all states — float precision */
    int match = 1;
    for (int i = 0; i < NUM_STATES; i++) {
        if (fabsf(ctx1->state_values[i] - ctx2->state_values[i]) > 1e-6f) {
            match = 0;
            fprintf(stderr, "  V3 mismatch at index %d: %.6f != %.6f\n",
                    i, ctx1->state_values[i], ctx2->state_values[i]);
            break;
        }
    }
    ASSERT(match, "V3 round-trip: all state values match");

    /* Spot-check the recognizable values */
    ASSERT_NEAR(ctx2->state_values[STATE_INDEX(0, 0)], 123.5, 1e-6,
                "V3: state (0,0) = 123.5");
    ASSERT_NEAR(ctx2->state_values[STATE_INDEX(63, 1)], 789.0, 1e-6,
                "V3: state (63,1) = 789.0");
    ASSERT_EQ_DBL(ctx2->state_values[STATE_INDEX(63, all_scored)], 50.0,
                  "V3: terminal state (63, all_scored) = 50.0");

    FreeYatzyContext(ctx1);
    FreeYatzyContext(ctx2);
}

/*
 * Test that loading a non-existent file returns 0.
 */
static void test_load_nonexistent(void) {
    YatzyContext *ctx = CreateYatzyContext();
    ASSERT(ctx != NULL, "Nonexistent: create ctx");

    int loaded = LoadAllStateValues(ctx, "/tmp/nonexistent_yatzy_v3.bin");
    ASSERT_EQ_INT(loaded, 0, "Loading nonexistent file returns 0");

    FreeYatzyContext(ctx);
}

/*
 * Test reachability table basic properties.
 */
static void test_reachability(void) {
    YatzyContext *ctx = CreateYatzyContext();
    ASSERT(ctx != NULL, "Reachability: create ctx");
    PrecomputeLookupTables(ctx);

    /* Base case: no upper cats scored, upper_score = 0 is reachable */
    ASSERT_EQ_INT(ctx->reachable[0][0], 1,
                  "Reachable: mask=0, up=0 (base case)");

    /* No upper cats scored, upper_score > 0 is unreachable */
    ASSERT_EQ_INT(ctx->reachable[0][1], 0,
                  "Unreachable: mask=0, up=1");
    ASSERT_EQ_INT(ctx->reachable[0][63], 0,
                  "Unreachable: mask=0, up=63");

    /* Only Ones scored (mask=0x01): up can be 0..5 */
    ASSERT_EQ_INT(ctx->reachable[1][0], 1, "Reachable: Ones scored, up=0");
    ASSERT_EQ_INT(ctx->reachable[1][5], 1, "Reachable: Ones scored, up=5");
    ASSERT_EQ_INT(ctx->reachable[1][6], 0, "Unreachable: Ones scored, up=6");

    /* Only Sixes scored (mask=0x20): up can be 0,6,12,18,24,30 */
    ASSERT_EQ_INT(ctx->reachable[0x20][0], 1, "Reachable: Sixes scored, up=0");
    ASSERT_EQ_INT(ctx->reachable[0x20][6], 1, "Reachable: Sixes scored, up=6");
    ASSERT_EQ_INT(ctx->reachable[0x20][30], 1, "Reachable: Sixes scored, up=30");
    ASSERT_EQ_INT(ctx->reachable[0x20][1], 0, "Unreachable: Sixes scored, up=1");
    ASSERT_EQ_INT(ctx->reachable[0x20][7], 0, "Unreachable: Sixes scored, up=7");

    /* All upper cats scored (mask=0x3F): up=63 should be reachable (>=63) */
    ASSERT_EQ_INT(ctx->reachable[0x3F][63], 1,
                  "Reachable: all upper cats scored, up=63 (>=63 cap)");

    FreeYatzyContext(ctx);
}

TEST_MAIN_BEGIN("test_storage")
    test_file_exists();
    test_v3_round_trip();
    test_load_nonexistent();
    test_reachability();
    cleanup_files();
TEST_MAIN_END
