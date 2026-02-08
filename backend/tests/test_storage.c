/*
 * test_storage.c — Round-trip I/O tests for per-level and consolidated files
 *
 * Creates a context, sets known state values, saves/loads, and compares.
 */
#include <string.h>
#include <unistd.h>
#include "test_helpers.h"
#include "context.h"
#include "storage.h"

#define TEST_PER_LEVEL_FILE  "/tmp/yatzy_test_states_15.bin"
#define TEST_MMAP_FILE       "/tmp/yatzy_test_all_states.bin"

static void cleanup_files(void) {
    unlink(TEST_PER_LEVEL_FILE);
    unlink(TEST_MMAP_FILE);
}

static void test_file_exists(void) {
    ASSERT_EQ_INT(FileExists("/tmp"), 1, "FileExists('/tmp') = 1");
    ASSERT_EQ_INT(FileExists("/tmp/nonexistent_yatzy_test_xyz"), 0,
                  "FileExists(nonexistent) = 0");
}

/*
 * Per-level round-trip: save level 15 states, load into fresh context, compare.
 */
static void test_per_level_round_trip(void) {
    YatzyContext *ctx1 = CreateYatzyContext();
    ASSERT(ctx1 != NULL, "Per-level: create ctx1");
    PrecomputeLookupTables(ctx1);

    /* Set known terminal state values */
    int all_scored = (1 << CATEGORY_COUNT) - 1;
    for (int up = 0; up <= 63; up++) {
        double val = (up >= 63) ? 50.0 : 0.0;
        ctx1->state_values[STATE_INDEX(up, all_scored)] = val;
    }

    /* Save */
    SaveStateValuesForCount(ctx1, 15, TEST_PER_LEVEL_FILE);
    ASSERT_EQ_INT(FileExists(TEST_PER_LEVEL_FILE), 1,
                  "Per-level file created");

    /* Load into fresh context */
    YatzyContext *ctx2 = CreateYatzyContext();
    ASSERT(ctx2 != NULL, "Per-level: create ctx2");
    PrecomputeLookupTables(ctx2);

    int loaded = LoadStateValuesForCount(ctx2, 15, TEST_PER_LEVEL_FILE);
    ASSERT_EQ_INT(loaded, 1, "Per-level: load succeeds");

    /* Compare */
    int match = 1;
    for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
        if (ctx2->scored_category_count_cache[scored] != 15) continue;
        for (int up = 0; up <= 63; up++) {
            int idx = STATE_INDEX(up, scored);
            if (fabs(ctx1->state_values[idx] - ctx2->state_values[idx]) > 1e-12) {
                match = 0;
                fprintf(stderr, "  Mismatch at up=%d scored=%d: %.12f != %.12f\n",
                        up, scored, ctx1->state_values[idx], ctx2->state_values[idx]);
                break;
            }
        }
        if (!match) break;
    }
    ASSERT(match, "Per-level round-trip: saved and loaded values match");

    FreeYatzyContext(ctx1);
    FreeYatzyContext(ctx2);
}

/*
 * Mmap round-trip: save all states, load via mmap, compare.
 */
static void test_mmap_round_trip(void) {
    YatzyContext *ctx1 = CreateYatzyContext();
    ASSERT(ctx1 != NULL, "Mmap: create ctx1");
    PrecomputeLookupTables(ctx1);

    /* Set some known values: terminal states + a few others */
    int all_scored = (1 << CATEGORY_COUNT) - 1;
    for (int up = 0; up <= 63; up++) {
        ctx1->state_values[STATE_INDEX(up, all_scored)] = (up >= 63) ? 50.0 : 0.0;
    }
    /* Set a recognizable value for state (0, 0) */
    ctx1->state_values[STATE_INDEX(0, 0)] = 123.456;
    /* Set a recognizable value for state (63, 1) */
    ctx1->state_values[STATE_INDEX(63, 1)] = 789.012;

    /* Save via mmap */
    SaveAllStateValuesMmap(ctx1, TEST_MMAP_FILE);
    ASSERT_EQ_INT(FileExists(TEST_MMAP_FILE), 1, "Mmap file created");

    /* Load into fresh context */
    YatzyContext *ctx2 = CreateYatzyContext();
    ASSERT(ctx2 != NULL, "Mmap: create ctx2");

    int loaded = LoadAllStateValuesMmap(ctx2, TEST_MMAP_FILE);
    ASSERT_EQ_INT(loaded, 1, "Mmap: load succeeds");

    /* Compare all states */
    int match = 1;
    for (int i = 0; i < NUM_STATES; i++) {
        if (fabs(ctx1->state_values[i] - ctx2->state_values[i]) > 1e-12) {
            match = 0;
            fprintf(stderr, "  Mmap mismatch at index %d: %.12f != %.12f\n",
                    i, ctx1->state_values[i], ctx2->state_values[i]);
            break;
        }
    }
    ASSERT(match, "Mmap round-trip: all state values match");

    /* Spot-check the recognizable values */
    ASSERT_EQ_DBL(ctx2->state_values[STATE_INDEX(0, 0)], 123.456,
                  "Mmap: state (0,0) = 123.456");
    ASSERT_EQ_DBL(ctx2->state_values[STATE_INDEX(63, 1)], 789.012,
                  "Mmap: state (63,1) = 789.012");
    ASSERT_EQ_DBL(ctx2->state_values[STATE_INDEX(63, all_scored)], 50.0,
                  "Mmap: terminal state (63, all_scored) = 50.0");

    FreeYatzyContext(ctx1);
    FreeYatzyContext(ctx2);
}

TEST_MAIN_BEGIN("test_storage")
    test_file_exists();
    test_per_level_round_trip();
    test_mmap_round_trip();
    cleanup_files();
TEST_MAIN_END
