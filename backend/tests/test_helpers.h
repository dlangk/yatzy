/*
 * test_helpers.h — Shared test macros for the Yatzy test suite
 *
 * Provides ASSERT macros and main() boilerplate so all test files
 * use a consistent framework.
 */
#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
    } else { \
        tests_passed++; \
        printf("PASS: %s\n", msg); \
    } \
} while (0)

#define ASSERT_EQ_INT(a, b, msg) ASSERT((a) == (b), msg)
#define ASSERT_EQ_DBL(a, b, msg) ASSERT(fabs((a) - (b)) < 1e-9, msg)
#define ASSERT_NEAR(a, b, tol, msg) ASSERT(fabs((a) - (b)) < (tol), msg)

#define TEST_MAIN_BEGIN(name) \
    int main(void) { \
        printf("=== %s ===\n\n", name);

#define TEST_MAIN_END \
        printf("\n%d/%d tests passed.\n", tests_passed, tests_run); \
        return (tests_passed == tests_run) ? 0 : 1; \
    }

#endif /* TEST_HELPERS_H */
