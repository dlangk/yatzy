/*
 * precompute.c — Entry point for precomputing all Yatzy state values.
 *
 * Computes optimal expected values for every game state via backward-induction DP
 * and saves results to disk. No server is started.
 */

#include <stdio.h>
#include <stdlib.h>

#include <context.h>
#include "utilities.h"
#include "state_computation.h"

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    SetWorkingDirectory();
    printf("Yatzy precomputation tool\n");

    YatzyContext *ctx = CreateYatzyContext();
    if (!ctx) {
        fprintf(stderr, "Failed to allocate YatzyContext\n");
        return EXIT_FAILURE;
    }

    PrecomputeLookupTables(ctx);
    ComputeAllStateValues(ctx);

    printf("Precomputation complete.\n");
    FreeYatzyContext(ctx);
    return 0;
}
