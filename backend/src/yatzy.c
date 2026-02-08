/*
 * yatzy.c — Entry point for the Delta Yatzy API server.
 *
 * Startup:
 *   1. Set up working directory and allocate game context
 *   2. Build static lookup tables
 *   3. Load precomputed state values (fall back to computing if not found)
 *   4. Start HTTP server
 *
 * Run yatzy_precompute first to generate data/all_states.bin for fast startup.
 */

#include <stdio.h>
#include <stdlib.h>

#include <context.h>
#include <webserver.h>
#include "utilities.h"
#include "storage.h"
#include "state_computation.h"

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    SetWorkingDirectory();
    printf("Starting yatzy API server...\n");

    YatzyContext *ctx = CreateYatzyContext();
    if (!ctx) {
        fprintf(stderr, "Failed to allocate YatzyContext\n");
        return EXIT_FAILURE;
    }

    PrecomputeLookupTables(ctx);

    if (!LoadAllStateValuesMmap(ctx, "data/all_states.bin")) {
        printf("No precomputed data found, computing (run yatzy_precompute to avoid this)...\n");
        ComputeAllStateValues(ctx);
    }

    start_webserver(ctx, 9000);

    FreeYatzyContext(ctx);
    return 0;
}
