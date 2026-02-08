/*
 * state_computation.c — Phase 2: COMPUTE_OPTIMAL_STRATEGY
 *
 * Backward induction orchestrator. Processes states from |C|=15 (terminal)
 * down to |C|=0 (game start), calling SOLVE_WIDGET for each state.
 *
 * See: theory/optimal_yahtzee_pseudocode.md, Phase 2: COMPUTE_OPTIMAL_STRATEGY.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "state_computation.h"
#include "storage.h"
#include "phase0_tables.h"
#include "widget_solver.h"
#include "timing.h"

/* ── Progress tracking ───────────────────────────────────────────────── */

typedef struct {
    int total_states;
    int completed_states;
    double start_time;
    double last_report_time;
    int states_per_level[16];
    double time_per_level[16];
} ComputeProgress;

static void InitProgress(ComputeProgress *progress) {
    memset(progress, 0, sizeof(ComputeProgress));
    progress->start_time = timer_now();
    progress->last_report_time = progress->start_time;

    for (int num_scored = 0; num_scored <= 15; num_scored++) {
        int states = 0;
        for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
            if (__builtin_popcount(scored) == num_scored) {
                states += 64;
            }
        }
        progress->states_per_level[num_scored] = states;
        progress->total_states += states;
    }
}

static void UpdateProgress(ComputeProgress *progress, int level, int states_completed) {
    progress->completed_states += states_completed;
    progress->time_per_level[level] = timer_now() - progress->start_time;
}

static void PrintProgress(ComputeProgress *progress, int current_level) {
    double now = timer_now();
    if (progress->completed_states < progress->total_states &&
        now - progress->last_report_time < 0.5) {
        return;
    }
    progress->last_report_time = now;

    double elapsed = now - progress->start_time;
    double pct = (double)progress->completed_states / progress->total_states * 100.0;
    double rate = progress->completed_states / elapsed;
    double eta = (progress->total_states - progress->completed_states) / rate;

    printf("\rProgress: %d/%d states (%.1f%%) | Level: %d | Elapsed: %.1fs | Rate: %.0f states/s | ETA: %.1fs     ",
           progress->completed_states, progress->total_states, pct,
           current_level, elapsed, rate, eta);
    fflush(stdout);
}

/* ── Phase 2 main loop ───────────────────────────────────────────────── */

void ComputeAllStateValues(YatzyContext *ctx) {
    ComputeProgress progress;
    InitProgress(&progress);

    printf("=== Starting State Value Computation ===\n");
    printf("Total states to compute: %d\n", progress.total_states);
    printf("States per level:\n");
    for (int i = 0; i <= 15; i++) {
        printf("  Level %2d: %6d states\n", i, progress.states_per_level[i]);
    }
    printf("\n");

    double total_start = timer_now();

    /* Try to load consolidated file first */
    const char *consolidated_file = "data/all_states.bin";
    if (LoadAllStateValuesMmap(ctx, consolidated_file)) {
        printf("Loaded pre-computed states from consolidated file\n");
        return;
    }

    /* Process states level by level, from game end (15) to game start (0) */
    for (int num_scored = 15; num_scored >= 0; num_scored--) {
        double level_start = timer_now();
        char level_filename[64];
        snprintf(level_filename, sizeof(level_filename), "data/states_%d.bin", num_scored);

        PrintProgress(&progress, num_scored);

        /* Level 15: terminal states (base case) */
        if (num_scored == 15) {
            if (!LoadStateValuesForCount(ctx, num_scored, level_filename)) {
                InitializeFinalStates(ctx);
                SaveStateValuesForCount(ctx, num_scored, level_filename);
            }
            UpdateProgress(&progress, num_scored, progress.states_per_level[num_scored]);
            continue;
        }

        /* Try loading from disk */
        if (LoadStateValuesForCount(ctx, num_scored, level_filename)) {
            UpdateProgress(&progress, num_scored, progress.states_per_level[num_scored]);
            continue;
        }

        /* Compute this level */
        printf("\nComputing level %d (%d states)...\n",
               num_scored, progress.states_per_level[num_scored]);

        int states_to_compute = progress.states_per_level[num_scored];
        int (*state_list)[2] = malloc(sizeof(int) * 2 * states_to_compute);
        if (!state_list) {
            fprintf(stderr, "Memory allocation failed for level %d\n", num_scored);
            return;
        }

        int state_index = 0;
        for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
            if (__builtin_popcount(scored) == num_scored) {
                for (int up = 0; up <= 63; up++) {
                    state_list[state_index][0] = up;
                    state_list[state_index][1] = scored;
                    state_index++;
                }
            }
        }

        const int BATCH = 1000;

        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < state_index; i++) {
            int up = state_list[i][0];
            int scored = state_list[i][1];

            YatzyState current = {up, scored};
            double ev = ComputeExpectedStateValue(ctx, &current);
            ctx->state_values[STATE_INDEX(up, scored)] = ev;

            if (i % BATCH == 0) {
                #pragma omp critical
                {
                    UpdateProgress(&progress, num_scored, BATCH);
                    PrintProgress(&progress, num_scored);
                }
            }
        }

        int remaining = state_index % BATCH;
        if (remaining > 0) {
            UpdateProgress(&progress, num_scored, remaining);
        }

        free(state_list);
        SaveStateValuesForCount(ctx, num_scored, level_filename);

        double level_dur = timer_now() - level_start;
        printf("\nLevel %d completed in %.2f seconds (%.0f states/sec)\n",
               num_scored, level_dur, progress.states_per_level[num_scored] / level_dur);
    }

    printf("\n\n=== Computation Complete ===\n");

    printf("\nSaving consolidated state file...\n");
    SaveAllStateValuesMmap(ctx, consolidated_file);

    double total_time = timer_now() - total_start;
    printf("\nTotal computation time: %.2f seconds\n", total_time);
    printf("Average processing rate: %.0f states/second\n", progress.total_states / total_time);

    printf("\nDetailed timing breakdown by level:\n");
    printf("Level | States  | Time (s) | Rate (states/s)\n");
    printf("------|---------|----------|----------------\n");

    double prev_time = 0;
    for (int level = 15; level >= 0; level--) {
        double level_time = progress.time_per_level[level] - prev_time;
        if (level_time > 0) {
            printf("  %2d  | %6d  | %7.2f  | %6.0f\n",
                   level, progress.states_per_level[level], level_time,
                   progress.states_per_level[level] / level_time);
        }
        prev_time = progress.time_per_level[level];
    }
}
