/*
 * state_computation.h — Phase 2: COMPUTE_OPTIMAL_STRATEGY
 *
 * Backward induction orchestrator that processes states from |C|=15
 * (terminal) down to |C|=0 (game start), calling SOLVE_WIDGET for
 * each state.
 *
 * See: theory/optimal_yahtzee_pseudocode.md, Phase 2.
 */
#ifndef STATE_COMPUTATION_H
#define STATE_COMPUTATION_H

#include "context.h"

/**
 * Compute expected values for all possible game states using backward
 * induction. Tries to load precomputed values from disk first; if not
 * found, computes level by level (|C|=15 down to 0) and saves results.
 */
void ComputeAllStateValues(YatzyContext *ctx);

#endif /* STATE_COMPUTATION_H */
