# Benchmark Results: Post Phase 5 Cleanup

Measured on: Apple M1 Max, 8 P-cores, RAYON_NUM_THREADS=8

## Results

| Benchmark | Measured | Threshold | Baseline | Status |
|-----------|----------|-----------|----------|--------|
| phase0_tables | 724 μs | 1619 μs | 743 μs | PASS |
| precompute_ev | 489.9 ms | 532.3 ms | 504.0 ms | PASS |
| simulate_lockstep_10k | 102.3 ms | 118.2 ms | 102.2 ms | PASS |
| simulate_lockstep_100k | 597.9 ms | 630.2 ms | 593.1 ms | PASS |
| api_evaluate_2rerolls | 9 μs | 12 μs | 8 μs | PASS |
| api_evaluate_0rerolls | 2 μs | 2 μs | 2 μs | PASS |
| api_evaluate_late_game | 7 μs | 7 μs | 7 μs | PASS |

**ALL BENCHMARKS PASSED** — no performance regressions from Phase 5 changes.

## Changes Made in Phase 5

### 5.1 Web Server Optimization

**Blocking I/O moved to `spawn_blocking`:**
- `handle_get_score_histogram`: reads CSV file — now uses `tokio::task::spawn_blocking`
- `handle_get_statistics`: reads JSON file — now uses `tokio::task::spawn_blocking`
- `handle_density`: already used `spawn_blocking` (no change needed)
- `handle_evaluate`: pure in-memory computation (2-9μs), no I/O — correctly stays on async runtime
- `handle_get_state_value`: O(1) array lookup (<1μs), no I/O — correctly stays on async runtime
- `handle_health_check`: trivial — correctly stays on async runtime

**Input validation (handlers MUST NOT panic on user input):**
- `/evaluate`: added bounds checks for `upper_score` (0-63), `scored_categories` (15-bit), dice values (1-6)
- `/state_value`: added bounds checks for `upper_score` and `scored_categories`
- `/density`: added bounds checks for `upper_score`, `scored_categories`, `accumulated_score`
- All validation returns HTTP 400 with descriptive error message, never panics

**Serialization audit:**
JSON is appropriate for the gameplay endpoints. The `/evaluate` response is ~2 KB — JSON overhead
is negligible relative to network latency. Binary formats (MessagePack, protobuf) would save <1 KB
per response and add dependency/complexity for no user-visible benefit. The hot path is the
computation (2-9μs), not serialization.

### 5.2 Simulation & Sweep Ergonomics

No changes needed. The existing simulation entry points are already well-structured:
- `simulate_batch()`, `simulate_batch_with_recording()`, `simulate_batch_summaries()` — typed structs
- `sweep.rs` — inventory scan, grid resolution, `ensure_strategy_table()` with clean entry points
- Output includes metadata (theta, games, seed) in both binary headers and filenames

### 5.3 Code Clarity

**`debug_assert!` invariants added (zero cost in release builds):**
- `state_index()`: validates upper_score < STATE_STRIDE and scored_categories < 2^15
- `YatzyContext::get_state_value()`: validates upper_score 0-63 and scored 15-bit
- `PolicyOracle::idx()`: validates state_idx < NUM_STATES and ds < NUM_DICE_SETS

**`// PERF: intentional` comments** were added in Phase 4 (batched_solver.rs, lockstep.rs).

**Module-level documentation** was already comprehensive — widget_solver.rs, batched_solver.rs,
simd.rs, state_computation.rs, server.rs, api_computations.rs, types.rs, constants.rs, storage.rs,
and simulation/engine.rs all have detailed `//!` doc comments explaining purpose, algorithms,
and performance characteristics.

**Dead code:** None found (verified in Phase 4 duplication audit).

### 5.4 What Was NOT Changed

- No changes to hot-path code (widget_solver.rs, batched_solver.rs, simd.rs, state_computation.rs)
- No new dependencies added
- No unnecessary error handling in HPC core — `unwrap()`/`expect()` on structural invariants retained
- No `#[inline]` annotations changed — existing annotations are correct
