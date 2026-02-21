# Performance Baseline

**Date:** 2026-02-21
**Machine:** Apple M1 Max (8 P-cores)
**Rust:** rustc 1.93.1 (01f6ddf75 2026-02-11)
**Threads:** RAYON_NUM_THREADS=8
**Phase:** 2b of OVERHAUL_PLAN.md

## Benchmarks

| Benchmark | Mean | σ | p95 | Threshold | Description |
|-----------|------|---|-----|-----------|-------------|
| `phase0_tables` | 0.7 ms | 0.3 ms | 1.6 ms | 1.6 ms | Lookup table construction (462 keeps, scores, reachability) |
| `precompute_ev` | 504 ms | 9.4 ms | 522 ms | 532 ms | Full θ=0 backward induction (1.43M states, 8 threads) |
| `simulate_lockstep_10k` | 102 ms | 5.3 ms | 116 ms | 118 ms | 10K games lockstep simulation |
| `simulate_lockstep_100k` | 593 ms | 12 ms | 615 ms | 630 ms | 100K games lockstep (end-to-end) |
| `api_evaluate_2rerolls` | 9 μs | 1 μs | 9 μs | 11 μs | Single /evaluate call, mid-game, 2 rerolls |
| `api_evaluate_0rerolls` | 2 μs | 0 μs | 2 μs | 2 μs | Single /evaluate call, mid-game, 0 rerolls |
| `api_evaluate_late_game` | 7 μs | 0 μs | 7 μs | 7 μs | Single /evaluate call, 12 categories scored, 2 rerolls |

## Threshold Formula

```
threshold = max(mean + 3σ, mean × 1.05)
```

The 3σ bound catches real regressions. The 5% floor prevents false failures on very tight benchmarks.

## Commands

```bash
# Record new baseline
just bench-baseline

# Check against baseline (exits non-zero on regression)
just bench-check

# Run benchmarks (print only)
just bench
```

## Implementation

- Binary: `solver/src/bin/bench_wall_clock.rs`
- Baseline data: `.overhaul/performance-baseline.json`
- Uses `compute_all_state_values_nocache()` to bypass mmap cache for honest precompute timing
- Uses `std::hint::black_box()` to prevent dead code elimination for API benchmarks
- Warmup run before each benchmark series

## Notes

- `precompute_ev` includes both Phase 0 (tables) and Phase 2 (backward induction)
- Simulation benchmarks require `data/strategy_tables/all_states.bin` to exist
- API benchmarks test computation only (no HTTP round-trip), measuring the `compute_roll_response()` function directly
- All timings are wall-clock (includes OS scheduling jitter)
