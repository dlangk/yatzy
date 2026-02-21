# Test Baseline

**Date:** 2026-02-21
**Machine:** Apple M1 Max
**Rust:** rustc 1.93.1 (01f6ddf75 2026-02-11)
**Node:** vitest 3.2.4

## Summary

| Component | Framework | Tests | Passed | Failed | Ignored | Duration |
|-----------|-----------|-------|--------|--------|---------|----------|
| Solver (lib) | cargo test | 138 | 136 | 0 | 2 | 2.86s |
| Solver (integration) | cargo test | 25 | 25 | 0 | 0 | 0.04s |
| Frontend | vitest | 34 | 34 | 0 | 0 | 305ms |
| Blog | vitest | 25 | 25 | 0 | 0 | 270ms |
| Analytics | — | 0 | — | — | — | — |

**Total: 222 tests, 220 passed, 0 failed, 2 ignored.**

## Pre-Existing Ignored Tests

These 2 tests are `#[ignore]`d in source and are excluded from the pass/fail contract:

1. `density::forward::tests::test_density_mean_matches_ev` — requires precomputed strategy table
2. `density::forward::tests::test_density_probability_conservation` — requires precomputed strategy table

## Clippy / Lint Status

- **Rust clippy**: 0 errors, 82 warnings (all pre-existing, not introduced by overhaul)
- **Rust fmt**: clean (after auto-format of bench_wall_clock.rs)
- **Frontend lint**: `eslint .` available via `npm run lint`

## Test Suites Detail

### Solver: `cargo test` (163 total)

**Unit tests** (`src/lib.rs`): 138 tests (136 pass, 2 ignored)
- `api_computations::tests` — 5 tests (fat response variants)
- `batched_solver::tests` — 2 tests (terminal, EV consistency)
- `dice_mechanics::tests` — 4 tests (count, find, sort, probability)
- `game_mechanics::tests` — 9 tests (scoring rules)
- `phase0_tables::tests` — 7 tests (252 sets, keep table, reachability)
- `rosetta::dsl::tests` — 4 tests (features, bonus pace)
- `rosetta::policy::tests` — 32 tests (CFG parsing, keep actions, semantic)
- `simd::tests` — 7 tests (NEON intrinsics)
- `simulation::adaptive::tests` — 7 tests (bonus, categories, policy lookup)
- `simulation::engine::tests` — 7 tests (game sim, recording, determinism)
- `simulation::fast_prng::tests` — 4 tests (PRNG range, distribution)
- `simulation::heuristic::tests` — 11 tests (heuristic keep/pick)
- `simulation::radix_sort::tests` — 4 tests (sort correctness)
- `simulation::raw_storage::tests` — 8 tests (binary I/O round-trip)
- `simulation::statistics::tests` — 3 tests (aggregate, percentiles, JSON)
- `simulation::strategy::tests` — 5 tests (game view, underdog)
- `storage::tests` — 3 tests (file I/O, v4 round-trip)
- `widget_solver::tests` — 4 tests (bench, reroll, late game)
- `density::forward::tests` — 2 tests (IGNORED)
- `density::transitions::tests` — 2 tests (transitions)

**Integration tests** (`tests/test_precomputed.rs`): 25 tests (all pass)
- Requires `data/strategy_tables/all_states.bin` on disk
- Tests: EV positivity, monotonicity, terminal states, optimal reroll, file format, etc.

### Frontend: `vitest run` (34 tests)

- `src/mask.test.ts` — 11 tests (bitmask operations)
- `src/reducer.test.ts` — 23 tests (game state reducer)

### Blog: `vitest run` (25 tests)

- `js/game/mask.test.js` — 11 tests (bitmask operations)
- `js/game/reducer.test.js` — 14 tests (game state reducer)

### Analytics: No tests

The Python analytics package has no test suite. Test files found were only in `.venv/` site-packages (third-party).

## Commands

```bash
# Run all tests
just test                    # Solver only (cargo test)

# Component-specific
cd solver && cargo test      # Rust solver (163 tests)
cd frontend && npm test      # Frontend vitest (34 tests)
cd blog && npm test          # Blog vitest (25 tests)

# Lints
cd solver && cargo fmt --check && cargo clippy
cd frontend && npm run lint
```

## Contract

From this point forward, `just test` (solver) must pass after every phase. Frontend and blog tests (`npm test` in each directory) must also pass. Any new test failure introduced by the overhaul must be fixed before proceeding.
