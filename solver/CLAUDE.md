# CLAUDE.md — Solver

Rust HPC engine: backward-induction DP, Monte Carlo simulation, REST API.

## Commands

```bash
cargo build --release                  # Build (~30s with LTO)
cargo build --features timing          # Build with per-level timing output enabled
cargo test                             # 189 tests (139 unit + 14 API + 8 property + 28 integration, 2 ignored)
cargo fmt --check                      # Formatting
cargo clippy                           # Lints

# From repo root (YATZY_BASE_PATH=.):
just precompute               # θ=0 strategy table (~1.1s)
just serve                    # API server on port 9000
just simulate                 # 1M games lockstep
just sweep                    # All 37 θ values
just bench-check              # Performance regression test
```

## Crate Structure

Single crate (not a workspace). Release profile: opt-level 3, fat LTO, codegen-units 1, panic=abort.

29 binary targets in `src/bin/`. Key ones:

| Binary | Purpose |
|--------|---------|
| `yatzy` | axum HTTP API server |
| `yatzy-precompute` | Backward induction DP (+ optional `--oracle`) |
| `yatzy-simulate` | Monte Carlo simulation |
| `yatzy-sweep` | θ sweep (resumable) |
| `yatzy-density` | Exact forward-DP PMF |
| `yatzy-bench` | Wall-clock performance benchmarks |

## Core Data Structures

### YatzyContext (`types.rs`)

Central context holding all precomputed tables and state values.

- `state_values: StateValues` — E[score | optimal play] for all 4,194,304 state slots
- `keep_table: KeepTable` — CSR-format probability matrix (462 unique keeps × 252 dice sets)
- `dice_set_probabilities: [f64; NUM_DICE_SETS]` — P(dice set) for each of 252 ordered combinations
- `precomputed_scores: [[i32; 15]; 252]` — s(r, c) for all dice sets and categories
- `reachable: [[bool; 64]; 64]` — reachability mask indexed by [upper_mask][upper_score]
- `theta: f32` — risk parameter (0 = EV-optimal)

### State Representation (`constants.rs`)

- S = (upper_score, scored_categories) where m ∈ [0,63], C = 15-bit bitmask
- `STATE_STRIDE = 128` (topological padding: indices 64-127 duplicate the capped value)
- `state_index(up, scored) = scored * 128 + up`
- `NUM_STATES = 4,194,304` (32,768 masks × 128 stride)
- Only ~1.43M states are reachable after pruning

### KeepTable (`types.rs`)

Sparse probability matrix in CSR format. For each dice set, maps to ≤462 unique kept-dice multisets with transition probabilities. Key dedup: 462 unique keeps vs 31 raw reroll masks (~47% reduction).

### PolicyOracle (`types.rs`)

Precomputed argmax decisions for θ=0. Three flat `Vec<u8>` arrays (~3.17 GB total):

- `oracle_cat[si * 252 + ds]` — best category (0-14)
- `oracle_keep1[si * 252 + ds]` — best keep with 1 reroll left
- `oracle_keep2[si * 252 + ds]` — best keep with 2 rerolls left

## Hot Paths

| Hot Path | Location | What It Computes | Why It's Fast | Do NOT |
|----------|----------|------------------|---------------|--------|
| SOLVE_WIDGET | `widget_solver.rs` | E(S) for one state | Ping-pong buffers, `#[inline(always)]`, `get_unchecked` | Add allocations, dynamic dispatch |
| Batched solver | `batched_solver.rs` | 64 upper scores simultaneously | SpMV→SpMM, cache-friendly | Break the 64-wide batch pattern |
| NEON SIMD | `simd.rs` | FMA, max, argmax over 64-element arrays | 14 NEON intrinsic kernels, fast exp | Use scalar fallbacks |
| Backward induction | `state_computation.rs` | All 1.43M states | rayon par_iter, AtomicPtr writes | Add synchronization |
| Lockstep sim | `simulation/lockstep.rs` | N games in parallel | Radix sort grouping, SplitMix64 PRNG | Add per-game branching |
| Oracle sim | `simulation/lockstep.rs` | N games with O(1) lookups | No DP recomputation | Change oracle encoding |
| Density evolution | `density/transitions.rs` | Exact PMF propagation | Sort-merge, prob-array propagation | Add Monte Carlo variance |

## API Reference

Server: axum on port 9000, stateless, `Arc<YatzyContext>` shared state.

| Method | Path | Request | Latency | Purpose |
|--------|------|---------|---------|---------|
| GET | `/health` | — | <1ms | Health check |
| GET | `/state_value` | `?upper_score=N&scored_categories=N` | <1ms | E_table[S] lookup |
| POST | `/evaluate` | `{dice, upper_score, scored_categories, rerolls_remaining}` | 2-9μs | All keep-mask EVs + category EVs |
| POST | `/density` | `{upper_score, scored_categories, accumulated_score}` | <10ms (precomputed) / ~50ms (MC) | Score distribution percentiles (hybrid: precomputed turns 0-4, MC turns 5-14) |

## Concurrency Model

- **tokio** (rt-multi-thread): async HTTP server
- **rayon** (defaults to `std::thread::available_parallelism()`, overridable via `RAYON_NUM_THREADS`): parallel backward induction and batch simulation
- No shared mutable state at runtime — context is immutable after precomputation
- Density endpoint uses `tokio::spawn_blocking` for CPU-heavy forward DP

## Risk-Sensitive Solver (θ parameter)

- θ=0: expected value (EV-optimal). θ<0: risk-averse. θ>0: risk-seeking.
- |θ| ≤ 0.15: utility-domain solver (same speed as EV)
- |θ| > 0.15: log-domain LSE solver (~2.7x slower)
- θ grid: 37 values from -3.0 to +3.0 in `configs/theta_grid.toml`
- Math: `theory/foundations/risk-parameter-theta.md`
- Strategy analysis: `theory/strategy/risk-sensitive-strategy.md`
- **CRITICAL**: Delete `data/strategy_tables/all_states_theta_*.bin` AND `data/strategy_tables/oracle.bin` after changing solver code! A stale oracle silently plays an old policy (`just simulate` mtime-checks it against the θ=0 table as a backstop).

## Storage Format

Binary files: 16-byte header (magic `0x59545A53` + version 6) + `f32[4,194,304]`.
Zero-copy mmap loading via memmap2 (<1ms). Files are ~16 MB each.

## Performance Testing

```bash
just bench-check    # Compare against baseline, PASS/FAIL
just bench-baseline # Record new baseline
just bench          # Print only
```

Baseline: `.benchmarks/performance-baseline.json`. Threshold: `max(mean + 3σ, mean × 1.10)`.

### Reference Numbers

| Benchmark | M1 Max (8 threads) | M5 Max (18 threads, 2026-07) |
|-----------|---------------------|----------------------|
| Precompute θ=0 (EV), DP compute | ~1.1s | 113 ms |
| Precompute θ=0, cold end-to-end (`just precompute`) | — | ~0.12s |
| Precompute \|θ\|≤0.15 (utility) | ~0.49s | ~0.12s |
| Precompute \|θ\|>0.15 (LSE) | ~2.7s | ~0.60s |
| Lockstep simulation (100k) | 342 ms | ~105 ms |
| Lockstep simulation (1M) | — | ~283 ms (~3.5M games/s) |
| Oracle simulation (1M) | — | ~81 ms (~12M games/s) |
| `just simulate` 1M end-to-end | — | ~0.09s (oracle) / ~0.30s (lockstep) |
| API `/evaluate` | 2-9μs | 2-6μs |
| Density evolution (oracle) | ~3.0s | — |
| Density evolution (non-oracle) | ~381s | — |

`just simulate` (θ=0, `--output`) picks the fastest available engine:
the PolicyOracle when `data/strategy_tables/oracle.bin` exists and is newer
than the θ=0 table (`just precompute-oracle`, one-time ~0.8s, 3.17 GB), else
lockstep. Pass `--vertical` to force the per-game engine (bit-compatible
with pre-2026-07 outputs). Lockstep persists per-keep EVs per state and
builds per-group DecisionTables for groups ≥384 games. Optimization history
and measured dead ends: `theory/lab-reports/hardware-and-hot-path.md` §6.

## Module Structure

### `src/` (core)

| File | Purpose |
|------|---------|
| `lib.rs` | Crate root, re-exports all public modules |
| `constants.rs` | STATE_STRIDE=128, state_index(), NUM_STATES, category constants |
| `types.rs` | YatzyContext, KeepTable (f32), StateValues, PolicyOracle |
| `dice_mechanics.rs` | Dice operations: face counting, sorting, index lookup, probability |
| `game_mechanics.rs` | Scoring rules s(S, r, c) and upper-score successor function |
| `phase0_tables.rs` | Phase 0: precompute all static lookup tables (rolls, scores, keeps, reachability) |
| `widget_solver.rs` | SOLVE_WIDGET with ping-pong buffers |
| `batched_solver.rs` | BatchedBuffers, 4 solver variants (EV/risk/utility/max), oracle builder |
| `state_computation.rs` | Phase 2 DP with rayon par_iter, optional oracle building |
| `simd.rs` | NEON intrinsic kernels (14 functions + fast exp) |
| `storage.rs` | Binary I/O, zero-copy mmap (version 6/7, 128-stride), oracle I/O |
| `forward_pass.rs` | Exact forward pass: state visitation probabilities for D3 visualizations |
| `env_config.rs` | Shared env config (YATZY_BASE_PATH, RAYON_NUM_THREADS, YATZY_PORT) |
| `api_computations.rs` | Computation logic for API endpoints |
| `server.rs` | axum HTTP server setup and route handlers |

### `src/simulation/`

| File | Purpose |
|------|---------|
| `mod.rs` | Module root, re-exports |
| `engine.rs` | Sequential simulate_game, simulate_batch, GameRecord |
| `lockstep.rs` | Horizontal lockstep simulation (radix sort + SplitMix64) |
| `fast_prng.rs` | SplitMix64 PRNG (8-byte state, 5 dice from single u64) |
| `radix_sort.rs` | 2-pass counting sort for O(N) grouping by state_index |
| `adaptive.rs` | Adaptive θ policies (turn-dependent risk) |
| `sweep.rs` | Theta sweep infrastructure (resumable) |
| `raw_storage.rs` | Binary I/O for simulation data (mmap) |
| `heuristic.rs` | Heuristic (human-like) strategy: pattern-matching rerolls + greedy scoring |
| `multiplayer.rs` | N-player round-robin games with opponent-aware strategies |
| `statistics.rs` | Statistics aggregation from GameRecord data (scores, categories, rerolls) |
| `strategy.rs` | Strategy abstraction: PlayerState, GameView, CLI spec parsing |

### `src/density/`

| File | Purpose |
|------|---------|
| `forward.rs` | Forward density evolution (dense Vec<f64> arrays), oracle variant |
| `transitions.rs` | Per-state transition computation, oracle variant (sort-merge) |

## Test Files

| File | Tests | Purpose |
|------|-------|---------|
| `src/**/*.rs` (unit) | 138 | Inline `#[cfg(test)]` modules |
| `tests/test_api.rs` | 14 | API endpoint integration tests |
| `tests/test_properties.rs` | 8 | Proptest property-based tests |
| `tests/test_precomputed.rs` | 25 | State value correctness (ignored without data) |
