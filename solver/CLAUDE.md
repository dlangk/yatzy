# CLAUDE.md — Solver

Rust HPC engine: backward-induction DP, Monte Carlo simulation, REST API.

## Commands

```bash
cargo build --release         # Build (~30s with LTO)
cargo test                    # 163 tests (136 unit + 25 integration + 2 ignored)
cargo fmt --check             # Formatting
cargo clippy                  # Lints

# From repo root (YATZY_BASE_PATH=.):
just precompute               # θ=0 strategy table (~504ms)
just serve                    # API server on port 9000
just simulate                 # 1M games lockstep
just sweep                    # All 37 θ values
just bench-check              # Performance regression test
```

## Crate Structure

Single crate (not a workspace). Release profile: opt-level 3, fat LTO, codegen-units 1, panic=abort.

28 binary targets in `src/bin/`. Key ones:

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
- `dice_set_probabilities: [f32; 252]` — P(dice set) for each of 252 ordered combinations
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
| POST | `/density` | `{upper_score, scored_categories, accumulated_score}` | ~100ms | Exact score distribution from mid-game |
| GET | `/score_histogram` | — | <1ms | Binned score distribution |
| GET | `/statistics` | — | <1ms | Aggregated game statistics |

## Concurrency Model

- **tokio** (rt-multi-thread): async HTTP server
- **rayon** (8 threads): parallel backward induction and batch simulation
- No shared mutable state at runtime — context is immutable after precomputation
- Density endpoint uses `tokio::spawn_blocking` for CPU-heavy forward DP

## Risk-Sensitive Solver (θ parameter)

- θ=0: expected value. θ<0: risk-averse. θ>0: risk-seeking.
- |θ| ≤ 0.15: utility-domain solver (same speed as EV)
- |θ| > 0.15: log-domain LSE solver (~2.7x slower)
- θ grid: 37 values from -3.0 to +3.0 in `configs/theta_grid.toml`
- **CRITICAL**: Delete `data/strategy_tables/all_states_theta_*.bin` after changing solver code!

## Storage Format

Binary files: 16-byte header (magic `0x59545A53` + version 5) + `f32[4,194,304]`.
Zero-copy mmap loading via memmap2 (<1ms). Files are ~16 MB each.

## Performance Testing

```bash
just bench-check    # Compare against baseline, PASS/FAIL
just bench-baseline # Record new baseline
just bench          # Print only
```

Baseline: `.overhaul/performance-baseline.json`. Threshold: `max(mean + 3σ, mean × 1.05)`.
