# CLAUDE.md — Solver

Instructions for Claude Code when working in this directory.

## Commands

```bash
cargo build --release         # Build
cargo test                    # Unit + integration tests
cargo fmt --check             # Formatting
cargo clippy                  # Lints

# Precompute state values (required once, ~2.3s, from repo root)
YATZY_BASE_PATH=. target/release/yatzy-precompute

# Start API server (port 9000, from repo root)
YATZY_BASE_PATH=. target/release/yatzy

# Simulate games (from repo root)
YATZY_BASE_PATH=. target/release/yatzy-simulate --games 1000000 --output data/simulations

# Theta sweep: simulate all thetas, resumable (from repo root)
YATZY_BASE_PATH=. target/release/yatzy-sweep --grid all --games 1000000
YATZY_BASE_PATH=. target/release/yatzy-sweep --list   # show inventory
```

## Source Layout

| Module | Pseudocode | Role |
|--------|------------|------|
| `constants.rs` | — | Game constants, `state_index()`, category definitions |
| `types.rs` | S, E_table, R_k | `YatzyContext`, `KeepTable`, `StateValues`, `YatzyState` |
| `dice_mechanics.rs` | R_{5,6} | Face counting, sorting, index lookup, P(r) |
| `game_mechanics.rs` | s(S,r,c), u(S,r,c) | Category scoring (15 categories), upper-score update |
| `phase0_tables.rs` | Phase 0 + Phase 1 | All 8 precomputation steps including reachability |
| `widget_solver.rs` | `SOLVE_WIDGET` | Groups 6-5-3-1 with ping-pong buffers |
| `state_computation.rs` | `COMPUTE_OPTIMAL_STRATEGY` | Level-by-level backward induction with rayon |
| `api_computations.rs` | — | API wrappers: best reroll, evaluate mask, distributions |
| `server.rs` | — | Axum HTTP router, 9 endpoints, CORS |
| `storage.rs` | — | Binary file I/O, zero-copy mmap via memmap2 |
| `simulation/engine.rs` | — | Game simulation with optimal strategy, recording |
| `simulation/statistics.rs` | — | Aggregate statistics from recorded games |
| `simulation/raw_storage.rs` | — | Binary I/O for raw simulation data (mmap) |
| `simulation/sweep.rs` | — | Theta sweep: inventory scan, grid resolution, ensure_strategy_table |

Entry points: `src/bin/precompute.rs`, `src/bin/server.rs`, `src/bin/simulate.rs`, `src/bin/sweep.rs`, `src/bin/category_sweep.rs`, `src/bin/pivotal_scenarios.rs`, `src/bin/yatzy_conditional.rs`.

## Algorithm Reference

The code implements `theory/pseudocode.md` (at repo root), adapted for Scandinavian Yatzy (15 categories, 50-point upper bonus, no Yahtzee bonus flag).

### Computation Pipeline

1. **Phase 0** (`phase0_tables.rs`): Build lookup tables — dice combinations, scores, keep-multiset transition table, probabilities, reachability
2. **Phase 2** (`state_computation.rs`): Backward induction from |C|=14 to |C|=0, calling SOLVE_WIDGET per state via rayon `par_iter`

### State Representation

S = (m, C) where m = upper score [0,63], C = 15-bit scored-categories bitmask. Index: `C * 64 + m` (2,097,152 slots).

## Key Patterns

- **Hot path**: `widget_solver.rs` inner loops use `#[inline(always)]`, `get_unchecked`, cached `sv` slice
- **Parallel writes**: `state_computation.rs` uses `AtomicPtr` + unsafe raw pointer writes (each state index unique)
- **Keep-multiset dedup**: 462 unique keeps vs 31 raw masks per dice set (~47% fewer dot products)
- **Storage**: 16-byte header + float32[2,097,152] (~8 MB), zero-copy mmap loading

## Important Notes

- Strategy tables live at `data/strategy_tables/` (repo root)
- Simulation output goes to `data/simulations/` (repo root)
- Analytics package lives at `analytics/` (repo root)
- All computation uses f32 throughout (storage + internal accumulation)
- `RAYON_NUM_THREADS=8` is optimal on Apple Silicon
