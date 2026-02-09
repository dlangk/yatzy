# CLAUDE.md — Backend

Instructions for Claude Code when working in this directory.

## Commands

```bash
cargo build --release         # Build
cargo test                    # 29 unit + 20 integration = 49 tests
cargo fmt --check             # Formatting
cargo clippy                  # Lints

# Precompute state values (required once, ~90s)
YATZY_BASE_PATH=. RAYON_NUM_THREADS=8 target/release/yatzy-precompute

# Start API server (port 9000)
YATZY_BASE_PATH=. target/release/yatzy
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

Entry points: `src/bin/precompute.rs` and `src/bin/server.rs`.

## Algorithm Reference

The code implements `theory/optimal_yahtzee_pseudocode.md` (at repo root), adapted for Scandinavian Yatzy (15 categories, 50-point upper bonus, no Yahtzee bonus flag).

### Computation Pipeline

1. **Phase 0** (`phase0_tables.rs`): Build lookup tables — dice combinations, scores, keep-multiset transition table, probabilities, reachability
2. **Phase 2** (`state_computation.rs`): Backward induction from |C|=14 to |C|=0, calling SOLVE_WIDGET per state via rayon `par_iter`

### State Representation

S = (m, C) where m = upper score [0,63], C = 15-bit scored-categories bitmask. Index: `m * 2^15 + C` (2,097,152 slots).

## Key Patterns

- **Hot path**: `widget_solver.rs` inner loops use `#[inline(always)]`, `get_unchecked`, cached `sv` slice
- **Parallel writes**: `state_computation.rs` uses `AtomicPtr` + unsafe raw pointer writes (each state index unique)
- **Keep-multiset dedup**: 462 unique keeps vs 31 raw masks per dice set (~47% fewer dot products)
- **Storage**: 16-byte header + float32[2,097,152] (~8 MB), zero-copy mmap loading

## Important Notes

- `data/` is a symlink to `../backend-legacy-c/data` (shared `all_states.bin`)
- DP uses f64 internally; only storage is f32
- `RAYON_NUM_THREADS=8` is optimal on Apple Silicon
