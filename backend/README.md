# Yatzy Backend

Rust backend for Delta Yatzy. Precomputes optimal expected values for all 2M+ game states using backward induction, then serves real-time strategy advice via a REST API.

## Quick Start

```bash
cargo build --release

# Precompute state values (required once, ~90s)
YATZY_BASE_PATH=. RAYON_NUM_THREADS=8 target/release/yatzy-precompute

# Start API server (port 9000)
YATZY_BASE_PATH=. target/release/yatzy
```

## Tests

```bash
cargo test           # 29 unit + 20 integration = 49 tests
cargo fmt --check    # formatting
cargo clippy         # lints
```

## Algorithm

The code implements the algorithm described in [`theory/optimal_yahtzee_pseudocode.md`](../theory/optimal_yahtzee_pseudocode.md). The pseudocode targets standard Yahtzee (13 categories, 35-point bonus); this implementation adapts it for Scandinavian Yatzy (15 categories, 50-point bonus, no Yahtzee bonus flag).

### State Representation

A game state **S = (m, C)** where:
- **m** ∈ [0, 63]: capped upper-section score
- **C**: 15-bit bitmask of scored categories (Ones = bit 0, Yatzy = bit 14)

Flat array index: `state_index(m, C) = m × 2^15 + C` → 2,097,152 total slots.

### Computation Pipeline

The computation runs in three phases, mapping directly to the pseudocode:

#### Phase 0 — Precompute lookup tables

**Pseudocode**: `PRECOMPUTE_ROLLS_AND_PROBABILITIES`
**Code**: `phase0_tables.rs` → `precompute_lookup_tables()`

Runs 8 sub-steps in dependency order (< 1ms total):

| Step | Function | What it builds |
|------|----------|----------------|
| 1 | `precompute_factorials` | 0!..5! for multinomial coefficients |
| 2 | `build_all_dice_combinations` | R_{5,6}: all 252 sorted 5-dice multisets + 5D reverse lookup |
| 3 | `precompute_category_scores` | s(S, r, c) for all 252 × 15 (dice set, category) pairs |
| 4 | `precompute_keep_table` | Sparse CSR transition matrix P(r'→r) with keep-multiset dedup |
| 5 | `precompute_dice_set_probabilities` | P(⊥→r) for each r ∈ R_{5,6} |
| 6 | `precompute_scored_category_counts` | Popcount cache: bitmask C → \|C\| |
| 7 | `initialize_final_states` | Terminal states: E(S) = 50 if m ≥ 63, else 0 |
| 8 | `precompute_reachability` | Phase 1 pruning: which (upper_mask, m) pairs are achievable |

Step 4 is the most important. It enumerates 462 keep-multisets R_k, computes P(K→T) via the multinomial formula, and builds dedup mappings so the inner loop iterates only unique keeps (avg 16.3 per dice set instead of 31 raw masks).

Step 8 implements the pseudocode's `COMPUTE_REACHABILITY`, eliminating ~31.8% of state space.

#### Phase 2 — Backward induction

**Pseudocode**: `COMPUTE_OPTIMAL_STRATEGY`
**Code**: `state_computation.rs` → `compute_all_state_values()`

Processes states level by level, from |C| = 14 down to |C| = 0 (terminal states at |C| = 15 are already set by Phase 0, step 7). For each level:

1. Build a list of all reachable (upper_score, scored_categories) pairs at this level
2. In parallel (rayon `par_iter`), call `SOLVE_WIDGET` for each state
3. Write E(S) directly into the state array via unsafe raw pointers (each state index is unique, so no data races)

#### SOLVE_WIDGET — The inner loop

**Pseudocode**: `SOLVE_WIDGET(S)`
**Code**: `widget_solver.rs` → `compute_expected_state_value()`

Computes E(S) for one turn-start state by evaluating 6 groups bottom-up. Uses two ping-pong buffers `e[0]` and `e[1]` (each 252 × f64) to avoid allocation:

```
Step 1: Group 6 → e[0]     E(S, r, 0) for all 252 dice sets r
        Best category for each final roll.
        Calls compute_best_scoring_value_for_dice_set_by_index()

Step 2: Group 5 → e[1]     E(S, r, 1) for all 252 dice sets r
        Best keep after 1st reroll. For each dice set, iterates
        deduplicated keep-multisets and picks max Σ P(r'→r'') · e[0][r''].
        Calls compute_max_ev_for_n_rerolls()

Step 3: Group 3 → e[0]     E(S, r, 2) for all 252 dice sets r
        Best keep from initial roll. Same structure as step 2,
        but reads from e[1] and writes back to e[0].
        Calls compute_max_ev_for_n_rerolls()

Step 4: Group 1 → E(S)     Weighted sum P(⊥→r) · e[0][r]
        Expected value over all possible initial rolls.
```

The pseudocode's Groups 2+3 and 4+5 are fused into single passes (steps 3 and 2 above).

### Code Flow Diagram

```
bin/precompute.rs          bin/server.rs
      │                          │
      ▼                          ▼
phase0_tables.rs ────────► phase0_tables.rs
  precompute_lookup_tables()     precompute_lookup_tables()
      │                          │
      ▼                          ▼
state_computation.rs         storage.rs
  compute_all_state_values()     load_all_state_values()  ← mmap, <1ms
      │                          │
      │  for each level:         │
      │    par_iter states        │
      │      │                    │
      │      ▼                   ▼
      │  widget_solver.rs    server.rs
      │    compute_expected_     create_router()
      │    state_value()           │
      │      │                     ▼
      │      │               api_computations.rs
      │      │                 (re-solves widgets
      │      │                  on demand for
      │      │                  API queries)
      │      ▼
      │  storage.rs
      │    save_all_state_values()
      ▼
   data/all_states.bin  (~8 MB)
```

## Source Modules

| Module | Pseudocode | Role |
|--------|------------|------|
| `constants.rs` | — | Game constants, `state_index()`, category definitions |
| `types.rs` | S, E_table, R_k | `YatzyContext`, `KeepTable`, `StateValues`, `YatzyState` |
| `dice_mechanics.rs` | R_{5,6} | Face counting, sorting, index lookup, P(⊥→r) |
| `game_mechanics.rs` | s(S,r,c), u(S,r,c) | Category scoring (15 categories), upper-score update |
| `phase0_tables.rs` | Phase 0 + Phase 1 | All 8 precomputation steps including reachability |
| `widget_solver.rs` | `SOLVE_WIDGET` | Groups 6→5→3→1 with ping-pong buffers |
| `state_computation.rs` | `COMPUTE_OPTIMAL_STRATEGY` | Level-by-level backward induction with rayon |
| `api_computations.rs` | — | API wrappers: best reroll, evaluate mask, distributions |
| `server.rs` | — | Axum HTTP router, 9 endpoints, CORS |
| `storage.rs` | — | Binary file I/O, zero-copy mmap via memmap2 |

## Key Optimizations

| Optimization | Effect |
|---|---|
| Keep-multiset dedup | 47% fewer dot products (16.3 vs 31 keeps per dice set) |
| Reachability pruning | 31.8% of (mask, score) pairs eliminated |
| Cached `sv` slice | Avoids enum match on every E_table lookup in hot path |
| `#[inline(always)]` | Forces inlining of `compute_max_ev_for_n_rerolls` and scoring |
| `get_unchecked` | Removes bounds checks in inner loops (indices from precomputation) |
| Unsafe direct writes | Eliminates `collect()` + scatter allocation in parallel DP |
| Sparse CSR | Only 4,368 non-zero entries stored across 462 keep rows (51 KB) |
| Fat LTO + codegen-units=1 | Whole-program optimization in release builds |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/state_value` | Look up E(S) for a given state |
| GET | `/score_histogram` | Binned score distribution from CSV |
| POST | `/evaluate_category_score` | Score for placing dice in a category |
| POST | `/available_categories` | List categories with scores and validity |
| POST | `/evaluate_all_categories` | EV for each available category (0 rerolls) |
| POST | `/evaluate_actions` | EV for all 32 reroll masks |
| POST | `/suggest_optimal_action` | Best reroll mask or best category |
| POST | `/evaluate_user_action` | EV of a user's chosen action |

## Binary Storage Format

- **Header** (16 bytes): magic `0x59545A53` ("STZY") + version `3` + total_states + reserved
- **Data**: float32[2,097,152] in `state_index(m, C)` order
- **File size**: 8,388,624 bytes (~8 MB)
- **Loading**: zero-copy mmap via memmap2 (< 1ms)
