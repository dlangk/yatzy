# Yatzy Backend (Legacy C)

> **This is the legacy C implementation.** The primary backend is now the [Rust version](../backend-rust/) (`backend-rust/`), which produces bit-identical output with better safety, simpler builds, and comparable performance. This code is kept for reference.

C-based API server and DP solver for Scandinavian Yatzy. Precomputes optimal expected values for all 2M+ game states using backward induction, then serves real-time strategy advice via a REST API.

## Features

- **Optimal Strategy**: Dynamic programming with backward induction computes E(S) for every game state
- **Fast Precomputation**: ~81s on Apple Silicon (M1 Max, 8 threads) for 1.4M reachable states
- **Keep-Multiset Dedup**: Collapses equivalent reroll masks into unique keep-multisets (avg 16.3 vs 31 masks per dice set, ~47% fewer dot products)
- **Zero-Copy Loading**: Memory-mapped I/O loads precomputed states in <1ms
- **REST API**: JSON endpoints for evaluating actions, retrieving scores, and optimal suggestions
- **Parallel Computation**: OpenMP for multi-threaded state computation

## Dependencies

- **CMake** (>= 3.10)
- **GCC** with OpenMP support (gcc-15 on macOS via Homebrew)
- **libmicrohttpd** (HTTP server)
- **json-c** (JSON handling)

### Install (macOS)

```bash
brew install cmake libmicrohttpd json-c gcc
```

### Install (Ubuntu)

```bash
sudo apt update
sudo apt install cmake libmicrohttpd-dev libjson-c-dev gcc build-essential -y
```

## Build

```bash
cd backend
mkdir -p build && cd build
cmake .. -DCMAKE_C_COMPILER=gcc-15   # gcc-15 on macOS; plain gcc on Linux
make -j8
```

## Run

```bash
# Precompute state values (required once, ~81s)
YATZY_BASE_PATH=.. OMP_NUM_THREADS=8 ./yatzy_precompute

# Start API server (port 9000)
YATZY_BASE_PATH=.. ./yatzy
```

## Tests

```bash
# Unit tests
for t in test_context test_dice_mechanics test_game_mechanics test_phase0 test_storage test_widget; do
  ./$t
done

# Precomputed validation (requires data/all_states.bin)
YATZY_BASE_PATH=.. ./test_precomputed
```

## Architecture

### Source Layout

```
src/
  yatzy.c               Entry point: API server startup
  precompute.c           Entry point: offline precomputation
  context.c              YatzyContext lifecycle and Phase 0 orchestration
  phase0_tables.c        Phase 0: lookup tables (scores, keep-multiset table, probabilities)
  state_computation.c    Phase 2: DP backward induction loop with progress tracking
  widget_solver.c        Phase 2a: SOLVE_WIDGET — computes E(S) for a single state
  api_computations.c     API wrappers composing widget solver primitives for HTTP handlers
  webserver.c            HTTP API server (libmicrohttpd + json-c)
  game_mechanics.c       Yatzy scoring rules: s(S, r, c)
  dice_mechanics.c       Dice operations and probability calculations
  storage.c              Binary file I/O (Storage v3 format, mmap)
  utilities.c            Logging and environment helpers

include/                 Header files for all modules
tests/                   Unit and integration tests (test_helpers.h framework)
data/                    Precomputed state values (all_states.bin, ~8 MB)
theory/                  Mathematical documentation
```

### Computation Pipeline

1. **Phase 0** — Precompute lookup tables:
   - Enumerate 252 sorted 5-dice multisets R_{5,6}
   - Precompute scores s(S, r, c) for all (dice_set, category) pairs
   - Build keep-multiset transition table (462 keeps, sparse per-row storage)
   - Compute P(empty -> r) initial roll probabilities
   - Mark reachable (upper_mask, upper_score) pairs (31.8% pruned)

2. **Phase 2** — Backward induction (|C|=15 down to |C|=0):
   - For each state S, SOLVE_WIDGET computes E(S) through Groups 6->5->3->1
   - Group 6: best category score for each final roll
   - Groups 5 & 3: best keep decision via deduplicated keep-multiset iteration
   - Group 1: expected value over initial roll distribution

### Key Data Structures

- **YatzyContext**: All precomputed tables + DP result array (E_table)
- **KeepTable**: Sparse per-row transition probabilities with mask dedup mappings. 462 unique keep-multisets, 4,368 non-zero entries, 51 KB. Inline in YatzyContext (no heap allocation)
- **State**: (upper_score in [0,63], scored_categories bitmask of 15 categories)
- **Storage v3**: 16-byte header + float[2,097,152] in STATE_INDEX order (~8 MB)

### Performance

| Optimization | Impact |
|---|---|
| Keep-multiset dedup | 47% fewer dot products (16.3 vs 31 masks/ds) |
| Reachability pruning | 31.8% of (mask, score) pairs eliminated |
| Storage v3 (float32) | 8 MB file, <1ms mmap load |
| OpenMP (8 threads) | Near-linear scaling on P-cores |
| Compiler flags | -O3 -mcpu=apple-m1 -flto -funroll-loops |

## Scandinavian Yatzy Rules

- 15 scoring categories (Ones through Yatzy)
- 50-point upper section bonus (not 35 as in American Yahtzee)
- 3 rolls per turn (initial + 2 rerolls)
- Upper section: sum of matching face values (capped at 63 for bonus tracking)
