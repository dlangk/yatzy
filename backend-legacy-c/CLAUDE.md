# Backend CLAUDE.md (Legacy C)

> **Legacy**: The primary backend is `backend-rust/`. This C code is kept for reference only.

## Build & Test

```bash
# Build (from repo root)
cmake -S backend -B backend/build -DCMAKE_C_COMPILER=gcc-15 -DCMAKE_BUILD_TYPE=Release
make -C backend/build -j8

# Unit tests
for t in test_context test_dice_mechanics test_game_mechanics test_phase0 test_storage test_widget; do
  backend/build/$t
done

# Precomputed validation (requires data/all_states.bin)
YATZY_BASE_PATH=backend backend/build/test_precomputed

# Precompute (required once, ~81s)
YATZY_BASE_PATH=backend OMP_NUM_THREADS=8 backend/build/yatzy_precompute

# Run API server (port 9000)
YATZY_BASE_PATH=backend backend/build/yatzy
```

## Architecture

### Computation Pipeline

**Phase 0** — Precompute lookup tables (< 1ms):
1. Factorials, dice combinations (252 multisets), category scores
2. Keep-multiset table: 462 unique keeps, sparse per-row storage (4,368 nnz, 51 KB)
3. Dice set probabilities P(empty -> r)
4. Reachability pruning (31.8% of upper pairs eliminated)

**Phase 2** — Backward induction (|C|=15 down to 0, ~81s with 8 threads):
- SOLVE_WIDGET computes E(S) via Groups 6 -> 5 -> 3 -> 1
- Hot path: iterate deduplicated keep-multisets (avg 16.3 vs 31 masks) with sparse dot products

### Key Source Files

| File | Role |
|------|------|
| `context.h` | YatzyContext and KeepTable data structures |
| `phase0_tables.c` | Phase 0: all lookup tables including PrecomputeKeepTable |
| `widget_solver.c` | SOLVE_WIDGET: ComputeMaxEVForNRerolls (DP hot path) |
| `state_computation.c` | Phase 2 DP loop, OpenMP parallelism, progress tracking |
| `api_computations.c` | API wrappers: best reroll, evaluate mask, distributions |
| `webserver.c` | HTTP handlers, CORS, JSON serialization |
| `storage.c` | Storage v3: 16-byte header + float[2M], mmap load/save |

### KeepTable (context.h)

The central optimization. A keep-multiset is the sorted multiset of dice kept before rerolling. Multiple 5-bit reroll masks can produce the same keep (e.g., dice [1,1,2,3,4] with masks 0b00001 and 0b00010 both keep {1,2,3,4}).

```
KeepTable:
  vals[4368], cols[4368]     Sparse probability entries (value + column index)
  row_start[463]             Per-keep row boundaries into vals/cols
  unique_count[252]          Deduplicated keeps per dice set (avg 16.3)
  unique_keep_ids[252][31]   Keep indices for each unique keep
  mask_to_keep[252*32]       Reroll mask -> keep index (for API input)
  keep_to_mask[252*32]       Unique keep index -> representative mask (for API output)
```

Inline in YatzyContext — no heap allocation needed.

### State Representation

- `upper_score` in [0, 63]: capped sum of upper category scores
- `scored_categories`: 15-bit bitmask (Ones=bit 0 through Yatzy=bit 14)
- `STATE_INDEX(up, scored) = up * 2^15 + scored` — flat array index
- Total: 2,097,152 states; 1,430,528 reachable after pruning

### Storage v3

- Header: magic `0x59545A53` + version `3` + total_states + reserved (16 bytes)
- Data: float[2,097,152] in STATE_INDEX order
- File size: 8,388,624 bytes (~8 MB)
- Load: zero-copy mmap (<1ms)

## Important Notes

- Must use gcc (not clang) for OpenMP — `gcc-15` from Homebrew on macOS
- IDE diagnostics showing `-fopenmp` errors are false positives from clang
- OMP_NUM_THREADS=8 is optimal on Apple Silicon; 10 threads hits E-cores
- Scandinavian Yatzy: 15 categories, 50-point upper bonus (not 35)
- DP uses double internally; only storage is float32 (7 significant digits)
- Tests use ASSERT_NEAR with 1e-4 tolerance for float precision
