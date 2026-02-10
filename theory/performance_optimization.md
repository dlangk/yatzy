# Performance Optimization: Precomputation from ~20 min to ~2.3s

This document traces the optimization of the backward induction precomputation
from the original C monolith to the current Rust implementation. Each step
targets a specific bottleneck in the cache hierarchy of the Apple M1 Max
(8 P-cores, 128KB L1d/core, 12MB shared L2, ~48MB SLC).

## Target hardware

| Level | Size | Latency | Role in solver |
|-------|------|---------|----------------|
| L1d | 128 KB per core | ~4 cycles | KeepTable (51KB), successor regions (256B), keep_ev (3.7KB) |
| L2 | 12 MB shared | ~12 cycles | State values array during parallel access |
| SLC | ~48 MB | ~40 cycles | Full 8MB state array fits here |
| DRAM | 64 GB | ~200+ cycles | Only touched on cold start |

## Summary

| Step | Change | Time | Speedup | Key insight |
|------|--------|------|---------|-------------|
| 0 | Original C monolith | ~20 min | — | Dense 16MB table thrashes L2 |
| 1 | Sparse CSR + OpenMP | ~4 min | 5x | CSR fits in 12MB shared L2 |
| 2 | Compiler flags + DP variant | ~3 min | 1.3x | NEON vectorization, branch elimination |
| 3 | float32 storage + reachability | ~111s | 1.6x | 8MB state array fits SLC; 31.8% fewer states |
| 4 | Keep-multiset dedup | ~81s | 1.4x | 51KB KeepTable fits L1 |
| 5 | Rust port | ~117s | 0.7x | Safety overhead (recovered next step) |
| 6 | Unsafe hot path tuning | ~92s | 1.3x | Eliminate bounds checks, enum dispatch |
| 7 | Keep-EV dedup + index swap + grouped par | ~2.3s | 40x | 462 vs 4108 dot products; L1 successor hits |
| 8 | f32 internal computation | ~2.3s | — | Enables GPU path (10.4 vs 0.3 TFLOPS); max 0.00046 pt diff |

**Total: ~500x end-to-end.**

---

## Step 0: Original C monolith (~20 min)

A single `yatzy.c` file (1,542 lines). The transition table was a dense 3D array:

```c
double transition_table[252][32][252];  // 16.3 MB, 79% zeros
```

This stored `P(keep → outcome)` for every combination of dice set (252),
reroll mask (32), and outcome dice set (252). The table alone exceeded the
M1 Max's 12MB shared L2 cache. Every widget solve thrashed the cache
hierarchy, pulling data from SLC or DRAM at ~200+ cycles per access.

State index layout: `upper_score * 2^15 + scored_categories`. This scatters
successor lookups across the full 8MB state array because scoring a category
changes the lower 15 bits unpredictably.

No parallelism. Single-threaded.

## Step 1: Sparse CSR + OpenMP (5x)

**Commit `fc899e7`**

Replaced the dense transition table with Compressed Sparse Row (CSR) format.
Only 21% of entries were non-zero, so CSR reduced memory from 16.3MB to 5.1MB
— fitting comfortably in the 12MB shared L2.

```
Dense:  16.3 MB → exceeds L2 → SLC/DRAM reads at ~200 cycles
Sparse: 5.1 MB  → fits in L2 → reads at ~12 cycles
```

Added OpenMP parallelism with `schedule(guided)` over states per level.
Confirmed that 8 P-core threads outperform 10 threads (E-cores hurt
throughput on Apple Silicon's big.LITTLE architecture).

Also decomposed the monolithic `computations.c` into focused modules:
`phase0_tables.c`, `widget_solver.c`, `api_computations.c`,
`state_computation.c`.

## Step 2: Compiler tuning + DP-only variant (1.3x)

**Commit `e5a12b6`**

- `-O3 -mcpu=apple-m1 -flto -funroll-loops` enables Firestorm-specific
  scheduling and NEON auto-vectorization
- Added `ComputeMaxEVForNRerolls` — a DP-only variant that drops mask
  tracking, eliminating bookkeeping branches from the hot path
- Ping-pong buffers `E[2][252]` replaced per-level heap allocation
- `restrict` qualifiers on sparse table pointers let the compiler prove
  no aliasing, enabling SIMD vectorization of dot product inner loops
- Replaced `omp critical` with `omp atomic` for the progress counter

## Step 3: float32 storage + reachability pruning (1.6x)

**Commit `0a9fb13`**

Stored state values as `f32` instead of `f64`, halving the binary from 16MB
to 8MB. The DP still uses f64 internally for accumulation precision — only
stored results use f32 (7 significant digits, sufficient for game EVs in
0-400 range).

The 8MB state array now fits in the SLC with room to spare. Group 6
successor lookups that miss L2 hit the SLC at ~40 cycles instead of DRAM
at ~200+ cycles.

Added reachability pruning: a small DP pre-marks which `(upper_mask,
upper_score)` pairs are achievable. 31.8% of states are unreachable
(e.g., upper_score=63 with only Ones scored). Pruning reduces the
workload from 2.1M to 1.43M states.

## Step 4: Keep-multiset deduplication (1.4x)

**Commit `dca29dd`**

Multiple reroll masks can produce the same kept-dice multiset. For example,
dice `[1,1,2,3,4]` with masks `0b00001` and `0b00010` both keep `{1,2,3,4}`.
The KeepTable collapses these duplicates:

```
Raw masks per dice set:    31
Unique keeps per dice set: 16.3 avg (462 total across all dice sets)
Non-zero CSR entries:      4,368 (was 424,368)
Table memory:              51 KB (was 5.1 MB)
```

At 51KB, the KeepTable fits entirely in L1 cache (128KB on Firestorm).
The sparse row data stays hot across all 252 dice sets within one widget
solve.

## Step 5: Rust port (parity)

**Commit `36819b0`**

Full rewrite: libmicrohttpd → axum, json-c → serde, OpenMP → rayon,
manual mmap → memmap2. Same binary format, same API contract, 49 tests.

Rayon's work-stealing scheduler adapts better to core heterogeneity than
OpenMP's static/guided scheduling. The `StateValues` enum gives zero-copy
mmap loading at runtime while allowing mutable writes during precomputation.

Initial Rust performance was ~117s (vs 81s in C) due to safety overhead.

## Step 6: Unsafe hot path tuning (1.3x)

**Commit `050282f`**

Eliminated overhead from Rust's safety guarantees in the inner loops:

- **`sv` slice caching**: `ctx.state_values.as_slice()` involves an enum
  match. Called billions of times, this dominated Group 6. Cache the slice
  once per widget.
- **`get_unchecked`**: Bounds checks on `vals[k] * e_ds_prev[cols[k]]`
  added ~15% overhead. Indices are guaranteed valid by construction.
- **Unsafe direct writes via `AtomicPtr`**: `par_iter().map().collect()`
  allocates a Vec and scatters results. Direct writes skip the allocation
  — safe because `state_index` is injective.
- **`#[inline(always)]`**: Forces inlining of widget solver hot functions.

## Step 7: Three-way optimization (40x)

**Commit `9ce3fe5`**

Three interlocking optimizations that amplify each other:

### 7a. Keep-EV deduplication (algorithmic — 9x fewer dot products)

Within one call to `compute_max_ev_for_n_rerolls`, the input array
`e_ds_prev` is the **same** for all 252 dice sets. The sparse dot product
for keep `kid` depends only on `kid` and `e_ds_prev`. But each `kid`
appears in an average of 8.9 dice sets — meaning the same dot product was
computed 8.9 times redundantly.

```
Before: 252 dice sets × 16.3 keeps/ds = 4,108 dot products per call
After:  462 unique dot products + 4,108 table lookups per call
```

The fix computes all 462 keep EVs in a single sequential pass over the
CSR rows (ideal for NEON auto-vectorization via `fmla` instructions), then
distributes results via array lookups from a 3.7KB buffer that sits
entirely in L1.

### 7b. State index layout swap (memory layout — 45x faster Group 6)

```rust
// Before: up * 32768 + scored  → scattered across 8MB
// After:  scored * 64 + up     → 256 bytes contiguous
fn state_index(up: usize, scored: usize) -> usize {
    scored * 64 + up
}
```

Group 6 iterates unscored categories. For each category `c`, it looks up
`sv[state_index(new_up, scored | (1<<c))]`. With the new layout, all 64
upper-score variants of the same scored mask occupy a contiguous **256-byte
region** (64 × 4 bytes). On Firestorm, an L1 cache line is 128 bytes —
two cache lines cover all possible successor upper scores for a given
scored mask.

### 7c. Grouped parallelism (amplifies 7b)

States are grouped by `scored_categories` before rayon dispatch. Each task
processes all upper scores for one scored mask sequentially:

```rust
groups.par_iter().for_each(|(scored, ups)| {
    for &up in ups {
        // Successive widgets reuse the same successor cache lines
        solve_widget(up, scored);
    }
});
```

This gives temporal locality on top of spatial locality — consecutive
widgets in the same group hit the same 256-byte successor regions that
7b made contiguous.

### 7d. Compiler tuning

- `panic = "abort"` eliminates unwinding tables, improving instruction
  cache density
- `target-cpu=native` enables the full Firestorm instruction set (NEON
  SIMD, fused multiply-add, 8-wide decode scheduling)
- The keep-EV sequential accumulation loop is a prime target for NEON
  `fmla` (fused multiply-accumulate) auto-vectorization

### 7e. Lower-category successor preloading

For lower categories (6-14), the successor state value
`sv[state_index(up, scored|(1<<c))]` is constant across all 252 dice sets
(it depends only on `up` and `scored`, not the roll). Preloading these
~5 values before the dice-set loop eliminates ~1,255 redundant L1 reads
per widget.

## Final measured breakdown

Measured with feature-gated timing instrumentation (`--features timing`):

```
Wall clock:         2.3s
Sum widget time:    20.3s (across 8 cores)
  Group 6  (sv):    3.5s  (17%)
  Group 53 (dot):  16.5s  (81%)
  Group 1  (sum):   0.3s  (2%)
Parallel efficiency: 96% of 8 cores
Per widget:         14.2 µs = ~45,000 cycles @ 3.2GHz
```

Group 53 dominates because the sparse dot product has a **gather load**
(`e_ds_prev[cols[k]]`) that defeats NEON vectorization — ARM NEON has no
hardware gather instruction, so each access is a scalar load. This is a
fundamental hardware limitation at the current sparsity (3.7% density).

## Theoretical floor

The absolute minimum work per widget:

| Component | Operations | Min cycles |
|-----------|-----------|------------|
| Group 6 | 252 × ~7.5 categories (score read + sv read + add + compare) | ~6,300 |
| Groups 5+3 | 2 × (4,368 FMAs + 4,108 comparisons) | ~17,000 |
| Group 1 | 252 multiply-accumulates | ~500 |
| **Total** | | **~24,000** |

At 24K cycles: `1.43M × 24K / 8 / 3.2 GHz = 1.34s`

Current performance (2.3s) is at **52% of theoretical peak**. The gap is
almost entirely the cost of scatter-gather memory access patterns in the
sparse CSR dot product — a fundamental limitation of the format on ARM.

## Step 8: f32 internal computation (precision validation)

**Commit `5ae9cdd`**

Switched all internal DP accumulation from f64 to f32. Previously, storage
was f32 but ping-pong buffers, keep_ev arrays, and dot product accumulators
used f64. Since every level boundary truncates to f32 anyway (for storage),
the f64 intermediate precision was wasted.

Empirical comparison across all 2,097,152 states:

```
States with any difference:  1,338,944 (64%)
Max absolute difference:     0.000458 points
Max relative difference:     4.05e-6
Game-start EV (f64):         248.439987
Game-start EV (f32):         248.440140
Difference:                  0.000153 points (0.00006%)
```

Decision impact analysis found 22 successor ordering flips out of 1.43M
reachable states, all involving pairs where the f64 values differ by less
than 0.0001 points. The EV loss from any individual "wrong" decision is
below 0.0002 points — completely irrelevant for gameplay.

This change enables future GPU acceleration: the M1 Max GPU runs f32 at
10.4 TFLOPS but f64 at only 325 GFLOPS (1/32 rate). With f32 throughout,
the GPU path becomes viable for a potential 5-10× additional speedup.

No performance change on CPU (the hot path was already operating on f32
values from the state array; the f64 accumulators just added unnecessary
widening/narrowing casts).

## Verification

All optimizations preserve numerical correctness:

- 49 tests (29 unit + 20 integration) pass on every change
- Integration tests check exact EVs (Yatzy=50, straights=15/20),
  optimal reroll masks, category choices, monotonicity across all 1.43M
  reachable states, and the upper bonus cliff
- Storage format versioned (v3 → v4) to prevent loading old-layout data
- f32 precision validated: max 0.00046 pt difference, zero decision impact
