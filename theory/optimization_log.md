# Optimization Log

Chronological record of all performance work on the Scandinavian Yatzy solver, from the original C monolith to the current NEON-optimized Rust implementation. All benchmarks on Apple M1 Max (8 Firestorm P-cores at 3.228 GHz, 128 KB L1d/core, 12 MB shared L2, ~48 MB SLC, 64 GB unified LPDDR5).

---

## Phase 1: C Implementation (Steps 0-4)

### Step 0: Original C Monolith (~20 min)

A single `yatzy.c` file (1,542 lines). The transition table was a dense 3D array of 16.3 MB (79% zeros), exceeding the 12 MB shared L2 cache. Every widget solve thrashed the cache hierarchy. State index layout `upper_score * 2^15 + scored_categories` scattered successor lookups across the full state array. No parallelism.

### Step 1: Sparse CSR + OpenMP (5x) — `fc899e7`

Replaced the dense transition table with Compressed Sparse Row (CSR) format, reducing memory from 16.3 MB to 5.1 MB — fitting in L2. Added OpenMP parallelism with `schedule(guided)`. Confirmed 8 P-core threads outperform 10 threads (E-cores hurt throughput on Apple Silicon's big.LITTLE architecture). Decomposed the monolithic C file into `phase0_tables.c`, `widget_solver.c`, `api_computations.c`, `state_computation.c`.

### Step 2: Compiler Tuning + DP-Only Variant (1.3x) — `e5a12b6`

Enabled `-O3 -mcpu=apple-m1 -flto -funroll-loops` for Firestorm-specific scheduling and NEON auto-vectorization. Added `ComputeMaxEVForNRerolls` — a DP-only variant dropping mask tracking. Ping-pong buffers `E[2][252]` replaced per-level heap allocation. `restrict` qualifiers on sparse table pointers enabled SIMD vectorization.

### Step 3: f32 Storage + Reachability Pruning (1.6x) — `0a9fb13`

Stored state values as f32 instead of f64, halving the binary from 16 MB to 8 MB. The state array now fits in the SLC. Group 6 successor lookups hit SLC at ~40 cycles instead of DRAM at ~200+ cycles. Added reachability pruning: 31.8% of states are unreachable (e.g., upper_score=63 with only Ones scored), reducing workload from 2.1M to 1.43M states.

### Step 4: Keep-Multiset Deduplication (1.4x) — `dca29dd`

Multiple reroll masks can produce the same kept-dice multiset. The KeepTable collapses these duplicates: 31 raw masks per dice set become 16.3 unique keeps on average (462 total). Table memory dropped from 5.1 MB to 51 KB — fitting entirely in L1 cache (128 KB on Firestorm). Non-zero CSR entries: 4,368 (was 424,368).

---

## Phase 2: Rust Port (Steps 5-8)

### Step 5: Rust Port (parity) — `36819b0`

Full rewrite: libmicrohttpd -> axum, json-c -> serde, OpenMP -> rayon, manual mmap -> memmap2. Same binary format, same API contract, 49 tests. Rayon's work-stealing adapts better to core heterogeneity than OpenMP's static/guided scheduling. `StateValues` enum gives zero-copy mmap loading at runtime. Initial Rust performance was ~117s (vs 81s C) due to safety overhead.

### Step 6: Unsafe Hot Path Tuning (1.3x) — `050282f`

Eliminated Rust safety overhead in inner loops:
- `sv` slice caching: `ctx.state_values.as_slice()` involves an enum match called billions of times; cache once per widget
- `get_unchecked`: bounds checks on `vals[k] * e_ds_prev[cols[k]]` added ~15% overhead
- Unsafe direct writes via `AtomicPtr`: skips `par_iter().collect()` allocation — safe because `state_index` is injective
- `#[inline(always)]`: forces inlining of hot functions

### Step 7: Three-Way Optimization (40x) — `9ce3fe5`

Three interlocking optimizations that amplify each other:

**7a. Keep-EV deduplication** — Within `compute_max_ev_for_n_rerolls`, the input array `e_ds_prev` is shared across all 252 dice sets. Each keep-multiset appears in avg 8.9 dice sets, so the same dot product was computed redundantly. Fix: compute all 462 keep EVs in a single pass, then distribute via array lookups from a 3.7 KB L1-resident buffer. Reduced from 4,108 to 462 dot products per call.

**7b. State index layout swap** — Changed from `up * 32768 + scored` to `scored * 64 + up`. All 64 upper-score variants of the same scored mask now occupy a contiguous 256-byte region — two L1 cache lines on Firestorm. Group 6 successor lookups hit at most 2 cache lines per scored mask.

**7c. Grouped parallelism** — States grouped by `scored_categories` before rayon dispatch. Each task processes all upper scores for one scored mask sequentially, giving temporal locality on top of spatial locality.

**7d. Compiler tuning** — `panic = "abort"` eliminates unwinding tables; `target-cpu=native` enables full Firestorm instruction set.

**7e. Lower-category successor preloading** — For lower categories (6-14), the successor state value is constant across all 252 dice sets. Preloading ~5 values eliminates ~1,255 redundant L1 reads per widget.

### Step 8: f32 Internal Computation — `5ae9cdd`

Switched all internal DP accumulation from f64 to f32. Validation across all 2,097,152 states: max absolute difference 0.000458 points, max relative difference 4.05e-6. Game-start EV: 248.439987 (f64) vs 248.440140 (f32). 22 successor ordering flips out of 1.43M reachable states, all below 0.0002 pt EV loss. Enables GPU path: M1 Max GPU runs f32 at 10.4 TFLOPS vs f64 at 325 GFLOPS.

### Summary After Phase 2

| Step | Change | Time | Cumulative Speedup |
|------|--------|------|-------------------|
| 0 | Original C monolith | ~20 min | 1x |
| 1 | Sparse CSR + OpenMP | ~4 min | 5x |
| 2 | Compiler flags + DP variant | ~3 min | 6.7x |
| 3 | f32 + reachability | ~111s | 10.8x |
| 4 | Keep-multiset dedup | ~81s | 14.8x |
| 5 | Rust port | ~117s | 10.3x |
| 6 | Unsafe hot path | ~92s | 13x |
| 7 | Three-way optimization | ~2.3s | 522x |
| 8 | f32 internal | ~2.3s | 522x |

Measured breakdown at 2.3s: Group 6 (sv lookups) 17%, Groups 5+3 (dot products) 81%, Group 1 (weighted sum) 2%. Parallel efficiency: 96% of 8 cores.

Theoretical floor: ~24K cycles/widget → 1.34s. Current performance at 52% of theoretical peak. The gap is the cost of scatter-gather memory access patterns in the CSR dot product — ARM NEON has no hardware gather instruction.

---

## Phase 3: Batched SoA Solver (SpMV -> SpMM)

### Hypothesis

Batching all 64 upper-score variants into a single solve operation would convert scattered SpMV inner loops into SpMM, where each CSR row is multiplied by a 64-wide vector instead of a scalar. Predicted 10-20x speedup.

### Implementation

`solve_widget_batched(ctx, sv, scored, bufs) -> [f32; 64]` processes all 64 upper scores simultaneously. Per-thread `BatchedBuffers` struct (247 KB: two 252x64 ping-pong matrices + one 462x64 keep_ev matrix) allocated once via rayon `for_each_init`, eliminating ~16 GB of heap allocation traffic.

### Result: 1.6x (not 10-20x)

| Configuration | Time | vs Baseline |
|---------------|------|-------------|
| Scalar solver (baseline) | 2.3s | 1.0x |
| Batched, per-call allocation | 1.53s | 1.5x |
| Batched, thread-local buffers | 1.40-1.42s | 1.6x |

### Why the Prediction Failed

1. **Scatter-gather persists**: The CSR outer loop still uses indirect `cols[k]` indexing — each iteration loads a different 256-byte row of `e_prev`. Batching amortizes the inner loop but not the outer loop's random access pattern.
2. **NEON is only 4-wide**: 64-element inner loop = 16 NEON iterations. With ~2-cycle overhead per CSR entry, the effective speedup on SpMM is 1.46x — closely matching the measured 1.6x.
3. **L1 pressure**: Working set grew from 3.8 KB (scalar) to 247 KB (batched), exceeding L1 and cycling through L2 instead.
4. **Group 6 was never the bottleneck**: Only 17% of baseline time; even a 64x Group 6 speedup would save ~0.15s.

### Conclusion

The batched SoA approach works but the dominant cost — indirect row access via CSR column indices — is inherent to the sparse format. To reach 10x+ would require replacing CSR with dense [462x252] matrix (viable via AMX/Accelerate) or GPU offload.

---

## Phase 4: NEON Intrinsics + Lockstep MC + Density Evolution

### Implementation

Three rewrites targeting M1 Max hardware natively:

**Solver**: STATE_STRIDE changed from 64 to 128 (topological padding — indices 64-127 duplicate the capped value). Eliminates `update_upper_score` branch, converts scattered sv loads into sequential reads. KeepTable.vals converted from f64 to f32, eliminating 4,368 `fcvt` conversions per widget. 14 hand-written NEON kernels in `simd.rs` including `neon_fast_exp_f32x4` (degree-5 minimax polynomial, ~2 ULP accuracy, ~8 NEON instructions vs ~20+ for libm `expf`).

**MC Simulation**: Lockstep horizontal processing — all N games advance through each turn together, sharing computation for games in the same state. Radix sort grouping (2-pass counting sort, O(N)) replaces HashMap for game grouping. SplitMix64 PRNG (8-byte state, 5 dice from single u64) replaces Xoshiro256++ (128-byte state).

**Density Evolution**: Dense `Vec<f64>` score distributions (size 384, padded for alignment) replacing HashMap. Parallel merge via rayon with bounded iteration (`max_i` tracking).

### Results

| Subsystem | Baseline | Optimized | Speedup |
|-----------|----------|-----------|---------|
| Solver (EV, θ=0) | 1.30s | 0.48s | 2.71x |
| Solver (utility, |θ|≤0.15) | 1.30s | ~0.49s | ~2.65x |
| Solver (LSE, |θ|>0.15) | 7.40s | ~2.7s | ~2.75x |
| MC simulation | 52K g/s | 242K g/s | 4.64x |
| Density evolution | 382s | 378s | 1.01x |

Solver speedup decomposition: topological padding ~1.4x, f32 probabilities ~1.1x, NEON intrinsics ~1.2x. Compound measured at 2.7x (super-linear due to cache effects from 128-aligned stride improving prefetch hit rate).

Density evolution showed no improvement because the optimization targeted the merge phase (~15% of runtime), but the bottleneck is transition computation (~85% of runtime). Each active state requires enumerating all 252 dice sets x all reroll decisions x all category choices.

---

## Phase 5: Policy Oracle

### Design

Precomputed argmax decisions: three flat `Vec<u8>` arrays (~1.057 GB each, ~3.17 GB total) indexed by `state_index * 252 + ds_index`. Built as side-effect of backward induction via scalar argmax pass after SIMD computation (Approach A). θ=0 only.

| Array | Content | Encoding |
|-------|---------|----------|
| `oracle_cat` | Best category (0-14) | u8 direct |
| `oracle_keep1` | Best keep, 1 reroll left | 0=keep-all, j+1=unique keep j |
| `oracle_keep2` | Best keep, 2 rerolls left | same |

### Results

| Subsystem | Before | After | Speedup |
|-----------|--------|-------|---------|
| Precompute (with oracle building) | 1.04s | 3.92s | 0.27x (3.77x overhead) |
| MC simulation | 107K g/s* | 1,572K g/s | 14.7x |
| Density evolution | ~378s | ~378s | 1.0x |

*Lockstep baseline under concurrent load; quiescent lockstep is 242K g/s.

Oracle transforms each MC game turn from full DP widget evaluation to three O(1) byte reads. At 0.6 µs/game (15 turns), 94% of per-core time is waiting on DRAM latency (3 sequential RAM fetches at ~100 ns each). This is the memory-bandwidth speed of light for sequential random access on M1 Max.

Density evolution confirmed at 1.0x — the bottleneck is transition path enumeration (looping over all 252 x 462 dice paths), not decision computation. The oracle eliminates decision lookup but not the stochastic enumeration itself.

---

## Phase 6: NEON Pipeline Fix + Density Condensation

### NEON `vbslq_u8` Argmax (2.5x oracle precompute)

The original `neon_max_64_argmax` used `vgetq_lane_u32` (scalar lane extraction) + conditional branches per 4-element group — 64 NEON-to-scalar pipeline stalls per invocation. Replaced with a vectorized narrowing chain: `vmovn_u32` x4 -> `vcombine_u16` x2 -> `vmovn_u16` x2 -> `vcombine_u8` x1 -> `vbslq_u8` (16-wide index blend). Zero scalar transfers, zero branches. The argmax pass itself sped up 6.5x (2.88s -> 0.44s overhead).

### Density Transition Condensation (126x oracle density)

Replaced path-by-path transition enumeration (~100K HashMap inserts per state) with probability-array propagation through two `[f64; 252]` arrays (4 KB working set, fully L1-resident).

Old algorithm traces individual paths:
```
for ds0 in 0..252:
  for ds_mid in reroll(ds0):     # ~20 outcomes
    for ds_final in reroll(ds_mid): # ~20 outcomes
      HashMap.insert(score(ds_final), p0 * p1 * p2)
```
~100K HashMap inserts per state.

New algorithm propagates probability arrays level-by-level:
```
cur_probs[252] = initial_probabilities
Pass 1: apply reroll-2 decisions -> nxt_probs[252]
Pass 2: apply reroll-1 decisions -> nxt_probs[252]
Pass 3: score each ds, sort-and-merge
```
~10K array writes per state, no hashing.

Three orthogonal improvements: algorithm reduction (~5x: additive O(252 x K) vs multiplicative O(252 x K x K)), HashMap elimination (~20-50x), cache locality (~2-5x: sequential oracle reads enable hardware prefetching).

### Results

| Subsystem | Before | After | Speedup |
|-----------|--------|-------|---------|
| Oracle precompute (compute only) | 3.92s | 1.54s | 2.5x |
| Oracle overhead vs baseline | 3.77x | 1.4x | 2.7x reduction |
| Oracle MC simulation | 1.57M g/s | 5.6M g/s | 3.6x (measurement conditions) |
| Oracle density evolution | ~378s | 3.0s | 126x |

The previous "speed of light" conclusion was structurally correct but missed that the probability tree has shared internal nodes. Path-by-path visits each leaf independently (O(paths)); level-by-level propagation visits each node once (O(edges)). The analogy: computing network flow by tracing every source-to-sink path vs relaxing edges level-by-level.

---

## Phase 7: Risk-Sensitive Solver (Theta Sweep Infrastructure)

### Three Solver Modes

| Mode | θ range | Stochastic nodes | Runtime |
|------|---------|-----------------|---------|
| EV | θ=0 | Weighted sum | ~1.1s |
| Utility domain | 0<|θ|≤0.15 | Weighted sum (identical to EV) | ~0.49s |
| Log domain (LSE) | |θ|>0.15 | Log-sum-exp | ~2.7s |

The utility domain works directly with U(S) = E[e^(θ * remaining) | S], where stochastic nodes are plain weighted sums (same SIMD-friendly inner loop as EV, no transcendentals). This is possible because e^(θx) is monotonic, so comparing U values gives the same decisions as comparing L values. A single conversion pass L(S) = ln(U(S)) runs after the DP completes (~3 ms for 2M values).

Safety cutoff at |θ| ≤ 0.15: f32 overflow at e^(θ*374) when θ > 0.235; mantissa erasure when e^(θ*100) > 2^24 at θ > 0.166. The 0.15 cutoff covers the entire interesting range where risk preferences produce meaningfully different strategies.

### Sweep Grids

| Grid | Count | Range | Precompute time |
|------|------:|-------|----------------:|
| dense | 19 | [-0.1, +0.1] | ~25s |
| sparse | 17 | [-3.0, +3.0] | ~83s |
| all | 37 | [-3.0, +3.0] | ~2.5 min |
| frontier | 89 | [-1.0, +1.0] | ~12 min |

Grid spacing follows the dimensionless control parameter |θ| * σ_EV: dense sampling in the 0.03 < |θ| < 0.1 range where the Pareto frontier bends sharply; coarse sampling in the degenerate |θ| ≥ 1 region where LSE collapses to max/min.

---

## Current Performance Summary

All figures on quiescent Apple M1 Max, 8 P-cores, RAYON_NUM_THREADS=8.

| Benchmark | Time/Throughput | Distance to Floor |
|-----------|----------------|-------------------|
| EV precompute (θ=0) | 1.10s | 2.2x (DAG barrier sync) |
| Oracle precompute | 1.54s | 1.4x overhead |
| Utility precompute (|θ|≤0.15) | ~0.49s | same class as EV |
| LSE precompute (|θ|>0.15) | ~2.7s | same class as EV |
| MC sequential | ~52K g/s | baseline |
| MC lockstep | ~232K g/s | 2.9x (widget amortization) |
| MC oracle | 5.6M g/s | 1.4x (DRAM random read) |
| Density non-oracle | ~381s | 3.0x (widget + HashMap) |
| Density oracle | 3.0s | 1.5x (DRAM oracle reads) |

Every benchmark is DRAM-latency-bound, not compute- or bandwidth-bound. The 400 GB/s peak bandwidth is irrelevant when access patterns are random at cache-line granularity.

---

## Remaining Opportunities

### AMX / Accelerate Dense MatVec

Replace the sparse CSR dot product (Step 1) with dense matrix-vector multiply: `keep_ev[462] = dense_probs[462x252] x e_prev[252]`. 26.7x more arithmetic (116K vs 4.4K FMAs) but fully vectorizable. The 466 KB dense table exceeds L1 but fits L2. Via AMX (dispatched through `cblas_sgemv`), potentially 10-50x faster than NEON for dense operations. Worth benchmarking.

### GPU Offload (Metal)

Dispatch entire levels to M1 Max GPU (32-core, 10.4 TFLOPS fp32). Unified memory eliminates data transfer bottleneck. Each widget as one threadgroup with 252 threads. At 20% GPU utilization: ~2 TFLOPS = 5x speedup. Most promising path for a large step change.

### Non-Oracle Density Improvement

The non-oracle density (381s) has a 3.0x gap to theoretical floor (~125s). Replacing HashMap with sort-merge (as done for oracle density) would close most of the gap. However, the full Group 6/5/3 widget computation per state is a hard floor without an oracle.

---

## Negative Results

### f16 (Half-Precision)

f16 has ~3.3 significant digits. At 256-point EVs, the ULP is 0.25 points. Accumulation error over ~9.5 FMAs: ~0.75 points. The upper bonus cliff at m=63 makes this intolerable. Not viable.

### Multi-Level Fusion (Groups 5+3)

The dependency chain `e[0] -> keep_ev_5 -> e[1] -> keep_ev_3 -> e[0]` cannot be fused because Group 3's input depends on Group 5's max-over-keeps output. The max-over-keeps operation is not distributive over the CSR multiply.

### Precomputed Unscored-Category Tables

The `is_category_scored` branch test is ~0.5 cycles amortized on Firestorm's aggressive predictor. A lookup table adds memory pressure for negligible benefit.
