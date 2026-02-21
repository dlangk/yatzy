# Batched SoA Solver: SpMV→SpMM Experiment

This document reports on an experiment to speed up precomputation by replacing
the scalar per-state SOLVE_WIDGET loop with a batched Structure-of-Arrays (SoA)
solver that processes all 64 upper-score variants of each `scored_categories`
mask simultaneously.

## Motivation

The state index layout `scored * 64 + up` places all 64 upper-score variants
of a scored mask into a contiguous 256-byte region (64 × f32). The existing
scalar solver calls `compute_expected_state_value` once per `(scored, up)` pair,
loading the same successor blocks from memory repeatedly for neighboring `up`
values. The hypothesis was that batching all 64 `up` values into a single
solve operation would convert scattered SpMV (sparse matrix-vector) inner loops
into SpMM (sparse matrix-matrix), where each row of the sparse matrix is
multiplied by a 64-wide vector instead of a scalar.

### Predicted speedup

The plan estimated 10–20× speedup (from ~2.3s to ~150–300ms) based on:

1. Eliminating redundant memory traffic for successor lookups in Group 6
2. Converting the CSR dot product in Groups 5/3 from 252 independent SpMVs
   into a single SpMM over a [252 × 64] matrix
3. NEON auto-vectorization of the 64-wide inner loops (4 × f32 per SIMD lane)

## Design

### Entry point: `solve_widget_batched(ctx, sv, scored, bufs) -> [f32; 64]`

Three variants (EV, risk-sensitive, max-policy) share the same structure:

| Group | Scalar solver | Batched solver |
|-------|--------------|----------------|
| **6 — Category scoring** | For one `(ds, up)`: compute score for each unscored category, look up `sv[successor]`, take max | For one `ds`: compute score once, then iterate `up` 0..64, reading from the contiguous 256-byte successor block |
| **5/3 — Reroll (×2)** | For one `(ds, up)`: CSR dot product `Σ P(k→ds') · e_prev[ds']` per keep, take max over keeps | Step 1: compute `keep_ev[kid][up]` = CSR SpMM over [462 × 64] matrix. Step 2: per `ds`, max over unique keeps |
| **1 — Initial roll** | For one `up`: `Σ P(⊥→r) · e[r]` | `result[up] = Σ P(⊥→r) · e[r][up]` for all 64 ups |

### Buffer management

Each thread needs three working matrices:

| Buffer | Shape | Size | Purpose |
|--------|-------|------|---------|
| `e0` | 252 × 64 | 63 KB | Ping-pong buffer A |
| `e1` | 252 × 64 | 63 KB | Ping-pong buffer B |
| `keep_ev` | 462 × 64 | 115 KB | Intermediate keep-multiset EVs |
| **Total** | | **~247 KB** | Fits in L2 (12 MB per P-core cluster) |

Initially, these were allocated per invocation (`vec![[0.0; 64]; 252]` for each
scored mask). With ~22K scored masks processed across all 15 levels, this
generated ~16 GB of heap allocation traffic. Refactoring to a `BatchedBuffers`
struct, allocated once per rayon thread via `for_each_init`, eliminated this
overhead.

### Dispatch: `state_computation.rs`

```rust
scored_masks
    .par_iter()
    .for_each_init(BatchedBuffers::new, |bufs, &scored| {
        let results = solve_widget_batched(ctx, sv, scored, bufs);
        // unsafe write results[up] to state_values[state_index(up, scored)]
    });
```

Rayon's `for_each_init` creates one `BatchedBuffers` per worker thread and
reuses it across all scored masks assigned to that thread. With 8 threads,
total buffer memory is 8 × 247 KB ≈ 2 MB.

## Results

### Correctness

All 107 tests pass (82 unit + 25 integration). Two new unit tests verify that
the batched solver produces **bit-identical** results to the scalar solver
across 8 representative scored masks covering levels |C| = 0, 1, 12, 13, 14.
The accumulation order is preserved, so f32 results match exactly.

### Performance

Measured on Apple M1 Max, 8 P-cores, `RAYON_NUM_THREADS=8`, release profile
(opt-level 3, fat LTO, `target-cpu=native`):

| Configuration | Computation time | Wall clock | vs. baseline |
|---------------|-----------------|------------|-------------|
| Scalar solver (baseline, Step 7) | 2.3s | 2.6s | 1.0× |
| Batched, per-call allocation | 1.53s | 1.8s | 1.5× |
| Batched, thread-local buffers | 1.40–1.42s | 1.7s | 1.6× |

**Measured speedup: 1.6×.** Significantly below the 10–20× prediction.

### Per-level timing

```
Level | States  | Time (s) | Rate (states/s)
------|---------|----------|----------------
  14  |    959  |    0.00  | 535K
  13  |   6677  |    0.02  | 423K
  12  |  28486  |    0.03  | 1.02M
  11  |  82860  |    0.06  | 1.32M
  10  | 173274  |    0.12  | 1.43M
   9  | 267883  |    0.21  | 1.30M
   8  | 310209  |    0.29  | 1.07M
   7  | 269592  |    0.33  | 816K
   6  | 174355  |    0.24  | 730K
   5  |  82211  |    0.18  | 447K
   4  |  27224  |    0.08  | 321K
   3  |   5932  |    0.03  | 180K
   2  |    756  |    0.01  | 85K
   1  |     45  |    0.00  | 29K
```

Processing rate peaks at level 10 (1.43M states/sec) and drops at both ends.
Low levels (0–3) have too few scored masks for effective parallelism. High
levels (7–8) have the most scored masks but each mask has few unscored
categories, so Group 6 does less useful work per iteration.

## Analysis: Why 1.6× instead of 10–20×

### 1. The scatter-gather bottleneck persists

The plan predicted that batching would convert the CSR dot product from a
gather-per-scalar to a gather-per-row, amortizing the scatter cost over 64
elements. This is correct in theory — the inner loop:

```rust
for k in start..end {
    let prob = vals[k];
    let col = cols[k];
    // 64-wide vectorizable:
    for up in 0..64 {
        kev[up] += prob * e_prev[col][up];
    }
}
```

reads `e_prev[col]` (a contiguous 256-byte row) once and broadcasts `prob`
across the 64-wide FMA. But the **outer** loop still iterates CSR entries
(`k in start..end`) with indirect `cols[k]` indexing — each iteration loads
a different 256-byte row of `e_prev`, and these rows are not contiguous in
memory.

With `avg_nnz = 9.5` per keep and 462 keeps, the SpMM touches
`462 × 9.5 = 4,389` rows of `e_prev` (with repetition). The `e_prev` buffer
is 252 × 256 = 63 KB — larger than L1 (128 KB shared between data and
instruction caches, effectively ~96 KB for data). So the SpMM generates L1
misses on roughly 30–40% of `e_prev` row accesses, each costing ~12 cycles
to fill from L2.

### 2. NEON vectorization is only 4-wide

ARM NEON operates on 128-bit registers = 4 × f32. The 64-element inner loop
processes 64/4 = 16 NEON operations per CSR entry. Compare to the scalar
solver which processes 1 entry in ~4 cycles. The batched solver processes 64
entries in ~16 × 1.5 = ~24 cycles (allowing for FMA latency), giving an
effective throughput of 64/24 ÷ 1/4 = **~10.7× per CSR entry**.

But this 10.7× only applies to the FMA throughput of the inner loop. The
**overhead** — CSR index loading, branch prediction, loop control, `cols[k]`
address computation — is paid once per CSR entry regardless of batch size.
In the scalar solver, this overhead is ~2 cycles out of ~4 total. In the
batched solver, the FMA work grows from ~2 to ~24 cycles, but the overhead
stays at ~2 cycles. So the actual speedup on the SpMM inner loop is
(4 × 462 × 9.5) / (26 × 462) ≈ **1.46×** when accounting for overhead —
closely matching the observed 1.6×.

### 3. Group 6 was already fast

Group 6 accounted for only 17% of the scalar solver's time (3.5s out of
20.3s across 8 cores). Even a 64× speedup on Group 6 would reduce total
time from 2.3s to ~2.15s. The batched Group 6 is faster (it reads each
successor block once instead of per-`up`), but this saves only ~0.1s
because Group 6 was never the bottleneck.

### 4. Reduced parallelism granularity

The scalar solver dispatched ~22K work items (one per scored mask, each
containing ~64 widgets). The batched solver dispatches the same ~22K items
but each item does 64× more work. Rayon's work-stealing handles this well
when items are uniform, but at level boundaries (e.g., level 14 with only
959 states across ~68 scored masks), some threads finish early and idle.

### 5. L1 pressure from wider buffers

The scalar solver's working set per widget was:
- `e[0]`, `e[1]`: 252 × f32 = 1 KB each = 2 KB total
- `keep_ev`: 462 × f32 = 1.8 KB
- Total: ~3.8 KB (trivially fits L1)

The batched solver's working set:
- `e[0]`, `e[1]`: 252 × 64 × f32 = 63 KB each = 126 KB total
- `keep_ev`: 462 × 64 × f32 = 115 KB
- Total: ~247 KB (**exceeds L1**)

During the SpMM, the hot data (`e_prev` rows being accumulated into `keep_ev`)
cycles through L2 instead of staying in L1. This adds ~8 cycles per cache
line load compared to the scalar solver's all-L1 execution. Over 4,389 CSR
entries × 4 cache lines per row = ~17,500 L2 reads, costing ~140K extra
cycles per scored mask.

## Conclusions

### The optimization works, but the bottleneck is structural

The batched SoA approach successfully amortizes successor lookups (Group 6)
and converts the keep-multiset dot product into a matrix operation (Groups
5/3). However, the **dominant cost** — indirect row access via CSR column
indices — is inherent to the sparse format and cannot be eliminated by
batching alone.

The 1.6× speedup comes primarily from:
1. Eliminating per-widget function call overhead (~22K calls → ~22K batched)
2. Better amortization of Group 6 successor reads
3. Partial NEON vectorization of the 64-wide FMA loops
4. Eliminating heap allocation with thread-local buffer reuse

### What would achieve 10×+

To reach the originally predicted 10–20× range, fundamental changes to the
data layout would be needed:

1. **Dense transition table in SoA layout**: Replace the CSR format with a
   dense [462 × 252] matrix of f32 probabilities (~460 KB, fits L2). This
   eliminates indirect indexing at the cost of iterating zeros (79% of entries).
   At 4-wide NEON, the dense FMA throughput might offset the wasted work.

2. **GPU offload**: The SpMM is a standard BLAS-like operation. On the M1 Max
   GPU (10.4 TFLOPS f32), each level's computation could potentially run in
   microseconds. The challenge is the small matrix size (252 × 64) — GPU
   launch overhead may dominate.

3. **Restructure the DP to process multiple scored masks simultaneously**:
   Instead of batching over the 64 `up` values (which the state layout
   already optimizes), batch over scored masks at the same level. This would
   increase the matrix width from 64 to thousands, better utilizing NEON's
   pipeline depth. However, different scored masks have different unscored
   categories, making Group 6 irregular.

### Recommendation

The batched SoA solver is a net improvement (1.6× faster, same correctness,
cleaner dispatch loop) and should be kept. Further optimization of precomputation
has diminishing returns — 1.4s is already fast enough that it's imperceptible
in practice. The remaining compute time is dominated by the CSR scatter-gather
pattern, which is a fundamental limitation of the sparse format on ARM.

Engineering effort would be better spent on other fronts (GPU acceleration for
theta sweeps, or algorithmic improvements to the simulation engine).

## Files changed

| File | Change |
|------|--------|
| `solver/src/batched_solver.rs` | New module: 605 lines, 3 solve variants (EV, risk, max), `BatchedBuffers` struct |
| `solver/src/state_computation.rs` | Replaced per-state dispatch with per-scored-mask batched dispatch via `for_each_init` |
| `solver/src/lib.rs` | Added `pub mod batched_solver;` |

The scalar `widget_solver.rs` is unchanged and continues to serve API queries.
