# Hardware-Aware Hot Path Optimization

## 1. Optimization History

### 1.1 Original baseline → CSR (5×)

The original C implementation used a dense 3D transition table:

```c
double transition_table[252][32][252];  // 16.3 MB, 79% zeros
```

This exceeded the M1 Max's 12 MB shared L2 cache. Every widget thrashed the cache hierarchy.

**Fix**: Sparse CSR format reduced memory from 16.3 MB to 5.1 MB — fitting in L2. Added OpenMP parallelism.

### 1.2 f32 storage + reachability pruning (1.6×)

Switched `E_table` from f64 to f32, halving the state array from 16 MB to 8 MB (now fits in the ~48 MB SLC). Added reachability pruning (31.8% fewer states).

### 1.3 Keep-multiset dedup (1.4×)

Multiple 5-bit masks can produce the same kept-dice multiset. E.g., dice `[1,1,2,3,4]` with masks `0b00001` and `0b00010` both keep `{1,2,3,4}`.

```
Raw masks per dice set:       31
Unique keeps per dice set:    16.3 avg
KeepTable memory:             51 KB (was 5.1 MB)
```

At 51 KB, the KeepTable fits entirely in L1 cache (128 KB on Apple M1 Firestorm cores).

### 1.4 Keep-EV dedup (9× fewer dot products)

Within one call to `compute_max_ev_for_n_rerolls`, the input `e_prev[252]` is shared across all 252 dice sets. The sparse dot product for keep $k$ depends only on $k$ and `e_prev`, but each keep appears in avg 8.9 dice sets — so the same dot product was computed 8.9× redundantly.

```
Before: 252 dice sets × 16.3 keeps/ds = 4,108 dot products per call
After:  462 unique dot products + 4,108 table lookups per call
```

The 462 keep EVs are stored in a 3.7 KB buffer that fits in L1.

### 1.5 State index layout swap (45× faster Group 6)

```
Before: state_index(m, C) = m × 32768 + C  → successor lookups scattered across 8 MB
After:  state_index(m, C) = C × 64 + m     → 256 bytes contiguous per scored mask
```

Group 6 looks up `sv[state_index(new_up, scored | (1<<c))]`. With the new layout, all 64 upper-score variants of one scored mask fit in 2 L1 cache lines (128 bytes each). Scoring a new category changes the scored mask, but consecutive calls within the same widget share the successor mask, so these 2 cache lines stay hot.

### 1.6 Grouped parallelism (amplifies 1.5)

States grouped by `scored_categories` before rayon dispatch. Each task processes all upper scores for one scored mask sequentially:

```rust
groups.par_iter().for_each(|(scored, ups)| {
    for &up in ups {
        solve_widget(up, scored);  // same successor cache lines reused
    }
});
```

This gives temporal locality on top of spatial locality.

### 1.7 Lower-category successor preloading

For lower categories (6–14), the successor EV `sv[state_index(up, scored|(1<<c))]` is constant across all 252 dice sets (depends only on `up` and `scored`, not the roll). Preloading ~5 values eliminates ~1,255 redundant L1 reads per widget.

### 1.8 Unsafe hot path

- `sv` slice cached once per widget (avoids `StateValues` enum match on every lookup)
- `get_unchecked` in inner loops (indices guaranteed valid by construction)
- `AtomicPtr` + unsafe direct writes (no `par_iter().collect()` allocation)
- `#[inline(always)]` on all hot functions
- `panic = "abort"` eliminates unwinding tables

### 1.9 f32 internal computation

Switched all internal DP accumulation from f64 to f32. Validation:

```
Max absolute difference:  0.000458 points
Max relative difference:  4.05e-6
Game-start EV (f64):      248.439987
Game-start EV (f32):      248.440140
Decision impact:          22 ordering flips out of 1.43M states,
                          all below 0.0002 pt EV loss
```

Enables GPU path: M1 Max GPU runs f32 at 10.4 TFLOPS vs f64 at 325 GFLOPS.

## 2. Target Hardware: Apple M1 Max

All optimization decisions are informed by the specific microarchitecture. This section provides a detailed hardware reference.

### 2.1 ISA: ARMv8.5-A (AArch64)

The M1 Max implements ARMv8.5-A — a 64-bit ISA with fixed-width 32-bit instruction encoding. Key features:

- **Registers**: 31 general-purpose 64-bit registers (X0–X30), plus SP, ZR, and a separate PC
- **Condition flags**: NZCV in PSTATE
- **ARMv8.1-A LSE**: Hardware atomics (LDADD, CAS, SWP) replacing LL/SC loops. Our `AtomicPtr` usage in Phase 2 benefits from this — though we only use relaxed loads (no contention, disjoint indices).
- **ARMv8.3-A PAC**: Pointer Authentication. Relevant because `panic = "abort"` eliminates unwinding tables that would otherwise interact with PAC-signed return addresses.
- **ARMv8.5-A BTI**: Branch Target Identification. Our `#[inline(always)]` annotations eliminate indirect branches in the hot path, sidestepping BTI overhead.

### 2.2 SIMD: NEON (Advanced SIMD)

- **32 × 128-bit vector registers** (V0–V31), aliasable as 64-bit D-regs or scalar H/S/D
- **Data types**: int8/16/32/64, fp16/fp32/fp64
- **No SVE/SVE2** — Apple chose not to implement scalable vectors. All vectorization targets fixed 128-bit NEON.
- **Key instruction**: `FMLA` (fused multiply-accumulate) — 4-cycle latency, 2/cycle throughput at fp64, 4/cycle at fp32

This is the fundamental constraint on our sparse dot product: NEON has no hardware gather instruction (`VGATHER` equivalent). Each `e_prev[cols[k]]` in the CSR inner loop is a scalar load, defeating auto-vectorization.

**NEON throughput on Firestorm:**
- 4 NEON execution pipes → 4 × 128-bit SIMD ops/cycle sustained
- At fp32: 4 pipes × 4 lanes × 2 (fma = mul + add) × 3.228 GHz ≈ **103 GFLOPS/core**
- At fp64: 4 pipes × 2 lanes × 2 × 3.228 GHz ≈ **52 GFLOPS/core**
- To fully utilize: interleave ≥8 independent FMLA chains to hide the 4-cycle latency

### 2.3 AMX (Apple Matrix Extensions)

Undocumented proprietary coprocessor with a hidden register file separate from NEON. Accessed via `AMX_LDX`, `AMX_FMA32`, etc. through the Accelerate framework (vDSP, BLAS). Responsible for the bulk of CPU matrix throughput.

**Relevance to solver**: The KeepTable CSR dot product is a SpMV operation. If reformulated as a dense MatVec (462 × 252 × f32), it could potentially be dispatched via `cblas_sgemv` → AMX. This would bypass the NEON gather limitation entirely. See §4.1.

### 2.4 Firestorm P-core Microarchitecture

The M1 Max has 8 Firestorm (performance) cores organized in 2 clusters of 4.

| Parameter | Value | Solver implication |
|-----------|-------|--------------------|
| Decode width | 8-wide | Can decode the entire Group 6 inner loop body (~6 instructions) in one cycle |
| ROB size | ~630 entries | Enormous — can sustain ~45 in-flight operations, hiding memory latency |
| Issue width | ~13 execution ports | Far wider than our ILP; not the bottleneck |
| Pipeline depth | ~13 stages | Relatively shallow → low branch mispredict penalty |
| Clock | ~3.228 GHz (fixed) | No turbo; performance is deterministic |
| Integer multiply | 3-cycle latency, pipelined | Relevant for `state_index = scored * 64 + up` (compiled to shift+add) |
| Load ports | 3 load + 2 store per cycle | 128-bit load width on NEON |
| ALU ports | 6 ALU, 4 FP/NEON, 2 branch | |
| L1I$ | 192 KB, 6-way | Extraordinarily large — our entire hot path (~2 KB code) fits trivially |
| L1D$ | 128 KB, 8-way | KeepTable (51 KB) + keep_ev (1.8 KB) + ping-pong buffers (2 KB) + lower_succ_ev (60 B) fit together |
| L2$ | 12 MB shared per cluster | 4 cores/cluster. 8 MB state array fits in ~67% of one cluster's L2 |

**Cache line size: 128 bytes** (not 64 like x86). This affects:
- The `state_index = C × 64 + m` layout puts all 64 upper-score variants (256 bytes) in exactly 2 cache lines. A single cache line covers 32 upper-score values.
- Successor lookups in Group 6 hit at most 2 cache lines per scored mask — both almost certainly hot.
- The `keep_ev[462]` array (1,848 bytes) spans 15 cache lines. During Step 2 lookups, L1 hits are guaranteed (written moments earlier in Step 1).
- False sharing boundary is 128 bytes. Our parallel writes use disjoint `state_index` values. Adjacent indices within the same scored mask differ by 1 (consecutive upper scores within a 256-byte block), but grouped parallelism assigns all upper scores for one scored mask to the same thread, so no false sharing occurs.

**Page size: 16 KB** (not 4 KB like x86 Linux). The 8 MB state array spans 512 pages. With the M1's large TLB, this is well within coverage — no TLB pressure.

### 2.5 Icestorm E-cores

2 Icestorm (efficiency) cores at ~2.064 GHz with 4-wide decode, ~100-entry ROB, 64 KB L1D$, shared 4 MB L2. Roughly Cortex-A75 class.

**Solver policy**: `RAYON_NUM_THREADS=8` excludes E-cores entirely. Testing showed that including E-cores (10 threads) reduces throughput — the E-cores are ~3× slower per-widget and create stragglers that delay level completion. Rayon's work-stealing mitigates this somewhat, but the asymmetry is too large.

### 2.6 Memory Hierarchy for Solver Data Structures

| Data structure | Size | Cache level | Access pattern |
|---------------|------|-------------|----------------|
| `e[0]`, `e[1]` (ping-pong) | 2 × 1,008 B | L1D$ | Sequential write (Step 1), random read (Step 2 gather) |
| `keep_ev[462]` | 1,848 B | L1D$ | Sequential write, then indexed read |
| `lower_succ_ev[15]` | 60 B | L1D$ (registers) | Written once, read 252 times |
| `KeepTable` (CSR) | 51 KB | L1D$ | Sequential scan of vals/cols per row |
| `precomputed_scores[252][15]` | 15,120 B | L1D$ | Sequential scan, one row per ds_i |
| `dice_set_probabilities[252]` | 2,016 B | L1D$ | Sequential read (Group 1 only) |
| `state_values` (E_table) | 8 MB | L2$ / SLC | Random access (Group 6 upper cats), semi-random (parallel workers) |
| `unique_keep_ids[252][31]` | 31,248 B | L1D$ | Sequential per dice set |

**Total L1 working set per widget**: ~105 KB — fits within the 128 KB L1D$. This is why Steps 1.3 (keep-multiset dedup) and 1.4 (keep-EV dedup) were so effective: they collapsed the working set from L2 to L1.

### 2.7 Memory Ordering

ARM uses a weakly ordered memory model. Loads and stores can be reordered with each other (no Total Store Order like x86). Our Phase 2 parallel writes use `AtomicPtr` with `Ordering::Relaxed` — this is correct because:

1. Each thread writes to a unique `state_index` (no data races)
2. Level barriers between `num_scored` iterations are enforced by rayon's `par_iter().for_each()` (implicit join before next level)
3. Successor reads within a level always access states from the *previous* level (higher `|C|`), which completed before the current level started

No explicit barriers (`DMB`, `DSB`) are needed beyond what rayon provides.

### 2.8 Compiler Flags

```
-mcpu=apple-m1        # implies armv8.5-a+crypto+fp16+dotprod+rcpc
target-cpu=native     # in .cargo/config.toml — same effect for rustc/LLVM
opt-level = 3         # maximum optimization
lto = "fat"           # cross-crate inlining (critical for widget_solver → types)
codegen-units = 1     # maximum inlining opportunity
panic = "abort"       # eliminates unwinding tables, improves I$ density
```

The `target-cpu=native` flag enables the full Firestorm instruction set including NEON `FMLA` scheduling. The keep-EV sequential accumulation loop (Step 1) is the prime candidate for NEON `FMLA` auto-vectorization — but the gather load (`e_prev[cols[k]]`) prevents it.

## 3. Measured Performance

### 3.1 Current breakdown

```
Wall clock:           2.3s
Sum widget time:      20.3s (across 8 cores)
  Group 6  (sv):       3.5s  (17%)
  Group 53 (dot):     16.5s  (81%)
  Group 1  (sum):      0.3s  (2%)
Parallel efficiency:  96% of 8 cores
Per widget:           14.2 µs = ~45,000 cycles @ 3.228 GHz
```

### 3.2 Theoretical floor

| Component | Operations | Min cycles | Notes |
|-----------|-----------|------------|-------|
| Group 6 | 252 × ~7.5 categories × (score read + sv read + add + cmp) | ~6,300 | 2 L1 loads + 1 add + 1 cmp per category per ds |
| Groups 5+3 Step 1 | 2 × 4,368 FMAs | ~4,400 | At 4 fp32 FMAs/cycle (NEON), if fully vectorizable |
| Groups 5+3 Step 2 | 2 × 4,108 comparisons | ~2,100 | Max-reduction over ~16.3 keeps per ds |
| Group 1 | 252 multiply-accumulates | ~130 | 4-wide NEON FMA, trivially vectorizable |
| **Total** | | **~13,000** | Assuming perfect NEON utilization |

The gap between ~13,000 (NEON-vectorized floor) and ~24,000 (our previous estimate) is the cost of scalar gather loads. The gap between ~24,000 and ~45,000 (measured) is pipeline overhead, branch mispredicts, and residual cache misses.

At the NEON-vectorized floor: $1.43\text{M} \times 13\text{K} / 8 / 3.228\text{ GHz} = 0.72\text{s}$

At the current measured floor: $1.43\text{M} \times 24\text{K} / 8 / 3.228\text{ GHz} = 1.33\text{s}$

**Current: 2.3s → 58% of achievable peak (with gather), 31% of theoretical NEON peak (without gather).**

### 3.3 Bottleneck analysis: the gather problem

The inner loop of Step 1 in Groups 5/3:

```rust
for k in start..end {
    ev += vals[k] * e_prev[cols[k]];  // cols[k] is indirect/scattered
}
```

`cols[k]` is a random index into `e_prev[252]` (1,008 bytes). NEON has no hardware gather instruction. Each `e_prev[cols[k]]` requires:

1. Load `cols[k]` from the CSR column array (sequential — vectorizable)
2. Use it as an address to load `e_prev[cols[k]]` (scalar — cannot vectorize)
3. Multiply by `vals[k]` and accumulate

The scalar load in step 2 serializes the loop. Even though `e_prev` fits in L1 (1 KB), the lack of gather means we get ~1 FMA/cycle instead of ~4 FMA/cycle. This is a **4× penalty** on what should be a trivially NEON-friendly computation.

The Firestorm core has 3 load ports, so up to 3 independent scalar loads can execute per cycle. With sufficient loop unrolling and independent accumulator chains, we could approach 3 FMA/cycle — a 3× improvement over the current ~1 FMA/cycle. However, the LLVM auto-vectorizer does not produce this pattern for indirect-indexed loops.

**Comparison to x86**: Intel AVX2/AVX-512 have `VGATHER` instructions that can load 4–16 scattered values in one instruction (~5-12 cycle latency). On a Zen 4 or Sapphire Rapids machine, this loop would vectorize trivially. ARM's lack of gather is the single largest performance gap.

## 4. Remaining Optimization Opportunities

Each opportunity is analyzed against the Firestorm microarchitecture.

### 4.1 Dense MatVec via AMX / Accelerate

**Idea**: Replace the sparse CSR dot product (Step 1) with a dense matrix-vector multiply: `keep_ev[462] = dense_probs[462×252] × e_prev[252]`.

**Arithmetic cost**: 462 × 252 = 116,424 FMAs (vs ~4,368 sparse FMAs). That's 26.7× more arithmetic.

**But**: The dense multiply is fully vectorizable. Via NEON alone at 4 fp32 FMAs/cycle: 116,424 / 4 = 29,106 cycles per call, ×2 calls = 58,212 cycles. Via AMX (if dispatched through `cblas_sgemv`): potentially 10-50× faster than NEON for dense MatVec, bringing it to ~1,000-5,000 cycles.

**Memory**: Dense table = 462 × 252 × 4B = **466 KB**. This exceeds L1D$ (128 KB) but fits in L2 (12 MB). At ~12-cycle L2 latency, streaming through 466 KB costs ~3,640 cache line loads × 12 cycles = ~44K cycles overhead — comparable to the computation itself.

**Tradeoff**: The question is whether 116K fully vectorized FMAs at L2 bandwidth beat 4,368 scalar-gather FMAs at L1 latency. Key factors:

- L2 bandwidth on Firestorm: one 128-byte line per ~12 cycles → ~34 GB/s per core. Streaming 466 KB takes ~14 µs — comparable to the entire widget time.
- AMX could change the calculus entirely if `cblas_sgemv` dispatches to it. AMX has its own register file and data path; it can sustain much higher throughput than NEON for dense operations.
- The dense table is **constant across all widgets** — it never changes. On the first widget, it loads into L2; subsequent widgets would hit L2 directly. With 8 cores sharing 12 MB L2, 466 KB is 3.9% — affordable.

**Verdict**: Worth benchmarking. The AMX path (via `cblas_sgemv`) is the most promising; pure NEON dense is marginal.

### 4.2 Manual loop unrolling with multiple accumulators

**Idea**: Hand-unroll the sparse CSR inner loop with 3–4 independent accumulator chains to exploit Firestorm's 3 load ports and 4 FP pipes:

```rust
// Pseudocode: 4-way unrolled sparse dot product
let (mut ev0, mut ev1, mut ev2, mut ev3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
let chunks = (end - start) / 4;
for i in 0..chunks {
    let k = start + i * 4;
    ev0 += vals[k+0] * e_prev[cols[k+0]];
    ev1 += vals[k+1] * e_prev[cols[k+1]];
    ev2 += vals[k+2] * e_prev[cols[k+2]];
    ev3 += vals[k+3] * e_prev[cols[k+3]];
}
ev = ev0 + ev1 + ev2 + ev3;
// + scalar tail for remainder
```

Firestorm has 3 load ports. With 4-way unrolling, 3 of the 4 `e_prev[cols[k+i]]` loads can issue per cycle. The `vals[k+i]` loads are sequential and can be prefetched. The FMAs use the 4 FP pipes with 4-cycle latency — with 4 independent chains, the pipeline is fully filled.

**Expected gain**: ~2-3× on the Step 1 inner loop, or ~1.3-1.5× on total Groups 5+3 time (Step 1 is ~60% of Groups 5+3; Step 2 is the rest).

**Risk**: LLVM may already partially unroll this. Need to inspect codegen (`--emit=asm`) to see if the auto-vectorizer is blocking manual unrolling. The `#[inline(always)]` + `get_unchecked` already remove the main barriers.

### 4.3 Batched widget dispatch to GPU (Metal)

**Idea**: Dispatch entire levels to the M1 Max GPU (32-core, 10.4 TFLOPS fp32).

**Architecture fit**: Each level has ~1,400–6,000 widgets (depending on `|C|`). Each widget is independent. The GPU can run one threadgroup per widget, with 252 threads per threadgroup handling the dice-set parallelism within Groups 6 and 5/3 Step 2.

**Data layout**:
- `state_values[8 MB]`: read-only texture or buffer, resident in unified memory (no copy needed on M1)
- `KeepTable CSR [51 KB]`: constant buffer, trivially fits in GPU shared memory
- `e[0]`, `e[1]` per widget: threadgroup shared memory (2 × 1 KB = 2 KB per threadgroup, well within the 32 KB limit)
- `keep_ev[462]`: threadgroup shared memory (1.8 KB)

**Challenges**:
1. **Step 1 (sparse dot product)**: Same gather problem as CPU — Metal has no native scatter-gather. Would need to use the dense MatVec approach (§4.1) on GPU, where it would be trivially parallelized across the 32 GPU cores.
2. **Group 6 (state value lookups)**: Random access into 8 MB — this is a texture fetch pattern. The GPU texture cache handles random access well, but 8 MB exceeds the per-CU L1 cache.
3. **Dispatch overhead**: Metal command buffer submission is ~2-5 µs. With ~1,400 widgets per level and 15 levels, that's 15 dispatches — negligible overhead.
4. **Writeback**: Results need to be written back to `state_values`. With unified memory, this is a simple store — no PCIe transfer.

**Expected gain**: The GPU has 32 cores × ~3.2 TFLOPS = 10.4 TFLOPS fp32. The CPU achieves ~0.4 TFLOPS across 8 cores. If the GPU can sustain even 20% utilization on this workload, that's ~2 TFLOPS — a **5× speedup**. Wall clock would drop from 2.3s to ~0.5s.

**Verdict**: Most promising path for a large speedup. The unified memory architecture of M1 eliminates the data transfer bottleneck that would kill GPU offload on discrete GPU systems.

### 4.4 SIMD-parallel Group 6 (SoA restructure)

**Idea**: Restructure Group 6 to process 4 dice sets simultaneously in one NEON register.

**Problem**: The successor EV load is again a gather — `state_index(new_up, scored|(1<<c))` differs per dice set because `new_up` depends on the score. For lower categories this doesn't apply (successor is constant — already preloaded). For upper categories, the 4 `new_up` values are generally different, requiring 4 scalar loads.

**Partial benefit**: The `precomputed_scores` access IS sequential (ds_i varies, category is fixed). Vectorizing just the score load + add + compare saves ~30% of Group 6 cycles.

**Expected gain**: ~1.1-1.2× on Group 6 time (17% of total) → ~1.02-1.04× overall. Not worth the complexity.

### 4.5 Precomputed unscored-category tables

**Idea**: For each of the $2^{15} = 32{,}768$ scored bitmasks, precompute a compact list of unscored category indices.

**Verdict**: Not worth it. The branch predictor handles the `is_category_scored` pattern well, and the table adds memory pressure.

### 4.6 Multi-level fusion (Groups 5+3)

The dependency chain is: `e[0] → keep_ev_5 → e[1] → keep_ev_3 → e[0]`. Steps 1a and 1b are identical CSR operations with different input vectors. They cannot be fused because Step 1b depends on the output of Steps 1a+2a.

**Verdict**: Not fusible without fundamentally changing the algorithm.

### 4.7 f16 (half-precision) computation

f16 has ~3.3 significant decimal digits (10-bit mantissa). Game EVs range 0–400; an f16 ULP at 256 is 0.25 points. The 462 keep EVs accumulate ~9.5 FMAs on average. Rounding error accumulates as $O(\sqrt{n}) \cdot \text{ULP}$ ≈ $3 \times 0.25 = 0.75$ points — unacceptable.

**Verdict**: f16 precision is insufficient for the DP accumulation.

## 5. Data Flow Summary

```
Phase 0: Build tables (~1 ms)
  ┌──────────────────────────────────────────────────┐
  │ R_{5,6}[252]  scores[252][15]  KeepTable(CSR)   │
  │ P(⊥→r)[252]  reachable[64][64]  terminals       │
  └──────────────────────────────────────────────────┘
                          │
Phase 2: Backward induction (~2.3s)
  for level = 14 → 0:
    ┌──────────────────────────────────────────────┐
    │ Group states by scored_categories bitmask    │
    │ par_iter over groups:                        │
    │   for each upper_score:                      │
    │     SOLVE_WIDGET:                            │
    │       Group 6: 252 × max(score + sv[succ])   │  ← 17% time
    │       Group 5: 462 dot prods + 252 max       │  ← 81% time
    │       Group 3: 462 dot prods + 252 max       │  (combined 5+3)
    │       Group 1: 252 weighted sum              │  ← 2% time
    │     → write E_table[state_index(up, scored)] │
    └──────────────────────────────────────────────┘
                          │
Output: E_table[2,097,152] → 8 MB binary file
  Game-start EV: E(0, ∅) ≈ 248.44 points
```
