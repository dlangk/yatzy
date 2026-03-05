:::section{#solver}

## The Solver

We are now ready to take on the ~1.43M states, connected through widgets and layered into 16 layers. Since each layer only depends on the previous layer, we can solve the entire game by working backward: start at layer 16 (where the answer is trivial: just add the bonus if you earned it), then solve layer 15 using those answers, then layer 14, and so on.

:::html
<div id="chart-backward-cascade"></div>
:::

This section explains what happens inside each state: how the solver [evaluates a single turn](https://github.com/dlangk/yatzy/blob/main/solver/src/widget_solver.rs#L430) and picks the best play.

### Scoring the Final Roll

After the last reroll, the solver has a final roll of five dice and must choose which category to score. Each unscored category gives an immediate score plus a future value (the expected score from the resulting state, already computed in a later layer). The solver tries every option and picks the best:

```
for each final roll (252 possibilities):
    best = -∞
    for each unscored category:
        value = score(roll, category) + future_value(next state)
        best = max(best, value)
    scoring_ev[roll] = best
```

This is the innermost decision. Everything else in the turn feeds into it.

### Choosing What to Keep

Before that final roll, the player chose which dice to hold. Each possible hold leads to a probability distribution over rerolls. The expected value of a hold is the weighted sum over those outcomes, using the scoring EVs just computed:

```
for each roll (252):
    best = -∞
    for each way to hold dice:
        ev = Σ P(hold → reroll) × scoring_ev[reroll]
        best = max(best, ev)
    keep_ev[roll] = best
```

The solver runs this twice: once for the second reroll (using scoring EVs as input) and once for the first reroll (using second-reroll EVs as input). Two identical passes, reading from one buffer and writing to the other.

### Transition Probabilities

Where does P(hold &rarr; reroll) come from? When you hold some dice and reroll the rest, each free die lands on 1 through 6 independently. The probability of reaching a specific outcome depends on how many ways the free dice can produce the required values.

Say you hold (3, 3) and reroll 3 dice. To end up with (1, 3, 3, 4, 5), the free dice need to show {1, 4, 5}. Three distinct values across three dice: 3! = 6 arrangements out of 6<sup>3</sup> = 216, so P = 6/216. To end up with (3, 3, 3, 4, 4) instead, the free dice need {3, 4, 4}: only 3!/2! = 3 arrangements, so P = 3/216.

In general, the probability is the multinomial coefficient (the number of distinguishable arrangements of the free dice) divided by 6<sup><var>k</var></sup>. These probabilities are precomputed once into a [sparse table](https://github.com/dlangk/yatzy/blob/main/solver/src/phase0_tables.rs#L82): 462 keeps &times; 252 outcomes, with only [4,368 non-zero entries](https://github.com/dlangk/yatzy/blob/main/solver/src/types.rs#L25-L40). The solver reads from this table in every widget evaluation, never recomputing it.

### The Keep Shortcut

Here is the key algorithmic insight. Many different rolls share the same hold. "Keep two Threes" from (1,3,3,5,6) and from (2,3,3,4,4) produces the same reroll distribution: two free dice, each uniform over 1 through 6. Instead of recomputing each hold's expected value for every roll, the solver computes it once for each of the 462 unique hold-multisets, then distributes:

```
Step 1: Compute EV for each unique hold (462)
    for each unique hold h:
        hold_ev[h] = Σ P(h → reroll) × prev_ev[reroll]

Step 2: For each roll, pick the best hold (252 lookups)
    for each roll r:
        keep_ev[r] = max over valid holds h of r: hold_ev[h]
```

Step 1 does the expensive probability work once. Step 2 is a cheap lookup. This separation reduces the inner loop work by an order of magnitude.

### The First Roll

At the start of a turn, before any dice are thrown, the expected value is just the weighted average over all 252 possible initial rolls:

```
E(state) = Σ P(initial roll = r) × keep_ev[r]   /   7776
```

One number comes out: the expected score from this state under optimal play.

### The Backward Sweep

All 1.43 million widgets are [solved layer by layer](https://github.com/dlangk/yatzy/blob/main/solver/src/state_computation.rs#L251), starting from the terminal states (layer 16, where the game is over) and working backward to layer 1. Each layer depends only on the next, and states within a layer are independent of each other. This means every state in a layer can be solved in parallel across all CPU cores. The full sweep completes in about one second.

:::html
<div class="chart-container" id="chart-widget-interactive">
  <div class="chart-controls">
    <select id="widget-scenario-select" class="chart-select">
      <option value="0">Scenario: Keep triple or chase straight?</option>
    </select>
    <button class="chart-btn" id="widget-solver-btn">Let Solver Play</button>
    <button class="chart-btn" id="widget-reset-btn">Reset</button>
  </div>
  <div id="widget-flow"></div>
  <div id="widget-fan-panel" class="widget-math"></div>
  <p class="chart-caption">Step through a concrete scenario to see how the widget solver evaluates keeps, rerolls, and scoring.</p>
</div>
:::


:::insight
**462 shared holds make the inner loop tractable.** Without the keep shortcut, each of the 252 rolls would need its own set of sparse dot products against the reroll distribution. Sharing the computation across the 462 unique hold-multisets reduces the work by an order of magnitude, turning a minutes-long solve into a one-second sweep.
:::

:::math

The solver evaluates the Bellman equation in four stages. Terminal states have a known value (just the bonus check). For non-terminal states, the evaluation proceeds bottom-up through the widget layers.

**Terminal value (layer 16):**

:::equation
<var>V</var>(<var>u</var>, <var>S</var>) = [<var>u</var> &ge; 63] &middot; 50
:::

**Best category for a final roll:**

:::equation
<var>Q</var><sub>cat</sub>(<var>u</var>, <var>S</var>, <var>d</var>) =
max<sub><var>c</var> &notin; <var>S</var></sub>
&big;( score(<var>d</var>, <var>c</var>) + <var>V</var>(<var>u</var>&prime;, <var>S</var> &cup; {<var>c</var>}) &big;)
:::

**Best keep at reroll stage <var>t</var> (ping-pong):**

:::equation
<var>Q</var><sub>keep</sub>(<var>u</var>, <var>S</var>, <var>d</var>, <var>t</var>) =
max<sub><var>h</var> &sube; <var>d</var></sub>
&sum;<sub><var>d</var>&prime;</sub> <var>P</var>(<var>h</var> &rarr; <var>d</var>&prime;) &middot;
<var>Q</var><sub>prev</sub>(<var>u</var>, <var>S</var>, <var>d</var>&prime;, <var>t</var>&minus;1)
:::

**Turn-start expected value:**

:::equation
<var>V</var>(<var>u</var>, <var>S</var>) =
&sum;<sub><var>d</var></sub> <var>w</var>(<var>d</var>) &middot;
<var>Q</var><sub>keep</sub>(<var>u</var>, <var>S</var>, <var>d</var>, 2)
&nbsp;/&nbsp; 7776
:::

**Keep-multiset deduplication.** The key to efficiency is factoring the keep-evaluation into two steps. Let <var>H</var> = {<var>h</var><sub>1</sub>, &hellip;, <var>h</var><sub>462</sub>} be the set of unique keep-multisets. For each <var>h</var> &in; <var>H</var>:

:::equation
<var>E</var>[<var>h</var>] = &sum;<sub><var>d</var>&prime;</sub> <var>P</var>(<var>h</var> &rarr; <var>d</var>&prime;) &middot; <var>Q</var><sub>prev</sub>(<var>d</var>&prime;)
:::

Then for each roll <var>d</var>, the best keep is just max over the (small) subset of <var>H</var> compatible with <var>d</var>.

:::

:::code

### Topological Padding ([source](https://github.com/dlangk/yatzy/blob/main/solver/src/constants.rs#L24-L28))

States are indexed as `scored * 128 + upper` using a stride of 128 instead of the natural 64. The extra 64 slots (indices 64 through 127) all duplicate the capped upper value. When an upper-category score pushes the total beyond 63, the access `sv[base + upper + score]` lands in the padding region instead of out of bounds. This eliminates a `min(upper + score, 63)` branch in the innermost loop, at the cost of 2&times; memory per strategy table (16 MB instead of 8 MB).

```rust
// constants.rs
pub const STATE_STRIDE: usize = 128;
pub const NUM_STATES: usize = STATE_STRIDE * (1 << 15); // 128 × 32,768 = 4,194,304

#[inline(always)]
pub fn state_index(upper_score: usize, scored_categories: usize) -> usize {
    scored_categories * STATE_STRIDE + upper_score
}
```

The NEON kernel that exploits this padding reads 64 consecutive state values via pointer arithmetic. When scoring an upper category with value `scr`, it reads from `sv[base + scr + 0], sv[base + scr + 1], ..., sv[base + scr + 63]`. If `scr + upper > 63`, the read falls into the padding region (indices 64 through 127), which contains the correct capped value. No branch needed.

```rust
// simd.rs — branchless upper-category access via topological padding
pub unsafe fn neon_add_max_offset_64(
    dst: &mut [f32; 64], sv: *const f32, base_plus_offset: usize, scalar: f32,
) {
    let s = vdupq_n_f32(scalar);
    let src = sv.add(base_plus_offset);  // sv + scored*128 + scr
    for i in (0..64).step_by(4) {
        let d = vld1q_f32(dst.as_ptr().add(i));
        let v = vld1q_f32(src.add(i));   // reads sv[base+scr+i..i+3]
        vst1q_f32(dst.as_mut_ptr().add(i), vmaxq_f32(d, vaddq_f32(v, s)));
    }
}
```

### Batched Solver ([source](https://github.com/dlangk/yatzy/blob/main/solver/src/batched_solver.rs#L24-L28))

Instead of solving one state at a time, the solver processes all 64 upper-score values for a given scored-category mask in a single batch. This turns sparse matrix-vector products (SpMV) into sparse matrix-matrix products (SpMM), keeping the working set at roughly 247 KB: small enough to fit in L2 cache.

```rust
// state_computation.rs
scored_masks.par_iter().for_each(|&scored| {
    let mut buffers = BatchedBuffers::new();
    for upper in 0..64 {
        let si = state_index(upper, scored);
        if !reachable[si] { continue; }
        let ev = solve_widget(&ctx, &buffers, upper, scored);
        state_values[si] = ev;
    }
});
```

### Sparse Transition Table ([source](https://github.com/dlangk/yatzy/blob/main/solver/src/types.rs#L25-L40))

The keep-to-outcome probability matrix is stored in Compressed Sparse Row (CSR) format. For each of the 462 keeps, `row_start[ki]` marks where its non-zero entries begin in the `vals` and `cols` arrays. The matrix has only 4,368 non-zero entries out of a potential 462 &times; 252 = 116,424 (96.2% sparse). Total storage: ~17 KB for the sparse data. The hot path reads it as:

```rust
// types.rs — CSR sparse dot product
for j in row_start[ki]..row_start[ki + 1] {
    ev += vals[j] * e_prev[cols[j] as usize];
}
```

### NEON SIMD Intrinsics ([source](https://github.com/dlangk/yatzy/blob/main/solver/src/simd.rs#L27))

The keep-evaluation kernel uses ARM NEON instructions to process four f32 values simultaneously. The critical operation is fused multiply-add followed by lane-wise maximum, avoiding round-trips to scalar registers. The solver includes 14 NEON kernels covering FMA, max, min, add-max, and mul-max operations.

```rust
// simd.rs — NEON FMA+max kernel
#[inline(always)]
pub unsafe fn neon_fma_max_f32x4(
    acc: float32x4_t,
    a: float32x4_t,
    b: float32x4_t,
    prob: float32x4_t,
) -> float32x4_t {
    let product = vmulq_f32(a, prob);
    let sum = vaddq_f32(b, product);
    vmaxq_f32(acc, sum)
}
```

### The f32 Decision

The entire solver uses f32 instead of f64. Across all 1.43 million reachable states, only 22 show a different optimal action under f32 than f64. The expected-value difference at the initial state is less than 0.001 points. The 2&times; bandwidth savings and native NEON f32 throughput make the tradeoff overwhelmingly worthwhile.

### Parallelism ([source](https://github.com/dlangk/yatzy/blob/main/solver/src/state_computation.rs#L251))

Rayon's `par_iter` distributes scored-category masks across all available cores. Each thread processes all 64 upper-score values for a given mask, ensuring contiguous memory access within the 128-stride layout. The thread count is pinned to 8 on Apple Silicon (matching the performance core count), achieving near-linear scaling.

### Zero-Copy Memory Mapping ([source](https://github.com/dlangk/yatzy/blob/main/solver/src/storage.rs#L50))

Strategy tables are stored as flat binary files: a 16-byte header (magic `0x59545A53`, version, state count) followed by `4,194,304 &times; 4 = 16,777,216` bytes of raw `f32` values. Loading uses `memmap2` to map the file directly into the process address space. The `StateValues` enum has two variants: `Owned(Vec<f32>)` for precomputation (writable) and `Mmap` for serving (read-only, zero-copy). Startup time: under 1 millisecond.

```rust
// types.rs — dual-mode state storage
pub enum StateValues {
    Owned(Vec<f32>),                        // precompute: writable
    Mmap { mmap: memmap2::Mmap },           // serve: zero-copy, read-only
}
```

### Performance

The full backward sweep over 1.43 million reachable states completes in **1.10 seconds** on an Apple M1 Max with 8 performance cores.

### Memory Lifecycle ([source](https://github.com/dlangk/yatzy/blob/main/solver/src/widget_solver.rs#L430))

The solver uses two flat `[f32; 252]` buffers (1,008 bytes each) in a ping-pong pattern. At each reroll stage, it reads expected values from one buffer (the results of the previous stage) and writes best-keep values into the other. Then it swaps: the output buffer becomes the input for the next stage. The same two buffers handle all three passes (scoring, second reroll, first reroll). Total scratch memory per widget: 2,016 bytes.

At the global level, the solver keeps one `f32` per state slot: `4,194,304 &times; 4 bytes = 16 MB` (of which ~1.43M entries are reachable). Because each layer only reads from the next, all layers share this single table. A naive approach that stored intermediate values for every roll at every stage of every widget would need roughly 8 GB.

:::html
<div class="chart-container" id="chart-memory-lifecycle">
  <p class="chart-caption">Memory footprint: one 8 MB table of expected scores, plus 2 KB of scratch space per widget.</p>
</div>
:::

### Memory Budget

| Structure | Size | Purpose |
|-----------|------|---------|
| State values | 16 MB | One f32 per state slot (4,194,304 &times; 4 bytes) |
| Precomputed scores | 15 KB | 252 dice sets &times; 15 categories &times; 4 bytes |
| Sparse keep table | 17 KB | 4,368 non-zero (f32, i32) pairs + row pointers |
| Reverse dice lookup | 32 KB | 5D index_lookup for O(1) dice &rarr; index |
| Keep frequency lookup | 182 KB | Horner hash table (46,656 entries) |
| Reachability mask | 4 KB | 64 &times; 64 booleans |
| Batched buffers | 247 KB/thread | Two ping-pong [f32; 64] &times; 252 + keep_ev |

**Total working set during precomputation:** <17 MB.
**During serving (mmap mode):** 16 MB mapped, ~250 KB active.

:::

:::
