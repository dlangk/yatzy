:::section{#solver}

## The Solver

Section 2 showed how to shrink the game to 1.43 million positions and how to process them layer by layer. This section explains what happens inside each position: how the solver evaluates a single turn and picks the best play. The entire computation finishes in about one second.

### Scoring the Final Roll

After the last reroll, the solver has a final roll of five dice and must choose which category to score. Each unscored category gives an immediate score plus a future value (the expected score from the resulting position, already computed in a later layer). The solver tries every option and picks the best:

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

One number comes out: the expected score from this position under optimal play.

### The Backward Sweep

All 1.43 million widgets are solved layer by layer, starting from the terminal states (layer 15, where the game is over) and working backward to layer 0. Each layer depends only on the next, and states within a layer are independent of each other. This means every state in a layer can be solved in parallel across all CPU cores. The full sweep completes in about one second.

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
  <div id="widget-fan-panel" class="math"></div>
  <p class="chart-caption">Step through a concrete scenario to see how the widget solver evaluates keeps, rerolls, and scoring.</p>
</div>
:::

:::html
<div class="chart-container" id="chart-backward-wave">
  <div class="chart-controls">
    <button class="chart-btn" id="wave-play-btn">&#9654; Play</button>
    <button class="chart-btn" id="wave-reset-btn">Reset</button>
    <input type="range" id="wave-scrub" class="chart-slider" min="0" max="15" value="15">
    <span class="scenario-label" id="wave-label">Layer 15: terminal states</span>
  </div>
  <div id="chart-backward-wave-svg"></div>
  <p class="chart-caption">The backward wave: each layer's states are evaluated in parallel before moving to the next.</p>
</div>
:::

:::insight
**462 shared holds make the inner loop tractable.** Without the keep shortcut, each of the 252 rolls would need its own set of sparse dot products against the reroll distribution. Sharing the computation across the 462 unique hold-multisets reduces the work by an order of magnitude, turning a minutes-long solve into a one-second sweep.
:::

:::math

The solver evaluates the Bellman equation in four stages. Terminal states have a known value (just the bonus check). For non-terminal states, the evaluation proceeds bottom-up through the widget layers.

**Terminal value (layer 15):**

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

**Risk-sensitive extension.** Replacing the linear expectation with exponential utility parameterised by &theta;:

:::equation
<var>V</var><sub>&theta;</sub> = (1/&theta;) &middot; log &sum;<sub><var>d</var></sub> <var>w</var>(<var>d</var>) &middot; exp(&theta; &middot; <var>Q</var>(<var>d</var>)) &nbsp;/&nbsp; 7776
:::

For |&theta;| &le; 0.15, a direct utility-domain formulation avoids logarithms entirely and runs at the same speed as the EV solver. For larger |&theta;|, a numerically stable log-sum-exp with max-subtraction is used, at roughly 2.5&times; the cost.

:::

:::code

### Topological Padding

States are indexed as `scored * 128 + upper` using a stride of 128 instead of the natural 64. The extra slots (indices 64 through 127) duplicate the capped upper value, so that scoring an upper category never requires a branch to clamp the index. This wastes half the storage (16 MB instead of 8 MB per table) but eliminates a conditional in the innermost loop. See also Section 2's code block for the index function.

```rust
// constants.rs
pub const STATE_STRIDE: usize = 128;
pub const NUM_STATES: usize = STATE_STRIDE * (1 << 15); // 4,194,304

#[inline(always)]
pub fn state_index(upper_score: usize, scored_categories: usize) -> usize {
    scored_categories * STATE_STRIDE + upper_score
}
```

### Batched Solver

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

### NEON SIMD Intrinsics

The keep-evaluation kernel uses ARM NEON instructions to process four f32 values simultaneously. The critical operation is fused multiply-add followed by lane-wise maximum, avoiding round-trips to scalar registers. The solver includes 14 NEON kernels covering FMA, max, min, add-max, mul-max, and a fast-exp polynomial (degree-5 Cephes, approximately 2 ULP accuracy) for the risk-sensitive log-sum-exp path.

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

### Parallelism

Rayon's `par_iter` distributes scored-category masks across all available cores. Each thread processes all 64 upper-score values for a given mask, ensuring contiguous memory access within the 128-stride layout. The thread count is pinned to 8 on Apple Silicon (matching the performance core count), achieving near-linear scaling.

### Zero-Copy Memory Mapping

Strategy tables are stored as flat binary files with a 16-byte header (magic number 0x59545A53, version 6 for &theta;=0, version 7 with embedded &theta; for non-zero values). Loading uses `memmap2` for zero-copy access: the OS maps the file directly into the process address space, achieving startup in under 1 millisecond.

### Performance

| Configuration | Time | Notes |
|---|---|---|
| &theta; = 0 (EV) | 1.10 s | Baseline |
| &#124;&theta;&#124; &le; 0.15 (utility) | 0.49 s | Faster: no log/exp |
| &#124;&theta;&#124; > 0.15 (LSE) | 2.7 s | log-sum-exp path |

All timings on Apple M1 Max with 8 performance cores.

### Memory Lifecycle

Because each layer only looks at the next one, the solver never needs to keep everything in memory at once. It stores one number per reachable position (the expected score from that point), taking about 8 MB. Each widget borrows about 2 KB of scratch space, computes its answer, and gives it back. A naive approach that stored all intermediate calculations would need roughly 8 GB.

:::html
<div class="chart-container" id="chart-memory-lifecycle">
  <p class="chart-caption">Memory footprint: one 8 MB table of expected scores, plus 2 KB of scratch space per widget.</p>
</div>
:::

:::

:::
