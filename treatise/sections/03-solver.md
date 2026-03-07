:::section{#solver}

## The Solver

We have established that there are ~1.43M reachable states, organized into 16 layers, connected by widgets. Every transition goes strictly forward (from layer *l* to layer *l*+1), and layer 16 is trivial: just check whether the upper bonus was earned. This one-directional structure is the key to solving the entire game. We can work backward: solve layer 16 first (trivial), then use those answers to solve layer 15, then layer 14, and so on until we reach layer 1.

:::html
<div id="chart-backward-cascade"></div>
:::

### Solving a Widget

To compute the value of a state, the solver [evaluates its widget](https://github.com/dlangk/yatzy/blob/main/solver/src/widget_solver.rs#L430): the complete decision tree of one turn. The trick is to work backward through the turn: start from the last decision (which category to score) and reason back toward the first roll. Since the solver already knows the value of every end state (from the next layer) and the probability of every dice outcome, it can compute the expected score of every possible choice.

The visualization below demonstrates this live, using the actual solver. It first highlights the backward reasoning, sweeping from the scoring decision up to the initial roll, then lets you play a turn yourself.

:::html
<div class="chart-container" id="chart-widget-interactive">
  <p class="chart-caption">Walk through the solver's backward reasoning, then play a turn with live evaluation.</p>
</div>
:::

### Scoring the Final Roll

The solver starts at the end of the turn. After the last reroll, there is a final roll of five dice and the player must choose which category to score. Each unscored category gives an immediate score plus a future value (the expected score from the resulting state, already computed in a later layer). The solver tries every option and picks the best:

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

One step further back: the player sees a roll and decides which dice to keep. Each possible keep leads to a probability distribution over rerolls. The expected score of a keep is the weighted sum over those outcomes, using the scoring expected scores just computed:

```
for each roll (252):
    best = -∞
    for each way to keep dice:
        exp = Σ P(keep → reroll) × scoring_exp[reroll]
        best = max(best, exp)
    keep_exp[roll] = best
```

The solver runs this logic twice, in two identical passes: once for the second reroll (using scoring expected scores as input) and once for the first reroll (using second-reroll expected scores as input). It reads from one buffer and writes to the other, then swaps.

### Transition Probabilities

Every time you decide which dice to keep, you are not just holding onto numbers you like. You are choosing which possible futures to allow, and quietly closing off all the others.

A transition probability is simply the answer to the question: given the dice I am holding, how likely is each outcome when I reroll? It is the bridge between where you are now and where you could end up.

#### How the probability is calculated

When you keep some dice and reroll the rest, each free die lands on 1 through 6 independently. The probability of any specific outcome depends on how many ways the rerolled dice can produce the values you need.

This compounds across a full turn. Each roll depends on what you kept from the previous one, so the probability of any specific three-roll sequence is the product of three individual probabilities, one per roll. That multiplication is where things get surprising.

#### Try it: construct a path

Use the tool below to set your dice at each roll, decide what to keep, and watch the joint probability of your path update in real time.

:::html
<div class="chart-container" id="chart-path-probability">
  <p class="chart-caption">Use the arrows to set dice values and click a die to toggle keep/reroll. Each step's probability multiplies into the path total.</p>
</div>
:::

The chain at the bottom shows what is happening: three probabilities being multiplied together. Each one is already a small fraction. Their product is almost always tiny, sometimes astronomically so.

This is one of the most counterintuitive things in probability. Each individual step can feel completely reasonable. Getting two fives on a reroll: sure, that happens. But the specific sequence of all three rolls, each turning out exactly as it did? That combination is almost always far rarer than it feels. If a million people made the exact same keep decisions as you, only a handful would trace the exact same path.

This does not mean your decisions were wrong. A rare path and a bad decision are completely different things. The best possible strategy still produces rare specific outcomes most of the time, because every specific path is rare. What good decisions do is aim your probability toward outcomes you want, not guarantee you arrive there.

#### The full map of possibilities

The interactive tool above shows one path at a time. The image below shows all of them at once.

:::html
<div class="chart-container" id="chart-transition-matrix">
  <p class="chart-caption">The 462 &times; 252 keep-to-outcome transition probability matrix. Each colored cell is a non-zero entry. Rows grouped by number of dice kept (k).</p>
</div>
:::

Each row is one of the 462 keeping decisions. Each column is one of the 252 possible outcomes. A colored cell means the outcome is reachable; a blank cell means the kept dice rule it out.

At the top (keep nothing), every outcome is possible: the row is fully filled. At the bottom (keep all five), only one outcome remains: the dice you already hold. In between, each die you lock narrows the future. The filled region drifts diagonally because both axes are sorted the same way: keeping low dice only reaches low outcomes, keeping sixes only reaches outcomes containing sixes. Your keeping decision does not just narrow the future; it aims it.

The solver [precomputes](https://github.com/dlangk/yatzy/blob/main/solver/src/phase0_tables.rs#L82) all 462 &times; 252 transition probabilities into a [sparse table](https://github.com/dlangk/yatzy/blob/main/solver/src/types.rs#L25-L40) with only 4,368 non-zero entries (roughly 4% of the full grid). Every widget on this page reads from that table rather than recomputing probabilities on demand.

### The Keep Shortcut

Here's a small insight that makes the whole solver fast.

Many different rolls lead to the same keeping decision. "Keep my two threes" from (1,3,3,5,6) and from (2,3,3,4,4) are the same situation in disguise: two threes in hand, three free dice to reroll. The reroll distribution is identical. So the expected score for that keep only needs to be calculated once.

The solver exploits this by splitting the work in two. First, it computes the expected score for each of the 462 unique keeps. This is the expensive step, done once. Then, for each actual roll, it simply looks up the best available keep from that precomputed table. This is a cheap step, done quickly.

The result is that the hard probability work happens 462 times instead of once per roll per game state. That's what makes real-time evaluation possible.

```
Step 1: Compute expected score for each unique keep (462)
    for each unique keep h:
        keep_ev[h] = Σ P(h → outcome) × prev_exp[outcome]

Step 2: For each roll, pick the best keep (252 lookups)
    for each roll r:
        keep_exp[r] = max over valid keeps h of r: keep_ev[h]
```

### The First Roll

At the start of a turn, before any dice are thrown, the expected score is just the weighted average over all 252 possible initial rolls:

```
E(state) = Σ P(initial roll = r) × keep_exp[r]   /   7776
```

One number comes out: the expected score from this state under optimal play.

### The Backward Sweep

That is how a single widget is solved. Now we zoom out. All 1.43 million widgets are [solved layer by layer](https://github.com/dlangk/yatzy/blob/main/solver/src/state_computation.rs#L251), starting from layer 16 and working backward to layer 1. Each layer depends only on the next, and states within a layer are independent of each other, so every state in a layer can be solved in parallel across all CPU cores. The full sweep completes in about one second.

:::insight
**462 shared keeps make the inner loop tractable.** Without the keep shortcut, each of the 252 rolls would need its own set of sparse dot products against the reroll distribution. Sharing the computation across the 462 unique keep-multisets reduces the work by an order of magnitude, turning a minutes-long solve into a one-second sweep.
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

**Turn-start expected score:**

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

The solver uses two flat `[f32; 252]` buffers (1,008 bytes each) in a ping-pong pattern. At each reroll stage, it reads expected scores from one buffer (the results of the previous stage) and writes best-keep scores into the other. Then it swaps: the output buffer becomes the input for the next stage. The same two buffers handle all three passes (scoring, second reroll, first reroll). Total scratch memory per widget: 2,016 bytes.

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
