:::section{#introduction}

## 1. An Introduction to Yatzy

Yatzy is easy to learn. I loved playing it as a kid, and I still do. I relate Yatzy a lot to family times, especially with my dad. He is the most fun person I know to play Yatzy with. ❤️

I think a big reason for why Yatzy is such a great family game is that it's easy to learn, and has enough randomness that you can have fun without worrying about strategy. You can just roll the dice, keep what you like, and assign to categories as you go.

Most family games require some mixture of skill and luck. The more skill, the less likely it is that your entire family will enjoy it together. For example, playing chess isn't really enjoyable unless both players are about as good.

At the same time, we want to feel responsible for winning, so some skill needs to be involved. Yatzy, I think, strikes a perfect balance. When you win, you feel skilled. When you loose, you had bad luck. Everyone is always happy!

Many years ago, I got it into my head to figure out: exactly how much skill is involved in Yatzy? Or put differently: how good can you be at Yatzy? This site is the result of that investigation.

::concept[Markov decision process]{markov-decision-process}
with over two million possible states. Solving it optimally requires
::concept[backward induction]{backward-induction}
across 1.43 million reachable positions, evaluating every legal dice-keep and
category-assignment at every step. The gap between naive play and the optimal
policy is roughly 82 points --an entire category's worth of score, hidden
in plain sight.

This treatise traces the full arc from game rules to solved game. We build
the ::concept[state space]{state-space},
engineer a solver fast enough to sweep an entire risk-preference axis in
minutes, decompose the score distribution into its structural components,
and ask what it means to play well when perfection is computationally
within reach.

### Ten Thousand Games

What happens when a perfect player sits down for ten thousand games?
Each dot is one complete game --watch the shape of optimal play emerge.

:::html
<div class="chart-container" id="chart-score-spray">
  <div class="chart-controls">
    <button class="chart-btn" id="spray-replay-btn">&#9654; Replay</button>
  </div>
  <div id="chart-score-spray-wrap" style="position: relative;"></div>
  <div class="spray-legend" id="spray-legend"></div>
  <div class="spray-stats" id="spray-stats"></div>
  <div id="spray-scorecard-popup" class="spray-popup hidden"></div>
</div>
:::

:::depth-2

Formally, optimal Yatzy play is a finite-horizon stochastic dynamic program.
The state at each turn is the pair (<var>u</var>, <var>S</var>) where
<var>u</var> &isin; {0, &hellip;, 63} is the accumulated upper-section score
and <var>S</var> &sube; {0, &hellip;, 14} is the set of categories already
scored. The ::concept[Bellman equation]{bellman-equation}
recurses over all 2<sup>15</sup> &times; 64 = 2,097,152 state slots, though
only about 1.43 million are reachable from the initial state (0, &empty;).

:::equation
<var>V</var>(<var>u</var>, <var>S</var>) = E<sub>dice</sub>[max<sub>action</sub>
<var>Q</var>(<var>u</var>, <var>S</var>, dice, action)]
:::

The expected value under optimal play is 248.4 points with a standard
deviation of 38.5 --far higher than any human achieves on average,
and far more variable than most players assume.

:::

:::depth-3

The solver is a three-phase pipeline. Phase 0 builds lookup tables: all 252
ordered dice outcomes, their multinomial weights, the 15 &times; 252 scoring matrix,
and a sparse transition table mapping each of 462 unique keep-multisets to its
reachable outcomes. Phase 1 marks which of the 2,097,152 state slots are
reachable from the initial position, pruning 31.8%. Phase 2 runs backward
induction: for each popcount level from 14 down to 0, it evaluates the
Bellman equation at every reachable state in parallel.

The hot path is `SOLVE_WIDGET` --the function called once per
reachable state. It uses two ping-pong buffers (`e[0]`,
`e[1]`, each 252 &times; f32) and works bottom-up through four fused
groups:

```
SOLVE_WIDGET(state, sv):
  // Group 6: best category for each of 252 final-roll outcomes
  for ds in 0..252:
    e[0][ds] = max over unscored categories c:
                 score(ds, c) + sv[successor(state, c)]

  // Group 5: best keep with 1 reroll left
  // Group 3: best keep with 2 rerolls left  (same logic, applied twice)
  for pass in [0, 1]:
    // Step 1: sparse mat-vec — 462 unique keep EVs
    for kid in 0..462:
      keep_ev[kid] = Σ P(kid → ds') · e_prev[ds']

    // Step 2: best keep per dice set
    for ds in 0..252:
      e_curr[ds] = max(e_prev[ds], max over keeps k of keep_ev[k])

  // Group 1: expected value over initial roll
  return Σ P(ds) · e[0][ds] / 7776
```

In the production solver, all 64 upper-score variants of a scored-category
mask are processed simultaneously --converting the inner sparse matrix-vector
product into a sparse matrix-*matrix* product. Each row of the
ping-pong buffers becomes a 252 &times; 64 array of f32, and the NEON SIMD kernels
operate on 4-wide vectors in the inner loop:

```
// simd.rs — NEON FMA kernel (Groups 5/3 Step 1)
unsafe fn neon_fma_64(dst: &mut [f32; 64], src: &[f32; 64], prob: f32) {
    let s = vdupq_n_f32(prob);
    for i in (0..64).step_by(4) {
        let d = vld1q_f32(dst.as_ptr().add(i));
        let r = vld1q_f32(src.as_ptr().add(i));
        vst1q_f32(dst.as_mut_ptr().add(i), vfmaq_f32(d, r, s));
    }
}

// simd.rs — NEON add-max kernel (Group 6)
unsafe fn neon_add_max_64(dst: &mut [f32; 64], sv: *const f32, scr: f32) {
    let s = vdupq_n_f32(scr);
    for i in (0..64).step_by(4) {
        let d = vld1q_f32(dst.as_ptr().add(i));
        let v = vaddq_f32(vld1q_f32(sv.add(i)), s);
        vst1q_f32(dst.as_mut_ptr().add(i), vmaxq_f32(d, v));
    }
}
```

The key optimizations, roughly in order of impact:

1. **Batched SoA layout.** Processing 64 upper scores at once
   turns scattered scalar lookups into contiguous 256-byte block reads. Total
   working set: ~247 KB per thread, which fits in L2 cache.
2. **Keep-multiset deduplication.** Of the 31 possible reroll
   masks per dice set, only 462 unique keep-multisets exist across all 252
   sets. Each reroll pass computes 462 sparse dot products instead of 4,108
   redundant ones --an 8.9× reduction.
3. **Topological padding.** `STATE_STRIDE=128`
   instead of the natural 64. Indices 64–127 duplicate the capped value
   (index 63), which eliminates a `min(up + score, 63)` branch on
   every upper-category state lookup. Costs 2× storage, saves billions of
   branches.
4. **NEON SIMD intrinsics.** 14 hand-written kernels: FMA for
   keep-EV accumulation, add-max for category scoring, mul-max for
   utility-domain, and a custom fast-exp polynomial (~2 ULP, degree-5
   Cephes) for the log-sum-exp risk solver.
5. **f32 throughout.** All state values, probabilities, and
   intermediate computations use f32. Across 1.43 million states, only 22
   differ in their optimal action versus f64 --a max error of 0.00046 points.
   The 2× bandwidth saving compounds with every SIMD operation.
6. **Rayon parallel layers.** Each popcount level is
   embarrassingly parallel: states at level <var>k</var> read from level
   <var>k</var>+1 (already computed). `par_iter` over scored
   masks, one `BatchedBuffers` per thread, unsafe raw pointer
   writes (unique indices guarantee no aliasing).

Result: 1.1 seconds for the full &theta;=0 solve on an M1 Max (8 performance
cores). The risk-sensitive solver adds one optimization layer: for
|&theta;| &le; 0.15, a utility-domain formulation avoids all exp/ln operations
(0.49 s), while larger |&theta;| uses numerically stable log-sum-exp (2.7 s).

:::

### The Score Distribution

Simulating one million games under the optimal policy reveals a score
distribution that is emphatically not Gaussian. Despite the game consisting of
fifteen rounds --enough, one might think, for the
::concept[central limit theorem]{central-limit-theorem}
to take hold --the distribution is visibly skewed and heavy-tailed to
the left. The reason lies in the structural dependencies between rounds.

:::html
<div class="chart-container" id="chart-score-distribution">
  <div id="chart-score-distribution-svg"></div>
</div>
:::

### Four Populations, One Game

The non-Gaussian shape is not random noise; it has a precise structural
explanation. Every game either hits or misses the
::concept[upper-section bonus]{upper-section-bonus}
(at a rate of roughly 90%), and every game either lands or misses Yatzy
(at about 39%). These two binary events create four sub-populations, each
with its own roughly normal distribution:

- Bonus missed, no Yatzy --mean around 164
- Bonus missed, Yatzy scored --mean around 213
- Bonus hit, no Yatzy --mean around 237
- Bonus hit, Yatzy scored --mean around 288

The overall distribution is a weighted mixture of these four components.
Because the bonus event dominates (it separates 90% of games from 10%), the
distribution looks roughly bimodal, with a long left tail formed by the
bonus-miss groups.

:::html
<div class="chart-container" id="chart-mixture">
  <div id="chart-mixture-svg"></div>
</div>
:::

### The Bonus Covariance

The ::concept[upper-section bonus]{upper-section-bonus}
is worth 50 points on paper. But its true impact is 72 points --50 from
the bonus itself, plus an additional 22 points of correlated scoring advantage.
Players who reach the bonus threshold tend to have rolled well in the upper
categories, which correlates with higher scores in the lower section too (more
dice combinations available for straights, full houses, and multi-of-a-kind
categories). This covariance structure means the bonus is not merely an additive
reward; it is a marker of a globally favourable game trajectory.

:::html
<div class="chart-container" id="chart-bonus-covariance">
  <div id="chart-bonus-covariance-svg"></div>
</div>
:::

:::insight
**Key insight:** The 72-point bonus gap explains why good Yatzy
players obsess over the upper section early in the game. The 50-point bonus
is visible in the rules, but the 22-point correlated advantage is invisible
--- discoverable only through simulation or exact computation.
:::

:::depth-2

The four-component mixture model can be written as:

:::equation
<var>f</var>(<var>x</var>) = &sum;<sub><var>i</var>=1</sub><sup>4</sup>
&pi;<sub><var>i</var></sub> &middot;
&phi;(<var>x</var>; &mu;<sub><var>i</var></sub>, &sigma;<sub><var>i</var></sub>)
:::

where the mixing weights &pi;<sub><var>i</var></sub> are determined by the
joint probability of bonus hit/miss and Yatzy hit/miss. Under optimal play,
the bonus hit rate is approximately 90% and the Yatzy probability is approximately
39%, giving mixing weights of roughly 0.055, 0.055, 0.555, and 0.335 for the
four components (miss/miss, miss/hit, hit/miss, hit/hit).

The 22-point correlated advantage can be decomposed further. Approximately
12 points come from the upper-section scores themselves being higher when
the bonus is reached (conditional expectation above threshold vs below),
and approximately 10 points come from a genuine positive correlation between
upper-section luck and lower-section scoring opportunities.

:::

:::depth-3

The mixture decomposition is computed from simulation data by conditioning
on the bonus and Yatzy indicators. The Monte Carlo engine records both
the final score and the bonus/Yatzy status for each simulated game:

```python
# analytics: mixture decomposition
import polars as pl

df = pl.read_parquet("outputs/aggregates/parquet/summary.parquet")
groups = df.group_by(["bonus_hit", "yatzy_hit"]).agg([
    pl.col("score").mean().alias("mean"),
    pl.col("score").std().alias("std"),
    pl.col("score").count().alias("n"),
])
# Mixing weights: n_i / N
groups = groups.with_columns(
    (pl.col("n") / pl.col("n").sum()).alias("weight")
)
```

:::

:::
