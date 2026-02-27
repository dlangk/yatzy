:::section{#state-space}

## The State Space

A Yatzy game is a sequence of rolls, keeps, and category assignments spread across
fifteen rounds. Played naively, the number of distinct game histories is
astronomical--roughly 1.7 &times; 10<sup>170</sup>, vastly exceeding the number of atoms
in the observable universe. Solving the game exactly requires a series of reductions
that compress this enormity into something a laptop can handle in about a second.

Each step in the chain replaces one view of the problem with a strictly smaller one,
without losing any information relevant to optimal play.

### Step 1: What Matters, Not What Happened

The key insight is that most of the game history is irrelevant. Two games that
have scored different categories in different orders, but arrived at the same
upper-section total and the same set of used categories, face identical futures.
The
::concept[state]{state-space}
of the game collapses to two numbers: the upper-section score so far (an integer
from 0 to 63, since any value above the
::concept[bonus]{upper-section-bonus}
threshold is equivalent) and the set of categories already used (a 15-bit bitmask).
This gives 64 &times; 2<sup>15</sup> = 2,097,152 slots--a reduction of roughly
100,000&times; from the raw game-history count.

:::html
<div class="chart-container" id="chart-reduction-funnel"></div>
:::

### Step 2: Prune the Impossible

Not every slot corresponds to a position that can actually arise in play.
An upper score of 60, for instance, cannot occur if only Ones and Twos have been
scored. A forward DP from the starting state identifies exactly which states are
::concept[reachable]{reachability-pruning}: the result is that
31.8% of slots are pruned, leaving roughly 1,430,000 states. This is exact--not
an approximation--and it reduces both computation time and memory proportionally.

The heatmap below shows reachability by upper score and number of upper categories
scored. Gray cells are provably unreachable. The triangular pattern reflects the
combinatorial constraint: few categories cannot produce high upper scores.

:::html
<div class="chart-container" id="chart-reachability-grid"></div>
:::

### Step 3: Decompose into Widgets

Within each state, a single turn has a fixed structure: roll, keep, reroll, keep,
final roll, then assign to a category. This six-layer decision tree--which we call a
::concept[widget]{solve-widget}--is
self-contained: 1 entry node, 252 chance nodes per roll layer, and 462 decision
nodes per keep layer. In total: 1,681 nodes and at most 21,000 edges, fitting in
about 3 KB of working memory.

Every reachable state gets one widget evaluation. The widget reads only the
successor expected values from a shared E-table and writes back a single number.
This decomposition is what makes the problem embarrassingly parallel at the
state level.

:::html
<div class="chart-container" id="chart-widget-structure"></div>
:::

### Step 4: Process in Topological Order

The states form a
::concept[Markov decision process]{markov-decision-process}
whose transitions only move forward: from layer <var>k</var> (with <var>k</var>
categories scored) to layer <var>k</var>+1. This means we can process all 16 layers
in strict topological order, starting at the terminal layer (all 15 categories
scored, where the only remaining value is the bonus) and working backward to
layer 0 (game start).

The number of widgets per layer follows
C(15, <var>k</var>)--a binomial distribution peaking at
<var>k</var> = 7 with 6,435 widgets. The animation below shows the wavefront
sweeping from terminals to start. No layer ever needs to revisit a previous one.

:::html
<div class="chart-container" id="chart-topological-cascade">
  <div class="chart-controls">
    <button class="chart-btn" id="cascade-play-btn">&#9654; Play</button>
    <button class="chart-btn" id="cascade-reset-btn">Reset</button>
    <input type="range" class="chart-slider" id="cascade-scrub" min="-1" max="15" value="-1">
    <span class="slider-value" id="cascade-label">All layers pending</span>
  </div>
  <div id="chart-topological-cascade-svg"></div>
</div>
:::

### Memory: What Lives and What Dies

The topological structure has a dramatic consequence for memory. The solver
maintains a single E-table of expected values--one float per reachable state,
about 8 MB total. Each widget borrows ~3 KB of working memory, computes its
answer, and releases it. At no point does the solver hold more than the
E-table plus one widget's workspace.

Compare this with a naive approach that materializes all intermediate values
for all states simultaneously: that would require roughly 8 GB. The actual
solver uses 1,000 times less memory.

:::html
<div class="chart-container" id="chart-memory-lifecycle"></div>
:::

:::insight

**The reduction chain: 10<sup>170</sup> &rarr; 8 MB.** Sufficient statistics
collapse histories to 2.1M states. Reachability pruning removes 32%. Widget
decomposition isolates each turn into 3 KB of working memory. Topological
ordering ensures we never look back. The final footprint is a single 8 MB table.

:::

:::math

The state space forms a layered DAG. Each layer <var>k</var> contains all states
with exactly <var>k</var> categories scored. Transitions go strictly from
layer <var>k</var> to layer <var>k</var>+1, enabling backward induction:

:::equation
|states| = &sum;<sub><var>k</var>=0</sub><sup>15</sup>
C(15,<var>k</var>) &times; 64 = 2,097,152
&nbsp;&nbsp;&rarr;&nbsp;&nbsp;
reachable &asymp; 1,430,000 &nbsp;(31.8% pruned)
:::

The 252 distinct dice outcomes correspond to multisets of size 5 drawn from
{1,&hellip;,6}, counted by C(10,5) = 252. Each keep choice is one of 462
unique multisets. The widget structure gives:

:::equation
nodes/widget = 1 + 252 + 462 + 252 + 462 + 252 = 1,681
:::

Reachability is computed via forward DP over the 6-bit upper-category subsets.
For each subset <var>M</var> &sube; {1,&hellip;,6} and score <var>n</var>:

:::equation
R[n, M] = &exist; face <var>d</var> &notin; M, count <var>c</var> &in; {1..5}:
R[n &minus; d&middot;c, M \setminus {d}]
:::

:::

:::code

In the solver, states are indexed as `scored * 128 + upper` using a
stride of 128 instead of 64. The extra padding (indices 64-127) duplicates
the capped upper value, allowing branchless access when an upper-category score
would push the total beyond 63. This topological padding eliminates a branch
in the innermost loop at the cost of 2&times; memory (16 MB per strategy table).

```rust
// constants.rs
pub const STATE_STRIDE: usize = 128;
pub const NUM_STATES: usize = 32768 * STATE_STRIDE; // 4,194,304

#[inline(always)]
pub fn state_index(upper: usize, scored: usize) -> usize {
    scored * STATE_STRIDE + upper
}
```

The widget solver evaluates one turn for a single state. It reads successor
expected values from the shared E-table and writes back a single optimal EV.
Simplified pseudocode:

```
SOLVE_WIDGET(S = (C, m)):
    // Layer 6: score final roll → choose best category
    for each roll r in 252 outcomes:
        E_exit[r] = max over unscored c of:
            score(S, r, c) + E[successor(S, r, c)]

    // Layer 5: choose best keep (1 reroll left)
    for each roll r:
        E_roll1[r] = max over keeps r' ⊆ r of:
            Σ P(r' → r'') · E_exit[r'']

    // Layers 3-4: choose best keep (2 rerolls left)
    for each roll r:
        E_roll2[r] = max over keeps r' ⊆ r of:
            Σ P(r' → r'') · E_roll1[r'']

    // Layer 1: expected value over initial roll
    E[S] = Σ P(∅ → r) · E_roll2[r]
```

:::

:::
