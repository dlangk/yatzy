:::section{#part-i}

:::part-title
Part I
:::

## The Geometry of a Game

Every game of Yatzy traces a path through a vast, invisible space. To solve
the game, we first need to map that space --to understand its shape, its
size, and the structure that makes computation feasible.

### The State Space

A Yatzy game has fifteen rounds. In each round, the player assigns a dice
result to one of the fifteen categories. The categories split into two
sections: six upper categories (Ones through Sixes) and nine lower categories
(One Pair through Yatzy). A running tally of the upper-section total determines
whether the player earns a 50-point
::concept[bonus]{upper-section-bonus} at the
end (the threshold is 63 points, equivalent to three of each die face).

The ::concept[state]{state-space} of the game
at any point between turns is fully described by two quantities: the upper-section
score so far (an integer from 0 to 63, since any value above the threshold is
equivalent), and the set of categories already used (a subset of fifteen elements).
This gives a theoretical space of 64 &times; 2<sup>15</sup> = 2,097,152 slots,
arranged as a grid where one axis counts points and the other encodes a bitmask
of scored categories.

:::html
<div class="chart-container" id="chart-state-space-counter">
  <div class="scroll-sticky" id="counter-sticky">
    <div id="counter-display"></div>
  </div>
  <div class="scroll-steps" id="counter-steps">
    <div class="step" data-step="0"><p>Fifteen categories, each scored or not — 2<sup>15</sup> = 32,768 possible scorecards.</p></div>
    <div class="step" data-step="1"><p>Each scorecard combined with an upper-section total (0–63) — now 2,097,152 slots.</p></div>
    <div class="step" data-step="2"><p>Forward reachability prunes 31.8% of these. The solver evaluates 1,430,000 states.</p></div>
    <div class="step" data-step="3"><p>Within each state, a single turn unfolds across 252 dice outcomes and 462 keep decisions.</p></div>
    <div class="step" data-step="4"><p>The solver computes the exact optimal decision for every one of these states. The result fits in 8 megabytes.</p></div>
  </div>
</div>
:::

### Pruning by Reachability

Not every slot in this grid corresponds to a position that can actually arise
in play. An upper score of 60, for example, cannot occur if only Ones and Twos
have been scored. A forward pass from the starting state (upper = 0, scored = empty)
identifies exactly which states are reachable. The result is striking: 31.8% of
slots are pruned, leaving roughly 1,430,000 states that the solver must evaluate.
This pruning is not an approximation --it is exact, and it reduces both
computation time and memory proportionally.

:::html
<div class="chart-container" id="chart-reachability-pruning">
  <div id="chart-reachability-pruning-svg"></div>
</div>
:::

### Dice and Decisions

Within each turn, the player faces a different combinatorial landscape. Five
six-sided dice produce 6<sup>5</sup> = 7,776 ordered outcomes, but only 252
distinct unordered results (since the order of dice does not matter). After
seeing the dice, the player chooses which to keep --a subset of the
current roll. The number of distinct
::concept[keep-multisets]{keep-multiset} is
462, each representing a unique combination of held dice values. Every keep
choice leads to a probability distribution over the next roll's 252 outcomes,
which the solver evaluates exhaustively.

The full decision tree within a single turn is therefore: see 252 possible dice
outcomes, choose among up to 462 keeps, observe a new outcome, choose again,
then assign to one of the remaining categories. This tree is evaluated at every
reachable state, and the optimal action is stored in a strategy table for
instant lookup during play.

### Anatomy of a Hard Decision

Not all decisions are obvious. Consider this endgame position --most players
take the safe guaranteed score, but the solver sees further.

:::html
<div class="chart-container" id="chart-decision-anatomy">
  <div id="chart-decision-anatomy-content"></div>
</div>
:::

:::depth-2

The state space forms a
::concept[Markov decision process]{markov-decision-process}
with a layered directed acyclic graph structure. Each layer corresponds to a
specific number of scored categories (the popcount of the bitmask). Transitions
only move forward: from layer <var>k</var> (with <var>k</var> categories scored)
to layer <var>k</var>+1. This topological ordering is what makes backward
induction possible --we solve layer 15 trivially (the game is over),
then work backward through layers 14, 13, &hellip;, 0.

:::equation
|states| = &sum;<sub><var>k</var>=0</sub><sup>15</sup>
C(15,<var>k</var>) &times; 64 = 2,097,152
&nbsp;&nbsp;&rarr;&nbsp;&nbsp;
reachable &asymp; 1,430,000 &nbsp;(31.8% pruned)
:::

The 252 distinct dice outcomes correspond to multisets of size 5 drawn from
{1,&hellip;,6}, counted by C(6+5&minus;1, 5) = C(10,5) = 252. Each outcome
has a multiplicity weight (the number of ordered permutations that map to it),
and the solver weights expectations accordingly.

:::

:::depth-3

In the solver, states are indexed as `scored * 128 + upper` using a
stride of 128 instead of 64. The extra padding (indices 64–127) duplicates
the capped upper value, allowing branchless access when an upper-category score
would push the total beyond 63. This topological padding eliminates a branch
in the innermost loop at the cost of 2× memory (16 MB per strategy table).

```rust
// constants.rs
pub const STATE_STRIDE: usize = 128;
pub const NUM_STATES: usize = 32768 * STATE_STRIDE; // 4,194,304

#[inline(always)]
pub fn state_index(upper: usize, scored: usize) -> usize {
    scored * STATE_STRIDE + upper
}
```

Keep-multiset deduplication maps the 2<sup>5</sup> = 32 subsets of a 5-die
roll to at most 462 unique multisets across all 252 outcomes. The precomputed
`KeepTable` stores the transition probabilities for each (keep, outcome)
pair as `f32`, eliminating a costly `f64 → f32` conversion
in the hot path.

:::

:::
