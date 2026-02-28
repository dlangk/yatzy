:::section{#optimal-strategy}

## The Optimal Strategy

With the solver in hand, we can study what optimal play actually looks like. The strategy table tells us the best action in every conceivable game state, but the aggregate statistics reveal something deeper: the structural forces that shape outcomes even under perfect play.

### Anatomy of a Hard Decision

Not all decisions are obvious. Consider this endgame position --most players
take the safe guaranteed score, but the solver sees further.

:::html
<div class="chart-container" id="chart-decision-anatomy">
  <div id="chart-decision-anatomy-content"></div>
</div>
:::

### The Score Distribution

A solved game does not mean a predictable game. Even under perfect play, a
single game of Yatzy is startlingly variable. The optimal policy produces a
mean score of 248.4 with a standard deviation of 38.5 --a coefficient
of variation above 15%. Two games played with identical, flawless strategy can
easily differ by 100 points or more. Understanding where this variance comes
from, and why it has the particular shape it does, is one of the most revealing
aspects of the analysis.

Simulating one million games under the optimal policy reveals a score
distribution that is emphatically not Gaussian. Despite the game consisting of
fifteen rounds --enough, one might think, for the
::concept[central limit theorem]{central-limit-theorem}
to take hold --the distribution is visibly skewed and heavy-tailed to
the left. The reason lies in the structural dependencies between rounds.

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

### Why the Maximum Fails

A natural first question: what is the highest possible score? If you compute the
maximum achievable value at each state --replacing the expected value in the
Bellman equation with a pure maximum --the answer is 374 points. This is
the theoretical ceiling, the score you would get if every dice roll came up
perfectly. But when you simulate this max-policy (which always chooses the action
with the highest ceiling) in actual play with random dice, the mean score drops
to just 118.7 --less than half the EV-optimal mean.

:::html
<div class="chart-container" id="chart-max-policy-failure">
  <div id="chart-max-policy-failure-svg"></div>
</div>
:::

The max-policy fails because it systematically overvalues unlikely outcomes. It
will hold a single Six hoping for Yatzy (five of a kind) when the expected value
of that gamble is far below a safe play like scoring a modest Full House. The
gap of 130 points between optimal and max-policy play is a vivid illustration
of why expected value, not best-case thinking, is the right objective for
sequential decision-making under uncertainty.

### The Forward Pass

The backward induction tells us the *value* of every state, but it does not
tell us how often each state is actually *visited* during optimal play. For that
we need the forward pass: starting from the initial state (turn 0, upper
score 0, no categories scored), propagate exact Markov transition probabilities
through the game tree. The result is three complementary views of what a
perfectly played game looks like in aggregate.

**The race to 63.** The upper-section bonus threshold at 63 points is the
single most important structural feature of Yatzy. This node-link diagram
shows the exact probability of reaching each upper score at each turn. The
bright channels trace the dominant flow of probability mass; most games
converge toward the bonus threshold by turns 10--12, but a persistent
low-probability channel hugs the bottom for games where lower-section
categories consumed the early turns.

:::html
<div class="chart-container" id="chart-race-to-63"></div>
:::

**When is each category scored?** The streamgraph below shows, for each turn,
the probability that each of the 15 categories is chosen. Early turns
favour high-value upper categories (Sixes, Fives, Fours) and multi-of-a-kind
combinations. The lower-value categories (Ones, Twos) and Chance tend to
be deferred to later turns, serving as dump slots when dice rolls are
unfavourable.

:::html
<div class="chart-container" id="chart-category-stream"></div>
:::

**The narrowing of uncertainty.** Before any dice are rolled, every game
starts with the same expected value of 248. As turns progress, the
distribution of conditional expected values fans out and then collapses
toward the actual score. The ridgeline plot below shows this process: each
row is one turn's distribution over expected final scores, from the single
spike at turn 0 to the wide spread of realized outcomes at turn 15.

:::html
<div class="chart-container" id="chart-ev-ridgeline"></div>
:::

:::math

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

:::code

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

The max-policy simulation uses the same solver infrastructure but replaces
the expectation operator with a pure maximum at every decision point. The
resulting policy table is then evaluated via standard Monte Carlo:

```rust
// batched_solver.rs — max-policy variant
fn solve_widget_max(ctx: &YatzyContext, ...) -> f32 {
    // Phase 2: replace weighted sum with max
    for ds in 0..252 {
        let mut best = f32::NEG_INFINITY;
        for keep in 0..n_keeps[ds] {
            let future_max = lookup_max(keep, state_values);
            best = best.max(future_max);
        }
        widget_values[ds] = best;
    }
    // Phase 3: max over dice outcomes (not weighted sum)
    widget_values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}
```

:::

:::
