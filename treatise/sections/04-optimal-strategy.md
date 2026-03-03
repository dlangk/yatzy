:::section{#optimal-strategy}

## The Optimal Strategy

Now we have a solver that knows the best move in every possible situation. What does perfect play actually look like? The answer is surprising: even when you play perfectly, Yatzy is still wildly unpredictable.

### Anatomy of a Hard Decision

Not all decisions are obvious. Consider this endgame position. Most players take the safe guaranteed score, but the solver sees further.

:::html
<div class="chart-container" id="chart-decision-anatomy">
  <div id="chart-decision-anatomy-content"></div>
  <p class="chart-caption">A hard endgame decision: the safe play vs. the solver's deeper calculation.</p>
</div>
:::

### The Score Distribution

Playing perfectly does not mean scoring the same every time. The optimal strategy averages 248.4 points per game, but the spread is enormous: a standard deviation of 38.5 means that two perfectly played games can easily differ by 100 points or more.

Simulating one million games shows that the scores do not follow the familiar bell curve. You might expect fifteen rounds to average out, but they don't. The distribution is lopsided, with a long tail stretching to the left. The reason? The rounds are not independent. What happens early in the game changes what is possible later.

### Four Groups, One Game

The lopsided shape has a clean explanation. Two things dominate your final score more than anything else: whether you hit the
::concept[upper-section bonus]{upper-section-bonus}
(about 90% of the time under perfect play), and whether you scored Yatzy (about 39%). These two yes-or-no outcomes split all games into four groups, each with its own score range:

- Missed bonus, no Yatzy: average around 164
- Missed bonus, scored Yatzy: average around 213
- Hit bonus, no Yatzy: average around 237
- Hit bonus, scored Yatzy: average around 288

The overall score distribution is these four groups stacked on top of each other. Because the bonus matters so much (it separates 90% of games from the unlucky 10%), you can see a clear dip where the two populations meet.

:::html
<div class="chart-container" id="chart-mixture">
  <div id="chart-mixture-svg"></div>
  <p class="chart-caption">The score distribution is a blend of four groups: bonus hit/miss × Yatzy hit/miss.</p>
</div>
:::

### The Bonus Is Worth More Than You Think

The ::concept[upper-section bonus]{upper-section-bonus}
says 50 points in the rules. But its true impact is 72 points. Where do the extra 22 come from? Players who reach the bonus threshold tend to have been rolling well throughout the game. Good upper-section rolls also mean more options in the lower section: more straights, more full houses, more of-a-kind combinations. So hitting the bonus is not just a 50-point reward; it is a sign that the whole game went well.

:::html
<div class="chart-container" id="chart-bonus-covariance">
  <div id="chart-bonus-covariance-svg"></div>
  <p class="chart-caption">The 50-point bonus is actually worth 72 points when you account for the 22 extra points from generally better rolls.</p>
</div>
:::

:::insight
**Key insight:** This is why good Yatzy players obsess over the upper section early in the game. The 50-point bonus is visible in the rules, but the 22-point hidden advantage is not. You can only discover it by running the numbers.
:::

### Why "Best Case" Thinking Fails

What if you always played for the highest possible outcome? The theoretical ceiling is 374 points (every roll comes up perfect). But a strategy that always chases the ceiling, picking whatever move has the best possible upside, averages just 118.7 in actual play. That is less than half the optimal score.

:::html
<div class="chart-container" id="chart-max-policy-failure">
  <div id="chart-max-policy-failure-svg"></div>
  <p class="chart-caption">Always chasing the best case averages just 118.7, less than half the optimal 248.4.</p>
</div>
:::

The ceiling-chasing strategy fails because it overvalues unlikely outcomes. It will hold a single Six hoping for Yatzy when the odds are terrible, instead of taking a safe Full House. The 130-point gap between optimal play and ceiling-chasing is a vivid reminder: in a game with dice, playing the odds beats wishful thinking every time.

### What Does a Typical Game Look Like?

The solver tells us the *value* of every position, but not how often each position actually comes up. To see that, we trace a million games forward from the starting position and record what happens along the way. Three patterns emerge.

**The race to 63.** The upper-section bonus at 63 points is the single most important target in the game. The chart below tracks how upper-section scores evolve turn by turn. Most games converge toward the bonus by turns 10 to 12, but a thin stream of unlucky games stays low throughout.

:::html
<div class="chart-container" id="chart-race-to-63">
  <p class="chart-caption">How upper-section scores build turn by turn, with most games converging toward the 63-point bonus.</p>
</div>
:::

**When is each category scored?** The chart below shows, for each turn, which category the solver picks most often. High-value upper categories (Sixes, Fives, Fours) tend to get scored early, while low-value ones (Ones, Twos) and Chance get used as dump slots late in the game when the dice don't cooperate.

:::html
<div class="chart-container" id="chart-category-stream">
  <p class="chart-caption">Category timing: high-value categories early, dump slots like Ones and Chance late.</p>
</div>
:::

**How uncertainty narrows.** Before any dice are rolled, every game has the same expected score of 248. As turns pass, some games get lucky and some don't, and the range of likely outcomes fans out. By the final turn, the spread covers everything from about 120 to 370.

:::html
<div class="chart-container" id="chart-ev-ridgeline">
  <p class="chart-caption">Expected score by turn: starts as a single point at 248, fans out to the full range of final scores.</p>
</div>
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
