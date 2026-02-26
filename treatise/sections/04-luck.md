:::section{#part-iii}

:::part-title
Part III
:::

## Luck, Variance, and the Shape of Outcomes

A solved game does not mean a predictable game. Even under perfect play, a
single game of Yatzy is startlingly variable. The optimal policy produces a
mean score of 248.4 with a standard deviation of 38.5 --a coefficient
of variation above 15%. Two games played with identical, flawless strategy can
easily differ by 100 points or more. Understanding where this variance comes
from, and why it has the particular shape it does, is one of the most revealing
aspects of the analysis.

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

:::depth-3

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
