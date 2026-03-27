:::section{#multiplayer}

## Multiplayer and Adaptive Strategies

Most people play Yatzy as if it was a single-player game, which makes sense. Two parallel Yatzy games are independent: the dice you roll do not affect the dice your opponent rolls. Nothing you do changes what is available to them.

But here is an interesting question: if you can see your opponent's running score, can you improve your probability of *winning*?

No fixed &theta; beats EV-optimal in head-to-head play. Two EV-optimal players each win exactly 50% of games by symmetry, and any deviation from &theta; = 0 reduces expected score, which reduces win probability. The best fixed challenger (&theta; = &minus;0.03) wins only 48.6% of games against an EV-optimal opponent.

Variance does matter, but only conditionally. When trailing, high variance is useful: you need a lucky swing, and a strategy that reliably adds points is nearly worthless if you cannot catch up. When leading, high variance is dangerous: it gives your opponent a chance to overtake. In a close game, EV-optimal play is correct.

This gives rise to a ::concept[threshold policy]{threshold-policy}: select &theta; at each turn based on the current score gap. But what is the theoretical ceiling? How much can adaptive play actually gain?

The chart below shows the answer. The bars compare two strategies for every possible range of opponent final scores: playing EV-optimal (&theta; = 0) the whole game versus playing the single best fixed &theta; for that score band. The colored bars are the oracle: it knows the opponent's final score *before the game begins*, and plays the optimal fixed &theta; for that band throughout.

:::html
<div class="chart-container" id="chart-oracle-winrate">
  <div id="chart-oracle-winrate-svg"></div>
  <p class="chart-caption">Oracle upper bound: conditional win rates for EV-optimal vs best fixed &theta; per opponent score band. The oracle knows the opponent's final score before the game starts and selects the best possible &theta; for that band.</p>
</div>
:::

The directional logic is clear: against a weak opponent (scored below 200) you want to play conservatively (negative &theta;), protecting your lead by minimising variance. Against a strong opponent (scored above 280) you need to have taken risks (positive &theta;), because cautious play cannot produce the tail scores required to win.

The overall oracle win rate is **50.15%**, a gain of just +0.146 percentage points over EV-optimal. This is the absolute theoretical ceiling for strategy-switching based on the opponent's *final* score, even with perfect advance knowledge. No adaptive policy, however sophisticated, can exceed this bound.

The reason the gain is so small is that the score distributions of the two players overlap heavily. A conservative adjustment that helps when the opponent scores low hurts roughly equally when the opponent scores high, and vice versa. The gain from knowing which regime you are in does not compound: each band offers only a fraction of a percentage point of room, and the weighted sum across bands is tiny.

A practical ::concept[threshold policy]{threshold-policy} does not have oracle knowledge. It selects &theta; based on the *current* score gap during the game, not the opponent's eventual total. The 30-point threshold corresponds to roughly 0.8 standard deviations of the score distribution, the point where the probability of overtaking shifts meaningfully.

:::math

### Adaptive Policy Implementation

The adaptive policy loads multiple strategy tables into memory (one per &theta; value) and selects at runtime based on the estimated score differential:

```rust
// Adaptive θ selection based on score gap
fn select_theta(my_score: f32, opp_estimate: f32, turns_left: u8) -> f32 {
    let gap = my_score - opp_estimate;
    let urgency = 1.0 - (turns_left as f32 / 15.0);

    if gap < -30.0 * (1.0 - urgency) {
        0.10  // Far behind: maximum risk-seeking
    } else if gap < -10.0 {
        0.05  // Moderately behind: mild risk-seeking
    } else if gap > 30.0 {
        -0.03 // Far ahead: protect the lead
    } else {
        0.0   // Close game: EV-optimal
    }
}
```

The strategy tables are memory-mapped (~16 MB each), so switching between them is a pointer swap with no I/O cost. The score-gap thresholds are derived from the mean-variance frontier: the 30-point threshold corresponds to roughly 0.8&sigma; of the score distribution, the point where the probability of overtaking shifts meaningfully.

```bash
# Simulate head-to-head with adaptive policy
just simulate --games 1000000 --adaptive --opponent-theta 0.0
```

:::

:::
