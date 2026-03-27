:::section{#multiplayer}

## Multiplayer and Adaptive Strategies

Most people play Yatzy as if it were a single-player game, which makes sense. Two parallel Yatzy games are independent: the dice you roll do not affect the dice your opponent rolls. Nothing you do changes what is available to them.

But here is an interesting question: if you can see your opponent's running score, can you improve your probability of *winning*?

**The short answer is: barely.** Let's say two players play 1000 games. If both play by the fixed, optimal strategy we expect them to win ~50% each. Any fixed deviation from &theta; = 0 lowers the expected score, and a lower-scoring strategy loses more often than it wins.

**Variance can be used to your advantage in two-player games.** If you are far behind your opponant, and only care about winning, then you can take more risk and hope for a big swing. The opposite is also true: if you are far ahead, variance is your enemy. This reasoning suggests we can play according to a ::concept[threshold policy]{threshold-policy}: choose &theta; based on the difference in score between you and your opponent at each turn. This policy would play conservatively when leading and aggressively when trailing. When it's close, it'd just aim to play optimally.

But before building that policy, a natural question is: what is the theoretical ceiling? If you could somehow know your opponent's final score *before the game began*, and pick the single best &theta; to play against that exact outcome, how much would it help?

We can answer this from the data. The chart below splits opponent final scores into seven bands and shows, for each band, the win rate of EV-optimal play alongside the win rate of the best possible fixed &theta; for that band. This "oracle" strategy has information no real player could have: it knows the opponent's total in advance and selects its risk posture accordingly.

:::html
<div class="chart-container" id="chart-oracle-winrate">
  <div id="chart-oracle-winrate-svg"></div>
  <p class="chart-caption">Conditional win rates per opponent score band. Gray: EV-optimal (&theta; = 0). Colored: best fixed &theta; for each band, colored by the optimal &theta; value (blue = conservative, orange = risk-seeking).</p>
</div>
:::

When the opponent scored below 200, the oracle plays slightly conservative (&theta; = &minus;0.03), protecting a lead that was almost certainly large. When the opponent scored above 280, it plays risk-seeking (&theta; = +0.04), because only tail outcomes can win against a strong score. The transition from negative to positive &theta; happens near the mean (around 250), where the two players' distributions overlap most.

The overall oracle win rate is **50.15%**: a gain of +0.146 percentage points over EV-optimal. That is the ceiling for this class of strategies (pick the best constant &theta; given advance knowledge of the opponent's final score band). A dynamic policy that switches &theta; turn by turn could in principle do slightly better, but the room is vanishingly small. The score distributions of the two players overlap so heavily that knowing which band the opponent lands in buys almost nothing. Each band offers a fraction of a percentage point of improvement, and the weighted sum across bands is tiny.

The practical takeaway: Yatzy is, for all intents and purposes, a single-player game. Playing for the highest expected score is also the best way to win. A threshold policy that adjusts &theta; based on the running score gap is directionally correct, but the gain it can extract is so small that it would take millions of games to measure it reliably.

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

The strategy tables are memory-mapped (~16 MB each), so switching between them is a pointer swap with no I/O cost. The 30-point gap threshold corresponds to roughly 0.8&sigma; of the score distribution, where the probability of overtaking shifts meaningfully.

```bash
# Simulate head-to-head with adaptive policy
just simulate --games 1000000 --adaptive --opponent-theta 0.0
```

:::

:::
