:::section{#part-v}

:::part-title
Part V
:::

## Multiplayer and Adaptive Play

Everything so far treats Yatzy as a solitaire optimisation problem: one player,
one score, maximise it. But Yatzy is usually played against opponents, and in
that setting the right question changes. It is no longer "what maximises my
expected score?" but "what maximises my probability of winning?"

Against an EV-optimal opponent, the EV-optimal policy wins exactly half the
time by symmetry. But a slightly risk-seeking policy --&theta; around
+0.05 to +0.07 --can gain an edge. The mechanism is straightforward:
risk-seeking play increases variance, which pushes more probability mass into
both tails. Since games are decided by the higher score, the increased upside
(p95 rising from 309 to 313) matters more than the increased downside when
trailing. The draw rate in head-to-head play is just 0.76%, with the typical
winning margin at 43.5 points.

The disagreement between &theta; = 0.07 and &theta; = 0 reveals where risk
tolerance changes decisions. In 12.5% of reroll situations and 12.4% of
category selections, the two policies diverge. These disagreements cluster
around two patterns. First,
::concept[Yatzy preservation]{backward-induction}:
at &theta; &asymp; 0.05–0.07, the solver keeps Yatzy-seeking dice
combinations in 42.8% of relevant states versus 38.8% at &theta; = 0 ---
the risk-seeking policy holds out for the 50-point jackpot more often. Second,
straight chasing: the risk-seeking solver is more willing to break a safe pair
to chase a Large Straight, accepting the lower hit rate for the higher payoff.

:::html
<div class="chart-container" id="chart-adaptive-winrate"><div id="chart-adaptive-winrate-svg"></div></div>
:::

The real insight is that the optimal &theta; depends on game state. A player
who is behind should increase &theta; to generate variance; a player who is
ahead should decrease it to protect the lead. This leads naturally to a
::concept[threshold policy]{threshold-policy}:
estimate the score differential at each decision point and select the strategy
table (indexed by &theta;) that maximises win probability given the current
gap. The adaptive policy does not require solving a new DP --it simply
switches between pre-computed strategy tables based on a score-differential
threshold.

The cost of risk-seeking is visible in category hit rates. The EV-optimal
solver achieves a 92% Full House hit rate across one million games. At
&theta; = +3.0, this collapses to 54% --the solver sacrifices reliable
mid-range categories to chase Yatzy and straights. The upper bonus achievement
rate at &theta; = 0 is 83.6%, and it declines monotonically as &theta;
increases. Both extremes converge to similarly poor mean scores: &theta; =
&minus;1.0 yields 198.2, and &theta; = +1.0 yields 194.5. Excessive caution
is as costly as excessive ambition.

:::html
<div class="chart-container" id="chart-threshold-policy"><div id="chart-threshold-policy-svg"></div></div>
:::

:::depth-2

### Win Probability and Variance

In a symmetric two-player game where both players use the same policy, the
win probability is exactly 50% minus half the draw rate. With the EV-optimal
policy, the draw rate is 0.76%, giving each player a 49.62% win probability.
The winning margin distribution has mean 43.5 points and heavy right tail
--- blowout wins of 80+ points are common, reflecting the high variance
(&sigma; &asymp; 38) of individual game scores.

For a risk-seeking player (&theta; = +0.07) facing an EV-optimal opponent,
the win rate shifts modestly. The increased variance is a double-edged sword:
p95 rises to 313 (+4 vs EV-optimal) but p5 drops to 159 (&minus;20). The
net effect on win probability depends on whether the opponent's score
distribution concentrates in the region where the risk-seeker's added
upside exceeds the added downside.

### Disagreement Anatomy

At &theta; = 0.07 versus &theta; = 0, category disagreements are concentrated
in the upper section. The risk-seeking solver is more willing to take a zero
in a lower-value upper category (Ones, Twos) to preserve the option of
chasing Yatzy or Full House later. Reroll disagreements centre on keep
decisions where a safe pair competes with a speculative straight draw. The
12.5% reroll disagreement rate translates to roughly 5–6 different keep
decisions per 15-round game.

:::

:::depth-3

### Adaptive Policy Implementation

The adaptive policy loads multiple strategy tables into memory (one per
&theta; value) and selects at runtime based on estimated score differential:

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

The strategy tables are memory-mapped (~16 MB each), so switching between
them is a pointer swap with no I/O cost. The score-gap thresholds are
derived from the mean-variance frontier: the 30-point threshold corresponds
to roughly 0.8&sigma; of the score distribution, the point where the
probability of overtaking shifts meaningfully.

```bash
# Simulate head-to-head with adaptive policy
just simulate --games 1000000 --adaptive --opponent-theta 0.0
```

:::

:::
