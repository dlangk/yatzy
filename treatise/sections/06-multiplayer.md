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

The overall oracle win rate is **50.14%**, against an EV-vs-EV baseline of 49.58% (the baseline sits below 50% because draws occur in about 0.8% of games): a gain of roughly +0.56 percentage points. That is the ceiling for this class of strategies (pick the best constant &theta; given advance knowledge of the opponent's final score band). A dynamic policy that switches &theta; turn by turn could in principle do slightly better, but the room is small. The score distributions of the two players overlap so heavily that knowing which band the opponent lands in buys little: six of the seven bands gain between 0.07 and 0.68 percentage points, and only the rare 280+ band gains more (1.26 points).

The practical takeaway: Yatzy is, for all intents and purposes, a single-player game. Playing for the highest expected score is also the best way to win. A threshold policy that adjusts &theta; based on the running score gap is directionally correct, but the gain it can extract is so small that it would take millions of games to measure it reliably.

:::math

### Adaptive Policy Implementation

The adaptive policies load multiple strategy tables into memory (one per
&theta; value, memory-mapped at ~16 MB each so switching is a pointer swap)
and select a table at runtime. Two families exist in the solver:

- **Solitaire adaptive policies** (`solver/src/simulation/adaptive.rs`):
  `select_theta_index(upper_score, scored, turn)` keys the &theta; choice on
  the player's own board state (e.g. bonus progress), not on an opponent.
  Run them with `yatzy-simulate --policy NAME`.
- **Opponent-aware strategies** (`solver/src/simulation/multiplayer.rs`):
  the multiplayer engine tracks both boards and picks &theta; from the score
  differential and turns remaining.

```bash
# Solitaire adaptive policy (see POLICY_CONFIGS for names)
just simulate games=1000000 -- --policy bonus-guard

# Opponent-aware head-to-head
just multiplayer args="--players ev,adaptive --games 1000000"
```

:::

:::
