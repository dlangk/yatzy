:::section{#optimal-strategy}

## The Optimal Strategy

The solver has computed the optimal action for every reachable position: 1.43 million numbers. To make sense of that, we look at what happens to each of the 15 categories across a million simulated games.

### The Category Landscape

Not all 15 boxes on the scorecard are equal. The chart below maps each category by when it gets scored (x-axis), how close to its ceiling it scores (y-axis), and how much it contributes to final score variance (bubble size).

:::html
<div class="chart-container" id="chart-category-landscape">
  <p class="chart-caption">Each bubble is a category. Position shows when and how well it scores; size shows variance contribution. Upper section in blue, lower in red.</p>
</div>
:::

Three clusters stand out. Top-left: Sixes, Fives, and Fours, scored early and close to ceiling. These are the earners. Bottom-right: Ones and Twos, scored late and far below ceiling. These are the dump slots, sacrificed to preserve options elsewhere. In between: the lower-section categories, each with a distinct strategic role. And one enormous outlier: Yatzy, whose bubble dwarfs everything else because it swings between 50 and 0.

### The Upper Section and the Bonus

The 50-point upper bonus at 63 points is the gravitational center of the entire strategy. The solver pursues upper categories in a clear hierarchy: Sixes first, then Fives, then Fours, then Threes. Each contributes more toward the 63 threshold and scores more raw points. Ones and Twos sit at the bottom: their ceilings (5 and 10) are so low that dumping a zero there barely hurts.

The chart below tracks upper-section score evolution across 15 turns. Most games converge toward the bonus by turns 10 to 12. A thin stream of unlucky games stays low throughout.

:::html
<div class="chart-container" id="chart-race-to-63">
  <p class="chart-caption">How upper-section scores build turn by turn, with most games converging toward the 63-point bonus.</p>
</div>
:::

**The bonus is worth 72, not 50.** The rules say 50 points, but the true impact is 72. Where do the extra 22 come from? Players who reach the bonus tend to have been rolling well throughout, which lifts their lower-section scores too. About 12 points come from the upper-section scores themselves being higher when the threshold is reached, and about 10 from genuine positive correlation between upper and lower sections.

:::html
<div class="chart-container" id="chart-bonus-covariance">
  <div id="chart-bonus-covariance-svg"></div>
  <p class="chart-caption">The 50-point bonus is actually worth 72 points when you account for the 22 extra points from generally better rolls.</p>
</div>
:::

:::insight
**Key insight:** This is why good Yatzy players obsess over the upper section early in the game. The 50-point bonus is visible in the rules, but the 22-point hidden advantage is not. You can only discover it by running the numbers.
:::

### The Lower Section: Roles and Wildcards

The nine lower-section categories play very different roles. They fall into four groups.

**Reliable scorers.** One Pair, Three of a Kind, and Chance are easy to fill. Their zero rates are tiny, and Chance accepts any five dice. These categories provide the scoring floor: steady points that are rarely wasted.

**Pattern specialists.** Two Pairs, Full House, and Four of a Kind need specific dice patterns. They have moderate zero rates (7% to 13%). The solver pursues them when dice align and takes zeros when they don't.

**Binary outcomes.** Small Straight (15 points) and Large Straight (20 points) are fixed-value patterns. Either you hit the sequence or you score zero. No partial credit.

**The lottery ticket.** Yatzy is worth 50 points but only hit about 39% of the time. It accounts for the single largest chunk of score variance in the entire game.

:::html
<div class="chart-container" id="chart-category-pmf-grid">
  <p class="chart-caption">Score distributions for each category. Some are smooth (upper categories), some are binary spikes (straights, Yatzy), and some fall in between.</p>
</div>
:::

### How a Game Unfolds

We have seen the roles. Now let's watch them play out. The chart below shows which category gets scored at each turn, aggregated across a million games.

:::html
<div class="chart-container" id="chart-category-stream">
  <p class="chart-caption">Category timing: high-value categories early, dump slots like Ones and Chance late.</p>
</div>
:::

The strategy has phases. Early game: invest in the bonus by accumulating Sixes, Fives, and Fours while keeping lower-section options open. Mid-game: fill pattern categories (straights, full house, pairs) as dice allow. Late game: damage control. Ones, Twos, and Chance absorb whatever is left.

The chart below shows how uncertainty evolves. Before any dice are rolled, every game has the same expected score of 248. As turns pass, outcomes fan out based on dice luck and category choices. By the final turn, the spread covers everything from about 120 to 370.

:::html
<div class="chart-container" id="chart-ev-ridgeline">
  <p class="chart-caption">Expected score by turn: starts as a single point at 248, fans out to the full range of final scores.</p>
</div>
:::

Even under perfect play, Yatzy is fundamentally a game of managed uncertainty. The solver does not eliminate luck; it manages it.

### The Shape of the Outcome

All of the above produces one final number each game. The distribution is not a bell curve. Two binary events dominate: whether you hit the
::concept[upper-section bonus]{upper-section-bonus}
(about 90% of the time under perfect play) and whether you scored Yatzy (about 39%). These two yes-or-no outcomes split all games into four sub-populations:

- Missed bonus, no Yatzy: average around 164
- Missed bonus, scored Yatzy: average around 213
- Hit bonus, no Yatzy: average around 237
- Hit bonus, scored Yatzy: average around 288

The overall distribution is these four groups stacked together. The bonus creates a visible dip where the two populations (90% vs 10%) meet.

:::html
<div class="chart-container" id="chart-mixture">
  <div id="chart-mixture-svg"></div>
  <p class="chart-caption">The score distribution is a blend of four groups: bonus hit/miss x Yatzy hit/miss. Toggle to highlight each component.</p>
</div>
:::

The non-normal shape now makes sense. It is not a failure of averaging; it is a fundamental property of a game where the two highest-value events are all-or-nothing.

Headline stats: mean 248.4, standard deviation 38.5, median 249, 5th percentile 179, 95th percentile 309.

### Explore a Position

The interactive tool below connects to the live solver API. Set the dice and scorecard state, then see the solver's evaluation: which dice to keep, which category to score, and the expected total for every option.

:::html
<div class="chart-container" id="chart-position-explorer">
  <p class="chart-caption">Set any position and see the solver's optimal play. Requires the backend (<code>just dev-backend</code>).</p>
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

The category landscape data is derived from Monte Carlo simulation.
The solver records per-category scores and fill turns for each game,
then computes means, zero rates, and variance contributions:

```rust
// category_sweep.rs — per-category statistics

for game in 0..n_games {
    let record = simulate_game(&ctx, &mut rng);
    for cat in 0..15 {
        stats[cat].add(record.category_scores[cat],
                       record.fill_turns[cat]);
    }
}
// Output: mean_score, zero_rate, mean_fill_turn,
//         score_pct_ceiling, variance_contribution
```

The position explorer uses the same `/evaluate` endpoint as the
game UI. Given dice, upper score, scored categories, and rerolls
remaining, the solver returns all 252 keep-mask EVs and 15
category EVs:

```rust
// server.rs — evaluate endpoint

async fn handle_evaluate(
    State(state): State<AppState>,
    Json(req): Json<EvaluateRequest>,
) -> Result<Json<EvaluateResponse>, StatusCode> {
    let resp = compute_roll_response(
        &state.ctx, req.dice, req.upper_score,
        req.scored_categories, req.rerolls_remaining,
    );
    Ok(Json(resp))
}
```

:::

:::
