:::section{#optimal-strategy}

## The Optimal Strategy

The optimal strategy exists in the form of ~1.43M numbers calculated by the solver. That means to play perfectly, we need ~1.43M parameters. In this section, we will explore what this strategy looks like, and see if there is anything we can learn in order to get better at playing Yatzy!

Since we cannot comprehend ~1.43M parameters, we need to find other ways to see how the optimal strategy behaves. We will start by searching for patterns. To do that, we simulate 1 million games played based on the optimal strategy. Why 1 million? Statistical precision scales with the square root of sample size. At 1 million games, averages are accurate to within a fraction of a point, and even rare events (like scoring zero on One Pair, which happens less than 0.1% of the time) show up hundreds of times. Going to 10 million would barely improve the precision while taking 10x longer to run. Going down to 100,000 would leave some of the rarer statistics too noisy to trust.

### The Shape of the Outcome

Let's start by looking at the big picture: The score distribution.

:::html
<div class="chart-container" id="chart-mixture">
  <div id="chart-mixture-svg"></div>
  <p class="chart-caption">Score distribution under optimal play (1M games). The 16 colored components are the sub-populations from four binary events: Bonus, Yatzy, Small Straight, Large Straight. Use the toggles to filter by hit/miss for each event. Dashed line: true (exact) distribution.</p>
</div>
:::

:::html
<div class="headline-stats">
  <div class="headline-stat"><div class="headline-stat-value">248.4</div><div class="headline-stat-label">Mean</div></div>
  <div class="headline-stat"><div class="headline-stat-value">249</div><div class="headline-stat-label">Median</div></div>
  <div class="headline-stat"><div class="headline-stat-value">38.5</div><div class="headline-stat-label">Std Dev</div></div>
  <div class="headline-stat"><div class="headline-stat-value">179</div><div class="headline-stat-label">P5</div></div>
  <div class="headline-stat"><div class="headline-stat-value">309</div><div class="headline-stat-label">P95</div></div>
</div>
:::

It's clearly not a normal distribution. The shape has bumps and shoulders that a single bell curve cannot explain. The reason: four categories in Yatzy are binary. You either score them or you don't, and each one shifts your total by a fixed amount. The upper-section bonus adds 50 points (hit ~90% of the time). Yatzy adds 50 points (~39%). Large Straight adds 20 points (~49%). Small Straight adds 15 points (~26%).

These four yes/no events create 16 sub-populations. Each sub-population is roughly Gaussian in its "residual" score (the non-binary categories), centered around 155 to 177 points, then shifted upward by whichever binary events fired. The peaks in the overall distribution appear where multiple sub-populations pile up at similar total scores. For example, the shoulder near 254 is where "Bonus + Large Straight + no Yatzy" (mean 244) overlaps with "Bonus + Small Straight + Large Straight" (mean 262) and "Bonus + Yatzy only" (mean 270). Use the toggles above to isolate each event and see how it splits the distribution.

### Understanding the Patterns

:::html
<div class="chart-container" id="chart-category-landscape">
  <p class="chart-caption">Each bubble is one of the 15 categories. Click an insight to see a guided view, or use the dropdowns to explore freely. Upper section in blue, lower in red.</p>
</div>
:::

The next chart shows how category scores correlate.

:::html
<div class="chart-container" id="chart-category-correlations">
  <p class="chart-caption">Pairwise correlation between category scores (including bonus). Blue = negative, red = positive. Color scale clamped to the off-diagonal range.</p>
</div>
:::

Three patterns are visible:

**Firstd** of all, the bonus row dominates. Bonus correlates with Sixes (+0.33), then Fives (+0.26), Fours (+0.22), and so on in value order. The bonus also correlates with lower-section categories. We know that the zero rate for Four of a Kind (+0.14) drops from 48% to 27% in games that secure the bonus. This is explained by the solver scoring zero in this category more often when still chasing the upper section bonus. In no-bonus games the solver fills Chance earlier (mean turn 7.3 vs 8.5) as a dump for mediocre rolls while still chasing upper categories. In bonus games, Chance can wait for a better roll.

**Second**, every upper-upper category pair shows negative correlation. The explanation for this is that upper categories compete for turns. If the solver spends an early turn scoring Sixes, that is one fewer turn available to retry Ones. Since the solver prioritizes high-value categories, the low-value ones get whatever is left. The further apart two categories are in value, the stronger the tradeoff.

**Third**, Yatzy is uncorrelated with almost everything. Hitting or missing Yatzy tells you essentially nothing about the rest of your game. It is a pure lottery ticket.

The next chart shows when each category gets filled.

:::html
<div class="chart-container" id="chart-fill-turn-heatmap">
  <p class="chart-caption">Probability of filling each category on each turn. Categories in scorecard order, with the bonus row showing when the last upper category is filled.</p>
</div>
:::

High-value upper categories, like Sixes, Fives and Fours, tend to be filled early, while Ones and Twos are deferred. Lower-section categories are bimodal: Two Pairs and Full House peak at turn 1 (grab them if you get them), while Four of a Kind and Large Straight peak at turn 15 (last resort).

The toggle reveals that what looks like one strategy is actually two different games superimposed. Since ~90% of games hit the bonus, the "Bonus scored" view is nearly identical to "All games." The real information is in "Bonus missed."

In no-bonus games, the solver fills lower-section categories earlier (Small Straight shifts from mean turn 8.4 to 6.5, One Pair from 7.3 to 6.1) to free up late turns for upper-section retries. The result: turn 15 belongs exclusively to upper categories (Sixes 24%, Fives 19%, Fours 19%), and the bonus row spikes to 93% at turn 15. These are not zero dumps. Sixes scored on turn 15 in no-bonus games average 9.5 points with only a 9% zero rate. The solver was genuinely trying to score well on these categories the whole game; the dice just never cooperated, so the upper categories kept getting pushed later until one landed on the final turn.

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
