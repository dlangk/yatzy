# Applications and Appendices

## 1. Player Card: Simulation-Backed Profile Analytics

After quiz completion, the player's estimated (θ, β, γ, d) parameters are translated into tangible gameplay metrics via pre-computed Monte Carlo simulations. The Player Card shows the player's expected score distribution and pinpoints where each cognitive imperfection costs them the most points.

### 1.1 Pre-Computed Simulation Grid

The blog is fully static (no server), so all simulation results are pre-computed for a 4D parameter grid:

| Parameter | Grid values | Count |
|---|---|---|
| θ | −0.05, −0.02, 0, 0.02, 0.05, 0.1 | 6 |
| β | 0.5, 1.0, 2.0, 4.0, 7.0, 10.0 | 6 |
| γ | 0.3, 0.6, 0.8, 0.9, 0.95, 1.0 | 6 |
| d | 8, 20, 999 | 3 |

Total: 648 combinations × 10K games each (~2.5 min on Apple Silicon). Output: `blog/data/player_card_grid.json` (~82 KB).

The simulation uses `simulate_game_profiled` — a streamlined noisy agent that plays full games using softmax(β·Q) action selection with γ-discounted values and depth noise σ_d. Q-value computation always uses EV-mode (never risk-sensitive LSE) — θ only affects which strategy table (state values) backs the decisions, matching the profiling estimator's model.

For each grid point, the simulation records: mean, std, p5, p10, p25, p50, p75, p90, p95, p99, and bonus rate. An optimal baseline (θ=0, perfect decisions) is also simulated for comparison.

### 1.2 Counterfactual Coaching

The key insight is that counterfactual analysis requires only 4 extra lookups. For a player with estimated parameters (θ̂, β̂, γ̂, d̂), each counterfactual fixes one parameter to its optimal value while keeping the others at the player's estimated values:

| Counterfactual | Lookup | Measures |
|---|---|---|
| Fix θ → 0 | (0, β̂, γ̂, d̂) | Cost of non-neutral risk attitude |
| Fix β → 10 | (θ̂, 10, γ̂, d̂) | Cost of imprecise decisions |
| Fix γ → 1.0 | (θ̂, β̂, 1.0, d̂) | Cost of myopia |
| Fix d → 999 | (θ̂, β̂, γ̂, 999) | Cost of shallow lookahead |

The difference in mean score between the counterfactual and the player's actual grid entry quantifies the point cost of each imperfection. These costs overlap (fixing one parameter changes the impact of others), so their sum exceeds the total gap to optimal. The Player Card displays these as "coaching bars" sorted by impact, with a note about overlap.

### 1.3 Spot-Check Validation

Grid entries at parameter extremes confirm expected behavior:

| Parameters | Expected | Observed mean |
|---|---|---|
| θ=0, β=10, γ=1.0, d=999 | ≈ optimal (248) | 248.3 |
| θ=0, β=0.5, γ=0.3, d=8 | Near-random (~140) | 155.3 |
| θ=0, β=2.0, γ=0.8, d=20 | Intermediate | ~195 |

The near-optimal entry (β=10, γ=1.0, d=999) matches the optimal baseline within sampling noise, confirming that the noisy agent model degrades gracefully to optimal play.


## 2. Open Questions

**Head-to-head win rate.** Maximizing E[score] and maximizing P(my\_score > opponent\_score) are different objectives. Against an opponent playing θ = 0, does there exist a θ that wins > 50% of head-to-head games? The opponent's full score distribution is computable from existing simulations; win rate for any challenger θ is an O(n²) convolution. The interesting case is whether distributional shape (skewness from the bonus cliff) can be exploited beyond what mean advantage predicts. An EV-vs-EV baseline (1M games, two θ=0 players) confirms the null: 49.5% vs 49.7% wins (0.76% draws), avg winning margin 43.5 points.

**Non-exponential utility.** The constant-θ frontier is tight under exponential utility. Whether strategies optimal under other utility classes (prospect-theoretic, rank-dependent) reach points inaccessible to any θ remains open.

**Optimal θ\* per percentile.** The coarse θ sweep (Section 2) identifies approximate peak-θ values per quantile, but the grid spacing (Δθ = 0.005–0.01) limits precision. A two-phase adaptive sweep — coarse grid to locate peaks, fine grid (Δθ = 0.002) around each peak — would pin down θ\* to ±0.002 for each percentile from p1 through p99.99.

**Profiling with more scenarios.** The current 30-scenario quiz achieves good recovery (θ RMSE 0.033, γ RMSE 0.197), but β recovery degrades for near-deterministic players (β > 4). Increasing to 40–50 scenarios or using adaptive scenario selection could improve β identifiability in the high-precision regime.


---

## Appendix A: Human Strategy Guide

Based on gap analysis of 100K games comparing a pattern-matching heuristic (mean 166) against the EV-optimal solver (mean 248). See the human cognition analysis for the full analytical treatment.

### Upper Bonus Targets

You need exactly 63 across all six upper categories (averaging 3-of-each-face). Surpluses in high categories offset shortfalls in low ones.

| Category | Target (3x) | Surplus from 4x | Surplus from 5x |
|----------|-------------|-----------------|-----------------|
| Ones     | 3           | +1              | +2              |
| Twos     | 6           | +2              | +4              |
| Threes   | 9           | +3              | +6              |
| Fours    | 12          | +4              | +8              |
| Fives    | 15          | +5              | +10             |
| Sixes    | 18          | +6              | +12             |
| **Total**| **63**      |                 |                 |

### Decision Flowchart

1. **Completed pattern?** (Yatzy, straight, full house) — Score it, unless the same dice give equal or better value in an open upper category.
2. **At-or-above par in an upper category?** — Strong candidate, but compare against available lower-section alternatives. Prefer upper when bonus is still reachable.
3. **High pair or better?** — Keep it. Reroll the rest targeting upper bonus faces or pattern completion.
4. **Nothing special?** — Keep your highest die(s) matching open upper categories. Reroll the rest.

### Dump Priority (When Scoring Zero)

1. Ones (max 5, contributes least to bonus)
2. Twos (max 10)
3. Yatzy (only 4.6% hit rate — if unscored by late game, dump it)
4. Small Straight (15 pts but hard to get late)
5. Large Straight (20 pts but very hard to get late)

### When to Abandon the Bonus Chase

Past turn 10, if more than 15 points below bonus pace (needing 4x or better in all remaining upper categories), shift to maximizing raw score: take highest-scoring available category, favor Chance for high-sum hands, dump low-value categories.


## Appendix B: Human-Plausible State Filtering

The solver computes optimal decisions for ~2M states, but most are irrelevant for human play. States can be rare for two independent reasons: (1) the dice sequences needed are extremely unlikely regardless of strategy, or (2) the upstream decisions needed are ones no human would make. Useful analysis requires filtering to the intersection.

### Methodology

Simulate games under the softmax choice model used in profiling:

$$P(a \mid s, \theta, \beta) \propto \exp(\beta \cdot V_\theta(a, s))$$

At β ≈ 0.3: obvious decisions (EV gap > 5) are followed >95% of the time; close decisions (EV gap < 1) are followed ~55-65%; catastrophic errors (gap > 15) are extremely rare. Mean score falls in the 210-235 range, representing a decent experienced player.

At each turn, sort states by visit frequency under the softmax policy. The smallest set covering 90% of games defines the human-plausible state set. This adapts to the hourglass structure: ~1 state at turn 1, a few thousand at the diffuse mid-game peak, ~20 at turn 15.

Each state in the filtered set is annotated with: optimal action and EV, second-best action and EV gap (decision difficulty), and θ-sensitivity (which θ values change the optimal action). This dataset supports worked examples, difficulty maps, θ-sensitive scenario cards, and heuristic policy design.


## Appendix C: Full θ Sweep Data

**37 θ values, 1M games each:**

| θ | Mean | Std | p5 | p50 | p95 | p99 | Max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| −3.00 | 188.5 | 37.0 | 139 | 181 | 257 | 293 | 344 |
| −2.00 | 190.6 | 36.9 | 142 | 183 | 259 | 295 | 343 |
| −1.50 | 192.5 | 37.0 | 143 | 185 | 261 | 296 | 343 |
| −1.00 | 198.2 | 37.9 | 148 | 191 | 269 | 300 | 344 |
| −0.75 | 203.2 | 38.6 | 151 | 196 | 275 | 303 | 341 |
| −0.50 | 211.6 | 39.5 | 156 | 209 | 283 | 308 | 345 |
| −0.30 | 221.7 | 39.8 | 161 | 226 | 290 | 313 | 356 |
| −0.20 | 227.5 | 39.3 | 165 | 233 | 294 | 315 | 353 |
| −0.15 | 231.8 | 38.9 | 167 | 237 | 298 | 317 | 354 |
| −0.10 | 239.2 | 35.8 | 172 | 241 | 300 | 319 | 355 |
| −0.07 | 242.4 | 35.0 | 175 | 243 | 302 | 320 | 355 |
| −0.05 | 245.1 | 35.7 | 179 | 245 | 305 | 322 | 357 |
| −0.04 | 246.5 | 36.3 | 182 | 246 | 306 | 322 | 357 |
| −0.03 | 247.3 | 36.7 | 183 | 247 | 307 | 323 | 350 |
| −0.02 | 247.9 | 37.2 | 183 | 248 | 308 | 323 | 351 |
| −0.015 | 248.1 | 37.4 | 182 | 248 | 308 | 324 | 351 |
| −0.01 | 248.3 | 37.7 | 182 | 248 | 308 | 324 | 356 |
| −0.005 | 248.4 | 38.0 | 181 | 249 | 309 | 324 | 352 |
| 0.000 | 248.4 | 38.5 | 179 | 249 | 309 | 325 | 352 |
| +0.005 | 248.3 | 39.0 | 177 | 249 | 310 | 325 | 353 |
| +0.010 | 248.2 | 39.4 | 175 | 249 | 310 | 325 | 353 |
| +0.015 | 248.0 | 39.8 | 174 | 248 | 310 | 326 | 359 |
| +0.020 | 247.7 | 40.3 | 172 | 248 | 311 | 326 | 359 |
| +0.030 | 246.9 | 41.3 | 169 | 247 | 311 | 326 | 359 |
| +0.040 | 245.7 | 42.3 | 166 | 245 | 312 | 327 | 359 |
| +0.050 | 244.5 | 43.2 | 164 | 244 | 312 | 327 | 359 |
| +0.070 | 240.9 | 45.3 | 159 | 241 | 313 | 328 | 362 |
| +0.100 | 235.9 | 46.9 | 155 | 237 | 312 | 329 | 362 |
| +0.150 | 229.2 | 47.9 | 152 | 231 | 311 | 329 | 362 |
| +0.200 | 223.8 | 48.4 | 148 | 225 | 308 | 329 | 358 |
| +0.300 | 215.7 | 48.6 | 142 | 215 | 301 | 327 | 358 |
| +0.500 | 204.7 | 47.6 | 135 | 200 | 287 | 319 | 361 |
| +0.750 | 198.1 | 47.0 | 130 | 192 | 280 | 313 | 360 |
| +1.000 | 194.5 | 46.3 | 128 | 188 | 276 | 309 | 359 |
| +1.500 | 189.2 | 45.5 | 125 | 182 | 270 | 305 | 359 |
| +2.000 | 186.0 | 44.8 | 123 | 179 | 266 | 300 | 359 |
| +3.000 | 186.5 | 44.2 | 124 | 179 | 266 | 301 | 359 |

## Appendix D: Visualization Index

**θ sweep:**
- `percentiles_vs_theta.png` / `percentiles_vs_theta_zoomed.png` — Score percentiles vs θ
- `cdf_full.png` — Full CDF curves colored by θ
- `tails_zoomed.png` — Survival curves for upper and lower tails
- `density.png` — KDE density estimates across θ
- `combined.png` — Multi-panel composite (CDF, density, percentiles, mean-std)

**Efficiency:**
- `efficiency.png` — MER, SDVA, CVaR deficit, combined frontier

**Mean-variance:**
- `mean_vs_std.png` — Two-branch structure in (μ, σ) space

**Per-category:**
- `category_score_heatmaps.png` / `category_rate_heatmaps.png` / `category_fill_turn_heatmaps.png`
- `category_sparklines_mean_score.png` / `category_sparklines_zero_rate.png` / `category_sparklines_mean_fill_turn.png`
- `category_bars.png` / `category_slope.png` / `category_slope_dense.png`

**Conditional Yatzy:**
- `yatzy_conditional_bars.png` — Hit rate by score band and θ
- `yatzy_unconditional_vs_tail.png` — Unconditional vs top-5% hit rate
- `yatzy_dump_gap.png` — Mean score gap between hit/miss games

**State flow:**
- `state_flow_bonus_alluvial.png` / `state_flow_dag.png` / `state_flow_streamgraph.png`

**Sensitivity:**
- `sensitivity_flip_rates.png` / `sensitivity_flip_theta.png` / `sensitivity_gaps.png` / `sensitivity_heatmap.png`
- `scenario_card_{hex_id}.png` — Individual decision flip scenarios

**Frontier:**
- `frontier_pareto.png` — Constant-θ curve vs adaptive policies
- `frontier_cdf.png` — Overlaid CDFs
- `frontier_delta.png` — Δμ bar chart

**Multiplayer (EV vs EV baseline):**
- `mp_score_scatter.png` — Player 1 vs Player 2 score scatter
- `mp_score_diff_hist.png` — Score difference distribution
- `mp_score_diff_per_turn.png` — Cumulative score difference by turn
- `mp_trajectories.png` — Sample game score trajectories
- `mp_win_margin.png` — Win margin distribution

**Modality (why scores aren't normal):**
- `modality_histogram_vs_kde.png` — Histogram vs KDE at narrow and reasonable bandwidths
- `modality_bonus_yatzy.png` — 2×2 decomposition by (bonus, Yatzy) sub-populations with normal fits
- `modality_category_pmf.png` — 3×5 per-category score PMF grid (orange = binary categories)
- `modality_variance_decomposition.png` — Variance and covariance contribution per category
- `modality_mixture_waterfall.png` — Impact of binary categories on total score (hit rate × Δmean)

**Surrogate compression:**
- `surrogate_pareto.png` — Pareto frontier: log(params) vs EV loss per game, DTs and MLPs across 3 decision types
- `surrogate_dt_vs_mlp.png` — Decision tree vs MLP comparison at matched parameter budgets
- `surrogate_feature_importance.png` — Random forest feature importances per decision type
- `surrogate_accuracy_by_turn.png` — Model accuracy heatmap by turn number

**Heuristic gap analysis:**
- `outputs/heuristic_gap/heuristic_gap.csv` — Per-decision disagreements with EV gaps
- `outputs/heuristic_gap/heuristic_gap_summary.json` — Top-20 mistake patterns and breakdown by decision type

**Other:**
- `quantile.png` — Quantile-quantile or quantile function plot
