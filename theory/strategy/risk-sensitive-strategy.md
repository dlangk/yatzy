# Risk-Sensitive Strategy

## 1. The θ Parameter

We generalize the solver to optimize exponential utility rather than expected value:

$$U(x) = -\frac{e^{-\theta x}}{\theta}$$

where θ > 0 is risk-seeking, θ < 0 is risk-averse, and θ → 0 recovers the EV-optimal policy. The DP recursion is identical except the objective at chance nodes becomes a certainty equivalent rather than an expectation.

This is CARA (constant absolute risk aversion) utility: the risk premium for a given gamble is independent of current wealth. The parameter θ controls how the solver weights upside vs downside at every decision point.

## 2. The θ Sweep

We solved the full DP for 37 θ values from −3.0 to +3.0 with progressive spacing (dense near zero: Δθ = 0.005 in [−0.015, +0.015], widening to Δθ = 1.0 at extremes). Each strategy was evaluated over 1M simulated games.

**Selected results:**

| θ | Mean | Std | p5 | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|---:|
| −1.00 | 198.2 | 37.9 | 148 | 191 | 269 | 300 |
| −0.10 | 239.2 | 35.8 | 172 | 241 | 300 | 319 |
| −0.05 | 245.1 | 35.7 | 179 | 245 | 305 | 322 |
| **0.00** | **248.4** | **38.5** | **179** | **249** | **309** | **325** |
| +0.05 | 244.5 | 43.2 | 164 | 244 | 312 | 327 |
| +0.07 | 240.9 | 45.3 | 159 | 241 | 313 | 328 |
| +0.10 | 235.9 | 46.9 | 155 | 237 | 312 | 329 |
| +0.20 | 223.8 | 48.4 | 148 | 225 | 308 | 329 |
| +1.00 | 194.5 | 46.3 | 128 | 188 | 276 | 309 |

**Best θ per quantile:**

| Metric | Best θ | Value |
|---:|---:|---:|
| p5 | −0.03 | 183 |
| p50 | −0.005 | 249 |
| p75 | +0.04 | 278 |
| p95 | +0.07 | 313 |
| p99 | +0.10 | 329 |

The optimal θ shifts from negative (risk-averse) for lower quantiles to positive (risk-seeking) for upper quantiles. This is a fundamental property of the risk-return tradeoff: protecting the floor requires different decisions than lifting the ceiling.

## 3. Two-Branch Structure in Mean-Variance Space

The (mean, σ) curve is not a single curve. It traces two distinct branches:

- **Risk-averse branch** (θ < 0): σ stays nearly flat at 35–38 as mean drops from 248 to 182. Variance compresses without moving much.
- **Risk-seeking branch** (θ > 0): σ rises to 48.4 (at θ ≈ 0.2), then falls back to 44 as mean drops from 248 to 186.

At the same mean (~190), the risk-seeking branch (θ ≈ 1) has σ ≈ 46 vs σ ≈ 37 for the risk-averse branch (θ ≈ −1.5). A quadratic fit across all points gives R² = 0.37 — confirming these are two distinct branches, not a single frontier.

This is analogous to a mean-variance frontier in portfolio theory, except the classical efficient frontier is a single hyperbola. The two-branch structure arises because risk-averse and risk-seeking policies achieve the same mean loss through fundamentally different mechanisms.

## 4. Near-Origin Behavior

Near θ = 0, both mean and std are well-approximated by quadratics:

$$\text{mean}(\theta) \approx 247.0 - 0.9\theta - 618\theta^2$$
$$\text{std}(\theta) \approx 39.5 + 53.5\theta - 17\theta^2$$

Mean loss is quadratic (second-order around the optimum, as expected from perturbing a maximum). Std gain is linear in θ (dominant term: 53.5θ). This asymmetry — quadratic cost, linear benefit — creates a narrow window where tail gains are cheap.

For p95 specifically: the Gaussian approximation p95 ≈ μ + 1.645σ predicts the peak location (θ ≈ 0.05) but overpredicts the level by ~4 points (predicted 316 vs actual 313). The Gaussian model fails because the score distribution's skewness changes with θ: mildly thin-tailed near θ = 0, increasingly heavy-tailed for θ > 0.2. Mean and σ locate the p95 peak; the third moment determines its height.

## 5. The Asymmetry of Risk

**The two directions are asymmetric.** At θ = −0.05, σ drops 2.8 for a mean cost of 3.3. At θ = +0.05, σ rises 4.7 for a mean cost of 3.9. Risk-seeking buys more σ per mean point (1.21 vs 0.85) but also costs more mean in absolute terms. The asymmetry is further visible in the branch structure (Section 3): the risk-averse branch compresses σ into a narrow band (35–38) regardless of mean, while the risk-seeking branch first expands σ (peaking at 48.4) then slowly contracts it.

**Both extremes degenerate symmetrically.** At θ = −3 (mean 188.5) and θ = +3 (mean 186.5), strategies are comparably poor. Refusing all variance is as costly as embracing all variance.

**Regime boundaries:**

| Regime | θ range | Behavior |
|---|---|---|
| Near-EV | −0.01 to +0.03 | Negligible difference from θ = 0 |
| Active risk-averse | −1 to −0.01 | Variance compresses, lower quantiles improve |
| Active risk-seeking | +0.03 to +0.20 | Upper quantiles peak, mean declines ~1 pt per 0.01θ |
| Diminishing returns | \|θ\| > 0.2 | Policy changes slow, all metrics decline |
| Degenerate | \|θ\| > 1.5 | Policy frozen, no further change |

## 6. Three Efficiency Metrics

**Marginal Exchange Rate (MER).** MER_q(θ) = [mean(0) − mean(θ)] / [q(θ) − q(0)]: mean points sacrificed per point of quantile gain.

| θ | Mean cost | MER\_p95 | MER\_p99 |
|---:|---:|---:|---:|
| 0.01 | 0.2 | 0.2 | dom |
| 0.05 | 3.9 | 1.3 | 1.9 |
| 0.08 | 9.3 | 2.3 | 3.1 |
| 0.10 | 12.5 | 4.2 | 3.1 |
| 0.20 | 24.6 | dom | 6.2 |

p95 becomes dominated (declines) at θ ≥ 0.18. p99 stays positive until θ ≈ 0.30. The peak-benefit θ shifts rightward for more extreme quantiles, but the MER keeps rising.

**Stochastic Dominance Violation Area (SDVA).** For each score x, D(x) = F_θ(x) − F_0(x). Where D > 0, θ puts more mass below x (worse). We decompose:

| θ | A\_worse | A\_better | Ratio | Crossing point |
|---:|---:|---:|---:|---:|
| 0.05 | 4.5 | 0.6 | 7.2 | 271 |
| 0.08 | 9.7 | 0.4 | 22 | 285 |
| 0.10 | 12.8 | 0.3 | 38 | 294 |

At θ = 0.08, you must score above ~285 (≈ p80 of baseline) for the risk-seeking policy to be beneficial. Below that, you are strictly worse off.

**CVaR (Expected Shortfall).** Mean score in the worst α-fraction of games.

| θ | Mean | CVaR\_5% | CVaR deficit |
|---:|---:|---:|---:|
| 0.00 | 248.4 | 157.4 | — |
| 0.05 | 244.5 | 149.0 | −8.4 |
| 0.08 | 239.1 | 144.2 | −13.2 |
| 0.10 | 235.9 | 142.6 | −14.8 |

At θ = 0.08, the CVaR deficit (−13.2) exceeds the mean deficit (−9.3). Risk-seeking hurts the worst games disproportionately.


## 7. How θ Changes Decisions

### 7.1 Disagreement Rates

At each decision point, θ = 0 and θ > 0 may prescribe different actions. Measured over 100K games following θ = 0's trajectory:

| Comparison | Reroll disagree | Category disagree | Per game (of 45) |
|---|---:|---:|---:|
| θ = 0 vs θ = 0.07 | 12.5% | 12.4% | 5.6 |
| θ = 0 vs θ = 0.50 | — | 32.0% | — |
| θ = 0 vs θ = 3.00 | — | 44.2% | — |

At the mild θ = 0.07 (peak p95), 1 in 8 decisions differs.

### 7.2 The Unified Pattern: Option Value

Every category disagreement follows one pattern: θ > 0 preserves high-ceiling categories and dumps low-ceiling ones. θ = 0 optimizes for probability of use; θ > 0 optimizes for ceiling of use.

**Breakdown at θ = 0.07 (186,520 disagreements across 100K games):**

| θ = 0.07 preserves | Count | Share | Avg immediate sacrifice |
|---|---:|---:|---:|
| Yatzy (ceiling 50) | 48,598 | 26.1% | +3.0 pts |
| Straights (ceiling 15/20) | 47,269 | 25.3% | −2.3 pts (gains) |
| Upper category over lower | 58,065 | 31.1% | −0.2 pts (gains) |
| Chance (ceiling 30) | 10,933 | 5.9% | +2.9 pts |
| Full House / Two Pairs / 4K | 13,948 | 7.4% | −1.3 pts (gains) |
| Other | 7,707 | 4.2% | — |

The net immediate sacrifice across all disagreements is **0.3 points per game** (on 1.9 disagreeing decisions). The overall mean difference is 7.5 points. The gap is explained by compounding: preserving hard-to-complete categories means more turns with high-variance slots open, lower bonus rates (89.8% → 72.6%), and suboptimal reroll targeting downstream.

### 7.3 Yatzy Preservation: The Costliest Disagreement

The largest single category of disagreement. Late game (avg turn 13), θ = 0 fills Yatzy with zero; θ = 0.07 dumps on another category to keep the 50-point jackpot alive.

θ = 0 writes off Yatzy because P(five-of-a-kind) ≈ 4.6% per turn — the EV of keeping it open (~2.3 pts) is usually less than the EV of keeping other categories. θ > 0 values the upside of 50 points beyond what its probability warrants under EV.

### 7.4 Per-Category Effects Across θ

**Upper section: Ones is the dump category.** Ones absorbs zeros increasingly with θ: zero rate rises from 9.2% (θ = 0) to 38.5% (θ = 3), filled at turn 3 rather than turn 8. Sixes migrates from turn 6.5 to turn 12.4.

**Yatzy peaks at moderate θ.** Mean Yatzy score is maximized at θ ≈ 0.05–0.07 (21.4 pts, 42.8% hit rate), up from 19.4 pts (38.8%) at θ = 0. Beyond θ ≈ 0.10, over-commitment to preserving Yatzy damages other categories, and the hit rate declines.

**Straights benefit substantially.** Small Straight hit rate rises from 26% to 44% at θ = 0.20. Large Straight peaks at 57% around θ = 0.15–0.20.

**Full House degrades.** Hit rate drops from 92% (θ = 0) to 54% (θ = 3) as it becomes a dump target for preserving higher-ceiling categories.

**Timing reversal.** θ > 0 fills low-ceiling categories early and preserves high-ceiling categories for late game:

| Category | Fill turn (θ = 0) | Fill turn (θ = 3) | Shift |
|---|---:|---:|---:|
| One Pair | 7.1 | 2.8 | −4.3 |
| Ones | 8.0 | 3.1 | −4.9 |
| Fives | 6.1 | 11.7 | +5.6 |
| Sixes | 6.5 | 12.4 | +5.9 |
| Yatzy | 10.6 | 13.4 | +2.8 |

### 7.5 Conditional Yatzy Hit Rate

**Puzzle:** As θ increases, unconditional Yatzy hit rate drops from 38.8% to 17.6% (θ = 1.1). Risk-seeking strategies should chase high-variance categories. Why does the highest-ceiling category suffer?

**Two hypotheses:**
- **H0 (dump):** The drop is driven by dumping Yatzy only in poor games. In the right tail, high-θ strategies hit Yatzy equally.
- **H1 (sacrifice):** The drop is fundamental — even top games at high θ skip Yatzy.

**Result: H1 confirmed.** Three signals agree:

1. **Top-5% Yatzy hit rate drops monotonically.** At θ ≤ 0.20, 100% of top-5% games include Yatzy. At θ = 0.50, 94.2%. At θ = 1.10, 78.1%.

2. **Conditional hit rates drop in every score band**, including 300+.

3. **The dump gap is constant.** Mean score difference between Yatzy-hit and Yatzy-miss games is 50–54 points across all θ. If H0 were true, this gap should widen at high θ.

**What the solver does instead of Yatzy.** At high θ, the solver trades one 50-point lottery ticket (39% hit rate) for many smaller bets: upper-section optimization, straight completion, four-of-a-kind improvement. At θ = 0.05, both Yatzy and other categories improve. Beyond θ ≈ 0.10, Yatzy becomes a casualty of portfolio reallocation.

### 7.6 Decision Sensitivity: Where θ Flips the Optimal Action

Simulating 100K games under θ = 0 and re-evaluating every decision (4.5M total: two rerolls + one category per turn) across 12 θ values in [0, 0.2] identifies **239 flip decisions** — game states where some θ > 0 prescribes a different optimal action than θ = 0.

Out of 1,384 frequently visited decisions (≥100 visits in realistic board states), 17.3% exhibit a flip. The rate is higher for category choices (18.8%) than rerolls (15.4–17.2%), and higher early in the game (26.7% for early-game category choices) than late (9.8%).

**Flip θ distribution.** The θ at which the first flip occurs is bimodal: 39 decisions flip at θ ≤ 0.01 (near-EV decisions where the optimal action is a coin toss) and 58 flip at θ = 0.10 (a distinct cluster of structurally different decisions). The gap between θ = 0.04 and θ = 0.10 suggests two populations: fragile near-ties and genuine risk-preference-sensitive decisions.

| First flip θ | Count | Share |
|---:|---:|---:|
| 0.005 | 19 | 7.9% |
| 0.01 | 20 | 8.4% |
| 0.04 | 42 | 17.6% |
| 0.07 | 25 | 10.5% |
| 0.10 | 58 | 24.3% |
| 0.15–0.20 | 45 | 18.8% |

**Concrete examples (tightest flips):**
- **Turn 1, dice [2,4,6,6,6], category choice:** θ = 0 scores Sixes (18 pts, guaranteed). θ > 0.005 scores Three of a Kind (18 pts, preserving Sixes for a potential 30-point ceiling later). EV gap at θ = 0: 0.03 points.
- **Turn 0, dice [3,3,4,5,6], reroll 1:** θ = 0 keeps [3,3] (targeting pairs/full house). θ > 0.01 keeps [5,6] (targeting Large Straight, a higher-ceiling category). EV gap at θ = 0: 0.08 points.
- **Turn 14, dice [x,6,6,6,6], reroll 2:** θ = 0 re-rolls the non-6 (chasing Yatzy). θ > 0.005 keeps all (taking Four of a Kind). The EV difference is negligible (0.008 pts), but the variance profiles differ.

These flip decisions form a natural diagnostic for estimating a player's risk preference. A player's choices across 15–20 such scenarios — selected by Fisher information to maximize discrimination power — can estimate their revealed θ within the empirically relevant range [−0.05, +0.15].


## 8. Board-State Frequency

Simulating 100K games under θ = 0 reveals an hourglass structure in the state space. Every game starts at one state (turn 1), fans out to a peak of 33,029 unique board states at turn 9, then collapses to 267 states by turn 15.

| Turn | Unique states | Top-10 concentration |
|---:|---:|---:|
| 1 | 1 | 100% |
| 5 | 6,275 | 1.9% |
| 9 | 33,029 | 1.9% |
| 12 | 11,614 | 10.0% |
| 15 | 267 | 82.1% |

Mid-game is maximally diffuse. Late-game concentrates sharply: at turn 15, the top 10 states account for 82% of games.

**Turn 1 fills opportunistically.** Two Pairs (10.5%) and Full House (10.0%) lead because they require specific patterns — when they hit, the solver takes them.

**Turn 15 reveals the dump hierarchy.** Four of a Kind (20.6%) and Large Straight (17.1%) are most often left for last. Both are high-variance categories with specific requirements. One Pair is almost never last (2.5%) — trivially easy to score.

**Bonus achievement.** 83.6% of games reach upper score 63 at turn 15. The optimal solver prioritizes the bonus aggressively.
