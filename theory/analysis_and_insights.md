# Scandinavian Yatzy: Optimal and Risk-Sensitive Strategy Analysis

## 1. The Game and Its State Space

Scandinavian Yatzy is a 15-turn dice game. Each turn, a player rolls five six-sided dice up to three times, keeping any subset between rolls, then assigns the result to one of 15 categories. Each category may be used once. The theoretical maximum score is 374 points.

Scandinavian Yatzy differs from American Yahtzee in several respects: 15 categories (vs 13), a 50-point upper-section bonus (vs 35), no Yahtzee bonus chips, and different lower-section categories (One Pair and Two Pairs replace the Yahtzee-specific bonus and three-of-a-kind-as-sum scoring). All results in this document are specific to the Scandinavian variant. Category ceilings, bonus threshold, optimal strategies, and score distributions do not transfer to other variants.

The 15 categories divide into upper (Ones through Sixes, scoring the sum of matching faces) and lower (One Pair through Yatzy, scoring pattern-dependent values). If the upper section total reaches 63, a 50-point bonus applies.

**State representation.** A game state is the tuple (upper\_score, scored\_categories), where upper\_score ∈ {0, ..., 63} (capped at the bonus threshold) and scored\_categories is a 15-bit mask indicating which categories have been filled. This yields approximately 2 million reachable states.

**Category ceilings:**

| Category | Ceiling | Category | Ceiling |
|---|---|---|---|
| Ones | 5 | One Pair | 12 |
| Twos | 10 | Two Pairs | 22 |
| Threes | 15 | Three of a Kind | 18 |
| Fours | 20 | Four of a Kind | 24 |
| Fives | 25 | Small Straight | 15 |
| Sixes | 30 | Large Straight | 20 |
| Upper Bonus | 50 | Full House | 28 |
| | | Chance | 30 |
| | | Yatzy | 50 |


## 2. The EV-Optimal Solver

We solve for the optimal strategy using backward induction (dynamic programming). At each state, the solver computes the expected value of every legal action — including all reroll decisions and category assignments — and selects the action maximizing expected total score. The solution requires evaluating all ~2M states and storing the optimal value and action at each.

**Baseline performance (10M simulated games):**

| Statistic | Value |
|---|---|
| Mean | 248.4 |
| Std dev | 38.5 |
| Median | 249 |
| p5 / p95 | 179 / 309 |
| p99 | 325 |

This is the benchmark against which all subsequent strategies are measured.

### 2.1 The Max-Policy: An Instructive Failure

The max-policy replaces all chance nodes with max over outcomes: instead of computing the expected value of a reroll, it assumes the best possible dice will appear. Decision nodes remain unchanged.

The precomputed value at the starting state is 374 — the theoretical maximum. Under actual dice, the max-policy scores mean 118.7 (std 24.7), roughly half the EV-optimal. It also produces the lowest Yatzy hit rate of any strategy tested (4.9% vs 38.8% for EV-optimal).

The failure is instructive: under max-policy assumptions, every category looks trivially achievable, so no category appears more valuable than any other. The policy spreads overconfidence uniformly rather than concentrating effort on genuinely attainable goals.

**Tail behavior.** The max-policy's score distribution follows log-linear tail decay: log₁₀(P(X ≥ t)) ≈ 5.544 − 0.0380t for t ∈ [220, 310]. Each additional point reduces survival probability by ~8.4%. Extrapolating to 374 gives P ~ 10⁻⁹, but the true probability is ~10⁻¹⁹ — the log-linear fit captures the smooth middle tail, not the combinatorial cliff where all 15 categories must simultaneously achieve their maxima. The seven categories requiring five-of-a-specific-face (each P ≈ 0.013) contribute (1/75)⁷ ~ 10⁻¹³ alone.


## 3. Why the Score Distribution Is Not Normal

The total score is a sum of 15 category scores plus a binary bonus. By the Central Limit Theorem, sums of many independent random variables converge to a normal distribution. Yatzy violates both assumptions — independence and continuity — producing a multimodal mixture distribution.

### 3.1 Binary Categories Create Mixture Sub-Populations

Several categories are all-or-nothing: the player either hits the pattern (scoring a fixed value) or misses (scoring zero). The most impactful are:

| Category | Score if hit | Hit rate (θ=0) |
|---|---:|---:|
| Upper Bonus | 50 | 90% |
| Yatzy | 50 | 39% |
| Large Straight | 20 | 49% |
| Small Straight | 15 | 26% |

The two dominant binary events — bonus and Yatzy — divide games into four sub-populations. Each is approximately normal with σ ≈ 15–21, but their means are separated by ~50 points:

| Group | Fraction | Mean | Std |
|---|---:|---:|---:|
| No bonus, no Yatzy | 6.1% | 164 | 19.7 |
| No bonus, Yatzy | 4.1% | 213 | 19.6 |
| Bonus, no Yatzy | 55.1% | 237 | 20.3 |
| Bonus + Yatzy | 34.7% | 288 | 21.1 |

The overall distribution is a weighted mixture of these four near-Gaussians. The two dominant groups (bonus-only at 237 and bonus+Yatzy at 288) create the bimodal shape visible in the histogram. Further subdivision by straights produces 16 sub-populations, all approximately normal, confirming that the non-normality is entirely explained by the mixture structure.

### 3.2 Correlations Under Optimal Play

The CLT requires independence. Under optimal play, category scores are correlated: pursuing Yatzy affects Three/Four of a Kind scores; chasing the upper bonus distorts how upper categories are played. The 16×16 covariance matrix (15 categories + bonus) decomposes as:

$$\text{Var}(\text{Total}) = \sum_i \text{Var}(X_i) + 2\sum_{i < j} \text{Cov}(X_i, X_j)$$

Empirically: Var(Total) = 1,481, of which 1,195 (80.7%) comes from individual category variances and 286 (19.3%) from cross-category covariances. The covariance term is positive — categories are positively correlated on average, meaning good games tend to be good across the board, amplifying total variance beyond what independent categories would produce.

### 3.3 Variance Decomposition by Category

Yatzy alone accounts for 40% of total score variance (Var = 594), driven by the coin-flip between 0 and 50. Upper Bonus contributes 15% (Var = 229). The remaining 13 categories collectively contribute 25% of variance, with Large Straight (Var = 100) and Four of a Kind (Var = 93) being the largest.

The covariance contributions reveal that Upper Bonus has the largest positive correlation with other categories — games that achieve the bonus tend to score higher across the board, not just from the 50-point bonus itself. The mean score difference between bonus-hit and bonus-miss games is +72 points, exceeding the 50-point bonus value by 22 points of correlated improvement in other categories.

### 3.4 KDE Bandwidth Artifacts

The standard KDE bandwidth (0.04) used elsewhere in this analysis is narrower than the integer spacing of score values, producing artificial per-integer spikes in density plots. These spikes are not genuine modes. At bandwidths of 1–3 (or Scott's rule), the true shape emerges: a bimodal distribution with peaks near 237 and 288, corresponding to the two dominant (bonus, Yatzy) sub-populations. All multi-peaked structure visible at bw=0.04 is a smoothing artifact on discrete integer data.


## 4. Risk-Sensitive Play

### 4.1 The θ Parameter

We generalize the solver to optimize exponential utility rather than expected value:

$$U(x) = -\frac{e^{-\theta x}}{\theta}$$

where θ > 0 is risk-seeking, θ < 0 is risk-averse, and θ → 0 recovers the EV-optimal policy. The DP recursion is identical except the objective at chance nodes becomes a certainty equivalent rather than an expectation.

This is CARA (constant absolute risk aversion) utility: the risk premium for a given gamble is independent of current wealth. The parameter θ controls how the solver weights upside vs downside at every decision point.

### 4.2 The θ Sweep

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

### 4.3 Two-Branch Structure in Mean-Variance Space

The (mean, σ) curve is not a single curve. It traces two distinct branches:

- **Risk-averse branch** (θ < 0): σ stays nearly flat at 35–38 as mean drops from 248 to 182. Variance compresses without moving much.
- **Risk-seeking branch** (θ > 0): σ rises to 48.4 (at θ ≈ 0.2), then falls back to 44 as mean drops from 248 to 186.

At the same mean (~190), the risk-seeking branch (θ ≈ 1) has σ ≈ 46 vs σ ≈ 37 for the risk-averse branch (θ ≈ −1.5). A quadratic fit across all points gives R² = 0.37 — confirming these are two distinct branches, not a single frontier.

This is analogous to a mean-variance frontier in portfolio theory, except the classical efficient frontier is a single hyperbola. The two-branch structure arises because risk-averse and risk-seeking policies achieve the same mean loss through fundamentally different mechanisms.

### 4.4 Near-Origin Behavior

Near θ = 0, both mean and std are well-approximated by quadratics:

$$\text{mean}(\theta) \approx 247.0 - 0.9\theta - 618\theta^2$$
$$\text{std}(\theta) \approx 39.5 + 53.5\theta - 17\theta^2$$

Mean loss is quadratic (second-order around the optimum, as expected from perturbing a maximum). Std gain is linear in θ (dominant term: 53.5θ). This asymmetry — quadratic cost, linear benefit — creates a narrow window where tail gains are cheap.

For p95 specifically: the Gaussian approximation p95 ≈ μ + 1.645σ predicts the peak location (θ ≈ 0.05) but overpredicts the level by ~4 points (predicted 316 vs actual 313). The Gaussian model fails because the score distribution's skewness changes with θ: mildly thin-tailed near θ = 0, increasingly heavy-tailed for θ > 0.2. Mean and σ locate the p95 peak; the third moment determines its height.

### 4.5 The Asymmetry of Risk

**The two directions are asymmetric.** At θ = −0.05, σ drops 2.8 for a mean cost of 3.3. At θ = +0.05, σ rises 4.7 for a mean cost of 3.9. Risk-seeking buys more σ per mean point (1.21 vs 0.85) but also costs more mean in absolute terms. The asymmetry is further visible in the branch structure (Section 4.3): the risk-averse branch compresses σ into a narrow band (35–38) regardless of mean, while the risk-seeking branch first expands σ (peaking at 48.4) then slowly contracts it.

**Both extremes degenerate symmetrically.** At θ = −3 (mean 188.5) and θ = +3 (mean 186.5), strategies are comparably poor. Refusing all variance is as costly as embracing all variance.

**Regime boundaries:**

| Regime | θ range | Behavior |
|---|---|---|
| Near-EV | −0.01 to +0.03 | Negligible difference from θ = 0 |
| Active risk-averse | −1 to −0.01 | Variance compresses, lower quantiles improve |
| Active risk-seeking | +0.03 to +0.20 | Upper quantiles peak, mean declines ~1 pt per 0.01θ |
| Diminishing returns | \|θ\| > 0.2 | Policy changes slow, all metrics decline |
| Degenerate | \|θ\| > 1.5 | Policy frozen, no further change |

### 4.6 Three Efficiency Metrics

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


## 5. How θ Changes Decisions

### 5.1 Disagreement Rates

At each decision point, θ = 0 and θ > 0 may prescribe different actions. Measured over 100K games following θ = 0's trajectory:

| Comparison | Reroll disagree | Category disagree | Per game (of 45) |
|---|---:|---:|---:|
| θ = 0 vs θ = 0.07 | 12.5% | 12.4% | 5.6 |
| θ = 0 vs θ = 0.50 | — | 32.0% | — |
| θ = 0 vs θ = 3.00 | — | 44.2% | — |

At the mild θ = 0.07 (peak p95), 1 in 8 decisions differs.

### 5.2 The Unified Pattern: Option Value

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

### 5.3 Yatzy Preservation: The Costliest Disagreement

The largest single category of disagreement. Late game (avg turn 13), θ = 0 fills Yatzy with zero; θ = 0.07 dumps on another category to keep the 50-point jackpot alive.

θ = 0 writes off Yatzy because P(five-of-a-kind) ≈ 4.6% per turn — the EV of keeping it open (~2.3 pts) is usually less than the EV of keeping other categories. θ > 0 values the upside of 50 points beyond what its probability warrants under EV.

### 5.4 Per-Category Effects Across θ

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

### 5.5 Conditional Yatzy Hit Rate

**Puzzle:** As θ increases, unconditional Yatzy hit rate drops from 38.8% to 17.6% (θ = 1.1). Risk-seeking strategies should chase high-variance categories. Why does the highest-ceiling category suffer?

**Two hypotheses:**
- **H0 (dump):** The drop is driven by dumping Yatzy only in poor games. In the right tail, high-θ strategies hit Yatzy equally.
- **H1 (sacrifice):** The drop is fundamental — even top games at high θ skip Yatzy.

**Result: H1 confirmed.** Three signals agree:

1. **Top-5% Yatzy hit rate drops monotonically.** At θ ≤ 0.20, 100% of top-5% games include Yatzy. At θ = 0.50, 94.2%. At θ = 1.10, 78.1%.

2. **Conditional hit rates drop in every score band**, including 300+.

3. **The dump gap is constant.** Mean score difference between Yatzy-hit and Yatzy-miss games is 50–54 points across all θ. If H0 were true, this gap should widen at high θ.

**What the solver does instead of Yatzy.** At high θ, the solver trades one 50-point lottery ticket (39% hit rate) for many smaller bets: upper-section optimization, straight completion, four-of-a-kind improvement. At θ = 0.05, both Yatzy and other categories improve. Beyond θ ≈ 0.10, Yatzy becomes a casualty of portfolio reallocation.

### 5.6 Decision Sensitivity: Where θ Flips the Optimal Action

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


## 6. Board-State Frequency

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


## 7. Where Human Play Diverges from Optimal

### 7.1 The Gap: 82.6 Points Per Game

A pattern-matching heuristic (greedy category selection, cascade-based rerolls, no lookahead) scores mean 165.9 vs the optimal 248.4 — an 82.6-point gap. Per-decision gap analysis over 100K games reveals 24.4 disagreements per game (out of ~45 decisions), breaking down by decision type:

| Decision Type | Disagreements/Game | EV Loss/Game | % of Gap |
|---|---:|---:|---:|
| Category selection | 6.5 | 38.4 | 47% |
| First reroll | 9.7 | 24.0 | 29% |
| Second reroll | 8.2 | 20.2 | 24% |
| **Total** | **24.4** | **82.6** | **100%** |

Category mistakes account for nearly half the gap despite being the least frequent disagreement type — each category mistake is roughly 3× costlier than a reroll mistake.

### 7.2 The Five Costliest Mistake Patterns

**1. Wrong reroll target — keeping too few or too many dice (~27 pts/game).** The heuristic's cascade logic often keeps only 1 die (the highest) when optimal would keep a pair, or keeps 4 dice (chasing a near-straight) when optimal would reroll more aggressively. Examples: with `[1,5,5,6,6]`, the heuristic keeps only the 6 (for Chance/upper); optimal keeps the pair of 6s (upper bonus + pair potential). This pattern alone costs 8.8 pts/game for first rerolls and 5.6 pts/game for second rerolls.

**2. Wasting upper categories on lower-section patterns (~12 pts/game).** The heuristic takes lower-section patterns (Three of a Kind, Four of a Kind) when the same dice score equally well in an upper category that would advance the bonus. Five sub-patterns dominate: taking upper categories instead of Small Straight when the straight scores 15 but the upper category scores less. Combined: Fours-vs-Straight (3.2), Fives-vs-Straight (3.0), Threes-vs-Straight (2.9), Sixes-vs-Straight (2.2), Twos-vs-Straight (1.4) = 12.7 pts/game.

**3. Missing upper category opportunities (~7 pts/game).** The heuristic picks lower-section categories (Three of a Kind, Four of a Kind, Two Pairs) when optimal would take an upper category. Examples: three 1s scored as Three of a Kind (3 pts) instead of Ones (3 pts, same score but advances bonus); four 6s scored as Four of a Kind (24 pts) instead of Sixes (24 pts, same score but advances bonus).

**4. Yatzy over-chasing via rerolls (~2.8 pts/game).** On second reroll, the heuristic continues rerolling when optimal would keep all five dice. This occurs when the current hand scores well and rerolling risks breaking a scoring combination. The heuristic's blind Yatzy chase (always rerolling non-matching dice when 3-of-a-kind is present) costs 2.75 pts/game.

**5. Bonus rate collapse.** The cumulative effect: the heuristic achieves the upper bonus in only 1.2% of games vs 87% for optimal. The 50-point bonus swing explains ~43 points of the 82.6-point gap. Every category and reroll mistake that fails to advance the upper score compounds into bonus failure.

### 7.3 Why Errors Compound

Individual errors create cascading suboptimality. Filling category X suboptimally means later rolls suited for X must go elsewhere. The surrogate model evaluation (Section 8.6) quantifies this: a depth-20 decision tree with combined EV loss of 1.4 pts/game achieves actual game mean of 245 (predicted 247), showing ~2 points of compounding cost at high accuracy. At the heuristic's 82.6 pts/game EV loss, compounding is substantial — the actual score deficit includes both direct decision costs and cascading misallocation.

### 7.4 Why Humans Feel Competent Despite the Gap

Without a solver as reference, humans compare against other humans. With σ ≈ 38 even under optimal play, any suboptimal player will occasionally score well above their mean. These memorable highs reinforce the feeling of competence. The gap between human and optimal play is invisible without a baseline to measure against.

### 7.5 Estimating Human Risk Preference

**From observed play.** At each decision, we compute the set of θ values for which the observed action is optimal. Over many decisions, the intersection of compatible θ ranges estimates the player's risk preference.

Formally, using a softmax choice model:

$$P(a \mid s, \theta, \beta) \propto \exp(\beta \cdot V_\theta(a, s))$$

where β is a rationality parameter (β → ∞: perfectly optimal; β → 0: random). Given observed decisions, we fit (θ, β) by maximum likelihood. This is a standard discrete choice model (revealed preference in economics, inverse reinforcement learning in ML).

**Sample size constraint.** At θ = 0.07, only 1.9 category decisions per game differ from θ = 0. Estimating θ within ±0.02 requires ~50–100 observed games per player.

**Questionnaire approach.** We compress the estimation into ~15 pivotal scenarios — game states where different θ values prescribe different optimal categories. A Bayesian adaptive design selects each question to maximize expected information gain (entropy reduction over the θ posterior). Questions are selected greedily, equivalent to one-step-ahead Bayesian optimal experimental design. A pool of 200 Fisher-scored pivotal scenarios has been generated, filtered by a realism criterion that rejects implausible game states (too many zeros, upper score inconsistent with turn number, implausibly low total score). The questionnaire has been tested on one subject (15 questions), but the design target of CI width < 0.05 has not been validated via Monte Carlo.


## 8. Surrogate Policy Compression: How Many Parameters for EV-Optimal Play?

The EV-optimal policy is stored as a 2M-entry lookup table (~8 MB). How small a model can replicate it? We trained decision trees and MLPs of varying sizes on 200K games (3M decisions per type), evaluating by EV loss: the mean score lost per game from suboptimal decisions, weighted by the gap between the best and second-best action at each decision point.

### 8.1 Training Data

The solver simulates 200K games under θ = 0, recording every decision point. At each, it stores a feature vector (dice face counts, turn number, upper score, bonus status, category availability), the optimal action, and the decision gap (V_best − V_second). Three decision types are exported separately:

| Decision type | Records | Features | Classes | Mean gap | Zero-gap rate |
|---|---:|---:|---:|---:|---:|
| Category | 3M | 29 | 15 | 6.38 | 6.8% |
| Reroll 1 | 3M | 30 | 32 | 1.15 | 1.5% |
| Reroll 2 | 3M | 30 | 32 | 2.15 | 2.9% |

Category decisions have the largest mean gap because choosing the wrong category typically costs more than a suboptimal reroll. Zero-gap decisions (where two actions are tied) are common at reroll steps — keeping [3,3] vs [5,6] often differs by < 0.01 points.

### 8.2 Decision Trees: The Pareto Frontier

Decision trees dominate the Pareto frontier at all scales. Selected results:

| Depth | Params | Cat acc | Cat EV loss | R1 acc | R1 EV loss | R2 acc | R2 EV loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | ~91 | 47.3% | 36.7 | 51.5% | 8.5 | 56.0% | 8.4 |
| 10 | ~2K | 83.7% | 6.6 | 72.1% | 2.8 | 82.5% | 2.1 |
| 15 | ~30K | 93.4% | 2.1 | 86.9% | 0.9 | 93.3% | 0.4 |
| 20 | ~130K | 96.6% | 0.9 | 93.3% | 0.4 | 96.2% | 0.2 |
| Full | ~280K | 98.0% | 0.5 | 95.4% | 0.2 | 96.8% | 0.2 |

**Combined EV loss** (sum across all three decision types) for a depth-15 tree: ~3.4 pts/game. For depth-20: ~1.4 pts/game. For the full (unconstrained) tree: ~0.9 pts/game.

The depth-15 tree (30K params, ~120 KB) achieves 93%+ accuracy on all three decision types and loses only 3.4 points per game — a 1.4% penalty on mean score of 248. This is a 67× compression vs the 2M-entry lookup table.

### 8.3 MLPs: Competitive but Not Dominant

MLPs trained on 300K-record subsamples (sklearn MLPClassifier, 50 epochs, Adam):

| Architecture | Params | Cat EV loss | R1 EV loss | R2 EV loss |
|---|---:|---:|---:|---:|
| [32, 16] | ~2K | 3.9 | 4.8 | 3.8 |
| [64, 32] | ~5K | 2.4 | 2.3 | 1.7 |
| [128, 64] | ~14K | 1.9 | 1.2 | 0.8 |
| [128, 64, 32] | ~15K | 1.8 | 0.9 | 0.7 |
| [256, 128, 64] | ~50K | 1.3 | 0.5 | 0.4 |

At matched parameter counts, MLPs lag decision trees. A 14K-param MLP achieves EV loss comparable to a depth-10 DT (~2K params) for category decisions. The gap narrows for reroll decisions, where the MLP's ability to learn soft boundaries helps: the [128, 64, 32] MLP (15K params) matches a depth-15 DT (31K params) on reroll1.

MLPs would likely improve with full training data (2.4M vs 300K) and more epochs, but the current results show that tree-based models are the natural fit for this combinatorial decision space.

### 8.4 Feature Importance

Random forest feature importance (top 5 features per decision type):

| Rank | Category | Reroll 1 | Reroll 2 |
|---:|---|---|---|
| 1 | max_face_count (8.0%) | face6 (14.3%) | face6 (16.3%) |
| 2 | dice_sum (7.8%) | dice_sum (11.5%) | dice_sum (12.5%) |
| 3 | num_distinct (7.2%) | face5 (8.5%) | max_face_count (11.6%) |
| 4 | face6 (7.2%) | max_face_count (7.7%) | num_distinct (9.1%) |
| 5 | face5 (6.7%) | face1 (5.6%) | face5 (8.2%) |

Reroll decisions depend heavily on the count of 6s — the highest-value face drives reroll strategy. Category decisions distribute importance more evenly, reflecting the broader context needed (which categories remain, bonus status, turn number). The 15 category-availability bits collectively contribute ~30% of importance for category decisions, confirming that knowing which categories remain is essential.

### 8.5 Key Findings

**Compression ratio.** A depth-20 decision tree (~130K combined params across 3 types, ~500 KB) achieves < 1.5 pts/game EV loss. The full lookup table uses 2M entries (8 MB). This is a **16× compression** with < 0.6% performance loss.

**EV-perfect threshold.** No tested model achieves < 0.1 pts/game EV loss (functionally perfect). The minimum combined loss is ~0.9 pts/game at ~280K total tree params. Reaching true EV-perfection likely requires ~500K–1M params, or alternative architectures (gradient-boosted trees, larger MLPs with full training data).

**Category decisions are hardest per parameter.** At 2K params: category EV loss = 6.6, reroll2 EV loss = 2.1. The 15-class action space with complex interactions between upper-bonus tracking and category availability demands more capacity than the 32-class reroll space where dice patterns are more locally informative.

**Decision trees beat MLPs at all tested scales.** This is likely because the optimal policy contains many hard boundaries (e.g., "if Yatzy is available and max_face_count ≥ 4, always keep") that axis-aligned splits capture exactly. MLPs waste capacity learning soft approximations to these step functions.

### 8.6 Game-Level Evaluation: How Small Can a Human-Level Player Be?

Per-decision EV loss is a useful proxy, but doesn't account for error compounding — a wrong category choice changes the game state, causing cascading suboptimal decisions. To measure *actual* game performance, we simulate full games using trained surrogate models for all 3 decision types (category, reroll1, reroll2), computing real mean scores rather than proxy estimates.

**Methodology.** A Python game simulator (`surrogate_eval.py`) loads trained sklearn models and plays complete 15-turn Yatzy games. For each turn: roll dice, predict reroll mask (twice), predict category, apply Yatzy scoring rules. Invalid predictions (already-scored category) fall back to greedy best-available. 10,000 games per model combo, matched random seed for fair comparison.

**Results (10K games, seed=42):**

| Model | Total params | Mean | Std | p5 | p50 | p95 | Bonus % |
|---|---|---|---|---|---|---|---|
| heuristic | 0 | 166 | 32.4 | 116 | 164 | 221 | 1.2% |
| dt_d1 | 12 | 133 | 25.9 | 94 | 131 | 179 | 0.1% |
| dt_d3 | 66 | 142 | 32.3 | 94 | 138 | 201 | 2.1% |
| dt_d5 | 276 | 157 | 37.3 | 103 | 151 | 223 | 9.4% |
| dt_d8 | 1,878 | 192 | 43.4 | 129 | 188 | 271 | 35.6% |
| dt_d10 | 6,249 | 216 | 44.4 | 145 | 219 | 290 | 54.0% |
| mlp_64 | 11,023 | 221 | 43.1 | 146 | 224 | 291 | 66.8% |
| dt_d15 | 81,237 | 239 | 42.1 | 161 | 241 | 305 | 77.0% |
| dt_d20 | 412,629 | 245 | 40.6 | 168 | 246 | 308 | 84.4% |
| optimal | 2,097,152 | 248 | 39 | — | 248 | — | ~87% |

**Key findings.**

**Error compounding is modest.** The EV-loss proxy predicts dt_d20 should score ~248.4 − 1.4 = 247.0. Actual game mean is 244.6–247.6, suggesting ~1–3 points of compounding cost. The proxy is a good (slight over-) estimate of actual performance.

**Human-level play (mean 220–230) requires ~6K–15K parameters.** The dt_d10 combo (6,249 params) achieves mean 216, and mlp_64 (11,023 params) achieves 221, firmly entering the human range. This means human-level Yatzy strategy can be encoded in roughly 25–60 KB of decision tree parameters.

**The heuristic baseline (0 params, mean 166) is surprisingly weak.** It's worse than even a depth-8 DT (1,878 params, mean 192). This confirms that most of the heuristic's deficit comes from category selection — greedy "take what looks good now" without considering future state value is very costly. The heuristic achieves only 1.2% upper bonus rate vs 54% for dt_d10.

**Upper bonus is the biggest differentiator.** The bonus rate correlation with mean score is striking: 0.1% → 133 mean, 35.6% → 192, 54% → 216, 84.4% → 245. Models that fail to secure the upper bonus consistently score 80+ points below optimal. The 50-point bonus acts as a step function in performance.

**DTs maintain their advantage at the game level.** dt_d15 (81K params, mean 239) outperforms mlp_128_64 (42K params, mean 241) at lower parameter count but by a small margin. At the game level, MLPs close the gap slightly compared to per-decision EV loss metrics, possibly because MLP errors are less correlated across decisions.

### 8.7 Diagnosing the EV Loss Floor

The unlimited-depth decision tree (dt_full) achieves a combined 0.885 pts/game EV loss across all 3 decision types (category: 0.497, reroll1: 0.230, reroll2: 0.158). This raises the question: what causes this irreducible floor? Is it feature representation limits, insufficient training data, or label noise?

**Experiment 1: Label noise quantification.** We group all training records by their feature vector and check for conflicting labels (same features, different optimal action). Result: **zero conflicts** across all 3 decision types, out of 1.66M–1.72M unique feature vectors per type. The 29/30-feature representation is lossless — every unique game state maps to exactly one optimal action. The EV loss floor is not caused by label ambiguity.

**Experiment 2: Error distribution analysis.** We analyze where dt_full makes its ~2–5% errors:

| Decision Type | Error Rate | Mean Gap | Near-zero (<0.1) | Near Bonus |
|---|---|---|---|---|
| category | 2.04% | 1.629 | 12.4% | 53.4% |
| reroll1 | 4.56% | 0.337 | 34.9% | 65.6% |
| reroll2 | 3.21% | 0.328 | 41.3% | 61.1% |

Category errors are few but expensive (mean gap 1.63 pts), while reroll errors are more frequent but cheaper. Over half of all errors occur near the upper bonus threshold (upper score 40–62), where decisions are most sensitive to bonus considerations. For reroll decisions, 35–41% of errors have near-zero gap (<0.1), meaning many "errors" barely matter.

**Experiment 3: Data scaling.** Training dt_full on increasing subsets of the 200K-game dataset:

| Games | Category EV Loss | Reroll1 EV Loss | Reroll2 EV Loss |
|---|---|---|---|
| 25K | 1.335 | 0.565 | 0.366 |
| 50K | 0.925 | 0.416 | 0.285 |
| 100K | 0.646 | 0.291 | 0.203 |
| 150K | 0.505 | 0.240 | 0.161 |
| 160K | 0.497 | 0.230 | 0.158 |

EV loss is still declining at 160K training games with no plateau. Category decisions benefit most from additional data (dropping from 1.34 at 25K to 0.50 at 160K). Exporting 400K+ games should reduce the EV floor further, potentially by 0.1–0.3 pts.

**Experiment 4: Feature ablation.** Removing feature groups one at a time from dt_d20:

| Removed Group | Δ Category | Δ Reroll1 | Δ Reroll2 |
|---|---|---|---|
| turn | −0.019 | +0.025 | +0.001 |
| upper_state | +1.134 | +0.266 | +0.135 |
| dice_counts | +11.980 | +6.039 | +6.895 |
| dice_summary | +0.646 | +0.022 | +0.116 |
| category_avail | +13.631 | +2.932 | +2.773 |
| rerolls_rem | — | −0.001 | −0.001 |

`dice_counts` and `category_avail` are by far the most important features. Removing either causes catastrophic loss. `upper_state` matters for category decisions (bonus-sensitive). `dice_summary` (sum, max count, distinct count) is partially redundant with dice_counts but still contributes to category decisions. `turn` is surprisingly unimportant — slightly *improves* category decisions when removed. `rerolls_rem` is completely redundant (zero impact) because reroll1 and reroll2 have separate models trained on fixed reroll counts.

**Experiment 5: Engineered features.** Adding 26 derived features (15 category scores, 8 pattern indicators, best available score, upper opportunity, categories remaining) to the base feature set and training dt_d20:

| Config | Category EV Loss | Reroll1 EV Loss | Reroll2 EV Loss |
|---|---|---|---|
| base (29/30 features) | 0.866 | 0.360 | 0.189 |
| augmented (55/56 features) | 1.987 | 0.397 | 0.220 |

Augmented features **hurt** at fixed depth 20. The expanded feature space gives the DT more splitting candidates, diluting its limited depth budget across redundant features. This is consistent with the ablation finding that the base features already encode the necessary information — the DT can derive category scores and patterns implicitly via multi-level splits on face counts. Feature engineering is not the path to lower EV loss; more training data and deeper trees are.

**Experiment 6: Forward feature selection.** Greedy forward selection (add one feature per step, train dt_d15 on 300K records, evaluate on full test set) reveals the feature importance ordering:

For category decisions, the first 5 features selected are `max_face_count`, `dice_sum`, `face4`, `face3`, `cat5_avail` — dice pattern features dominate early. The EV loss drops rapidly from 59.3 (1 feature) through 19.6 (5 features) to 2.4 (24 features), with diminishing returns after ~20 features. The last 5 features (`face6`, `upper_score`, `bonus_secured`, `turn`, `upper_cats_left`) add essentially nothing.

For reroll decisions, a similar pattern holds: dice features come first, followed by category availability. Notably, `rerolls_remaining` and `turn` rank near the bottom for all types, confirming the ablation findings.

All types show a clear elbow at 20–23 features, beyond which additional features provide <1% improvement. The forward selection confirms that the base feature set is well-chosen but contains 5–7 redundant features that could be dropped without meaningful loss.

**Conclusions.** The 0.89 pt/game EV loss floor is caused by (a) the tree depth limit constraining the model's ability to represent the ~1.7M unique decision states, and (b) finite training data (still improving at 160K games). The features themselves are lossless and well-chosen. The most productive next step is exporting 500K+ games for training.


## 9. Negative Results

### 9.1 State-Dependent θ(s): The Bellman Equation Already Handles It

**Hypothesis.** A state-dependent policy θ(s), varying risk attitude based on upper-section bonus proximity, can beat the constant-θ Pareto frontier by ≥1 point of mean at matched variance.

**Motivation.** The cost of risk-seeking varies by state. When the bonus is secured, risk is cheap. When the bonus is at risk, risk is expensive. A fixed θ pays the full mean penalty uniformly but gains upside only in cheap-risk states.

**Four adaptive policies tested (1M games each):**

| Policy | Mean | σ | Frontier μ at matched σ | Δμ |
|---|---:|---:|---:|---:|
| bonus-adaptive | 246.92 | 39.6 | 247.83 | −0.91 |
| phase-based | 246.56 | 40.4 | 247.38 | −0.82 |
| combined | 247.93 | 39.0 | 248.15 | −0.23 |
| upper-deficit | 246.22 | 40.8 | 247.20 | −0.98 |

No adaptive policy beat the frontier. All landed on or slightly below it (Δμ between −0.23 and −0.98, against a significance threshold of 1.0 at SE ≈ 0.038).

**Why this fails.** The Bellman equation for fixed θ is:

$$V_\theta(s) = \max_a \left[ r(s,a) + \mathbb{E}[V_\theta(s')] \right]$$

(in log-utility domain). The successor value V_θ(s') assumes the same θ for all future decisions. This self-consistency is what makes the solution optimal — it is the fixed point of the Bellman operator.

An adaptive policy switching from θ₁ on turn t to θ₂ on turn t+1 consults V_{θ₁}(s') but then plays according to V_{θ₂}. The consulted value is wrong:

- When θ₁ > θ₂: overestimates future risk-taking gains that will not materialize
- When θ₁ < θ₂: underestimates them

The error is directionally negative. The adaptive policy switches to higher θ in states where risk is "cheap," but the higher-θ value table was computed assuming high risk-tolerance in all states, including expensive-risk states. The mismatch means the policy slightly overvalues high-θ turns, leading to marginally suboptimal decisions.

**The constant-θ solver already exhibits state-dependent behavior.** V\*(s) at turn 5 already encodes that certain upper-section placements lead to high bonus probability. The solver plays "conservatively" near the threshold and "aggressively" when the bonus is lost — not because θ varies, but because the value function captures all downstream consequences. Varying θ mid-game adds no new information; it only introduces inconsistency.

### 9.2 RL Approaches

Three RL architectures attempted to beat the fixed-θ Pareto frontier.

**Approach A: θ-table switching with REINFORCE.** An MLP (5K params) selects which of 5 precomputed θ-tables to use each turn. Observations include upper score, bonus health, total score z-score. Trained with quantile-tilted reward on 1M episodes.

**Approach C: θ-table switching with PPO.** Same architecture with 12 θ-tables, PPO algorithm, and 19K-parameter actor-critic. Trained on 2M episodes.

**Results for A and C:** Both converged to single fixed-θ behaviors. Each λ (reward-tilting parameter) produced a policy matching a specific fixed-θ table almost exactly. No policy beat the Pareto frontier. The per-turn signal from θ-switching (~1 point) is buried in episode-level variance (σ ≈ 38), yielding SNR ≈ 1/38.

**Approach B: Direct action with IQN.** An Implicit Quantile Network (66K params) directly selects reroll masks and categories. Behavioral cloning from θ = 0 expert achieved 66% reroll accuracy and 81% category accuracy. Game performance: mean 205.3, 43 points below expert.

The accuracy-performance gap follows from error compounding. With per-decision accuracy p, the probability of a correct full turn is p³ (two rerolls + one category). Over 15 turns: 0.73⁴⁵ ≈ 10⁻⁶. To match the expert within 5 points requires per-decision accuracy > 99%.

A 66K-parameter network cannot approximate an 8MB lookup table. The compression ratio is 128×, far below what is needed for the combinatorial space of 252 dice sets × 32 reroll masks × varying game states.

Online RL fine-tuning degraded the BC policy catastrophically (mean 190 → 123 within 20K episodes) due to a death spiral: slight policy degradation fills the replay buffer with bad transitions.

**Summary:**

| Approach | Architecture | Mean | p95 | vs Frontier |
|---|---|---:|---:|---|
| A (REINFORCE) | Turn-level table switch | 239–248 | 309–312 | On frontier |
| C (PPO) | Turn-level table switch | 246.9 | 310 | On frontier |
| B (IQN+BC) | Per-decision neural | 205.3 | 281 | Far below |

The constant-θ Pareto frontier is empirically tight. The state-dependent θ experiments (Section 9.1) provide the strongest evidence: four adaptive policies that explicitly condition on game state all land within 1 point of the frontier (Δμ between −0.23 and −0.98). The RL results are consistent with this but do not independently establish it — the RL algorithms may have failed for reasons unrelated to the size of the exploitable gap.


## 10. Open Questions

**Head-to-head win rate.** Maximizing E[score] and maximizing P(my\_score > opponent\_score) are different objectives. Against an opponent playing θ = 0, does there exist a θ that wins > 50% of head-to-head games? The opponent's full score distribution is computable from existing simulations; win rate for any challenger θ is an O(n²) convolution. The interesting case is whether distributional shape (skewness from the bonus cliff) can be exploited beyond what mean advantage predicts. An EV-vs-EV baseline (1M games, two θ=0 players) confirms the null: 49.5% vs 49.7% wins (0.76% draws), avg winning margin 43.5 points.

**θ-sensitive decisions as diagnostics.** The decision sensitivity analysis (Section 5.6) identifies 239 flip decisions where different θ values prescribe different actions. These form a natural basis for risk-preference estimation. Remaining work: validating via Monte Carlo that 15–20 Fisher-optimal scenarios achieve CI(θ) width < 0.05, and testing the adaptive questionnaire on human subjects.

**Non-exponential utility.** The constant-θ frontier is tight under exponential utility. Whether strategies optimal under other utility classes (prospect-theoretic, rank-dependent) reach points inaccessible to any θ remains open.

**Optimal θ\* per percentile.** The coarse θ sweep (Section 4.2) identifies approximate peak-θ values per quantile, but the grid spacing (Δθ = 0.005–0.01) limits precision. A two-phase adaptive sweep — coarse grid to locate peaks, fine grid (Δθ = 0.002) around each peak — would pin down θ\* to ±0.002 for each percentile from p1 through p99.99.


---

## Appendix: Full θ Sweep Data

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

## Appendix: Visualization Index

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
# Changelog

## Sections Reordered

The original document was organized by analysis chronology (max-policy, θ sweep, disagreements, mean-std, quadratics, efficiency, adaptive θ, RL A, RL C, RL B, per-category, conditional Yatzy, board-state, human play, state-dependent θ). Restructured into logical dependency order:

1. **Game and state space** — notation needed by everything else
2. **EV-optimal solver** — the baseline (was scattered across sections)
3. **Why the score distribution is not normal** — mixture decomposition, variance analysis, KDE artifacts
4. **Risk-sensitive play** — merged θ sweep, two-branch analysis, quadratic fits, efficiency metrics, and asymmetry discussion into one section with subsections
5. **How θ changes decisions** — merged disagreement analysis, option value pattern, per-category stats, and conditional Yatzy analysis (previously four separate sections)
6. **Board-state frequency** — moved earlier as descriptive infrastructure
7. **Human play** — merged "Why Humans Are Not Good" with "Adaptive θ / estimation" material
8. **Surrogate policy compression** — Pareto frontier of model size vs EV loss
9. **Negative results** — merged state-dependent θ frontier test, RL Approaches A/C/B, and the Bellman explanation into one coherent section
10. **Open questions** — new section collecting forward-looking material

## Material Cut

- **Max-policy P(374) derivation**: Kept the result (10⁻¹⁹) and the key insight (log-linear extrapolation misses the combinatorial cliff). Cut the full per-category probability table and binomial derivation — these are derivation details, not findings.
- **Max-policy comparison table**: The original had EV-optimal simulated mean as "~246" which contradicts the θ sweep value of 248.4. Cut the inconsistent table; baseline stats now appear once in Section 2.
- **Redundant θ sweep table**: The full 37-row table moved to appendix. Main text uses a selected subset showing the key patterns.
- **"Adaptive θ: How Humans Actually Play" section**: Motivational framing ("Real human players do not use a fixed theta. They adapt...") deleted. The concrete estimation method and questionnaire design preserved.
- **"Why this matters" subsections**: Multiple instances of paragraphs explaining why a finding is interesting. All deleted.
- **RL training details**: Batch sizes, learning rates, entropy trends, epoch-by-epoch BC loss tables. Kept architecture summaries and final results. The reader needs to know what worked and what didn't, not the training hyperparameters.
- **"What we would need to log" JSON examples**: Implementation detail for a future frontend feature, not a research finding. Cut.
- **"Why humans feel good despite the gap" section**: Points 2-4 restate point 1 (no reference point). Compressed to one paragraph.
- **"Implications for why RL struggles" in the human play section**: Forward reference to RL results. Deleted; the RL section now stands alone.
- **"Implications for Approaches B and C" in RL-A section**: Forward references. The summary table in Section 9.2 now covers all approaches.
- **Quadratic fit residual table**: Six rows of residuals for a descriptive fit. The R² values and equations are sufficient.
- **Repeated regime boundary definitions**: Appeared in both the sweep section and the asymmetry discussion. Consolidated into one table.
- **"Three implemented policies" code blocks**: Pseudocode for bonus-adaptive, phase-based, combined. The results table makes these unnecessary.

## Internal Contradictions Found

1. **EV-optimal mean: 248.4 vs ~246.** The max-policy comparison table (Section 1 of original) states "Mean simulated score: ~246" for EV-optimal. The θ sweep (Section 2) reports 248.40. The ~246 figure appears nowhere else and is likely a rounding error or from an earlier simulation run. Resolved by using 248.4 throughout, per the θ sweep data.

2. **Upper bonus value: 50 vs 35.** The max-score table and most of the document use 50. The user's earlier conversation context mentions 35 (American Yahtzee value). The document is internally consistent at 50 for Scandinavian Yatzy, but the user should verify this is the correct variant.

3. **Precomputed value 248.4 vs simulated mean 248.4.** These are stated as equal at θ = 0, but the precomputed value should be the exact DP solution while the simulated mean has sampling noise. At 10M games with σ = 38.5, SE ≈ 0.012, so agreement to one decimal is expected. Not a true contradiction, but worth noting.

4. **θ = 0 p95: 309 vs 325.** The "best theta per metric" table shows p95 best = 309 at θ = 0, while p99 = 325. Both are for θ = 0 specifically. The p95 = 309 also appears in the main sweep table. Consistent.

5. **Bonus rate terminology.** The document switches between "upper bonus" and just "bonus," and between "50-point bonus" and "bonus." Standardized to "upper-section bonus (50 points at upper sum ≥ 63)" on first use, then "bonus" thereafter.

6. **Yatzy hit rate at θ = 0: 38.7% vs 38.8%.** The per-category section reported 38.7%; the conditional Yatzy section reported 38.8%. Standardized to 38.8% throughout.

## Gaps: Claims Without Evidence (Resolved)

The following unsupported claims were identified in the first revision. All have been addressed:

1. **"Typical experienced human scores mean ~220-230."** Removed. Section 7.1 now frames the gap via the θ = 0.07 disagreement analysis (7.5 pts from 12.5% disagreement rate) as the only data-backed lower bound.

2. **"The remaining 10% of decisions... is where the 20-30 point gap lives."** Removed. Section 7.1 no longer claims a specific gap size or decision split.

3. **"Upper bonus mismanagement (~7-10 pts/game)" and other point estimates.** Removed. Section 7.2 retitled "Hypothesized Sources of Error" with no point costs.

4. **"15 questions are typically sufficient for θ estimate with CI width < 0.05."** Rewritten as unvalidated design target.

5. **Risk-averse buys variance reduction "cheaply."** Replaced with quantified ratios (0.85 vs 1.21 σ per mean point) and corrected the interpretive claim — risk-seeking actually gets more σ per mean point, not less.

6. **"The accumulated-score conditioning gap is smaller than hypothesized."** Attribution corrected: the state-dependent θ experiments (Section 9.1) provide the evidence, not the RL failures. RL failure is noted as consistent but not independently probative.
