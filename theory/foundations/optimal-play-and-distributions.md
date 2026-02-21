# Scandinavian Yatzy: Optimal Play and Score Distributions

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
