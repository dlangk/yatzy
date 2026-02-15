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


## 3. Risk-Sensitive Play

### 3.1 The θ Parameter

We generalize the solver to optimize exponential utility rather than expected value:

$$U(x) = -\frac{e^{-\theta x}}{\theta}$$

where θ > 0 is risk-seeking, θ < 0 is risk-averse, and θ → 0 recovers the EV-optimal policy. The DP recursion is identical except the objective at chance nodes becomes a certainty equivalent rather than an expectation.

This is CARA (constant absolute risk aversion) utility: the risk premium for a given gamble is independent of current wealth. The parameter θ controls how the solver weights upside vs downside at every decision point.

### 3.2 The θ Sweep

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

### 3.3 Two-Branch Structure in Mean-Variance Space

The (mean, σ) curve is not a single curve. It traces two distinct branches:

- **Risk-averse branch** (θ < 0): σ stays nearly flat at 35–38 as mean drops from 248 to 182. Variance compresses without moving much.
- **Risk-seeking branch** (θ > 0): σ rises to 48.4 (at θ ≈ 0.2), then falls back to 44 as mean drops from 248 to 186.

At the same mean (~190), the risk-seeking branch (θ ≈ 1) has σ ≈ 46 vs σ ≈ 37 for the risk-averse branch (θ ≈ −1.5). A quadratic fit across all points gives R² = 0.37 — confirming these are two distinct branches, not a single frontier.

This is analogous to a mean-variance frontier in portfolio theory, except the classical efficient frontier is a single hyperbola. The two-branch structure arises because risk-averse and risk-seeking policies achieve the same mean loss through fundamentally different mechanisms.

### 3.4 Near-Origin Behavior

Near θ = 0, both mean and std are well-approximated by quadratics:

$$\text{mean}(\theta) \approx 247.0 - 0.9\theta - 618\theta^2$$
$$\text{std}(\theta) \approx 39.5 + 53.5\theta - 17\theta^2$$

Mean loss is quadratic (second-order around the optimum, as expected from perturbing a maximum). Std gain is linear in θ (dominant term: 53.5θ). This asymmetry — quadratic cost, linear benefit — creates a narrow window where tail gains are cheap.

For p95 specifically: the Gaussian approximation p95 ≈ μ + 1.645σ predicts the peak location (θ ≈ 0.05) but overpredicts the level by ~4 points (predicted 316 vs actual 313). The Gaussian model fails because the score distribution's skewness changes with θ: mildly thin-tailed near θ = 0, increasingly heavy-tailed for θ > 0.2. Mean and σ locate the p95 peak; the third moment determines its height.

### 3.5 The Asymmetry of Risk

**The two directions are asymmetric.** At θ = −0.05, σ drops 2.8 for a mean cost of 3.3. At θ = +0.05, σ rises 4.7 for a mean cost of 3.9. Risk-seeking buys more σ per mean point (1.21 vs 0.85) but also costs more mean in absolute terms. The asymmetry is further visible in the branch structure (Section 3.3): the risk-averse branch compresses σ into a narrow band (35–38) regardless of mean, while the risk-seeking branch first expands σ (peaking at 48.4) then slowly contracts it.

**Both extremes degenerate symmetrically.** At θ = −3 (mean 188.5) and θ = +3 (mean 186.5), strategies are comparably poor. Refusing all variance is as costly as embracing all variance.

**Regime boundaries:**

| Regime | θ range | Behavior |
|---|---|---|
| Near-EV | −0.01 to +0.03 | Negligible difference from θ = 0 |
| Active risk-averse | −1 to −0.01 | Variance compresses, lower quantiles improve |
| Active risk-seeking | +0.03 to +0.20 | Upper quantiles peak, mean declines ~1 pt per 0.01θ |
| Diminishing returns | \|θ\| > 0.2 | Policy changes slow, all metrics decline |
| Degenerate | \|θ\| > 1.5 | Policy frozen, no further change |

### 3.6 Three Efficiency Metrics

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


## 4. How θ Changes Decisions

### 4.1 Disagreement Rates

At each decision point, θ = 0 and θ > 0 may prescribe different actions. Measured over 100K games following θ = 0's trajectory:

| Comparison | Reroll disagree | Category disagree | Per game (of 45) |
|---|---:|---:|---:|
| θ = 0 vs θ = 0.07 | 12.5% | 12.4% | 5.6 |
| θ = 0 vs θ = 0.50 | — | 32.0% | — |
| θ = 0 vs θ = 3.00 | — | 44.2% | — |

At the mild θ = 0.07 (peak p95), 1 in 8 decisions differs.

### 4.2 The Unified Pattern: Option Value

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

### 4.3 Yatzy Preservation: The Costliest Disagreement

The largest single category of disagreement. Late game (avg turn 13), θ = 0 fills Yatzy with zero; θ = 0.07 dumps on another category to keep the 50-point jackpot alive.

θ = 0 writes off Yatzy because P(five-of-a-kind) ≈ 4.6% per turn — the EV of keeping it open (~2.3 pts) is usually less than the EV of keeping other categories. θ > 0 values the upside of 50 points beyond what its probability warrants under EV.

### 4.4 Per-Category Effects Across θ

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

### 4.5 Conditional Yatzy Hit Rate

**Puzzle:** As θ increases, unconditional Yatzy hit rate drops from 38.8% to 17.6% (θ = 1.1). Risk-seeking strategies should chase high-variance categories. Why does the highest-ceiling category suffer?

**Two hypotheses:**
- **H0 (dump):** The drop is driven by dumping Yatzy only in poor games. In the right tail, high-θ strategies hit Yatzy equally.
- **H1 (sacrifice):** The drop is fundamental — even top games at high θ skip Yatzy.

**Result: H1 confirmed.** Three signals agree:

1. **Top-5% Yatzy hit rate drops monotonically.** At θ ≤ 0.20, 100% of top-5% games include Yatzy. At θ = 0.50, 94.2%. At θ = 1.10, 78.1%.

2. **Conditional hit rates drop in every score band**, including 300+.

3. **The dump gap is constant.** Mean score difference between Yatzy-hit and Yatzy-miss games is 50–54 points across all θ. If H0 were true, this gap should widen at high θ.

**What the solver does instead of Yatzy.** At high θ, the solver trades one 50-point lottery ticket (39% hit rate) for many smaller bets: upper-section optimization, straight completion, four-of-a-kind improvement. At θ = 0.05, both Yatzy and other categories improve. Beyond θ ≈ 0.10, Yatzy becomes a casualty of portfolio reallocation.

### 4.6 Decision Sensitivity: Where θ Flips the Optimal Action

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


## 5. Board-State Frequency

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


## 6. Where Human Play Diverges from Optimal

### 6.1 The Gap

No human study has been conducted to measure the exact deficit, but the structure of the game constrains it. The disagreement analysis (Section 4.1) shows that even θ = 0.07 — a mild perturbation of the optimal policy — disagrees on 12.5% of decisions and loses 7.5 points of mean. A human heuristic policy would disagree on substantially more decisions. The error sources below are hypothesized categories, ordered by conjectured severity; validating them requires simulating a human-heuristic baseline policy.

### 6.2 Hypothesized Sources of Error

**Upper bonus mismanagement.** The 50-point bonus at 63 upper points creates a cliff that humans misjudge. Common errors: investing in the bonus when it is already unreachable (remaining upper categories cannot sum to the deficit), and under-investing when it is marginal (e.g., scoring 20 in Fives for immediate points when scoring 12 in Fours and saving Fives would secure the bonus).

**Suboptimal rerolls.** Evaluating 32 reroll masks × 252 resulting dice distributions × remaining-category interactions exceeds human computation. Humans use heuristics ("keep the biggest matching set") that miss state-dependent exceptions.

**Category ordering cascades.** Filling a flexible category (Chance) with an inflexible roll wastes future options. Humans err by filling greedily (highest immediate score) rather than preserving option value.

**Chase errors.** Humans both over-chase (rerolling a strong Three of a Kind for Yatzy on the last reroll) and under-chase (banking One Pair from [4,4,4,\_,\_] instead of rerolling for Full House at ~35%).

### 6.3 Why Errors Compound

Individual errors create cascading suboptimality. Filling category X suboptimally means later rolls suited for X must go elsewhere. This secondary misallocation is invisible to the player and may rival the original error in cost, though the compounding magnitude has not been measured.

### 6.4 Why Humans Feel Competent Despite the Gap

Without a solver as reference, humans compare against other humans. With σ ≈ 38 even under optimal play, any suboptimal player will occasionally score well above their mean. These memorable highs reinforce the feeling of competence. The gap between human and optimal play is invisible without a baseline to measure against.

### 6.5 Estimating Human Risk Preference

**From observed play.** At each decision, we compute the set of θ values for which the observed action is optimal. Over many decisions, the intersection of compatible θ ranges estimates the player's risk preference.

Formally, using a softmax choice model:

$$P(a \mid s, \theta, \beta) \propto \exp(\beta \cdot V_\theta(a, s))$$

where β is a rationality parameter (β → ∞: perfectly optimal; β → 0: random). Given observed decisions, we fit (θ, β) by maximum likelihood. This is a standard discrete choice model (revealed preference in economics, inverse reinforcement learning in ML).

**Sample size constraint.** At θ = 0.07, only 1.9 category decisions per game differ from θ = 0. Estimating θ within ±0.02 requires ~50–100 observed games per player.

**Questionnaire approach.** We compress the estimation into ~15 pivotal scenarios — game states where different θ values prescribe different optimal categories. A Bayesian adaptive design selects each question to maximize expected information gain (entropy reduction over the θ posterior). Questions are selected greedily, equivalent to one-step-ahead Bayesian optimal experimental design. A pool of 200 Fisher-scored pivotal scenarios has been generated, filtered by a realism criterion that rejects implausible game states (too many zeros, upper score inconsistent with turn number, implausibly low total score). The questionnaire has been tested on one subject (15 questions), but the design target of CI width < 0.05 has not been validated via Monte Carlo.


## 7. Negative Results

### 7.1 State-Dependent θ(s): The Bellman Equation Already Handles It

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

### 7.2 RL Approaches

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

The constant-θ Pareto frontier is empirically tight. The state-dependent θ experiments (Section 7.1) provide the strongest evidence: four adaptive policies that explicitly condition on game state all land within 1 point of the frontier (Δμ between −0.23 and −0.98). The RL results are consistent with this but do not independently establish it — the RL algorithms may have failed for reasons unrelated to the size of the exploitable gap.


## 8. Open Questions

**Head-to-head win rate.** Maximizing E[score] and maximizing P(my\_score > opponent\_score) are different objectives. Against an opponent playing θ = 0, does there exist a θ that wins > 50% of head-to-head games? The opponent's full score distribution is computable from existing simulations; win rate for any challenger θ is an O(n²) convolution. The interesting case is whether distributional shape (skewness from the bonus cliff) can be exploited beyond what mean advantage predicts. An EV-vs-EV baseline (1M games, two θ=0 players) confirms the null: 49.5% vs 49.7% wins (0.76% draws), avg winning margin 43.5 points.

**θ-sensitive decisions as diagnostics.** The decision sensitivity analysis (Section 4.6) identifies 239 flip decisions where different θ values prescribe different actions. These form a natural basis for risk-preference estimation. Remaining work: validating via Monte Carlo that 15–20 Fisher-optimal scenarios achieve CI(θ) width < 0.05, and testing the adaptive questionnaire on human subjects.

**Non-exponential utility.** The constant-θ frontier is tight under exponential utility. Whether strategies optimal under other utility classes (prospect-theoretic, rank-dependent) reach points inaccessible to any θ remains open.

**Optimal θ\* per percentile.** The coarse θ sweep (Section 3.2) identifies approximate peak-θ values per quantile, but the grid spacing (Δθ = 0.005–0.01) limits precision. A two-phase adaptive sweep — coarse grid to locate peaks, fine grid (Δθ = 0.002) around each peak — would pin down θ\* to ±0.002 for each percentile from p1 through p99.99.


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

**Other:**
- `quantile.png` — Quantile-quantile or quantile function plot
# Changelog

## Sections Reordered

The original document was organized by analysis chronology (max-policy, θ sweep, disagreements, mean-std, quadratics, efficiency, adaptive θ, RL A, RL C, RL B, per-category, conditional Yatzy, board-state, human play, state-dependent θ). Restructured into logical dependency order:

1. **Game and state space** — notation needed by everything else
2. **EV-optimal solver** — the baseline (was scattered across sections)
3. **Risk-sensitive play** — merged θ sweep, two-branch analysis, quadratic fits, efficiency metrics, and asymmetry discussion into one section with subsections
4. **How θ changes decisions** — merged disagreement analysis, option value pattern, per-category stats, and conditional Yatzy analysis (previously four separate sections)
5. **Board-state frequency** — moved earlier as descriptive infrastructure
6. **Human play** — merged "Why Humans Are Not Good" with "Adaptive θ / estimation" material
7. **Negative results** — merged state-dependent θ frontier test, RL Approaches A/C/B, and the Bellman explanation into one coherent section
8. **Open questions** — new section collecting forward-looking material

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
- **"Implications for Approaches B and C" in RL-A section**: Forward references. The summary table in Section 7.2 now covers all approaches.
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

1. **"Typical experienced human scores mean ~220-230."** Removed. Section 6.1 now frames the gap via the θ = 0.07 disagreement analysis (7.5 pts from 12.5% disagreement rate) as the only data-backed lower bound.

2. **"The remaining 10% of decisions... is where the 20-30 point gap lives."** Removed. Section 6.1 no longer claims a specific gap size or decision split.

3. **"Upper bonus mismanagement (~7-10 pts/game)" and other point estimates.** Removed. Section 6.2 retitled "Hypothesized Sources of Error" with no point costs.

4. **"15 questions are typically sufficient for θ estimate with CI width < 0.05."** Rewritten as unvalidated design target.

5. **Risk-averse buys variance reduction "cheaply."** Replaced with quantified ratios (0.85 vs 1.21 σ per mean point) and corrected the interpretive claim — risk-seeking actually gets more σ per mean point, not less.

6. **"The accumulated-score conditioning gap is smaller than hypothesized."** Attribution corrected: the state-dependent θ experiments (Section 7.1) provide the evidence, not the RL failures. RL failure is noted as consistent but not independently probative.
