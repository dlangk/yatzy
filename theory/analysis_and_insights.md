# Analysis and Insights

This is a living document that accumulates empirical findings, simulation results, and statistical insights about Scandinavian Yatzy strategy. When a conversation produces new statistical or theoretical insights, append them here. Check existing content before adding to avoid duplication.

---

## Max-Policy (Maximax) Analysis

### What is the max-policy?

The max-policy replaces all **chance nodes** (dice outcome expectations) with **max** over outcomes:

- Standard solver: `E(keep) = Sum P(keep -> r'') * V(r'')` (expected value)
- Max-policy: `E(keep) = max_{r''} V(r'')` (best possible outcome)

Decision nodes (category choice, keep selection) are unchanged -- already argmax.

This creates an "optimistic" policy that always picks the action assuming the best possible dice roll will happen.

### Precomputed state value at (0, 0)

The max-policy starting state value is **374.0** -- the theoretical maximum score in Scandinavian Yatzy. This makes sense: if every chance node delivers the best outcome, you can max every category.

### Maximum score breakdown (374 points)

| Category        | Dice needed   | Score |
|-----------------|---------------|-------|
| Ones            | [1,1,1,1,1]   |     5 |
| Twos            | [2,2,2,2,2]   |    10 |
| Threes          | [3,3,3,3,3]   |    15 |
| Fours           | [4,4,4,4,4]   |    20 |
| Fives           | [5,5,5,5,5]   |    25 |
| Sixes           | [6,6,6,6,6]   |    30 |
| **Upper bonus** |               |  **50** |
| One pair        | [_,_,_,6,6]   |    12 |
| Two pairs       | [_,5,5,6,6]   |    22 |
| Three of a kind | [_,_,6,6,6]   |    18 |
| Four of a kind  | [_,6,6,6,6]   |    24 |
| Small straight  | [1,2,3,4,5]   |    15 |
| Large straight  | [2,3,4,5,6]   |    20 |
| Full house      | [5,5,6,6,6]   |    28 |
| Chance          | [6,6,6,6,6]   |    30 |
| Yatzy           | [n,n,n,n,n]   |    50 |
| **Total**       |               | **374** |

### Simulation results (10M games)

The max-policy makes decisions assuming the best dice outcome, but actual dice are random.

| Statistic | Value |
|-----------|-------|
| Games     | 10,000,000 |
| Mean      | 118.74 |
| Std dev   | 24.7 |
| Min       | 35 |
| Max       | 313 |
| Median    | 117 |

The mean is ~119 -- about half the EV-optimal policy's ~246. The max-policy makes terrible real-world decisions because it always assumes the best dice outcome will happen. For example, it might reroll a decent hand chasing five-of-a-kind, since under the max assumption that always succeeds.

### Tail distribution (empirical)

| Threshold | Count >= | Fraction | 1 in |
|-----------|----------|----------|------|
| 200 | 41,651 | 4.17e-3 | 240 |
| 220 | 12,305 | 1.23e-3 | 813 |
| 240 | 2,859 | 2.86e-4 | 3,497 |
| 260 | 492 | 4.92e-5 | 20,325 |
| 280 | 72 | 7.20e-6 | 138,889 |
| 300 | 5 | 5.00e-7 | 2,000,000 |
| 310 | 2 | 2.00e-7 | 5,000,000 |
| 320+ | 0 | -- | -- |

### Log-linear tail fit

Fitting `log10(P(X >= t)) = a + b*t` to the range 220-310:

```
log10(P(X >= t)) = 5.544 - 0.0380 * t
```

Each additional point reduces the survival probability by a factor of ~0.916 (i.e., ~8.4% less likely per point).

Extrapolating this to 374 gives P ~ 10^-8.7 ~ 2e-9, or ~500M games. **However**, this extrapolation is misleading -- it captures the smooth middle tail but not the combinatorial cliff at the extreme.

### Analytical estimate: P(score = 374)

To score exactly 374, every category must yield its maximum. The per-category probabilities (with 3 rolls, optimal keeping) are:

| Category | Requirement | P(max) | 1 in |
|----------|-------------|--------|------|
| Ones     | Five 1s | 0.01327 | 75 |
| Twos     | Five 2s | 0.01327 | 75 |
| Threes   | Five 3s | 0.01327 | 75 |
| Fours    | Five 4s | 0.01327 | 75 |
| Fives    | Five 5s | 0.01327 | 75 |
| Sixes    | Five 6s | 0.01327 | 75 |
| One pair | >= 2 sixes | 0.699 | 1.4 |
| Two pairs | >= 2 fives + 2 sixes | 0.275 | 3.6 |
| Three of a kind | >= 3 sixes | 0.355 | 2.8 |
| Four of a kind | >= 4 sixes | 0.104 | 9.6 |
| Small straight | [1,2,3,4,5] | 0.197 | 5.1 |
| Large straight | [2,3,4,5,6] | 0.197 | 5.1 |
| Full house | [5,5,6,6,6] | 0.062 | 16 |
| Chance | Five 6s | 0.01327 | 75 |
| Yatzy | Any five-of-a-kind | 0.080 | 13 |

**Key formula: P(five of a specific face)**

With 3 rolls of 5 dice, keeping matching dice:

```
P = (91/216)^5 ~ 0.01327 ~ 1 in 75
```

Derivation: After roll 1 with k matches, need (5-k) more in 2 rolls. The probability of getting n specific faces in 2 rolls of n dice (keeping matches between rolls) is `(11/36)^n`. By the binomial theorem:

```
P = Sum_{k=0}^{5} C(5,k) (1/6)^k (5/6)^{5-k} (11/36)^{5-k}
  = (1/6 + 5*11/(6*36))^5
  = (91/216)^5
```

**Combined probability (assuming independence):**

```
P(374) ~ (0.01327)^7 * 0.699 * 0.275 * 0.355 * 0.104 * 0.197 * 0.197 * 0.062 * 0.080
       ~ 10^{-19}
```

**~10^19 games needed** (10 quintillion). At 50,000 games/sec: ~6.5 million years.

### Why the log-linear extrapolation underestimates

The log-linear fit (P ~ 10^-9 at 374) captures scores 220-310, where high scores come from maxing *some* categories while getting decent scores in others. But 374 requires *every single category* maxed -- a qualitatively different event that sits on a combinatorial cliff far below the smooth tail extrapolation.

The 7 categories requiring five-of-a-specific-face (each 1-in-75) are the dominant bottleneck, contributing (1/75)^7 ~ 10^-13 alone.

### Comparison with EV-optimal policy

| Metric | EV-optimal | Max-policy |
|--------|-----------|------------|
| Precomputed value at (0,0) | 248.4 | 374.0 |
| Mean simulated score | ~246 | ~119 |
| Std dev | ~38 | ~25 |
| Max (10M games) | ~374* | 313 |

*The EV-optimal policy has a much better chance of high scores because it makes realistic decisions. The max-policy actively hurts performance by chasing unrealistic best-case outcomes.

---

## Risk-Sensitive theta Sweep Results

### Coarse sweep (1M games each, 21 theta values)

| theta | Mean | Std | p5 | p95 (actual) | p95 (Gaussian) | Delta |
|------:|------:|-----:|----:|---:|---:|------:|
| -20.000 | 181.9 | 36.0 | 134 | 247 | 241.1 | +5.9 |
| -10.000 | 182.7 | 36.2 | 134 | 249 | 242.2 | +6.8 |
| -5.000 | 186.6 | 37.1 | 137 | 255 | 247.7 | +7.3 |
| -1.000 | 198.2 | 37.9 | 148 | 269 | 260.5 | +8.5 |
| -0.100 | 239.2 | 35.9 | 172 | 300 | 298.2 | +1.8 |
| -0.050 | 245.1 | 35.7 | 179 | 305 | 303.9 | +1.1 |
| -0.020 | 247.9 | 37.2 | 183 | 308 | 309.1 | -1.1 |
| -0.010 | 248.3 | 37.7 | 182 | 308 | 310.4 | -2.4 |
| **0.000** | **248.4** | **38.5** | **179** | **309** | **311.7** | **-2.7** |
| +0.010 | 248.2 | 39.4 | 175 | 310 | 313.0 | -3.0 |
| +0.020 | 247.7 | 40.3 | 172 | 311 | 314.1 | -3.1 |
| +0.030 | 246.9 | 41.3 | 169 | 311 | 314.9 | -3.9 |
| +0.040 | 245.7 | 42.3 | 166 | 312 | 315.3 | -3.3 |
| +0.050 | 244.5 | 43.2 | 164 | 312 | 315.6 | -3.6 |
| +0.100 | 235.9 | 46.9 | 155 | 312 | 313.1 | -1.1 |
| +0.200 | 223.8 | 48.4 | 148 | 308 | 303.3 | +4.7 |
| +0.500 | 204.7 | 47.7 | 135 | 287 | 283.1 | +3.9 |
| +1.000 | 194.5 | 46.3 | 128 | 276 | 270.6 | +5.4 |
| +5.000 | 186.5 | 44.1 | 124 | 266 | 259.1 | +6.9 |
| +10.000 | 185.8 | 43.9 | 124 | 265 | 258.1 | +6.9 |
| +20.000 | 185.7 | 43.9 | 124 | 265 | 257.9 | +7.1 |

### Dense sweep (1M games each, seed=42)

Grid: dense (delta_theta = 0.01) in the interesting range 0-0.20, sparser (delta_theta = 0.1-0.2) in the saturated range 0.3-3.0.

| theta | Mean | StdDev | Min | p5 | p25 | p50 | p75 | p95 | Max | Bonus% |
|------:|------:|-------:|----:|---:|----:|----:|----:|----:|----:|-------:|
| 0.000 | 248.40 | 38.5 | 97 | 179 | 225 | 249 | 276 | 309 | 352 | 89.8% |
| 0.010 | 248.21 | 39.4 | 90 | 175 | 225 | 249 | 277 | 310 | 353 | 88.4% |
| 0.020 | 247.73 | 40.3 | 90 | 172 | 224 | 248 | 277 | 311 | 359 | 86.7% |
| 0.030 | 246.92 | 41.3 | 88 | 169 | 223 | 247 | 277 | 311 | 359 | 84.7% |
| 0.040 | 245.71 | 42.3 | 88 | 166 | 221 | 245 | 278 | 312 | 359 | 82.5% |
| 0.050 | 244.53 | 43.2 | 79 | 164 | 219 | 244 | 277 | 312 | 359 | 79.9% |
| 0.060 | 242.94 | 44.2 | 79 | 162 | 216 | 243 | 276 | 312 | 362 | 77.0% |
| 0.070 | 240.86 | 45.3 | 84 | 159 | 213 | 241 | 274 | 313 | 362 | 72.6% |
| 0.080 | 239.12 | 46.0 | 78 | 158 | 210 | 239 | 273 | 313 | 362 | 69.4% |
| 0.090 | 237.47 | 46.5 | 82 | 156 | 207 | 238 | 270 | 313 | 362 | 66.3% |
| 0.100 | 235.92 | 46.9 | 79 | 155 | 204 | 237 | 268 | 312 | 362 | 63.6% |
| 0.110 | 234.40 | 47.2 | 81 | 154 | 202 | 236 | 266 | 312 | 362 | 61.7% |
| 0.120 | 233.05 | 47.4 | 81 | 154 | 199 | 235 | 265 | 312 | 362 | 59.6% |
| 0.130 | 231.69 | 47.6 | 74 | 153 | 196 | 234 | 263 | 311 | 362 | 57.7% |
| 0.140 | 230.43 | 47.8 | 74 | 152 | 194 | 232 | 262 | 311 | 362 | 55.9% |
| 0.150 | 229.23 | 47.9 | 74 | 152 | 192 | 231 | 261 | 311 | 362 | 54.4% |
| 0.160 | 228.02 | 48.1 | 74 | 151 | 191 | 230 | 260 | 310 | 362 | 53.0% |
| 0.170 | 226.82 | 48.2 | 78 | 150 | 189 | 229 | 260 | 310 | 358 | 51.8% |
| 0.180 | 225.74 | 48.3 | 75 | 149 | 188 | 228 | 259 | 309 | 358 | 50.6% |
| 0.190 | 224.75 | 48.3 | 78 | 149 | 186 | 227 | 258 | 308 | 358 | 49.6% |
| 0.200 | 223.76 | 48.4 | 67 | 148 | 185 | 225 | 257 | 308 | 358 | 48.5% |
| 0.300 | 215.74 | 48.6 | 76 | 142 | 177 | 215 | 251 | 301 | 358 | 44.0% |
| 0.400 | 209.82 | 48.3 | 67 | 138 | 171 | 207 | 245 | 293 | 358 | 40.7% |
| 0.500 | 204.73 | 47.6 | 60 | 135 | 168 | 200 | 240 | 287 | 361 | 38.0% |
| 0.700 | 199.29 | 47.1 | 51 | 131 | 163 | 194 | 234 | 281 | 360 | 34.7% |
| 0.900 | 195.29 | 46.3 | 57 | 128 | 160 | 189 | 230 | 276 | 359 | 33.2% |
| 1.100 | 193.73 | 46.1 | 51 | 127 | 158 | 187 | 228 | 275 | 359 | 32.2% |
| 1.300 | 190.56 | 45.8 | 51 | 125 | 156 | 184 | 223 | 272 | 359 | 30.0% |
| 1.500 | 189.22 | 45.5 | 55 | 125 | 155 | 182 | 221 | 270 | 359 | 28.8% |
| 1.700 | 187.28 | 45.2 | 57 | 123 | 153 | 180 | 219 | 268 | 359 | 27.3% |
| 1.900 | 186.17 | 44.8 | 48 | 123 | 152 | 179 | 217 | 267 | 359 | 27.1% |
| 2.000 | 186.04 | 44.8 | 48 | 123 | 152 | 179 | 217 | 266 | 359 | 27.1% |
| 3.000 | 186.49 | 44.2 | 50 | 124 | 154 | 179 | 216 | 266 | 359 | 25.7% |

---

## Key Observations from the theta Sweep

### Where does p95 peak?

**p95 peaks at 313** across theta = 0.07-0.09 (|theta| * sigma ~ 0.7-0.9). This is +4 points over the EV-optimal baseline (p95 = 309). The gain comes at a meaningful cost to the mean: theta = 0.08 has mean 239.1 vs baseline 248.4 (~9 points lower).

With the coarser grid, p95 = 312 was observed across theta = 0.04-0.10. The dense grid refines this to a peak of 313 at theta = 0.07-0.09.

### Can the p95 peak be predicted from mean and std alone?

**Directionally yes, quantitatively no.**

The Gaussian approximation `p95 ~ mean + 1.645 * sigma` predicts the p95 peak location reasonably well -- the Gaussian p95 also peaks around theta ~ +0.05 (at 315.6). But there are important failures:

**1. The delta flips sign with theta.** Near theta = 0, the actual p95 is *lower* than the Gaussian prediction (delta = -2.7), meaning the right tail is thinner than Normal. At extreme theta (both positive and negative), the actual p95 *exceeds* the Gaussian prediction (delta = +5 to +7), meaning the tail becomes heavier. The distribution shape changes with theta.

**2. The Gaussian overpredicts the peak p95 by ~4 points.** The Gaussian model says the best achievable p95 is ~316; the actual peak is 313. The thin-tail effect near the optimum eats into the predicted gain.

**3. The actual p95 plateau is flatter than predicted.** The Gaussian curve has a sharp peak at theta ~ 0.05; the actual p95 holds at 312-313 across theta = 0.04-0.10 because the growing heavy-tail effect at higher theta partially compensates for the declining Gaussian prediction.

### Why the distribution shape changes with theta

As theta increases (risk-seeking), the policy increasingly "gambles" -- chasing high-variance plays that pay off rarely but spectacularly. This doesn't just shift mean down and sigma up; it changes the **skewness**:

- At theta = 0: the distribution is mildly thin-tailed on the right (delta negative)
- At theta > 0.2: the distribution becomes heavy-tailed on the right (delta positive)
- The crossover happens around theta ~ 0.1-0.2

Mean and sigma get you in the right ballpark for *where* the p95 peaks, but you need the third moment (skewness) or the full distribution to predict the actual level. The two-moment Gaussian model systematically mischaracterizes the tail behavior because it cannot capture how risk-sensitive policies reshape the score distribution's asymmetry.

### Mean loss is monotonic and concave

Mean drops from 248.4 (theta = 0) to 186.0 (theta = 2.0), a total loss of 62.4 points. The loss is steepest in [0, 0.10]: ~1.25 pts per 0.01 theta. By theta = 0.20, mean loss rate halves to ~1.0 pt per 0.01 theta.

### StdDev rises rapidly then plateaus

StdDev rises from 38.5 to 46.9 in the first 0.10 of theta (+8.4 pts, +22%), then barely increases beyond that (peak 48.6 at theta = 0.3).

### p75 peaks at theta = 0.04

p75 = 278 vs 276 at theta = 0. The "best theta" shifts left for lower percentiles and right for higher percentiles -- a fundamental property of the risk-variance tradeoff.

### Bonus rate as a proxy for policy divergence

Bonus rate drops 2.3% per 0.01 theta in [0, 0.10]. The upper section bonus is the most sensitive indicator of when the policy starts making different decisions.

### Regime boundaries (refined with dense grid)

- **theta < 0.03** (|theta| * sigma < 0.3): Near-EV behavior. Mean loss < 1.5 pts, all percentiles within 2 pts of baseline.
- **0.03 < theta < 0.20** (|theta| * sigma ~ 0.3-2): **Active tradeoff regime.** Upper percentiles (p95, p99) peak, mean declines ~1 pt per 0.01 theta.
- **theta > 0.20** (|theta| * sigma > 2): Diminishing returns. Policy changes slow down, all metrics decline monotonically.
- **theta > 1.5** (|theta| * sigma > 15): Full saturation. Policy frozen, no further change.

---

## Mean-Std Two-Branch Analysis

### Does (mean, std) trace a single curve across theta?

No. It traces **two separate branches** that do not overlap:

```
theta = -0.1:  mean=239, std=35.8
theta = +0.1:  mean=236, std=46.9   <-- 11 points higher std for similar mean!

theta = -5.0:  mean=187, std=37.1
theta = +5.0:  mean=187, std=44.1   <-- 7 points higher std for same mean!
```

At the same mean score, the risk-seeking policy (theta > 0) has 7-11 points higher std than the risk-averse policy (theta < 0). The two branches trace distinct paths in (mean, std) space:

- **Risk-averse branch** (theta < 0): std stays nearly flat at 36-38 as mean drops from 248 to 182. Risk-averse play compresses variance without moving it much.
- **Risk-seeking branch** (theta > 0): std rises to 48.4 (at theta ~ +0.2) then falls back to 44 as mean drops from 248 to 186. Risk-seeking play first inflates variance, then both mean and variance collapse as the policy becomes degenerate.

A single-curve quadratic fit `std = a + b*mean + c*mean^2` across all points gives **R^2 = 0.37** -- a poor fit, confirming these are two distinct branches.

The (mean, std) plot across all theta values is analogous to a **mean-variance frontier** from portfolio theory -- except unlike the classical efficient frontier (which is a single hyperbola), the Yatzy version has two branches because risk-averse and risk-seeking policies achieve the same mean loss through fundamentally different mechanisms (variance compression vs. variance inflation).

---

## Quadratic Fits for mean(theta) and std(theta)

When plotted against theta (not against each other), both mean and std are well-described by quadratics near the origin:

```
mean(theta) ~ 247.0 - 0.9*theta - 618*theta^2     (R^2 = 0.94)
std(theta)  ~ 39.5  + 53.5*theta - 17*theta^2      (R^2 = 0.91)
```

| theta | Mean | Mean fit | Resid | Std | Std fit | Resid |
|------:|------:|--------:|------:|-----:|-------:|------:|
| -0.100 | 239.2 | 240.9 | -1.7 | 35.8 | 33.9 | +1.9 |
| -0.050 | 245.1 | 245.5 | -0.4 | 35.7 | 36.7 | -1.0 |
| 0.000 | 248.4 | 247.0 | +1.4 | 38.5 | 39.5 | -1.0 |
| +0.050 | 244.5 | 245.4 | -0.9 | 43.2 | 42.1 | +1.1 |
| +0.100 | 235.9 | 240.7 | -4.8 | 46.9 | 44.6 | +2.3 |
| +0.200 | 223.8 | 222.1 | +1.7 | 48.4 | 49.5 | -1.1 |

The quadratic for mean(theta) is an **inverted parabola** peaking near theta ~ 0 (technically at theta = -0.0007). This makes sense: theta = 0 is the EV-optimal policy by definition, and small perturbations in either direction cause second-order losses in the mean -- exactly what you expect from perturbing an optimum (quadratic loss around the stationary point, or equivalently, the mean function has a negative-definite second derivative at theta = 0).

The quadratic for std(theta) is a **right-leaning parabola** peaking around theta ~ +0.2. The dominant term is linear: `std ~ 39.5 + 53.5*theta`, meaning risk-seeking policies increase variance roughly in proportion to theta, with a quadratic rolloff at larger theta as the policy degenerates.

---

## Adaptive theta: How Humans Actually Play

### The intuition

Real human players do not use a fixed theta. They adapt their risk tolerance based on the situation:

- Holding four 6s with 1 reroll left: go for Yatzy even if keeping four-of-a-kind has higher EV
- Leading late in a multiplayer game: play conservatively (risk-averse)
- Behind with 3 turns left: chase high-variance plays (risk-seeking)
- Early in the game with Yatzy still open: take speculative rerolls

This is an **adaptive policy** where theta = theta(S, d, n) -- risk tolerance varies with the game state S, the current dice d, and the rerolls remaining n. The fixed-theta solver is a special case where theta is constant.

### Why this matters

A fixed theta is a blunt instrument. The sweep shows:

| theta | Mean | p95 | Tradeoff |
|------:|------:|----:|----------|
| 0.000 | 248.4 | 309 | Maximum average score |
| +0.050 | 244.5 | 312 | Best p95, costs 4 points on average |
| +0.100 | 235.9 | 312 | Same p95, costs 12.5 points on average |

A human who is risk-seeking only in the right moments could potentially get the p95 benefit (+3 points) without paying the full mean penalty. The adaptive policy dominates any fixed-theta policy.

### Estimating theta from observed human decisions

At each decision point in a game, the player chooses an action (reroll mask or category). For most decisions, all theta values agree -- keep Yatzy if you have it, reroll garbage, etc. But at **pivotal decisions**, different theta values prescribe different actions. These pivotal decisions reveal the player's risk preference.

Rough estimate of pivotal decisions per game:

| theta | Mean loss vs theta=0 | ~Pivotal decisions per game |
|------:|------:|------:|
| +0.01 | 0.2 | ~0 |
| +0.05 | 3.9 | ~1-2 |
| +0.10 | 12.5 | ~4 |
| +0.20 | 24.6 | ~8 |

At theta = 0.05, only 1-2 decisions per game differ from theta = 0. This means we need many observed games to estimate a human's theta with any precision -- a single game has very little signal.

### Method: revealed theta per decision

For each observed decision (state, dice, action), we can compute:

1. **theta-optimal action map**: For a grid of theta values (e.g., -0.1 to +0.2 in steps of 0.01), compute the optimal action at that state.
2. **Compatible theta range**: The set of theta values for which the player's observed action IS the optimal action.
3. **Revealed theta**: The midpoint of the compatible range, or a likelihood-weighted estimate.

More formally, using a **softmax (logit) choice model**:

```
P(action a | state, theta, beta) ~ exp(beta * V_theta(a, state))
```

where V_theta(a, state) is the value of action a under policy theta, and beta is a "rationality" parameter (beta -> infinity means perfectly optimal play, beta -> 0 means random). Given a sequence of observed decisions, we fit (theta, beta) by maximum likelihood.

This is a standard **discrete choice model** from econometrics, applied to game decisions. The framework is known as **inverse reinforcement learning** in the ML literature, or **revealed preference** in economics.

### What we would need to log

The frontend already sends decisions to the backend. We would need per-decision records:

```json
{
  "upper_score": 12,
  "scored_categories": "0b000_0100_0001_0011",
  "dice": [2, 3, 5, 5, 5],
  "rerolls_remaining": 1,
  "action_type": "reroll",
  "action": "0b00110",
  "timestamp": "2026-02-11T21:30:00Z"
}
```

For category decisions:

```json
{
  "upper_score": 12,
  "scored_categories": "0b000_0100_0001_0011",
  "dice": [5, 5, 5, 5, 6],
  "rerolls_remaining": 0,
  "action_type": "category",
  "action": "fives",
  "timestamp": "2026-02-11T21:30:05Z"
}
```

### What we would learn

With enough logged games, we could estimate:

1. **Global theta per player**: Their average risk preference. Casual players likely have theta > 0 (risk-seeking -- they chase big scores). Experienced players likely hover near theta = 0.

2. **Situational theta**: How risk tolerance varies with:
   - Game phase (early/mid/late)
   - Current score relative to expected
   - Specific dice patterns (four-of-a-kind chasing Yatzy)
   - Whether Yatzy/straights are still available
   - Multiplayer position (ahead vs. behind)

3. **Rationality parameter beta**: How close to optimal the player is, independent of risk preference. Low beta means noisy/random decisions; high beta means the player is making deliberate risk-adjusted choices.

4. **Where humans deviate most from EV-optimal**: Which specific game situations trigger the largest theta shifts. Hypotheses:
   - Near-Yatzy hands (4-of-a-kind with 1 reroll): strong risk-seeking spike
   - Upper bonus threshold (close to 63): risk-averse to protect the 50-point bonus
   - Last 2-3 turns: theta depends on score gap in multiplayer

### Practical feasibility

The backend already has all the infrastructure needed:

- `api_computations.rs` computes optimal actions for a given state
- The risk-sensitive solver computes optimal actions for any theta
- Extending the API to return "optimal action for theta = [grid]" is straightforward
- The frontend already tracks game state and player decisions

The main challenge is **sample size**. With only 1-2 pivotal decisions per game (at the resolution of theta ~ 0.05), we would need ~50-100 observed games per player to estimate their theta within +/-0.02. For situational theta (e.g., "theta when holding four-of-a-kind"), we would need even more data since that situation arises in maybe 1 in 5 games.
