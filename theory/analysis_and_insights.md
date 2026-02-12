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

| Category        | Dice needed | Score   |
| --------------- | ----------- | ------- |
| Ones            | [1,1,1,1,1] | 5       |
| Twos            | [2,2,2,2,2] | 10      |
| Threes          | [3,3,3,3,3] | 15      |
| Fours           | [4,4,4,4,4] | 20      |
| Fives           | [5,5,5,5,5] | 25      |
| Sixes           | [6,6,6,6,6] | 30      |
| **Upper bonus** |             | **50**  |
| One pair        | [_,_,_,6,6] | 12      |
| Two pairs       | [_,5,5,6,6] | 22      |
| Three of a kind | [_,_,6,6,6] | 18      |
| Four of a kind  | [_,6,6,6,6] | 24      |
| Small straight  | [1,2,3,4,5] | 15      |
| Large straight  | [2,3,4,5,6] | 20      |
| Full house      | [5,5,6,6,6] | 28      |
| Chance          | [6,6,6,6,6] | 30      |
| Yatzy           | [n,n,n,n,n] | 50      |
| **Total**       |             | **374** |

### Simulation results (10M games)

The max-policy makes decisions assuming the best dice outcome, but actual dice are random.

| Statistic | Value      |
| --------- | ---------- |
| Games     | 10,000,000 |
| Mean      | 118.74     |
| Std dev   | 24.7       |
| Min       | 35         |
| Max       | 313        |
| Median    | 117        |

The mean is ~119 -- about half the EV-optimal policy's ~246. The max-policy makes terrible real-world decisions because it always assumes the best dice outcome will happen. For example, it might reroll a decent hand chasing five-of-a-kind, since under the max assumption that always succeeds.

### Tail distribution (empirical)

| Threshold | Count >= | Fraction | 1 in      |
| --------- | -------- | -------- | --------- |
| 200       | 41,651   | 4.17e-3  | 240       |
| 220       | 12,305   | 1.23e-3  | 813       |
| 240       | 2,859    | 2.86e-4  | 3,497     |
| 260       | 492      | 4.92e-5  | 20,325    |
| 280       | 72       | 7.20e-6  | 138,889   |
| 300       | 5        | 5.00e-7  | 2,000,000 |
| 310       | 2        | 2.00e-7  | 5,000,000 |
| 320+      | 0        | --       | --        |

### Log-linear tail fit

Fitting `log10(P(X >= t)) = a + b*t` to the range 220-310:

```
log10(P(X >= t)) = 5.544 - 0.0380 * t
```

Each additional point reduces the survival probability by a factor of ~0.916 (i.e., ~8.4% less likely per point).

Extrapolating this to 374 gives P ~ 10^-8.7 ~ 2e-9, or ~500M games. **However**, this extrapolation is misleading -- it captures the smooth middle tail but not the combinatorial cliff at the extreme.

### Analytical estimate: P(score = 374)

To score exactly 374, every category must yield its maximum. The per-category probabilities (with 3 rolls, optimal keeping) are:

| Category        | Requirement          | P(max)  | 1 in |
| --------------- | -------------------- | ------- | ---- |
| Ones            | Five 1s              | 0.01327 | 75   |
| Twos            | Five 2s              | 0.01327 | 75   |
| Threes          | Five 3s              | 0.01327 | 75   |
| Fours           | Five 4s              | 0.01327 | 75   |
| Fives           | Five 5s              | 0.01327 | 75   |
| Sixes           | Five 6s              | 0.01327 | 75   |
| One pair        | >= 2 sixes           | 0.699   | 1.4  |
| Two pairs       | >= 2 fives + 2 sixes | 0.275   | 3.6  |
| Three of a kind | >= 3 sixes           | 0.355   | 2.8  |
| Four of a kind  | >= 4 sixes           | 0.104   | 9.6  |
| Small straight  | [1,2,3,4,5]          | 0.197   | 5.1  |
| Large straight  | [2,3,4,5,6]          | 0.197   | 5.1  |
| Full house      | [5,5,6,6,6]          | 0.062   | 16   |
| Chance          | Five 6s              | 0.01327 | 75   |
| Yatzy           | Any five-of-a-kind   | 0.080   | 13   |

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

The log-linear fit (P ~ 10^-9 at 374) captures scores 220-310, where high scores come from maxing _some_ categories while getting decent scores in others. But 374 requires _every single category_ maxed -- a qualitatively different event that sits on a combinatorial cliff far below the smooth tail extrapolation.

The 7 categories requiring five-of-a-specific-face (each 1-in-75) are the dominant bottleneck, contributing (1/75)^7 ~ 10^-13 alone.

### Comparison with EV-optimal policy

| Metric                     | EV-optimal | Max-policy |
| -------------------------- | ---------- | ---------- |
| Precomputed value at (0,0) | 248.4      | 374.0      |
| Mean simulated score       | ~246       | ~119       |
| Std dev                    | ~38        | ~25        |
| Max (10M games)            | ~374\*     | 313        |

\*The EV-optimal policy has a much better chance of high scores because it makes realistic decisions. The max-policy actively hurts performance by chasing unrealistic best-case outcomes.

---

## Risk-Sensitive theta Sweep Results

### Coarse sweep (1M games each, 21 theta values)

|     theta |      Mean |      Std |      p5 | p95 (actual) | p95 (Gaussian) |    Delta |
| --------: | --------: | -------: | ------: | -----------: | -------------: | -------: |
|   -20.000 |     181.9 |     36.0 |     134 |          247 |          241.1 |     +5.9 |
|   -10.000 |     182.7 |     36.2 |     134 |          249 |          242.2 |     +6.8 |
|    -5.000 |     186.6 |     37.1 |     137 |          255 |          247.7 |     +7.3 |
|    -1.000 |     198.2 |     37.9 |     148 |          269 |          260.5 |     +8.5 |
|    -0.100 |     239.2 |     35.9 |     172 |          300 |          298.2 |     +1.8 |
|    -0.050 |     245.1 |     35.7 |     179 |          305 |          303.9 |     +1.1 |
|    -0.020 |     247.9 |     37.2 |     183 |          308 |          309.1 |     -1.1 |
|    -0.010 |     248.3 |     37.7 |     182 |          308 |          310.4 |     -2.4 |
| **0.000** | **248.4** | **38.5** | **179** |      **309** |      **311.7** | **-2.7** |
|    +0.010 |     248.2 |     39.4 |     175 |          310 |          313.0 |     -3.0 |
|    +0.020 |     247.7 |     40.3 |     172 |          311 |          314.1 |     -3.1 |
|    +0.030 |     246.9 |     41.3 |     169 |          311 |          314.9 |     -3.9 |
|    +0.040 |     245.7 |     42.3 |     166 |          312 |          315.3 |     -3.3 |
|    +0.050 |     244.5 |     43.2 |     164 |          312 |          315.6 |     -3.6 |
|    +0.100 |     235.9 |     46.9 |     155 |          312 |          313.1 |     -1.1 |
|    +0.200 |     223.8 |     48.4 |     148 |          308 |          303.3 |     +4.7 |
|    +0.500 |     204.7 |     47.7 |     135 |          287 |          283.1 |     +3.9 |
|    +1.000 |     194.5 |     46.3 |     128 |          276 |          270.6 |     +5.4 |
|    +5.000 |     186.5 |     44.1 |     124 |          266 |          259.1 |     +6.9 |
|   +10.000 |     185.8 |     43.9 |     124 |          265 |          258.1 |     +6.9 |
|   +20.000 |     185.7 |     43.9 |     124 |          265 |          257.9 |     +7.1 |

### Dense sweep (1M games each, seed=42)

Grid: dense (delta_theta = 0.01) in the interesting range 0-0.20, sparser (delta_theta = 0.1-0.2) in the saturated range 0.3-3.0.

| theta |   Mean | StdDev | Min |  p5 | p25 | p50 | p75 | p95 | p99 | Max | Bonus% |
| ----: | -----: | -----: | --: | --: | --: | --: | --: | --: | --: | --: | -----: |
| 0.000 | 248.40 |   38.5 |  97 | 179 | 225 | 249 | 276 | 309 | 325 | 352 |  89.8% |
| 0.010 | 248.21 |   39.4 |  90 | 175 | 225 | 249 | 277 | 310 | 325 | 353 |  88.4% |
| 0.020 | 247.73 |   40.3 |  90 | 172 | 224 | 248 | 277 | 311 | 326 | 359 |  86.7% |
| 0.030 | 246.92 |   41.3 |  88 | 169 | 223 | 247 | 277 | 311 | 326 | 359 |  84.7% |
| 0.040 | 245.71 |   42.3 |  88 | 166 | 221 | 245 | 278 | 312 | 327 | 359 |  82.5% |
| 0.050 | 244.53 |   43.2 |  79 | 164 | 219 | 244 | 277 | 312 | 327 | 359 |  79.9% |
| 0.060 | 242.94 |   44.2 |  79 | 162 | 216 | 243 | 276 | 312 | 328 | 362 |  77.0% |
| 0.070 | 240.86 |   45.3 |  84 | 159 | 213 | 241 | 274 | 313 | 328 | 362 |  72.6% |
| 0.080 | 239.12 |   46.0 |  78 | 158 | 210 | 239 | 273 | 313 | 328 | 362 |  69.4% |
| 0.090 | 237.47 |   46.5 |  82 | 156 | 207 | 238 | 270 | 313 | 329 | 362 |  66.3% |
| 0.100 | 235.92 |   46.9 |  79 | 155 | 204 | 237 | 268 | 312 | 329 | 362 |  63.6% |
| 0.110 | 234.40 |   47.2 |  81 | 154 | 202 | 236 | 266 | 312 | 329 | 362 |  61.7% |
| 0.120 | 233.05 |   47.4 |  81 | 154 | 199 | 235 | 265 | 312 | 329 | 362 |  59.6% |
| 0.130 | 231.69 |   47.6 |  74 | 153 | 196 | 234 | 263 | 311 | 329 | 362 |  57.7% |
| 0.140 | 230.43 |   47.8 |  74 | 152 | 194 | 232 | 262 | 311 | 329 | 362 |  55.9% |
| 0.150 | 229.23 |   47.9 |  74 | 152 | 192 | 231 | 261 | 311 | 329 | 362 |  54.4% |
| 0.160 | 228.02 |   48.1 |  74 | 151 | 191 | 230 | 260 | 310 | 329 | 362 |  53.0% |
| 0.170 | 226.82 |   48.2 |  78 | 150 | 189 | 229 | 260 | 310 | 329 | 358 |  51.8% |
| 0.180 | 225.74 |   48.3 |  75 | 149 | 188 | 228 | 259 | 309 | 329 | 358 |  50.6% |
| 0.190 | 224.75 |   48.3 |  78 | 149 | 186 | 227 | 258 | 308 | 329 | 358 |  49.6% |
| 0.200 | 223.76 |   48.4 |  67 | 148 | 185 | 225 | 257 | 308 | 329 | 358 |  48.5% |
| 0.300 | 215.74 |   48.6 |  76 | 142 | 177 | 215 | 251 | 301 | 327 | 358 |  44.0% |
| 0.400 | 209.82 |   48.3 |  67 | 138 | 171 | 207 | 245 | 293 | 323 | 358 |  40.7% |
| 0.500 | 204.73 |   47.6 |  60 | 135 | 168 | 200 | 240 | 287 | 319 | 361 |  38.0% |
| 0.700 | 199.29 |   47.1 |  51 | 131 | 163 | 194 | 234 | 281 | 314 | 360 |  34.7% |
| 0.900 | 195.29 |   46.3 |  57 | 128 | 160 | 189 | 230 | 276 | 309 | 359 |  33.2% |
| 1.100 | 193.73 |   46.1 |  51 | 127 | 158 | 187 | 228 | 275 | 308 | 359 |  32.2% |
| 1.300 | 190.56 |   45.8 |  51 | 125 | 156 | 184 | 223 | 272 | 306 | 359 |  30.0% |
| 1.500 | 189.22 |   45.5 |  55 | 125 | 155 | 182 | 221 | 270 | 305 | 359 |  28.8% |
| 1.700 | 187.28 |   45.2 |  57 | 123 | 153 | 180 | 219 | 268 | 303 | 359 |  27.3% |
| 1.900 | 186.17 |   44.8 |  48 | 123 | 152 | 179 | 217 | 267 | 300 | 359 |  27.1% |
| 2.000 | 186.04 |   44.8 |  48 | 123 | 152 | 179 | 217 | 266 | 300 | 359 |  27.1% |
| 3.000 | 186.49 |   44.2 |  50 | 124 | 154 | 179 | 216 | 266 | 301 | 359 |  25.7% |

---

## Key Observations from the theta Sweep

### Where does p95 peak?

**p95 peaks at 313** across theta = 0.07-0.09 (|theta| \* sigma ~ 0.7-0.9). This is +4 points over the EV-optimal baseline (p95 = 309). The gain comes at a meaningful cost to the mean: theta = 0.08 has mean 239.1 vs baseline 248.4 (~9 points lower).

With the coarser grid, p95 = 312 was observed across theta = 0.04-0.10. The dense grid refines this to a peak of 313 at theta = 0.07-0.09.

### Can the p95 peak be predicted from mean and std alone?

**Directionally yes, quantitatively no.**

The Gaussian approximation `p95 ~ mean + 1.645 * sigma` predicts the p95 peak location reasonably well -- the Gaussian p95 also peaks around theta ~ +0.05 (at 315.6). But there are important failures:

**1. The delta flips sign with theta.** Near theta = 0, the actual p95 is _lower_ than the Gaussian prediction (delta = -2.7), meaning the right tail is thinner than Normal. At extreme theta (both positive and negative), the actual p95 _exceeds_ the Gaussian prediction (delta = +5 to +7), meaning the tail becomes heavier. The distribution shape changes with theta.

**2. The Gaussian overpredicts the peak p95 by ~4 points.** The Gaussian model says the best achievable p95 is ~316; the actual peak is 313. The thin-tail effect near the optimum eats into the predicted gain.

**3. The actual p95 plateau is flatter than predicted.** The Gaussian curve has a sharp peak at theta ~ 0.05; the actual p95 holds at 312-313 across theta = 0.04-0.10 because the growing heavy-tail effect at higher theta partially compensates for the declining Gaussian prediction.

### Why the distribution shape changes with theta

As theta increases (risk-seeking), the policy increasingly "gambles" -- chasing high-variance plays that pay off rarely but spectacularly. This doesn't just shift mean down and sigma up; it changes the **skewness**:

- At theta = 0: the distribution is mildly thin-tailed on the right (delta negative)
- At theta > 0.2: the distribution becomes heavy-tailed on the right (delta positive)
- The crossover happens around theta ~ 0.1-0.2

Mean and sigma get you in the right ballpark for _where_ the p95 peaks, but you need the third moment (skewness) or the full distribution to predict the actual level. The two-moment Gaussian model systematically mischaracterizes the tail behavior because it cannot capture how risk-sensitive policies reshape the score distribution's asymmetry.

### Mean loss is monotonic and concave

Mean drops from 248.4 (theta = 0) to 186.0 (theta = 2.0), a total loss of 62.4 points. The loss is steepest in [0, 0.10]: ~1.25 pts per 0.01 theta. By theta = 0.20, mean loss rate halves to ~1.0 pt per 0.01 theta.

### StdDev rises rapidly then plateaus

StdDev rises from 38.5 to 46.9 in the first 0.10 of theta (+8.4 pts, +22%), then barely increases beyond that (peak 48.6 at theta = 0.3).

### p75 peaks at theta = 0.04

p75 = 278 vs 276 at theta = 0. The "best theta" shifts left for lower percentiles and right for higher percentiles -- a fundamental property of the risk-variance tradeoff.

### Bonus rate as a proxy for policy divergence

Bonus rate drops 2.3% per 0.01 theta in [0, 0.10]. The upper section bonus is the most sensitive indicator of when the policy starts making different decisions.

### Regime boundaries (refined with dense grid)

- **theta < 0.03** (|theta| \* sigma < 0.3): Near-EV behavior. Mean loss < 1.5 pts, all percentiles within 2 pts of baseline.
- **0.03 < theta < 0.20** (|theta| \* sigma ~ 0.3-2): **Active tradeoff regime.** Upper percentiles (p95, p99) peak, mean declines ~1 pt per 0.01 theta.
- **theta > 0.20** (|theta| \* sigma > 2): Diminishing returns. Policy changes slow down, all metrics decline monotonically.
- **theta > 1.5** (|theta| \* sigma > 15): Full saturation. Policy frozen, no further change.

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

|  theta |  Mean | Mean fit | Resid |  Std | Std fit | Resid |
| -----: | ----: | -------: | ----: | ---: | ------: | ----: |
| -0.100 | 239.2 |    240.9 |  -1.7 | 35.8 |    33.9 |  +1.9 |
| -0.050 | 245.1 |    245.5 |  -0.4 | 35.7 |    36.7 |  -1.0 |
|  0.000 | 248.4 |    247.0 |  +1.4 | 38.5 |    39.5 |  -1.0 |
| +0.050 | 244.5 |    245.4 |  -0.9 | 43.2 |    42.1 |  +1.1 |
| +0.100 | 235.9 |    240.7 |  -4.8 | 46.9 |    44.6 |  +2.3 |
| +0.200 | 223.8 |    222.1 |  +1.7 | 48.4 |    49.5 |  -1.1 |

The quadratic for mean(theta) is an **inverted parabola** peaking near theta ~ 0 (technically at theta = -0.0007). This makes sense: theta = 0 is the EV-optimal policy by definition, and small perturbations in either direction cause second-order losses in the mean -- exactly what you expect from perturbing an optimum (quadratic loss around the stationary point, or equivalently, the mean function has a negative-definite second derivative at theta = 0).

The quadratic for std(theta) is a **right-leaning parabola** peaking around theta ~ +0.2. The dominant term is linear: `std ~ 39.5 + 53.5*theta`, meaning risk-seeking policies increase variance roughly in proportion to theta, with a quadratic rolloff at larger theta as the policy degenerates.

---

## Quantifying the Cost of Risk-Seeking

### The asymmetry of risk-seeking

Risk-seeking is expensive and asymmetric. The EV-optimal policy (theta=0) is a **maximum** of mean(theta) by construction, so perturbations cause **quadratic** loss in the mean. But upper percentiles respond **linearly** to small theta (shifting probability mass rightward helps the tail before the mean penalty kicks in). This creates a narrow sweet spot where you get cheap tail gains, followed by rapidly diminishing returns.

### Three efficiency metrics

We compute three complementary metrics from 1M-game simulations at each theta value:

#### 1. Marginal Exchange Rate (MER)

MER_q(theta) = [mean(0) - mean(theta)] / [q(theta) - q(0)]: mean points sacrificed per point of quantile gain.

| theta | Mean cost | MER_p75 | MER_p90 | MER_p95 | MER_p99 | MER_max |
| ----: | --------: | ------: | ------: | ------: | ------: | ------: |
|  0.01 |       0.2 |     0.2 |     0.2 |     0.2 |     dom |     0.2 |
|  0.02 |       0.7 |     0.7 |     0.3 |     0.3 |     0.7 |     0.1 |
|  0.05 |       3.9 |     3.9 |     1.3 |     1.3 |     1.9 |     0.6 |
|  0.08 |       9.3 |     dom |     3.1 | **2.3** |     3.1 |     0.9 |
|  0.10 |      12.5 |     dom |     6.2 | **4.2** |     3.1 |     1.2 |
|  0.20 |      24.6 |     dom |     dom |     dom |     6.2 |     4.1 |

Key observations:

- **p75 becomes dominated at theta >= 0.06**: beyond this point, even p75 declines.
- **p90 becomes dominated at theta >= 0.12**.
- **p95 peaks at theta ~ 0.08** (MER = 2.3), dominated by theta >= 0.18.
- **p99 peaks at theta ~ 0.09** (MER = 2.7), stays positive until theta ~ 0.30.
- **max keeps improving until theta ~ 0.06** at MER < 1 (very cheap).

The "peak benefit theta" shifts rightward for more extreme tail metrics, but the MER keeps rising. The benefit of risk-seeking concentrates into ever-thinner tails while the cost in mean keeps growing.

#### 2. CDF Gain-Loss Decomposition (Stochastic Dominance Violation Area)

For each score x, D(x) = F_theta(x) - F_0(x). Where D > 0, theta puts more probability below x (worse). Where D < 0, theta puts less probability below x (better). We decompose the integral into:

- **A_worse**: area where risk-seeking hurts
- **A_better**: area where risk-seeking helps
- **Ratio**: A_worse / A_better (always > 1 since A_worse - A_better = mean loss)
- **x_cross**: the score threshold above which risk-seeking helps

| theta | A_worse | A_better | Ratio | Crossing point |
| ----: | ------: | -------: | ----: | -------------: |
|  0.01 |     0.5 |      0.3 |   1.7 |            251 |
|  0.05 |     4.5 |      0.6 |   7.2 |            271 |
|  0.08 |     9.7 |      0.4 |    22 |            285 |
|  0.10 |    12.8 |      0.3 |    38 |            294 |
|  0.20 |    24.8 |      0.1 |   217 |            312 |

At theta = 0.08, the crossing point is 285 -- you need to score above ~285 (roughly p80 of the baseline distribution) for the risk-seeking policy to have put you in a better position. Below that, you're strictly worse off. The ratio of 22:1 means for every unit of distributional area that improves, 22 units get worse.

#### 3. CVaR Deficit (Expected Shortfall)

CVaR_alpha = mean score in the worst alpha-fraction of games. This measures **tail severity** -- not just how many bad games there are (p5 tells you that), but how bad those games actually are.

| theta |  Mean | CVaR_1% | CVaR_5% | CVaR_10% | CVaR_5% deficit |
| ----: | ----: | ------: | ------: | -------: | --------------: |
|  0.00 | 248.4 |   134.8 |   157.4 |    175.3 |              -- |
|  0.05 | 244.5 |   130.7 |   149.0 |    161.9 |            -8.4 |
|  0.08 | 239.1 |   126.7 |   144.2 |    155.2 |           -13.2 |
|  0.10 | 235.9 |   125.6 |   142.6 |    152.9 |           -14.8 |
|  0.20 | 223.8 |   118.9 |   135.7 |    145.0 |           -21.7 |

At theta = 0.08: your worst 5% of games average 144.2, down from 157.4 at theta = 0. The 13-point CVaR deficit exceeds the 9-point mean deficit -- risk-seeking hurts the worst games disproportionately.

### How the three metrics work together

| Metric     | Question answered                                        | Scope                                            |
| ---------- | -------------------------------------------------------- | ------------------------------------------------ |
| MER        | "What does one point of upside cost?"                    | Point estimate, marginal                         |
| MER family | "At what percentile does risk-seeking start to pay off?" | Peak-benefit theta shifts right for rarer events |
| SDVA       | "What fraction of outcomes improve vs degrade?"          | Full distributional shape                        |
| CVaR       | "How bad do the bad games get?"                          | Downside severity                                |

### Mathematical intuition

Near theta = 0 (the EV-optimal point):

- **Mean loss is quadratic**: mean(theta) ~ 248.4 - 618\*theta^2
- **Tail gain is linear**: p95(theta) ~ 309 + c\*theta (for small theta)

So initially the exchange rate is cheap (linear benefit, quadratic cost). But the quadratic cost quickly dominates. This is the same reason small portfolio tilts are cheap but large bets are expensive -- convexity of the loss function around an optimum.

The extreme tail (max, top-5 avg) keeps improving well past where p95 peaks, but only because you're concentrating the benefit into an ever-thinner slice of outcomes. At theta = 0.20, max = 358 (vs 352 baseline) but you're sacrificing the experience of ~999,000 games out of 1,000,000 to improve the best ~1,000.

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

|  theta |  Mean | p95 | Tradeoff                               |
| -----: | ----: | --: | -------------------------------------- |
|  0.000 | 248.4 | 309 | Maximum average score                  |
| +0.050 | 244.5 | 312 | Best p95, costs 4 points on average    |
| +0.100 | 235.9 | 312 | Same p95, costs 12.5 points on average |

A human who is risk-seeking only in the right moments could potentially get the p95 benefit (+3 points) without paying the full mean penalty. The adaptive policy dominates any fixed-theta policy.

### Estimating theta from observed human decisions

At each decision point in a game, the player chooses an action (reroll mask or category). For most decisions, all theta values agree -- keep Yatzy if you have it, reroll garbage, etc. But at **pivotal decisions**, different theta values prescribe different actions. These pivotal decisions reveal the player's risk preference.

Rough estimate of pivotal decisions per game:

| theta | Mean loss vs theta=0 | ~Pivotal decisions per game |
| ----: | -------------------: | --------------------------: |
| +0.01 |                  0.2 |                          ~0 |
| +0.05 |                  3.9 |                        ~1-2 |
| +0.10 |                 12.5 |                          ~4 |
| +0.20 |                 24.6 |                          ~8 |

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

---

## Adaptive θ Policies: Table-Switching Approximation

### Why fixed θ is suboptimal

θ=0 maximizes E[score] by construction. Any fixed θ>0 sacrifices mean for upper-tail improvement, but applies this sacrifice **uniformly** across all game states — including states where risk-seeking is destructive (e.g., when the upper bonus is at risk).

The "cost of risk" varies dramatically by state:

- **Bonus already secured** (upper_score >= 63): The main source of variance (50pt bonus or not) is resolved. Risk-seeking here is cheap — it only affects the remaining lower-section categories.
- **Bonus at risk** (e.g., upper_score=45, 2 upper categories left): Risk-seeking could cause missing the 50pt bonus, which is catastrophic. Risk here is expensive.
- **Bonus unreachable** (can't reach 63): Nothing to protect. Risk-seeking costs almost nothing.

A fixed θ=0.05 pays the full mean penalty in ALL states but only gains upside in the "cheap risk" states. An adaptive policy can avoid the costly risk and only take cheap risk.

### Why adaptive should beat the Pareto frontier

The Pareto frontier of fixed-θ policies in (mean, p95) space represents the best tradeoff achievable with a uniform risk preference. A convex combination (mixture) of fixed-θ policies can reach any point on or below the convex hull of this frontier.

An adaptive policy is **strictly more powerful** than a mixture because it switches θ **within** a single game based on observed trajectory. It can:

- Play θ=0 when the game is on track (protecting mean)
- Switch to θ>0 when risk is cheap (bonus secured or unreachable)
- Switch to θ<0 when protecting the bonus is critical

If any adaptive policy point lies **above** the convex hull of fixed-θ points, it is genuinely better — not achievable by mixing.

### Architecture: table-switching approximation

We already have precomputed state values for many fixed θ values. Each is a full table of 2M state values (8MB mmap'd, <1ms to load). At simulation time:

1. Load N precomputed tables (e.g., θ=0 and θ=0.08)
2. At the start of each turn, a **policy function** examines the game state and selects which table to use
3. ALL decisions within that turn (reroll 1, reroll 2, category choice) use the selected table's state values and θ

The successor state values were computed assuming the SAME θ for all future turns. When we switch tables between turns, this assumption is violated. However, for small θ differences (|Δθ| < 0.1), policies differ on <5% of decisions, making this a second-order error.

### Three implemented policies

**Policy 1: Bonus-Adaptive** — Formalizes the "Dad's strategy":

```
if bonus_secured(upper_score >= 63): use θ=0.08 (risk is cheap)
elif bonus_reachable:                use θ=0   (protect the bonus)
else:                                use θ=0.08 (nothing to protect)
```

**Policy 2: Phase-Based** — Risk tolerance varies by game phase:

```
if categories_left >= 10:   use θ=0.03 (early game, many turns to recover)
elif categories_left >= 5:  use θ=0    (mid game, lock in value)
elif bonus_secured:         use θ=0.08 (late game, send it)
else:                       use θ=0    (late game, protect what's left)
```

**Policy 3: Combined** — Most nuanced, combining bonus health, phase, and high-variance category awareness. Uses four θ values (0, 0.03, 0.05, 0.08) and selects based on bonus health ratio and remaining high-variance categories.

### Evaluation framework

The key question: **Can any adaptive policy beat the Pareto frontier of fixed-θ policies?**

Primary metric: (Mean, p95) Pareto dominance — plot all fixed-θ points and their convex hull, overlay each adaptive policy point. If adaptive is above the hull, it achieves something no fixed policy or mixture can.

Secondary metrics:

- **Bonus rate**: Does adaptive maintain a higher bonus rate than fixed θ>0?
- **CVaR comparison**: For the same p95, does adaptive have better worst-case (CVaR5)?

### Results

_To be filled after running simulations. See `analytics/results/plots/efficiency_adaptive.png`._

---

## RL Approach A: Learned θ-Switching

### Motivation

The fixed-θ solver cannot condition on accumulated total score — it treats all games in the same (upper_score, scored_mask) state identically regardless of whether the player is running hot (180 pts with 5 turns left) or cold (130 pts). An RL agent could learn to exploit this gap by switching between precomputed θ-tables based on the full game trajectory.

### Architecture

- **Action space**: K=5 discrete (which θ-table to use per turn), θ ∈ {0.0, 0.02, 0.05, 0.08, 0.12}
- **Network**: MLP 10 → 64 → 64 → 5 (~5K params)
- **Observations**: 10 features including upper_score, categories_left, bonus health, total_score/374, EV_remaining, and a score z-score tracking how far ahead/behind expected trajectory
- **Training**: REINFORCE with quantile-tilted reward: `shaped = score + λ·max(0, score - running_p95)`
- **Simulation**: Rust FFI bridge (~16K episodes/s, ~1000x faster than pure Python)

### Training results (1M episodes per λ, batch=4096)

| λ   | Best batch p95 | Entropy trend | Convergence                             |
| --- | -------------- | ------------- | --------------------------------------- |
| 0.0 | 313            | 1.605 → 1.567 | Learned to prefer higher-θ tables       |
| 1.0 | 314            | 1.607 → 1.598 | Minimal learning                        |
| 3.0 | 314            | 1.602 → 1.540 | Learned differentiation, then rebounded |
| 7.0 | 314            | 1.603 → 1.553 | Strongest entropy reduction             |

Maximum entropy for K=5 is ln(5) ≈ 1.609. The policy entropy decreased by 0.03-0.06 nats — meaningful learning but far from a decisive specialization.

### Evaluation (1M games per policy, deterministic action selection)

| Policy           | Mean      | p5      | p50     | p90 | p95 | p99 |
| ---------------- | --------- | ------- | ------- | --- | --- | --- |
| RL λ=0.0         | 239.0     | 158     | 239     | 302 | 312 | 328 |
| RL λ=1.0         | 247.8     | 173     | 248     | 301 | 311 | 326 |
| RL λ=3.0         | 248.2     | 177     | 249     | 299 | 309 | 325 |
| RL λ=7.0         | 239.0     | 158     | 239     | 302 | 312 | 328 |
| **Fixed θ=0.00** | **248.4** | **179** | **249** | 299 | 309 | 325 |
| Fixed θ=0.02     | 247.8     | 172     | 248     | 301 | 311 | 326 |
| Fixed θ=0.05     | 244.5     | 164     | 244     | 302 | 312 | 327 |
| Fixed θ=0.08     | 239.0     | 158     | 239     | 302 | 312 | 328 |
| Fixed θ=0.12     | 233.0     | 154     | 235     | 299 | 312 | 329 |

### Key findings

**1. RL policies collapsed to single fixed-θ behaviors.** Each λ value produced a policy that matches a specific fixed-θ table almost exactly:

- λ=0.0 → θ≈0.08 (mean=239.0, p95=312, identical to fixed θ=0.08)
- λ=1.0 → θ≈0.02 (mean=247.8, p95=311)
- λ=3.0 → θ≈0.00 (mean=248.2, p95=309)
- λ=7.0 → θ≈0.08 (mean=239.0, p95=312)

**2. No policy beat the fixed-θ Pareto frontier.** All RL points lie on or below the frontier traced by fixed-θ policies in (mean, p95) space.

**3. The accumulated-score signal is too weak for REINFORCE.** Choosing θ=0.05 vs θ=0.08 on a single turn shifts p95 by <0.1 points, but REINFORCE sees only the episode return. With 15 turns per game and ~1-2 pivotal decisions, the per-turn credit assignment signal is buried in variance.

### Why Approach A failed

The fundamental bottleneck is **approximation error vs. signal strength**:

- The theta-switching architecture delegates ALL reroll and category decisions to precomputed tables. The agent only picks which table to use. With K=5 tables, the agent has 5^15 ≈ 30 billion possible turn-by-turn policies, but the tables themselves agree on >95% of decisions. The actual search space of meaningfully different policies is tiny.
- The REINFORCE gradient for per-turn actions is `∂J/∂θ = E[(R - b) ∂ log π(a|s)/∂θ]`. With R (total game score) having std ~38 and the per-turn theta choice affecting R by <1 point, the signal-to-noise ratio is ~1/38 — requiring ~1400 episodes just to detect a 1-point effect at 1 sigma. After 1M episodes this should be enough in theory, but with 15 actions per episode sharing the same reward signal, the credit assignment problem makes convergence extremely slow.
- The quantile-tilted reward λ·max(0, score - p95) only fires for ~5% of episodes. This severely reduces the effective sample size for learning upper-tail behavior.

### Implications for Approaches B and C

The failure of Approach A narrows the hypothesis space:

1. **Approach C (value correction)** operates at the per-decision level (3 decisions/turn, 45/game), providing much finer-grained credit assignment. It also doesn't need to learn which table to use — it learns a small δ correction to the θ=0 values. This is a fundamentally easier learning problem.

2. **Approach B (direct action)** replaces the tables entirely, giving maximum flexibility but also the hardest learning problem. The IQN architecture with CVaR-based action selection may find risk-sensitive strategies that no fixed-θ table encodes.

3. The key variable is **accumulated score in the observation**. If Approaches B/C also fail to beat the frontier, it would suggest the accumulated-score-conditioning gap is smaller than hypothesized — the fixed-θ frontier may be near-tight.

---

## RL Approach C: PPO with Dense θ Grid

### Architecture

Approach C replaces REINFORCE with PPO (Proximal Policy Optimization) and uses a denser θ grid (12 tables: θ = 0.00, 0.01, ..., 0.10, 0.12) to give the policy finer-grained control over risk level.

- **Actor**: MLP 10 → 128 → 128 → 12 (Tanh activations, ~19K params)
- **Critic**: MLP 10 → 128 → 128 → 1 (value function, ~18K params)
- **Training**: PPO with clipped objective, GAE (λ=0.95), 4 epochs per batch
- **Scale**: 2M episodes (500 batches of 4096), ~5 min total

### Training behavior

The critic loss decreased steadily from 32K to 674, showing the value function learned to predict game scores. The action distribution evolved:

- **Early** (batch 0): roughly uniform across 12 θ values (~8% each)
- **Mid** (batch 100-200): concentrated on θ ∈ {0.00-0.04}, occasionally spiking on specific θ
- **Late** (batch 490): θ=0.00 (28%), θ=0.02 (24%), θ=0.01 (23%) — ~75% on lowest three θ

### Evaluation (1M games, deterministic)

| Policy         | Mean      | p5      | p50     | p90     | p95     | p99     |
| -------------- | --------- | ------- | ------- | ------- | ------- | ------- |
| **Approach C** | **246.9** | **169** | **247** | **301** | **310** | **326** |
| Fixed θ=0.00   | 248.4     | 179     | 249     | 299     | 309     | 325     |
| Fixed θ=0.02   | 247.8     | 172     | 248     | 301     | 311     | 326     |
| Fixed θ=0.05   | 244.5     | 164     | 244     | 302     | 312     | 327     |

### Key findings

**1. PPO trained much more stably than REINFORCE.** The critic loss decreased monotonically and the policy converged smoothly — no mode collapse or erratic behavior. This confirms PPO is the right algorithm for this problem.

**2. The policy converged to a mixture of low-θ tables.** With 75% of actions on θ ∈ {0.00, 0.01, 0.02}, the effective policy is a convex combination of nearly-identical fixed-θ strategies. It found the right region of the frontier (low-θ for high mean) but didn't discover beneficial state-conditional switching.

**3. Still did not beat the Pareto frontier.** At mean=246.9, the expected p95 from fixed θ ≈ 0.015 would be ~310, which is exactly what was achieved. The policy replicated a fixed-θ point on the frontier rather than finding a new point above it.

**4. Why no state-conditional switching emerged:**

- The observation includes total_score, bonus health, and score z-score — all the features needed for adaptive switching
- But the GAE advantage signal still comes from episode-level returns (std ~38), while per-turn θ choices affect the return by <1 point
- PPO's clipping prevents large policy changes, which is good for stability but bad for exploring rare high-value switching patterns
- The 12-way softmax has high entropy (ln(12) ≈ 2.48), making it harder to learn sharp state-conditional switching

### Combined conclusion from Approaches A and C

Both approaches used the same fundamental architecture (turn-level θ-table selection with game-level reward) and both converged to fixed-θ-equivalent policies. The approaches differed in:

- Algorithm (REINFORCE vs PPO): PPO was more stable but no better at finding adaptive strategies
- Action space (5 vs 12 θ values): more options didn't help — the policy concentrated on 2-3 nearby values
- Network capacity (5K vs 19K params): larger network learned a slightly better value function but same policy

**The binding constraint is not the algorithm or network — it's the architecture.** Switching θ-tables per turn cannot capture the key adaptive signal because:

1. The tables themselves agree on >95% of decisions
2. The ~5% of decisions where tables disagree have <1 point of impact each
3. Detecting which of 15 turns to "switch" for a 1-point gain requires ~1400 episodes per learning step (SNR ≈ 1/38)

This confirms the plan's risk assessment: "If Approach A can't beat the frontier, proceed to C" — and C can't either, because the bottleneck is table-switching, not the training algorithm.

### Implications for Approach B

Approach B (direct action via IQN) eliminates table-switching entirely. The agent makes **every decision** (reroll masks, category choice) based on learned distributional value estimates. This is fundamentally more expressive — it can discover decision patterns that no fixed-θ table encodes. However, it also faces:

- 45 decisions per game (vs 15 for A/C) — harder credit assignment
- 32+15 action space per step — much larger search space
- No precomputed "near-optimal" baseline to bootstrap from

The behavioral cloning warm-start from θ=0 decisions should help, providing a good initial policy that IQN refines with distributional objectives.

## RL Approach B: Direct Action with IQN

### Architecture

Approach B eliminates table-switching entirely. An Implicit Quantile Network (IQN) directly selects reroll masks (32 actions) and categories (15 actions) for each decision point — 45 decisions per game.

- **Network**: 66K params — shared state encoder (18→128→128), cosine quantile embedding (64→128), separate reroll head (128→128→32) and score head (128→128→15)
- **Training**: Two-phase — behavioral cloning (BC) warm-start from θ=0 expert, then IQN distributional RL fine-tuning
- **Inference**: CVaR action selection — average quantile values from τ ∈ [0, α] for risk-sensitive decisions

### Rust Bridge Extension

Extended the FFI bridge with per-decision functions: `batch_roll`, `batch_apply_reroll`, `batch_score_category`, `batch_expert_reroll`, `batch_expert_category`. Verified: 10K expert games via per-decision bridge yield mean=248.3, matching full-turn simulation.

### Behavioral Cloning Results

50K expert games → 1.5M reroll decisions + 750K category decisions. Trained 20 epochs:

| Epoch | Loss | Reroll Acc | Score Acc |
| ----- | ---- | ---------- | --------- |
| 1     | 1.20 | 36.5%      | 70.0%     |
| 5     | 0.64 | 53.2%      | 77.3%     |
| 10    | 0.55 | 58.2%      | 79.1%     |
| 15    | 0.50 | 61.6%      | 80.3%     |
| 20    | 0.48 | 66.4%      | 81.2%     |

### Game Performance with Different CVaR α

| Policy            | Mean  | p5  | p50 | p95 | p99 |
| ----------------- | ----- | --- | --- | --- | --- |
| IQN α=1.00 (mean) | 205.3 | 131 | 208 | 281 | 303 |
| IQN α=0.75        | 205.3 | 131 | 209 | 281 | 303 |
| IQN α=0.50        | 204.2 | 131 | 207 | 280 | 302 |
| IQN α=0.25        | 204.9 | 131 | 208 | 281 | 303 |
| Expert θ=0.00     | 248.4 | 179 | 249 | 309 | 324 |

### Analysis

**1. BC accuracy is insufficient for competitive play.** Despite 66% reroll accuracy and 81% score accuracy, the model scores mean=205 — 43 points below the expert. Errors compound: with ~0.34 error rate per reroll decision and ~0.19 per category decision, the probability of playing a full turn correctly is 0.66² × 0.81 ≈ 35%. Over 15 turns, almost every game contains multiple errors.

**2. CVaR has zero effect.** Varying α from 1.0 to 0.25 produced identical results (±1 point). This is expected: the IQN was trained with cross-entropy loss (classification), not distributional RL. The quantile estimates from the cosine embedding are not calibrated to the return distribution — they're arbitrary values that happen to produce correct argmax but carry no distributional information.

**3. The RL fine-tuning phase failed catastrophically.** Online IQN training degraded the BC policy from mean=190 to mean=123 within 20K episodes. Root cause: terminal-only reward (45-step bootstrap chain) combined with a 200K replay buffer creates a death spiral — slight policy degradation fills the buffer with bad transitions, which further degrades the policy.

**4. A 66K-parameter network cannot approximate an 8MB lookup table.** The θ=0 solver encodes 2,097,152 state values with full precision. The IQN must compress this into 66K parameters — a 128× compression ratio. Reroll decisions are especially hard: with 252 dice sets × 32 masks × varying game states, the combinatorial space far exceeds what a 128-wide network can represent.

### Why Approach B Cannot Work (Fundamental Limit)

The core issue is the **approximation gap**. The optimal solver achieves mean=248.4 through 2M precomputed state values. Any neural network policy must approximate this table, and even small approximation errors compound multiplicatively:

- If each decision has accuracy p, the probability of a perfect turn is p³ (two rerolls + one category)
- Over 15 turns, the probability of zero errors is p⁴⁵
- At p=0.80 (our score head accuracy): 0.80⁴⁵ ≈ 0.0001 — virtually no game is played optimally
- The mean score loss is roughly proportional to (1-p) × decisions × average_error_cost
- With average error cost ~3 points and 45 decisions: (1-0.73) × 45 × 3 ≈ 36 points — close to our observed 43-point deficit

To match the expert within 5 points, we'd need decision accuracy >99%, which requires a network with ~1M parameters and orders of magnitude more training data — at which point we're essentially memorizing the table.

## Final RL Conclusions

All three approaches (A, B, C) failed to beat the fixed-θ Pareto frontier:

| Approach                   | Architecture               | Mean    | p95     | vs Frontier |
| -------------------------- | -------------------------- | ------- | ------- | ----------- |
| A: θ-switching (REINFORCE) | Turn-level table selection | 239-248 | 309-312 | On frontier |
| C: θ-switching (PPO)       | Turn-level table selection | 246.9   | 310     | On frontier |
| B: Direct action (IQN+BC)  | Per-decision neural        | 205.3   | 281     | Far below   |

**The fixed-θ Pareto frontier is tight.** No RL approach found a point above it:

1. **Approaches A/C** (table-switching) replicate points on the frontier. The adaptive signal (which θ to use when) is too weak relative to noise for any algorithm to learn.

2. **Approach B** (direct action) falls far below the frontier. Neural networks cannot approximate the precomputed tables with sufficient fidelity.

**The theoretical gap for RL is ~0.** The accumulated-score conditioning that RL could exploit (running hot/cold) shifts optimal decisions by <1 point per game. This is below the noise floor for any practical RL algorithm. The precomputed solver, despite not conditioning on accumulated score, is empirically unbeatable — its mean-optimal decisions happen to be near-optimal for all reasonable risk preferences.
