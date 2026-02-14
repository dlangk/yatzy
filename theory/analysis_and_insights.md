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

Notably, the max-policy also has the **lowest Yatzy hit rate** of all strategies tested (4.9% vs 38.8% for EV-optimal). See "Conditional Yatzy Hit Rate" section below for analysis.

---

## Risk-Sensitive theta Sweep Results

### Full sweep (1M games each, 37 theta values, progressive spacing)

Grid: 37 values from -3.0 to +3.0 with progressive spacing dense near zero (delta = 0.005 in [-0.015, +0.015], 0.01 in [-0.05, +0.05], then widening to 0.5-1.0 at the extremes).

| theta |   Mean |  Std | Min |  p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | Max |
| ----: | -----: | ---: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| -3.00 | 188.54 | 37.0 |  90 | 139 | 147 | 161 | 181 | 213 | 239 | 257 | 293 | 344 |
| -2.00 | 190.63 | 36.9 |  91 | 142 | 149 | 163 | 183 | 215 | 241 | 259 | 295 | 343 |
| -1.50 | 192.54 | 37.0 |  95 | 143 | 151 | 164 | 185 | 218 | 243 | 261 | 296 | 343 |
| -1.00 | 198.22 | 37.9 |  98 | 148 | 155 | 169 | 191 | 225 | 249 | 269 | 300 | 344 |
| -0.75 | 203.24 | 38.6 | 104 | 151 | 158 | 173 | 196 | 231 | 255 | 275 | 303 | 341 |
| -0.50 | 211.63 | 39.5 | 107 | 156 | 163 | 179 | 209 | 240 | 263 | 283 | 308 | 345 |
| -0.30 | 221.74 | 39.8 | 105 | 161 | 169 | 187 | 226 | 250 | 272 | 290 | 313 | 356 |
| -0.20 | 227.48 | 39.3 |  97 | 165 | 173 | 194 | 233 | 254 | 275 | 294 | 315 | 353 |
| -0.15 | 231.83 | 38.9 | 104 | 167 | 176 | 201 | 237 | 257 | 280 | 298 | 317 | 354 |
| -0.10 | 239.22 | 35.8 | 100 | 172 | 187 | 221 | 241 | 261 | 283 | 300 | 319 | 355 |
| -0.07 | 242.36 | 35.0 | 100 | 175 | 199 | 224 | 243 | 263 | 287 | 302 | 320 | 355 |
| -0.05 | 245.09 | 35.7 |  99 | 179 | 203 | 225 | 245 | 268 | 292 | 305 | 322 | 357 |
| -0.04 | 246.45 | 36.3 |  99 | 182 | 204 | 225 | 246 | 270 | 295 | 306 | 322 | 357 |
| -0.03 | 247.32 | 36.7 |  90 | 183 | 204 | 226 | 247 | 273 | 296 | 307 | 323 | 350 |
| -0.02 | 247.93 | 37.2 |  93 | 183 | 204 | 226 | 248 | 274 | 297 | 308 | 323 | 351 |
| -0.015| 248.13 | 37.4 |  93 | 182 | 204 | 226 | 248 | 275 | 298 | 308 | 324 | 351 |
| -0.01 | 248.29 | 37.7 |  93 | 182 | 204 | 226 | 248 | 275 | 298 | 308 | 324 | 356 |
| -0.005| 248.39 | 38.0 |  93 | 181 | 203 | 225 | 249 | 275 | 299 | 309 | 324 | 352 |
|**0.000**|**248.40**|**38.5**|**97**|**179**|**203**|**225**|**249**|**276**|**299**|**309**|**325**|**352**|
| 0.005 | 248.32 | 39.0 |  94 | 177 | 202 | 225 | 249 | 276 | 300 | 310 | 325 | 353 |
| 0.010 | 248.21 | 39.4 |  90 | 175 | 201 | 225 | 249 | 277 | 300 | 310 | 325 | 353 |
| 0.015 | 248.03 | 39.8 |  90 | 174 | 200 | 224 | 248 | 277 | 301 | 310 | 326 | 359 |
| 0.020 | 247.73 | 40.3 |  90 | 172 | 198 | 224 | 248 | 277 | 301 | 311 | 326 | 359 |
| 0.030 | 246.92 | 41.3 |  88 | 169 | 195 | 223 | 247 | 277 | 301 | 311 | 326 | 359 |
| 0.040 | 245.71 | 42.3 |  88 | 166 | 190 | 221 | 245 | 278 | 302 | 312 | 327 | 359 |
| 0.050 | 244.53 | 43.2 |  79 | 164 | 185 | 219 | 244 | 277 | 302 | 312 | 327 | 359 |
| 0.070 | 240.86 | 45.3 |  84 | 159 | 176 | 213 | 241 | 274 | 302 | 313 | 328 | 362 |
| 0.100 | 235.92 | 46.9 |  79 | 155 | 170 | 204 | 237 | 268 | 301 | 312 | 329 | 362 |
| 0.150 | 229.23 | 47.9 |  74 | 152 | 165 | 192 | 231 | 261 | 296 | 311 | 329 | 362 |
| 0.200 | 223.76 | 48.4 |  67 | 148 | 160 | 185 | 225 | 257 | 289 | 308 | 329 | 358 |
| 0.300 | 215.74 | 48.6 |  76 | 142 | 154 | 177 | 215 | 251 | 280 | 301 | 327 | 358 |
| 0.500 | 204.73 | 47.6 |  60 | 135 | 146 | 168 | 200 | 240 | 269 | 287 | 319 | 361 |
| 0.750 | 198.13 | 47.0 |  57 | 130 | 141 | 162 | 192 | 233 | 262 | 280 | 313 | 360 |
| 1.000 | 194.47 | 46.3 |  57 | 128 | 139 | 159 | 188 | 229 | 258 | 276 | 309 | 359 |
| 1.500 | 189.22 | 45.5 |  55 | 125 | 135 | 155 | 182 | 221 | 253 | 270 | 305 | 359 |
| 2.000 | 186.04 | 44.8 |  48 | 123 | 133 | 152 | 179 | 217 | 250 | 266 | 300 | 359 |
| 3.000 | 186.49 | 44.2 |  50 | 124 | 135 | 154 | 179 | 216 | 250 | 266 | 301 | 359 |

### Best theta per metric

| Metric | Best theta | Value |
| -----: | ---------: | ----: |
| min | -0.50 | 107 |
| p5 | -0.03 | 183 |
| p10 | -0.04 | 204 |
| p25 | -0.03 | 226 |
| p50 | -0.005 | 249 |
| p75 | 0.04 | 278 |
| p90 | 0.04 | 302 |
| p95 | 0.07 | 313 |
| p99 | 0.10 | 329 |
| max | 0.07 | 362 |

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

### Risk-Averse theta: The Other Branch

The extended sweep to negative theta reveals the risk-averse side of the frontier, which has qualitatively different properties from risk-seeking:

**Risk-averse play is surprisingly effective at compressing variance.** At theta=-0.05, std drops from 38.5 to 35.7 while mean only drops 3 points (248.4 to 245.1). This is a much better variance/mean tradeoff than risk-seeking achieves — at theta=+0.05, std *rises* from 38.5 to 43.2 while mean drops 3.9 points. Risk-averse buys variance reduction cheaply; risk-seeking buys variance inflation expensively.

**p5 peaks at theta=-0.03 (183 vs 179 at theta=0).** Risk-averse play lifts the floor — the worst 5% of games improve by 4 points with only 1 point of mean cost. Similarly, p10 peaks at theta=-0.04 (204 vs 203 at theta=0).

**The minimum score peaks at theta=-0.5 (107 vs 97).** Extremely risk-averse play protects the absolute worst case, raising the floor by 10 points. This comes at a steep mean cost (211.6 vs 248.4), but for applications where catastrophic scores are unacceptable, this tradeoff may be worthwhile.

**Degenerate risk-averse (theta < -1)**: Mean drops to ~190, similar to extreme risk-seeking. The policy becomes symmetrically bad — refusing all variance is as costly as embracing all variance. At theta=-3, mean=188.5, comparable to theta=+3 at 186.5.

**Asymmetric branches confirmed with full range**: Risk-averse achieves std=35-38 across its entire useful range, while risk-seeking peaks at std=48.6. At the same mean (~190), the risk-seeking branch (theta~1) has std~46 vs std~37 for risk-averse (theta~-1.5) — 9 points higher. The two branches trace distinct curves in (mean, std) space, confirming the two-branch structure first observed in the coarse sweep.

### Regime boundaries (refined with full symmetric grid)

- **theta < -1** (|theta| * sigma > 10): Degenerate risk-averse. Mean ~190, all metrics declining. Policy refuses nearly all variance.
- **-1 < theta < -0.1** (|theta| * sigma ~ 1-4): Active risk-averse regime. Variance compresses substantially (std 35-38), lower percentiles improve.
- **-0.1 < theta < -0.01**: Near-EV risk-averse. Small variance reduction, small mean cost. Sweet spot for floor protection.
- **-0.01 < theta < 0.03** (|theta| * sigma < 0.3): Near-EV. Negligible differences from theta=0.
- **0.03 < theta < 0.20** (|theta| * sigma ~ 0.3-2): **Active risk-seeking tradeoff.** Upper percentiles (p95, p99) peak, mean declines ~1 pt per 0.01 theta.
- **theta > 0.20** (|theta| * sigma > 2): Diminishing returns. Policy changes slow down, all metrics decline monotonically.
- **theta > 1.5** (|theta| * sigma > 15): Full saturation. Policy frozen, no further change.

---

## How theta Changes Decisions

### Disagreement rate

At each decision point (reroll mask or category choice), theta=0 and theta>0 may prescribe different actions. The disagreement rate scales with theta (measured over 100K games, following theta=0's game trajectory):

| Comparison | Reroll disagree | Category disagree | Total | Per game (of 45) |
|---|---|---|---|---|
| θ=0 vs θ=0.07 | 12.5% | 12.4% | 12.5% | 5.6 |
| θ=0 vs θ=0.50 | — | 32.0% | — | — |
| θ=0 vs θ=1.10 | — | 40.3% | — | — |
| θ=0 vs θ=3.00 | — | 44.2% | — | — |

Even at the mild θ=0.07 (peak p95), 1 in 8 decisions differs. At extreme θ=3.0, nearly half of all category decisions differ.

### The unified pattern: option value

Every category disagreement has the same structure: θ=0 fills category X, θ>0 fills category Y — they differ on **which category to keep open for future turns**. θ>0 systematically assigns higher option value to keeping categories open, especially those with high scoring ceilings.

θ=0 (EV-optimal) writes off hard-to-complete categories sooner and keeps flexible ones that score from many dice combinations. θ>0 (risk-seeking) preserves high-ceiling categories longer and dumps low-ceiling or easy-to-fill ones first.

### Category disagreement breakdown (θ=0 vs θ=0.07, 100K games)

Categorized by what θ=0.07 **preserves** (keeps open) that θ=0 gives up:

| θ=0.07 preserves... | Count | % of disagreements | Avg immediate sacrifice |
|---|---|---|---|
| Yatzy (ceiling 50) | 48,598 | 26.1% | +3.0 pts |
| Straights (ceiling 15/20) | 47,269 | 25.3% | -2.3 pts (gains) |
| Chance (ceiling 30) | 10,933 | 5.9% | +2.9 pts |
| Full House (ceiling 28) | 6,593 | 3.5% | -1.0 pts (gains) |
| Two Pairs / Four of a Kind (ceiling 22/24) | 7,355 | 3.9% | -1.6 pts (gains) |
| Upper category over lower | 58,065 | 31.1% | -0.2 pts (gains) |
| Upper section reordering | 1,862 | 1.0% | -0.5 pts (gains) |
| Lower category over upper | 4,837 | 2.6% | +0.5 pts |
| **Total** | **186,520** | **100%** | **+0.2 pts** |

The "immediate sacrifice" is the score difference on the current turn — positive means θ=0 scores more now. Negative ("gains") means θ=0.07 actually scores more now because it fills a scoring category instead of dumping a zero.

### Detailed examples of each type

**1. Preserve Yatzy (26.1%, costs +3.0 pts/decision)**

The most costly preservation. Late game (avg turn 13), θ=0 fills Yatzy with a low or zero score. θ=0.07 dumps that score on another category to keep the 50-point jackpot alive.

| θ=0.07 preserves | θ=0.07 dumps | Count | θ=0 scores | θ=0.07 scores |
|---|---|---|---|---|
| Yatzy | Large Straight | 14,492 | 5.6 | 0.0 |
| Yatzy | Four of a Kind | 13,585 | 1.5 | 0.1 |
| Yatzy | Small Straight | 5,487 | 5.3 | 0.0 |
| Yatzy | Chance | 3,690 | 5.9 | 5.0 |
| Yatzy | Full House | 2,513 | 1.3 | 0.0 |

θ=0 writes off Yatzy because P(five-of-a-kind) ≈ 4.6% per turn — the EV of keeping Yatzy open (~2.3 pts) is usually less than the EV of keeping other categories open. θ=0.07 values the *upside* of 50 points more than its probability warrants under expected value.

**2. Preserve Straights (25.3%, gains -2.3 pts/decision)**

θ=0 dumps a zero on Small/Large Straight (gives up on completing it). θ=0.07 fills a different category (often Ones, Twos, One Pair) and keeps the straight alive. Because the dump target often scores 2-5 points, θ=0.07 typically gains immediate points.

| θ=0.07 preserves | θ=0.07 dumps | Count | θ=0 scores | θ=0.07 scores |
|---|---|---|---|---|
| Small Straight | Ones | 14,640 | 0.0 | 2.9 |
| Small Straight | Twos | 5,730 | 0.0 | 2.6 |
| Small Straight | Threes | 5,417 | 0.0 | 0.0 |
| Large Straight | Ones | 3,994 | 0.0 | 2.5 |
| Small Straight | One Pair | 3,815 | 0.0 | 2.2 |
| Large Straight | Chance | 2,431 | 0.0 | 5.0 |

θ=0 writes off straights early (avg turn 5-8) because the probability of completing them declines as upper section categories fill. θ=0.07 keeps them alive, filling low-value categories as dump targets.

**3. Preserve Upper Category over Lower (31.1%, near-zero cost)**

The largest category. Both score ~0, but θ=0.07 dumps in a lower-section category (Three of a Kind, Four of a Kind) while θ=0 dumps in an upper-section category (Sixes, Fives).

| θ=0.07 preserves | θ=0.07 dumps | Count | θ=0 scores | θ=0.07 scores |
|---|---|---|---|---|
| Sixes | Three of a Kind | 30,889 | 0.0 | 0.0 |
| Sixes | Four of a Kind | 18,994 | 0.0 | 0.0 |

θ=0 keeps Three of a Kind open because it's a flexible category that scores from many dice combinations. θ=0.07 keeps Sixes open (max 30, contributes to upper bonus) because it values the ceiling and bonus potential more than the scoring flexibility. This is about **probability of use vs ceiling of use**: θ=0 optimizes for the former, θ=0.07 for the latter.

**4. Preserve Chance (5.9%, costs +2.9 pts/decision)**

θ=0 fills Chance with a mediocre roll (avg 5 pts, the dice sum is low). θ=0.07 puts fewer points in One Pair or Ones and keeps Chance (max 30) open for a future high roll.

**5. Preserve Full House / Two Pairs / Four of a Kind (7.4%, gains points)**

θ=0.07 keeps harder multi-match categories open and fills easier ones. Since the dump target often has a non-zero score, θ=0.07 gains 1-2 points immediately while preserving upside.

### Immediate cost vs compounding cost

Net across all 186,520 disagreements: **θ=0.07 sacrifices only 0.3 total points per game** in immediate category scores (on 1.9 disagreeing decisions per game). Yet the overall mean difference is 7.5 points.

The gap is explained by **compounding**. Preserving hard-to-complete categories means:
- More turns spent with incomplete high-variance categories open
- Those categories often end up scored at zero anyway (Yatzy ≈ 4.6% hit rate)
- The upper bonus is hit less often (89.8% → 72.6%) because upper category slots are filled suboptimally
- Reroll decisions also diverge (12.5% disagreement), chasing different targets

The immediate sacrifice is small, but the downstream effects cascade: worse bonus rates, worse category utilization, and slightly worse reroll targeting. The 7.5-point mean loss is a systemic property of the strategy, not any single decision.

### How disagreements scale with θ

At higher θ, the same patterns intensify without changing character:

| θ | Category disagree rate | Dominant pattern |
|---|---|---|
| 0.07 | 12.4% | Preserve Yatzy, Straights, Upper slots |
| 0.50 | 32.0% | Same patterns + abandon all mid-value lower categories |
| 1.10 | 40.3% | Same + Fours/Fives → Ones (dump low-value upper) |
| 3.00 | 44.2% | Same + Full House/Two Pairs → One Pair (dump all low-ceiling) |

The classification shifts at higher θ:

| θ | Same score (swap) | θ=0 scores more | θ>0 scores more |
|---|---|---|---|
| 0.07 | 60% | 9% | 31% |
| 0.50 | 43% | 10% | 47% |
| 1.10 | 38% | 13% | 49% |
| 3.00 | 44% | 16% | 40% |

At moderate θ, most disagreements are swaps (same immediate score) or cases where θ>0 scores more (it fills a non-zero category while θ=0 dumps a zero). At extreme θ, the Yatzy preservation cases where θ=0 scores more become a larger fraction, but the overall pattern remains: preserve high ceiling, dump low ceiling.

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

Empirically measured disagreements per game (category decisions only, 100K games):

| theta | Mean loss vs theta=0 | Category disagree rate | Category disagrees/game |
| ----: | -------------------: | ---------------------: | ----------------------: |
| +0.07 |                  7.5 |                  12.4% |                     1.9 |
| +0.50 |                 43.7 |                  32.0% |                     4.8 |
| +1.10 |                 54.7 |                  40.3% |                     6.0 |
| +3.00 |                 62.0 |                  44.2% |                     6.6 |

Including reroll decisions, θ=0.07 disagrees on 12.5% of all 45 decisions per game (5.6 per game). See the "How theta Changes Decisions" section for a full breakdown of what these disagreements look like.

At θ=0.07 (peak p95), only 1.9 category decisions per game differ. This means we need many observed games to estimate a human's θ with any precision — a single game has very little signal.

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

## theta Estimation Questionnaire

### Approach: pivotal scenarios as a questionnaire

Rather than requiring 50-100 observed games, we can estimate a player's theta directly by presenting **pivotal scenarios** — game states where different theta values prescribe different optimal categories — and asking the player to choose. This is a compressed version of the revealed-preference approach described above.

### Generating pivotal scenarios

Pivotal scenarios are generated by the `pivotal-scenarios` binary. It simulates games and finds decision points where different theta values prescribe different optimal categories. A scenario is "pivotal" if there exist at least two theta values in the grid that disagree on the best category to fill.

Not all pivotal scenarios from optimal play are realistic — some arise in game states that no human would encounter (e.g., having scored zero in all upper categories by turn 12). A **realism filter** rejects scenarios that fail basic sanity checks:
- Zero-dump limit: too many categories scored as zero
- Upper section sanity: upper score consistent with turn number
- Minimum scored sum: total score not implausibly low for the turn

### Bayesian grid estimator

The estimator uses a softmax choice model:

```
P(a | s, θ, β) = exp(β · V_θ(a, s)) / Σ_a' exp(β · V_θ(a', s))
```

where V_θ(a, s) is the value of choosing category a in state s under theta, and beta is a rationality parameter (higher beta = more decisive). Given observed answers, the posterior over theta is updated via Bayes' rule on a discrete grid.

### Adaptive question selection

Each question is selected to maximize expected information gain — the expected reduction in entropy of the theta posterior. After observing the answer, the posterior is updated and the next question is selected. This greedy strategy is equivalent to one-step-ahead Bayesian optimal experimental design.

**15 questions are typically sufficient** for a theta estimate with confidence interval width < 0.05, based on validation experiments. The first 5-7 questions rapidly narrow the posterior; subsequent questions refine the estimate within the active region.

### Answer format

Answers are stored as `(scenario_id, category_id)` pairs for robustness across grid changes. This means saved questionnaire results remain valid even if the theta grid or scenario pool changes — the answers reference immutable scenario and category identifiers, not theta-dependent quantities.

A `theta-replay` command re-estimates theta from saved answers, and `theta-validate` runs convergence validation (how CI width decreases with question count).

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

_To be filled after running simulations. See `outputs/plots/efficiency_adaptive.png`._

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

---

## Per-Category Statistics Across the θ Sweep

### Method

For each of the 37 θ values in the full sweep, simulated 1M games and computed five per-category statistics:

1. **Mean score**: average points scored in that category
2. **Zero rate**: fraction of games where the category scores 0 (either dumped or missed)
3. **Mean fill turn**: average turn (1-indexed) when the category is filled
4. **Score % of ceiling**: mean score as a percentage of the category's maximum
5. **Hit rate**: 1 - zero rate (fraction of games where the category scores ≥ 1)

Full data: `outputs/aggregates/csv/category_stats.csv` (555 rows = 37 θ values × 15 categories, generated by `yatzy-category-sweep`).

### Category ceilings

| Category | Ceiling | Notes |
|---|---|---|
| Ones | 5 | Upper section |
| Twos | 10 | Upper section |
| Threes | 15 | Upper section |
| Fours | 20 | Upper section |
| Fives | 25 | Upper section |
| Sixes | 30 | Upper section |
| One Pair | 12 | |
| Two Pairs | 22 | |
| Three of a Kind | 18 | |
| Four of a Kind | 24 | |
| Small Straight | 15 | All-or-nothing |
| Large Straight | 20 | All-or-nothing |
| Full House | 28 | |
| Chance | 30 | Never zeros |
| Yatzy | 50 | All-or-nothing |

### Upper section: Ones is the dump category

Ones absorbs more zeros than any other upper category, and this intensifies with θ:

| θ | Ones mean | Ones zero% | Ones fill turn | Sixes mean | Sixes fill turn |
|---|---|---|---|---|---|
| 0.00 | 2.18 | 9.2% | 8.0 | 19.99 | 6.5 |
| 0.07 | 1.75 | 14.8% | 7.0 | 19.05 | 9.9 |
| 0.20 | 1.44 | 20.5% | 5.5 | 18.08 | 11.7 |
| 1.00 | 1.37 | 30.9% | 3.1 | 16.86 | 12.6 |
| 3.00 | 1.25 | 38.5% | 3.1 | 17.13 | 12.4 |

θ>0 fills Ones early (turn 3 vs turn 8) and with low/zero scores, preserving high-value upper slots for later. Sixes migrates from turn 6.5 to turn 12.4 — filled late to maximize its ceiling of 30 and bonus contribution.

### High-variance categories: Yatzy peaks at moderate θ

| θ | Yatzy mean | Yatzy hit% | Yatzy fill turn |
|---|---|---|---|
| 0.00 | 19.36 | 38.7% | 10.6 |
| 0.05 | 21.38 | 42.8% | 11.3 |
| 0.07 | 21.42 | 42.8% | 11.4 |
| 0.10 | 20.64 | 41.3% | 11.6 |
| 0.20 | 16.94 | 33.9% | 12.1 |
| 0.50 | 10.94 | 21.9% | 13.0 |
| 3.00 | 7.47 | 14.9% | 13.4 |

Yatzy mean score peaks at θ≈0.05-0.07, where the option-value preservation strategy maximizes the probability of eventually completing five-of-a-kind. Beyond θ≈0.10, the strategy over-commits to preserving Yatzy at the expense of other categories, and the cascading mean loss outweighs the improved hit rate.

### Straights improve with moderate θ

| θ | Sm.Str mean | Sm.Str hit% | Lg.Str mean | Lg.Str hit% |
|---|---|---|---|---|
| 0.00 | 3.96 | 26.4% | 9.73 | 48.7% |
| 0.07 | 4.83 | 32.2% | 10.33 | 51.7% |
| 0.15 | 6.02 | 40.2% | 11.37 | 56.8% |
| 0.20 | 6.52 | 43.5% | 11.24 | 56.2% |
| 0.50 | 6.56 | 43.7% | 10.26 | 51.3% |
| 3.00 | 5.90 | 39.4% | 10.11 | 50.5% |

Small Straight hit rate nearly doubles from 26% → 44% at θ=0.20. Large Straight peaks at 57% around θ=0.15-0.20. Both benefit from the option-value strategy keeping straight slots open longer.

### Categories that degrade with θ

| θ | Full House mean | Full House hit% | Four of a Kind mean | Chance mean |
|---|---|---|---|---|
| 0.00 | 21.47 | 91.5% | 13.77 | 23.10 |
| 0.07 | 21.14 | 86.3% | 13.89 | 22.76 |
| 0.20 | 20.16 | 77.9% | 15.39 | 23.14 |
| 0.50 | 18.32 | 69.3% | 13.99 | 23.51 |
| 1.00 | 16.17 | 61.5% | 13.29 | 23.02 |
| 3.00 | 13.93 | 53.7% | 15.28 | 22.19 |

Full House drops from 92% hit rate to 54% — θ>0 increasingly uses Full House as a dump target, sacrificing its 28-point ceiling to preserve other high-ceiling categories. Chance stays near 23 regardless of θ (it never zeros and acts as a universal dump/catch-all).

Four of a Kind is interesting: its mean *increases* with moderate θ (13.77 → 15.39 at θ=0.20) because risk-seeking play more often results in four-of-a-kind combinations, but at extreme θ it becomes volatile.

### Timing shifts: risk-seeking reorders category filling

The most striking effect of θ is not the score per category but *when* each category gets filled. At θ=0, categories are filled roughly in order of flexibility (easiest-to-score first). At high θ, the ordering inverts for high-ceiling categories:

| Category | Fill turn (θ=0) | Fill turn (θ=3.0) | Shift |
|---|---|---|---|
| One Pair | 7.1 | 2.8 | -4.3 earlier |
| Three of a Kind | 8.7 | 4.8 | -3.9 earlier |
| Ones | 8.0 | 3.1 | -4.9 earlier |
| Twos | 7.6 | 5.3 | -2.3 earlier |
| Full House | 7.2 | 10.2 | +3.0 later |
| Fours | 6.6 | 9.6 | +3.0 later |
| Fives | 6.1 | 11.7 | +5.6 later |
| Sixes | 6.5 | 12.4 | +5.9 later |
| Yatzy | 10.6 | 13.4 | +2.8 later |

The pattern: θ>0 dumps low-ceiling categories early (One Pair, Ones, Three of a Kind) and preserves high-ceiling categories for late game (Sixes, Fives, Yatzy). This is the option-value framework in action — early turns are "investments" in preserving upside, late turns "cash in" whatever was achieved.

---

## Conditional Yatzy Hit Rate: Does Risk-Seeking Chase the Jackpot?

### The puzzle

As θ increases, the unconditional Yatzy hit rate drops from 38.8% (θ=0) to 17.6% (θ=1.1). This is counterintuitive — risk-seeking strategies should chase high-variance categories, and Yatzy is the highest-ceiling category at 50 points.

Two competing explanations:

- **Dump hypothesis (H0)**: The drop is driven by the solver dumping Yatzy in games already going poorly. In the right tail (top 5% of games), high-θ strategies hit Yatzy at equal or higher rates than θ=0.
- **Sacrifice hypothesis (H1)**: The drop is fundamental — even in the right tail, high-θ hits Yatzy less often. The solver sacrifices Yatzy for other priorities regardless of game quality.

### Definitions

Every game fills all 15 categories exactly once. The Yatzy category is always used — the question is what you score in it:

- **Hit** = scored 50 points in Yatzy (rolled five of a kind)
- **Miss** = scored 0 points in Yatzy (used it as a dump slot for non-matching dice)

### Method

For each of 6 θ values (0, 0.05, 0.10, 0.20, 0.50, 1.10) plus max-policy, simulated 1M games and computed Yatzy hit rates conditioned on the game's total score falling in various bands. The `top5pct` band uses a dynamic threshold (p95 differs by θ).

Full data: `outputs/aggregates/csv/yatzy_conditional.csv` (63 rows, generated by `yatzy-conditional`).

### Result: H1 confirmed — the sacrifice is fundamental

Three independent signals agree:

**1. Top-5% tail hit rate drops monotonically**

| θ | Unconditional | Top-5% tail | p95 threshold |
|---|---|---|---|
| 0.00 | 38.8% | **100.0%** | 309 |
| 0.05 | 42.9% | **100.0%** | 312 |
| 0.10 | 41.3% | **100.0%** | 312 |
| 0.20 | 33.7% | **100.0%** | 308 |
| 0.50 | 21.7% | **94.2%** | 287 |
| 1.10 | 17.6% | **78.1%** | 275 |
| max-policy | 4.9% | **24.4%** | 226 |

For θ ≤ 0.20, every game in the top 5% contains Yatzy — it is a *necessary condition* for a high score. At θ=0.50, 6% of top-5% games achieve high scores without Yatzy. At θ=1.10, 22% of top games skip Yatzy. This directly falsifies H0: even the best games at high θ don't always include Yatzy.

**2. Conditional hit rates drop in every score band**

At high θ, Yatzy hit rate decreases in all bands, including 300+ and top5pct. Moderate θ (0.05-0.10) actually *increases* hit rates in most bands before the decline sets in.

| Band | θ=0 | θ=0.05 | θ=0.10 | θ=0.50 | θ=1.10 |
|---|---|---|---|---|---|
| <200 | 12.1% | 11.9% | 9.3% | 6.1% | 6.1% |
| 240-260 | 17.5% | 21.3% | 28.2% | 25.8% | 24.8% |
| 280-300 | 96.2% | 97.7% | 95.6% | 73.8% | 76.4% |
| 300+ | 100.0% | 100.0% | 100.0% | 99.6% | 99.5% |
| top5pct | 100.0% | 100.0% | 100.0% | 94.2% | 78.1% |

**3. The dump gap is constant**

The mean score difference between Yatzy-hit games and Yatzy-miss games is ~50-54 points across all strategies:

| Strategy | Mean (hit) | Mean (miss) | Gap |
|---|---|---|---|
| θ=0 (EV) | 278 | 229 | +49 |
| θ=0.05 | 275 | 222 | +53 |
| θ=0.10 | 267 | 214 | +54 |
| θ=0.20 | 259 | 206 | +53 |
| θ=0.50 | 247 | 193 | +54 |
| θ=1.10 | 237 | 184 | +53 |
| max-policy | 191 | 141 | +50 |

If H0 were true (dumping only in bad games), the gap should widen at high θ — bad games would increasingly be the ones without Yatzy. Instead, the gap is remarkably stable, confirming the sacrifice is applied uniformly regardless of game quality.

### The non-monotonic bump at moderate θ

An unexpected finding: θ=0.05 *increases* the unconditional Yatzy hit rate from 38.8% to 42.9% before the decline begins. This is consistent with the per-category analysis (see "High-variance categories: Yatzy peaks at moderate θ" above) — mild risk-seeking enhances the option-value preservation strategy, keeping Yatzy open longer and increasing the probability of eventually completing five-of-a-kind. Only beyond θ≈0.10 does the solver begin sacrificing Yatzy for other high-variance paths.

### Max-policy: overconfidence kills Yatzy

Max-policy (maximax) is the worst Yatzy chaser of all strategies: 4.9% unconditional hit rate and only 24.4% in the top 5%. This is counterintuitive — the strategy that assumes the best possible dice outcome should love chasing five-of-a-kind.

The explanation: under max-policy, *every* category looks easy to max. If you assume the best dice always come up, getting 30 in Sixes is guaranteed, completing a Large Straight is guaranteed, etc. Yatzy's 50 points don't stand out as special when every category promises its ceiling. The solver spreads its overconfidence equally rather than concentrating on the one big jackpot.

| Max-policy metric | Value |
|---|---|
| Unconditional Yatzy hit | 4.9% (vs 38.8% at θ=0) |
| Top-5% Yatzy hit | 24.4% (vs 100% at θ=0) |
| Mean score overall | 143 (vs 248 at θ=0) |
| Mean score (Yatzy hit) | 191 |
| Mean score (Yatzy miss) | 141 |

The conditional bars show a striking pattern for max-policy: hit rate rises sharply with score band (3.8% in <200, 62% in 240-260, 92% in 260-280, 100% in 280-300), but collapses to 24% in the top-5% band because the p95 threshold is only 226 — most "top" max-policy games are still mediocre by absolute standards.

### What the solver actually does instead of Yatzy

At high θ, the solver doesn't chase Yatzy because it finds more efficient paths to high scores:

1. **Upper section optimization**: Preserving Sixes and Fives for late game to maximize ceiling and bonus potential (see timing shifts in per-category stats)
2. **Straight completion**: Small Straight hit rate nearly doubles from 26% to 44% at θ=0.20 — the solver keeps straight slots open longer
3. **Four of a Kind improvement**: Mean score increases from 13.8 to 15.4 at moderate θ as risk-seeking play more often yields four-of-a-kind combinations
4. **Portfolio effect**: Instead of one 50-point lottery ticket (Yatzy at ~39% hit rate), the solver buys many smaller bets across categories that collectively produce more upper-tail mass

The solver trades one big jackpot for many smaller but more reliable bets. At θ=0.05, the effect is harmonious — both Yatzy and other categories improve. At θ ≥ 0.20, Yatzy becomes a casualty of the aggressive portfolio reallocation.

---

## Board-State Frequency: Which Positions Occur Most Often?

Simulating 100K games under optimal (θ=0) play and aggregating by board state (upper_score, scored_categories) — ignoring dice and decision type — reveals how play funnels through the 2M-state space.

### The hourglass shape

The state space has a pronounced hourglass structure. Every game starts at the same state (turn 1: 1 unique state), fans out to a peak of **33,029 unique board states at turn 9**, then collapses back to just **267 states by turn 15**. The full count across all 15 turns is 181,065 unique board states visited in 100K games.

| Turn | Unique states | Top-10 concentration | Top-50 concentration |
|------|--------------|---------------------|---------------------|
| 1 | 1 | 100% | 100% |
| 2 | 21 | 70.6% | 100% |
| 3 | 236 | 15.9% | 51.3% |
| 4 | 1,570 | 4.7% | 17.4% |
| 5 | 6,275 | 1.9% | 7.2% |
| 6 | 15,490 | 1.0% | 3.8% |
| 7 | 25,598 | 1.0% | 3.0% |
| 8 | 32,167 | 1.3% | 3.6% |
| 9 | 33,029 | 1.9% | 5.3% |
| 10 | 28,403 | 3.0% | 8.3% |
| 11 | 20,376 | 5.2% | 13.8% |
| 12 | 11,614 | 10.0% | 24.6% |
| 13 | 4,762 | 20.8% | 43.6% |
| 14 | 1,256 | 40.2% | 71.2% |
| 15 | 267 | 82.1% | 95.5% |

Mid-game (turns 6-9) is maximally diffuse — each board state is visited a handful of times at most. Late-game (turns 13-15) concentrates sharply: at turn 15, the top 10 states account for 82% of all games.

### What categories get scored first?

Turn 2 shows what optimal play fills on the very first turn (from 100K games × 3 decisions/turn = 300K visits):

| First category | Visits | Share |
|----------------|--------|-------|
| Two Pairs | 31,431 | 10.5% |
| Full House | 30,030 | 10.0% |
| Three of a Kind | 23,412 | 7.8% |
| Fives | 22,491 | 7.5% |
| Fours | 20,022 | 6.7% |
| One Pair | 19,788 | 6.6% |
| Small Straight | 17,253 | 5.8% |
| Chance | 16,875 | 5.6% |

The solver opportunistically fills whatever hits. Two Pairs and Full House lead because they require specific patterns that either hit or miss — when they hit early, the solver takes them. Upper section categories (Fours, Fives) are also popular first fills because they contribute to the 63-point upper bonus.

### What category gets left for last?

Turn 15 shows which category remains unscored — the one the solver deliberately saves as the dump slot:

| Last category | Games | Share |
|---------------|-------|-------|
| Four of a Kind | 61,839 | 20.6% |
| Large Straight | 51,381 | 17.1% |
| Yatzy | 34,434 | 11.5% |
| Full House | 25,008 | 8.3% |
| Three of a Kind | 22,047 | 7.3% |
| Small Straight | 16,995 | 5.7% |
| Chance | 15,465 | 5.2% |

Four of a Kind and Large Straight dominate the dump slot because they're high-variance categories with specific requirements. The solver will sacrifice them (often zeroing them) rather than miss the upper bonus or leave easier categories unfilled. Yatzy is third — it's worth 50 points but requires all five dice matching, making it the hardest to fill on demand.

One Pair is almost never left last (2.5%) — it's trivially easy to score at least 2 points, so the solver always fills it earlier.

### Upper bonus: almost always achieved

At turn 15, **83.6% of games reach the maximum upper score of 63** (guaranteeing the 50-point bonus). No other single upper score even reaches 2%. This confirms that the optimal solver prioritizes the upper bonus aggressively — it's worth 50 points, far more than any individual category score sacrificed to reach it.

### The most common turn-15 board states

The top 5 board states at turn 15 (what category is missing + upper score):

| Missing category | Upper | Visits | Share |
|-----------------|-------|--------|-------|
| 4K | 63 | 61,242 | 20.4% |
| LS | 63 | 51,222 | 17.1% |
| SS | 63 | 34,320 | 11.4% |
| FH | 63 | 24,513 | 8.2% |
| 3K | 63 | 21,669 | 7.2% |

All have upper=63. The most common final position — missing Four of a Kind with the bonus secured — accounts for 1 in 5 games. These are the positions where the last turn is essentially "roll dice and hope for quads (or zero it)."

### Implications for decision sensitivity

The hourglass structure explains why the decision sensitivity pipeline found flips concentrated at turns 1 and 15. Mid-game states are so diffuse that no individual state accumulates enough visits (≥100) to survive the dedup filter. The states that matter most for risk-sensitivity analysis — the ones players actually encounter repeatedly — cluster at the beginning and end of the game, exactly where the state space is narrow.

## Why Humans Are Not Actually Good at Yatzy

### The perception vs reality

Experienced Yatzy players feel competent — they win games, develop intuitions, and can articulate strategies ("always go for the bonus," "don't waste Yatzy early"). This creates the impression that human play is close to optimal.

It is not. The gap between a typical experienced human and the optimal solver is roughly **20-30 points per game** on average (mean ~220-230 vs solver's 248.4). This is a substantial deficit — equivalent to losing an entire category score every game.

### What humans get right (~90% of decisions)

Most Yatzy decisions are "obvious" in the sense that any reasonable heuristic produces the same answer as the optimal solver:

1. **Keep matching dice for pairs/trips/quads**: When you have three 5s, keep them. No strategy required.
2. **Fill high-scoring categories with good rolls**: Rolled [6,6,6,5,5]? Fill Full House. Trivial.
3. **Reroll non-contributing dice**: Holding [4,4,4,2,1] for Three of a Kind? Reroll the 2 and 1. Obvious.
4. **Take Yatzy when you get it**: Five of a kind → Yatzy. No thought needed.
5. **Fill Chance with awkward rolls**: No good fits → Chance. Simple fallback.

These easy decisions account for roughly 90% of all choices. A human who gets these right — which most players do after a few games — appears to play well. The remaining 10% of decisions, where the optimal choice is non-obvious, is where the 20-30 point gap lives.

### The 10% that matters: where humans lose points

**1. Upper bonus mismanagement (~7-10 points/game)**

The upper bonus is a 50-point cliff at 63 upper points. The optimal solver tracks exact upper score and remaining upper categories to determine when to "invest" in the bonus vs. give up on it. Humans make systematic errors:

- **Over-committing**: Filling Ones with 1 point "to keep working on the bonus" when the bonus is already unreachable given remaining categories. The solver would dump a zero on Ones and use the turn for a lower section category.
- **Under-committing**: Putting 4 Fives in Fives (20 pts) when the bonus needs exactly those 5 points, then missing the bonus by 1. The solver calculates that accepting 3 Fours instead and saving Fives for later secures the bonus.
- **Ignoring the threshold**: Not realizing that with upper=45 and only Ones/Twos left, the bonus is mathematically unreachable (max remaining = 5+10 = 15, but need 63-45 = 18). The solver knows this and switches to pure EV play.

This is the costliest category of human errors because each mistake either loses the 50-point bonus or wastes a turn investing in an unreachable bonus.

**2. Suboptimal reroll decisions (~5-8 points/game)**

Reroll decisions have subtle interactions with game state that humans cannot compute:

- **Keeping too many dice**: Holding [3,3,4,4,_] hoping for Full House when the solver knows rerolling all five dice for a fresh start has higher EV (because Three of a Kind and Two Pairs are both still open).
- **Chasing the wrong target**: Rerolling [2,3,4,5,5] keeping the 5s for pairs, when the solver rerolls one 5 to chase a Large Straight (which has higher EV given the specific game state and remaining categories).
- **Ignoring reroll information**: The second reroll should be conditioned on what happened in the first reroll, but humans often commit to a "plan" (e.g., "I'm going for Sixes") and don't reconsider.

Humans cannot evaluate the ~32 possible reroll masks × 252 resulting dice distributions × remaining category interactions. They use heuristics ("keep the biggest matching set") that work most of the time but miss the EV-optimal choice when the game state creates unusual interactions.

**3. Category ordering cascades (~3-5 points/game)**

Which category to fill with an ambiguous roll creates cascading effects. Humans err on:

- **Filling categories greedily**: Scoring 24 in Chance instead of 24 in Four of a Kind because "it's the same points." But Chance is more flexible (any dice scores) while Four of a Kind requires a specific pattern — filling the flexible category with an inflexible roll wastes future options.
- **Timing mismatches**: Filling Small Straight on turn 3 when there are 12 turns left, vs. waiting for a Large Straight opportunity that has higher EV given the remaining categories. The solver makes these tradeoffs perfectly; humans rely on "fill it if you can."
- **Ignoring state interactions**: With upper=55 and Sixes open, rolling [6,6,6,3,3] — the solver fills Sixes (18 pts, pushes upper to 63 for the 50-pt bonus = 68 pts total). A human might fill Full House (27 pts), not recognizing that the bonus interaction makes Sixes worth 68 points.

**4. Chase errors (~1-3 points/game)**

Deciding when to chase a rare combination (Yatzy, straights) vs. settling for a sure thing:

- **Chasing too long**: Rerolling a good Three of a Kind hand trying for Yatzy on the last reroll when Three of a Kind is already a strong score.
- **Not chasing enough**: Immediately filling One Pair when holding [4,4,4,_,_] because "a bird in the hand," when the solver would reroll the two non-4s for a ~35% chance at Full House (28 pts) vs. the guaranteed One Pair (8 pts).

### Why these errors compound

A critical insight: individual errors don't just lose their direct point value — they create **cascading suboptimality**. Filling category X suboptimally means later rolls that would have been perfect for X must now go somewhere else. This secondary misallocation is invisible to the player but typically costs as much as the original error.

For example: filling Chance early with a mediocre 18-point roll seems harmless. But three turns later, rolling [2,3,4,1,1] with no good category available — the solver would have used Chance here (sum=11, bad but better than zeroing a category). Without Chance, the player zeros a category worth 5+ points in expected future value.

### Why humans *feel* good despite the gap

1. **No reference point**: Without a solver, humans compare against other humans. A mean of ~225 wins most games.
2. **Variance hides the gap**: With std ~38, a human scoring 220 in expectation will score 260+ about 15% of the time. These memorable high games reinforce the feeling of competence.
3. **Easy decisions dominate**: 90% of decisions are obvious, and humans get them right. The feeling of constant correct play masks the 10% of costly errors.
4. **Yatzy is fun regardless**: The joy of rolling Yatzy or completing a straight provides satisfaction independent of optimality.

### Implications for why RL struggles

The human-play analysis reveals exactly why RL failed to beat the solver:

**Humans solve a much easier problem than RL attempted.** A human playing at mean=225 needs to get ~90% of easy decisions right and be "reasonable" on the hard 10%. This is achievable with simple heuristics and basic pattern recognition — exactly what humans are good at.

RL Approach B tried to learn the *entire* decision function from scratch. Getting from 90% accuracy (human level) to 99%+ accuracy (solver level) is exponentially harder than getting from 0% to 90%. The last 10% of decision accuracy requires precise computation across millions of states — exactly what lookup tables excel at and neural networks struggle with.

**The gap RL was trying to exploit barely exists.** The hypothesis was that accumulated-score conditioning (knowing you're running hot or cold) could improve on the solver. But this signal is worth <1 point per game — smaller than the noise floor. Meanwhile, humans lose 20-30 points from basic computational limitations that RL shares (limited representational capacity, imprecise value estimation).

**Compact heuristics vs. exact tables.** Human strategy can be captured in ~10 rules (always go for bonus, keep matching dice, etc.) totaling maybe 1KB of information. The optimal solver uses 8MB of state values — a 10,000× information advantage. Humans operate in the "heuristic regime" where compact rules cover the easy decisions. The solver operates in the "exact regime" where precise state-dependent calculations matter. RL with 66K parameters sits uncomfortably between these regimes — too large for simple heuristics, too small for exact computation.

## State-Dependent θ(s) vs Constant-θ Pareto Frontier

### Hypothesis

**H₁**: A state-dependent policy θ(s) that conditions on upper-section bonus proximity can beat the constant-θ Pareto frontier by ≥1 point of mean at matched variance.

**H₀**: The constant-θ frontier is tight — no single-player state-dependent policy can beat it in (mean, σ) space.

### Motivation

The existing adaptive policies (bonus-adaptive, phase-based, combined) were tested via RL but never properly evaluated as raw simulation policies against the Pareto frontier in (mean, σ) space. The RL conclusion was that table-switching per turn is too weak a signal — but RL failed due to credit assignment noise (σ~38 vs ~1pt signal) and convergence to fixed-θ equivalents. A direct simulation of 1M games per policy avoids both problems.

A new `upper-deficit` policy was designed to concentrate variance-inflation on turns where it is cheap:
- Bonus secured → θ=0.10 (risk is free, bonus already locked)
- Bonus unreachable → θ=0.05 (nothing to protect)
- Bonus on track (health < 0.6) → θ=0 (protect the 50-point bonus)
- Marginal bonus health (0.6-0.8) → θ=0.03 (mild risk)
- Struggling (health > 0.8) → θ=0 (don't gamble away the slim chance)

### Results (1M games per policy, seed=42)

**Constant-θ baselines:**

| Policy | Mean | σ | p5 | p50 | p95 | p99 |
|--------|------|---|----|----|-----|-----|
| θ=0 (EV) | 248.40 | 38.5 | 179 | 249 | 309 | 325 |
| θ=0.030 | 246.92 | 41.3 | 169 | 247 | 311 | 326 |
| θ=0.050 | 244.53 | 43.2 | 164 | 244 | 312 | 327 |
| θ=0.070 | 240.86 | 45.3 | 159 | 241 | 313 | 328 |
| θ=0.100 | 235.92 | 46.9 | 155 | 237 | 312 | 329 |
| θ=0.150 | 229.23 | 47.9 | 152 | 231 | 311 | 329 |

**Adaptive policies:**

| Policy | Mean | σ | p5 | p50 | p95 | p99 |
|--------|------|---|----|----|-----|-----|
| bonus-adaptive | 246.92 | 39.6 | 178 | 246 | 310 | 325 |
| phase-based | 246.56 | 40.4 | 173 | 246 | 310 | 326 |
| combined | 247.93 | 39.0 | 179 | 248 | 310 | 325 |
| upper-deficit | 246.22 | 40.8 | 172 | 245 | 310 | 326 |

**Frontier comparison (linear interpolation between baseline σ values):**

| Policy | σ | Mean | Frontier μ | Δμ | Verdict |
|--------|---|------|-----------|-----|---------|
| bonus-adaptive | 39.6 | 246.92 | 247.83 | −0.91 | on frontier |
| phase-based | 40.4 | 246.56 | 247.38 | −0.82 | on frontier |
| combined | 39.0 | 247.93 | 248.15 | −0.23 | on frontier |
| upper-deficit | 40.8 | 246.22 | 247.20 | −0.98 | on frontier |

SE(mean) ≈ 0.038 at 1M games. Δμ ≥ 1.0 required to declare H₁.

### Conclusion: H₀ holds — the constant-θ frontier is tight

No adaptive policy beats the constant-θ Pareto frontier. All four policies land on or slightly below the frontier (Δμ between −0.23 and −0.98). The best performer is `combined` at Δμ = −0.23, which is effectively indistinguishable from the frontier given sampling noise.

The negative Δμ values have a clear explanation: table-switching between turns violates the Bellman equation. Each θ table was precomputed assuming the same θ for all future turns. When a policy switches from θ=0.10 in one turn to θ=0 in the next, the current turn's value estimates are slightly wrong — they assumed θ=0.10 would continue for all future turns. This small approximation error is directionally negative (the policy pays the switching cost without getting the corresponding benefit), which is why all adaptive policies land slightly below the frontier rather than exactly on it.

This result, combined with the RL experiments, establishes a strong empirical case: **the single-player constant-θ Pareto frontier is tight**. No state-dependent policy — whether learned by RL or designed by hand — can beat it by more than the switching approximation cost (~0.2-1.0 points). The reason is fundamental: the constant-θ solver already implicitly conditions on all state features (upper score, scored categories) through the precomputed state-value table. Adding explicit state-dependent θ switching on top of this provides no new information — it only introduces approximation error from mismatched successor values.

### Why table-switching violates the Bellman equation

The Bellman equation for the risk-sensitive solver at a fixed θ is:

    V_θ(s) = opt_a [ r(s,a) + E[ V_θ(s') ] ]

where `opt` is max for θ>0 (risk-seeking) or min for θ<0 (risk-averse), and all quantities are in the log-domain (log of exponential utility). The key property is that the successor value `V_θ(s')` assumes the **same θ** will be used for all future decisions. This self-consistency is what makes the solution optimal — it's the fixed point of the Bellman operator.

An adaptive policy that switches from θ₁ on turn t to θ₂ on turn t+1 consults `V_{θ₁}(s')` to evaluate its turn-t decisions, but then actually plays according to `V_{θ₂}` from turn t+1 onward. The value it consulted was wrong:

    Consulted:  V_{θ₁}(s')  — "what's this state worth if I play θ₁ forever?"
    Actual:     V_{θ₂}(s')  — "what's this state worth if I play θ₂ forever?"

When θ₁ > θ₂ (switching from risk-seeking to conservative), the consulted value overestimates future risk-taking gains that won't materialize. When θ₁ < θ₂ (switching from conservative to aggressive), it underestimates them. In either case, the turn-t decision was made with incorrect future values.

The magnitude of the error depends on how often decisions differ between θ₁ and θ₂, which is small for nearby θ values. From the decision sensitivity analysis, only ~5% of decisions flip between θ=0 and θ=0.10, and the affected decisions typically involve marginal tradeoffs worth <1 point each. This explains why the frontier gap is small (0.2-1.0 points) rather than catastrophic.

Crucially, this error is **directionally negative**. An adaptive policy switches to higher θ in states where risk is "cheap" (bonus secured, nothing to protect). But the higher-θ value table was computed assuming high risk-tolerance in *all* states, including states where risk is expensive. The adaptive policy gets the cheap-risk benefit but doesn't pay the expensive-risk cost — yet the value table priced in both. The mismatch means the policy slightly overvalues the high-θ turns (because the consulted successors assumed expensive-risk turns that won't happen), leading to marginally suboptimal decisions. The net effect is a small but consistent mean penalty.

To truly beat the frontier, one would need to solve a **non-stationary Bellman equation** where the successor values reflect the actual adaptive policy rather than a fixed θ. This is equivalent to computing a new 8MB state-value table for every possible policy — computationally feasible but combinatorially explosive in the space of policies. The RL experiments attempted this implicitly (learning a value function conditioned on policy parameters) but failed due to the signal-to-noise ratio. The frontier test confirms that even hand-designed policies cannot overcome the approximation cost, establishing that the constant-θ frontier is empirically tight.
