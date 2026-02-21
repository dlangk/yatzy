# Risk-Sensitive Solver: The theta Parameter

This document describes the mathematical framework for risk-sensitive Yatzy solving. The standard solver maximizes expected score. The risk-sensitive solver generalizes this by introducing a parameter theta that controls the tradeoff between expected value and score variance.

---

## Exponential Utility

The standard solver maximizes expected total score:

```
E[total_score]
```

The risk-sensitive solver maximizes exponential utility:

```
u(x) = e^(theta * x)
```

- **theta = 0**: Reduces exactly to the standard EV-optimal solver.
- **theta > 0**: Risk-seeking -- overweights high scores, underweights low scores. The policy chases high-variance plays.
- **theta < 0**: Risk-averse -- overweights low scores, underweights high scores. The policy prefers safe, consistent plays.

The certainty equivalent (CE) is the score that provides the same utility as the lottery:

```
CE = (1/theta) * ln(E[e^(theta * total)])
```

For theta > 0, CE > E[total] (risk premium is positive). For theta < 0, CE < E[total].

---

## Log-Domain Storage and Computation

Rather than storing raw exponential utilities (which overflow for theta * score ~ 250 * 0.1 = 25), the solver works entirely in the **log domain**:

```
L(S) = ln(E[e^(theta * total) | S])
```

All recurrence relations are reformulated in log space. The key substitution is at **chance nodes**: the expected value under exponential utility becomes a **log-sum-exp** (LSE).

### Standard solver (chance node):

```
E(keep) = Sum_r P(keep -> r) * V(r)
```

### Risk-sensitive solver (chance node):

```
L(keep) = LSE_r { ln(P(keep -> r)) + L(r) }
```

where LSE (log-sum-exp) is computed with the standard numerical stability trick:

```
LSE(x_1, ..., x_n) = m + ln(Sum exp(x_i - m))
where m = max(x_i)
```

### Decision nodes

Decision nodes (category choice, keep selection) use **maximize** for theta > 0 (risk-seeking) and **minimize** for theta < 0 (risk-averse). In the standard EV solver both are just argmax; the sign flip for theta < 0 comes from the monotonicity reversal of the exponential utility when theta changes sign.

### theta = 0 equivalence

When theta = 0, the exponential utility is constant (e^0 = 1), so the risk-sensitive formulation is degenerate. The implementation detects theta = 0 and falls back to the exact EV code path with zero overhead.

---

## The LSE Transition: EV to Max

At each chance node, the EV solver computes a weighted mean:

```
E[V] = Sum p_i * V_i
```

The risk-sensitive solver replaces this with LSE. Taylor-expanding LSE around the weighted mean mu = Sum p_i * V_i:

```
Let delta_i = V_i - mu  (deviations from mean)

Sum p_i * exp(delta_i) = Sum p_i * (1 + delta_i + delta_i^2/2 + ...)
                       = 1 + 0 + sigma^2/2 + ...

where sigma^2 = Sum p_i * delta_i^2  (probability-weighted variance at this node)

LSE ~ mu + ln(1 + sigma^2/2 + ...) ~ mu + sigma^2/2 + O(sigma^3)
```

So **LSE is approximately EV + sigma^2/2** when the variance sigma^2 is small. But sigma here is in the **log domain**, not raw scores.

### Connecting log-domain spread to theta and raw scores

In the log domain, state values are L(S) = ln(E[e^(theta * total)]). For small theta, the cumulant expansion gives:

```
L(S) ~ theta * E[total|S] + theta^2 * Var[total|S] / 2 + ...
```

So the log-domain values at a chance node are approximately L_i ~ theta * EV_i, and the log-domain deviations are delta_i ~ theta * (EV_i - mean_EV). Therefore:

```
sigma^2_log ~ theta^2 * sigma^2_EV
```

where sigma_EV is the std of raw-score-domain state values across dice outcomes at this chance node.

### The transition criterion

```
LSE ~ EV   when  |theta| * sigma_EV << 1   (probability weighting dominates)
LSE -> max  when  |theta| * sigma_EV >> 1   (best outcome dominates exponentially)

theta_critical = 1 / sigma_EV
```

---

## The Dimensionless Control Parameter

The product |theta| * sigma_EV is a **dimensionless control parameter** that determines the regime. It does not matter whether theta is large and sigma_EV is small, or vice versa -- only their product matters.

### Physical interpretation

The exponential utility e^(theta * x) amplifies score differences. An outcome that is better by Delta points gets exponential weight e^(theta * Delta):

- When |theta| * Delta << 1: e^(theta * Delta) ~ 1 + theta * Delta -- linear weighting, probability dominates
- When |theta| * Delta >> 1: e^(theta * Delta) >> 1 -- exponential amplification, best outcome dominates

The transition happens at **|theta| * Delta ~ 1**, giving **theta ~ 1/Delta** where Delta is the characteristic score spread at a chance node.

### Regime map

Using sigma_node ~ 10 (middle estimate for Yatzy):

| |theta| | |theta| * sigma | Regime | Behavior |
|--------:|----------------:|--------|----------|
| 0.00 | 0.0 | EV-optimal | Exact expected value maximization |
| 0.01 | 0.1 | Near-EV | Indistinguishable from theta = 0 |
| 0.02 | 0.2 | Near-EV | Policy barely differs |
| 0.05 | 0.5 | **Interesting** | Meaningful risk/reward tradeoffs |
| 0.10 | 1.0 | **Interesting** | Strong risk sensitivity |
| 0.20 | 2.0 | Mostly degenerate | LSE ~ max at most nodes |
| 0.50 | 5.0 | Fully degenerate | Policy frozen |
| 1.00 | 10.0 | Fully degenerate | No further change |

Boundaries:

- **|theta| * sigma < 0.3**: LSE correction < 5%. Policy is essentially EV-optimal.
- **|theta| * sigma ~ 0.5-2**: The interesting range. Meaningfully different decisions, structured distributions, active mean-variance-percentile tradeoffs.
- **|theta| * sigma > 3**: LSE ~ max/min. Policy degenerate -- further increasing |theta| changes almost nothing.

---

## Estimating sigma_EV for Yatzy

Three independent methods converge on theta_critical ~ 0.07-0.17.

**Method 1: Variance decomposition.** Total game score std is sigma_total ~ 38.5. A game has ~45 chance nodes (15 turns * 3 rolls). If independent:

```
sigma_node ~ sigma_total / sqrt(45) ~ 38.5 / 6.7 ~ 5.7
-> theta_critical ~ 1/5.7 ~ 0.17
```

This is a lower bound on sigma_node because chance nodes within a turn are correlated (same state), so the actual per-node spread is larger.

**Method 2: Per-turn score spread.** Category scores range from 0 to 50, but successor state values moderate this. A bad roll costs ~5-15 points in total game value (future turns partially compensate). Estimated sigma_EV at a typical reroll chance node: ~8-15 points.

```
theta_critical ~ 1/sigma_EV ~ 0.07 to 0.12
```

**Method 3: Certainty equivalent expansion.** CE(S) = L(S)/theta ~ E[total] + (theta/2) * Var[total]. The risk correction is small when:

```
|theta/2| * Var << E[total]
|theta| << 2 * 248 / 1482 ~ 0.33
```

This is a weaker bound (whole-game level); individual chance nodes degenerate first.

---

## The Boltzmann / Statistical Mechanics Analogy

The LSE with parameter theta is mathematically identical to the **free energy** in statistical mechanics at inverse temperature beta = theta. The transition from EV to max is the **zero-temperature limit**: when the "temperature" 1/|theta| drops below the energy gap between states, the system freezes into the ground state.

In this analogy:

| Physics concept | Yatzy concept |
|----------------|---------------|
| Energy gaps between states | Score differences between dice outcomes |
| Inverse temperature beta | theta |
| Free energy | LSE value |
| Characteristic energy scale | sigma_EV |

The condition theta_critical = 1/sigma_EV is exactly the statement that the system freezes when the temperature drops below the energy gap -- a universal result in statistical physics.

---

## Useful theta Range for Yatzy

Combining the three estimates and the simulation data, the useful range is:

- **|theta| < 0.03**: Indistinguishable from EV-optimal. No need to sweep.
- **0.03 < |theta| < 0.20**: The active tradeoff regime. This is where risk preferences produce meaningfully different strategies.
- **|theta| > 0.5**: Fully degenerate. Mean loss saturates at ~62 points.

For practical purposes, a sweep over theta in [-0.10, +0.20] with step 0.01 covers the entire interesting range.
