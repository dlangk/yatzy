# Central Limit Theorem

The **Central Limit Theorem** (CLT) states that the sum of many independent random variables converges to a normal distribution, regardless of the underlying distribution of each variable. It is one of the most powerful results in probability theory.

## The Standard Result

If X_1, X_2, ..., X_n are independent and identically distributed with mean mu and variance sigma^2, then:

> (X_1 + X_2 + ... + X_n - n*mu) / (sigma * sqrt(n)) --> N(0, 1)

as n grows. The convergence is remarkably fast -- even n = 30 often gives a good Gaussian approximation. This is why the normal distribution appears everywhere in science: any quantity that arises as a sum of many small independent contributions will be approximately normal.

## Why It Doesn't Fully Apply to Yatzy

A Yatzy score is the sum of 15 category scores, but the CLT's assumptions break down in two important ways:

1. **Not independent**: category choices are coupled through the upper bonus. Taking a low score in Threes to chase a Yatzy affects your Fours decision later. The 50-point bonus at upper total >= 63 creates a non-linear dependency across all upper categories. When you score Sixes, it changes the probability that you'll reach the bonus threshold, which in turn changes the optimal strategy for every remaining upper category.

2. **Not identically distributed**: each category has a different scoring distribution. A Yatzy (50 points) has fundamentally different statistics than Ones (1--5 points). The distributions also depend on the *policy*, which adapts to game state. A risk-averse player sees different per-category distributions than a risk-seeking player.

There is a third, subtler issue: the number of summands (15) is small. The CLT is an asymptotic result, and n = 15 is not large enough for the convergence to be tight, especially when the individual distributions are skewed.

## The Observed Shape

The actual score distribution under optimal play is **not normal**. Exact density evolution reveals:

- A heavier left tail (bad games where the bonus is missed)
- A pronounced mode around 250 (the bonus-secured region)
- A lighter right tail (constrained by the maximum achievable score of ~374)
- Slight multimodality near the bonus boundary

The distribution is better described as a **4-component mixture** than a single Gaussian. The components roughly correspond to: games with no bonus and bad luck, games with no bonus and average luck, games with bonus and average luck, and games with bonus and good luck.

## When CLT Is Still Useful

The CLT does apply in two contexts:

- The **upper section subtotal** in isolation is somewhat more symmetric and closer to normal, since those six categories have more similar distributions
- **Monte Carlo estimation**: when averaging over millions of independent simulated games, the sample mean of scores does satisfy CLT conditions. This is why simulation gives reliable confidence intervals for expected values even though individual scores are non-normal

## Implications for Risk Analysis

The non-normality of Yatzy scores means that standard risk measures based on the normal distribution (e.g., "mean plus or minus two standard deviations covers 95% of outcomes") are unreliable. The heavier left tail means worst-case outcomes are more probable than a Gaussian would predict.

This is why the project uses **exact density evolution** rather than Gaussian approximations. The forward-DP computation of the full probability mass function captures all the structural features -- multimodality, asymmetric tails, bonus-induced discontinuities -- that the CLT smooths away.

For risk-sensitive strategy evaluation, getting the tails right is crucial. A risk-averse player (theta < 0) cares specifically about the left tail, which is precisely where the CLT approximation is worst. Exact PMFs from density evolution provide the ground truth that neither CLT approximations nor finite Monte Carlo samples can match.

## Quantifying the Deviation

Several statistical tests confirm the non-normality:

- **Skewness**: the score distribution under optimal play has negative skew (~-0.3), reflecting the heavier left tail. A normal distribution has skewness exactly 0.
- **Kurtosis**: slightly positive excess kurtosis (~0.2), indicating heavier tails than Gaussian. The 4-component mixture structure produces this through mode separation.
- **Kolmogorov-Smirnov test**: rejects normality at any reasonable significance level for sample sizes above ~1,000 games.

These deviations are small enough that a Gaussian approximation gives reasonable results for casual analysis (mean +/- std captures the bulk of the distribution), but large enough to matter for risk-sensitive optimization where tail behavior drives decisions.
