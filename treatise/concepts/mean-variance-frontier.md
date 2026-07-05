# Mean-Variance Frontier

The **mean-variance frontier** (also called the efficient frontier) is a curve showing the best achievable tradeoff between expected return and risk.

## In Yatzy

Each risk parameter θ produces a different strategy, and each strategy has a mean score and a standard deviation. Plotting all of these traces out a frontier:

- **x-axis**: standard deviation (risk/volatility)
- **y-axis**: mean score (expected return)

Points on the frontier represent **efficient** strategies — you cannot increase the mean without also increasing variance, and vice versa.

## Key Insight: Asymmetry

The Yatzy frontier is asymmetric:

- **Risk-averse side** (theta < 0): hedging is shallow and quickly exhausted. The minimum-variance point (theta = -0.05) trims sigma by only 7% (38.5 to 35.7) at a cost of 3.3 mean points, and more negative theta is strictly dominated (theta = -0.20 has both lower mean and higher sigma than theta = 0).
- **Risk-seeking side** (theta > 0): no extra mean exists to gain (theta = 0 is the mean-optimum); positive theta buys variance (sigma up to 47-48) while giving up mean. What it purchases is tail mass, not average performance.

## Origin

The concept comes from modern portfolio theory (Markowitz, 1952), where it describes the optimal mix of financial assets. Here we apply the same framework to dice game strategies — each θ is an "asset allocation" between safe and speculative play.
