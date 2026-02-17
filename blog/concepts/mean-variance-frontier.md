# Mean-Variance Frontier

The **mean-variance frontier** (also called the efficient frontier) is a curve showing the best achievable tradeoff between expected return and risk.

## In Yatzy

Each risk parameter θ produces a different strategy, and each strategy has a mean score and a standard deviation. Plotting all of these traces out a frontier:

- **x-axis**: standard deviation (risk/volatility)
- **y-axis**: mean score (expected return)

Points on the frontier represent **efficient** strategies — you cannot increase the mean without also increasing variance, and vice versa.

## Key Insight: Asymmetry

The Yatzy frontier is asymmetric:

- **Risk-averse side** (left/downward): the curve is steep. A small sacrifice in mean score buys a large reduction in variance. This is "cheap insurance."
- **Risk-seeking side** (right/upward): the curve is flat. You must accept a lot more variance to gain a little more mean. High scores are expensive to chase.

## Origin

The concept comes from modern portfolio theory (Markowitz, 1952), where it describes the optimal mix of financial assets. Here we apply the same framework to dice game strategies — each θ is an "asset allocation" between safe and speculative play.
