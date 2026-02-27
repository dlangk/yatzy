# Exponential Utility

**Exponential utility** is a mathematical function that models how a decision-maker values outcomes when they care about risk, not just averages.

## The Function

Instead of maximizing E[score], we maximize:

> E[e^(θ · score)]

where θ (theta) is the **risk parameter**. In practice, we work in the log domain for numerical stability:

> L(S) = ln(E[e^(θ · total)])

## What θ Controls

- **θ = 0**: recovers standard expected value maximization (risk-neutral)
- **θ < 0**: risk-averse — the player prefers consistent scores and avoids catastrophic outcomes
- **θ > 0**: risk-seeking — the player chases high scores, accepting more bad games

## Why Exponential?

Exponential utility has a special property: **constant absolute risk aversion (CARA)**. The degree of risk aversion doesn't depend on your current score — you're equally cautious whether you're at 100 or 200 points. This makes it tractable for backward induction because the risk attitude doesn't change during the game.

## Practical Range

For Yatzy (scores ~250), the useful range is |θ| < 0.5. Beyond that, the utility function saturates and every strategy looks the same.
