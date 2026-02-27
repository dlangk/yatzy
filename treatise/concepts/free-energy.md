# Free Energy

The **free energy** is a concept from statistical mechanics that connects the risk-sensitive value function to physics. The optimal value under exponential utility has the exact same mathematical form as the Helmholtz free energy in thermodynamics.

## The Formula

The risk-sensitive value of a state is:

> V(s) = -(1/theta) * log E[exp(-theta * score | s)]

This is the **certainty equivalent** under CARA utility. In statistical mechanics, the identical expression appears as the Helmholtz free energy:

> F = -(1/beta) * log Z

where beta = 1/(k_B * T) is the inverse temperature and Z = sum of exp(-beta * E_i) is the **partition function** summing over energy levels E_i.

## The Analogy

| Statistical Mechanics | Risk-Sensitive Yatzy |
|---|---|
| Inverse temperature beta | Risk parameter theta |
| Energy level E_i | Negative score -score_i |
| Partition function Z | E[exp(-theta * score)] |
| Free energy F | Value function V(s) |
| Boltzmann distribution | Optimal mixed strategy weights |

High theta (risk-seeking) corresponds to high temperature: the system explores widely and rare high-energy (high-score) states contribute more to the partition function. Low theta (risk-averse) corresponds to low temperature: the system concentrates on the most probable, moderate outcomes. At theta = 0, we recover the ground state (maximum expected value), analogous to zero temperature.

## Why This Matters

The analogy is more than cosmetic. It imports powerful mathematical results from physics:

- **Convexity**: the free energy is convex in theta, guaranteeing a smooth, well-behaved interpolation between risk-averse and risk-seeking strategies
- **Derivatives**: dV/d(theta) at theta=0 gives the variance of the score distribution, connecting risk sensitivity to score volatility. Higher derivatives give cumulants (skewness, kurtosis).
- **Large deviations**: the free energy framework naturally handles tail probabilities via the Legendre transform, which is exactly what risk-averse players care about
- **Phase transitions**: abrupt changes in optimal strategy as theta varies correspond to phase transitions in the physical system

## Practical Consequence

The free energy perspective explains several empirical observations:

1. The mean-variance frontier is smooth and convex -- guaranteed by thermodynamic convexity
2. Strategy changes are gradual for small |theta| but can be abrupt at critical values -- analogous to phase transitions
3. The risk premium (E[X] - CE) has a natural interpretation as the entropy contribution to free energy

The physical intuition also guides algorithm design. The partition function Z can be computed efficiently using the same dynamic programming structure as the value function, and the log-sum-exp trick is the standard physicist's method for stable computation of log Z.

## Beyond Analogy

The free energy framework is not merely a useful analogy -- it is a mathematical identity. The same theorems that govern thermodynamic systems apply directly to risk-sensitive Yatzy. This means results from 150 years of statistical mechanics can be imported without re-derivation:

- The Gibbs variational principle provides an alternative proof that the DP solution is optimal
- Fluctuation-dissipation relations connect the variance of optimal-play scores to the sensitivity of the value function to theta
- Replica methods from disordered systems theory could, in principle, analyze the typical-case complexity of the DP computation

The connection also flows in reverse: Yatzy provides a tractable, exactly solvable test case for ideas from statistical physics that are typically studied only in approximate or asymptotic regimes.
