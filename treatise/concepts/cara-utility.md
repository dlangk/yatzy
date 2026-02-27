# CARA Utility

**Constant Absolute Risk Aversion** (CARA) is a class of utility functions where the decision-maker's risk attitude does not depend on their current wealth or score. The canonical form is the exponential utility function.

## The Function

> u(x) = -exp(-theta * x)

where theta is the **risk parameter**:

- **theta = 0**: the agent is risk-neutral and maximizes expected value
- **theta < 0**: risk-averse -- the agent prefers a certain 240 over a 50/50 gamble between 200 and 280
- **theta > 0**: risk-seeking -- the agent prefers the gamble, chasing the upside

The negative sign and the negative exponent are conventions that ensure u(x) is increasing in x (more points are always better) and concave for theta < 0 (diminishing marginal value of additional points).

## Why "Constant"?

The **Arrow-Pratt measure** of absolute risk aversion is:

> A(x) = -u''(x) / u'(x) = theta

For exponential utility, this measure is a constant -- it does not depend on x. Compare this to logarithmic utility (u(x) = log(x)), where risk aversion decreases as wealth grows (wealthy people take bigger gambles). With CARA, you are equally cautious whether your current score is 50 or 200.

This constancy is not just mathematically elegant -- it is economically meaningful. It says the player's risk attitude is a stable personality trait, not something that shifts based on how the game is going.

## Why CARA Is Perfect for Yatzy

The constant risk aversion property makes the risk-sensitive solver tractable. Because A(x) doesn't change with the accumulated score, the backward induction algorithm can still decompose the game into independent per-state computations. The value of a state depends only on the state variables (upper progress, scored mask), not on the total score accumulated so far.

If risk aversion varied with score (as with logarithmic or power utility), the total accumulated score would become a state variable. This would expand the state space by a factor of ~300 (possible total scores), making exact solution computationally prohibitive.

## Practical Range

In the solver, theta values in the range [-3.0, +3.0] span the full spectrum from extreme risk aversion to extreme risk seeking. The practical range where strategies meaningfully differ is roughly |theta| < 0.5; beyond that, the utility function saturates and strategies converge.

For Yatzy scores around 250, the key scale is 1/score ~ 0.004. Values of |theta| much larger than ~1 produce such extreme curvature that the utility function becomes nearly flat over the relevant score range, and the optimal action reduces to min-variance (theta << 0) or max-variance (theta >> 0).

## Connection to Certainty Equivalents

The **certainty equivalent** of a random outcome X under CARA utility is:

> CE(X) = -(1/theta) * log(E[exp(-theta * X)])

This is the guaranteed score the agent considers equally desirable to the random outcome. For theta < 0, CE < E[X]: the agent would accept a lower guaranteed score to avoid uncertainty. The gap E[X] - CE is the **risk premium** -- the price the agent would pay for insurance.

## Alternatives and Why They Fail

Other utility functions exist but create problems for backward induction:

- **Logarithmic utility** u(x) = log(x): risk aversion decreases with score (DRRA). The total accumulated score becomes a state variable, expanding the state space by ~300x.
- **Power utility** u(x) = x^(1-gamma)/(1-gamma): same DRRA problem. Also undefined for negative scores.
- **Quadratic utility** u(x) = x - a*x^2: risk preference depends on x, and the function eventually decreases (more points become bad), which is nonsensical for scoring games.
- **Prospect theory** (Kahneman-Tversky): reference-dependent utility with loss aversion. The reference point shifts during the game, making backward induction intractable.

CARA is the unique utility family that combines risk sensitivity with state-space tractability. This is not a limitation but a feature: the single parameter theta provides a clean, one-dimensional spectrum from extreme risk aversion to extreme risk seeking, covering the full range of plausible human preferences.
