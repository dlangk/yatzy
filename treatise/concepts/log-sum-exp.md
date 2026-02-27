# Log-Sum-Exp

The **log-sum-exp** (LSE) operation computes log(sum of exp(x_i)) in a numerically stable way. It appears whenever you need to combine exponential terms without overflow or underflow.

## The Problem

Naively computing log(exp(x_1) + exp(x_2) + ... + exp(x_n)) fails when the x_i values are large or very negative. In Yatzy's risk-sensitive solver, the arguments are theta * score, where scores can reach 300+ and theta can be several units. The product theta * 300 = 900 far exceeds the float32 exponent range (~88), causing exp to overflow to infinity.

Even with float64, large theta values would eventually cause overflow. And for negative theta * score values, exp underflows to zero, losing all information about the relative magnitudes of different terms.

## The Trick

Factor out the maximum value:

> LSE(x_1, ..., x_n) = max(x) + log( sum of exp(x_i - max(x)) )

Now every exponent is <= 0, so no term overflows. The largest term contributes exp(0) = 1, so the sum is >= 1 and the log is >= 0. Both overflow and underflow are eliminated.

This is mathematically exact -- no approximation is involved. It is purely a rearrangement that avoids numerical pathology.

## Why It Matters for Risk-Sensitive DP

The risk-sensitive Bellman equation replaces the standard max with a soft-max:

> V(s) = (1/theta) * LSE(theta * Q(s, a_1), ..., theta * Q(s, a_n))

As theta -> 0, this converges to max(Q(s, a_i)) -- the risk-neutral case. As |theta| grows, the soft-max interpolates between max (theta > 0) and min (theta < 0) of the Q-values.

For |theta| <= 0.15, the solver works in the utility domain directly (exp values are manageable in float32). For |theta| > 0.15, it switches to the log domain and uses the LSE trick throughout. This threshold was determined empirically by measuring where float32 precision begins to degrade.

## SIMD Implementation

The solver implements LSE using NEON SIMD intrinsics, processing 4 floats at a time. The critical inner loop:

1. Compute the max of all terms (using NEON vmaxq)
2. Subtract the max from each term (vsub)
3. Apply fast exp via degree-5 Cephes polynomial (~2 ULP accuracy)
4. Sum the results (vaddq + horizontal reduce)
5. Take the log and add back the max

The fast polynomial exp avoids the cost of hardware transcendental instructions while maintaining sufficient accuracy. The combination of the LSE trick and fast exp keeps the risk-sensitive solver within 2--3x of the risk-neutral solver's speed.

## Connection to Machine Learning

LSE appears throughout machine learning under different names. The softmax function used in neural network classifiers is:

> softmax(x_i) = exp(x_i) / sum(exp(x_j)) = exp(x_i - LSE(x))

The same numerical stability trick applies: subtract the max before exponentiating. Cross-entropy loss, attention mechanisms, and energy-based models all rely on LSE computations.

In the Yatzy solver, LSE serves an analogous role to softmax in neural networks: it provides a smooth, differentiable approximation to the max operator. The temperature parameter theta plays exactly the role of the temperature in softmax: as theta -> infinity, LSE -> max; as theta -> 0, LSE -> average.

## Why Not Use Float64?

Float64 (double precision) has an exponent range up to ~709, which would handle theta * score products up to about 2.3. This covers part of the useful theta range but not all of it. More importantly, the solver processes billions of floating-point operations, and float32 SIMD processes twice as many values per instruction as float64. The LSE trick makes float32 sufficient for all theta values, preserving the 2x throughput advantage.
