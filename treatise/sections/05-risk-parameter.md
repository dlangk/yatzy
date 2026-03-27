:::section{#risk-parameter}

## Taking Risk

We defined the optimal strategy as "maximize the expected score." Such a strategy will get the highest possible *average* score. But, reality is, most people don't play Yatzy to get a good average. We like to play to win! We all remember that one time when all stars aligned and we got 322 points, and want to experience it again! So, in that spirit, let's see if we can design a solver that takes more risk.

**How can we encode risk-taking?** An expected value is calculated as the probability weighted sum E[<var>x</var>] = &Sigma;<sub>i</sub> <var>p(x<sub>i</sub>)</var> &middot; <var>x</var><sub>i</sub>. That means all scores are treated equally: going from 200 to 201 is worth exactly as much as going from 300 to 301. To make the solver care more about high scores, we transform each score using e<sup>&theta;<var>x</var></sup> before averaging. A single parameter &theta; (theta) controls the shape of this transformation. Use the slider to see: when &theta; &gt; 0, points at high scores are worth exponentially more; when &theta; &lt; 0, points at low scores dominate. At &theta; = 0, the transformation is flat and we recover the plain expected value.

:::html
<div class="chart-container chart-compact" id="chart-utility-curves">
  <div class="chart-controls">
    <label>&theta; = <span class="slider-value">0</span></label>
    <input type="range" class="chart-slider" min="0" max="28" value="13">
  </div>
  <div id="chart-utility-curves-svg"></div>
  <p class="chart-caption">Marginal value of one additional point at each score level. EV (dashed): flat, every point matters equally. Risk-weighted (solid): the curve bends toward the scores the solver cares most about.</p>
</div>
:::

**Think of &theta; as a risk-dial.** Turn it right and the solver takes larger risks chasing a legendary score. Turn it left and it plays like someone who really, really doesn't want to embarrass themselves. At zero, it plays for the best *average* outcome. The solver's willingness to gamble depends only on &theta; and the specific dice situation, never on how many points have been scored so far. Whether you're at 50 or 150 going into a round, the same &theta; produces the same risk attitude.

Use the slider below to see how the score distribution changes across the &theta; range. At &theta; = 0 the EV-optimal distribution is shown. As you decrease &theta; the distribution compresses with lower mean and less variance. As &theta; is increased, variance increases, pushing the tails outward before collapsing at extreme values. The tail probabilities displayed below the curve show how outcomes for different score thresholds compare to &theta; = 0. The ::concept[mean-variance frontier]{free-energy} at the bottom traces where each &theta; lands in risk-return space.

:::html
<div class="chart-container" id="chart-risk-theta">
  <div id="chart-risk-theta-distribution"></div>
  <div class="chart-controls chart-controls-center">
    <label>&theta; = <span class="slider-value">0</span></label>
    <input type="range" class="chart-slider" min="0" max="28" value="13">
  </div>
  <div id="chart-risk-theta-stats" class="tail-stats"></div>
  <div id="chart-risk-theta-frontier"></div>
  <p class="chart-caption">Score distribution, tail probabilities, and mean-variance frontier. Drag the slider to explore how θ reshapes the distribution and traces the frontier.</p>
</div>
:::

The best &theta; for protecting the 5th percentile (the floor) is &minus;0.03, which lifts p5 to 183, four points above the EV-optimal floor of 179. The best &theta; for the 95th percentile (the ceiling) is around +0.04 to +0.10, pushing p95 to 312 versus 309 at &theta; = 0. The best for p99 is +0.10, reaching 329.

:::html
<div class="chart-container" id="chart-risk-reward">
  <div id="chart-risk-reward-svg"></div>
  <div class="chart-controls chart-controls-center">
    <label>&theta; = <span class="slider-value">0</span></label>
    <input type="range" class="chart-slider" min="0" max="29" value="13">
  </div>
  <div class="chart-stats-panel"></div>
  <p class="chart-caption">Practical sweet spots: small θ shifts yield measurable percentile gains at precise cost in expected value.</p>
</div>
:::

### The Extreme Tail

The solver computes the *exact* probability of every possible final score via forward dynamic programming density propagation. This let's us model things far beyond what we can afford to simulate. The charts below show the right tail of the distribution (scores 280+) for several &theta; values, using exact probabilities down to 10<sup>&minus;15</sup>.

:::html
<div class="chart-container" id="chart-tail-games-needed">
  <div id="chart-tail-games-needed-svg"></div>
  <p class="chart-caption">Expected number of games to observe at least one game reaching each score. Horizontal lines mark sample sizes from a human lifetime to a trillion simulated games.</p>
</div>
:::

:::html
<div class="chart-container" id="chart-tail-bars">
  <div id="chart-tail-bars-svg"></div>
  <p class="chart-caption">Exact probability of reaching each score threshold, grouped by θ. Higher θ lifts the bars at every threshold, but the effect is tiny compared to the exponential drop-off.</p>
</div>
:::

:::math

### The Exponential Utility Framework

The utility function is <var>u</var>(<var>x</var>) = e<sup>&theta;<var>x</var></sup>. The solver maximizes the *certainty equivalent* (CE): the guaranteed score that the solver considers equally desirable to playing the random game. Imagine someone offers you a deal: skip the game and receive a fixed number of points instead. How many points would it take? That number is the CE.

:::equation
CE = (1/&theta;) &middot; ln E[e<sup>&theta; &middot; total</sup>]
:::

For &theta; &gt; 0, CE &gt; E[total]: the risk-seeker values a gamble above its expected value, because the chance of a high score is worth something on its own. For &theta; &lt; 0, CE &lt; E[total]: the risk-averter would accept fewer guaranteed points to avoid the chance of a disaster. The gap CE &minus; E[total] is the risk premium, and it scales with both &theta; and the variance of the outcome distribution.

This framework is known as ::concept[Constant Absolute Risk Aversion]{cara-utility} (CARA) in decision theory. Its defining property: the solver's risk attitude depends only on &theta; and the shape of the gamble, never on how many points have already been scored. Scoring 100 in the first five rounds does not make the solver more reckless in round six.

### From Weighted Sum to Log-Sum-Exp

The EV solver and the &theta; solver have the same structure. Both use probability-weighted averaging at chance nodes and argmax at decision nodes. The only difference is *what gets averaged*.

At a chance node, the EV solver computes a weighted sum of raw state values:

:::equation
E(keep) = &Sigma;<sub>r</sub> P(keep &rarr; r) &middot; V(r)
:::

The &theta; solver computes the same weighted sum, but of *exponentiated* state values:

:::equation
exp(L(keep)) = &Sigma;<sub>r</sub> P(keep &rarr; r) &middot; exp(L(r))
:::

where <var>L</var>(<var>S</var>) = ln E[e<sup>&theta; &middot; total</sup> | <var>S</var>]. The exponential e<sup>&theta;<var>x</var></sup> amplifies extreme values before the averaging happens. High scores contribute exponentially more when &theta; &gt; 0, low scores when &theta; &lt; 0.

Direct computation of exp(&theta; &middot; score) overflows for moderate &theta; and typical scores around 250. The solver works entirely in the log domain, where this weighted sum becomes a log-sum-exp (LSE):

:::equation
<var>L</var>(keep) = LSE<sub>r</sub> { ln(<var>P</var>(keep &rarr; r)) + <var>L</var>(r) }
:::

LSE is computed with the standard numerical stability trick: LSE(<var>x</var><sub>1</sub>, &hellip;, <var>x</var><sub>n</sub>) = <var>m</var> + ln(&Sigma; exp(<var>x</var><sub>i</sub> &minus; <var>m</var>)), with <var>m</var> = max(<var>x</var><sub>i</sub>).

At decision nodes, the solver picks the action that maximizes <var>L</var> when &theta; &gt; 0, or minimizes it when &theta; &lt; 0 (because dividing CE = <var>L</var>/&theta; by a negative &theta; flips the direction).

### Policy Degeneration at Large |&theta;|

When |&theta;| is large, the exponential amplification becomes so extreme that only the single best (or worst) outcome matters. The LSE at a chance node approaches a plain max:

:::equation
LSE &asymp; max<sub>i</sub>(<var>x</var><sub>i</sub>) &nbsp;&nbsp; when |&theta;| &middot; &sigma; &gt;&gt; 1
:::

where &sigma; is the spread of state values at that node. The transition depends on a dimensionless control parameter: the product |&theta;| &middot; &sigma;<sub>node</sub>. In Yatzy, the typical per-node score spread is &sigma;<sub>node</sub> &asymp; 10, giving three regimes:

| |&theta;| | |&theta;| &middot; &sigma; | Regime |
|-------:|------------------:|--------|
| &lt; 0.03 | &lt; 0.3 | Near-EV: policy barely differs from &theta; = 0 |
| 0.03&ndash;0.20 | 0.3&ndash;2 | Active: meaningful risk/return tradeoffs |
| &gt; 0.5 | &gt; 5 | Degenerate: LSE &asymp; max, policy frozen |

The marginal utility chart tells the same story from the score side: at |&theta;| = 0.5, the marginal value is effectively zero everywhere except the extreme tail. The solver is running a soft-max over outcomes, concentrating all its attention on the single best reachable score. Increasing |&theta;| further cannot change which action achieves that maximum, so the policy stops changing.

The degenerate max-policy does reach higher ceilings: the observed maximum across 1M games rises from 352 at &theta; = 0 to 362 at &theta; &asymp; 0.1. But it pays for those ceilings by sacrificing everything else (Full House hit rate drops from 92% to 54%, mean score from 248 to 210). The max-policy's best games are spectacular but vanishingly rare. With only 1M simulations, the observed maximum barely changes beyond &theta; &asymp; 0.1 because the ceiling-achieving games have become so improbable that even a million rolls cannot reliably sample them. In the limit, the max-policy converges to the theoretical maximum of the game, but with a probability so low it would take billions of simulated games to observe.

### f32 Numerical Stability

The solver uses f32 throughout for performance. This creates a numerical stability boundary. At |&theta;| &le; 0.15, the exponential utility can be computed in the "utility domain" via direct weighted averaging, with the same speed as the EV solver. Beyond |&theta;| &asymp; 0.166, mantissa erasure in f32 subtraction corrupts the weighted sums. At |&theta;| &asymp; 0.235, raw exponentials overflow f32 entirely. The implementation switches to log-domain LSE for |&theta;| &gt; 0.15, which is numerically stable but roughly 2.5&times; slower due to the transcendental function calls.

### Near-Origin Approximations

Near &theta; = 0, the mean and standard deviation are well-approximated by quadratics:

:::equation
mean(&theta;) &asymp; 247.0 &minus; 0.9&theta; &minus; 618&theta;<sup>2</sup>
:::

:::equation
std(&theta;) &asymp; 39.5 + 53.5&theta; &minus; 17&theta;<sup>2</sup>
:::

Mean loss is quadratic (second-order around the optimum, as expected from perturbing a maximum). Standard deviation gain is linear in &theta; (dominant term: 53.5&theta;). This asymmetry (quadratic cost, linear benefit) creates the narrow window where tail gains are cheap.

### Win Probability and Variance

In a symmetric two-player game where both players use the same policy, the win probability is exactly 50% minus half the draw rate. With the EV-optimal policy, the draw rate is 0.76%, giving each player a 49.62% win probability. The winning margin distribution has mean 43.5 points and heavy right tail; blowout wins of 80+ points are common, reflecting the high variance (&sigma; &asymp; 38) of individual game scores.

For a risk-seeking player (&theta; = +0.07) facing an EV-optimal opponent, the win rate shifts modestly. The increased variance is a double-edged sword: p95 rises to 313 (+4 vs EV-optimal) but p5 drops to 159 (&minus;20). The net effect on win probability depends on whether the opponent's score distribution concentrates in the region where the risk-seeker's added upside exceeds the added downside.

### Disagreement Anatomy

At &theta; = 0.07 versus &theta; = 0, category disagreements are concentrated in the upper section. The risk-seeking solver is more willing to take a zero in a lower-value upper category (Ones, Twos) to preserve the option of chasing Yatzy or Full House later. Reroll disagreements centre on keep decisions where a safe pair competes with a speculative straight draw. The 12.5% reroll disagreement rate translates to roughly 5–6 different keep decisions per 15-round game.

:::

:::code

### Implementation: Solver Dispatch

The three code paths correspond to the three computational regimes. The EV path uses direct weighted sums. The utility path computes the same weighted sum in exponential space without leaving f32. The LSE path works in log domain for numerical stability, at the cost of transcendental function calls. All three produce the same data structure: a strategy table mapping every reachable state to an optimal action.

```rust
// batched_solver.rs — solver dispatch by θ regime
match theta.classify() {
    ThetaClass::Zero     => solve_ev(buffers, ctx),      // ~1.1s
    ThetaClass::Utility  => solve_utility(buffers, ctx),  // |θ| ≤ 0.15, ~0.49s
    ThetaClass::LogDomain => solve_lse(buffers, ctx),     // |θ| > 0.15, ~2.7s
}
```

The utility-domain path is actually *faster* than EV because the NEON FMA kernels can fuse the utility weighting into the existing multiply-accumulate pipeline. The LSE path requires per-element `exp()` calls via the fast Cephes polynomial approximation in `neon_fast_exp_f32x4`, adding ~2 ULP error but keeping the solve under 3 seconds for any &theta;.

```rust
// Log-sum-exp at a chance node (simplified)
fn lse_chance_node(log_probs: &[f32], log_values: &[f32]) -> f32 {
    let terms: Vec<f32> = log_probs.iter()
        .zip(log_values)
        .map(|(lp, lv)| lp + lv)
        .collect();
    let m = terms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    m + terms.iter().map(|t| (t - m).exp()).sum::<f32>().ln()
}
```

The full &theta; sweep (37 values) is resumable: each strategy table is written to `data/strategy_tables/all_states_theta_*.bin` and skipped on subsequent runs if the file already exists. Deleting these files after changing solver code is mandatory.

:::

:::
