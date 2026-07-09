:::section{#risk-parameter}

## The Risk Parameter

We defined the optimal strategy as "maximize the expected score." Such a strategy will get the highest possible *average* score. But, reality is, most people don't play Yatzy to get a good average. We like to play to win! 😜 We all remember that one time when all stars aligned and we got 322 points, and want to experience it again! So, in that spirit, let's see if we can design a solver that takes more risk.

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

**Use the slider below to see how the score distribution changes across the &theta; range.** At &theta; = 0 the EV-optimal distribution is shown. As you decrease &theta; the distribution compresses with lower mean and less variance. As &theta; is increased, variance increases, pushing the tails outward before collapsing at extreme values. The tail probabilities displayed below the curve show how outcomes for different score thresholds compare to &theta; = 0. The ::concept[mean-variance frontier]{free-energy} at the bottom traces where each &theta; lands in risk-return space.

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

**In the chart below you see what &theta; is optimal for different percentiles.** One conclusion is that a sligth increase in risk-taking might be worth it for most players. p75 is bumped a few points for &theta; = 0.04. At that level, you are 20% more likely to get a score above 310, but only pay a small price for it in the form of a slightly lower mean score.

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

Here's some help on how to read the graph below: The x-axis is the target score. The y-axis shows how many games you need to play to reach that score for a certain theta. As you can see, theta = 0.5 is "worse", i.e. requires more games, to reach scores up to 333. It's only the best policy when targeting scores of 348 and above. The only problem with this insight is that it only matters if you are able to play around 2000 games, and it then shows up as a single, mindblowing score.

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
CE = \frac{1}{\theta} \cdot \ln E[e^{\theta \cdot \text{total}}]
:::

For &theta; &gt; 0, CE &gt; E[total]: the risk-seeker values a gamble above its expected value, because the chance of a high score is worth something on its own. For &theta; &lt; 0, CE &lt; E[total]: the risk-averter would accept fewer guaranteed points to avoid the chance of a disaster. The gap CE &minus; E[total] is the risk premium, and it scales with both &theta; and the variance of the outcome distribution.

This framework is known as ::concept[Constant Absolute Risk Aversion]{cara-utility} (CARA) in decision theory. Its defining property: the solver's risk attitude depends only on &theta; and the shape of the gamble, never on how many points have already been scored. Scoring 100 in the first five rounds does not make the solver more reckless in round six.

### From Weighted Sum to Log-Sum-Exp

The EV solver and the &theta; solver have the same structure. Both use probability-weighted averaging at chance nodes and argmax at decision nodes. The only difference is *what gets averaged*.

At a chance node, the EV solver computes a weighted sum of raw state values:

:::equation
E(\text{keep}) = \sum_{r} P(\text{keep} \to r) \cdot V(r)
:::

The &theta; solver computes the same weighted sum, but of *exponentiated* state values:

:::equation
\exp(L(\text{keep})) = \sum_{r} P(\text{keep} \to r) \cdot \exp(L(r))
:::

where <var>L</var>(<var>S</var>) = ln E[e<sup>&theta; &middot; total</sup> | <var>S</var>]. The exponential e<sup>&theta;<var>x</var></sup> amplifies extreme values before the averaging happens. High scores contribute exponentially more when &theta; &gt; 0, low scores when &theta; &lt; 0.

Direct computation of exp(&theta; &middot; score) overflows for moderate &theta; and typical scores around 250. The solver works entirely in the log domain, where this weighted sum becomes a log-sum-exp (LSE):

:::equation
L(\text{keep}) = \text{LSE}_{r} \{ \ln(P(\text{keep} \to r)) + L(r) \}
:::

LSE is computed with the standard numerical stability trick: LSE(<var>x</var><sub>1</sub>, &hellip;, <var>x</var><sub>n</sub>) = <var>m</var> + ln(&Sigma; exp(<var>x</var><sub>i</sub> &minus; <var>m</var>)), with <var>m</var> = max(<var>x</var><sub>i</sub>).

At decision nodes, the solver picks the action that maximizes <var>L</var> when &theta; &gt; 0, or minimizes it when &theta; &lt; 0 (because dividing CE = <var>L</var>/&theta; by a negative &theta; flips the direction).

### Policy Degeneration at Large |&theta;|

When |&theta;| is large, the exponential amplification becomes so extreme that only the single best (or worst) outcome matters. The LSE at a chance node approaches a plain max:

:::equation
\text{LSE} \approx \max_{i}(x_{i}) \quad \text{when} \quad |\theta| \cdot \sigma \gg 1
:::

where &sigma; is the spread of state values at that node. The transition depends on a dimensionless control parameter: the product |&theta;| &middot; &sigma;<sub>node</sub>. In Yatzy, the typical per-node score spread is &sigma;<sub>node</sub> &asymp; 10, giving three regimes:

| |&theta;| | |&theta;| &middot; &sigma; | Regime |
|-------:|------------------:|--------|
| &lt; 0.03 | &lt; 0.3 | Near-EV: policy barely differs from &theta; = 0 |
| 0.03&ndash;0.20 | 0.3&ndash;2 | Active: meaningful risk/return tradeoffs |
| 0.20&ndash;1.5 | 2&ndash;15 | Extreme: policy still shifting (disagreement vs &theta; = 0 grows from 32% at &theta; = 0.5 to 44% at &theta; = 3) |
| &gt; 1.5 | &gt; 15 | Degenerate: LSE &asymp; max, policy effectively frozen |

The marginal utility chart tells the same story from the score side: at |&theta;| = 0.5, the marginal value is effectively zero everywhere except the extreme tail. The solver is running a soft-max over outcomes, concentrating all its attention on the single best reachable score. Increasing |&theta;| further cannot change which action achieves that maximum, so the policy stops changing.

The extreme regime does reach higher ceilings: the observed maximum rises from 354 at &theta; = 0 (1M games) to 362 at &theta; &asymp; 0.1 (200K games). But it pays for those ceilings by sacrificing everything else (at &theta; = 3, the Full House hit rate drops from 92% to 54% and the mean score from 248 to about 186). The best games are spectacular but vanishingly rare: the ceiling-achieving games become so improbable that even a million rolls cannot reliably sample them. In the limit, the max-policy converges to the theoretical maximum of the game, but with a probability on the order of 10<sup>&minus;19</sup>, far beyond what any feasible simulation could observe.

### f32 Numerical Stability

The solver uses f32 throughout for performance. This creates a numerical stability boundary. At |&theta;| &le; 0.15, the exponential utility can be computed in the "utility domain" via direct weighted averaging, with the same speed as the EV solver. Beyond |&theta;| &asymp; 0.166, mantissa erasure in f32 subtraction corrupts the weighted sums. At |&theta;| &asymp; 0.235, raw exponentials overflow f32 entirely. The implementation switches to log-domain LSE for |&theta;| &gt; 0.15, which is numerically stable but roughly 2.5&times; slower due to the transcendental function calls.

### Near-Origin Approximations

Near &theta; = 0, the mean and standard deviation are well-approximated by quadratics:

:::equation
\text{mean}(\theta) \approx 247.0 - 0.9\theta - 618\theta^{2}
:::

:::equation
\text{std}(\theta) \approx 39.5 + 53.5\theta - 17\theta^{2}
:::

Mean loss is quadratic (second-order around the optimum, as expected from perturbing a maximum). Standard deviation gain is linear in &theta; (dominant term: 53.5&theta;). This asymmetry (quadratic cost, linear benefit) creates the narrow window where tail gains are cheap.

### The Mean-Variance Frontier and the Marginal Rate of Substitution

The ::concept[mean-variance frontier]{mean-variance-frontier} in the chart above traces where each &theta; lands in (standard deviation, mean) space. Each &theta; table maximizes a specific certainty equivalent. Using the cumulant generating function K(&theta;) = ln E[e<sup>&theta;X</sup>], the CE expands as:

:::equation
CE_{\theta} \approx E + \frac{\theta}{2} \cdot \text{Var} + \frac{\theta^{2}}{6} \cdot \kappa_{3} + \ldots
:::

where &kappa;<sub>3</sub> is the third cumulant (skewness). For small |&theta;|, the CE is dominated by the first two terms. Each &theta; table maximizes this CE, and on the efficient frontier CE is locally constant (you cannot increase one term without decreasing the other). Taking the total derivative and setting dCE = 0:

:::equation
dE + \frac{\theta}{2} \cdot dV = 0 \Rightarrow \frac{\partial E}{\partial V} = -\frac{\theta}{2}
:::

This is the **marginal rate of substitution** (MRS): the exact price, in expected points, that the solver pays for each unit of variance. Running the solver with &theta; = 0.10 finds the point on the frontier where the tangent slope is &minus;0.05: the solver accepts a drop of 1 mean point for every 20 units of variance gained. At &theta; = 0 the slope is zero (the peak: maximum EV, variance ignored). At &theta; = &minus;0.10 the slope is +0.05: the solver sacrifices mean to crush variance.

**Where the approximation breaks down.** The MRS formula &part;E/&part;V = &minus;&theta;/2 is a first-order result. It holds well for |&theta;| &lt; 0.05, the "active" regime where the &theta;<sup>2</sup> skewness term is small. Beyond that range, two effects distort the frontier:

1. **Skewness.** Yatzy scores have positive skewness (heavy right tail from Yatzy category and straights). For &theta; &gt; 0, the &kappa;<sub>3</sub> term gives the solver "free" CE from the right tail, so it trades less mean per unit of variance than &minus;&theta;/2 predicts. For &theta; &lt; 0, the skewness penalty compounds with the variance penalty.

2. **Phase transitions.** Around &theta; &asymp; &minus;0.15, the solver abruptly abandons the 50-point upper bonus (the expected value of chasing it no longer justifies the variance cost under strong risk aversion). This creates a discontinuous jump in the frontier that no smooth tangent line can capture.

These effects explain the two-branch structure visible in the frontier chart: the risk-averse branch (&theta; &lt; 0) and the risk-seeking branch (&theta; &gt; 0) follow different paths in mean-variance space because they activate different strategic levers (eliminating downside vs. chasing upside) with different skewness profiles. The MRS formula accurately predicts the local geometry near &theta; = 0, and the multiplayer section uses this relationship to derive the optimal adaptive risk parameter for head-to-head play.

### Win Probability and Variance

In a symmetric two-player game where both players use the same policy, the win probability is exactly 50% minus half the draw rate. With the EV-optimal policy, the draw rate is 0.76%, giving each player a 49.62% win probability. The winning margin distribution has mean 43.5 points and heavy right tail; blowout wins of 80+ points are common, reflecting the high variance (&sigma; &asymp; 38) of individual game scores.

For a risk-seeking player (&theta; = +0.07) facing an EV-optimal opponent, the win rate shifts modestly. The increased variance is a double-edged sword: p95 rises to 313 (+4 vs EV-optimal) but p5 drops to 159 (&minus;20). The net effect on win probability depends on whether the opponent's score distribution concentrates in the region where the risk-seeker's added upside exceeds the added downside.

### Disagreement Anatomy

At &theta; = 0.07 versus &theta; = 0, category disagreements are concentrated in the upper section. The risk-seeking solver is more willing to take a zero in a lower-value upper category (Ones, Twos) to preserve the option of chasing Yatzy or Full House later. Reroll disagreements centre on keep decisions where a safe pair competes with a speculative straight draw. The 12.5% reroll disagreement rate translates to roughly 4 different keep decisions per 15-round game (about 5.6 changed decisions per game when category picks are included).

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

**Domain boundary artifact.** The utility and LSE paths are mathematically equivalent but accumulate f32 rounding differently, so the optimal action can disagree at a handful of states near the |&theta;|&nbsp;=&nbsp;0.15 boundary. The effect is small (order 1 point on mean score) and comparable to the 22-state f32/f64 divergence measured for the EV path.

:::

:::
