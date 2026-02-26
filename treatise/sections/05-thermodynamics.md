:::section{#part-iv}

:::part-title
Part IV
:::

## Thermodynamics of Risk

The EV-optimal solver answers one question: what maximises expected score? But
expected score is not always the right objective. A player trailing by 40 points
in the final rounds needs upside, not consistency. A player protecting a lead
needs the floor, not the ceiling. The question becomes: how do we parametrise the
tradeoff between expected value and variance?

The answer is ::concept[exponential utility]{exponential-utility}.
Instead of maximising the raw expected score E[<var>x</var>], the solver maximises
E[&minus;exp(&minus;&theta;<var>x</var>)], where &theta; is a single real-valued
parameter that controls risk attitude. When &theta; = 0, the exponential flattens
and the solver recovers exactly the EV-optimal policy. When &theta; &lt; 0, the
solver becomes risk-averse --it overweights bad outcomes and prefers
consistent scores. When &theta; &gt; 0, it becomes risk-seeking --it chases
high-variance plays that lift the ceiling at the cost of the floor.

This is the ::concept[CARA]{cara-utility}
(constant absolute risk aversion) model from decision theory. Its defining
property is that the risk premium for any gamble is independent of current
wealth --the solver's willingness to take a risky reroll depends only on
&theta; and the gamble's shape, never on how many points have already been
scored. This makes &theta; a clean, interpretable dial: turn it left for safety,
turn it right for ambition.

:::html
<div class="chart-container" id="chart-exponential-utility"><div id="chart-exponential-utility-svg"></div></div>
:::

The solver sweeps 37 values of &theta; from &minus;3.0 to +3.0, with dense
spacing near zero (19 values with &Delta;&theta; as small as 0.005) and sparser
coverage at the extremes. Each produces a complete strategy table via backward
induction, and each is evaluated over one million simulated games. The results
trace a ::concept[mean-variance frontier]{free-energy}
that reveals the price of risk in exact quantitative terms.

:::html
<div class="chart-container" id="chart-mean-variance-frontier"><div id="chart-mean-variance-frontier-svg"></div></div>
:::

The frontier is not a single curve. It splits into two branches. The risk-averse
branch (&theta; &lt; 0) compresses variance into a narrow band of 35–38
while mean score drops from 248.4 to 198.2 at &theta; = &minus;1.0. The
risk-seeking branch (&theta; &gt; 0) first expands variance to a peak of 48.4
at &theta; &asymp; 0.2, then contracts it as the mean falls to 194.5 at
&theta; = +1.0. At the same mean of roughly 190, the two branches have
different standard deviations --risk-seeking play is fundamentally
noisier than risk-averse play at matched expected value.

The practical sweet spots are narrow. The best &theta; for protecting the 5th
percentile (the floor) is &minus;0.03, which lifts p5 to 183 --four
points above the EV-optimal floor of 179. The best &theta; for the 95th
percentile (the ceiling) is +0.07, pushing p95 to 313 versus 309 at &theta; = 0.
The best for p99 is +0.10, reaching 329. These gains are small but real, and
they come at a precise, quantifiable cost in expected value.

:::html
<div class="chart-container" id="chart-risk-reward"><div id="chart-risk-reward-svg"></div></div>
:::

There is a deep analogy to statistical mechanics. The
::concept[log-sum-exp]{log-sum-exp} operation
that replaces weighted averaging in the risk-sensitive Bellman equation is
identical to the computation of
::concept[free energy]{free-energy} in a
thermodynamic partition function. The parameter &theta; plays the role of
inverse temperature: at &theta; = 0, all outcomes are weighted equally
(infinite temperature); as |&theta;| grows, the solver increasingly
concentrates on extreme outcomes (low temperature). The mean-variance frontier
is the equation of state. The two-branch structure is a phase diagram.

:::html
<div class="chart-container" id="chart-free-energy-phase"><div id="chart-free-energy-phase-svg"></div></div>
:::

:::depth-2

### The Exponential Utility Framework

The utility function is <var>u</var>(<var>x</var>) = &minus;exp(&minus;&theta;<var>x</var>).
The certainty equivalent --the deterministic score that provides equal
utility to the random outcome --is:

:::equation
CE = (1/&theta;) &middot; ln(E[exp(&theta; &middot; total)])
:::

For &theta; &gt; 0, CE &gt; E[total]: the risk-seeker values a gamble above
its expected value. For &theta; &lt; 0, CE &lt; E[total]: the risk-averter
discounts the gamble below its expectation. The gap CE &minus; E[total] is
the risk premium, and it scales with both &theta; and the variance of the
outcome distribution.

### Log-Domain Computation

Direct computation of exp(&theta; &middot; score) overflows for moderate
&theta; and typical scores around 250. The solver works entirely in the
log domain, storing <var>L</var>(<var>S</var>) = ln(E[exp(&theta; &middot; total) | <var>S</var>]).
The chance-node recurrence becomes a log-sum-exp:

:::equation
<var>L</var>(keep) = LSE<sub>r</sub> { ln(<var>P</var>(keep &rarr; r)) + <var>L</var>(r) }
:::

where LSE is computed with the standard numerical stability trick:
LSE(<var>x</var><sub>1</sub>, &hellip;, <var>x</var><sub>n</sub>) =
<var>m</var> + ln(&Sigma; exp(<var>x</var><sub>i</sub> &minus; <var>m</var>)),
with <var>m</var> = max(<var>x</var><sub>i</sub>).

### f32 Numerical Stability

The solver uses f32 throughout for performance. This creates a numerical
stability boundary. At |&theta;| &le; 0.15, the exponential utility can be
computed in the "utility domain" --direct weighted averaging ---
with the same speed as the EV solver. Beyond |&theta;| &asymp; 0.166,
mantissa erasure in f32 subtraction corrupts the weighted sums. At
|&theta;| &asymp; 0.235, raw exponentials overflow f32 entirely. The
implementation switches to log-domain LSE for |&theta;| &gt; 0.15, which
is numerically stable but roughly 2.5× slower due to the transcendental
function calls.

### Near-Origin Approximations

Near &theta; = 0, the mean and standard deviation are well-approximated by
quadratics:

:::equation
mean(&theta;) &asymp; 247.0 &minus; 0.9&theta; &minus; 618&theta;<sup>2</sup>
:::

:::equation
std(&theta;) &asymp; 39.5 + 53.5&theta; &minus; 17&theta;<sup>2</sup>
:::

Mean loss is quadratic (second-order around the optimum, as expected from
perturbing a maximum). Standard deviation gain is linear in &theta;
(dominant term: 53.5&theta;). This asymmetry --quadratic cost, linear
benefit --creates the narrow window where tail gains are cheap.

:::

:::depth-3

### Implementation: Solver Dispatch

The batched solver dispatches to one of three code paths based on &theta;:

```rust
match theta.classify() {
    ThetaClass::Zero     => solve_ev(buffers, ctx),      // ~1.1s
    ThetaClass::Utility  => solve_utility(buffers, ctx),  // |θ| ≤ 0.15, ~0.49s
    ThetaClass::LogDomain => solve_lse(buffers, ctx),     // |θ| > 0.15, ~2.7s
}
```

The utility-domain path is actually *faster* than EV because the
NEON FMA kernels can fuse the utility weighting into the existing multiply-
accumulate pipeline. The LSE path requires per-element `exp()`
calls via the fast Cephes polynomial approximation in
`neon_fast_exp_f32x4`, adding ~2 ULP error but keeping the
solve under 3 seconds for any &theta;.

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

The full &theta; sweep (37 values) is resumable: each strategy table is
written to `data/strategy_tables/all_states_theta_*.bin` and
skipped on subsequent runs if the file already exists. Deleting these files
after changing solver code is mandatory.

:::

:::
