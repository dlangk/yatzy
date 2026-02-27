:::section{#risk-parameter}

## Adding a Risk Parameter

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

### Multiplayer and Adaptive Strategies

Everything so far treats Yatzy as a solitaire optimisation problem: one player,
one score, maximise it. But Yatzy is usually played against opponents, and in
that setting the right question changes. It is no longer "what maximises my
expected score?" but "what maximises my probability of winning?"

Against an EV-optimal opponent, the EV-optimal policy wins exactly half the
time by symmetry. But a slightly risk-seeking policy --&theta; around
+0.05 to +0.07 --can gain an edge. The mechanism is straightforward:
risk-seeking play increases variance, which pushes more probability mass into
both tails. Since games are decided by the higher score, the increased upside
(p95 rising from 309 to 313) matters more than the increased downside when
trailing. The draw rate in head-to-head play is just 0.76%, with the typical
winning margin at 43.5 points.

The disagreement between &theta; = 0.07 and &theta; = 0 reveals where risk
tolerance changes decisions. In 12.5% of reroll situations and 12.4% of
category selections, the two policies diverge. These disagreements cluster
around two patterns. First,
::concept[Yatzy preservation]{backward-induction}:
at &theta; &asymp; 0.05–0.07, the solver keeps Yatzy-seeking dice
combinations in 42.8% of relevant states versus 38.8% at &theta; = 0 ---
the risk-seeking policy holds out for the 50-point jackpot more often. Second,
straight chasing: the risk-seeking solver is more willing to break a safe pair
to chase a Large Straight, accepting the lower hit rate for the higher payoff.

:::html
<div class="chart-container" id="chart-adaptive-winrate"><div id="chart-adaptive-winrate-svg"></div></div>
:::

The real insight is that the optimal &theta; depends on game state. A player
who is behind should increase &theta; to generate variance; a player who is
ahead should decrease it to protect the lead. This leads naturally to a
::concept[threshold policy]{threshold-policy}:
estimate the score differential at each decision point and select the strategy
table (indexed by &theta;) that maximises win probability given the current
gap. The adaptive policy does not require solving a new DP --it simply
switches between pre-computed strategy tables based on a score-differential
threshold.

The cost of risk-seeking is visible in category hit rates. The EV-optimal
solver achieves a 92% Full House hit rate across one million games. At
&theta; = +3.0, this collapses to 54% --the solver sacrifices reliable
mid-range categories to chase Yatzy and straights. The upper bonus achievement
rate at &theta; = 0 is 83.6%, and it declines monotonically as &theta;
increases. Both extremes converge to similarly poor mean scores: &theta; =
&minus;1.0 yields 198.2, and &theta; = +1.0 yields 194.5. Excessive caution
is as costly as excessive ambition.

:::html
<div class="chart-container" id="chart-threshold-policy"><div id="chart-threshold-policy-svg"></div></div>
:::

:::math

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

### Win Probability and Variance

In a symmetric two-player game where both players use the same policy, the
win probability is exactly 50% minus half the draw rate. With the EV-optimal
policy, the draw rate is 0.76%, giving each player a 49.62% win probability.
The winning margin distribution has mean 43.5 points and heavy right tail
--- blowout wins of 80+ points are common, reflecting the high variance
(&sigma; &asymp; 38) of individual game scores.

For a risk-seeking player (&theta; = +0.07) facing an EV-optimal opponent,
the win rate shifts modestly. The increased variance is a double-edged sword:
p95 rises to 313 (+4 vs EV-optimal) but p5 drops to 159 (&minus;20). The
net effect on win probability depends on whether the opponent's score
distribution concentrates in the region where the risk-seeker's added
upside exceeds the added downside.

### Disagreement Anatomy

At &theta; = 0.07 versus &theta; = 0, category disagreements are concentrated
in the upper section. The risk-seeking solver is more willing to take a zero
in a lower-value upper category (Ones, Twos) to preserve the option of
chasing Yatzy or Full House later. Reroll disagreements centre on keep
decisions where a safe pair competes with a speculative straight draw. The
12.5% reroll disagreement rate translates to roughly 5–6 different keep
decisions per 15-round game.

:::

:::code

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

### Adaptive Policy Implementation

The adaptive policy loads multiple strategy tables into memory (one per
&theta; value) and selects at runtime based on estimated score differential:

```rust
// Adaptive θ selection based on score gap
fn select_theta(my_score: f32, opp_estimate: f32, turns_left: u8) -> f32 {
    let gap = my_score - opp_estimate;
    let urgency = 1.0 - (turns_left as f32 / 15.0);

    if gap < -30.0 * (1.0 - urgency) {
        0.10  // Far behind: maximum risk-seeking
    } else if gap < -10.0 {
        0.05  // Moderately behind: mild risk-seeking
    } else if gap > 30.0 {
        -0.03 // Far ahead: protect the lead
    } else {
        0.0   // Close game: EV-optimal
    }
}
```

The strategy tables are memory-mapped (~16 MB each), so switching between
them is a pointer swap with no I/O cost. The score-gap thresholds are
derived from the mean-variance frontier: the 30-point threshold corresponds
to roughly 0.8&sigma; of the score distribution, the point where the
probability of overtaking shifts meaningfully.

```bash
# Simulate head-to-head with adaptive policy
just simulate --games 1000000 --adaptive --opponent-theta 0.0
```

:::

:::
