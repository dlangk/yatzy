:::section{#multiplayer}

## Multiplayer and Adaptive Strategies

I think very few people play Yatzy by themselves 😅 It's possible I have during the course of this porject, but let's not talk about that.

But, even if you (like other sane people) mostly play with others, you likely make decisions *as if you played alone*. That makes sense. Two parallel Yatzy games are independent. The dice you roll do not affect the dice your opponents roll. You cannot do anything about the other players game.

But, it turns out, sequential play introduces a slight asymmetry. Let's assume standard format two player Yatzy. Player 1 goes first, Player 2 goes second. Each player takes turn, and after 15 turns, the game is over. This setup gives Player 2 the ability to observe Player 1's score, category and upper-section progress before making decisions.

The obvious question then becomes: Can Player 2 use their knowledge of Player 1's game to improve their probability of *winning*?

**The short answer: yes, but only by 0.9% percentage points.** The obvious baseline is that if both players use the fixed, EV-optimal strategy, they each win 50% of games. Let's explore what limits how far we can push the win percentage for Player 2 by using information about Player 1's game.

:::html
<div class="chart-container" id="chart-game-replay">
  <div id="chart-game-replay-svg"></div>
  <p class="chart-caption">A curated game illustrating variance-scaled adaptation. Top: cumulative scores (gray: Player 1, colored: Player 2). Bottom: Player 2's risk parameter &theta; (bars) against the combined remaining variance (shaded area). Player 1 scores Yatzy on turn 10, opening a 61-point gap. &theta; spikes to its ceiling. Player 2 answers with their own Yatzy on turn 11. As the variance envelope shrinks toward zero, the same score gap produces increasingly aggressive &theta; adjustments.</p>
</div>
:::

### Adaptive Risk Taking

The mechanism by which a player can adapt to their opponent is by taking more or less risk. In practice, that means trading expected value for variance according to the frontier mapped out in Section 5. There, we introduced the ::concept[exponential utility]{cara-utility} U(x) = e<sup>&theta;x</sup>, where &theta; controls risk taking. We also showed how precomputed strategy tables across a grid of &theta; values produce diffrent points on the ::concept[mean-variance frontier]{mean-variance-frontier}.

The relationship (derived in the math section below) is the **marginal rate of substitution**: &part;E/&part;V = &minus;&theta;/2.

In practice, Player 2 does not know the opponent's final score. Player 2 can only observe Player 1's trajectory unfold turn by turn. From the visible state, Player 2 can estimate the opponent's expected final score via a state-value table lookup: E[final] = current_total + EV_remaining(state). The challenge is determining exactly how much risk to take based on the EV deficit (the opponent's expected final minus own expected final).

### Why Variance Matters

My first attempt was to scale risk (&theta;) linearly with the EV deficit, but you quickly realize that a 15-point deficit at turn 2 is just noise. There are 13 turns left, each adding uncertainty, so what we know at that point isn't particularly useful. The same deficit at turn 14, however, is very informative. If there are only one or two turns left, and the only way to win is e.g. getting a Yatzy, then you have nothing to loose from taking risk. So, to find the optimal amount of risk at each turn, we need to account for how much uncertainty remains.

**The key here is realizing that Player 2's goal is not to maximize their own score, or their average score. It is to maximize P(S<sub>2</sub> &gt; S<sub>1</sub>): the probability that their final score exceeds Player 1's.**

Because 15 turns of dice rolls produce many independent random events, final scores are approximately Gaussian. The *difference* S<sub>2</sub> &minus; S<sub>1</sub> is therefore also Gaussian, with mean E<sub>2</sub> &minus; E<sub>1</sub> and variance V<sub>1</sub> + V<sub>2</sub> (a standard result: variances add when subtracting independent random variables).

Player 2 wins when this difference is positive. For a Gaussian, the probability of exceeding zero is entirely determined by the ::concept[Z-score]{z-score}: Z = (E<sub>2</sub> &minus; E<sub>1</sub>) / &radic;(V<sub>1</sub> + V<sub>2</sub>). A higher Z-score means the center of the bell curve is further from the losing side, which means a larger probability of winning. Maximizing the Z-score is therefore identical to maximizing win probability.

Player 2 controls their own expected score and variance by choosing a &theta; table. Taking more risk (higher &theta;) increases variance but reduces expected score. There is a tension: increasing E<sub>2</sub> pushes the numerator up, but if it comes at the cost of inflating V<sub>2</sub>, the denominator grows and the Z-score can actually drop.

The calculus (detailed in the math section below) resolves this. By differentiating the Z-score with respect to V<sub>2</sub> and equating the result with the tables' built-in exchange rate between mean and variance (&part;E/&part;V = &minus;&theta;/2), we get:

:::equation
\theta^* = \frac{E_1 - E_2}{V_1 + V_2}
:::

This formula describes exactly how much risk to take. It is positive when trailing (seek risk), negative when leading (protect), and vanishes when scores are even. Crucially, it scales inversely with remaining uncertainty. Early in the game V<sub>1</sub> + V<sub>2</sub> exceeds 3000, suppressing &theta;* even for large deficits. Late in the game it drops below 500, amplifying small deficits into decisive risk adjustments. If the opponent is on the bubble for the 50-point upper bonus, their remaining variance V<sub>1</sub> is large; the formula commands patience. The moment the opponent locks or fails the bonus, V<sub>1</sub> collapses, &theta;* spikes, and the policy pivots.

Use the slider below to see how the same deficit produces different &theta; at different turns. The variance-scaled policy (solid) is conservative early and aggressive late. The linear policy (dashed) applies the same &theta; regardless of turn. The faded line shows unclamped &theta;*, which becomes completely degenerate towards the later turns.

:::html
<div class="chart-container" id="chart-risk-horizon">
  <div class="chart-controls chart-controls-center">
    <label>EV deficit = <span class="slider-value">+20</span> points</label>
    <input type="range" class="chart-slider" min="-30" max="50" value="20">
  </div>
  <div id="chart-risk-horizon-svg"></div>
  <p class="chart-caption">Risk parameter &theta; across 15 turns for a fixed EV deficit. Orange: variance-scaled (clamped). Gray dashed: linear policy. Faded: unclamped &theta;* before the safety clamp. Shaded zones mark where CARA utility degenerates.</p>
</div>
:::

The remaining variance is extracted in O(1) from two table lookups (see math section), and the raw &theta;* is clamped to [&minus;0.05, +0.07] to prevent the utility degeneration described in Section 5.

### The Theoretical Ceiling

Before reporting results, let's establish the theoretical limits. How much advantage can Player 2 possibly extract?

**The constant-&theta; oracle.** As a first bound, suppose Player 2 knew the opponent's final score before the game began, and could pick the single best &theta; for that outcome. The chart below splits opponent final scores into seven bands and shows, for each band, the win rate of EV-optimal play alongside the win rate of the best possible fixed &theta; for that band.

:::html
<div class="chart-container" id="chart-oracle-winrate">
  <div id="chart-oracle-winrate-svg"></div>
  <p class="chart-caption">Conditional win rates per opponent score band. Gray: EV-optimal (&theta; = 0). Colored: best fixed &theta; for each band, colored by the optimal &theta; value (blue = conservative, orange = risk-seeking).</p>
</div>
:::

When the opponent scored below 200, the oracle plays conservatively (&theta; = &minus;0.03). We know we can reduce variance by reducing risk, and if the loss in mean is less than the variance, and we know the final score of the other player, we can improve our chances of winning. Reversely, if the opponent scored above 280, the Oracle will be more risk-seeking (&theta; = +0.04). That's because we know it will take a tail-outcome to beat Player 1. The transition from negative to positive &theta; happens near the mean (~250), where the two distributions overlap most. The overall oracle win rate is **50.6% H2H**: a gain of +0.60 percentage points over EV-optimal.

**The unconstrained clairvoyant.** The constant-&theta; oracle is limited to selecting one precomputed &theta; table per game. A stronger bound comes from solving a different game entirely: Player 1 plays all 15 turns in isolation and writes their final score on a piece of paper. Player 2 then plays knowing exactly what number they must beat.

This is a fundamentally different format from interleaved play, but *information relaxation* makes it a rigorous ceiling. First, Player 1 uses the EV-optimal policy and ignores Player 2, so turn order does not affect Player 1's score distribution. To Player 2, Player 1 is not a reactive opponent; they are a complex random number generator. Second, more information never hurts: a policy with perfect future information always performs at least as well as one with partial present information. Third, revealing the final score collapses the state space. The exact turn-by-turn optimum requires a joint-state MDP with approximately 164 trillion states. By revealing the final score, Player 1's game tree reduces to a single target number T, shrinking the problem to 1.43 million states &times; 384 possible targets, solvable in under 6 minutes.

We solved a 3D backward-induction DP over the state space (upper_score, scored_categories, target_score_needed). For each reachable state and target score, the solver maximizes P(final_score &gt; T). Unlike the &theta; tables, the optimal category depends on the target: at T = 200, the solver prefers safe scoring; at T = 300, it makes entirely different trade-offs. No continuous utility function can express this step-function logic, where winning by 1 point is exactly as valuable as winning by 100.

The clairvoyant ceiling is **55.3% H2H**: a gain of +5.3 percentage points. The 4.7pp gap between the constant-&theta; oracle (50.6%) and the clairvoyant (55.3%) isolates the structural cost of continuous proxy utilities. Both have perfect information; the difference is purely in the expressiveness of their decision-making.

### Turn-by-Turn Adaptation

With the theoretical bounds established, we can now evaluate practical strategies. We tested five adaptive policies at 10 million games each, alongside the two theoretical bounds:

| Strategy | H2H Win Rate | Net Advantage |
|----------|-------------|---------------|
| EV-optimal baseline | 50.04% | 0.00pp |
| Seek-only (linear) | 50.44% | +0.41pp |
| Protect-only (linear) | 50.20% | +0.16pp |
| Symmetric (linear) | 50.57% | +0.53pp |
| Constant-&theta; oracle | 50.6% | +0.60pp |
| **Variance-scaled [&minus;0.05, +0.07]** | **50.9%** | **+0.86pp** |
| Unconstrained clairvoyant | 55.3% | +5.3pp |

The linear policies scale &theta; proportionally with the EV deficit but ignore remaining variance. Risk-seeking when trailing contributes roughly 75% of the advantage; protecting when leading adds the remaining 25%. We also tested "Hail Mary" variants that play EV-optimal until the last 2-3 turns then spike &theta;. Every such variant performed worse than the linear policies: under large &theta;, the exponential utility becomes a max-score chaser rather than a target optimizer.

The variance-scaled policy surpasses the constant-&theta; oracle despite having less information, because it pays the mean-score penalty only on turns where &theta; deviates from zero. It is 2x more conservative than the symmetric policy in early turns and 2x more aggressive in late turns, with the crossover at turn 8. A parameter sweep over 9 clamp configurations confirmed the landscape is flat above +0.07 on the seek side and insensitive to the protect clamp.

**The stochastic washout effect.** Even with clairvoyant knowledge and unconstrained decisions, the maximum edge is ~5.3 percentage points. Under realistic sequential observation, it collapses to +0.86pp: fewer than 9 extra wins per 1000 games. The aleatoric entropy of 15 turns of dice rolls overwhelmingly dominates the strategic bandwidth of reactive play. The true sequential optimum lies somewhere in [50.9%, 55.3%], and our best heuristic captures the lower end of this range.

:::math

### The Marginal Rate of Substitution

The certainty equivalent of the exponential utility is CE<sub>&theta;</sub> = K(&theta;)/&theta;, where K(&theta;) = ln E[e<sup>&theta;X</sup>] is the cumulant generating function. Expanding via the Maclaurin series:

:::equation
CE_{\theta} \approx E[X] + \frac{\theta}{2} \cdot \text{Var}(X) + \frac{\theta^2}{6} \cdot \kappa_3(X) + \ldots
:::

where &kappa;<sub>3</sub> is the third cumulant (third central moment). Each precomputed &theta; table maximizes this CE. On the efficient frontier, CE is locally constant: improving one term requires degrading another. Taking the total derivative (setting dCE = 0):

:::equation
dE + \frac{\theta}{2} \cdot dV = 0 \Rightarrow \frac{\partial E}{\partial V} = -\frac{\theta}{2}
:::

This is the marginal rate of substitution on the mean-variance frontier. Running the solver with a specific &theta; finds the point on the frontier where the tangent line has slope &minus;&theta;/2.

### The Z-Score Derivation

**Setup.** Model final scores S<sub>1</sub>, S<sub>2</sub> as independent distributions with means E<sub>i</sub> and variances V<sub>i</sub>. Their difference S<sub>2</sub> &minus; S<sub>1</sub> has mean E<sub>2</sub> &minus; E<sub>1</sub> and variance V<sub>1</sub> + V<sub>2</sub> (variances add for independent variables). Under the Gaussian approximation, the probability that Player 2 wins is:

:::equation
P(S_2 > S_1) = \Phi(Z) \quad \text{where} \quad Z = \frac{E_2 - E_1}{\sqrt{V_1 + V_2}}
:::

Since &Phi; is monotonically increasing, maximizing Z is equivalent to maximizing win probability.

**The optimization.** Player 2 controls (E<sub>2</sub>, V<sub>2</sub>) by choosing a &theta; table. Player 1's (E<sub>1</sub>, V<sub>1</sub>) are fixed. Taking dZ/dV<sub>2</sub> = 0 via the quotient rule:

:::equation
\frac{dZ}{dV_2} = \frac{\frac{\partial E_2}{\partial V_2} \cdot \sqrt{V_1 + V_2} - \frac{E_2 - E_1}{2\sqrt{V_1 + V_2}}}{V_1 + V_2} = 0
:::

The numerator vanishes when:

:::equation
\frac{\partial E_2}{\partial V_2} = \frac{E_2 - E_1}{2(V_1 + V_2)}
:::

**Connecting to the &theta; tables.** From the MRS derived above, each &theta; table satisfies &part;E/&part;V = &minus;&theta;/2 at its operating point. Substituting:

:::equation
-\frac{\theta^*}{2} = \frac{E_2 - E_1}{2(V_1 + V_2)}
:::

The factors of 2 cancel, yielding:

:::equation
\theta^* = \frac{E_1 - E_2}{V_1 + V_2}
:::

**Interpretation.** The formula is positive when trailing (E<sub>1</sub> &gt; E<sub>2</sub>), commanding risk-seeking play. It is negative when leading, commanding conservative play. It vanishes when scores are even. The denominator is the key: the same point deficit produces a small &theta;* early (V<sub>1</sub> + V<sub>2</sub> &gt; 3000) and a large &theta;* late (V<sub>1</sub> + V<sub>2</sub> &lt; 500).

### The O(1) Variance Extraction

The cumulant generating function K(&theta;) = ln E[e<sup>&theta;X</sup>] expands as:

:::equation
K(\theta) = \theta \mu + \frac{\theta^2}{2} \cdot \sigma^2 + \frac{\theta^3}{6} \cdot \kappa_3 + \ldots
:::

The solver stores L(S) = K(&theta;) as state values. For &theta; = 0, L/&theta; &rarr; &mu;, so sv<sub>0</sub> = E[X]. For &theta; = 0.01:

:::equation
sv_{0.01} / 0.01 = \mu + 0.005 \cdot \sigma^2 + O(\theta^2)
:::

Subtracting the baseline:

:::equation
V(\text{state}) = 200 \cdot (sv_{0.01} / 0.01 - sv_0)
:::

The cubic residual contributes &theta;<sup>2</sup>/6 &middot; &kappa;<sub>3</sub> &asymp; 0.000017 &middot; &kappa;<sub>3</sub>, less than 1% of the variance term in the Yatzy score range. Verified: for the initial state (no categories scored), this yields V = 1482, matching the exact density standard deviation of 38.5 (38.5<sup>2</sup> = 1482.25).

### Why Continuous Utilities Are Target-Blind

Under CARA utility, the optimal policy maximizes E[e<sup>&theta; &middot; total</sup>], equivalent to maximizing:

:::equation
CE = \mu + \frac{\theta}{2} \cdot \sigma^2 + \frac{\theta^2}{6} \cdot \kappa_3 + \ldots
:::

This is a smooth, monotone function of the score. It assigns strictly positive marginal value to every additional point regardless of whether that point helps beat a specific target. The head-to-head objective P(score &gt; T) has a discontinuous reward: a step function at T. No member of the exponential family can approximate this step function.

The 3D threshold tablebase avoids this pathology by optimizing P(score &gt; T) directly for each T. The 4.7pp action gap (constant-&theta; oracle at 50.6% vs. unconstrained clairvoyant at 55.3%) quantifies the exact cost of this structural mismatch.

### Clairvoyant as Upper Bound

The clairvoyant ceiling (55.3% H2H) is a strict upper bound on the true sequential win rate. Knowing the opponent's final score is strictly more information than observing turn-by-turn progress. The true sequential optimum therefore lies in [50.9%, 55.3%].

The full sequential MDP has state space (C<sub>1</sub>, U<sub>1</sub>, C<sub>2</sub>, U<sub>2</sub>, &Delta;), where C<sub>i</sub> is the 15-bit scored-categories mask, U<sub>i</sub> is the upper-section progress, and &Delta; is the running score difference. The reachable joint states total approximately 164 trillion. Solving this exactly is intractable, but the bracketing [50.9%, 55.3%] places the answer within a 4.4pp window.

:::

:::code

### Implementation: Variance-Scaled Policy

The variance-scaled policy loads the &theta; = 0 and &theta; = 0.01 tables at startup. Each turn, it computes the EV deficit and remaining variance for both players via O(1) table lookups, then selects the nearest precomputed &theta; table.

```rust
// strategy.rs — variance-scaled risk selection
fn select_theta_index(&self, view: &GameView, tables: &[ThetaTable]) -> usize {
    if view.turn <= 1 { return self.ev_idx; }  // too early for signal
    let ev_sv = tables[self.ev_idx].sv.as_slice();
    let var_sv = tables[self.var_idx].sv.as_slice();

    // Expected final scores (O(1) table lookups)
    let my_e = my.total_score as f32 + ev_sv[state_index(my.upper_score, my.scored)];
    let opp_e = opp.total_score as f32 + ev_sv[state_index(opp.upper_score, opp.scored)];

    // Remaining variance via cumulant expansion: V ≈ 200·(CE_0.01 − E_0)
    let my_v = 200.0 * (var_sv[my_si] / 0.01 - ev_sv[my_si]);
    let opp_v = 200.0 * (var_sv[opp_si] / 0.01 - ev_sv[opp_si]);

    // θ* = deficit / total_variance, clamped to safe range
    let deficit = opp_e - my_e;
    let theta_raw = deficit / (my_v + opp_v).max(1.0);
    let theta_clamped = theta_raw.clamp(-0.05, 0.07);

    // Snap to nearest precomputed table
    tables.iter().enumerate()
        .min_by(|(_, a), (_, b)| /* nearest theta */)
        .map(|(i, _)| i).unwrap()
}
```

### Implementation: 3D Clairvoyant Solver

The clairvoyant solver extends the standard scalar solver by replacing each state's single float with a 384-element array: V[state][t] = P(remaining_score &gt; t | optimal play for target t).

```rust
// clairvoyant.rs — 3D Group 6 (simplified)
// For each dice set, choose the category that maximizes win probability
// at EACH target independently (element-wise max over categories).
for ds_i in 0..252 {
    let mut best = [0.0f32; 384];
    for c in available_categories {
        let score = precomputed_scores[ds_i][c];
        let next_si = state_index(new_upper(c), scored | (1 << c));
        for t in 0..384 {
            let val = if t < score { 1.0 }  // already exceeded target
                      else { v_table[next_si][t - score] };
            if val > best[t] { best[t] = val; }
        }
    }
    e0[ds_i] = best;
}
```

The critical difference from the scalar solver: the optimal category depends on the target. At target 200, the solver might prefer a safe category. At target 300, it might choose a risky one. The element-wise max captures this target-dependent behavior that no single &theta; can express.

Memory: 6.4 GB (4.2M state slots &times; 384 targets &times; 4 bytes). Compute: 346 seconds on Apple M1 Max with 8 threads. The dot product of the initial-state win-probability array with the opponent's exact PMF yields the clairvoyant ceiling of 55.3% H2H.

:::

:::
