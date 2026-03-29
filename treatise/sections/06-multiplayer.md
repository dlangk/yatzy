:::section{#multiplayer}

## Multiplayer and Adaptive Strategies

Most people play Yatzy as if it were a single-player optimization problem, and for good reason. Two parallel Yatzy games are independent: the dice you roll do not affect the dice your opponent rolls. Nothing you do changes what is available to them. In simultaneous play, the optimal strategy is simply to maximize expected value.

But sequential play introduces an asymmetry. In the standard format, Player 2 observes Player 1's trajectory (running score, categories consumed, upper-section progress) before making each decision. This raises a foundational question: can a player who adapts their risk posture based on the opponent's visible progress improve their probability of *winning*?

**The short answer: yes, but barely.** If both players use the fixed, EV-optimal strategy, they each win roughly 50% of decisive games. A player who dynamically adapts their risk can push this to 50.9%: fewer than nine extra wins per thousand games. For all practical purposes, Yatzy remains a game of parallel solitaire.

:::html
<div class="chart-container" id="chart-game-replay">
  <div id="chart-game-replay-svg"></div>
  <p class="chart-caption">A curated game illustrating variance-scaled adaptation. Top: cumulative scores (gray: Player 1, colored: Player 2). Bottom: Player 2's risk parameter &theta; (bars) against the combined remaining variance (shaded area). Player 1 scores Yatzy on turn 10, opening a 61-point gap. &theta; spikes to its ceiling. Player 2 answers with their own Yatzy on turn 11. As the variance envelope shrinks toward zero, the same score gap produces increasingly aggressive &theta; adjustments.</p>
</div>
:::

The path to that 0.9% is surprisingly rich. It involves a variance-scaling formula derived from Z-score maximization, exposes the structural failure of proxy utility functions in threshold games, and demands a 6.4 GB backward-induction computation to establish the absolute theoretical ceiling.

### The Framework: Trading EV for Variance

To adapt to an opponent, a player needs the ability to intentionally trade expected value for variance. Section 5 introduced the ::concept[exponential utility]{cara-utility} U(x) = e<sup>&theta;x</sup>, where &theta; controls the risk posture, and showed how precomputed strategy tables across a grid of &theta; values produce a spectrum of risk postures.

Each &theta; table finds a different point on the ::concept[mean-variance frontier]{mean-variance-frontier}. The key relationship (derived in the math section below) is the **marginal rate of substitution**: &part;E/&part;V = &minus;&theta;/2. This gives &theta; a precise geometric meaning. At &theta; = 0, the solver sits at peak EV. At &theta; = 0.10, it slides right along the frontier, accepting a drop of 1 point of EV per 20 units of variance gained. At &theta; = &minus;0.10, it slides left, paying EV to crush variance. This relationship is the engine connecting static risk tables to dynamic win-probability optimization.

The baseline head-to-head (H2H) win rate, where both players use the EV-optimal &theta; = 0 policy (excluding draws), is **50.04%**.

### The Oracle Ceiling

Before building any adaptive strategy, we establish a ceiling. If you knew your opponent's final score before the game began, and picked the single best constant &theta; for that exact outcome, how much would it help?

The chart below splits opponent final scores into seven bands and shows, for each band, the win rate of EV-optimal play alongside the win rate of the best possible fixed &theta; for that band. This "oracle" has information no real player could have: it knows the opponent's total in advance and selects its risk posture accordingly.

:::html
<div class="chart-container" id="chart-oracle-winrate">
  <div id="chart-oracle-winrate-svg"></div>
  <p class="chart-caption">Conditional win rates per opponent score band. Gray: EV-optimal (&theta; = 0). Colored: best fixed &theta; for each band, colored by the optimal &theta; value (blue = conservative, orange = risk-seeking).</p>
</div>
:::

When the opponent scored below 200, the oracle plays conservatively (&theta; = &minus;0.03), protecting a near-certain lead. When the opponent scored above 280, it plays risk-seeking (&theta; = +0.04), because only tail outcomes can win against a strong score. The transition from negative to positive &theta; happens near the mean (~250), where the two distributions overlap most.

The overall oracle win rate is **~50.6% H2H**: a gain of +0.60 percentage points over EV-optimal. That is the ceiling for any strategy that picks one constant &theta; per game.

### Turn-by-Turn Adaptation

In practice, Player 2 does not know the opponent's final score. They observe it unfold turn by turn. From the visible state, Player 2 can estimate the opponent's expected final score via a state-value table lookup: E[final] = current_total + EV_remaining(state).

The simplest adaptive policy scales &theta; linearly with the *EV deficit* (opponent's expected final minus own expected final). When trailing, increase &theta;; when leading, decrease it. Three variants tested at 10 million games each:

| Policy | Description | H2H Win Rate | Net Advantage |
|--------|-------------|-------------|---------------|
| Seek-only | &theta; ramps 0 &rarr; +0.05 when trailing | 50.44% | +0.41pp |
| Protect-only | &theta; ramps 0 &rarr; &minus;0.03 when leading | 50.20% | +0.16pp |
| Symmetric | Seek when trailing + protect when leading | 50.57% | +0.53pp |

The symmetric policy approaches the oracle ceiling (+0.53pp vs. +0.60pp) despite having far less information. It manages this because turn-by-turn switching pays the mean-score penalty only for turns where &theta; deviates from zero, while the oracle pays for all 15 turns. Risk-seeking when trailing contributes roughly 75% of the advantage; protecting when leading adds the remaining 25%.

An intuitive extension is to play EV-optimal for most of the game, then "go for broke" in the last 2-3 turns when trailing. Every such "Hail Mary" variant performed worse than the symmetric policy, and most were worse than the EV-optimal baseline. The failure is structural: under large &theta;, the exponential utility e<sup>&theta;x</sup> amplifies extreme scores so aggressively that the solver chases maximum-score outcomes rather than outcomes that beat the target. Under U(x) = e<sup>0.50x</sup>, scoring 50 points has over 3 million times the utility of scoring 20 points. The table becomes a *max-score chaser* rather than a *target optimizer*. Continuous proxy utilities cannot express the step-function logic of a head-to-head match, where winning by 1 point is exactly as valuable as winning by 100.

### Variance-Scaled Risk

The symmetric policy has a structural flaw: it is horizon-blind. A 15-point deficit receives the same &theta; at turn 2 and turn 14, even though early deficits are noise (remaining variance is large) and late deficits are emergencies (remaining variance is small).

The fix comes from a clean derivation (detailed in the math section). If we model the final scores as approximately Gaussian, Player 2 maximizes P(S<sub>2</sub> &gt; S<sub>1</sub>) by maximizing the Z-score of the score difference. Equating the match-optimal slope with the tables' built-in MRS yields:

:::equation
\theta^* = \frac{E_1 - E_2}{V_1 + V_2}
:::

This is positive when trailing (seek risk), negative when leading (protect), and scales inversely with remaining uncertainty. Early in the game V<sub>1</sub> + V<sub>2</sub> exceeds 3000, suppressing &theta;* even for large deficits. Late in the game it drops below 500, amplifying small deficits into decisive risk adjustments. If the opponent is on the bubble for the 50-point upper bonus, their remaining variance V<sub>1</sub> is large; the formula commands patience. The moment the opponent locks or fails the bonus, V<sub>1</sub> collapses, &theta;* spikes, and the policy pivots.

Use the slider below to see how the same deficit produces different &theta; at different turns. The variance-scaled policy (solid) is conservative early and aggressive late. The linear policy (dashed) applies the same &theta; regardless of turn. The faded line shows unclamped &theta;*, which shoots into the degeneration zone in the final turns.

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

The remaining variance is extracted in O(1) from two table lookups (see math section), and the raw &theta;* is clamped to [&minus;0.05, +0.07] to prevent the utility degeneration described above.

| Policy | H2H Win Rate | Net Advantage |
|--------|-------------|---------------|
| Variance-scaled [&minus;0.05, +0.07] | **50.9%** | **+0.86pp** |
| Symmetric (reference) | 50.6% | +0.53pp |

The variance-scaled policy is 2x more conservative than symmetric in early turns and 2x more aggressive in late turns, with the crossover at turn 8. A parameter sweep over 9 clamp configurations confirmed the landscape is flat above +0.07 on the seek side and insensitive to the protect clamp. The policy has reached its architectural ceiling within the &theta;-table-switching framework.

### The Absolute Ceiling: 3D Threshold Tablebase

The variance-scaled policy is constrained to selecting one precomputed &theta; table per turn. How close is this to the theoretical maximum? To answer that, we solve a different game entirely.

**The clairvoyant game.** Player 1 plays all 15 turns in isolation and writes their final score on a piece of paper. Player 2 then plays knowing exactly what number they must beat. This is a fundamentally different format from interleaved play. So why compute it?

The answer is *information relaxation*, a technique from dynamic programming. Three properties make this "breached" game a rigorous ceiling for the real game:

First, **Player 1 is a fixed environment.** Because Player 1 uses the EV-optimal policy and ignores Player 2, turn order does not affect Player 1's score distribution. To Player 2, Player 1 is not a reactive opponent; they are a complex random number generator.

Second, **more information never hurts.** A policy with perfect future information always performs at least as well as one with partial present information. The clairvoyant never wastes EV on a risk it did not need, and never plays it safe when the opponent is about to roll a Yatzy on turn 14.

Third, **it collapses the state space.** The exact turn-by-turn optimum requires a joint-state MDP with approximately 164 trillion states. By revealing the final score, Player 1's game tree reduces to a single target number T, shrinking the problem to 1.43 million states, solvable in under 6 minutes.

We solved a 3D backward-induction DP over the state space (upper_score, scored_categories, target_score_needed). For each of 1.43 million reachable states and 384 target scores, the solver maximizes P(final_score &gt; T). Unlike the &theta; tables, the optimal category depends on the target: at T = 200, the solver prefers safe scoring; at T = 300, it makes entirely different trade-offs.

| Strategy | H2H Win Rate | Net Advantage | % of Ceiling |
|----------|-------------|---------------|-------------|
| EV-optimal baseline | 50.04% | 0.00pp | 0% |
| Constant-&theta; oracle | ~50.6% | +0.60pp | 11% |
| Variance-scaled (best heuristic) | 50.9% | +0.86pp | 16% |
| **Unconstrained clairvoyant** | **~55.3%** | **+5.3pp** | **100%** |

**The action gap.** The 4.7pp gap between the constant-&theta; oracle (50.6%) and the unconstrained clairvoyant (55.3%) isolates the structural cost of continuous proxy utilities. Both have perfect information; the difference is purely in decision-making. The clairvoyant executes non-linear sacrifice plays (intentionally zeroing a risky category to lock in exact points needed) that no continuous utility function can express.

**The stochastic washout effect.** Even with clairvoyant knowledge and unconstrained decisions, the maximum edge is ~5.3 percentage points. Under realistic sequential observation, it collapses to +0.86pp: fewer than 9 extra wins per 1000 games. The aleatoric entropy of 15 turns of dice rolls overwhelmingly dominates the strategic bandwidth of reactive play. Our best heuristic captures 16% of the combined ceiling; the true sequential optimum lies somewhere in [50.9%, 55.3%], and the heuristic captures more than 16% of the realistically achievable advantage.

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

Model final scores as independent distributions with means E<sub>i</sub> and variances V<sub>i</sub>. Player 2 maximizes:

:::equation
P(S_2 > S_1) = \Phi(Z) \quad \text{where} \quad Z = \frac{E_2 - E_1}{\sqrt{V_1 + V_2}}
:::

Player 2 controls (E<sub>2</sub>, V<sub>2</sub>) by choosing &theta;. Taking dZ/dV<sub>2</sub> = 0:

:::equation
\frac{dZ}{dV_2} = \frac{\frac{\partial E_2}{\partial V_2} \cdot \sqrt{V_1 + V_2} - \frac{E_2 - E_1}{2\sqrt{V_1 + V_2}}}{V_1 + V_2} = 0
:::

The numerator vanishes when:

:::equation
\frac{\partial E_2}{\partial V_2}(V_1 + V_2) = \frac{E_2 - E_1}{2}
:::

Substituting &part;E<sub>2</sub>/&part;V<sub>2</sub> = &minus;&theta;/2 from the MRS:

:::equation
\left(-\frac{\theta^*}{2}\right)(V_1 + V_2) = \frac{E_2 - E_1}{2}
:::

The 1/2 factors cancel, yielding:

:::equation
\theta^* = \frac{E_1 - E_2}{V_1 + V_2}
:::

**Properties.** The formula is positive when trailing (E<sub>1</sub> &gt; E<sub>2</sub>), negative when leading, and vanishes when even. It scales inversely with total remaining variance: early in the game V<sub>1</sub> + V<sub>2</sub> &gt; 3000, suppressing &theta;* even for large deficits. Late in the game V<sub>1</sub> + V<sub>2</sub> &lt; 500, amplifying small deficits into meaningful risk adjustments.

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
