# Sequential Multiplayer Yatzy: Measuring the Adaptive Advantage

## I. Abstract

In Scandinavian Yatzy played with sequential turns, Player 2 observes Player 1's game state before each decision. We ask: how much can Player 2 increase their win rate by adapting their risk parameter based on this information?

We tested four classes of adaptive policies, pre-registering quantitative hypotheses for each. The best heuristic policy (variance-scaled θ switching) achieves a 50.86% head-to-head win rate (+0.86pp over the 50.04% baseline). We computed the theoretical ceiling via a 3D backward-induction DP: a clairvoyant player who knows the opponent's final score in advance and makes per-state optimal decisions achieves 55.3% H2H. Our heuristic captures 16% of this theoretical maximum. Crucially, the gap reflects both the clairvoyant's information advantage (knowing the final score vs. observing mid-game state) and the θ-table architecture's limited expressiveness.

## II. Problem Statement

Two players play 15-category Scandinavian Yatzy (50-point upper bonus). Player 1 uses the EV-optimal strategy (θ=0) and goes first every turn. Player 2 goes second and, before making each decision, observes Player 1's full public state: running total, scored categories, and upper-section progress.

Player 2 has access to precomputed strategy tables for risk parameters θ from -0.50 to +0.50. Each table contains optimal decisions under exponential utility e^(θx) for every reachable game state. Player 2 may select a different table each turn based on observed game state, but each turn's decisions must come from a single table.

**Question**: What head-to-head win rate can Player 2 achieve?

**Definitions**:
- *Head-to-head win rate (H2H)*: P2 wins / (P2 wins + P1 wins), excluding draws. Baseline (both θ=0): 50.04%.
- *Net advantage*: H2H minus baseline (50.04%).
- *EV deficit*: best opponent expected final score minus own expected final score, computed via state-value table lookups. Accounts for bonus progress and remaining categories.

## III. Experimental Setup

**Hardware**: Apple M1 Max (10-core, 64 GB), 8 rayon threads.

**Game loop**: `solver/src/simulation/multiplayer.rs`. Players take turns sequentially within each of 15 rounds. After Player 1 completes a turn, their state is updated before Player 2 decides.

**Statistical design**: 1M games per matchup (SE = 0.05pp). Effects of 0.5pp and above are detected at >10σ. Reference matchups confirmed at 10M games (SE = 0.016pp). All simulations use deterministic seeding for reproducibility.

**Available θ tables**: 90+ precomputed via backward-induction DP, covering θ from -0.50 to +0.50 with dense spacing near zero. Each table: 16 MB, memory-mapped.

## IV. Experiment 1: Linear-Scale EV-Based Policies

### Hypothesis

Three policies adapt θ based on EV deficit with a linear ramp and fixed scale parameter of 50:

1. **Seek-only**: θ ramps 0 → +0.05 when trailing (deficit > 0). θ=0 when leading.
2. **Protect-only**: θ ramps 0 → -0.03 when leading (deficit < 0). θ=0 when trailing.
3. **Symmetric**: Combines seek and protect.

All use `UnderdogPolicy::ev_deficit()` (state-value table lookups), not raw score gaps. Signal delay: θ=0 for turns 0-1 (insufficient information).

Pre-registered predictions were 0.01-0.08pp for the symmetric policy, based on an information-revelation model: the EV funnel data shows only ~12% of final-score variance is explained by mid-game state. This led to an estimated effective signal-leverage product of ~0.03pp.

### Results (10M games each)

| Policy | H2H | Net | P2 Mean | P2 Std |
|--------|-----|-----|---------|--------|
| Symmetric | 50.57% | +0.53pp | 247.4 | 39.8 |
| Seek-only (θ_max=0.05, scale=50) | 50.44% | +0.41pp | 247.5 | 40.2 |
| Seek-only (θ_max=0.07, scale=40) | 50.45% | +0.41pp | 246.5 | 41.5 |
| Protect-only (θ_max=0.03, scale=50) | 50.20% | +0.16pp | 247.9 | 39.3 |
| Baseline (ev vs ev) | 50.04% | 0.00pp | 248.4 | 38.5 |

### Analysis

The pre-registered prediction (0.01-0.08pp) was wrong by ~10x. The information-revelation model correctly identified that the mid-game signal about the opponent's final score is weak, but missed a key mechanism: **turn-by-turn θ switching pays the mean-score penalty only for turns where θ≠0**, while a constant-θ policy pays for all 15 turns. The symmetric policy's mean cost is 1.0 point (248.4 - 247.4), compared to 3.9 points for a full-game θ=0.05 policy. This asymmetry in cost structure makes the adaptive policy far more efficient than the information-revelation model predicted.

Seek-only contributes approximately 75% of the symmetric advantage (+0.41 of +0.53pp). Protection when leading contributes the remaining 25%.

Parameter sweeps (42 underdog configs, 90 symmetric configs, all at 100K games) showed the effect is parameter-insensitive: all reasonable configurations produce H2H between 50.4% and 50.7%.

## V. Experiment 2: Per-Turn Diagnostics

### Method

Instrumented the game loop to record, for each turn of each game, the actual θ selected by P2 and the EV deficit at that decision point. Used `UnderdogPolicy::ev_deficit()` for the deficit computation (O(1) state-value table lookup per player). Sequential simulation, 1M games.

### Results (symmetric policy)

| Turn | θ=0% | θ>0% | θ<0% | Mean\|θ\| | Mean deficit | Std deficit |
|------|------|------|------|-----------|-------------|-------------|
| 0 | 100.0 | 0.0 | 0.0 | 0.0000 | 0.0 | 9.1 |
| 1 | 100.0 | 0.0 | 0.0 | 0.0000 | 0.0 | 15.8 |
| 2 | 14.5 | 43.7 | 41.8 | 0.0124 | 0.0 | 20.5 |
| 5 | 9.0 | 46.5 | 44.5 | 0.0183 | 0.1 | 30.7 |
| 8 | 7.0 | 47.3 | 45.7 | 0.0218 | 0.2 | 38.8 |
| 11 | 5.8 | 47.7 | 46.5 | 0.0242 | 0.5 | 46.2 |
| 14 | 5.1 | 48.0 | 46.9 | 0.0261 | 0.9 | 54.3 |

### Analysis

The symmetric policy deviates from θ=0 in 85-95% of games from turn 2 onwards. Mean|θ| grows only 2.1x from turn 2 (0.012) to turn 14 (0.026), despite the EV deficit standard deviation growing 2.6x (20.5 to 54.3). This indicated the fixed scale parameter (deficit/50) was not horizon-aware: it applied similar θ magnitudes early and late in the game.

## VI. Experiment 3: Concentrated Late-Game Risk ("Hail Mary")

### Hypothesis

A policy that plays θ=0 for most of the game and switches to aggressive θ (0.15-0.50) only in the last 2-3 turns when trailing should outperform the symmetric policy. The analytical model predicted +1.0 to +2.0pp:

- 21.3% of games are close losses (P2 loses by 1-30 points)
- Two turns of θ=0.2 add ~10.7 points of extra std at a mean cost of ~3.1 points
- Estimated 5-10% of close losses flip to wins, yielding +1.0-2.1pp net

### Results (1M games each)

| Policy | H2H | Net |
|--------|-----|-----|
| Hail Mary (θ=0.10, turns 12-14, deficit > 5) | 50.2% | +0.16pp |
| Hail Mary (θ=0.15, turns 12-14) | 50.1% | +0.06pp |
| Hail Mary (θ=0.20, turns 12-14) | 50.0% | -0.04pp |
| Hail Mary (θ=0.20, turns 13-14) | 50.1% | +0.06pp |
| Hail Mary (θ=0.30, turns 12-14) | 49.8% | -0.24pp |
| Hail Mary (θ=0.50, turns 12-14) | 49.6% | -0.44pp |
| Hail Mary (θ=0.20, turns 10-14) | 49.5% | -0.54pp |
| Combined (symmetric turns 2-11 + Hail Mary turns 12-14) | 50.3% | +0.26pp |
| Symmetric (reference) | 50.6% | +0.56pp |

### Analysis

The hypothesis was falsified. Every Hail Mary variant performed worse than the symmetric policy, and most were worse than baseline. Higher θ and more activation turns both degraded performance monotonically.

**Root cause: extreme convexity of CARA utility, not horizon blindness.** The θ tables are computed via backward-induction DP and possess perfect horizon awareness: they assign exactly zero probability to outcomes that cannot physically materialize in the remaining turns. The failure mechanism is the extreme convexity of exponential utility U(x) = e^(θx). At θ=0.50, the utility of scoring 50 points (e^25) dwarfs that of scoring 20 points (e^10) by a factor of e^15 ≈ 3.3 million. The DP therefore willingly sacrifices a guaranteed 20 points (which might ensure a match win) for a small probability of 50 points, because the expected utility is dominated by the tail. The table does not ignore the horizon; it mathematically acts as a max-score chaser rather than a target optimizer.

The analytical model's error was in treating the high-θ variance boost as useful, target-neutral noise. In reality, high-θ tables concentrate variance into one extreme tail (the maximum reachable score), sacrificing the mid-range outcomes that would actually flip close losses into wins.

## VII. Experiment 4: Variance-Scaled Risk

### Hypothesis

Proposed by Gemini Deep Think. The core insight: the symmetric policy's fixed scale parameter (deficit/50) is horizon-unaware. A 15-point deficit at turn 2 (remaining variance ~3000) is noise; the same deficit at turn 14 (remaining variance ~200) is an emergency. The variance-scaled formula:

θ* = (E₁ - E₂) / (V₁ + V₂)

normalizes the deficit by the combined remaining variance. V is extracted in O(1) from two table lookups using the cumulant expansion CE_θ ≈ E + θ/2 · Var:

V(state) = 200 · (sv_{θ=0.01}[state] / 0.01 - sv_{θ=0}[state])

Verified: for the initial state (no categories scored), this yields V = 1482, matching the exact density std of 38.5 (38.5² = 1482).

The raw θ is clamped to avoid policy degeneration: θ_action = clamp(θ*, -0.03, +0.05) or (-0.05, +0.07).

Pre-registered prediction: +0.7 to +0.9pp, based on the reasoning that the improvement comes purely from better timing of risk (conservative early, aggressive late), not from accessing higher θ values. The early-game EV savings from reduced θ (~0.01 per turn for ~10 turns) translate to roughly 0.1-0.2pp of additional win rate.

### Results (1M games each)

| Policy | H2H | Net | P2 Mean |
|--------|-----|-----|---------|
| Varscaled (clamp [-0.05, +0.07]) | 50.9% | +0.86pp | 247.0 |
| Varscaled (clamp [-0.03, +0.05]) | 50.8% | +0.76pp | 247.5 |
| Varscaled (seek-only, clamp [0, +0.05]) | 50.6% | +0.56pp | 247.6 |
| Symmetric (reference) | 50.6% | +0.56pp | 247.4 |

### Per-turn diagnostics comparison

| Turn | Symmetric Mean\|θ\| | Varscaled Mean\|θ\| | Ratio |
|------|---------------------|---------------------|-------|
| 2 | 0.0124 | 0.0066 | 0.53x |
| 5 | 0.0183 | 0.0127 | 0.69x |
| 8 | 0.0218 | 0.0213 | 0.98x |
| 11 | 0.0242 | 0.0353 | 1.46x |
| 14 | 0.0261 | 0.0560 | 2.15x |

### Analysis

The pre-registered prediction (+0.7-0.9pp) was confirmed. The variance-scaled policy with wider clamping achieved +0.86pp, a 53% improvement over the symmetric baseline (+0.56pp).

The per-turn diagnostics confirm the mechanism: the variance-scaled policy is 2x more conservative than symmetric in early turns and 2x more aggressive in late turns. The crossover occurs around turn 8. Mean|θ| grows 8.5x from turn 2 to turn 14 (vs. 2.1x for symmetric), reflecting the natural horizon scaling of the V₁+V₂ denominator.

### Clamp sensitivity sweep

To determine whether wider clamping bounds could improve the variance-scaled policy further, we swept all combinations of positive clamp ∈ {0.05, 0.07, 0.10} and negative clamp ∈ {0.03, 0.05, 0.07} (9 configurations, 1M games each).

| Positive clamp | Negative clamp | H2H |
|----------------|----------------|-----|
| +0.05 | -0.03 | 50.8% |
| +0.05 | -0.05 | 50.8% |
| +0.05 | -0.07 | 50.8% |
| +0.07 | -0.03 | 50.9% |
| +0.07 | -0.05 | 50.9% |
| +0.07 | -0.07 | 50.9% |
| +0.10 | -0.03 | 50.9% |
| +0.10 | -0.05 | 50.9% |
| +0.10 | -0.07 | 50.9% |

The landscape is completely flat above a positive clamp of +0.07: the variance-scaled formula θ* = deficit/(V₁+V₂) naturally stays below 0.07, so wider positive clamps have no effect. The negative clamp is entirely insensitive across the tested range. This confirms that +0.07/-0.03 is the optimal configuration and that the policy has reached its architectural ceiling within the θ-table-switching framework.

## VIII. Theoretical Ceiling: Clairvoyant Upper Bound

### Constant-θ clairvoyant

If Player 2 knows Player 1's exact final score T before the game begins and selects the single best constant θ for that score, the maximum win rate is 50.22% (+0.60pp). Computed by optimizing over 99 exact PMFs (from forward-DP density evolution) for each possible opponent score, weighted by the θ=0 score distribution.

Our variance-scaled policy (+0.86pp) exceeds this ceiling because turn-by-turn switching is a strictly more powerful strategy class: it pays the mean-score penalty only for turns where θ≠0, while the constant-θ clairvoyant pays for all 15 turns.

### Unconstrained clairvoyant (3D threshold tablebase)

The true clairvoyant ceiling: Player 2 knows T and makes per-state, per-dice optimal decisions to maximize P(score > T), unconstrained by θ tables.

**Method**: 3D backward-induction DP over the state space (upper_score, scored_categories, target_score_needed). For each of 1.43M reachable states, a 384-element array stores P(remaining_score > t) for each target t ∈ [0, 383], under the policy that maximizes that probability. The solver follows the same 6-group structure as the scalar EV solver (category selection → keep decisions → initial roll expectation), but operates on arrays instead of scalars.

- Memory: 6.4 GB (4,194,304 state slots × 384 targets × 4 bytes)
- Compute: 346 seconds on M1 Max (8 threads)
- Implementation: `solver/src/bin/clairvoyant.rs`

The clairvoyant win rate is computed by dotting the initial-state win-probability array with Player 1's exact score distribution: ceiling = Σ_T P(P1 = T) · P_win(start_state, T).

### Results

All values below are head-to-head win rates (excluding draws). Assuming a ~0.7% draw rate, Player 1's absolute win rate is ~44.43%. The clairvoyant's H2H win rate is therefore 54.87 / (54.87 + 44.43) = 55.26%, which we round to 55.3%.

| Strategy | H2H | Net advantage | % of ceiling |
|----------|-----|---------------|-------------|
| Baseline (θ=0 vs θ=0) | 50.04% | 0.00pp | 0% |
| Constant-θ clairvoyant | ~50.6% | +0.60pp | 11% |
| Variance-scaled (best heuristic) | 50.9% | +0.86pp | 16% |
| **Unconstrained clairvoyant** | **~55.3%** | **+5.3pp** | **100%** |

### Analysis

The unconstrained clairvoyant achieves ~55.3% H2H (+5.3pp over baseline). Our best heuristic captures 16% of this gap (0.86 / 5.3).

The remaining 84% gap reflects two distinct, inseparable factors:

1. **Information gap**: The clairvoyant knows the opponent's exact final score from turn 1. Our heuristic observes only mid-game state, which (as shown in Experiment 1) contains weak predictive signal for the final score.
2. **Action gap**: The clairvoyant makes per-state, per-target-score optimal decisions. Exponential utility tables are a one-parameter family that cannot express target-specific trade-offs. At target 200 (weak opponent), the unconstrained policy achieves P(win) = 94.3% vs. 90.9% for θ=0. At target 300 (strong opponent), it achieves P(win) = 13.1% vs. 9.4%.

These factors cannot be cleanly separated without solving the intractable joint-state MDP (~164 trillion states). The clairvoyant ceiling is a strict upper bound on the sequential game (knowing the final score is strictly more information than observing turn-by-turn progress). The true sequential optimum lies in [50.86%, ~55.3%], and our heuristic captures more than 16% of the realistically achievable sequential advantage.

## IX. Summary of All Policies

| Policy | H2H | Net | Mechanism | Outcome |
|--------|-----|-----|-----------|---------|
| Varscaled ([-0.05, +0.07]) | 50.9% | +0.86pp | Variance-normalized θ switching | Best heuristic |
| Varscaled ([-0.03, +0.05]) | 50.8% | +0.76pp | Same, narrower clamp | |
| Symmetric (scale=50) | 50.6% | +0.56pp | Fixed-scale EV-deficit ramp | Superseded |
| Seek-only | 50.4% | +0.41pp | Risk-seeking when trailing only | |
| Hail Mary + symmetric | 50.3% | +0.26pp | Symmetric early, aggressive late | Failed |
| Protect-only | 50.2% | +0.16pp | Risk-averse when leading only | |
| Hail Mary (θ=0.10) | 50.2% | +0.16pp | EV-optimal early, mild aggression late | Failed |
| Baseline (ev vs ev) | 50.0% | 0.00pp | Both players θ=0 | Reference |
| Trailing-risk (θ=0.08, raw scores) | 48.9% | -1.10pp | Binary switch on raw score gap | Harmful |

## X. Conclusions

1. **The adaptive advantage is real but small.** The best heuristic wins 50.86% of decisive games, roughly 9 extra wins per 1000 games. For practical play, Yatzy remains effectively a single-player game.

2. **Variance-normalized risk scaling is the key mechanism.** Dividing the EV deficit by combined remaining variance produces the right θ magnitude at each game phase without manual tuning. The formula θ* = deficit / (V₁ + V₂) has a natural interpretation as the Z-score of the score difference.

3. **The clairvoyant ceiling reveals both an information gap and an action gap.** The unconstrained clairvoyant (~55.3% H2H) benefits from two advantages our heuristic lacks: perfect knowledge of the opponent's final score, and per-state decisions optimized for that specific target. Our heuristic captures 16% of the combined gap, but since the true sequential ceiling is strictly lower than the clairvoyant ceiling, the fraction of the achievable sequential advantage captured is higher than 16%.

4. **CARA utility degenerates under extreme θ, even with perfect horizon awareness.** The Hail Mary failure is not due to horizon blindness (the backward-induction tables have exact horizon information). It is caused by the extreme convexity of e^(θx): large θ forces the DP to chase maximum-score outcomes regardless of whether those outcomes would actually win the match.

5. **The true sequential optimum is bounded by [50.9%, ~55.3%] H2H.** Tightening this bound would require either a partial 3D tablebase conditioned on estimated opponent score, or solving the full joint-state MDP (approximately 164 trillion states; intractable).

## XI. Reproducibility

All simulation code: `solver/src/simulation/strategy.rs` (policies), `solver/src/simulation/multiplayer.rs` (game loop and diagnostics), `solver/src/bin/multiplayer.rs` (CLI), `solver/src/bin/clairvoyant.rs` (3D DP).

Commands:
```bash
# EV-based policies (10M games)
yatzy-multiplayer --strategy ev --strategy "mp:symmetric" --games 10000000 --seed 1007

# Variance-scaled (1M games with diagnostics)
yatzy-multiplayer --strategy ev --strategy "mp:varscaled:0.07:0.05" --games 1000000 --seed 42 --diagnostics

# Clairvoyant ceiling (346 seconds, 6.4 GB)
yatzy-clairvoyant
```

## XII. References

- `theory/foundations/risk-parameter-theta.md`: CARA utility and θ parameter
- `theory/strategy/risk-sensitive-strategy.md`: θ sweep results and two-branch structure
- `theory/research/3d-threshold-tablebase.md`: 3D DP design document
- `solver/src/bin/clairvoyant.rs`: unconstrained clairvoyant solver implementation
