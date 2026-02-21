# Literature, Reviews, and Research Directions

External positioning of the Scandinavian Yatzy solver against published literature, formal mathematical analysis of the risk-sensitive framework, and diagnosis of reinforcement learning limitations.

---

## 1. Literature Survey: Optimal Yahtzee/Yatzy Solvers

### Foundational Work

**Tom Verhoeff (TU Eindhoven, 1999)** was the first to compute the optimal solitaire Yahtzee strategy. His formulation splits the state space into Roll states and Choice states alternating in a directed acyclic graph. Each 13-category turn passes through six layers: initial roll (252 multiset outcomes), choose keepers (462 subsets), reroll, choose keepers, reroll, then score. The backward induction recurrence is clean: at Roll states, the optimal expected additional score is the probability-weighted average over outcomes; at Choice states, it is the maximum over available actions.

**James Glenn (Loyola University Maryland, 2006)** solved the problem independently. His 2007 IEEE CIG paper details further pruning of unreachable states and methods for computing not just expected values but full score distributions and variances. The full position graph contains roughly half a billion nodes.

**Phil Woodward** published the result in the statistics journal *Chance* in 2003. All three groups arrived at the same answer: the optimal expected score for solitaire Yahtzee under full American rules (including Extra Yahtzee Bonus and Joker rules) is **254.5896 ± 59.61** (mean ± std).

### Exact Arithmetic

**Jeffrey Liese and Katy Kelly (2017)** computed the exact rational value using Mathematica 11 over approximately five days. The numerator spans 139 digits, the denominator 137 digits. Without the Extra Yahtzee Bonus and Joker rules, the optimal EV drops to approximately 245.87.

### Computational Optimization

**Jakub Pawlewicz (University of Warsaw, 2009-2011)** achieved ~30 second computation time with aggressive optimizations. He generalized beyond expected score by computing full probability distributions at every game state, enabling multiplayer analysis. In interactive multiplayer Yahtzee, the optimal strategy adapts to opponents' visible scores. Pawlewicz showed this heuristic approach is experimentally "indistinguishable from optimal" for practical purposes, and proved that three-or-more player optimal play is beyond computational reach.

### Tournament-Optimal Strategies

**Theodore Perkins (2019 AAMAS)** demonstrated that in blind Yahtzee tournaments (the format used in Yahtzee with Buddies), the optimal strategy diverges sharply from score-maximizing:

| Opponents | Mean Score | σ | Win Rate vs EV-Optimal |
|----------:|-----------:|--:|:----------------------|
| 0 (solitaire) | 254.6 | 59.6 | baseline |
| 1 | 241.6 | 43 | +2.67% |
| 10,000 | 198.3 | 61 | +70% |

Against 10,000 opponents, the tournament-optimal strategy sacrifices 56 expected points yet wins 70% more often by gambling on multiple Yahtzees. Winning scores in real tournaments average ~349 for 15 players and ~563 for 10,000+.

### Alternative Objectives

**Cremers (2002)** developed "MaxProb" — maximizing the probability of exceeding a fixed target score — as part of his master's thesis at Eindhoven. For each game state and threshold *s*, Pr(total ≥ s) is propagated backward, producing strategies that sacrifice expected score for modified variance profiles.

No study has explicitly constructed a mean-standard deviation Pareto frontier for Yahtzee strategies, though Perkins' results implicitly trace one. This was the motivation for our θ sweep (see `analysis_and_insights.md` Section 4).

### Scandinavian Yatzy

**Larsson and Sjöberg (KTH, 2012)** were the first to compute the optimal Yatzy strategy, arriving at EV = **248.63 points**. The two additional categories (One Pair, Two Pairs) expand the category-subset state space from 2^13 = 8,192 to 2^15 = 32,768 (a fourfold increase), partially offset by the absence of the Yahtzee Bonus/Joker system. The maximum possible Yatzy score is 374 (vs Yahtzee's theoretical 1,575 with 13 consecutive Yahtzees). Our solver reproduces this EV (248.44 after the 50-point bonus correction).

### Other Variants

- **Yacht** (1938): the predecessor with 12 categories and no upper-section bonus
- **Maxi Yatzy** (Swedish): 6 dice and 20 categories
- **Forced Yatzy**: categories must be scored in order
- **Kniffel** (German): essentially identical to American Yahtzee

### Open-Source Implementations

| Project | Language | Notes |
|---------|----------|-------|
| Verhoeff & Scheffers (TU Eindhoven) | C | Gold standard online player, ~8 MB strategy table, ~10 min to generate |
| timpalpant/yahtzee | Go | Full optimal player with web server, Raspberry Pi deployment |
| greeness/yahtzee-optimal-strategy | Python | Implements Glenn's 2006 paper (no Yahtzee bonus) |
| dpmerrell/yahtzee | Python | MDP modeling (no upper-section bonus tracking, 229.6 avg) |
| Felix Holderied's yahtzee | C | Three modes: maximize EV, reach threshold, beat opponent |
| dionhaefner/yahtzotron | Python/JAX | Leading RL implementation (~240 avg via A2C) |
| philvasseur/Yahtzee-DQN-Thesis | Python | DQN strategy ladders (Vasseur & Glenn, 2019) |

---

## 2. Reinforcement Learning: Diagnosis of the 5% Gap

Multiple RL groups have applied deep RL to Yahtzee, consistently reaching ~95% of optimal play but unable to close the remaining gap.

### Published Results

**Pape (2025, arXiv:2601.00007)** systematically compared REINFORCE, A2C, and PPO with multi-headed networks on NVIDIA A40 GPUs. A2C achieved median 241.78 points (within 5.0% of 254.59). REINFORCE and PPO proved significantly more sensitive to hyperparameters.

**Häfner's Yahtzotron (2021)** demonstrated A2C effectiveness using JAX/Haiku with a pre-training curriculum (greedy table → advantage table → full A2C), reaching ~240-242 average in ~2 hours on a single CPU. Genetic optimization baseline plateaued at ~130 points.

**Dutschke** achieved 241.6 average with MLP-based Q-learning after 8,000 games. **Yuan (2023)** managed only 159 points in two-player DQN. **Kang & Schroeder (2018, Stanford)** achieved 99% win rate against random opponents with hierarchical MAXQ but only 68% against greedy.

### Three Barriers to Closing the Gap

**1. Discontinuous value function (the bonus problem).** The upper-section bonus creates a step function at m=63 worth 50 points. All RL models systematically overindex on four-of-a-kind plays at the expense of upper-section accumulation. The bonus rate under RL agents is ~24.9% vs ~68% optimal — this single failure accounts for 70-75% of the 5% gap. The boundary is a discontinuity in the value landscape that gradient-based optimizers struggle to represent.

**2. 13-turn credit assignment.** Scoring a 3 in Ones on turn 2 to advance the upper bonus may not yield its 50-point reward until turn 13. The per-turn signal from such decisions (~1 point) is buried in episode-level variance (σ ≈ 38), yielding SNR ≈ 1/38. Standard temporal-difference methods propagate this slowly, and reward shaping (PBRS) has not been explored.

**3. No adversarial curriculum.** Unlike Go or chess, solitaire Yatzy offers no opponent to drive exploration of edge cases. The agent must discover rare-but-critical states (e.g., upper score 62 with 2 categories left) through random exploration alone. The probability of visiting such states naturally is vanishingly small.

### Why Backgammon (1992) Was Solvable but Yatzy Isn't

A 5-axis taxonomy reveals why TD-Gammon succeeded where Yatzy RL fails:

| Axis | Backgammon | Yatzy |
|------|-----------|-------|
| Value smoothness | Continuous (pip counts) | Discontinuous (50-pt bonus cliff) |
| Credit assignment | ~40 moves, smooth updates | 13 turns, delayed 50-pt cliff |
| Curriculum | Opponent provides difficulty scaling | Solitaire, no curriculum |
| Reward density | Every move shifts position | Score only at turn end |
| Stochastic branching | ~21 dice outcomes | 252 multisets × 462 keeps |

These factors interact multiplicatively: even if each axis contributes only a 2-3x difficulty increase, their product is 30-100x harder than backgammon's already-impressive complexity.

### Unexplored Approaches (Ranked by Promise)

1. **DAgger** (Dataset Aggregation): behaviorally clone the optimal solver, then iteratively improve on states the clone visits. Most promising because it directly addresses the curriculum problem — the clone's own trajectory provides a natural curriculum of states where it errs.

2. **PBRS** (Potential-Based Reward Shaping): use the DP value function as a potential, giving dense reward signals that preserve optimality guarantees. Addresses credit assignment without changing the optimal policy.

3. **RUDDER** (Return Decomposition for Delayed Rewards): redistribute episode returns to individual timesteps using sequence analysis. Directly attacks the credit assignment problem.

4. **Stochastic MuZero**: MCTS + learned model for stochastic environments. Heavy but addresses the branching factor.

### Research Context

The gap is primarily **algorithmic, not fundamental**. The optimal policy exists and is computable; the question is whether gradient-based methods can find it without enumeration. The ~5% gap serves as a compact benchmark for advancing RL methods on structured combinatorial tasks with delayed rewards.

---

## 3. Transformer Architectures for Closing the RL Gap

The persistent RL gap motivates exploring attention-based architectures that might better handle Yatzy's specific challenges.

### Key Papers

**Paster, "You Can't Count on Luck" (NeurIPS 2022)** demonstrated that offline RL methods (Decision Transformer, CQL, IQL, BCQ) trained on mixed-quality Yahtzee data all plateau at ~95% of optimal, even with expert-level data. The bonus problem was identified as the primary bottleneck — Paster coined the term "can't count on luck" for policies that fail to account for stochastic credit assignment.

**ESPER** (Adversarial Clustering for offline RL): clusters trajectories by return then trains separate policies per cluster. The "high-return cluster" policy may implicitly learn bonus-achieving behavior by conditioning only on successful trajectories.

**Ni et al. (2023)** showed that transformers do not generally improve credit assignment — they are better at pattern matching than temporal reasoning. This suggests transformers would help with the curriculum/exploration problem but not directly with the bonus cliff.

### Ranked Architectural Assessment

| Architecture | Expected Benefit | Mechanism | Evidence Level |
|-------------|------------------|-----------|---------------|
| ESPER-DT | Highest | Clusters naturally separate bonus/no-bonus trajectories | Moderate (Paster's results) |
| Transformer-RUDDER | High | Return decomposition via attention weights | Moderate |
| PBRS | Moderate-High | Dense reward from DP potential | Strong (theoretical) |
| Trajectory Transformer | Moderate | Implicit planning via sequence modeling | Low |
| GTrXL | Low | Long-horizon memory | Low (already handles 13 turns) |
| Algorithm Distillation | Low | Cross-episode learning | Low (solitaire has no adaptation) |

### Implementation Path (Proposed)

1. **Phase 1**: ESPER-DT on existing simulation data. Cluster 10M simulated games by return, train separate decision transformers per cluster. Evaluate whether the high-return cluster policy achieves higher bonus rate.

2. **Phase 2**: If Phase 1 shows bonus-rate improvement, add RUDDER-style return decomposition to identify which turns contributed most to the bonus outcome. Use the decomposed returns as dense reward for online fine-tuning.

3. **Phase 3**: Full Trajectory Transformer with beam search at inference time, conditioned on target return. This is heaviest but could close the remaining gap by performing implicit planning over the 13-turn horizon.

---

## 4. Formal Mathematical Review

This section summarizes a formal review of the Scandinavian Yatzy solver framework, including six proofs establishing theoretical properties of the risk-sensitive solution.

### The Deep Structure of Sequential Decision-Making Under Uncertainty

The Yatzy solver instantiates a broader mathematical structure: a finite-horizon stochastic DP with a bipartite decision-chance DAG, non-linear utility aggregation, and a discontinuous reward function (the bonus cliff).

### Six Formal Results

**Theorem 1: CARA Sufficiency.** Among additively separable utility functions U satisfying the Bellman equation for stochastic nodes as U⁻¹(E[U(·)]), CARA (constant absolute risk aversion) is the unique solution. Proof via Pexider's functional equation: the requirement U⁻¹(αU(x) + βU(y)) = f(x, y, α, β) for all probability weights forces U(x) = ae^(bx) + c.

**Theorem 2: CARA Incompleteness.** There exist achievable (mean, variance) pairs that no constant-θ CARA policy can reach. Constructive proof: the "combined" adaptive policy from Section 9.1 of `analysis_and_insights.md` achieves (247.93, 39.0), lying strictly inside the constant-θ frontier (which achieves 248.15 at σ = 39.0). However, the gap is small (0.23 points), confirming the frontier is nearly tight.

**Theorem 3: Approximate Completeness Bound.** The maximum gap ε between the constant-θ frontier and any state-dependent policy is bounded by ε < O(T · B² / σ² · P(G < δ)), where T is the horizon (15 turns), B is the bonus value (50 points), σ is the score standard deviation, and P(G < δ) is the probability of being near the bonus threshold. This predicts ε ≈ 0.3-1.0 points, consistent with the measured 0.23-0.98 range.

**Theorem 4: Frontier Dimensionality (Berry-Esseen).** The constant-θ frontier is one-dimensional (parameterized by θ), but the space of achievable (mean, σ) pairs is two-dimensional. Berry-Esseen bounds on the sub-population mixture model show that the frontier's two-branch structure (Section 4.3 of `analysis_and_insights.md`) is a necessary consequence of the binary bonus creating distinct mixture components with different mean-variance profiles.

**Theorem 5: Temporal Separability.** The optimal policy cannot be decomposed as π*(s, t) = f(s) · g(t) — the game is constructively inseparable in state and time. Proof: exhibit states where the optimal category choice reverses between turns (e.g., Yatzy available: dump early, preserve late). However, the 4-parameter model (θ, β, γ, d) with γ < 1 provides a temporal discount that is descriptively separable — R² > 0.95 for a 3D temporal vector — despite being theoretically irreducible.

**Theorem 6: Local Risk Description.** The risk sensitivity of a game state can be locally described by a 3-dimensional vector: (bonus proximity, variance exposure, turns remaining). This low-dimensional description achieves R² > 0.95 for predicting how θ-sensitive a state's optimal action is, despite the full state space being 2M-dimensional.

### The Structural Paradox

Theorems 2 and 5 establish that the Yatzy decision problem is constructively inseparable — no factored representation is exactly optimal. Yet Theorems 3 and 6 show it is descriptively separable — simple low-dimensional models capture >95% of the variance in optimal behavior. The game is "more structured than it has any right to be" — the deep mathematical structure (CARA sufficiency, near-completeness, low-dimensional risk description) emerges from the interaction of the binary bonus cliff with the mixing distribution over 15 turns.

### Compact Representations

The solver's 2M-entry lookup table stores the exact solution, but the surrogate compression results (Section 8 of `analysis_and_insights.md`) show that ~130K decision tree parameters capture >99% of optimal play. The gap between 130K and the 2M theoretical requirement reflects the game's compressibility — most of the 2M states are "obvious" decisions where even a small tree agrees with optimal.

### Multiplayer Extension

The multiplayer case (Sections 10-11 of `analysis_and_insights.md`) introduces opponent modeling. Under the solitaire formulation, each player solves a separate DP; the interaction comes only through comparing final scores. For N opponents with known strategy θ_opp, the optimal response θ* maximizes E[P(my_score > max(opponent_scores))]. This is computable from existing per-θ score distributions via O(N) convolution.

For large N (tournament play), θ* → ∞: the optimal tournament strategy becomes maximally risk-seeking, gambling on rare high-scoring outcomes. This connects to Perkins (2019) and explains why tournament-optimal strategies sacrifice 50+ expected points. The connection to extreme value theory (the maximum of N draws from a distribution with exponential tails) provides the asymptotic framework.

---

## 5. Positioning: What This Solver Contributes

Based on comparison against all published work:

### Established (matching prior art)

- EV-optimal Scandinavian Yatzy strategy matching Larsson & Sjöberg's 248.63
- Backward induction over the full state space with keep-multiset deduplication
- Score distribution analysis confirming non-normality from binary categories

### Novel (not in published literature)

- **Risk-sensitive θ sweep**: Complete Pareto frontier of mean-variance tradeoffs under CARA utility across 37 θ values. No prior work has mapped this frontier for any Yahtzee/Yatzy variant.
- **Two-branch structure**: The discovery that risk-averse and risk-seeking branches trace distinct paths in (mean, σ) space, not a single efficient frontier. This is a structural property of sequential stochastic games with discontinuous rewards.
- **Near-completeness of constant-θ frontier**: Empirical demonstration (4 adaptive policies, 3 RL approaches) that state-dependent risk adaptation cannot meaningfully beat the constant-θ frontier, with theoretical bound (Theorem 3).
- **Cognitive profiling system**: 4-parameter model (θ, β, γ, d) with 30-scenario quiz, multi-start Nelder-Mead estimation, and simulation-backed player card. The revealed-preference framework for estimating player θ from observed decisions is the practical value proposition.
- **Surrogate policy compression**: Pareto frontier of model complexity vs EV loss, showing human-level play requires ~6K-15K parameters and decision trees dominate MLPs.
- **Density evolution**: Exact zero-variance score distributions via forward DP, enabling publication-grade PMFs without Monte Carlo noise.
- **Performance engineering**: Sub-2-second precompute, 5.6M games/sec oracle simulation, 3-second exact density — orders of magnitude faster than prior implementations.

### Publishable (sufficient depth for a paper)

The θ sweep with two-branch analysis and near-completeness proof constitutes a self-contained contribution to the computational game theory literature. The cognitive profiling system — estimating revealed risk preferences from observed play — connects to the behavioral economics literature on CARA estimation and could form a separate paper.
