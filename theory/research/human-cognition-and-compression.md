# Human Cognition and Policy Compression

## 1. Where Human Play Diverges from Optimal

### 1.1 The Gap: 82.6 Points Per Game

A pattern-matching heuristic (greedy category selection, cascade-based rerolls, no lookahead) scores mean 165.9 vs the optimal 248.4 — an 82.6-point gap. Per-decision gap analysis over 100K games reveals 24.4 disagreements per game (out of ~45 decisions), breaking down by decision type:

| Decision Type | Disagreements/Game | EV Loss/Game | % of Gap |
|---|---:|---:|---:|
| Category selection | 6.5 | 38.4 | 47% |
| First reroll | 9.7 | 24.0 | 29% |
| Second reroll | 8.2 | 20.2 | 24% |
| **Total** | **24.4** | **82.6** | **100%** |

Category mistakes account for nearly half the gap despite being the least frequent disagreement type — each category mistake is roughly 3× costlier than a reroll mistake.

### 1.2 The Five Costliest Mistake Patterns

**1. Wrong reroll target — keeping too few or too many dice (~27 pts/game).** The heuristic's cascade logic often keeps only 1 die (the highest) when optimal would keep a pair, or keeps 4 dice (chasing a near-straight) when optimal would reroll more aggressively. Examples: with `[1,5,5,6,6]`, the heuristic keeps only the 6 (for Chance/upper); optimal keeps the pair of 6s (upper bonus + pair potential). This pattern alone costs 8.8 pts/game for first rerolls and 5.6 pts/game for second rerolls.

**2. Wasting upper categories on lower-section patterns (~12 pts/game).** The heuristic takes lower-section patterns (Three of a Kind, Four of a Kind) when the same dice score equally well in an upper category that would advance the bonus. Five sub-patterns dominate: taking upper categories instead of Small Straight when the straight scores 15 but the upper category scores less. Combined: Fours-vs-Straight (3.2), Fives-vs-Straight (3.0), Threes-vs-Straight (2.9), Sixes-vs-Straight (2.2), Twos-vs-Straight (1.4) = 12.7 pts/game.

**3. Missing upper category opportunities (~7 pts/game).** The heuristic picks lower-section categories (Three of a Kind, Four of a Kind, Two Pairs) when optimal would take an upper category. Examples: three 1s scored as Three of a Kind (3 pts) instead of Ones (3 pts, same score but advances bonus); four 6s scored as Four of a Kind (24 pts) instead of Sixes (24 pts, same score but advances bonus).

**4. Yatzy over-chasing via rerolls (~2.8 pts/game).** On second reroll, the heuristic continues rerolling when optimal would keep all five dice. This occurs when the current hand scores well and rerolling risks breaking a scoring combination. The heuristic's blind Yatzy chase (always rerolling non-matching dice when 3-of-a-kind is present) costs 2.75 pts/game.

**5. Bonus rate collapse.** The cumulative effect: the heuristic achieves the upper bonus in only 1.2% of games vs 87% for optimal. The 50-point bonus swing explains ~43 points of the 82.6-point gap. Every category and reroll mistake that fails to advance the upper score compounds into bonus failure.

### 1.3 Why Errors Compound

Individual errors create cascading suboptimality. Filling category X suboptimally means later rolls suited for X must go elsewhere. The surrogate model evaluation (Section 2.6) quantifies this: a depth-20 decision tree with combined EV loss of 1.4 pts/game achieves actual game mean of 245 (predicted 247), showing ~2 points of compounding cost at high accuracy. At the heuristic's 82.6 pts/game EV loss, compounding is substantial — the actual score deficit includes both direct decision costs and cascading misallocation.

### 1.4 Why Humans Feel Competent Despite the Gap

Without a solver as reference, humans compare against other humans. With σ ≈ 38 even under optimal play, any suboptimal player will occasionally score well above their mean. These memorable highs reinforce the feeling of competence. The gap between human and optimal play is invisible without a baseline to measure against.

### 1.5 Estimating Human Cognitive Profile

The goal is to estimate a player's decision-making parameters from a short quiz. Rather than requiring 50–100 observed full games, we compress estimation into 30 diagnostic scenarios — game states where the optimal action depends on the player's cognitive profile.

**The 4-parameter model.** Player behavior is modeled as noisy optimization of a modified value function:

$$P(a \mid s) \propto \exp(\beta \cdot Q_{\theta,\gamma,d}(a, s))$$

The four parameters capture distinct cognitive dimensions:

| Parameter | Meaning | Range | Optimal |
|---|---|---|---|
| θ (risk) | Risk attitude — which strategy table backs decisions | [−0.05, 0.1] | 0 |
| β (precision) | Decision precision — inverse temperature of action selection | [0.5, 10] | ∞ |
| γ (myopia) | Discount on future value — how far ahead the player looks | [0.3, 1.0] | 1.0 |
| d (depth) | Search depth — number of future turns considered | {8, 20, 999} | 999 |

θ affects which pre-computed strategy table (state values) is used. β controls how deterministically the player follows Q-values (β → ∞ is perfectly optimal, β → 0 is random). γ discounts future state values (γ < 1 makes the player myopic — undervaluing moves that pay off later). d adds noise to the Q-value computation (lower d = shallower lookahead = noisier values).

**Q-value computation.** For each scenario, Q-values are pre-computed across a 6θ × 6γ × 3d = 108 parameter grid. The Q-grid maps parameter combinations to action-value vectors, enabling rapid likelihood evaluation during estimation. Q-values always use EV-mode computation — θ only selects the strategy table, not the aggregation method. This matches the simulation model: players are modeled as noisily optimizing expected value under a θ-shifted strategy table.

**Scenario generation.** 30 quiz scenarios are selected via diversity-constrained stratified sampling:

1. Simulate 100K games with a noisy agent (β=3, γ=0.85, σ_d=4) to visit realistic game states
2. At each decision point, collect the full Q-grid and identify scenarios where different parameter values prescribe different optimal actions
3. Filter by realism (plausible upper score, consistent scoring pattern) and minimum visit count (≥100)
4. Build a master pool (~595 candidates) by scoring for discriminative power
5. Select 30 scenarios via semantic bucketing (3 game phases × 3 decision types × 4 tension types = 36 buckets) with action-fatigue constraints (no action label in top-2 more than 4 times) and fingerprint deduplication (blocks functionally equivalent scenarios with identical top-3 actions and EVs)

**Estimation.** Given a player's 30 answers, the estimator fits (θ, β, γ, d) by maximum likelihood using multi-start Nelder-Mead optimization (8 start points × 3 d values = 24 optimizations). A weak log-normal prior on β centered at 3 prevents drift in flat regions. The estimator runs in JavaScript for client-side estimation (no server needed).

**Validation (Monte Carlo).** 100 synthetic trials across 6 archetypes (cautious, impulsive, myopic, strategic, random, expert) confirm parameter recovery:

| Parameter | Median RMSE | Threshold | Status |
|---|---|---|---|
| θ | 0.033 | < 0.2 | PASS |
| β | 1.79 | < 2.0 | PASS |
| γ | 0.197 | < 0.2 | PASS |
| d | 86% accuracy | ≥ 60% | PASS |

Known limitation: when β > 4 and d = 999, the player is near-deterministic and β becomes poorly identifiable (many β values produce similar likelihoods).


## 2. Surrogate Policy Compression: How Many Parameters for EV-Optimal Play?

The EV-optimal policy is stored as a 2M-entry lookup table (~8 MB). How small a model can replicate it? We trained decision trees and MLPs of varying sizes on 200K games (3M decisions per type), evaluating by EV loss: the mean score lost per game from suboptimal decisions, weighted by the gap between the best and second-best action at each decision point.

### 2.1 Training Data

The solver simulates 200K games under θ = 0, recording every decision point. At each, it stores a feature vector (dice face counts, turn number, upper score, bonus status, category availability), the optimal action, and the decision gap (V_best − V_second). Three decision types are exported separately:

| Decision type | Records | Features | Classes | Mean gap | Zero-gap rate |
|---|---:|---:|---:|---:|---:|
| Category | 3M | 29 | 15 | 6.38 | 6.8% |
| Reroll 1 | 3M | 30 | 32 | 1.15 | 1.5% |
| Reroll 2 | 3M | 30 | 32 | 2.15 | 2.9% |

Category decisions have the largest mean gap because choosing the wrong category typically costs more than a suboptimal reroll. Zero-gap decisions (where two actions are tied) are common at reroll steps — keeping [3,3] vs [5,6] often differs by < 0.01 points.

### 2.2 Decision Trees: The Pareto Frontier

Decision trees dominate the Pareto frontier at all scales. Selected results:

| Depth | Params | Cat acc | Cat EV loss | R1 acc | R1 EV loss | R2 acc | R2 EV loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | ~91 | 47.3% | 36.7 | 51.5% | 8.5 | 56.0% | 8.4 |
| 10 | ~2K | 83.7% | 6.6 | 72.1% | 2.8 | 82.5% | 2.1 |
| 15 | ~30K | 93.4% | 2.1 | 86.9% | 0.9 | 93.3% | 0.4 |
| 20 | ~130K | 96.6% | 0.9 | 93.3% | 0.4 | 96.2% | 0.2 |
| Full | ~280K | 98.0% | 0.5 | 95.4% | 0.2 | 96.8% | 0.2 |

**Combined EV loss** (sum across all three decision types) for a depth-15 tree: ~3.4 pts/game. For depth-20: ~1.4 pts/game. For the full (unconstrained) tree: ~0.9 pts/game.

The depth-15 tree (30K params, ~120 KB) achieves 93%+ accuracy on all three decision types and loses only 3.4 points per game — a 1.4% penalty on mean score of 248. This is a 67× compression vs the 2M-entry lookup table.

### 2.3 MLPs: Competitive but Not Dominant

MLPs trained on 300K-record subsamples (sklearn MLPClassifier, 50 epochs, Adam):

| Architecture | Params | Cat EV loss | R1 EV loss | R2 EV loss |
|---|---:|---:|---:|---:|
| [32, 16] | ~2K | 3.9 | 4.8 | 3.8 |
| [64, 32] | ~5K | 2.4 | 2.3 | 1.7 |
| [128, 64] | ~14K | 1.9 | 1.2 | 0.8 |
| [128, 64, 32] | ~15K | 1.8 | 0.9 | 0.7 |
| [256, 128, 64] | ~50K | 1.3 | 0.5 | 0.4 |

At matched parameter counts, MLPs lag decision trees. A 14K-param MLP achieves EV loss comparable to a depth-10 DT (~2K params) for category decisions. The gap narrows for reroll decisions, where the MLP's ability to learn soft boundaries helps: the [128, 64, 32] MLP (15K params) matches a depth-15 DT (31K params) on reroll1.

MLPs would likely improve with full training data (2.4M vs 300K) and more epochs, but the current results show that tree-based models are the natural fit for this combinatorial decision space.

### 2.4 Feature Importance

Random forest feature importance (top 5 features per decision type):

| Rank | Category | Reroll 1 | Reroll 2 |
|---:|---|---|---|
| 1 | max_face_count (8.0%) | face6 (14.3%) | face6 (16.3%) |
| 2 | dice_sum (7.8%) | dice_sum (11.5%) | dice_sum (12.5%) |
| 3 | num_distinct (7.2%) | face5 (8.5%) | max_face_count (11.6%) |
| 4 | face6 (7.2%) | max_face_count (7.7%) | num_distinct (9.1%) |
| 5 | face5 (6.7%) | face1 (5.6%) | face5 (8.2%) |

Reroll decisions depend heavily on the count of 6s — the highest-value face drives reroll strategy. Category decisions distribute importance more evenly, reflecting the broader context needed (which categories remain, bonus status, turn number). The 15 category-availability bits collectively contribute ~30% of importance for category decisions, confirming that knowing which categories remain is essential.

### 2.5 Key Findings

**Compression ratio.** A depth-20 decision tree (~130K combined params across 3 types, ~500 KB) achieves < 1.5 pts/game EV loss. The full lookup table uses 2M entries (8 MB). This is a **16× compression** with < 0.6% performance loss.

**EV-perfect threshold.** No tested model achieves < 0.1 pts/game EV loss (functionally perfect). The minimum combined loss is ~0.9 pts/game at ~280K total tree params. Reaching true EV-perfection likely requires ~500K–1M params, or alternative architectures (gradient-boosted trees, larger MLPs with full training data).

**Category decisions are hardest per parameter.** At 2K params: category EV loss = 6.6, reroll2 EV loss = 2.1. The 15-class action space with complex interactions between upper-bonus tracking and category availability demands more capacity than the 32-class reroll space where dice patterns are more locally informative.

**Decision trees beat MLPs at all tested scales.** This is likely because the optimal policy contains many hard boundaries (e.g., "if Yatzy is available and max_face_count ≥ 4, always keep") that axis-aligned splits capture exactly. MLPs waste capacity learning soft approximations to these step functions.

### 2.6 Game-Level Evaluation: How Small Can a Human-Level Player Be?

Per-decision EV loss is a useful proxy, but doesn't account for error compounding — a wrong category choice changes the game state, causing cascading suboptimal decisions. To measure *actual* game performance, we simulate full games using trained surrogate models for all 3 decision types (category, reroll1, reroll2), computing real mean scores rather than proxy estimates.

**Results (10K games, seed=42):**

| Model | Total params | Mean | Std | p5 | p50 | p95 | Bonus % |
|---|---|---|---|---|---|---|---|
| heuristic | 0 | 166 | 32.4 | 116 | 164 | 221 | 1.2% |
| dt_d1 | 12 | 133 | 25.9 | 94 | 131 | 179 | 0.1% |
| dt_d3 | 66 | 142 | 32.3 | 94 | 138 | 201 | 2.1% |
| dt_d5 | 276 | 157 | 37.3 | 103 | 151 | 223 | 9.4% |
| dt_d8 | 1,878 | 192 | 43.4 | 129 | 188 | 271 | 35.6% |
| dt_d10 | 6,249 | 216 | 44.4 | 145 | 219 | 290 | 54.0% |
| mlp_64 | 11,023 | 221 | 43.1 | 146 | 224 | 291 | 66.8% |
| dt_d15 | 81,237 | 239 | 42.1 | 161 | 241 | 305 | 77.0% |
| dt_d20 | 412,629 | 245 | 40.6 | 168 | 246 | 308 | 84.4% |
| optimal | 2,097,152 | 248 | 39 | — | 248 | — | ~87% |

**Key findings.**

**Error compounding is modest.** The EV-loss proxy predicts dt_d20 should score ~248.4 − 1.4 = 247.0. Actual game mean is 244.6–247.6, suggesting ~1–3 points of compounding cost. The proxy is a good (slight over-) estimate of actual performance.

**Human-level play (mean 220–230) requires ~6K–15K parameters.** The dt_d10 combo (6,249 params) achieves mean 216, and mlp_64 (11,023 params) achieves 221, firmly entering the human range. This means human-level Yatzy strategy can be encoded in roughly 25–60 KB of decision tree parameters.

**The heuristic baseline (0 params, mean 166) is surprisingly weak.** It's worse than even a depth-8 DT (1,878 params, mean 192). This confirms that most of the heuristic's deficit comes from category selection — greedy "take what looks good now" without considering future state value is very costly. The heuristic achieves only 1.2% upper bonus rate vs 54% for dt_d10.

**Upper bonus is the biggest differentiator.** The bonus rate correlation with mean score is striking: 0.1% → 133 mean, 35.6% → 192, 54% → 216, 84.4% → 245. Models that fail to secure the upper bonus consistently score 80+ points below optimal. The 50-point bonus acts as a step function in performance.

**DTs maintain their advantage at the game level.** dt_d15 (81K params, mean 239) outperforms mlp_128_64 (42K params, mean 241) at lower parameter count but by a small margin. At the game level, MLPs close the gap slightly compared to per-decision EV loss metrics, possibly because MLP errors are less correlated across decisions.

### 2.7 Diagnosing the EV Loss Floor

The unlimited-depth decision tree (dt_full) achieves a combined 0.885 pts/game EV loss across all 3 decision types (category: 0.497, reroll1: 0.230, reroll2: 0.158). This raises the question: what causes this irreducible floor? Is it feature representation limits, insufficient training data, or label noise?

**Experiment 1: Label noise quantification.** We group all training records by their feature vector and check for conflicting labels (same features, different optimal action). Result: **zero conflicts** across all 3 decision types, out of 1.66M–1.72M unique feature vectors per type. The 29/30-feature representation is lossless — every unique game state maps to exactly one optimal action. The EV loss floor is not caused by label ambiguity.

**Experiment 2: Error distribution analysis.** We analyze where dt_full makes its ~2–5% errors:

| Decision Type | Error Rate | Mean Gap | Near-zero (<0.1) | Near Bonus |
|---|---|---|---|---|
| category | 2.04% | 1.629 | 12.4% | 53.4% |
| reroll1 | 4.56% | 0.337 | 34.9% | 65.6% |
| reroll2 | 3.21% | 0.328 | 41.3% | 61.1% |

Category errors are few but expensive (mean gap 1.63 pts), while reroll errors are more frequent but cheaper. Over half of all errors occur near the upper bonus threshold (upper score 40–62), where decisions are most sensitive to bonus considerations. For reroll decisions, 35–41% of errors have near-zero gap (<0.1), meaning many "errors" barely matter.

**Experiment 3: Data scaling.** Training dt_full on increasing subsets of the 200K-game dataset:

| Games | Category EV Loss | Reroll1 EV Loss | Reroll2 EV Loss |
|---|---|---|---|
| 25K | 1.335 | 0.565 | 0.366 |
| 50K | 0.925 | 0.416 | 0.285 |
| 100K | 0.646 | 0.291 | 0.203 |
| 150K | 0.505 | 0.240 | 0.161 |
| 160K | 0.497 | 0.230 | 0.158 |

EV loss is still declining at 160K training games with no plateau. Category decisions benefit most from additional data (dropping from 1.34 at 25K to 0.50 at 160K). Exporting 400K+ games should reduce the EV floor further, potentially by 0.1–0.3 pts.

**Experiment 4: Feature ablation.** Removing feature groups one at a time from dt_d20:

| Removed Group | Δ Category | Δ Reroll1 | Δ Reroll2 |
|---|---|---|---|
| turn | −0.019 | +0.025 | +0.001 |
| upper_state | +1.134 | +0.266 | +0.135 |
| dice_counts | +11.980 | +6.039 | +6.895 |
| dice_summary | +0.646 | +0.022 | +0.116 |
| category_avail | +13.631 | +2.932 | +2.773 |
| rerolls_rem | — | −0.001 | −0.001 |

`dice_counts` and `category_avail` are by far the most important features. Removing either causes catastrophic loss. `upper_state` matters for category decisions (bonus-sensitive). `dice_summary` (sum, max count, distinct count) is partially redundant with dice_counts but still contributes to category decisions. `turn` is surprisingly unimportant — slightly *improves* category decisions when removed. `rerolls_rem` is completely redundant (zero impact) because reroll1 and reroll2 have separate models trained on fixed reroll counts.

**Experiment 5: Engineered features.** Adding 26 derived features (15 category scores, 8 pattern indicators, best available score, upper opportunity, categories remaining) to the base feature set and training dt_d20:

| Config | Category EV Loss | Reroll1 EV Loss | Reroll2 EV Loss |
|---|---|---|---|
| base (29/30 features) | 0.866 | 0.360 | 0.189 |
| augmented (55/56 features) | 1.987 | 0.397 | 0.220 |

Augmented features **hurt** at fixed depth 20. The expanded feature space gives the DT more splitting candidates, diluting its limited depth budget across redundant features. This is consistent with the ablation finding that the base features already encode the necessary information — the DT can derive category scores and patterns implicitly via multi-level splits on face counts. Feature engineering is not the path to lower EV loss; more training data and deeper trees are.

**Experiment 6: Forward feature selection.** Greedy forward selection (add one feature per step, train dt_d15 on 300K records, evaluate on full test set) reveals the feature importance ordering:

For category decisions, the first 5 features selected are `max_face_count`, `dice_sum`, `face4`, `face3`, `cat5_avail` — dice pattern features dominate early. The EV loss drops rapidly from 59.3 (1 feature) through 19.6 (5 features) to 2.4 (24 features), with diminishing returns after ~20 features. The last 5 features (`face6`, `upper_score`, `bonus_secured`, `turn`, `upper_cats_left`) add essentially nothing.

For reroll decisions, a similar pattern holds: dice features come first, followed by category availability. Notably, `rerolls_remaining` and `turn` rank near the bottom for all types, confirming the ablation findings.

All types show a clear elbow at 20–23 features, beyond which additional features provide <1% improvement. The forward selection confirms that the base feature set is well-chosen but contains 5–7 redundant features that could be dropped without meaningful loss.

**Conclusions.** The 0.89 pt/game EV loss floor is caused by (a) the tree depth limit constraining the model's ability to represent the ~1.7M unique decision states, and (b) finite training data (still improving at 160K games). The features themselves are lossless and well-chosen. The most productive next step is exporting 500K+ games for training.


## 3. Negative Results

### 3.1 State-Dependent θ(s): The Bellman Equation Already Handles It

**Hypothesis.** A state-dependent policy θ(s), varying risk attitude based on upper-section bonus proximity, can beat the constant-θ Pareto frontier by ≥1 point of mean at matched variance.

**Motivation.** The cost of risk-seeking varies by state. When the bonus is secured, risk is cheap. When the bonus is at risk, risk is expensive. A fixed θ pays the full mean penalty uniformly but gains upside only in cheap-risk states.

**Four adaptive policies tested (1M games each):**

| Policy | Mean | σ | Frontier μ at matched σ | Δμ |
|---|---:|---:|---:|---:|
| bonus-adaptive | 246.92 | 39.6 | 247.83 | −0.91 |
| phase-based | 246.56 | 40.4 | 247.38 | −0.82 |
| combined | 247.93 | 39.0 | 248.15 | −0.23 |
| upper-deficit | 246.22 | 40.8 | 247.20 | −0.98 |

No adaptive policy beat the frontier. All landed on or slightly below it (Δμ between −0.23 and −0.98, against a significance threshold of 1.0 at SE ≈ 0.038).

**Why this fails.** The Bellman equation for fixed θ is:

$$V_\theta(s) = \max_a \left[ r(s,a) + \mathbb{E}[V_\theta(s')] \right]$$

(in log-utility domain). The successor value V_θ(s') assumes the same θ for all future decisions. This self-consistency is what makes the solution optimal — it is the fixed point of the Bellman operator.

An adaptive policy switching from θ₁ on turn t to θ₂ on turn t+1 consults V_{θ₁}(s') but then plays according to V_{θ₂}. The consulted value is wrong:

- When θ₁ > θ₂: overestimates future risk-taking gains that will not materialize
- When θ₁ < θ₂: underestimates them

The error is directionally negative. The adaptive policy switches to higher θ in states where risk is "cheap," but the higher-θ value table was computed assuming high risk-tolerance in all states, including expensive-risk states. The mismatch means the policy slightly overvalues high-θ turns, leading to marginally suboptimal decisions.

**The constant-θ solver already exhibits state-dependent behavior.** V\*(s) at turn 5 already encodes that certain upper-section placements lead to high bonus probability. The solver plays "conservatively" near the threshold and "aggressively" when the bonus is lost — not because θ varies, but because the value function captures all downstream consequences. Varying θ mid-game adds no new information; it only introduces inconsistency.

### 3.2 RL Approaches

Three RL architectures attempted to beat the fixed-θ Pareto frontier.

**Approach A: θ-table switching with REINFORCE.** An MLP (5K params) selects which of 5 precomputed θ-tables to use each turn. Observations include upper score, bonus health, total score z-score. Trained with quantile-tilted reward on 1M episodes.

**Approach C: θ-table switching with PPO.** Same architecture with 12 θ-tables, PPO algorithm, and 19K-parameter actor-critic. Trained on 2M episodes.

**Results for A and C:** Both converged to single fixed-θ behaviors. Each λ (reward-tilting parameter) produced a policy matching a specific fixed-θ table almost exactly. No policy beat the Pareto frontier. The per-turn signal from θ-switching (~1 point) is buried in episode-level variance (σ ≈ 38), yielding SNR ≈ 1/38.

**Approach B: Direct action with IQN.** An Implicit Quantile Network (66K params) directly selects reroll masks and categories. Behavioral cloning from θ = 0 expert achieved 66% reroll accuracy and 81% category accuracy. Game performance: mean 205.3, 43 points below expert.

The accuracy-performance gap follows from error compounding. With per-decision accuracy p, the probability of a correct full turn is p³ (two rerolls + one category). Over 15 turns: 0.73⁴⁵ ≈ 10⁻⁶. To match the expert within 5 points requires per-decision accuracy > 99%.

A 66K-parameter network cannot approximate an 8MB lookup table. The compression ratio is 128×, far below what is needed for the combinatorial space of 252 dice sets × 32 reroll masks × varying game states.

Online RL fine-tuning degraded the BC policy catastrophically (mean 190 → 123 within 20K episodes) due to a death spiral: slight policy degradation fills the replay buffer with bad transitions.

**Summary:**

| Approach | Architecture | Mean | p95 | vs Frontier |
|---|---|---:|---:|---|
| A (REINFORCE) | Turn-level table switch | 239–248 | 309–312 | On frontier |
| C (PPO) | Turn-level table switch | 246.9 | 310 | On frontier |
| B (IQN+BC) | Per-decision neural | 205.3 | 281 | Far below |

The constant-θ Pareto frontier is empirically tight. The state-dependent θ experiments (Section 3.1) provide the strongest evidence: four adaptive policies that explicitly condition on game state all land within 1 point of the frontier (Δμ between −0.23 and −0.98). The RL results are consistent with this but do not independently establish it — the RL algorithms may have failed for reasons unrelated to the size of the exploitable gap.
