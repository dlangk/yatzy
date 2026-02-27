:::section{#compression}

## Compression

The solver produces a 2-million-entry lookup table --a complete map from
every reachable game state to the optimal action. It is perfect and utterly
opaque. No human could memorise it, and no simple rule set can replicate it.
But how small a model can approximate it? This is the question of
::concept[supervised distillation]{supervised-distillation}:
compressing an oracle's knowledge into a learnable function.

The natural first instinct is reinforcement learning --let an agent
play millions of games and learn from reward. But Yatzy presents a severe
::concept[credit assignment]{credit-assignment}
problem. A single game spans 15 rounds with roughly 45 decisions. The final
score is a single number. When the agent scores poorly, which of the 45
decisions was the mistake? The reward is sparse, delayed, and entangled
across a long horizon. Standard RL algorithms (PPO, DQN) struggle to
propagate credit through this many steps without enormous sample budgets,
and even then they plateau well below optimal play.

:::html
<div class="chart-container" id="chart-rl-barrier-diagram"><div id="chart-rl-barrier-diagram-svg"></div></div>
:::

Supervised distillation sidesteps credit assignment entirely. The solver
generates 200,000 games under optimal play, recording every decision point:
the game state, the optimal action, and the value gap between the best and
second-best action. This produces roughly 3 million labelled examples per
decision type (category selection, first reroll, second reroll). A surrogate
model then learns to map states to actions by pure classification, with no
need to discover the reward structure.

The results reveal a clear winner:
::concept[decision trees]{decision-tree}
dominate MLPs at every parameter budget. A depth-20 decision tree ensemble
(413K total parameters across three decision types) achieves a mean game
score of 245 --just 3 points below the optimal 248. A comparably-sized
MLP achieves only 241. The gap widens at smaller scales: at 2,000 parameters,
a depth-10 decision tree scores 216 while a two-layer MLP of similar size
manages 221 with five times the parameters.

:::html
<div class="chart-container" id="chart-surrogate-pareto"><div id="chart-surrogate-pareto-svg"></div></div>
:::

Why do trees win? The optimal policy contains many hard boundaries: "if Yatzy
is available and you have four of a kind, always keep." These axis-aligned
splits are exactly what decision trees learn. MLPs must waste capacity learning
smooth approximations to step functions. The combinatorial, discrete nature of
the action space --15 categories, 32 possible keeps --favours the
tree's ability to carve the feature space into crisp regions.

The compression curve reveals where human-level play sits. A mean score of
220–230 (the range most experienced human players achieve) requires
roughly 6,000 to 15,000 parameters. Below this, the upper bonus rate collapses:
a depth-8 tree (1,878 parameters) secures the bonus in only 35.6% of games
versus 84.4% for the depth-20 tree. The 50-point bonus is the single biggest
differentiator between weak and strong play, and a model must have enough
capacity to track the upper-section score and route decisions accordingly.
This is a concrete expression of
::concept[resource rationality]{resource-rationality}:
human cognitive limits place us at a specific point on the Pareto frontier
between model complexity and playing strength.

:::html
<div class="chart-container" id="chart-compression-long-tail"><div id="chart-compression-long-tail-svg"></div></div>
:::

At the other end, even the best surrogate (a full unconstrained tree at 280K
parameters) still loses 0.9 points per game to the lookup table. The residual
errors cluster near the upper bonus threshold --states where the score
is between 40 and 62 and the bonus decision is razor-thin. These are the
hardest decisions in the game, and they resist compression because they depend
on subtle interactions between remaining categories, current upper score, and
future reroll distributions.

### The Rosetta Stone

The solver's 16 MB strategy table encodes the optimal action for every reachable
state. No human can read it. Can we distill that knowledge into something a
person could memorize? A
::concept[sequential decision list]{sequential-decision-list}
of 100 category-selection rules --each a simple IF-THEN with one or two
conditions --recovers a mean score of 227.5 when paired with optimal
rerolls. Against the oracle's 248.4, that closes 59.7% of the gap, leaving a
40.3% incompressible residual of 20.9 EV points.

The rules emerge from greedy sequential covering over 370 candidate conditions
drawn from 56 semantic features (dice topology, bonus tracking, category
availability). Each condition is a simple threshold or equality test; rules
combine at most two conditions via conjunction. The algorithm searches
single conditions, then top-20 by top-30 conjunctions, selecting whichever
minimizes mean regret over uncovered states. Early rules are obvious ---
"if you have a large straight, score it." By rule 30, the algorithm discovers
structural strategies that experienced players miss: when to sacrifice
upper-section points for a better bonus trajectory, how to arbitrage Full
House against Chance, and which upper category to prioritize when multiple
score equally.

The scaling curve has three phases. The first 20 rules add only 2.7 EV,
picking off zero-regret corner cases like Yatzy and completed straights.
Then the curve accelerates: rules 26–40 deliver peak marginal value
of 0.74 EV/rule, discovering the structural core of Yatzy ---
upper-section management, Sixes accumulation, One Pair as the high-frequency
workhorse category. After 70 rules, returns diminish to 0.15 EV/rule as the
algorithm mops up context-dependent edge cases. The key landmarks: 0 rules
gives 196.7, 50 rules gives 216.5, 100 rules gives 227.5. Notably, roughly
67 rules suffice to surpass a trained neural network baseline at 221 EV.

:::html
<div class="chart-container" id="chart-rosetta-rules"><div id="chart-rosetta-rules-svg"></div></div>
:::

Reroll rules are harder. The natural action space is a 5-bit position bitmask,
but "keep positions 0, 1, 2" means entirely different things for [1,1,1,4,6]
versus [2,3,4,5,6]. Bitmask rules achieve only 168.7 EV. The fix is a
::concept[composable filter grammar]{composable-filter-grammar}
of 15 semantic actions --"keep the triple," "keep face 6," "keep the
straight draw" --that describe intent rather than position. Each verb
compiles to the correct bitmask at runtime. This recovers 184.4 EV, a +15.7
improvement that confirms position-dependence was the bottleneck.

:::html
<div class="chart-container" id="chart-filter-grammar"><div id="chart-filter-grammar-svg"></div></div>
:::

Yet semantic rules (184.4) still trail category-only mode (227.5) by 43 points.
Reroll decisions are inherently more speculative --they require reasoning
about potential future states, not just the current hand. The 15-verb vocabulary
cannot represent every oracle decision, and the 56 features may not carry
enough signal for a 100-rule list to approximate forward-looking reroll
strategy. The practical ceiling for this rule-based framework may be
category selection alone, with reroll decisions remaining in the domain
of lookup tables or learned surrogates.

The incompressible residual of 20.9 EV concentrates near the upper bonus
threshold (upper scores 40–62), where the value function is discontinuous.
Over half of all surrogate model errors land in this zone. No short rule list
captures the fine-grained state-dependence where a single category assignment
swings the 50-point bonus. This is the knowledge that resists compression
into language.

:::math

### Training Data and Feature Representation

The training data is generated by simulating 200K games under &theta; = 0
optimal play. At each decision point, the feature vector encodes:

:::html
<table>
  <thead>
    <tr><th>Feature Group</th><th class="num">Count</th><th>Description</th></tr>
  </thead>
  <tbody>
    <tr><td>Dice face counts</td><td class="num">6</td><td>Count of each face value (1–6)</td></tr>
    <tr><td>Dice aggregates</td><td class="num">4</td><td>Sum, max count, distinct faces, max face</td></tr>
    <tr><td>Category availability</td><td class="num">15</td><td>Binary: is each category still open?</td></tr>
    <tr><td>Game state</td><td class="num">4</td><td>Turn number, upper score, bonus achieved, rerolls left</td></tr>
  </tbody>
</table>
:::

This 29–30 feature representation is verified to be *lossless*:
across 1.7 million unique feature vectors per decision type, zero label
conflicts exist. Every unique game state maps to exactly one optimal action.

### The Pareto Frontier

:::html
<table>
  <thead>
    <tr><th>Model</th><th class="num">Params</th><th class="num">Mean Score</th><th class="num">Bonus %</th><th class="num">EV Loss</th></tr>
  </thead>
  <tbody>
    <tr><td>Heuristic</td><td class="num">0</td><td class="num">166</td><td class="num">1.2%</td><td class="num">82.6</td></tr>
    <tr><td>DT depth-5</td><td class="num">276</td><td class="num">157</td><td class="num">9.4%</td><td class="num">53.7</td></tr>
    <tr><td>DT depth-8</td><td class="num">1,878</td><td class="num">192</td><td class="num">35.6%</td><td class="num">11.7</td></tr>
    <tr><td>DT depth-10</td><td class="num">6,249</td><td class="num">216</td><td class="num">54.0%</td><td class="num">11.5</td></tr>
    <tr><td>MLP [64,32]</td><td class="num">5,000</td><td class="num">221</td><td class="num">66.8%</td><td class="num">6.4</td></tr>
    <tr><td>DT depth-15</td><td class="num">81,237</td><td class="num">239</td><td class="num">77.0%</td><td class="num">3.4</td></tr>
    <tr><td>DT depth-20</td><td class="num">412,629</td><td class="num">245</td><td class="num">84.4%</td><td class="num">1.4</td></tr>
    <tr><td>Optimal (lookup)</td><td class="num">2,097,152</td><td class="num">248</td><td class="num">~87%</td><td class="num">0</td></tr>
  </tbody>
</table>
:::

Error compounding is modest: the EV-loss proxy predicts the depth-20 tree
should score 247.0, and actual game simulation yields 245 --roughly
2 points of cascading cost from correlated errors.

### Why MLPs Lag

At matched parameter counts, MLPs trail decision trees on category decisions
--- the hardest task (15 classes, complex bonus interactions). A 14K-param
MLP achieves category EV loss comparable to a 2K-param depth-10 tree. The gap
narrows for reroll decisions, where the MLP's soft decision boundaries help
with near-tied actions. But overall, the combinatorial structure of Yatzy
strongly favours axis-aligned partitioning over learned embeddings.

### Rosetta Scaling Law

The scaling law follows a three-phase structure. Let <var>N</var> be the number
of category rules and EV(<var>N</var>) the mean score under category-only
evaluation with optimal rerolls:

:::equation
EV(0) = 196.7 &nbsp;&rarr;&nbsp; EV(50) = 216.5 &nbsp;&rarr;&nbsp; EV(100) = 227.5
:::

Phase 1 (<var>N</var> = 0–20): foundation, +2.7 EV. Phase 2
(<var>N</var> = 25–70): acceleration, +21.9 EV with peak marginal value
of 0.74 EV/rule in the 26–40 band. Phase 3 (<var>N</var> = 70–100):
saturation, +6.2 EV. The phase transition at <var>N</var> &approx; 25
coincides with the algorithm's discovery of upper-section management rules
--- the structural backbone of good Yatzy play.

The greedy covering algorithm operates on a condition space of 370 candidates
generated from the 56-feature DSL: boolean features produce `== 0`
and `== 1` conditions, discrete features produce `==`,
`≥`, `≤` comparisons, and continuous features use
percentile thresholds at p10/p25/p50/p75/p90. Condition masks are pre-computed
as a dense (370 × 300K) uint8 matrix, enabling vectorized coverage
counting via `dot(mask, alive)`. Each rule selects the condition (or
two-condition conjunction from top-20 × top-30 candidates) that minimizes
mean regret over uncovered records.

The 15 semantic reroll verbs decompose into three families: face-specific
(Keep Face 1–6, 6 actions), pattern-based (Keep Pair, Keep Two Pairs,
Keep Triple, Keep Quad, Keep Triple+Highest, Keep Pair+Kicker, 6 actions),
and structural (Reroll All, Keep Straight Draw, Keep All, 3 actions). Projection
from 32-column bitmask regret to 15-column semantic regret is lossy: when no
semantic action matches the oracle's optimal bitmask, all 15 columns
receive a penalty value of 999.0. The regret-minimizing rule selector then
picks the least-bad semantic option.

:::

:::code

### Generating Training Data

```bash
# Export decision records from 200K optimal games
YATZY_BASE_PATH=. solver/target/release/yatzy-export-training-data \
    --games 200000 \
    --output data/training/

# Produces three files:
#   category_decisions.csv   (~3M records, 29 features)
#   reroll1_decisions.csv    (~3M records, 30 features)
#   reroll2_decisions.csv    (~3M records, 30 features)
```

### Training a Decision Tree Ensemble

```python
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Train separate trees for each decision type
for dtype in ['category', 'reroll1', 'reroll2']:
    df = pd.read_csv(f'data/training/{dtype}_decisions.csv')
    features = [c for c in df.columns if c not in ['action', 'gap']]

    dt = DecisionTreeClassifier(
        max_depth=20,
        min_samples_leaf=5,
        class_weight='balanced',
    )
    dt.fit(df[features], df['action'], sample_weight=df['gap'])
    # Weight by gap: mistakes on high-gap decisions cost more
```

### Error Analysis: Where the Last Point Hides

The irreducible error floor (0.9 pts/game for the full tree) concentrates
near the upper bonus threshold. At upper scores between 40 and 62, 53.4%
of category errors occur --these are states where a 1-point
difference in upper score determines whether the 50-point bonus is
achievable. For reroll decisions, 35–41% of errors have near-zero
gap (&lt;0.1 points), meaning many "errors" are functionally irrelevant
ties.

```python
# Error distribution near bonus threshold
errors_near_bonus = errors[errors['upper_score'].between(40, 62)]
print(f"Errors near bonus: {len(errors_near_bonus) / len(errors):.1%}")
# Category: 53.4%, Reroll1: 65.6%, Reroll2: 61.1%

# Near-zero gap errors (functional ties)
trivial = errors[errors['gap'] < 0.1]
print(f"Trivial errors: {len(trivial) / len(errors):.1%}")
# Category: 12.4%, Reroll1: 34.9%, Reroll2: 41.3%
```

### Rosetta Rule Pipeline

The four-phase pipeline runs in approximately 12 minutes on an M1 Max:

```bash
# Phase 1-2: Export regret data (200K games, 8 threads)
YATZY_BASE_PATH=. solver/target/release/yatzy-regret-export \
  --games 200000                           # 8.9s, produces ~4 GB

# Phase 3: Greedy rule induction (Python)
yatzy-analyze induce-rules \
  --subsample 300000 --max-rules 100       # 673s for 3 x 100 rules

# Phase 4: Evaluate category-only (1M games)
YATZY_BASE_PATH=. solver/target/release/yatzy-eval-policy \
  --rules outputs/rosetta/rules_cat.json \
  --mode category-only --games 1000000     # 21.6s, EV = 227.5

# Evaluate full semantic rules (1M games)
YATZY_BASE_PATH=. solver/target/release/yatzy-eval-policy \
  --rules outputs/rosetta/rules_all.json \
  --mode full --games 1000000              # 2.5s, EV = 184.4
```

The Rust evaluator compiles semantic actions to position-specific bitmasks at
runtime. The `compile_to_mask(dice, action)` function maps each of
the 15 `SemanticRerollAction` variants to the appropriate 5-bit mask
given sorted dice. For example, `KeepTriple` with dice
`[2,5,5,5,6]` produces mask `0b10001` (reroll positions
0 and 4, keep positions 1–3). The compilation is O(1) per action and
handles edge cases (no valid triple, no pair, etc.) by falling back to
reroll-all.

The top 5 induced category rules illustrate the zero-regret foundation:

```
1. IF has_large_straight         THEN Large Straight   (regret 0.0000)
2. IF face_count_1 == 5          THEN Yatzy             (regret 0.0000)
3. IF categories_left <= 1      THEN Four of a Kind    (regret 0.0000)
4. IF face_count_1 == 4          THEN Ones              (regret 0.0000)
5. IF !cat_avail_fives AND
      face_count_5 >= 5         THEN Yatzy             (regret 0.0000)
```

:::

:::
