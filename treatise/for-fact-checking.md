# The Mathematics of Optimal Yatzy: Fact-Check Document

This document contains the prose from Sections 1 through 4 of the treatise, with all chart data and referenced statistics inlined for external verification. All numerical claims can be cross-referenced against the data tables below.

Note: This treatise covers Scandinavian Yatzy, not American Yahtzee. Scandinavian Yatzy has 15 scoring categories (Yahtzee has 13), awards a 50-point upper bonus (Yahtzee gives 35), and has no Yahtzee bonus.

---

## Section 1: The Family Game

I loved playing Yatzy as a kid. In fact, I still do. Yatzy, for me, is closely associated with having fun as a family. In particular, I associate it with my wonderful dad; he is the best person I know to play Yatzy with!

Yatzy has really interesting properties that make it fun to both play and pick apart. To understand the game, let's start by taking a look at what happens if an optimal Yatzy player sits down and plays a lifetime worth of games. I estimate that's about two thousand games.

[CHART: Score Spray. ~2,000 games displayed as a scatter plot (subsampled from 10,000 simulated games to represent a lifetime of play). Each dot is one game's total score.]

As you can see, scores vary wildly even for an optimal player. Even if you play perfectly, 1 out of 10 games will be below 203 points. And you will, during an entire lifetime, experience maybe a handful of games above 320 points.

This explains why Yatzy is fun for families: It has room for skill, but it's actually mostly luck! The best player will win slightly more, but anyone can get lucky and beat the strongest player by a wide margin. Parents will win a bit more, but their kids will regularly beat them.

Most family-games require some mixture of skill and luck. The more skill, the less likely it is that your entire family will enjoy it together. For example, playing chess isn't really fun unless both players are about as good. There is zero luck involved in chess, just pure skill.

At the same time, too much luck and you don't feel like you earned your win anymore. There needs to be decisions that matter, and outcomes to celebrate. Yatzy strikes a nice balance. When you win, you feel skilled. When you lose, you were unlucky.

Many years ago, I got it into my head to figure out: exactly how much skill is involved in Yatzy? Or put differently: how good can you be at Yatzy? Along the way, it also turned out that Yatzy was a good workbench for testing AI technologies. A few of the things that makes Yatzy interesting are:

- The Bonus Cliff. The hardest thing about Yatzy is that you get a bonus based on your upper score. This makes reinforcement learning much harder to use successfully.
- The Significant Inherent Randomness. Five dice with six sides that are rolled over and over again means a whole lot of randomness.
- The Huge State Space. There are about 10^197 possible Yatzy games. That's significantly larger than chess. And the universe. There are about 10^80 atoms in the universe.
- It Can Still Be Solved. Clever mathematicians have figured out how to reduce the state space drastically, and as a result, Yatzy can be solved. I've implemented my own solver based on this math, and used AI to push that solver to extreme performance.

### Score Spray Data (backing the chart)

Source: 10,000 Monte Carlo games under optimal play (theta=0).

Summary statistics (from 10K MC sample; canonical exact values from forward-DP are mean=248.4, std=38.5):
- Mean: 248.2 (MC sample estimate; exact DP value is 248.4)
- Std Dev: 38.6
- Median: 248
- Range: 101 to 344

Population breakdown (from score_spray_meta.json):

| Sub-population | Fraction | Mean |
|----------------|----------|------|
| No bonus, no Yatzy | 6.06% | 164.0 |
| No bonus, Yatzy | 4.15% | 214.5 |
| Bonus, no Yatzy | 55.73% | 236.3 |
| Bonus + Yatzy | 34.06% | 286.7 |

---

## Section 2: The State Space

A Yatzy game is a sequence of rolls, keeps, and scoring decisions across fifteen rounds. The total number of possible games is roughly 10^197. That vastly exceeds the number of atoms in the observable universe, estimated at "only" about 10^80.

How do you solve something that big? You don't. You find ways to make it smaller. This section walks through five simplifications that shrink the problem from impossibly large to something a laptop can handle in about a second.

### Step 1: Order Doesn't Matter

When you roll five dice, their "order" don't change anything. A roll of [3,5,2,6,4] is the same as [2,3,4,5,6]. This immediately means the 7,776 ways to roll five six-sided dice can be collapsed into just 252 unique combinations. These 252 unique rolls can then be grouped into seven patterns. Finally, there are up to 32 ways to choose which dice to keep, depending on the dice pattern.

[CHART: Dice Symmetry. Interactive explorer of the 252 unique rolls grouped into seven family patterns.]

**Verification: 7,776 = 6^5. 252 = C(10,5) = C(5+6-1, 5).**

### Step 2: Forget the Past

The second big insight is: that (most of) the past doesn't matter. Suppose two games took completely different paths; you scored different categories in different orders, but ended up with the same upper-section total and the same categories remaining. As long as those two things are the same, the best strategy for the rest of the game is identical. All history except scored categories and total upper score is irrelevant. It doesn't matter if you scored a good One Pair or a bad one. It doesn't even matter what your total score is.

A game state is fully described by just two things: the upper-section score so far (actually, all you need to know is the number between 0 to 63, since anything above the bonus threshold is equivalent) and which of the 15 categories have been used. That gives 64 x 2^15 = 2,097,152 possible states. Between turns, you are in one of these ~2.1M states.

**Verification: 64 x 32,768 = 2,097,152. Upper score caps at 63 because the bonus threshold is 63. Maximum raw upper sum = 5+10+15+20+25+30 = 105, but values in [63,105] are strategically equivalent.**

### Step 3: Prune the Impossible

Not all of those ~2.1M states can actually occur in a real game. For example, an upper score of 60 is impossible if you've only scored Ones and Twos. By checking which states are actually reachable, we can prune 31.8%, leaving ~1.43M.

[CHART: Reachability Grid. Heatmap showing reachable vs impossible states.]

**Reachability data:**

```json
{
  "total_states": 2097152,
  "reachable_states": 1430000,
  "pruned_fraction": 0.318
}
```

Reachable states by layer (number of categories scored):

NOTE: The "Reachable" column in reachability.json contains approximate (rounded) values used for chart visualization. The per-layer counts do not match exact combinatorial computation. For example, layer 1 (1 category scored) has exactly 45 reachable states (6 upper categories x 6 possible scores each + 9 lower categories x 1 upper-score value), not 340. The rounded values sum to ~1,415,205, which does not match the stated total of 1,430,000. The per-layer breakdown should be recomputed from the solver's exact reachability DP for precise verification.

The aggregate claims in the treatise prose (~1.43M reachable states, 31.8% pruned) are correct.

| Categories Scored | Total States | Reachable (approx) |
|-------------------|-------------|-------------------|
| 0 (Start) | 64 | 1 |
| 1 | 960 | ~340 (exact: 45) |
| 2 | 6,720 | ~3,200 |
| 3 | 29,120 | ~16,500 |
| 4 | 87,360 | ~55,000 |
| 5 | 192,192 | ~130,000 |
| 6 | 320,320 | ~225,000 |
| 7 | 411,840 | ~290,000 |
| 8 | 411,840 | ~280,000 |
| 9 | 320,320 | ~210,000 |
| 10 | 192,192 | ~120,000 |
| 11 | 87,360 | ~52,000 |
| 12 | 29,120 | ~26,000 |
| 13 | 6,720 | ~6,200 |
| 14 | 960 | ~900 |
| 15 (All scored) | 64 | 64 |

**Verification: Total column sums to 2,097,152. Total states per layer = C(15, k) x 64. Per-layer reachable counts are approximate and need recomputation.**

### Step 4: One Turn at a Time

Now we know there are ~1.43M possible states we can visit while playing. Next, we need to understand how we can transition between these states. Every transition works the same: roll all dice, keep some and roll the rest, then keep some and roll the rest again, then finally score a category. Every turn starts in one of the ~1.43M states and transitions to another through these intermediate steps. We refer to this transition mechanism as a widget.

[CHART: Widget Structure. The six layers of a widget, from initial roll to final scoring decision.]

Let's look more closely at each widget. The 252 unique rolls were explained in Step 1. But where does 462 come from? After each roll, you choose which dice to keep. With five dice there are 2^5 = 32 ways to select which dice to keep. But for most sets of dice, many of those 32 choices lead to the same outcome. What matters is how many of each face you keep, not which physical dice you picked.

[CHART: Keep Equivalence. Shows how 32 keep choices reduce to fewer unique keeps for a given roll.]

[CHART: Keep Funnel. Visualization of how the 32 unique keep choices reduce into fewer unique keeps.]

For a Full House, 32 possible keep choices reduce to just 12 unique keeps. But how many unique keeps exist in total? Since dice showing the same face are interchangeable, it becomes a multiset combination problem:

| Dice kept | Unique keeps | Example |
|-----------|-------------|---------|
| 0 | 1 | (reroll all) |
| 1 | 6 | keep one 3 |
| 2 | 21 | keep 3, 3 |
| 3 | 56 | keep 1, 3, 5 |
| 4 | 126 | keep 2, 3, 3, 5 |
| 5 | 252 | (keep all) |
| **Total** | **462** | |

**Verification: Each row = C(5+k, k) where k = dice kept. 1+6+21+56+126+252 = 462 = C(11,5). Widget node count: 1 + 252 + 462 + 252 + 462 + 252 = 1,681.**

### Step 5: Always Forward

There is one final structural property that simplifies Yatzy: it is "one-directional", i.e. you can never "unscore" a category. This means the ~1.43M states can be layered. The first layer has no categories scored, the second layer has exactly one scored, and so on up to the last layer where the game is over. Every transition goes strictly from one layer to the next. This is the final reduction of the state space. With this, we are ready to start solving.

From 10^197 to 1.43 million. Ignoring dice order cuts 7,776 rolls to 252 unique combinations. Ignoring history further reduces the space down to ~2.1M states. Pruning unreachable states removes another ~32%. Breaking each turn into a widget makes every state independently solvable. And because the graph is directed and acyclic, each layer depends only on the next.

---

## Section 3: The Solver

We have established that there are ~1.43M reachable states, organized into 16 layers, connected by widgets. Every transition goes strictly forward (from layer L to layer L+1), and layer 16 is trivial: just check whether the upper bonus was earned. This one-directional structure is necessary for us to be able to solve the game because it allows us to work backward: solve layer 16 first (trivial), then use those answers to solve layer 15, then layer 14, and so on until we reach layer 1.

[CHART: Backward Cascade. Visual of the 16-layer backward induction sweep.]

### Solving a Widget

To compute the value of a state, the solver evaluates its widget: the complete decision tree of one turn. The trick is to work backward through the turn: start from the last decision (which category to score) and reason back toward the first roll. Since the solver already knows the value of every end state (since we already solved the next layer) and the probability of every dice outcome, it can compute the expected score of every possible choice.

[CHART: Widget Interactive. Step through the solver's backward reasoning, then play a turn with live evaluation.]

### Scoring A Category

The solver starts at the end of the turn. After the last reroll the player must choose which category to score. Each unscored category gives an immediate score plus a future value (the expected score from the resulting state, already computed in a later layer). The solver evaluates every option and picks the best:

```
for each final roll (252 possibilities):
    best = -infinity
    for each unscored category:
        value = score(roll, category) + future_value(next state)
        best = max(best, value)
    scoring_ev[roll] = best
```

### Choosing What to Keep

One step back, the player has five dice and needs to decide which to keep. Each possible keep leads to a probability distribution over possible outcomes. The expected score of a keep is the weighted sum of the value of all possible outcomes:

```
for each roll (252):
    best = -infinity
    for each way to keep dice:
        exp = SUM P(keep -> reroll) x scoring_exp[reroll]
        best = max(best, exp)
    keep_exp[roll] = best
```

The solver runs this logic twice: once for the second reroll (using scoring expected scores as input) and once for the first reroll (using second-reroll expected scores as input).

### Transition Probabilities

A transition probability answers the question: given the dice I have decided to keep, what is the probability distribution over possible outcomes of rerolling? Each reroll depends on what you kept from the previous one, so the probability of any specific three-roll sequence is the product of three individual probabilities, one per roll.

[CHART: Path Probability. Interactive tool to set dice values and see joint probability of a three-roll path.]

The chain at the bottom shows what is happening: three probabilities being multiplied together. Most specific sequences of events are very rare in Yatzy.

[CHART: Transition Matrix. The 462 x 252 keep-to-outcome transition probability matrix.]

**Transition matrix properties:**
- Dimensions: 462 keeps x 252 outcomes
- Non-zero entries: 4,368 (out of 116,424 total = 3.75% dense / 96.25% sparse)
- Storage: Compressed Sparse Row (CSR) format, ~35 KB

### The Keep Shortcut

There's a trick that makes the solver a lot faster. Many different rolls lead to the same keeping decision. "Keep my two threes" from (1,3,3,5,6) and from (2,3,3,4,4) are the same situation: two threes in hand, three free dice to reroll. The reroll distribution is identical. The solver exploits this by splitting the work:

```
Step 1: Compute expected score for each unique keep (462)
    for each unique keep h:
        keep_ev[h] = SUM P(h -> outcome) x prev_exp[outcome]

Step 2: For each roll, pick the best keep (252 lookups)
    for each roll r:
        keep_exp[r] = max over valid keeps h of r: keep_ev[h]
```

### The First Roll

Before any dice are thrown, we don't know which of the 252 possible rolls will appear. Each roll r has a probability P(r) (its multinomial coefficient divided by 6^5 = 7,776). The state's expected value is the weighted average:

```
E(state) = SUM P(r) x keep_exp[r]
```

### The Backward Sweep

All ~1.43M widgets are solved in the same way, starting from layer 16 and working backward to layer 1. Each layer depends only on the next, and states within a layer are independent of each other, which allows for complete parallelization.

462 shared keeps make the inner loop tractable. Without the keep shortcut, each of the 252 rolls would need its own set of sparse dot products against the reroll distribution. Sharing the computation across the 462 unique keep-multisets reduces the work by an order of magnitude, turning a minutes-long solve into a one-second sweep.

### Memory Budget

| Structure | Size | Purpose |
|-----------|------|---------|
| State values | 16 MB | One f32 per state slot (4,194,304 x 4 bytes; stride-128 padding for branchless access) |
| Precomputed scores | 15 KB | 252 dice sets x 15 categories x 4 bytes |
| Sparse keep table | 35 KB | 4,368 non-zero (f32, i32) pairs + row pointers |
| Reverse dice lookup | 32 KB | 5D index_lookup for O(1) dice to index |
| Keep frequency lookup | 182 KB | Horner hash table (46,656 entries) |
| Reachability mask | 4 KB | 64 x 64 booleans |
| Batched buffers | 247 KB/thread | Two ping-pong [f32;64] x 252 + keep_ev |

Total working set during precomputation: <17 MB.
During serving (mmap mode): 16 MB mapped, ~250 KB active.

**Performance:** Full backward sweep over 1.43 million reachable states completes in 1.10 seconds on an Apple M1 Max with 8 performance cores.

**Verification: 4,194,304 = 128 x 32,768 (STATE_STRIDE=128 for topological padding). 128 x 4 bytes = 512 bytes per scored-category mask. 15 KB = 252 x 15 x 4 bytes. 46,656 = 6^6.**

---

## Section 4: The Optimal Strategy

The optimal strategy exists in the form of ~1.43M numbers calculated by the solver. That means to play perfectly, we need ~1.43M parameters. In this section, we will explore what this strategy looks like, and see if there is anything we can learn in order to get better at playing Yatzy!

Since we cannot comprehend ~1.43M parameters, we need to find other ways to see how the optimal strategy behaves. We will start by searching for patterns. To do that, we simulate 1 million games played based on the optimal strategy. Why 1 million? Statistical precision scales with the square root of sample size. At 1 million games, averages are accurate to within a fraction of a point, and even rare events (like scoring zero on One Pair, which happens less than 0.1% of the time) show up hundreds of times. Going to 10 million would barely improve the precision while taking 10x longer to run. Going down to 100,000 would leave some of the rarer statistics too noisy to trust.

### The Shape of the Outcome

Let's start by looking at the big picture: The score distribution.

[CHART: Mixture Decomposition. Score distribution under optimal play (1M games). The 16 colored components are the sub-populations from four binary events: Bonus, Yatzy, Small Straight, Large Straight. Dashed line: true (exact) distribution.]

**Headline statistics (exact forward-DP, not Monte Carlo):**

| Statistic | Value |
|-----------|-------|
| Mean | 248.4 |
| Median | 249 |
| Std Dev | 38.5 |
| P5 | 179 |
| P95 | 309 |

**Full exact distribution (density_exact.json):**

Source: Forward-DP exact PMF computation (zero variance, not Monte Carlo).

```
theta: 0
mean: 248.4400669879
variance: 1481.9728864272
std_dev: 38.4964009542

Percentiles:
  p1:  145
  p5:  179
  p10: 203
  p25: 225
  p50: 249
  p75: 276
  p90: 299
  p95: 309
  p99: 325

Score range: 10 to 374 (non-zero probability)
```

**Claims to verify against this data:**
- "1 out of 10 games will be below 203 points" (Section 1) -> p10 = 203. Correct.
- Mean 248.4 -> 248.4400669879. Correct (rounded).
- Median 249 -> p50 = 249. Correct.
- Std Dev 38.5 -> 38.4964009542. Correct (rounded).
- P5 = 179. Correct.
- P95 = 309. Correct.

It's clearly not a normal distribution. The shape has bumps and shoulders that a single bell curve cannot explain. The reason: four categories in Yatzy are binary. You either score them or you don't, and each one shifts your total by a fixed amount. The upper-section bonus adds 50 points (hit ~90% of the time). Yatzy adds 50 points (~39%). Large Straight adds 20 points (~49%). Small Straight adds 15 points (~26%).

**Verification of binary event rates (from category_landscape.json):**
- Bonus hit rate: ~90% (derived from mixture data: sum of bonus=true fractions = 0.2062+0.1993+0.0671+0.0787+0.129+0.1203+0.0452+0.0521 = 0.8979 = ~90%)
- Yatzy hit rate: 0.388 = ~39%. Text says "~39%". Correct.
- Large Straight hit rate: 0.486 = ~49%. Text says "~49%". Correct.
- Small Straight hit rate: 0.2634 = ~26%. Text says "~26%". Correct.
- Yatzy ceiling: 50 points. Correct.
- Large Straight ceiling: 20 points. Correct.
- Small Straight ceiling: 15 points. Correct.
- Bonus: 50 points. Correct.

These four yes/no events create 16 sub-populations. Each sub-population is roughly Gaussian in its "residual" score (the non-binary categories), centered around 154 to 177 points, then shifted upward by whichever binary events fired. The peaks in the overall distribution appear where multiple sub-populations pile up at similar total scores. For example, the shoulder near 254 is where "Bonus + Large Straight + no Yatzy" (mean 244) overlaps with "Bonus + Small Straight + Large Straight" (mean 262) and "Bonus + Yatzy only" (mean 270).

**16-component mixture data (mixture.json):**

| # | Bonus | Yatzy | Sm.Str | Lg.Str | Fraction | Mean | Std |
|---|-------|-------|--------|--------|----------|------|-----|
| 1 | No | No | No | No | 0.0316 | 154.2 | 15.9 |
| 2 | No | No | No | Yes | 0.0173 | 174.7 | 16.4 |
| 3 | No | No | Yes | No | 0.0081 | 169.0 | 16.1 |
| 4 | No | No | Yes | Yes | 0.0038 | 190.0 | 16.7 |
| 5 | No | Yes | No | No | 0.0212 | 202.6 | 16.1 |
| 6 | No | Yes | No | Yes | 0.0117 | 222.8 | 16.4 |
| 7 | No | Yes | Yes | No | 0.0057 | 217.2 | 16.4 |
| 8 | No | Yes | Yes | Yes | 0.0028 | 237.6 | 16.8 |
| 9 | Yes | No | No | No | 0.2062 | 220.2 | 14.6 |
| 10 | Yes | No | No | Yes | 0.1993 | 243.5 | 14.2 |
| 11 | Yes | No | Yes | No | 0.0671 | 236.7 | 14.7 |
| 12 | Yes | No | Yes | Yes | 0.0787 | 261.6 | 13.5 |
| 13 | Yes | Yes | No | No | 0.1290 | 269.8 | 15.4 |
| 14 | Yes | Yes | No | Yes | 0.1203 | 293.1 | 15.2 |
| 15 | Yes | Yes | Yes | No | 0.0452 | 286.4 | 15.5 |
| 16 | Yes | Yes | Yes | Yes | 0.0521 | 311.5 | 14.8 |

**Verification of prose claims about sub-populations:**
- "centered around 154 to 177 points" -> Residual means (subtracting binary contributions): Pop 1 = 154.2 (no events), Pop 9 = 220.2 - 50(bonus) = 170.2, Pop 13 = 269.8 - 50 - 50 = 169.8, Pop 16 = 311.5 - 50 - 50 - 15 - 20 = 176.5. Range of residuals: 154.2 to 176.5. Correct.
- "Bonus + Large Straight + no Yatzy" (mean 244) -> Pop 10: mean 243.5. Text says 244. Close (rounded).
- "Bonus + Small Straight + Large Straight" (mean 262) -> Pop 12: mean 261.6. Text says 262. Close (rounded).
- "Bonus + Yatzy only" (mean 270) -> Pop 13: mean 269.8. Text says 270. Close (rounded).
- All 16 fractions sum to: 0.0316+0.0173+0.0081+0.0038+0.0212+0.0117+0.0057+0.0028+0.2062+0.1993+0.0671+0.0787+0.129+0.1203+0.0452+0.0521 = 1.0001. Correct (rounding).

### Understanding the Patterns

[CHART: Category Landscape. Each bubble is one of the 15 categories with multiple statistical dimensions.]

**Category landscape data (category_landscape.json):**

| Category | Section | Ceiling | Mean Score | Zero Rate | Hit Rate | Var. Contrib. | Mean Fill Turn | Score % Ceiling | Bonus Dep. | Mean (Bonus) | Mean (No Bonus) |
|----------|---------|---------|------------|-----------|----------|---------------|----------------|-----------------|------------|-------------|----------------|
| Ones | upper | 5 | 2.18 | 0.0937 | 0.9063 | -3.8 | 8.04 | 0.4358 | -0.09 | 2.17 | 2.26 |
| Twos | upper | 10 | 5.67 | 0.0053 | 0.9947 | 1.9 | 7.60 | 0.5666 | 0.41 | 5.71 | 5.29 |
| Threes | upper | 15 | 9.29 | 0.0023 | 0.9977 | 9.3 | 6.96 | 0.6195 | 1.13 | 9.41 | 8.28 |
| Fours | upper | 20 | 12.79 | 0.0022 | 0.9978 | 18.0 | 6.58 | 0.6397 | 1.98 | 13.00 | 11.02 |
| Fives | upper | 25 | 16.24 | 0.0019 | 0.9981 | 26.5 | 6.13 | 0.6497 | 2.66 | 16.52 | 13.85 |
| Sixes | upper | 30 | 19.97 | 0.0023 | 0.9977 | 44.3 | 6.48 | 0.6658 | 4.18 | 20.40 | 16.22 |
| One Pair | lower | 12 | 11.18 | 0.0008 | 0.9992 | 2.2 | 7.14 | 0.9321 | 0.16 | 11.20 | 11.04 |
| Two Pairs | lower | 22 | 19.38 | 0.0128 | 0.9872 | 17.4 | 6.60 | 0.8811 | 0.56 | 19.44 | 18.88 |
| Three of a Kind | lower | 18 | 15.51 | 0.0417 | 0.9583 | 27.6 | 8.73 | 0.8617 | 1.05 | 15.62 | 14.57 |
| Four of a Kind | lower | 24 | 13.66 | 0.2875 | 0.7125 | 131.2 | 10.96 | 0.5693 | 4.39 | 14.11 | 9.72 |
| Small Straight | lower | 15 | 3.95 | 0.7366 | 0.2634 | 62.7 | 8.24 | 0.2634 | 1.07 | 4.06 | 2.99 |
| Large Straight | lower | 20 | 9.72 | 0.5140 | 0.4860 | 138.0 | 10.38 | 0.4860 | 3.07 | 10.03 | 6.97 |
| Full House | lower | 28 | 21.46 | 0.0844 | 0.9156 | 76.8 | 7.23 | 0.7665 | 1.50 | 21.61 | 20.12 |
| Chance | lower | 30 | 23.09 | 0.0000 | 1.0000 | 15.8 | 8.39 | 0.7697 | 0.79 | 23.17 | 22.38 |
| Yatzy | lower | 50 | 19.40 | 0.6120 | 0.3880 | 583.0 | 10.56 | 0.3880 | -0.92 | 19.30 | 20.23 |

**Verification of prose claims:**
- "scoring zero on One Pair, which happens less than 0.1% of the time" -> zero_rate = 0.0008 = 0.08%. Correct.

The next chart shows how category scores correlate.

[CHART: Category Correlations. Pairwise correlation between category scores (including bonus). Blue = negative, red = positive.]

**Correlation matrix (category_correlations.json):**

Categories in order: Ones, Twos, Threes, Fours, Fives, Sixes, Bonus, One Pair, Two Pairs, Three of a Kind, Four of a Kind, Small Straight, Large Straight, Full House, Chance, Yatzy.

|  | Ones | Twos | Threes | Fours | Fives | Sixes | Bonus | OnePr | TwoPr | 3oK | 4oK | SmStr | LgStr | FH | Chance | Yatzy |
|--|------|------|--------|-------|-------|-------|-------|-------|-------|-----|-----|-------|-------|-----|--------|-------|
| Ones | 1.0 | -0.035 | -0.057 | -0.086 | -0.102 | -0.118 | -0.021 | -0.038 | -0.028 | -0.030 | -0.024 | -0.052 | -0.027 | -0.003 | -0.043 | -0.063 |
| Twos | -0.035 | 1.0 | -0.067 | -0.092 | -0.094 | -0.112 | 0.068 | -0.010 | -0.009 | -0.021 | -0.004 | -0.015 | -0.006 | 0.008 | -0.009 | -0.015 |
| Threes | -0.057 | -0.067 | 1.0 | -0.075 | -0.091 | -0.114 | 0.149 | 0.008 | 0.008 | -0.010 | 0.021 | 0.008 | 0.020 | 0.013 | 0.007 | 0.001 |
| Fours | -0.086 | -0.092 | -0.075 | 1.0 | -0.090 | -0.104 | 0.219 | 0.019 | 0.013 | -0.002 | 0.047 | 0.028 | 0.043 | 0.018 | 0.028 | 0.013 |
| Fives | -0.102 | -0.094 | -0.091 | -0.090 | 1.0 | -0.108 | 0.255 | 0.024 | 0.023 | 0.020 | 0.070 | 0.045 | 0.060 | 0.028 | 0.047 | 0.019 |
| Sixes | -0.118 | -0.112 | -0.114 | -0.104 | -0.108 | 1.0 | 0.331 | 0.031 | 0.048 | 0.099 | 0.097 | 0.059 | 0.080 | 0.041 | 0.090 | 0.031 |
| Bonus | -0.021 | 0.068 | 0.149 | 0.219 | 0.255 | 0.331 | 1.0 | 0.041 | 0.049 | 0.078 | 0.138 | 0.049 | 0.093 | 0.060 | 0.106 | -0.012 |
| One Pair | -0.038 | -0.010 | 0.008 | 0.019 | 0.024 | 0.031 | 0.041 | 1.0 | 0.029 | -0.008 | -0.009 | -0.001 | 0.013 | -0.005 | 0.015 | -0.013 |
| Two Pairs | -0.028 | -0.009 | 0.008 | 0.013 | 0.023 | 0.048 | 0.049 | 0.029 | 1.0 | 0.001 | 0.019 | 0.014 | 0.028 | -0.009 | 0.058 | -0.003 |
| 3 of a Kind | -0.030 | -0.021 | -0.010 | -0.002 | 0.020 | 0.099 | 0.078 | -0.008 | 0.001 | 1.0 | 0.034 | 0.022 | 0.039 | 0.049 | 0.095 | -0.010 |
| 4 of a Kind | -0.024 | -0.004 | 0.021 | 0.047 | 0.070 | 0.097 | 0.138 | -0.009 | 0.019 | 0.034 | 1.0 | 0.051 | 0.084 | 0.040 | 0.103 | -0.033 |
| Sm Straight | -0.052 | -0.015 | 0.008 | 0.028 | 0.045 | 0.059 | 0.049 | -0.001 | 0.014 | 0.022 | 0.051 | 1.0 | 0.043 | 0.031 | 0.034 | 0.017 |
| Lg Straight | -0.027 | -0.006 | 0.020 | 0.043 | 0.060 | 0.080 | 0.093 | 0.013 | 0.028 | 0.039 | 0.084 | 0.043 | 1.0 | 0.053 | 0.080 | -0.007 |
| Full House | -0.003 | 0.008 | 0.013 | 0.018 | 0.028 | 0.041 | 0.060 | -0.005 | -0.009 | 0.049 | 0.040 | 0.031 | 0.053 | 1.0 | 0.034 | 0.000 |
| Chance | -0.043 | -0.009 | 0.007 | 0.028 | 0.047 | 0.090 | 0.106 | 0.015 | 0.058 | 0.095 | 0.103 | 0.034 | 0.080 | 0.034 | 1.0 | -0.010 |
| Yatzy | -0.063 | -0.015 | 0.001 | 0.013 | 0.019 | 0.031 | -0.012 | -0.013 | -0.003 | -0.010 | -0.033 | 0.017 | -0.007 | 0.000 | -0.010 | 1.0 |

Three patterns are visible:

First of all, the bonus row dominates. Bonus correlates with Sixes (+0.33), then Fives (+0.26), Fours (+0.22), and so on in value order. The bonus also correlates with lower-section categories. We know that the zero rate for Four of a Kind (+0.14) drops from 48% to 27% in games that secure the bonus. This is explained by the solver scoring zero in this category more often when still chasing the upper section bonus. In no-bonus games the solver fills Chance earlier (mean turn 7.3 vs 8.5) as a dump for mediocre rolls while still chasing upper categories. In bonus games, Chance can wait for a better roll.

**Verification of correlation claims:**
- Bonus-Sixes: 0.3307. Text says "+0.33". Correct.
- Bonus-Fives: 0.2553. Text says "+0.26". Correct (rounded).
- Bonus-Fours: 0.2186. Text says "+0.22". Correct (rounded).
- Bonus-Four of a Kind: 0.1375. Text says "+0.14". Correct (rounded).

Second, every upper-upper category pair shows negative correlation. The explanation for this is that upper categories compete for turns. If the solver spends an early turn scoring Sixes, that is one fewer turn available to retry Ones. Since the solver prioritizes high-value categories, the low-value ones get whatever is left. In general, the further apart two categories are in value, the stronger the tradeoff (for example, Ones-Sixes at -0.12 vs Ones-Twos at -0.04), though this is not perfectly monotonic across all pairs.

**Verification:** All upper-upper off-diagonal correlations are negative. Ones-Sixes = -0.118 (largest magnitude, furthest apart in value). The general trend holds on average by distance, but individual exceptions exist (e.g., Fives-Sixes = -0.108 is stronger than Fours-Sixes = -0.104 despite being closer).

Third, Yatzy is uncorrelated with almost everything. Hitting or missing Yatzy tells you essentially nothing about the rest of your game. It is a pure lottery ticket.

**Verification:** Yatzy row/column: all off-diagonal values between -0.063 and +0.031. Strongest is Yatzy-Ones = -0.063. All near zero. Confirmed.

The next chart shows when each category gets filled.

[CHART: Fill Turn Heatmap. Probability of filling each category on each turn, with toggles for bonus/no-bonus and zero/non-zero score filters.]

High-value upper categories, like Sixes, Fives and Fours, tend to be filled early, while Ones and Twos are deferred. Lower-section categories are bimodal: Two Pairs and Full House peak at turn 1 (grab them if you get them), while Four of a Kind and Large Straight peak at turn 15 (last resort).

**Verification of fill turn claims (from category_landscape.json):**
- Sixes mean fill turn: 6.48 (early). Correct.
- Fives mean fill turn: 6.13 (early). Correct.
- Fours mean fill turn: 6.58 (early). Correct.
- Ones mean fill turn: 8.04 (deferred). Correct.
- Twos mean fill turn: 7.60 (deferred relative to high-value). Correct.
- Two Pairs mean fill turn: 6.60 (early). Correct.
- Full House mean fill turn: 7.23 (early-ish). Correct.
- Four of a Kind mean fill turn: 10.96 (late). Correct.
- Large Straight mean fill turn: 10.38 (late). Correct.

The toggle reveals that what looks like one strategy is actually two different games superimposed. Since ~90% of games hit the bonus, the "Bonus scored" view is nearly identical to "All games." The real information is in "Bonus missed."

In no-bonus games, the solver fills lower-section categories earlier (Small Straight shifts from mean turn 8.4 to 6.5, One Pair from 7.3 to 6.1) to free up late turns for upper-section retries. The result: turn 15 belongs exclusively to upper categories (Sixes 24%, Fives 19%, Fours 19%), and the bonus row spikes to 93% at turn 15. These are not zero dumps. Sixes scored on turn 15 in no-bonus games average 9.5 points with only a 9% zero rate. The solver was genuinely trying to score well on these categories the whole game; the dice just never cooperated, so the upper categories kept getting pushed later until one landed on the final turn.

NOTE: The fill-turn heatmap has 9 filtered variants (3 bonus states x 3 score states). The specific numbers cited above (mean turn shifts in no-bonus games, turn-15 probabilities) come from the filtered heatmap data. The fill_turn_heatmap.json file is 39 KB and contains a "matrices" dict with keys like "all__all", "no_bonus__all", "bonus__all", "no_bonus__zeroed", "no_bonus__scored", "bonus__zeroed", "bonus__scored", "all__zeroed", "all__scored".

### Bonus Covariance

[CHART: Bonus Covariance. The 50-point bonus is actually worth 72 points.]

**Bonus covariance data (bonus_covariance.json):**

```json
{
  "total_gap": 72.0,
  "components": [
    { "label": "Bonus itself", "value": 50 },
    { "label": "Better upper scores", "value": 10.3 },
    { "label": "Better lower scores", "value": 11.7 }
  ]
}
```

The text states: "The bonus is worth 72, not 50." The rules say 50 points, but the true impact is 72. About 10 points come from the upper-section scores themselves being higher when the threshold is reached, and about 12 from genuine positive correlation between upper and lower sections.

**Verification:** Data says 10.3 (upper) + 11.7 (lower) = 22.0 extra points. Text correctly attributes ~10 to upper and ~12 to lower. Total gap = 50 + 10.3 + 11.7 = 72.0. Matches claim.

### Mixture Model (mathematical detail)

The four-component mixture model can be written as:

f(x) = SUM(i=1 to 4) pi_i * phi(x; mu_i, sigma_i)

where the mixing weights pi_i are determined by the joint probability of bonus hit/miss and Yatzy hit/miss. Under optimal play, the bonus hit rate is approximately 90% and the Yatzy probability is approximately 39%, giving mixing weights of roughly 0.061, 0.041, 0.551, and 0.347 for the four components (miss/miss, miss/hit, hit/miss, hit/hit).

**Verification of 4-component mixing weights (aggregated from 16-component data):**
- No bonus, no Yatzy: 0.0316+0.0173+0.0081+0.0038 = 0.0608. Text says "0.061". Correct (rounded).
- No bonus, Yatzy: 0.0212+0.0117+0.0057+0.0028 = 0.0414. Text says "0.041". Correct (rounded).
- Bonus, no Yatzy: 0.2062+0.1993+0.0671+0.0787 = 0.5513. Text says "0.551". Correct (rounded).
- Bonus, Yatzy: 0.129+0.1203+0.0452+0.0521 = 0.3466. Text says "0.347". Correct (rounded).

The 22-point correlated advantage can be decomposed further. Approximately 10 points come from the upper-section scores themselves being higher when the bonus is reached (conditional expectation above threshold vs below), and approximately 12 points come from a genuine positive correlation between upper-section luck and lower-section scoring opportunities.

**Verification:** See bonus_covariance.json above. Data says 10.3 (upper) + 11.7 (lower) = 22.0. Text correctly attributes ~10 to upper and ~12 to lower. Total = 22. Correct.

---

## Data Source Summary

All data in this document comes from two sources:

1. **Exact forward-DP computation** (density_exact.json): Zero-variance PMF over all possible games. Produces the exact mean (248.44), percentiles, and full score distribution. This is mathematically exact, not sampled.

2. **Monte Carlo simulation** (1,000,000 games under optimal play, theta=0): Produces per-category statistics (means, zero rates, fill turns, correlations), mixture decomposition, and fill-turn heatmaps. At 1M games, standard error of the mean is ~0.04 points.

The solver itself uses backward induction over ~1.43M reachable states with f32 arithmetic. The strategy table is a flat binary file of 4,194,304 x 4 bytes = ~16 MB.
