:::section{#state-space}

## The State Space

A Yatzy game is a sequence of rolls, keeps, and scoring decisions across fifteen rounds. The total number of possible games is roughly 10<sup>197</sup>. That vastly exceeds the number of atoms in the observable universe, estimated at "only" about 10<sup>80</sup>! 🤯

**How do you solve something that big? You don't.** You find ways to make it smaller. This section walks through five simplifications that shrink the problem from impossibly large to something a laptop can handle in about a second.

### Step 1: Order Doesn't Matter

When you roll five dice, their "order" don't change anything. A roll of ⚂⚄⚁⚅⚃ is the same as ⚁⚂⚃⚄⚅. This immediately means the 7,776 ways to roll five six-sided dice can be collapsed into just 252 unique combinations. These 252 unique rolls can then be grouped into seven patterns. Finally, there are up to 32 ways to choose which dice to keep, depending on the dice pattern. Roll the dice below to explore these patterns:

:::html
<div class="chart-container" id="chart-dice-symmetry">
  <p class="chart-caption">Roll dice and explore the 252 unique rolls, grouped into seven family patterns.</p>
</div>
:::

### Step 2: Forget the Past

The second big insight is: that (most of) the past doesn't matter. Suppose two games took completely different paths; you scored different categories in different orders, but ended up with the same upper-section total and the same categories remaining. As long as upper-section total and scored categories are the same, the best strategy for the rest of the game is **identical**. All history beyond that is irrelevant. It doesn't matter if you scored a good One Pair or a bad one.

**It doesn't even matter what your total score is.**

This means the game state is fully described by just two things: the upper-section score so far (actually, all you need to know is the number between 0 to 63, since anything above the ::concept[bonus]{upper-section-bonus} threshold is equivalent) and which of the 15 categories have been used. That gives 64 &times; 2<sup>15</sup> = 2,097,152 possible states. Between turns, you are always in one of these ~2.1M states.

### Step 3: Prune the Impossible

Not all of those ~2.1M states can actually occur in a real game. For example, an upper score of 60 is impossible if you've only scored Ones and Twos. By checking which states are actually reachable, we can prune 31.8%, leaving ~1.43M. The heatmap below shows which states exist (colored) and which are impossible (gray).

:::html
<div class="chart-container" id="chart-reachability-grid">
  <p class="chart-caption">Reachability heatmap: gray cells are impossible states, pruning 31.8% of the state space.</p>
</div>
:::

### Step 4: One Turn at a Time

Alright, now know there are ~1.43M possible states we can visit while playing. Next, we need to understand how we can *transition* between these states.

**Every turn starts in one of these states**, and it always transitions to another state using the same mechanism: roll all dices, keep some and roll the rest, then keep some and roll the rest again, then finally score a category.

If you roll a Yatzy on the first roll, we can just represent that as "keep all dices twice and then score Yatyz". We will refer this transition mechanism as a **widget** and we can visualize it like this:

:::html
<div class="chart-container" id="chart-widget-structure">
  <p class="chart-caption">The six layers of a widget, from initial roll to final scoring decision.</p>
</div>
:::

**Let's look more closely at each widget.** The 252 unique rolls were explained in Step 1. But where does 462 come from?

**After each roll, you choose which dice to keep.** With five dice there are 2<sup>5</sup>&nbsp;=&nbsp;32 ways to select which dice to keep. But for most sets of dices, many of those 32 choices lead to the same outcome. Say you rolled 3, 3, 3, 5, 5. Keeping dice #1 and #2, or #1 and #3, are different reroll choices but the result is the same: you will be keeping two Threes either way. What matters is *how many of each face* you keep, not *which physical dice* you picked.

:::html
<div id="chart-keep-equivalence"></div>
:::

The chart below starts with exactly that roll. Each colored cell is one of the 32 possible keep choices. If two keep choices have the same color that means they lead to the same unique keep. Hover a keep on the right to highlight which subsets map to it, or click Roll to try another combination.

:::html
<div class="chart-container" id="chart-keep-funnel">
  <p class="chart-caption">How the 32 unique keep choices reduce into fewer unique keeps.</p>
</div>
:::

For a Full House, 32 possible keep choices reduce to just 12 unique keeps. But how many unique keeps exist in total? Since dice showing the same face are interchangeable, it becomes a ::concept[multiset combination problem]{multiset-combination}. The table below enumerates every case:

| Dice kept | Unique keeps | Example |
|-----------|-------------:|---------|
| 0 | 1 | *(reroll all)* |
| 1 | 6 | keep one **3** |
| 2 | 21 | keep **3,&thinsp;3** |
| 3 | 56 | keep **1,&thinsp;3,&thinsp;5** |
| 4 | 126 | keep **2,&thinsp;3,&thinsp;3,&thinsp;5** |
| 5 | 252 | *(keep all)* |
| **Total** | **462** | |

Not every roll uses all 462. A Yatzy roll (e.g. five Threes) collapses to just 6 keeps. Only an all-different roll offers the full 32 choices. But 462 is the total number of strategically distinct keep decisions across all possible rolls: the complete branching factor the solver must reason over at each decision point. The math section below derives this via the stars-and-bars identity.

### Step 5: Always Forward

There is one final structural property that simplifies Yatzy: it is "one-directional", i.e. you can never "unscore" a category. This means the ~1.43M states can be layered. The first layer has no categories scored, the second layer has exactly one scored, and so on up to the last layer where the game is over. Every transition goes strictly from one layer to the next. This is the final reduction of the state space. With this, we are ready to start solving.

:::insight

**From 10<sup>197</sup> to 1.43 million.** Ignoring dice order cuts 7,776 rolls to 252 unique combinations. Ignoring history further reduces the space down to ~2.1M states. Pruning unreachable states removes another ~32%. Breaking each turn into a widget makes every state independently solvable. And because the graph is directed and acyclic, each layer depends only on the next. [1] [2] [3] [4]

:::

:::math

**State representation.** A game state is a tuple <var>S</var> = (<var>m</var>, <var>C</var>) where <var>m</var> &in; [0, 63] is the capped upper-section score and <var>C</var> &sube; {0, &hellip;, 14} is the set of scored categories, stored as a 15-bit bitmask. The upper score is capped because scores above 63 are strategically equivalent: the 50-point bonus is earned at &ge; 63, and nothing further depends on how far above the threshold you are. The maximum raw upper sum is 5 + 10 + 15 + 20 + 25 + 30 = 105, but all values in [63, 105] map to the same strategic state.

**State space size.** With 64 possible upper scores and 2<sup>15</sup> = 32,768 category masks:

:::equation
|<var>S</var>| = 64 &times; 2<sup>15</sup> = 2,097,152
:::

**Layered DAG.** The states form a directed acyclic graph with 16 layers, numbered 1 through 16. Layer <var>l</var> contains all states where |<var>C</var>| = <var>l</var> &minus; 1 (exactly <var>l</var> &minus; 1 categories scored). Every transition goes strictly from layer <var>l</var> to layer <var>l</var>+1, since scoring a category adds exactly one element to <var>C</var>. Layer 1 has a single category mask (the empty set), layer 16 has a single mask (all categories scored). The number of states per layer is:

:::equation
|layer <var>l</var>| = C(15, <var>l</var> &minus; 1) &times; 64
:::

**Dice multisets.** A roll of 5 dice from {1, &hellip;, 6} is represented as a frequency vector (<var>f</var><sub>1</sub>, &hellip;, <var>f</var><sub>6</sub>) where <var>f<sub>i</sub></var> counts face <var>i</var> and &sum; <var>f<sub>i</sub></var> = 5. The number of such vectors is the multiset coefficient:

:::equation
|<var>R</var><sub>5,6</sub>| = C(5 + 6 &minus; 1, 5) = C(10, 5) = 252
:::

**Keep multisets.** A keep is a sub-multiset of a roll: 0 to 5 dice chosen from {1, &hellip;, 6}. Since dice showing the same face are interchangeable, the number of distinct keeps of size <var>k</var> is the stars-and-bars multiset coefficient:

:::equation
C(6 + <var>k</var> &minus; 1, <var>k</var>) = C(5 + <var>k</var>, <var>k</var>)
:::

Summing over all possible keep sizes:

:::equation
|<var>K</var>| = &sum;<sub><var>k</var>=0</sub><sup>5</sup> C(5 + <var>k</var>, <var>k</var>) = 1 + 6 + 21 + 56 + 126 + 252 = 462 = C(11, 5)
:::

Many of the 2<sup>5</sup> = 32 binary keep/reroll masks per roll produce the same keep-multiset. For instance, keeping dice #1 and #2 from (3, 3, 3, 5, 5) is identical to keeping dice #1 and #3. This deduplication reduces work by roughly 47%.

**Widget structure.** Each turn decomposes into a six-layer sub-DAG. Let <var>D</var> = <var>R</var><sub>5,6</sub> (252 dice multisets), <var>H</var> = <var>K</var> (462 keep multisets), and <var>C</var><sub>avail</sub> = 15 &minus; |<var>C</var>| available categories. The layers alternate between chance nodes (dice outcomes) and decision nodes (keep or score choices):

:::equation
nodes/widget = 1 + |<var>D</var>| + |<var>H</var>| + |<var>D</var>| + |<var>H</var>| + |<var>D</var>| = 1 + 252 + 462 + 252 + 462 + 252 = 1,681
:::

**Reachability pruning.** Not every (<var>m</var>, <var>C</var>) pair can occur in a real game. A state is reachable only if its upper score <var>m</var> can be produced by some combination of scores from the upper categories already in <var>C</var>. Let <var>M</var> = <var>C</var> &cap; {0, &hellip;, 5} be the subset of scored upper categories. Reachability is computed via forward DP:

:::equation
<var>R</var>(0, &empty;) = true &nbsp;&nbsp;&nbsp; (base case)
:::

:::equation
<var>R</var>(<var>n</var>, <var>M</var>) = &exist; face <var>x</var> &in; <var>M</var>, count <var>c</var> &in; {0, &hellip;, 5} : <var>c</var> &middot; <var>x</var> &le; <var>n</var> &and; <var>R</var>(<var>n</var> &minus; <var>c</var> &middot; <var>x</var>, <var>M</var> &setminus; {<var>x</var>})
:::

For each upper-category face <var>x</var>, the player could have scored 0 to 5 copies of it (0 meaning the category was used but contributed nothing, which is only possible if the roll contained no <var>x</var>s). This DP runs over all 64 subsets of the 6 upper categories and all 64 upper scores. The result: 31.8% of states are pruned, leaving approximately 1,430,000 reachable states.

**Successor function.** When scoring category <var>c</var> on roll <var>r</var> from state (<var>m</var>, <var>C</var>), the successor state is:

:::equation
(<var>m</var>&prime;, <var>C</var>&prime;) = (min(<var>m</var> + <var>u</var>(<var>r</var>, <var>c</var>), 63), &nbsp; <var>C</var> &cup; {<var>c</var>})
:::

where <var>u</var>(<var>r</var>, <var>c</var>) is the upper-score contribution (non-zero only for categories 0 through 5, equal to the sum of matching face values). The min(·, 63) capping ensures the successor index stays within [0, 63].

:::

:::code

### Dice Multiset Enumeration ([source](https://github.com/dlangk/yatzy/blob/main/solver/src/phase0_tables.rs#L32))

All 252 sorted 5-dice multisets are enumerated once via nested loops with `b >= a, c >= b, ...`, guaranteeing no duplicates. A 5D reverse lookup table (`index_lookup[6][6][6][6][6]`, 7,776 entries, 32 KB) provides O(1) conversion from any sorted dice tuple back to its index.

```rust
// phase0_tables.rs — enumerate C(10,5) = 252 sorted multisets
for a in 1..=6 {
    for b in a..=6 {
        for c in b..=6 {
            for d in c..=6 {
                for e in d..=6 {
                    ctx.all_dice_sets[idx] = [a, b, c, d, e];
                    ctx.index_lookup[a-1][b-1][c-1][d-1][e-1] = idx;
                    idx += 1;
                }
            }
        }
    }
}
```

### Keep Deduplication via Horner Hashing ([source](https://github.com/dlangk/yatzy/blob/main/solver/src/phase0_tables.rs#L82))

Each keep-multiset is represented as a frequency vector [<var>f</var><sub>1</sub>, &hellip;, <var>f</var><sub>6</sub>] and hashed to a scalar via Horner's method: `key = ((((f1*6 + f2)*6 + f3)*6 + f4)*6 + f5)*6 + f6`. This maps into a flat lookup table of 6<sup>6</sup> = 46,656 entries, giving O(1) conversion from any frequency vector to its keep index. The table is sparse (only 462 entries are valid), but 46,656 integers is just 182 KB.

For each of the 252 dice sets, the solver iterates over all 31 non-trivial reroll masks (5-bit bitmasks), computes the resulting kept-dice frequency vector, looks up its keep index, and deduplicates. On average, 31 raw masks reduce to 16.3 unique keeps per dice set (47% reduction). The mapping is stored in `unique_keep_ids[252][31]`, which the hot path reads instead of recomputing.

### Precomputed Score Table ([source](https://github.com/dlangk/yatzy/blob/main/solver/src/phase0_tables.rs#L53))

All 252 &times; 15 = 3,780 category scores are precomputed into a flat `[[i32; 15]; 252]` array (15 KB). The hot path never evaluates scoring rules; it reads `precomputed_scores[dice_set][category]` directly. Scoring functions like "find the highest pair" involve reverse iteration and early exit, but they run only once during precomputation.

:::

:::
