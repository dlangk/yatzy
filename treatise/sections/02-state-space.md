:::section{#state-space}

## The State Space

A Yatzy game is a sequence of rolls, keeps, and scoring decisions across fifteen rounds. The total number of possible games is 1.7 &times; 10<sup>170</sup>. That vastly exceeds the number of atoms in the observable universe, estimated at "only" about 10^80! 🤯

How do you solve something that big? You don't. You find ways to make it smaller. This section walks through five simplifications that shrink the problem from impossibly large to something a laptop can handle in about a second.

### Step 1: Order Doesn't Matter

When you roll five dice, the order they land in doesn't change anything. A roll of ⚂⚄⚁⚅⚃ is the same as ⚁⚂⚃⚄⚅. This means the 7,776 ways to roll five six-sided dice collapse into just 252 unique rolls, which fall into seven families (like "three of a kind" or "two pair"). On top of that, there are at most 32 ways to choose which dice to keep. Roll the dice below to explore:

:::html
<div class="chart-container" id="chart-dice-symmetry">
  <p class="chart-caption">Roll dice and explore the 252 unique rolls, grouped into seven family patterns.</p>
</div>
:::

### Step 2: Forget the Past

The second big insight: the past doesn't matter. Suppose two games took completely different paths, scored different categories in different orders, but ended up with the same upper-section total and the same categories remaining. From here on, the best strategy is identical. The history is irrelevant.

This means a game position is fully described by just two things: the upper-section score so far (0 to 63, since anything above the ::concept[bonus]{upper-section-bonus} threshold is equivalent) and which of the 15 categories have been used. That gives 64 &times; 2<sup>15</sup> = 2,097,152 possible positions. A massive reduction from 10<sup>170</sup>.

### Step 3: Prune the Impossible

Not all of those 2 million positions can actually occur in a real game. For example, an upper score of 60 is impossible if you've only scored Ones and Twos. By checking which positions are actually reachable, we can throw out 31.8% of them, leaving about 1.43 million. The heatmap below shows which positions exist (colored) and which are impossible (gray).

:::html
<div class="chart-container" id="chart-reachability-grid">
  <p class="chart-caption">Reachability heatmap: gray cells are impossible positions, pruning 31.8% of the state space.</p>
</div>
:::

### Step 4: One Turn at a Time

Within each position, a single turn always follows the same structure: roll, keep some dice, reroll, keep again, reroll again, then pick a category. We call this six-step decision tree a ::concept[widget]{solve-widget}.

:::html
<div class="chart-container" id="chart-widget-structure">
  <p class="chart-caption">The six layers of a single-turn widget, from initial roll to final scoring decision.</p>
</div>
:::

The 252 unique rolls were explained in Step 1. But where does 462 come from? After each roll you choose which dice to keep. With five dice there are 2<sup>5</sup>&nbsp;=&nbsp;32 ways to tick boxes on a "keep or reroll" checklist. But many of those 32 choices lead to the same outcome. Say you rolled 3, 3, 3, 5, 5. Keeping dice #1, #2, #3 or keeping dice #1, #2, #5 are very different checklist ticks, but if #3 and #5 both show a 3, the result is identical: you're holding three Threes either way. What matters is *how many of each face* you hold, not *which physical dice* you picked.

:::html
<div class="keep-equiv">
  <div class="keep-equiv-row">
    <span class="keep-equiv-label">Mask A</span>
    <div class="keep-equiv-dice">
      <div class="keep-die"><svg viewBox="0 0 48 48"><rect x="1" y="1" width="46" height="46" rx="8" fill="var(--bg)" stroke="var(--accent)" stroke-width="3"/><circle cx="14" cy="14" r="4.5" fill="var(--text)"/><circle cx="24" cy="24" r="4.5" fill="var(--text)"/><circle cx="34" cy="34" r="4.5" fill="var(--text)"/></svg></div>
      <div class="keep-die"><svg viewBox="0 0 48 48"><rect x="1" y="1" width="46" height="46" rx="8" fill="var(--bg)" stroke="var(--accent)" stroke-width="3"/><circle cx="14" cy="14" r="4.5" fill="var(--text)"/><circle cx="24" cy="24" r="4.5" fill="var(--text)"/><circle cx="34" cy="34" r="4.5" fill="var(--text)"/></svg></div>
      <div class="keep-die"><svg viewBox="0 0 48 48"><rect x="1" y="1" width="46" height="46" rx="8" fill="none" stroke="var(--border)" stroke-width="2" stroke-dasharray="4 3"/><circle cx="14" cy="14" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="24" cy="24" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="34" cy="34" r="4.5" fill="var(--text)" opacity="0.2"/></svg></div>
      <div class="keep-die"><svg viewBox="0 0 48 48"><rect x="1" y="1" width="46" height="46" rx="8" fill="none" stroke="var(--border)" stroke-width="2" stroke-dasharray="4 3"/><circle cx="14" cy="14" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="34" cy="14" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="24" cy="24" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="14" cy="34" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="34" cy="34" r="4.5" fill="var(--text)" opacity="0.2"/></svg></div>
      <div class="keep-die"><svg viewBox="0 0 48 48"><rect x="1" y="1" width="46" height="46" rx="8" fill="none" stroke="var(--border)" stroke-width="2" stroke-dasharray="4 3"/><circle cx="14" cy="14" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="34" cy="14" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="24" cy="24" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="14" cy="34" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="34" cy="34" r="4.5" fill="var(--text)" opacity="0.2"/></svg></div>
    </div>
    <span class="keep-equiv-arrow">&rarr;</span>
    <div class="keep-equiv-result">
      <div class="keep-die"><svg viewBox="0 0 48 48"><rect x="1" y="1" width="46" height="46" rx="8" fill="var(--bg)" stroke="var(--accent)" stroke-width="3"/><circle cx="14" cy="14" r="4.5" fill="var(--text)"/><circle cx="24" cy="24" r="4.5" fill="var(--text)"/><circle cx="34" cy="34" r="4.5" fill="var(--text)"/></svg></div>
      <div class="keep-die"><svg viewBox="0 0 48 48"><rect x="1" y="1" width="46" height="46" rx="8" fill="var(--bg)" stroke="var(--accent)" stroke-width="3"/><circle cx="14" cy="14" r="4.5" fill="var(--text)"/><circle cx="24" cy="24" r="4.5" fill="var(--text)"/><circle cx="34" cy="34" r="4.5" fill="var(--text)"/></svg></div>
    </div>
  </div>
  <div class="keep-equiv-row">
    <span class="keep-equiv-label">Mask B</span>
    <div class="keep-equiv-dice">
      <div class="keep-die"><svg viewBox="0 0 48 48"><rect x="1" y="1" width="46" height="46" rx="8" fill="var(--bg)" stroke="var(--accent)" stroke-width="3"/><circle cx="14" cy="14" r="4.5" fill="var(--text)"/><circle cx="24" cy="24" r="4.5" fill="var(--text)"/><circle cx="34" cy="34" r="4.5" fill="var(--text)"/></svg></div>
      <div class="keep-die"><svg viewBox="0 0 48 48"><rect x="1" y="1" width="46" height="46" rx="8" fill="none" stroke="var(--border)" stroke-width="2" stroke-dasharray="4 3"/><circle cx="14" cy="14" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="24" cy="24" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="34" cy="34" r="4.5" fill="var(--text)" opacity="0.2"/></svg></div>
      <div class="keep-die"><svg viewBox="0 0 48 48"><rect x="1" y="1" width="46" height="46" rx="8" fill="var(--bg)" stroke="var(--accent)" stroke-width="3"/><circle cx="14" cy="14" r="4.5" fill="var(--text)"/><circle cx="24" cy="24" r="4.5" fill="var(--text)"/><circle cx="34" cy="34" r="4.5" fill="var(--text)"/></svg></div>
      <div class="keep-die"><svg viewBox="0 0 48 48"><rect x="1" y="1" width="46" height="46" rx="8" fill="none" stroke="var(--border)" stroke-width="2" stroke-dasharray="4 3"/><circle cx="14" cy="14" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="34" cy="14" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="24" cy="24" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="14" cy="34" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="34" cy="34" r="4.5" fill="var(--text)" opacity="0.2"/></svg></div>
      <div class="keep-die"><svg viewBox="0 0 48 48"><rect x="1" y="1" width="46" height="46" rx="8" fill="none" stroke="var(--border)" stroke-width="2" stroke-dasharray="4 3"/><circle cx="14" cy="14" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="34" cy="14" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="24" cy="24" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="14" cy="34" r="4.5" fill="var(--text)" opacity="0.2"/><circle cx="34" cy="34" r="4.5" fill="var(--text)" opacity="0.2"/></svg></div>
    </div>
    <span class="keep-equiv-arrow">&rarr;</span>
    <div class="keep-equiv-result">
      <div class="keep-die"><svg viewBox="0 0 48 48"><rect x="1" y="1" width="46" height="46" rx="8" fill="var(--bg)" stroke="var(--accent)" stroke-width="3"/><circle cx="14" cy="14" r="4.5" fill="var(--text)"/><circle cx="24" cy="24" r="4.5" fill="var(--text)"/><circle cx="34" cy="34" r="4.5" fill="var(--text)"/></svg></div>
      <div class="keep-die"><svg viewBox="0 0 48 48"><rect x="1" y="1" width="46" height="46" rx="8" fill="var(--bg)" stroke="var(--accent)" stroke-width="3"/><circle cx="14" cy="14" r="4.5" fill="var(--text)"/><circle cx="24" cy="24" r="4.5" fill="var(--text)"/><circle cx="34" cy="34" r="4.5" fill="var(--text)"/></svg></div>
    </div>
  </div>
  <span class="keep-equiv-result-label">Same keep</span>
</div>
:::

The chart below starts with exactly that roll. Each colored cell is one of the 32 raw subsets; same color means same unique keep. Hover a keep on the right to highlight which subsets map to it, or click Roll to try another combination.

:::html
<div class="chart-container" id="chart-keep-funnel">
  <p class="chart-caption">How 32 raw keep subsets collapse into unique keeps. Same color = same unique keep.</p>
</div>
:::

For this Full House roll, 32 subsets collapse to just 12 unique keeps. But the total across all 252 rolls is 462. Counting all possible holds by how many dice you keep:

| Dice kept | Unique keeps | Example |
|-----------|----------:|---------|
| 0 | 1 | *(reroll all)* |
| 1 | 6 | keep one **3** |
| 2 | 21 | keep **3,&thinsp;3** |
| 3 | 56 | keep **1,&thinsp;3,&thinsp;5** |
| 4 | 126 | keep **2,&thinsp;3,&thinsp;3,&thinsp;5** |
| 5 | 252 | *(keep all)* |
| **Total** | **462** | |

Not every roll uses all 462. A Yatzy roll (e.g. five Threes) collapses to just 6 keeps; an all-different roll stays at the full 32.

The solver doesn't need to think about all of this at once. It processes the widget one layer at a time, using two small buffers of 252 numbers each. It reads from one, writes to the other, then swaps. The total working memory per widget is about 2 KB, and when it's done it produces a single number: the expected score from this position.

### Step 5: Start from the End

Here's the trick that ties it all together. Because scoring a category always moves the game forward (you never un-score a category), we can group all 1.43 million positions into 16 layers: layer 0 has no categories scored, layer 1 has one scored, and so on up to layer 15 where the game is over. Each layer only depends on the one after it.

That means we can solve the entire game by working backward: start at layer 15 (where the answer is trivial: just add the bonus if you earned it), then solve layer 14 using those answers, then layer 13, and so on. The first two dominoes make the whole cascade click:

:::html
<div class="backward-cascade">

  <div class="cascade-card">
    <div class="cascade-card-header">
      <span class="cascade-layer-badge">Layer 15</span>
      <span class="cascade-card-subtitle">Game over</span>
    </div>
    <div class="cascade-scorecard">
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
    </div>
    <p class="cascade-desc">All 15 categories scored. Nothing to decide.</p>
    <div class="cascade-bonus-fork">
      <div class="cascade-bonus-box">
        <span class="cascade-bonus-label">upper &lt; 63</span>
        <span class="cascade-bonus-value">bonus: 0</span>
      </div>
      <div class="cascade-bonus-box cascade-bonus-earned">
        <span class="cascade-bonus-label">upper &ge; 63</span>
        <span class="cascade-bonus-value">bonus: +50 &#9733;</span>
      </div>
    </div>
    <p class="cascade-footnote">1 state-check per position. No dice, no decisions.</p>
  </div>

  <div class="cascade-arrow">
    <div class="cascade-arrow-line"></div>
    <span class="cascade-arrow-label">already solved</span>
    <div class="cascade-arrow-line"></div>
  </div>

  <div class="cascade-card">
    <div class="cascade-card-header">
      <span class="cascade-layer-badge">Layer 14</span>
      <span class="cascade-card-subtitle">One category left</span>
    </div>
    <div class="cascade-scorecard">
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq empty"></div>
    </div>
    <p class="cascade-desc">One category remains. You must score it. Roll, keep, reroll, keep, reroll, score. Then look up the final answer from Layer 15.</p>
    <p class="cascade-footnote">15 widgets &middot; 1 scoring choice each</p>
  </div>

  <div class="cascade-arrow">
    <div class="cascade-arrow-line"></div>
    <span class="cascade-arrow-label">already solved</span>
    <div class="cascade-arrow-line"></div>
  </div>

  <div class="cascade-card">
    <div class="cascade-card-header">
      <span class="cascade-layer-badge">Layer 13</span>
      <span class="cascade-card-subtitle">Two categories left</span>
    </div>
    <div class="cascade-scorecard">
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq filled"></div>
      <div class="cascade-sq empty"></div>
      <div class="cascade-sq empty"></div>
    </div>
    <p class="cascade-desc">Two categories remain. Try scoring each one. Each choice leads to a Layer 14 state, already solved. Pick whichever gives the higher expected score.</p>
    <p class="cascade-footnote">105 widgets &middot; 2 scoring choices each</p>
  </div>

  <div class="cascade-ellipsis">
    <div class="cascade-arrow-line"></div>
    <span class="cascade-ellipsis-dots">&middot; &middot; &middot;</span>
    <div class="cascade-arrow-line"></div>
  </div>

  <div class="cascade-card cascade-card-final">
    <div class="cascade-card-header">
      <span class="cascade-layer-badge">Layer 0</span>
      <span class="cascade-card-subtitle">Game start</span>
    </div>
    <div class="cascade-scorecard">
      <div class="cascade-sq empty"></div>
      <div class="cascade-sq empty"></div>
      <div class="cascade-sq empty"></div>
      <div class="cascade-sq empty"></div>
      <div class="cascade-sq empty"></div>
      <div class="cascade-sq empty"></div>
      <div class="cascade-sq empty"></div>
      <div class="cascade-sq empty"></div>
      <div class="cascade-sq empty"></div>
      <div class="cascade-sq empty"></div>
      <div class="cascade-sq empty"></div>
      <div class="cascade-sq empty"></div>
      <div class="cascade-sq empty"></div>
      <div class="cascade-sq empty"></div>
      <div class="cascade-sq empty"></div>
    </div>
    <p class="cascade-desc">All 15 open. 1 widget. <strong>E[start] = 248.4</strong></p>
    <p class="cascade-footnote">The answer to the entire game.</p>
  </div>

</div>
:::

:::insight

**From 10<sup>170</sup> to 1.43 million.** Ignoring dice order cuts 7,776 rolls to 252. Forgetting history collapses all possible games into 2.1 million positions. Pruning unreachable states removes another 32%. Breaking each turn into a six-layer widget makes every position solvable independently. Working backward, layer by layer, means we never revisit a decision.

:::

:::math

The state space forms a layered DAG. Each layer <var>k</var> contains all states
with exactly <var>k</var> categories scored. Transitions go strictly from
layer <var>k</var> to layer <var>k</var>+1, enabling backward induction:

:::equation
|states| = &sum;<sub><var>k</var>=0</sub><sup>15</sup>
C(15,<var>k</var>) &times; 64 = 2,097,152
&nbsp;&nbsp;&rarr;&nbsp;&nbsp;
reachable &asymp; 1,430,000 &nbsp;(31.8% pruned)
:::

The 252 distinct dice outcomes correspond to multisets of size 5 drawn from
{1,&hellip;,6}, counted by C(10,5) = 252. Each keep choice is one of 462
unique multisets. The widget structure gives:

:::equation
nodes/widget = 1 + 252 + 462 + 252 + 462 + 252 = 1,681
:::

Reachability is computed via forward DP over the 6-bit upper-category subsets.
For each subset <var>M</var> &sube; {1,&hellip;,6} and score <var>n</var>:

:::equation
R[n, M] = &exist; face <var>d</var> &notin; M, count <var>c</var> &in; {1..5}:
R[n &minus; d&middot;c, M \setminus {d}]
:::

:::

:::code

In the solver, states are indexed as `scored * 128 + upper` using a
stride of 128 instead of 64. The extra padding (indices 64-127) duplicates
the capped upper value, allowing branchless access when an upper-category score
would push the total beyond 63. This topological padding eliminates a branch
in the innermost loop at the cost of 2&times; memory (16 MB per strategy table).

```rust
// constants.rs — notation: upper_score = m, scored_categories = C
pub const STATE_STRIDE: usize = 128;
pub const NUM_STATES: usize = STATE_STRIDE * (1 << 15); // 128 × 32,768 = 4,194,304

#[inline(always)]
pub fn state_index(upper_score: usize, scored_categories: usize) -> usize {
    scored_categories * STATE_STRIDE + upper_score
}
```

:::

:::
