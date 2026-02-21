# The Mathematics of Optimal Yatzy

## Part I: The Game as a Mathematical Object

### Rules and Structure

Scandinavian Yatzy is played over exactly 15 turns. The scorecard contains 15 categories: the Upper Section (Ones through Sixes, scoring the sum of matching faces) and the Lower Section (One Pair, Two Pairs, Three of a Kind, Four of a Kind, Small Straight, Large Straight, Full House, Chance, and Yatzy).

On each turn, a player rolls five standard dice, may keep any subset and reroll the rest up to two times, then assigns the final dice to exactly one empty category. If the dice do not fit the category's requirements, the player scores zero.

**The Upper Bonus.** If the sum of the Upper Section reaches 63 or more, a flat 50-point bonus is awarded. This single rule drives the majority of strategic depth.

**Disambiguation from American Yahtzee.** Much of the published literature (Verhoeff 1999, Glenn 2006, Pape 2025) solves American Yahtzee, which features 13 categories, a 35-point bonus, variable straight scoring, and a complex 100-point "Yahtzee Bonus" mechanic. The EV-optimal score for American Yahtzee is 254.59. This document strictly analyzes Scandinavian Yatzy, where the EV-optimal score is **248.4**. Readers must not conflate the baseline numbers, though the structural conclusions map between variants.

### Scandinavian Yatzy Categories

| # | Category | Scoring rule | Ceiling |
|---|----------|-------------|---------|
| 1 | Ones | Sum of 1s | 5 |
| 2 | Twos | Sum of 2s | 10 |
| 3 | Threes | Sum of 3s | 15 |
| 4 | Fours | Sum of 4s | 20 |
| 5 | Fives | Sum of 5s | 25 |
| 6 | Sixes | Sum of 6s | 30 |
| 7 | One Pair | Sum of highest pair | 12 |
| 8 | Two Pairs | Sum of two different pairs | 22 |
| 9 | Three of a Kind | Sum of 3 matching dice | 18 |
| 10 | Four of a Kind | Sum of 4 matching dice | 24 |
| 11 | Small Straight | [1,2,3,4,5] = 15 points | 15 |
| 12 | Large Straight | [2,3,4,5,6] = 20 points | 20 |
| 13 | Full House | 3 of one + 2 of another = sum of all 5 | 28 |
| 14 | Chance | Sum of all 5 dice | 30 |
| 15 | Yatzy | 5 of a kind = 50 points | 50 |

Upper bonus: 50 points if the sum of categories 1-6 reaches or exceeds 63.

### Why Yatzy Is a Good Object of Study

Yatzy occupies a "Goldilocks zone" of complexity:

1. **Finite Horizon:** The game always ends in exactly 15 turns.
2. **Perfect Information:** No hidden state (unlike Poker).
3. **Stochastic Transitions:** The player controls decisions (which dice to keep, which category to score), but probability controls transitions (reroll outcomes).
4. **Combinatorial State Space:** Every decision permanently alters the scorecard, changing the geometry of all future decisions.

It is perfectly tractable: the state space is finite and small enough to solve exactly. Yet it is phenomenally hostile to approximation techniques — massive variance, discontinuous rewards, and deep sequential dependencies make it practically immune to Deep Reinforcement Learning.

### Formalizing the State Space

In optimal control theory, a "state" must contain all historical information necessary to make the best possible future decision (the **Markov Property**). In Yatzy, the exact sequence of dice rolled on Turn 1 does not matter on Turn 10. The only thing that dictates future options is the current scorecard.

A state is encoded as:

$$S = (C, m)$$

where:
- $C \subseteq \{0, \ldots, 14\}$: set of scored categories (15-bit bitmask)
- $m \in [0, 63]$: upper-section total, capped at the bonus threshold

Total state slots: $64 \times 2^{15} = 2{,}097{,}152$. After reachability pruning (31.8% of states are mathematically impossible), the game collapses to **~1,430,000 reachable states**.

---

## Part II: Solving the Game — Backward Induction

### The Bellman Equation for Yatzy

To find the mathematically perfect move on Turn 1, we start at the end of the game and work backward — **Backward Induction**.

At a **Chance Node** (rolling dice), the value is the probability-weighted sum of the outcomes. At a **Decision Node** (keeping dice or scoring a category), the value is the maximum over all available actions.

### Notation

| Symbol | Meaning |
|--------|---------|
| $\mathcal{C}$ | Set of all 15 categories |
| $S = (C, m)$ | Game state: $C$ = scored categories, $m$ = upper section total (capped) |
| $R_{5,6}$ | Set of all 252 distinct multisets of 5 dice from {1..6} |
| $R_k$ | Set of all "keeps": multisets of 0..5 dice from {1..6} (462 elements) |
| $P(r' \to r)$ | Probability of ending with roll $r$ after keeping $r'$ and rerolling the rest |
| $s(S, r, c)$ | Score from placing roll $r$ in category $c$ given state $S$ |
| $u(S, r, c)$ | Upper-section points gained: $= s(S,r,c)$ for upper categories, $0$ otherwise |
| $n(S, r, c)$ | Successor state: $(C \cup \{c\},\; \min(m + u(S,r,c),\; 63))$ |

### Recurrence Relations

**Terminal states** ($|C| = 15$, all categories scored):

$$E(C, m) = \begin{cases} 50 & \text{if } m \geq 63 \\ 0 & \text{otherwise} \end{cases}$$

**End of turn** — choose best category ($n = 0$ rerolls):

$$E(S, r, 0) = \max_{c \notin C} \Big[ s(S, r, c) + E\big(n(S, r, c)\big) \Big]$$

**After keeping dice** — expected value over reroll outcomes ($n = 1, 2$):

$$E(S, r', n) = \sum_{r \in R_{5,6}} P(r' \to r) \cdot E(S, r, n-1)$$

**After seeing a roll** — choose best keep ($n = 1, 2$):

$$E(S, r, n) = \max_{r' \subseteq r} E(S, r', n)$$

**Turn start** — expected value over initial roll:

$$E(S) = \sum_{r \in R_{5,6}} P(\perp \to r) \cdot E(S, r, 2)$$

### The Computation Pipeline

**Phase 0: Precompute Lookup Tables.**
1. Enumerate all 252 distinct 5-dice multisets and all 462 keep-multisets
2. Compute transition probabilities $P(K \to T)$ via the multinomial formula, stored in CSR (Compressed Sparse Row) format
3. For each (dice set, mask), compute which keep-multiset the mask produces, then deduplicate — multiple reroll masks can produce the same kept-dice multiset (462 unique keeps vs 31 raw masks, ~47% fewer dot products)
4. Precompute all 252 × 15 (roll, category) scores, dice set probabilities, and popcount caches

**Phase 1: Reachability Pruning.**
A DP over the 6 upper categories marks which (upper_mask, upper_score) pairs are achievable:

$$R(n, M) = \exists\, x \in M,\; \exists\, k \in [0,5]: k \cdot x \leq n \;\wedge\; R(n - k \cdot x,\; M \setminus \{x\})$$

Result: 31.8% of states pruned.

**Phase 2: Backward Induction.**
Process states in decreasing order of $|C|$ (number of scored categories), from $|C| = 15$ down to $|C| = 0$. This ensures that when we compute $E(S)$, all successor states have already been computed.

### The Turn "Widget" (SOLVE_WIDGET)

Each widget corresponds to a single turn starting in state $S$. It contains 1,681 internal states organized in 6 groups, computed bottom-up:

```
Group 6 → e[0]: For each final roll, choose the best category
    For each ds in 0..252:
        best ← max over unscored categories of (score + successor EV)
    e[0][ds] ← best

Groups 5 & 3 → Keep-EV dedup (the key optimization):
    Step 1: Compute EV for each of 462 unique keep-multisets
        keep_ev[kid] ← Σ vals[k] × e_prev[cols[k]]  (one sparse dot product)

    Step 2: For each dice set, find max over its unique keeps
        e_current[ds] ← max(e_prev[ds], max over unique keeps of keep_ev[kid])

    Called twice (ping-pong buffers):
        COMPUTE_MAX_EV(e[0], e[1])  → Group 5 (1 reroll left)
        COMPUTE_MAX_EV(e[1], e[0])  → Group 3 (2 rerolls left)

Group 1 → E(S): Weighted sum over initial rolls
    E(S) ← Σ P(⊥→r) × e[0][r]
```

### Memory Model

- **Persistent**: `E_table[S]` for all ~1.43M reachable states → ~8 MB (2,097,152 × f32)
- **Per-widget**: Two ping-pong buffers `e[0]`, `e[1]` each 252 × f32 → ~2 KB (freed after each widget)
- **Strategy**: ~1 byte per decision point (write-only during computation)

### The EV-Optimal Baseline

The exact Expected Value of an optimal game of Scandinavian Yatzy is **248.4 points** ($\sigma = 38.5$).

The hallmark of the EV-optimal policy is **Option Value**. Early in the game, the algorithm rarely forces difficult combinations. It takes zeros in hard categories (like Yatzy or Four of a Kind) if the dice are cold, preserving "easy" categories (like Chance or One Pair) as safety nets. An empty category is not just a scoring receptacle — it is a put-option that absorbs the game's inherent variance.

### Combinatorial Complexity

| Dimension | Count |
|-----------|-------|
| Total state slots | $2^{15} \times 64 = 2{,}097{,}152$ |
| Reachable states | $\approx 1{,}430{,}000$ |
| States per widget | 1,681 |
| Distinct 5-dice rolls | 252 |
| Distinct partial keeps | 462 |
| Max keep choices per roll | 32 (pattern $abcde$); as few as 6 ($aaaaa$) |
| Possible complete games | $\approx 1.7 \times 10^{170}$ |
| Deterministic strategies | $\sim 10^{10^{100}}$ |

---

## Part III: An Instructive Failure — The Max-Policy

To understand why the Bellman equation works, consider what happens when it is intentionally broken.

The **Max-Policy** replaces the expected value at chance nodes with the maximum over outcomes — assuming perfect dice luck. Decision nodes remain unchanged.

Under this assumption, the solver outputs a precomputed value of **374 points** — a perfect game. But when forced to play with real, randomized dice, the max-policy scores a mean of **118.7** — worse than random play.

**Why it fails.** The max-policy spreads overconfidence uniformly. Because it believes luck is guaranteed, it places zero value on safe fallback options. It reaches Turn 15 having thrown away numerous guaranteed scores in pursuit of jackpots that never materialized.

**The lesson.** In sequential decision-making, uncertainty is not noise that blurs the optimal path — **uncertainty is the signal that creates strategic depth.** The need to hedge, build safety nets, and preserve options exists *because* of variance. The EV-optimal algorithm is smart specifically because it quantifies its own likelihood of failure.

---

## Part IV: Risk-Sensitive Play — The θ Parameter

### Beyond Expected Value: CARA Utility

Expected Value assumes you are playing an infinite number of games and only care about the long-term average. In tournaments, this is incorrect — if trailing by 40 points, scoring 250 is as useless as scoring 0.

To model risk-sensitive play, we optimize for Utility rather than raw points, using exponential utility:

$$U(x) = e^{\theta x}$$

- **θ = 0**: Reduces exactly to the standard EV-optimal solver.
- **θ > 0**: Risk-seeking — overweights high scores.
- **θ < 0**: Risk-averse — overweights low scores.

Why exponential utility specifically? Because it possesses **Constant Absolute Risk Aversion (CARA)**:

$$e^{\theta(X_{\text{past}} + X_{\text{future}})} = e^{\theta X_{\text{past}}} \times e^{\theta X_{\text{future}}}$$

The past banked score factors completely out of the decision. The solver does not need to track accumulated score to know how to value the future. This is a computational miracle: the entire risk frontier uses the same compressed state space.

### Log-Domain Computation

To avoid numerical overflow (θ × 250 ≈ 25 for θ = 0.1), the solver works in the **log domain**:

$$L(S) = \ln\mathbb{E}[e^{\theta \cdot \text{total}} | S]$$

**Chance nodes** become log-sum-exp:

$$L(\text{keep}) = \text{LSE}_r \{ \ln P(\text{keep} \to r) + L(r) \}$$

where LSE uses the standard numerical stability trick:

$$\text{LSE}(x_1, \ldots, x_n) = m + \ln\sum e^{x_i - m}, \quad m = \max(x_i)$$

**Decision nodes** maximize $L$ for θ > 0, minimize for θ < 0 (dividing by negative θ flips the ordering of the certainty equivalent $CE = L/\theta$).

**θ = 0 equivalence**: When θ = 0, the exponential utility is constant. The implementation detects this and falls back to the exact EV code path.

### The LSE Transition: EV to Max

Taylor-expanding LSE around the weighted mean μ:

$$\text{LSE} \approx \mu + \frac{\sigma^2_{\log}}{2} + O(\sigma^3)$$

So LSE is approximately EV + variance/2 when spread is small. In the log domain, with values $L_i \approx \theta \cdot EV_i$:

$$\sigma^2_{\log} \approx \theta^2 \cdot \sigma^2_{EV}$$

This gives the transition criterion:

$$\text{LSE} \approx \text{EV} \quad \text{when } |\theta| \cdot \sigma_{EV} \ll 1$$
$$\text{LSE} \to \max \quad \text{when } |\theta| \cdot \sigma_{EV} \gg 1$$

### The Dimensionless Control Parameter

The product $|\theta| \cdot \sigma_{EV}$ determines the regime. It does not matter whether θ is large and σ is small, or vice versa — only their product matters.

This is mathematically identical to the **free energy** in statistical mechanics at inverse temperature β = θ. The transition from EV to max is the zero-temperature limit: when "temperature" $1/|\theta|$ drops below the "energy gap" between states, the system freezes into the ground state.

| Physics concept | Yatzy concept |
|----------------|---------------|
| Energy gaps between states | Score differences between dice outcomes |
| Inverse temperature β | θ |
| Free energy | LSE value |
| Characteristic energy scale | σ_EV |

### Estimating θ_critical for Yatzy

Three independent methods converge on θ_critical ~ 0.07–0.17:

**Method 1 (variance decomposition):** σ_total ≈ 38.5 across ~45 chance nodes gives σ_node ≈ 5.7, so θ_critical ≈ 1/5.7 ≈ 0.17.

**Method 2 (per-turn score spread):** Bad rolls cost ~5–15 points in game value. Estimated σ_EV at a typical reroll node: ~8–15 points, giving θ_critical ≈ 0.07–0.12.

**Method 3 (certainty equivalent expansion):** The risk correction is small when |θ| ≪ 2 × 248 / 1482 ≈ 0.33.

### Regime Map

| |θ| | |θ| × σ | Regime | Behavior |
|----:|-------:|--------|----------|
| 0.00 | 0.0 | EV-optimal | Exact expected value maximization |
| 0.01 | 0.1 | Near-EV | Indistinguishable from θ = 0 |
| 0.05 | 0.5 | **Interesting** | Meaningful risk/reward tradeoffs |
| 0.10 | 1.0 | **Interesting** | Strong risk sensitivity |
| 0.20 | 2.0 | Mostly degenerate | LSE ≈ max at most nodes |
| 0.50 | 5.0 | Fully degenerate | Policy frozen |

The useful range is **|θ| ∈ [0.03, 0.20]** — the active tradeoff regime where risk preferences produce meaningfully different strategies.

### Three Solver Modes

The solver exploits the regime structure:

| Mode | Theta range | Stochastic nodes | Runtime |
|------|-------------|-----------------|---------|
| EV | θ = 0 | Weighted sum | ~1.1s |
| Utility domain | |θ| ≤ 0.15 | Weighted sum (same as EV) | ~0.5s |
| Log domain (LSE) | |θ| > 0.15 | Log-sum-exp | ~2.7s |

For small θ, the utility domain $U(S) = \mathbb{E}[e^{\theta \cdot \text{remaining}} | S]$ avoids transcendentals entirely — stochastic nodes are plain weighted sums, identical to the EV solver. The numerical safety cutoff at |θ| ≤ 0.15 prevents f32 overflow ($e^{0.15 \times 374}$) and mantissa erasure ($e^{0.15 \times 100} > 2^{24}$).

---

## Part V: CARA Sufficiency and Its Limits

### Why CARA Is Special (Proof)

Can any objective $J(\pi) = \mathbb{E}[g(X^\pi)]$ be optimized using only the reduced state $(C, m)$ without tracking the running "sunk score" $x_{\text{sunk}}$?

For Bellman optimality to drop $x_{\text{sunk}}$, the preference over lotteries must be globally invariant to deterministic wealth shifts. By the VNM Utility Theorem, $g(x+y)$ must be a positive affine transformation of $g(y)$. Solving the Pexider functional equation yields exactly the linear (EV, θ = 0) and exponential (CARA, θ ≠ 0) solutions.

**Implication:** CARA is the *unique* mathematical framework capable of manipulating risk without exponentially exploding the state space. Any non-CARA objective (threshold step-functions, exact P(score ≥ t)) strictly requires the expanded state $(C, m, x_{\text{sunk}})$.

### CARA Incompleteness

Does there exist a threshold $t$ where the threshold-optimal policy strictly beats the best CARA policy?

**Yes** (constructive proof). At Turn 15 with only Chance remaining and dice [2,2,2,3,3]: if $x_{\text{sunk}} = 150$ (need 12 to reach t = 162), keeping all dice guarantees the win. If $x_{\text{sunk}} = 145$ (need 17), keeping guarantees a loss — one must reroll the 3s. Because CARA evaluates actions independently of $x_{\text{sunk}}$, it must output the same action for both histories, forcing strict suboptimality.

### Approximate Completeness

Despite this theoretical incompleteness, the gap is vanishingly small. The CARA suboptimality is bounded by:

$$\epsilon \leq \mathcal{O}\left( T \cdot \frac{B^2}{\sigma^2} \cdot \mathbb{P}\left(G < \frac{B^2}{\sigma}\right) \right)$$

Because the EV gap $G$ between best and second-best actions is typically massive, the "flippable set" where sunk-score knowledge changes the decision is vanishingly small. The structural error is choked to < 1%.

### The Structural Paradox

A tension exists between two facts about the policy space:

1. **Constructive inseparability** (Proof 4.5 in the extended theory): The game is globally coupled across all 15 turns. Consuming a category on Turn 1 removes it from Turn 15's action space, and the value of that category depends on Turn 15's risk posture. The minimal valid factorization is irreducible (p = 1).

2. **Descriptive separability** (Proof 4.6): Despite this global coupling, a 3-dimensional temporal risk summary $\bar{\boldsymbol{\theta}} = (\bar{\theta}_{\text{early}}, \bar{\theta}_{\text{mid}}, \bar{\theta}_{\text{late}})$ predicts the terminal score distribution with $R^2 > 0.95$.

**Resolution:** The Central Limit Theorem's smoothing across 15 bounded turns collapses the high-dimensional policy space back to ~1 effective dimension. You can *read* the policy locally, but you cannot *write* it locally without breaking Bellman optimality.

---

## Part VI: Why Reinforcement Learning Struggles

Modern RL conquered Backgammon, Chess, Go, and Poker. Yet the best neural networks stall at ~5% below the exact DP baseline in Yatzy. Three interacting barriers explain this:

1. **The Discontinuous Value Cliff.** At 62 upper points, the bonus is worth nothing. At 63, it is worth 50 points. Neural networks are continuous function approximators that "blur" this cliff. With $2^{15} = 32{,}768$ possible category-fill patterns each requiring a distinct value surface, a typical 66K-parameter network allocates only ~8 parameters per combinatorial regime — far too few.

2. **Devastating Signal-to-Noise Ratio.** A brilliant sacrifice on Turn 3 gains +0.5 EV, but game-level noise has σ ≈ 38.5. The true gradient of that decision is buried under stochastic noise. Detecting a 1-point improvement requires ~1,480 episodes at 1σ confidence.

3. **No Adversarial Curriculum.** RL thrives in zero-sum games via self-play (TD-Gammon, AlphaGo). Yatzy is a solitaire optimization problem. If the agent learns a "safe but mediocre" policy, there is no adversary to punish its blind spots.

These factors interact **multiplicatively**. Yatzy combines rough value functions *plus* sparse delayed rewards *plus* high stochastic branching *plus* no adversarial curriculum — a combination that defeats every standard RL approach.

---

## Part VII: Compact Representations

The exact solver requires an 8 MB lookup table. Can we compress it?

Decision Trees dominate MLPs at all parameter scales. A depth-20 DT (~130K parameters, ~500 KB) achieves 93%+ accuracy and loses only 1.4 points/game EV — a 16× compression. MLPs of matched size suffer roughly double the EV loss.

**Why DTs win:** The optimal policy is piecewise-constant with hard, combinatorial boundaries (exactly crossing the 63-point threshold). Decision trees handle axis-aligned step functions perfectly; neural networks waste capacity learning soft approximations of mathematical cliffs.

---

## Part VIII: Implications for Human Players

The mathematical optimum operates ~20–30 points higher than the average casual player. The gap decomposes into a strict skill ladder:

1. **Upper Bonus Awareness (largest gain):** Humans chronically undervalue the 50-point injection. The solver aggressively takes zeros in Four-of-a-Kind to force the 63-point threshold.

2. **Flexibility Preservation (medium gain):** Humans use Chance or Ones when they get a bad roll early. The solver guards these categories, saving them to absorb catastrophic variance in Turns 13 and 14.

3. **Combinatorial Micro-Hedging (final 5%):** The realm of the DP solver. Knowing exactly when to hold a pair versus chase a straight based on the exact fractional EV of the remaining scorecard.

---

## Appendix: State Indexing

```
state_index(m, C) = C × STATE_STRIDE + m
```

where STATE_STRIDE = 128 (topological padding). Indices 64–127 contain duplicates of the capped value at index 63, enabling branchless upper-category access: `sv[succ_base + up + scr]` where `up + scr` can safely exceed 63.

This places all 128 upper-score variants of the same bitmask $C$ in a contiguous block, perfectly aligning with L1/L2 cache lines during successor lookups.
