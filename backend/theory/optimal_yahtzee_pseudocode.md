# Pseudocode: Computing the Optimal Solitaire Yahtzee Strategy

The algorithm computes the optimal expected score for every possible game state using **backward induction** (retrograde analysis) over a directed acyclic graph of game states. The key insight is that the graph decomposes into $2^{19}$ independent "widgets" (one per turn-start state), and widgets can be processed and discarded one at a time, keeping only a table of $E(S)$ values across widgets.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $\mathcal{C}$ | Set of all 13 categories: {ones, twos, ..., sixes, 3ofk, 4ofk, full house, sm straight, lg straight, chance, yahtzee} |
| $S = (C, m, f)$ | Game state: $C \subseteq \mathcal{C}$ = used categories, $m \in [0, 63]$ = upper section total (capped), $f$ = Yahtzee bonus flag |
| $R_{5,6}$ | Set of all 252 distinct multisets of 5 dice from {1..6} |
| $R_k$ | Set of all "keeps": multisets of 0..5 dice from {1..6} (462 elements) |
| $r$ | A full roll $\in R_{5,6}$ |
| $r'$ | A partial keep $\in R_k$, where $r' \subseteq r$ |
| $\perp$ | The empty roll (keeping zero dice) |
| $P(r' \to r)$ | Probability of ending with roll $r$ after keeping $r'$ and rerolling the rest |
| $s(S, r, c)$ | Score from placing roll $r$ in category $c$ given state $S$ (includes Yahtzee Bonus, excludes Upper Bonus) |
| $u(S, r, c)$ | Upper-section points gained: $= s(S,r,c)$ for upper categories, $0$ otherwise |
| $f(S, r, c)$ | New Yahtzee Bonus flag after scoring $r$ in $c$ |
| $n(S, r, c)$ | Successor state: $(C \cup \{c\},\; m + u(S,r,c),\; f(S,r,c))$ |

---

## Phase 0: Precompute Combinatoric Tables

```
PRECOMPUTE_ROLLS_AND_PROBABILITIES():
    // Enumerate all 252 distinct 5-dice multisets -> R_5,6
    // Enumerate all 462 partial keeps (0..5 dice)  -> R_k
    //
    // For each keep r' in R_k and each full roll r in R_5,6:
    //   Compute P(r' -> r) = probability of rerolling (5 - |r'|) dice
    //   and ending with exactly r.
    //
    //   This is a multinomial coefficient over the rerolled dice
    //   divided by 6^(5 - |r'|).
    //
    // For each full roll r in R_5,6:
    //   Compute P(⊥ -> r) = probability of rolling r from scratch
    //   = multinomial(5; counts of each face in r) / 6^5
    //
    // For each full roll r in R_5,6:
    //   Enumerate all valid sub-multisets r' ⊆ r (the "keeps")
    //   Store this mapping for use in the max-over-keeps step.
```

---

## Phase 1: Prune Unreachable States (Section 4.3)

Not all $(C, m)$ combinations are reachable. Use dynamic programming to determine which upper totals $m$ are achievable with a given subset of upper categories.

```
COMPUTE_REACHABILITY():
    // S_upper ⊆ {1,2,3,4,5,6} indicates which upper categories are used
    // n ∈ [0, 63] is the upper section total
    //
    // R(n, S_upper) = true iff it is possible to score exactly n points
    //                 in the upper section using categories in S_upper

    for all S_upper ⊆ {1,2,3,4,5,6}:
        R(0, S_upper) ← true

    for all n ≥ 1:
        R(n, ∅) ← false

    for all S_upper ≠ ∅ and n ≥ 1:
        R(n, S_upper) ← false
        for each face x ∈ S_upper:            // x ∈ {1,2,3,4,5,6}
            for k = 0 to 5:                    // number of x's scored
                if k * x ≤ n AND R(n - k*x, S_upper - {x}):
                    R(n, S_upper) ← true
                    break both loops

    // Result: 1260 of the 4096 (S_upper, n) pairs are unreachable.
    // A state (C, m, f) is valid only if R(m, C ∩ {1..6}) is true.
    // This prunes >25% of the state space.

    VALID_STATES ← { (C, m, f) : R(m, C ∩ {1..6}) = true
                                  AND 0 ≤ m ≤ 63
                                  AND f ∈ {true, false} }
```

---

## Phase 2: Backward Induction — Main Algorithm (Section 5)

Process states in **decreasing order of $|C|$** (number of filled categories), from $|C| = 13$ down to $|C| = 0$. This ensures that when we compute $E(S)$, all successor states $n(S, r, c)$ have already been computed.

```
COMPUTE_OPTIMAL_STRATEGY():

    // ─── Allocate persistent storage ───
    // E_table[S] for all valid turn-start states S = (C, m, f)
    // Strategy can optionally be recorded (one byte per decision point)

    // ═══════════════════════════════════════════════════
    // STEP 1: BASE CASE — Terminal states (|C| = 13)
    // ═══════════════════════════════════════════════════

    for each state S = (C, m, f) where C = 𝒞:    // all categories filled
        if m ≥ 63:
            E_table[S] ← 35                       // upper bonus earned
        else:
            E_table[S] ← 0                        // no upper bonus

    // ═══════════════════════════════════════════════════
    // STEP 2: INDUCTION — Process widgets from |C|=12 down to |C|=0
    // ═══════════════════════════════════════════════════

    for num_filled = 12 downto 0:
        for each valid state S = (C, m, f) with |C| = num_filled:

            SOLVE_WIDGET(S)

    // ═══════════════════════════════════════════════════
    // RESULT
    // ═══════════════════════════════════════════════════

    // E_table[(∅, 0, false)] is the optimal expected score
    // ≈ 254.59 with Yahtzee bonuses, ≈ 245.87 without
```

---

## Phase 2a: Solve One Widget (The Inner Loop)

Each widget corresponds to a single turn starting in state $S$. It contains 1681 internal states organized in 6 groups. We compute expected values bottom-up within the widget: from the exit states (group 6) back to the entry state (group 1).

```
SOLVE_WIDGET(S):
    // S = (C, m, f) is the turn-start state
    // We compute E(S) and optionally record the optimal decisions.
    //
    // Widget structure (6 groups):
    //   Group 1: entry point (1 state)
    //   Group 2: initial roll outcomes (252 states)          [chance]
    //   Group 3: first keep choices (462 states)             [choice]
    //   Group 4: first reroll outcomes (252 states)          [chance]
    //   Group 5: second keep choices (462 states)            [choice]
    //   Group 6: final roll outcomes / exit (252 states)     [chance]

    // ─── GROUP 6 → EXIT: Score the final roll (n = 0 rerolls left) ───
    // This is the category-assignment decision.
    // Edges leave the widget to successor turn-start states.

    for each roll r ∈ R_5,6:
        best_value ← -∞
        best_category ← nil
        for each category c ∉ C:                    // unused categories only
            successor ← n(S, r, c)                  // next turn-start state
            value ← s(S, r, c) + E_table[successor] // immediate score + future
            if value > best_value:
                best_value ← value
                best_category ← c
        E_exit[r] ← best_value                      // = E(S, r, 0)
        // optionally record: strategy_category[S, r] ← best_category

    // ─── GROUP 5: Choose which dice to keep (second keep, n = 1) ───
    // Player sees roll r after first reroll, picks best sub-multiset r' ⊆ r.

    for each roll r ∈ R_5,6:

        // First compute E(S, r', n=1) for each possible keep r' ⊆ r:
        //   = expected value after keeping r' and rerolling, with 0 rerolls after
        //   = Σ P(r' → r'') · E_exit[r'']   over all final outcomes r''

        best_value ← -∞
        best_keep ← nil
        for each sub-multiset r' ⊆ r:              // possible keeps from this roll
            expected ← 0
            for each outcome r'' ∈ R_5,6:
                expected ← expected + P(r' → r'') · E_exit[r'']
            if expected > best_value:
                best_value ← expected
                best_keep ← r'
        E_roll1[r] ← best_value                    // = E(S, r, 1)
        // optionally record: strategy_keep1[S, r] ← best_keep

    // ─── GROUP 4 + 3: First reroll outcome + first keep choice (n = 2) ───
    // Same structure, but the "exit" of this stage feeds into E_roll1.

    for each roll r ∈ R_5,6:

        // E(S, r', n=2) for each keep r' ⊆ r:
        //   = Σ P(r' → r'') · E_roll1[r'']  over all first-reroll outcomes r''

        best_value ← -∞
        best_keep ← nil
        for each sub-multiset r' ⊆ r:
            expected ← 0
            for each outcome r'' ∈ R_5,6:
                expected ← expected + P(r' → r'') · E_roll1[r'']
            if expected > best_value:
                best_value ← expected
                best_keep ← r'
        E_roll2[r] ← best_value                    // = E(S, r, 2)
        // optionally record: strategy_keep2[S, r] ← best_keep

    // ─── GROUP 1: Entry point — expected value over initial roll ───

    E_table[S] ← 0
    for each roll r ∈ R_5,6:
        E_table[S] ← E_table[S] + P(⊥ → r) · E_roll2[r]

    // Widget is now fully solved.
    // E_roll1, E_roll2, E_exit are temporary and can be freed.
    // Only E_table[S] persists.
```

---

## Summary of Recurrence Relations

The pseudocode above implements these four equations from Section 5, evaluated bottom-up:

**Terminal states** ($C = \mathcal{C}$, all categories filled):

$$E(C, m, f) = \begin{cases} 35 & \text{if } m \geq 63 \\ 0 & \text{otherwise} \end{cases}$$

**End of turn** — choose best category ($n = 0$ rerolls):

$$E((C,m,f),\; r,\; 0) = \max_{c \notin C} \Big[ s\big((C,m,f),\, r,\, c\big) + E\big(n((C,m,f),\, r,\, c)\big) \Big]$$

**After keeping dice** — expected value over reroll outcomes ($n = 1, 2$):

$$E(S,\; r',\; n) = \sum_{r \in R_{5,6}} P(r' \to r) \cdot E(S,\; r,\; n-1)$$

**After seeing a roll** — choose best keep ($n = 1, 2$):

$$E(S,\; r,\; n) = \max_{r' \subseteq r} E(S,\; r',\; n)$$

**Turn start** — expected value over initial roll:

$$E(S) = \sum_{r \in R_{5,6}} P(\perp \to r) \cdot E(S,\; r,\; 2)$$

---

## Memory Optimization (Section 6)

The critical implementation insight is that **groups 2–5 within a widget depend only on each other**, and **group 6 depends only on E_table values from other widgets** (successor states with more categories filled, which have already been computed).

Therefore:

```
MEMORY_STRATEGY:
    // Persistent:  E_table[S] for all ~524,288 valid states     ≈ 4 MB
    //              (or ~786,432 states with Yahtzee Bonuses      ≈ 6 MB)
    //
    // Per-widget:  E_exit[252], E_roll1[252], E_roll2[252]       ≈ 6 KB
    //              (freed after each widget)
    //
    // Strategy:    ~1 byte per decision point per widget
    //              (only needs to be written, never read during computation)
    //
    // Total edges examined: ~11 billion across all widgets
    // Reported runtime: ~1 hour on entry-level CPU (circa 2006)
```

---

## Complexity

| Dimension | Count |
|-----------|-------|
| Turn-start states (widgets) | $\leq 2^{19} = 524{,}288$ (minus unreachable) |
| States per widget | 1,681 |
| Edges per widget | $\leq 21{,}000$ |
| Total edges in graph | $\approx 11 \times 10^9$ |
| Distinct 5-dice rolls | $\binom{10}{5} = 252$ |
| Distinct partial keeps | $\binom{10}{5} + \binom{9}{4} + \binom{8}{3} + \binom{7}{2} + \binom{6}{1} + \binom{5}{0} = 462$ |
| Max keep choices per roll | 32 (pattern $abcde$); as few as 6 (pattern $aaaaa$) |
