# Optimal Scandinavian Yatzy: Algorithm and DP

## 1. Problem Formulation

Scandinavian Yatzy is a 15-turn solitaire game. Each turn: roll 5 dice, optionally reroll any subset (up to 2 rerolls), then assign the final roll to one unscored category. The goal is to maximize total score, including a 50-point bonus if upper-section categories (Ones–Sixes) sum to ≥63.

### State space

A game state is a tuple $S = (m, C)$ where:
- $m \in [0, 63]$: capped upper-section score
- $C \subseteq \{0, \ldots, 14\}$: set of scored categories (15-bit bitmask)

Total states: $64 \times 2^{15} = 2{,}097{,}152$. After reachability pruning, ~1,430,000 are reachable.

### Combinatorial quantities

| Symbol | Count | Description |
|--------|-------|-------------|
| $R_{5,6}$ | $\binom{10}{5} = 252$ | Distinct 5-dice multisets from \{1..6\} |
| $R_k$ | 462 | Keep-multisets: 0–5 dice from \{1..6\} |
| Reroll masks per dice set | 31 (non-trivial) | 5-bit masks, but many are equivalent |
| Unique keeps per dice set | 16.3 avg | After multiset dedup |
| Categories | 15 | Ones through Yatzy |

## 2. Backward Induction

The algorithm computes $E(S)$ — the expected score under optimal play starting from state $S$ — for every reachable state via backward induction over the DAG of game states.

### State indexing

```
state_index(m, C) = C × 64 + m
```

This places all 64 upper-score variants of the same bitmask $C$ in a contiguous 256-byte block (64 × 4 bytes at f32 precision). This is critical for Group 6 cache performance (§5.3).

### Recurrence relations

**Terminal states** ($|C| = 15$, all categories scored):

$$E(C, m) = \begin{cases} 50 & \text{if } m \geq 63 \\ 0 & \text{otherwise} \end{cases}$$

**Category assignment** (0 rerolls remaining):

$$E(S, r, 0) = \max_{c \notin C} \big[ s(S, r, c) + E(n(S, r, c)) \big]$$

where $n(S, r, c) = (C \cup \{c\},\; \min(m + u(S,r,c),\; 63))$ is the successor state.

**After keeping dice** (expected value over reroll outcomes):

$$E(S, r', n) = \sum_{r \in R_{5,6}} P(r' \to r) \cdot E(S, r, n-1)$$

**After seeing a roll** (choose best keep):

$$E(S, r, n) = \max_{r' \subseteq r} E(S, r', n)$$

**Turn start** (expected value over initial roll):

$$E(S) = \sum_{r \in R_{5,6}} P(\perp \to r) \cdot E(S, r, 2)$$

### Memory model

- **Persistent**: `E_table[S]` for all ~1.43M reachable states → ~8 MB (2,097,152 × f32)
- **Per-widget**: Two ping-pong buffers `e[0]`, `e[1]` each 252 × f32 → ~2 KB (freed after each widget)

## 3. Computation Pipeline

### Phase 0: Precompute lookup tables

8 sub-steps in dependency order:

1. **Factorials** — $0!$ through $5!$ for multinomial coefficients
2. **Dice combinations** — enumerate all 252 sorted 5-dice multisets $R_{5,6}$, build 5D reverse lookup table for O(1) index access
3. **Category scores** — precompute $s(S, r, c)$ for all 252 × 15 = 3,780 (roll, category) pairs
4. **Keep-multiset table** — the most important table. Three sub-steps:
   - **3a**: Enumerate all 462 keep-multisets as frequency vectors $[f_1, \ldots, f_6]$
   - **3b**: Compute $P(K \to T)$ for each keep $K$ and target $T$ via the multinomial formula:
     $$P(K \to T) = \frac{n!}{\prod_{i=1}^{6} d_i!} \cdot \frac{1}{6^n}$$
     where $n = 5 - |K|$ (dice rerolled), $d_i = t_i - k_i$ (rerolled dice per face). Stored in CSR format (vals/cols arrays with row_start boundaries).
   - **3c**: For each (dice set, mask), compute which keep-multiset the mask produces, then deduplicate. Builds `unique_keep_ids`, `unique_count`, `mask_to_keep`, `keep_to_mask`.
5. **Dice set probabilities** — $P(\perp \to r) = \binom{5}{n_1, \ldots, n_6} / 6^5$
6. **Popcount cache** — bitmask → $|C|$
7. **Terminal states** — base case of DP
8. **Reachability pruning** — DP over 6 upper categories to mark achievable $(m, C)$ pairs

### Phase 1: Reachability pruning

A small DP determines which (upper_mask, upper_score) pairs are achievable:

$$R(n, M) = \exists\, \text{face}\; x \in M,\; \exists\, k \in [0,5]: k \cdot x \leq n \;\wedge\; R(n - k \cdot x,\; M \setminus \{x\})$$

Base case: $R(0, M) = \text{true}$. Upper score 63 means "≥63" (exact values 63–105 are OR'd together).

Result: **31.8% of states pruned** (e.g., upper_score=63 with only Ones scored is impossible).

### Phase 2: Backward induction

Process levels $|C| = 14$ down to $|C| = 0$:

```
for num_scored = 14 downto 0:
    group states by scored_categories bitmask
    par_iter over groups:
        for each upper_score in group:
            E_table[state_index(up, scored)] ← SOLVE_WIDGET(up, scored)
```

## 4. SOLVE_WIDGET: The Inner Loop

Each widget represents one turn starting in state $S$. It has 1,681 internal states in 6 groups, computed bottom-up using ping-pong buffers `e[0]`, `e[1]` (each 252 × f32):

### Group 6 → e[0]: Best category for each final roll

```
// Preload lower-category successor EVs (constant across all 252 dice sets)
for c in 6..15:
    if c not scored:
        lower_succ_ev[c] ← sv[state_index(up, scored | (1<<c))]

for each ds_i in 0..252:
    best_val ← -∞
    // Upper categories (0-5): sv lookup varies per roll (new_up depends on score)
    for c in 0..6:
        if c not scored:
            scr ← precomputed_scores[ds_i][c]
            new_up ← min(up + scr, 63)
            val ← scr + sv[state_index(new_up, scored | (1<<c))]
            best_val ← max(best_val, val)
    // Lower categories (6-14): use preloaded value (zero extra sv reads)
    for c in 6..15:
        if c not scored:
            scr ← precomputed_scores[ds_i][c]
            val ← scr + lower_succ_ev[c]
            best_val ← max(best_val, val)
    e[0][ds_i] ← best_val
```

### Groups 5 & 3 → keep-EV dedup (the key optimization)

```
COMPUTE_MAX_EV_FOR_N_REROLLS(e_prev, e_current):
    // Step 1: Compute EV for each of the 462 unique keep-multisets
    //         (one sparse dot product each)
    for kid in 0..462:
        keep_ev[kid] ← Σ vals[k] × e_prev[cols[k]]
                        for k in row_start[kid]..row_start[kid+1]

    // Step 2: For each dice set, find max over its unique keeps
    for ds_i in 0..252:
        best ← e_prev[ds_i]       // mask=0: keep all
        for j in 0..unique_count[ds_i]:
            kid ← unique_keep_ids[ds_i][j]
            best ← max(best, keep_ev[kid])
        e_current[ds_i] ← best
```

The widget calls this twice:
- `COMPUTE_MAX_EV_FOR_N_REROLLS(e[0], e[1])` → Group 5 (1 reroll left)
- `COMPUTE_MAX_EV_FOR_N_REROLLS(e[1], e[0])` → Group 3 (2 rerolls left)

### Group 1 → E(S): Weighted sum over initial rolls

```
E(S) ← Σ P(⊥→r) × e[0][r]   for r in 0..252
```

## 5. Risk-Sensitive Extension

For $\theta \neq 0$, the solver uses exponential utility $u(x) = e^{\theta x}$, computed in the log domain $L(S) = \ln\mathbb{E}[e^{\theta \cdot \text{total}}]$ to avoid numerical overflow.

The recurrence changes:
- **Decision nodes** (category assignment, keep selection): maximize $L$ for $\theta > 0$ (risk-seeking), minimize for $\theta < 0$ (risk-averse), because $CE = L/\theta$ and dividing by negative $\theta$ flips the ordering.
- **Stochastic nodes** (reroll outcomes): the dot product $\sum P \cdot x$ becomes a log-sum-exp: $\text{LSE}(x; w) = m + \ln\sum w_i \cdot e^{x_i - m}$ where $m = \max(x_i)$ for numerical stability.
- **Group 6**: $\text{val} = \theta \cdot \text{scr} + sv[\text{successor}]$ (log-domain contribution)
- **Group 1**: LSE over initial rolls (stochastic node, always the same regardless of $\theta$ sign)

Runtime: ~7s for $\theta \neq 0$ vs ~2.3s for $\theta = 0$ (the 2-pass LSE is ~3× slower than a single-pass dot product per keep).
