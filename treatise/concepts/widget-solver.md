# Widget Solver

The **widget solver** is the computation kernel that evaluates a single Yatzy state: given the set of already-scored categories and upper section progress, what is the optimal expected value?

## The Six Groups

Within one state, the solver must process a chain of interleaved chance and decision nodes:

1. **Group 1** (chance): enumerate all 252 distinct dice outcomes for the initial roll
2. **Group 2** (decision): for each outcome, evaluate all 462 unique keep actions with 2 rerolls remaining
3. **Group 3** (chance): for each keep, compute the expected value over all possible second rolls
4. **Group 4** (decision): for each second-roll outcome, evaluate all 462 keeps with 1 reroll remaining
5. **Group 5** (chance): for each keep, compute the expected value over the final roll
6. **Group 6** (decision): for each final dice configuration, choose the best category to score

The groups alternate between "nature rolls dice" (chance) and "player decides" (decision). Information flows backward: Group 6 feeds into Group 5, which feeds into Group 4, and so on up to Group 1.

## Scale of the Computation

Each widget evaluation touches approximately 45,000 intermediate values. Multiply by the ~1.43 million reachable states, and the full backward pass involves roughly 64 billion floating-point operations. Despite this scale, careful SIMD vectorization and memory layout allow the entire computation to complete in about 1 second on Apple Silicon.

The per-state cost breaks down roughly as:

- **Groups 5-6** (final roll): 252 outcomes x open categories ~ small
- **Groups 3-4** (second roll): 252 x 462 keeps ~ dominant cost
- **Groups 1-2** (first roll): 252 x 462 keeps ~ dominant cost

The keep-evaluation phases (Groups 2 and 4) account for most of the computation because they iterate over all 462 unique keep multisets for each dice outcome.

## Ping-Pong Buffers

The solver alternates between two pre-allocated buffers. One holds the values being computed (current layer); the other holds the already-solved future values (next layer). After each scored-mask group is complete, the buffers swap roles. This avoids allocation overhead and keeps memory usage constant at ~32 MB regardless of the number of states processed.

The buffer swap is a pointer exchange, not a data copy. This zero-cost swap pattern is critical when processing millions of states.

## Why "Widget"?

The name reflects that this is a self-contained unit of work. Each state's computation is independent of other states in the same layer (same number of scored categories), making it trivially parallelizable across CPU cores with Rayon. The solver distributes widget evaluations across all available cores, achieving near-linear speedup.

## Four Solver Variants

The widget has four variants optimized for different theta regimes:

- **EV solver** (theta = 0): standard max aggregation, fastest path
- **Utility solver** (|theta| <= 0.15): works in exp-utility space directly
- **LSE solver** (|theta| > 0.15): log-domain with log-sum-exp trick for numerical stability
- **Max solver**: computes the best possible score (upper bound), used for analysis

Each variant shares the same six-group structure but differs in how values are aggregated at decision nodes. The EV solver uses max; the LSE solver uses log-sum-exp; the utility solver uses weighted exponential sums. The compiler generates specialized code for each variant, avoiding branch overhead in the hot path.

## Memory Layout

The widget solver's performance depends critically on memory access patterns. The state values are stored in a flat array with STATE_STRIDE=128, so that adjacent upper-progress values for the same scored-mask are contiguous in memory. This enables efficient SIMD loads: processing 4 consecutive upper-progress values with a single NEON instruction.

The KeepTable and dice outcome probabilities are arranged for sequential access within the inner loops. Cache misses in these tables would dominate runtime; the current layout keeps the working set within L2 cache for the critical keep-evaluation phases.
