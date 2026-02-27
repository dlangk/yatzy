# Bellman Equation

The **Bellman equation** expresses the value of a state as the best immediate reward plus the value of the best reachable future state. It is the recursive backbone of dynamic programming.

## The Equation

For a finite-horizon MDP with no discounting:

> V(s) = max_a [ R(s,a) + sum over s' of P(s'|s,a) * V(s') ]

In words: the value of state s equals the maximum, over all actions a, of the immediate reward R(s,a) plus the expected value of the next state. The expectation accounts for randomness -- in Yatzy, this is the dice roll.

The equation has a natural base case: terminal states (all 15 categories scored) have value equal to the final score. There are no future actions, so V(terminal) = accumulated_score.

## Solving It Exactly

Because the game has a finite horizon, the recursion terminates. **Backward induction** fills in V(s) starting from terminal states, then states with 14 categories scored, then 13, and so on back to the empty board.

Each state is visited exactly once. There are no iterative convergence loops, no approximation errors, and no hyperparameters to tune. The solution is provably optimal.

This is fundamentally different from RL approaches that *estimate* V(s) through sampling. The Bellman equation gives us the exact value -- we just need enough compute to evaluate all states.

## Nested Structure in Yatzy

The Bellman equation appears at multiple levels within a single turn:

1. **Category selection**: V(s) = max over categories c of [score(c) + V(next_state)]
2. **Keep decision (1 reroll left)**: V(dice, 1) = max over keeps k of E_roll[V(resulting_dice, 0)]
3. **Keep decision (2 rerolls left)**: V(dice, 2) = max over keeps k of E_roll[V(resulting_dice, 1)]

These nest together: the category selection at the bottom feeds into the keep decisions above it. The solver evaluates all three levels for each state, producing a complete strategy table.

The nesting means a single state evaluation involves thousands of Bellman updates: 252 dice outcomes times 462 possible keeps times 2 reroll levels times the number of open categories. This is the computational core of the widget solver.

## Why It Works

The key insight is **optimal substructure**: the optimal strategy from any state onward doesn't depend on how you arrived there. This lets us decompose a 15-round game into 15 independent layers, each solved in terms of the next.

Without optimal substructure, you would need to consider all possible game histories -- an exponentially large set. The Bellman equation reduces this to a polynomial computation: linear in the number of states, with a per-state cost determined by the action and outcome spaces.

## Risk-Sensitive Extension

For the risk-sensitive solver (theta != 0), the Bellman equation generalizes to:

> V(s) = (1/theta) * log( sum_a exp(theta * [R(s,a) + sum P(s'|s,a) * V(s')]) )

The max is replaced by a **soft-max** (log-sum-exp). When theta -> 0, this recovers the standard max. When theta < 0, it biases toward actions with lower variance. The mathematical structure is identical; only the aggregation operator changes.

## Historical Note

Richard Bellman introduced this equation in the 1950s as part of his work on dynamic programming at the RAND Corporation. He coined the term "dynamic programming" partly to obscure the mathematical nature of his research from government funders who were skeptical of pure mathematics. The equation bears his name and remains the foundation of optimal control theory, reinforcement learning, and operations research.

In Yatzy, we use the Bellman equation in its simplest and most powerful form: exact solution via backward induction over a finite state space. This is the setting Bellman originally studied, before the equation was extended to continuous states, infinite horizons, and approximate solution methods that dominate modern RL.

## Computational Complexity

The total cost of solving the Bellman equation for all Yatzy states is:

> O(|S| * |A| * |S'|)

where |S| is the number of states (~1.43M reachable), |A| is the number of actions per state (up to 462 keeps x 15 categories), and |S'| is the number of stochastic outcomes per action (up to 252 dice outcomes). This product is large but finite and fixed -- roughly 64 billion floating-point operations for the full backward pass.

The key insight is that this cubic-looking cost is actually traversed once, not iteratively. Unlike value iteration (which may need dozens of passes to converge), backward induction requires exactly one pass through the state space. This single-pass property is what makes the ~1-second solve time possible.
