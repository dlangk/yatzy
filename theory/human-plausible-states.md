# Human-Plausible State Filtering for Yatzy Analysis

## Problem

The Yatzy solver computes optimal decisions for ~2M states, but most are irrelevant for human play. States can be rare for two independent reasons:

1. **Dice rarity**: the dice sequences needed to reach the state are extremely unlikely regardless of strategy
2. **Behavioral rarity**: the upstream decisions needed to reach the state are ones no human would make, even though the dice allowed it

To produce analysis useful for humans (e.g., decision guides, difficulty maps, worked examples), we need to filter to the intersection of dice-plausible AND human-reachable states.

## Approach: Softmax Policy Simulation

We already have the full value tables V_θ(a, s) for all actions at every state from the DP solver. Instead of following the optimal (argmax) policy, we simulate games under a **softmax choice model** that represents a skilled but imperfect human:

```
P(action a | state s, θ, β) ∝ exp(β · V_θ(a, s))
```

Where:

- `θ` controls risk preference (use θ = 0 for EV-optimal reference)
- `β` controls rationality / skill level:
  - β → ∞: always picks the optimal action (the solver)
  - β → 0: picks uniformly at random
  - β ≈ 0.3: picks optimally when the EV gap between best and second-best action is large (>5 pts), but makes frequent errors when alternatives are close (within 1-2 pts)

This applies to BOTH reroll decisions and category assignment decisions. At each decision point in a turn (up to 2 reroll decisions + 1 category decision), sample from the softmax distribution over all legal actions using the corresponding V_θ values.

## Implementation Steps

### Step 1: Calibrate β

Simulate 100K games under softmax(θ=0, β) for β ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0}. For each β, record:

- Mean score and std
- Score distribution (histogram or percentiles)
- Per-decision accuracy vs optimal (what fraction of decisions match argmax)

Select β such that the mean score falls in the 210-235 range. This represents a decent experienced human player. β = 0.3 is the initial guess; adjust based on results.

**Validation check**: at the chosen β, verify that:

- "Obvious" decisions (EV gap > 5) are followed >95% of the time
- "Close" decisions (EV gap < 1) are followed ~55-65% of the time
- The player almost never makes catastrophic errors (EV gap > 15)

### Step 2: Simulate and Record State Visits

Run 100K games under the calibrated softmax policy. At each turn, record the full game state: `(turn, upper_score, scored_categories_mask)`.

Also run 100K games under the optimal policy (β = ∞, θ = 0) for comparison.

### Step 3: Filter by Cumulative Coverage

At each turn t (1-15), sort states by visit frequency under the softmax policy. Keep the smallest set of states that covers **90% of all games**. This is the human-plausible state set for turn t.

This adapts to the hourglass structure automatically:

- Turn 1: 1 state (trivial)
- Turn 5: probably a few hundred states
- Turn 9: probably a few thousand states (peak diffusion)
- Turn 15: probably ~20 states (late-game concentration)

### Step 4: Annotate States with Decision Data

For each state in the human-plausible set, compute (from existing DP tables):

- The optimal action and its EV
- The second-best action and its EV
- The EV gap (decision difficulty)
- The set of θ values for which the optimal action changes (θ-sensitivity)

For category decisions specifically, also record which dice outcomes (after final roll) produce a decision where the gap is < 2 points — these are the "hard" decisions at this state.

### Step 5: Produce the Filtered Dataset

Output: a table with columns:

```
turn, upper_score, scored_mask, visit_frequency_softmax, visit_frequency_optimal, num_hard_decisions, avg_ev_gap, theta_sensitive
```

This dataset is the foundation for:

- Worked examples in a strategy guide (sample from high-frequency, low-gap states)
- Decision difficulty maps (aggregate gap statistics by turn and state type)
- θ-sensitive scenario cards (states where risk preference changes the optimal action)
- Heuristic policy design (focus rules on high-frequency, high-gap states where humans err most)

## Notes

- The softmax simulation needs V_θ(a, s) for ALL legal actions, not just the optimal one. This data exists in the DP tables but needs to be accessible during simulation.
- For reroll decisions, "all legal actions" means all 32 reroll masks. For category decisions, it means all unfilled categories.
- The V values used in the softmax should be the full remaining-game values (not just immediate scores), so the softmax player is "trying to be optimal but noisy" rather than "greedy with noise."
- If computing softmax over reroll masks is expensive (32 actions × many dice states), an acceptable approximation is to use optimal rerolls (β = ∞ for rerolls only) and softmax only for category decisions. This models a human who thinks carefully about rerolls but errs on category assignment — a defensible simplification since category errors are the ones that compound across turns.
