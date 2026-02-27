# Composable Filter Grammar

A **composable filter grammar** defines reroll actions using semantic building blocks -- named operations like "Keep Pair" or "Keep Face 6" -- that can be combined to express complex dice-keeping strategies.

## The Problem with Raw Bitmasks

A raw keep action is a 5-bit bitmask specifying which of the 5 dice to retain. There are 31 non-trivial masks (excluding "reroll everything"). These masks are opaque: mask 10110 means "keep dice in positions 1, 3, and 4," which says nothing about *why* those dice are being kept.

Humans don't think in bitmasks. They think: "keep the pair of fives" or "keep everything that's part of the straight." The gap between how the solver represents actions (bitmasks or multiset indices) and how humans conceptualize them (semantic intentions) is the problem the filter grammar addresses.

## Semantic Verbs

The composable filter grammar replaces 31 positional bitmasks with approximately **15 semantic verbs**:

- **Keep Face N**: retain all dice showing face value N (6 verbs, one per face)
- **Keep Pair**: retain the highest pair
- **Keep Triple**: retain three-of-a-kind
- **Keep Straight**: retain dice contributing to a straight sequence
- **Keep All**: retain all five dice (stop rerolling)
- **Keep None**: reroll everything
- **Keep High**: retain dice above a face-value threshold

Each verb is a function from a dice configuration to a keep action. Given dice [2, 3, 5, 5, 6], "Keep Pair" produces the mask for the two 5s, and "Keep Face 6" produces the mask for the single 6.

## Composition

Verbs can be composed to express compound intentions:

- "Keep Face 6 AND Keep Pair" -- retain all sixes and any pair among the remaining dice
- "Keep Straight OR Keep Triple" -- use whichever produces a better partial hand
- "Keep High AND Keep Face 1" -- keep high dice plus any ones (chasing a specific combination)

The grammar supports AND (intersection), OR (union), and NOT (complement) combinators. This gives 2^15 possible compound actions from 15 atomic verbs -- far more expressive than the individual verbs alone, but still structured and interpretable.

## Performance Impact

Replacing bitmask enumeration with a verb grammar for reroll decisions improved a heuristic agent's mean score by **+15.7 points**. The grammar constrains the action space to semantically meaningful keeps, which:

1. Eliminates nonsensical keeps (e.g., keeping one die from each face value for no strategic reason)
2. Reduces the effective branching factor from 31 to ~15 per decision point
3. Makes the resulting policy human-readable and explainable

## Why It Bridges Human and Machine Play

The solver operates on multiset indices (0--461) for mathematical precision. Humans operate on intentions ("keep the pair, reroll the rest"). The filter grammar provides a translation layer: each semantic verb maps to a specific multiset index given the current dice, and the resulting policy can be expressed as "Keep Pair when you have a pair and need the upper bonus" rather than "select keep index 127."

This makes the grammar a natural output format for distilled policies. A decision tree that outputs verb names instead of multiset indices produces a strategy that a human can actually follow without lookup tables.

## Expressiveness vs. Optimality

The grammar cannot express every possible keep action. Some optimal keeps are combinations that no single verb or small composition captures -- for example, keeping a 2 and a 5 specifically because of an unusual interaction between upper-section progress and the remaining open categories. These cases are rare enough that the grammar's +15.7 EV improvement far outweighs the occasional suboptimal keep.

The key insight is that the vast majority of optimal reroll decisions have a semantic explanation: "I'm building toward a straight," "I'm collecting sixes for the bonus," "I'm preserving my three-of-a-kind." The grammar formalizes these explanations as executable code.

## Design Principles

The grammar follows three design principles:

1. **Face-value grounding**: every verb refers to observable properties of the dice (face values, counts, sequences), not to internal solver state
2. **Composition closure**: any combination of verbs produces a valid keep action (possibly the empty set, meaning reroll everything)
3. **Monotonicity**: adding more verbs to an AND composition never increases the number of kept dice; adding to an OR composition never decreases it

These properties ensure the grammar is predictable and easy to reason about, even for compound actions.
