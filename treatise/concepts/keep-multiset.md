# Keep Multiset

A **keep multiset** is the set of dice a player chooses to retain after a roll. Order does not matter: keeping {1, 1, 3} is the same action as keeping {1, 3, 1}.

## From Masks to Multisets

When you roll 5 dice, there are 2^5 = 32 subsets you could keep (including the empty set, where you reroll everything). But many of these subsets produce identical results because dice of the same face value are interchangeable.

For example, if you rolled [2, 3, 2, 5, 2], keeping "positions 1 and 3" versus "positions 1 and 5" both result in keeping two 2s. The probability distributions over subsequent rolls are identical in both cases. What matters is the *multiset* of kept face values, not which physical dice are held.

The distinction is between **positional masks** (which slots to keep) and **value multisets** (which face-value combination to retain). Two different masks that produce the same multiset lead to identical future expected values.

## The Numbers

Across all possible 5d6 outcomes:

- **31 non-empty bitmasks** per outcome (2^5 - 1)
- **462 unique keep multisets** across the entire outcome space (multisets of 0-5 elements from {1,2,3,4,5,6})
- **4,368 non-zero entries** in the precomputed KeepTable (mapping outcome x keep to transition probabilities)

For any single dice outcome, the number of unique keeps is much smaller -- typically 10 to 25, depending on how many repeated face values appear. An outcome like [1,2,3,4,5] (all distinct) has fewer duplicate masks than [2,2,2,3,3] (multiple repeats).

## Why Deduplication Matters

Without deduplication, the solver would evaluate redundant keeps that produce identical future expected values. Collapsing 31 masks down to the unique multisets saves roughly **85% of the work** in the keep-evaluation phase.

This is not a micro-optimization. The keep-evaluation phase is the dominant cost in the widget solver. An 85% reduction in redundant evaluations translates directly into a ~6x speedup in the most expensive part of the computation.

## Counting Multisets

The number 462 comes from a standard combinatorial formula. The number of multisets of size k drawn from an alphabet of size n is C(n+k-1, k). Summing over all keep sizes from 0 to 5 dice, with 6 face values:

- Keep 0 dice: C(5,0) = 1
- Keep 1 die: C(6,1) = 6
- Keep 2 dice: C(7,2) = 21
- Keep 3 dice: C(8,3) = 56
- Keep 4 dice: C(9,4) = 126
- Keep 5 dice: C(10,5) = 252

Total: 1 + 6 + 21 + 56 + 126 + 252 = 462. Not all of these are reachable from every dice outcome -- you can only keep face values that actually appear in your roll -- but 462 is the universal upper bound.

## Implementation

The solver precomputes a `KeepTable` that maps each (dice_outcome, keep_multiset) pair to the probability distribution over next-roll outcomes. Each multiset is encoded as a small sorted tuple of face values and assigned a canonical index (0--461) via a precomputed lookup.

During the widget pass, only canonical keeps are evaluated. The max-over-keeps at decision nodes iterates over deduplicated indices, ensuring no work is duplicated. The KeepTable itself is computed once at startup and reused across all 1.43 million state evaluations.

## Analogy

Think of multiset deduplication as recognizing that "pick up the two red chips from the left side" and "pick up the two red chips from the right side" are the same action when both sides have identical red chips. The chips are fungible; only the count and type matter.

In the solver, this insight is baked into the data structures. The KeepTable never stores positional information -- only face-value multisets. This means the entire keep-evaluation pipeline operates in the mathematically natural space of multisets rather than the artificially larger space of positional masks.

## Connection to Combinatorial Game Theory

The multiset representation is an example of a broader principle in combinatorial game theory: exploiting symmetry to reduce the size of the game tree. Just as chess engines recognize that symmetric board positions need not be analyzed twice, the Yatzy solver recognizes that symmetric dice configurations (same face values in different positions) need not be evaluated independently.

This symmetry reduction is exact -- no information is lost. The deduplication produces identical results to evaluating all 31 masks, because the Yatzy scoring function depends only on face-value counts, not on which physical die shows which value. Any scoring function with this property (formally: invariant under permutations of dice positions) admits the same multiset optimization.
