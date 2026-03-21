# Multiset Combination Problem

A **multiset** (or "bag") is a collection where elements can repeat but order does not matter. The multiset combination problem asks: how many ways can you choose *k* items from *n* types, with repetition allowed?

## Formula

The answer is the "stars and bars" formula:

> C(n + k - 1, k) = (n + k - 1)! / (k! * (n - 1)!)

## In Yatzy

Rolling 5 dice with 6 faces is equivalent to choosing a multiset of size 5 from 6 types: C(6 + 5 - 1, 5) = C(10, 5) = **252** unique rolls. Similarly, keeping *k* dice from 6 possible faces gives C(6 + k - 1, k) unique keeps for each value of *k*. Summing across all keep sizes (0 through 5) gives **462** total unique keeps.
