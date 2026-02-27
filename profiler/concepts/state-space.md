# State Space

A **state space** is the set of all possible configurations a system can be in. For games, it's every distinct situation a player might face.

## Yatzy's State Space

A Yatzy game state is determined by two things:

- **Scored categories**: which of the 15 categories have been filled (a 15-bit bitmask → 32,768 combinations)
- **Upper section progress**: how many points you've accumulated toward the 63-point bonus threshold (0 to 63)

This gives 32,768 × 64 = **2,097,152 unique states**.

## Why It Matters

A finite, enumerable state space means we can solve the game *exactly*. We compute the optimal expected value for every single state, store the results in an 8 MB lookup table, and answer any "what should I do?" question with a simple table lookup.

Games with astronomically large state spaces (like Go, with ~10^170 states) require approximation. Yatzy's 2M states fit comfortably in memory, making exact solutions feasible.
