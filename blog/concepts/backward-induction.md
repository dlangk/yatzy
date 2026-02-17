# Backward Induction

**Backward induction** is an algorithm for solving sequential decision problems by starting at the end and working backward to the beginning.

## How It Works

1. **Start at terminal states**: at the end of a Yatzy game (all 15 categories filled), the value is simply the final score
2. **Step backward**: for each earlier state, consider all possible actions and their consequences. The value of a state is the best expected outcome achievable from that point
3. **Repeat** until you reach the starting state

At each step, you're answering: "If I play optimally from here, what's my expected score?"

## Why Backward?

Working forward doesn't work because you don't know the future. But working backward, every future state is already solved. When you evaluate keeping vs. rerolling dice, you can look up the exact expected value of every possible outcome.

## In Yatzy

The solver processes all 2,097,152 states grouped by how many categories remain. It starts with states where 14 of 15 categories are filled (one turn left), then 13, and so on. Each group depends only on already-computed future states.

The entire backward pass takes ~2.3 seconds and produces a complete strategy table.
