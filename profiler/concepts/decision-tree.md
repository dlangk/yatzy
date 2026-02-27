# Decision Trees

A **decision tree** is a machine learning model that makes predictions by asking a sequence of yes/no questions about the input features, organized in a tree structure.

## How They Work

Starting at the root, each internal node tests a condition (e.g., "is dice_count_5 >= 3?"). Based on the answer, you follow the left or right branch. At a leaf node, you get the predicted action.

## Why They Excel at Yatzy

Yatzy decisions are fundamentally **combinatorial**: the optimal action depends on discrete thresholds:

- Do I have 3 or more fives? → Keep them for the upper section
- Is the straight complete? → Score it now
- Is upper progress ≥ 50? → The bonus is almost secured, shift strategy

Decision trees encode these thresholds *natively* as split points. Neural networks must approximate the same step-function boundaries using smooth activations — a much harder task.

## Compression Results

| Depth | Parameters | Mean Score | vs Optimal |
|-------|-----------|------------|------------|
| 5     | 276       | 157        | -91        |
| 10    | 6,249     | 216        | -32        |
| 15    | 81K       | 239        | -9         |
| 20    | 413K      | 245        | -3         |

A depth-20 tree with 413K parameters achieves 98.6% of optimal play, compressing the 8 MB lookup table by ~20×.
