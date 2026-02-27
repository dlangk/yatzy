# Supervised Distillation

**Knowledge distillation** is a technique where a small model (the student) is trained to reproduce the behavior of a large model or exact solver (the teacher). The goal is to compress the teacher's knowledge into a more compact, deployable form.

## How It Works

1. **Generate training data**: run the teacher (the DP oracle) on a large set of game states to produce (state, optimal_action) pairs
2. **Train the student**: fit a smaller model -- a decision tree, neural network, or rule list -- to predict the teacher's actions from state features
3. **Evaluate**: simulate games with the student's policy and measure the mean score relative to the teacher

The student never needs to understand *why* the teacher's actions are optimal. It only learns to mimic them. This is supervised learning in its purest form: the oracle provides perfect labels, and the student learns to reproduce them.

## Training Data

The DP oracle can label any game state instantly via table lookup. A typical training set consists of 200,000 simulated games, each producing 15 state-action pairs (one per round), giving ~3 million labeled examples. The features include:

- Dice face counts (how many 1s, 2s, ..., 6s)
- Upper section progress (0--63)
- Which categories are open (15 binary features)
- Rerolls remaining (0, 1, or 2)

## Compression Results

| Student Model | Parameters | Mean Score | EV Loss vs Optimal |
|---|---|---|---|
| Decision tree (depth 5) | 276 | 157 | -91 |
| Decision tree (depth 10) | 6,249 | 216 | -32 |
| Decision tree (depth 15) | 81K | 239 | -9 |
| Decision tree (depth 20) | 413K | 245 | -3 |
| MLP (64 hidden) | ~15K | 222 | -26 |

Decision trees outperform neural networks at equal parameter counts because Yatzy decisions are fundamentally threshold-based. The optimal action often changes at sharp boundaries (e.g., "if upper_gap <= 7, change strategy"), which trees encode natively as split points but MLPs must approximate with smooth activations.

## The Distillation Tradeoff

Every compression discards information. The question is *which* information matters most:

- **Common states**: the student should get these right, since they dominate expected score
- **Rare states**: mistakes here have smaller impact on average but may cause catastrophic outcomes
- **Bonus-boundary states**: errors near upper_total = 63 are disproportionately costly because they can cause a 50-point bonus miss

A well-designed distillation pipeline weights training examples by state frequency and bonus proximity to minimize EV loss per parameter.

## Why Not Just Use the Table?

The full oracle table requires the solver infrastructure to generate and ~16 MB to store per decision type (category selection, keep-with-1-reroll, keep-with-2-rerolls). A distilled model can run on any device, requires no precomputation, provides interpretable explanations for its decisions, and opens the door to human-comprehensible strategy guides. The tradeoff between model size and play quality defines a Pareto frontier of practical interest.

## Distillation vs. Direct RL

An alternative to distillation is training the student model directly via reinforcement learning, without a teacher. This avoids the teacher dependency but introduces several challenges:

- RL must rediscover the optimal policy from scratch, requiring millions of simulated games
- Credit assignment for the upper bonus is difficult (see credit-assignment.md)
- The resulting policy has no guarantee of approaching optimal play

Distillation sidesteps all of these issues by leveraging the exact oracle. The student's training signal is perfect (no noise, no delay, no sparsity), and convergence is fast because supervised learning on clean labels is a much easier optimization problem than RL.

## Human-Level Play

An interesting benchmark is human performance: typical experienced Yatzy players score 220--230. A distilled decision tree at depth 10 (6,249 parameters) matches this level. This suggests that human-level Yatzy requires roughly 6K parameters of "strategy knowledge" -- a surprisingly small amount, compressed from the 16 MB oracle.
