# Markov Decision Process

A **Markov decision process** (MDP) is a mathematical framework for modeling sequential decision-making where outcomes are partly random and partly under the control of a decision-maker.

## The Four Components

An MDP is defined by:

- **States (S)**: every distinct situation the agent can be in
- **Actions (A)**: the choices available in each state
- **Transitions P(s'|s,a)**: the probability of reaching state s' after taking action a in state s
- **Rewards R(s,a)**: the immediate payoff for taking action a in state s

The **Markov property** means the future depends only on the current state, not on how you got there. History is irrelevant once you know where you are now. This memoryless property is what makes MDPs tractable: you only need to solve each state once, regardless of the exponentially many paths that might lead to it.

## Finite-Horizon MDPs

When the process has a fixed number of steps, it is called **finite-horizon**. There is no need for discount factors or infinite-sum convergence arguments. You simply work backward from the last step, computing optimal actions at each stage.

This is in contrast to infinite-horizon MDPs (used in most RL applications), where you need a discount factor gamma < 1 to ensure values converge. Yatzy's 15-round structure means the finite-horizon formulation applies exactly.

## Yatzy as an MDP

A game of Yatzy is a finite-horizon MDP with 15 rounds:

- **States**: upper section progress (0--63) crossed with scored categories (15-bit mask), giving ~1.43 million reachable states out of 2,097,152 total slots
- **Actions**: which dice to keep (reroll decisions) and which category to score
- **Transitions**: determined by the dice probability distribution -- rolling 5d6 produces 7,776 equally likely outcomes, which collapse to 252 distinct sorted outcomes
- **Rewards**: the points earned when scoring a category, plus the 50-point bonus triggered at upper total >= 63

Because the state space is finite and enumerable, the MDP can be solved exactly by backward induction. No approximation, no neural networks, no simulation -- just exhaustive computation over every reachable state.

## Why the Markov Property Holds

It might seem like history matters -- "I already used Threes, so I can't use it again." But that information is encoded in the scored-categories bitmask. The state captures everything relevant about the past. Two games that arrive at the same (upper_progress, scored_mask) pair face identical futures, regardless of the path taken.

This is a design choice in the state representation. If we had omitted upper-section progress from the state and tried to track only which categories are scored, the Markov property would break: the future value would depend on the invisible history of upper scores. The 64-level upper-progress variable restores the Markov property by making all decision-relevant history visible.

## Comparison to Other Games

Not all games are this clean. Poker has hidden information (opponents' cards), making it a partially observable MDP. Go has ~10^170 states, making exact solution impossible. Backgammon is an MDP but with a much larger state space than Yatzy.

Yatzy sits in a sweet spot: complex enough to be strategically interesting (1.43M states, non-trivial bonus structure), but small enough for exact solution on consumer hardware in about one second.

## From MDP to Algorithm

Once a problem is formulated as an MDP, a family of solution methods becomes available:

- **Value iteration**: iteratively apply the Bellman equation until convergence (for infinite-horizon MDPs)
- **Policy iteration**: alternate between evaluating a policy and improving it
- **Backward induction**: solve each time step once, from last to first (for finite-horizon MDPs)
- **Reinforcement learning**: estimate the value function from sampled trajectories (when the model is unknown)

For Yatzy, backward induction is the right tool: the horizon is finite (15 rounds), the state space is enumerable (2M states), and the transition model is known exactly (dice probabilities). No other method can match its combination of speed and exactness for this problem.

## Single-Player vs. Multi-Player

The MDP formulation treats Yatzy as a single-player optimization problem: maximize your own score regardless of opponents. This is valid because Scandinavian Yatzy has no interaction between players during a game -- your dice rolls and decisions are independent of other players' actions.

In a competitive context, the relevant question shifts from "maximize expected score" to "maximize probability of winning." This transforms the problem from an MDP into a stochastic game, which is significantly harder to solve. The risk-sensitive parameter theta provides a partial bridge: a risk-seeking strategy (theta > 0) implicitly optimizes for high-score outcomes that are more likely to win, even though it does not explicitly model the opponent.
