# Credit Assignment

The **credit assignment problem** asks: when a reward is received long after the decisions that caused it, how do you determine which decisions deserve credit?

## The Problem in General

In reinforcement learning, an agent takes many actions before receiving a reward signal. A chess player might make 40 moves before winning or losing. Which moves were good? Which were mistakes? Distributing credit across a long sequence of decisions is one of the hardest problems in RL.

The difficulty scales with two factors: the **delay** between action and reward, and the **entanglement** of multiple actions contributing to a single outcome. When both are large, credit assignment becomes extremely challenging.

## Credit Assignment in Yatzy

The upper section bonus is the most challenging credit assignment problem in Yatzy:

- The **50-point bonus** is awarded only after all 15 categories have been scored
- But every upper-section decision (Ones through Sixes) contributes to whether the 63-point threshold is reached
- A weak Threes score in round 3 might cause a bonus miss in round 15 -- twelve turns later

The bonus reward is **delayed and non-decomposable**. You cannot attribute 50/6 = 8.3 points of credit to each upper category, because the contributions are highly non-linear. Scoring 12 in Fours contributes nothing if the rest of the upper section already totals 63, but it is decisive if the rest totals 52.

This non-linearity means the "true credit" for the bonus depends on the entire sequence of upper scores, not on any individual decision in isolation.

## Why Backward Induction Solves It

The DP solver sidesteps the credit assignment problem entirely. Because it works backward from terminal states, the value of every intermediate state already accounts for all future consequences, including the bonus.

When the solver evaluates "score 9 in Threes with upper_total = 38," it knows exactly how that affects bonus probability. The value function V(upper=47, scored=...) already encodes the expected bonus contribution given optimal future play. No credit needs to be assigned after the fact -- it is baked into the value at computation time.

This is a fundamental advantage of model-based planning over model-free learning. The model (transition probabilities and reward structure) makes all consequences immediately computable.

## Why RL Struggles

Model-free RL methods (Q-learning, policy gradient) must discover the bonus's importance from experience:

- The signal is **sparse**: binary (bonus or no bonus), occurring once per game
- The signal is **delayed**: up to 15 turns between the first upper-section decision and the bonus award
- The signal is **entangled**: all six upper-section decisions jointly determine the bonus

Agents typically need millions of games to learn reliable bonus-tracking behavior. Even then, they often undervalue early upper-section investments because the reward signal is so diluted across the episode.

Temporal difference (TD) methods partially address this by propagating value estimates backward through the episode. But even TD learning struggles with the non-linear threshold at 63: the value function has a sharp discontinuity that is hard to approximate with smooth function approximators.

## Broader Significance

The credit assignment problem is not unique to Yatzy. It appears whenever rewards are sparse and delayed:

- In Go, a single win/loss signal must be distributed across 200+ moves
- In robotics, a "task completed" reward must be traced back through hundreds of motor commands
- In drug discovery, a successful molecule must be attributed to specific structural choices

What makes Yatzy interesting as a case study is that we have the exact solution (via backward induction) and can precisely measure how much each decision actually contributes to the bonus. This ground truth is rarely available in more complex domains, making Yatzy a useful testbed for studying credit assignment algorithms.

The DP solution reveals that the "true credit" for the bonus is highly concentrated: the last 2--3 upper-section decisions before the threshold typically carry 80%+ of the marginal bonus value. Early decisions matter primarily through their effect on the *distribution* of upper progress, not through direct threshold proximity.

## Reward Shaping

One technique for alleviating the credit assignment problem in RL is **reward shaping**: adding intermediate rewards that guide the agent toward the delayed goal. For Yatzy, this might mean giving a small positive reward for each point of upper-section progress, rather than waiting for the 50-point bonus at the end.

However, naive reward shaping can distort the optimal policy. The agent might chase intermediate rewards at the expense of the true objective. Potential-based reward shaping (Ng et al., 1999) avoids this problem by constructing the shaping function from the value function itself -- but if you already have the value function, you don't need RL.

This circularity illustrates why backward induction is so powerful for Yatzy: it solves credit assignment as a byproduct of exact computation, while RL must struggle to rediscover the same information through trial and error.
