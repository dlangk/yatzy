:::section{#conclusion}

## The Conclusions

This project started because one evening, when playing with my family, we got to wondering whether you can actually be good at Yatzy. It turns out the answer is:

**Yes, you can be great at Yatzy.** In fact, there is even a "perfect" policy that can be computed. There is a way to reduce the state space of Yatzy to ~1.43M by ignoring the dice order, forgetting the past, pruning the impossible, breaking each turn into a widget, and always moving forward. As a result, a laptop solves the whole thing in seconds.

**An optimal Scandinavian Yatzy player scores 248.4 on average, with a standard deviation of 38.5.**

**There is a predictable reward for taking risk.** Most of us don't play for a good average, we play to win. We have added a risk-dial &theta; that lets the solver reach for tail outcomes. We carefully mapped the mean-variance frontier, learning that a small amount of extra risk buys a decent bump in the top percentiles of outcomes. But, if you push it too far, the policy collapses into a max-score gambler whose best game shows up with a probability around 10<sup>&minus;19</sup>.

**There is a way to win slightly more than 50% of games if you get to play after your opponent.** If the objective is to beat another player in a specific game, instead of maximizing e.g. your average score, it turns out the player who gets to go second has a small advantage. If they adapt their risk-taking turn by turn, the very best they can squeeze out is about 9 extra wins per 1000 games. Even a clairvoyant player, who somehow knew your final score before the game started, would win only 55.3% of the time.

**This is why Yatzy is such a good family game!** There is enough skill that the optimal strategy gives a significant bump in average score. But there is enough luck involved that, even playing perfectly, a seasoned veteran will regularly lose to a novice. Across a lifetime of games, you will likely see a handful of scores above 330. So, Yatzy has room for skill, but it is mostly luck. When you win, you feel skilled. When you lose, you were unlucky. This is the balance a great family game needs. 😊

:::math

The key results, across the parts of the game a solver can reach:

:::insight
Solver: 248.4 EV, 38.5 std, 1.1s solve, 37-value \theta sweep\\
Distribution: a 16-component mixture from 4 binary events (bonus, Yatzy, two straights)\\
Risk: exponential-utility \theta, mean-variance frontier, exact tail to 10^{-15}\\
Multiplayer: best adaptive policy +0.86pp, clairvoyant ceiling 55.3\%
:::

The mathematical spine connecting risk and multiplayer is a single relationship: on the mean-variance frontier, &part;E/&part;V = &minus;&theta;/2. Running the solver at a given &theta; lands on the exact point where it pays that many mean points per unit of variance. Push the same identity through a Gaussian win-probability model and it hands you the optimal adaptive risk for head-to-head play, &theta;* = (E<sub>1</sub> &minus; E<sub>2</sub>) / (V<sub>1</sub> + V<sub>2</sub>): positive when trailing, negative when leading, and vanishing when scores are even. It scales inversely with the variance still left in the game, which is why the right amount of risk is almost nothing early and can be decisive on the final turn.

:::

:::code

Everything here is reproducible from source:

```bash
# Solve, simulate, analyze
just setup              # Build solver + install analytics
just precompute         # theta=0 strategy table (1.1s)
just sweep              # All 37 theta values (resumable)
just simulate           # 1M Monte Carlo games
just pipeline           # compute + plot + categories + efficiency
just density            # Exact forward-DP PMFs
```

:::

:::
