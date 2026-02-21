# The Mathematics of Optimal Yatzy

**A Pedagogical Report on Decision-Making Under Uncertainty**

To a casual player, Scandinavian Yatzy is a game of luck mitigated by intuition. You roll five dice, you keep some, you reroll the rest, and you write a number on a scorecard.

But to a computational mathematician, Yatzy is a perfectly engineered trap. It sits in a "Goldilocks zone" of complexity: its state space is just small enough to be solved to absolute mathematical perfection, yet its statistical properties—massive variance, discontinuous rewards, and deep sequential dependencies—make it practically immune to modern Artificial Intelligence architectures like Deep Reinforcement Learning (RL).

This report takes you from the basic mechanics of rolling dice to the mathematical frontier of dynamic programming and risk-sensitive utility theory. We will deconstruct how the game was exactly solved, explore the geometry of risk, and examine why the AI algorithms that conquered Chess and Poker still fall roughly 5% short of perfect Yatzy.

---

## Part I: The Game as a Mathematical Object

Before we can solve a game, we must translate its physical components into a formal mathematical structure.

### Rules and Structure

Scandinavian Yatzy is played over exactly 15 turns. The scorecard contains 15 categories: the "Upper Section" (Ones through Sixes) and the "Lower Section" (One Pair, Two Pairs, Three of a Kind, Four of a Kind, Small Straight, Large Straight, Full House, Chance, and Yatzy).

On each turn, you roll five standard dice. You may keep any subset of these dice and reroll the rest, up to two times. After the final roll, you must assign the five dice to exactly one empty category on your scorecard. If the dice do not fit the category's requirements, you score a zero.

A critical structural rule drives the entire strategy of the game: **The Upper Bonus**. If the sum of your scores in the Upper Section reaches 63 or more, you are awarded a flat 50-point bonus.

What makes Yatzy mathematically fascinating?

1. **Finite Horizon:** The game always ends in exactly 15 turns. There are no infinite loops.
2. **Perfect Information:** There is no hidden state (unlike Poker). You know exactly what categories are open and what dice you have.
3. **Stochastic Transitions:** You control the *decisions* (which dice to keep, which category to score), but probability controls the *transitions* (the outcome of the rerolls).
4. **Combinatorial State Space:** Every decision permanently alters your scorecard, fundamentally changing the geometry of all future decisions.

### Formalizing the State Space

In optimal control theory, a "state" must contain all the historical information necessary to make the best possible future decision. The past only matters if it restricts the future. This is the **Markov Property**.

In Yatzy, the exact sequence of dice you rolled on Turn 1 does not matter on Turn 10. The only thing that dictates your future options is your current scorecard. Therefore, *the scorecard is the state*.

We encode a state mathematically using just two pieces of information:

1. **The Category Bitmask:** A 15-bit string representing which of the 15 categories are already filled ( possibilities).
2. **The Upper Section Progress:** An integer from 0 to 63 tracking the sum of the Ones through Sixes. (Any score above 63 is mathematically identical to exactly 63, as the bonus is already secured).

Multiplying these gives a theoretical volume of  states. However, we can apply *reachability pruning*. For example, it is physically impossible to have an upper score of 63 if the only category you have filled is "Ones" (which maxes out at 5 points). By pruning these mathematical ghosts, the game collapses to exactly **~536,000 reachable turn-start states**.

### The Turn "Widget"

Because Yatzy obeys the Markov Property (the dice have no memory), we can cleanly isolate a single turn. A turn is not a single decision; it is a combinatorial "widget" composed of six alternating layers of chance and choice.

> **Visual Prompt 1: The State-Space "Hourglass"**
> * **Chart Type:** Directed Acyclic Graph (DAG) arranged vertically.
> * **Axes/Layout:** Top-to-bottom flow representing a single turn.
> * **Data Shown:** Top node (Turn Start) expands via a chance node into 252 branches (Roll 1 distinct sorted outcomes). These funnel into a decision node (Keep 1), which contracts the branches. The graph expands again (Roll 2 chance), contracts (Keep 2 decision), expands (Roll 3 chance), and finally contracts through the Category Assignment decision into discrete "Next State" nodes.
> * **Key Annotations:** Label the expanding layers "Stochastic Expansion (Nature)" and the contracting layers "Decision Compression (Player)".
> * **Insight:** Illustrates how stochastic chance nodes exponentially expand the game tree, while rational decision nodes act as funnels that compress the chaos back into manageable, deterministic paths.
>
>

---

## Part II: Solving the Game

How do you find the mathematically perfect move on Turn 1 when the consequences of that move won't be fully realized until 14 turns later? You cheat time. You start at the end of the game and work backward—a technique known as **Backward Induction**.

### The Bellman Equation for Yatzy

To understand Backward Induction, imagine the simplest possible scenario: Turn 15, after your third roll. You have exactly one empty category left (say, *Chance*). There is no strategic dilemma to solve. The Expected Value (EV) of this state is simply the sum of the dice on the table.

Now, step back one layer. You have just finished your *second* roll. You must make a **Keep Decision**. How do you choose?
For every possible subset of dice you could keep, you calculate the probability-weighted average of all possible third rolls. Because you already solved the absolute end of the game, you know exactly what those third rolls will be worth.

Mathematically, this is the Bellman Equation:

* At a **Chance Node** (rolling dice), the value is the *expected value* (, the probability-weighted sum) of the outcomes.
* At a **Decision Node** (keeping dice or scoring a category), the value is the *maximum* () over all available actions.

> **Visual Prompt 2: Worked Backward-Induction Example**
> * **Chart Type:** Bottom-up numerical decision tree.
> * **Axes:** Branching upward from terminal leaves to a root decision.
> * **Data Shown:** A late-game scenario. The player rolled `[6, 6, 2, 3, 4]` on Roll 2. Only the *Sixes* category is open.
> * **Key Annotations:** A square Decision Node branches into "Keep [6,6]" and "Keep All". "Keep All" leads directly to a terminal score of 12. "Keep [6,6]" leads to a circular Chance Node, which branches into binomial probabilities (rolling zero, one, two, or three more 6s on the final roll). The EV of the chance node evaluates to 16.0. The `max(12, 16.0)` bubble lights up, selecting "Keep [6,6]".
> * **Insight:** Demystifies dynamic programming. Shows exactly how taking a statistical average of future probabilistic states dictates the optimal deterministic action today.
>
>

### The EV-Optimal Baseline

By recursively applying this equation from Turn 15 all the way back to Turn 1 across all 536,000 states, we achieve mathematical perfection. We have mapped the absolute ceiling of the game.

The exact Expected Value (EV) of an optimal game of Scandinavian Yatzy is **248.4 points**, with a standard deviation () of **38.5 points**.

The hallmark of the EV-optimal policy is **Option Value**. Early in the game, the algorithm rarely forces difficult combinations. It happily takes zeros in hard categories (like *Yatzy* or *Four of a Kind*) if the dice are cold, fiercely preserving "easy" categories (like *Chance* or *One Pair*) as safety nets. The algorithm understands that an empty category is not just a scoring receptacle; it is a financial put-option that absorbs the game's inherent variance. Late in the game, it ruthlessly cashes in this stored flexibility.

> **Visual Prompt 3: The Score Distribution Histogram**
> * **Chart Type:** Probability Density histogram.
> * **Axes:** X-axis = Final Score (0 to 374), Y-axis = Probability Density.
> * **Data Shown:** The exact probability mass function of the EV-optimal DP solver. A wide, left-skewed bell curve.
> * **Key Annotations:** Vertical dashed lines marking the Mean (248.4), Median (~252), p5 (~183), and p95 (~312). Highlight the distinct bimodal "bump" on the left tail.
> * **Insight:** Shows that playing optimally does not guarantee a high score; it guarantees the best *distribution* of scores. The massive variance () is a fundamental property of the game. The left-tail bump represents "doomsday" games where the 50-point upper bonus is missed despite perfect play.
>
>

---

## Part III: An Instructive Failure

To truly understand why the Bellman equation is so brilliant, we must see what happens when we intentionally break it.

### The Max-Policy

Let's alter our algorithm. Instead of taking the *average* () over the stochastic dice rolls at chance nodes, we replace it with a *maximum* (). We are asking the AI: "If you could magically dictate the exact outcome of every dice roll, what is the best possible strategy?"

If we evaluate the game using this "Max-Policy," the solver proudly outputs a precomputed Expected Value of **374 points**—a mathematically perfect game.

But what happens when we force this delusional AI to play a million simulated games with real, randomized dice? **Its actual mean score collapses to an abysmal 118.7 points**—worse than a player choosing actions entirely at random.

### Why It Fails

The Max-Policy fails due to **uniform overconfidence**. On Turn 1, if it rolls `[1, 2, 3, 4, 6]`, it looks at the `[6]` and thinks, "I will keep this 6, and because I am infinitely lucky, my next roll will be four more 6s, giving me a Yatzy!" It throws away a guaranteed Large Straight, rolls garbage, and takes a zero.

Because it believes luck is guaranteed, it places zero value on safe fallback options. It zeroes out premium categories in pursuit of jackpots it will never hit. It reaches Turn 15, fails to roll perfectly, and is forced to take zeroes in its highest-value slots.

**The Lesson:** In sequential decision-making, uncertainty is not "noise" that blurs the optimal path; **uncertainty is the signal that creates strategic depth.** The need to hedge, to build safety nets, and to preserve options only exists *because* of variance. The EV-optimal algorithm is smart specifically because it mathematically quantifies its own likelihood of failure.

---

## Part IV: Risk and the  Frontier

Expected Value assumes you are playing an infinite number of games and only care about the long-term average. But Yatzy is often played in tournaments. If you are trailing by 40 points entering the final three turns, playing for the highest "average" score is mathematically incorrect. Scoring 250 is exactly as useless as scoring 0 if you need 260 to win. You need a risk-seeking strategy.

### Beyond Expected Value: CARA Utility

To model risk-sensitive play without destroying our solver, we must optimize for *Utility*, not points. We pass the final score  through an Exponential Utility Function: .

* If , the algorithm is **Risk-Seeking** (it heavily weights jackpots).
* If , the algorithm is **Risk-Averse** (it is terrified of zeroes).

Why use  specifically? Because it possesses **Constant Absolute Risk Aversion (CARA)**. Exponents factor addition into multiplication:
$$ e^{\theta(X_{\text{past}} + X_{\text{future}})} = e^{\theta X_{\text{past}}} \times e^{\theta X_{\text{future}}} $$

Because your past banked score () acts as a strictly positive constant multiplier, it factors completely out of the  decision. **The AI's risk premium is independent of its accumulated wealth.**
This is a computational miracle. It means the solver does *not* need to track your total accumulated score to know how to value the future. We can map the entire frontier of human risk using the exact same compressed 536,000 states, avoiding an exponential explosion in RAM.

### The  Sweep and the Two-Branch Frontier

By running the DP solver 37 times, sweeping  from  to , we map the complete behavioral frontier of Yatzy. Plotting these strategies in Mean-Variance space reveals a stunning two-branch structure:

1. **The Risk-Averse Branch ():** The algorithm panics at zeroes. It aggressively cashes in points to establish a floor. The mean score collapses rapidly, but the variance drops very slowly (you cannot easily remove variance from dice). Hedging in Yatzy is incredibly expensive.
2. **The Risk-Seeking Branch ():** The algorithm hunts Yahtzees and premium bonuses. The variance explodes linearly, buying massive right-tail upside for a relatively small cost to the mean.

**The Surprise:** Near the origin (), we observe a geometric anomaly. In a strict mean-variance sense, the EV-optimal policy is actually slightly inefficient. Because the game's outcomes are heavily right-skewed by massive bonuses, pushing  slightly positive actually increases the *median* score and the frequency of tournament-winning blowouts at an almost imperceptible cost to the overall mathematical mean.

> **Visual Prompt 4: The Two-Branch Mean-Variance Scatter Plot**
> * **Chart Type:** Scatter plot forming a sideways "V" or horseshoe shape (Pareto frontier).
> * **Axes:** X-axis = Standard Deviation (), Y-axis = Mean Score ().
> * **Data Shown:** 37 dots representing the DP policies, color-coded by  (Blue for , Grey for , Red for ).
> * **Key Annotations:** The peak EV (248.4) is marked at . The red points push out to the right (higher variance) before dipping down. The blue points drop sharply down and to the left.
> * **Insight:** Proves that risk in Yatzy is asymmetric. Near the peak, buying upside variance (a "Hail Mary") is remarkably cheap, but buying safety is incredibly expensive.
>
>

### When  Matters

Risk manipulation alters gameplay dynamically across four regimes of the Upper Bonus: *Secured, Comfortable, Marginal,* and *Unreachable*.

If you roll `[5, 5, 5, 2, 3]` early in the game in the **Marginal** regime:

* The **Risk-Averse** bot scores it in *Fives*. It wants guaranteed, mathematical progress toward the 63-point upper bonus.
* The **Risk-Seeking** bot scores it in *Three-of-a-Kind*. It intentionally leaves *Fives* open, hoping to roll four or five 5s later to maximize its upside ceiling, happily accepting the risk of scoring zero in *Fives* later.

---

## Part V: Boundaries of Exact Solution

If we have solved Yatzy for EV and CARA utility, have we *solved* it? Is CARA enough?

No. CARA models a *constant* appetite for risk. But in a real tournament, if you are down by 10 points on Turn 14, your appetite for risk should be incredibly dynamic. If you suddenly roll a Yatzy and take the lead, you must instantly switch to playing safely.

You need a **Threshold Policy**: maximize the probability of scoring exactly .
Threshold policies require *state-dependent risk*. Because CARA has a constant risk appetite, it cannot dynamically shift gears mid-game. To solve exact threshold policies, we must use an -reparameterization technique to track the exact running score, ballooning the state space by a factor of ~375. CARA represents the absolute boundary of what can be exactly solved without the memory footprint exploding into billions of states.

### Why Reinforcement Learning Struggles

If the expanded state space is too big for dynamic programming, standard practice is to use Deep Reinforcement Learning (RL). Modern RL conquered Backgammon, Chess, Go, and Poker. Yet, despite massive compute, the best Neural Networks stall at ~236 points, roughly 5% below the exact DP baseline in Yatzy.

Why does RL fail at dice? Three interacting barriers:

1. **The Discontinuous Value Cliff:** At 62 upper points, you get nothing. At 63, you get +50. Neural networks are continuous function approximators; they "blur" the cliff, physically struggling to model an infinitely steep, discontinuous mathematical jump without massive parameter bloat.
2. **Devastating Signal-to-Noise Ratio (Credit Assignment):** Yatzy has a standard deviation of 38.5. An agent can make a mathematically brilliant sacrifice on Turn 3 that gains +0.5 EV, but then roll garbage for the next 12 turns. The true gradient of that brilliant decision is buried under a mountain of stochastic noise, shattering the network's ability to assign credit to early-game moves.
3. **No Adversarial Curriculum:** RL thrives in zero-sum games (like Poker) via self-play. As the agent improves, the opponent improves, creating a perfectly scaling curriculum of difficulty. Yatzy is a solitaire optimization problem. The agent is fighting the raw, static math of the dice. If it learns a "safe but mediocre" policy, there is no adversary to punish its specific blind spots, so it stagnates in local optima.

> **Visual Prompt 5: The RL Gap Number Line**
> * **Chart Type:** Horizontal Number Line / Ruler.
> * **Axes:** Mean Score from 100 to 260.
> * **Data Shown:** Markers placing the agents: Max-Policy Simulation (~118), Random Play (~140), Average Human (~215), Best RL Agent (~236), Exact DP Optimum (248.4).
> * **Key Annotations:** A bracket highlighting the ~12-point "RL Gap" between neural approximation and absolute mathematical truth. Label it "The Signal-to-Noise Chasm."
> * **Insight:** Visually scales the "last mile" problem in AI. Neural networks easily learn the heuristics that beat humans, but fail to close the final 5% gap to mathematical perfection due to structural noise.
>
>

### The Complexity of Optimal Play

When we analyze the exact DP table, we discover why the neural network plateaus: the "mostly simple with a complex tail" gap distribution.

For 80% of states, the optimal choice dominates the second-best choice by several points. These are the "obvious" moves. But the remaining 20% form a thick tail of micro-decisions where the difference between the best and second-best move is less than 0.05 points. These tie-breakers depend on bizarre, non-linear combinatorial interactions between the upper score and the exact remaining categories. Neural networks easily memorize the 80%, but scaling an agent to perfectly resolve the micro-gradient tail requires an exponential explosion in its parameter count (Minimum Description Length).

> **Visual Prompt 6: Gap Distribution Histogram**
> * **Chart Type:** Log-Linear Histogram.
> * **Axes:** X-axis = EV difference between Best and Second-Best action (points). Y-axis = Log Frequency of states.
> * **Data Shown:** A massive spike near zero, trailing off to the right.
> * **Key Annotations:** Label the regions "Obvious Decisions" (large EV gap) and "The Combinatorial Micro-Optimization Fog" (tiny EV gap).
> * **Insight:** Explains *why* the game is hard to compress: most decisions matter a tiny bit, making heuristic neural approximation hopelessly leaky in the edge cases.
>
>

---

## Part VI: Implications

### What This Means for Human Players

The mathematical optimum operates roughly 20 to 30 points higher than the average casual human player. By comparing human heuristics to the DP tables, we can decompose this gap into a strict skill ladder:

1. **Upper Bonus Awareness (The largest gain):** Humans chronically undervalue the 50-point injection. The solver will aggressively take a zero in *Four-of-a-Kind* to force the 63-point upper threshold.
2. **Flexibility Preservation (The medium gain):** Humans use *Chance* or *Ones* when they get a bad roll early. The solver guards these categories with its life, saving them to absorb catastrophic variance in Turns 13 and 14.
3. **Combinatorial Micro-Hedging (The final 5%):** The realm of the DP solver. Knowing exactly when to hold a generic Pair versus a 3-to-a-Straight based on the exact fractional EV of the remaining scorecard.

> **Visual Prompt 7: The Skill Ladder**
> * **Chart Type:** Waterfall chart / Stepped Bar chart.
> * **Axes:** X-axis = Strategic principles in learning order. Y-axis = Cumulative Expected Mean Score.
> * **Data Shown:** Starting from a baseline (Random play ~140), stepping up as principles are added (Greedy Category Matching +45, Upper Bonus Focus +30, Option Preservation/Safety +20, DP Micro-hedging +13.4) to reach Peak (248.4).
> * **Insight:** Breaks down the impenetrable 248.4 optimum into sequential, learnable human principles, showing exactly how much EV each heuristic unlocks.
>
>

### Conclusion: The Geometry of Uncertainty

Scandinavian Yatzy occupies a perfect mathematical Goldilocks zone. It is simple enough to map entirely into silicon and solve with exact Dynamic Programming, yet combinatorial and stochastic enough to utterly defeat modern neural approximations.

It proves that uncertainty is not an obstacle to strategy; it is the canvas. When you strip away probability (the Max-Policy), strategy disappears into delusion. When you attempt to approximate it (Reinforcement Learning), the variance blinds the algorithm. Only by staring directly into the combinatorial explosion of the future, weighting every possible universe by its exact likelihood, and working backward to the present, can we find the perfect move.
