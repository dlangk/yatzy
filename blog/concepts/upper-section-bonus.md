# Upper Section Bonus

In Scandinavian Yatzy, if your upper section total (Ones through Sixes) reaches **63 or more**, you earn a **50-point bonus**. This is far more valuable than the 35-point bonus in American Yahtzee.

## Why 63?

The threshold of 63 equals exactly three of each die face: 3×1 + 3×2 + 3×3 + 3×4 + 3×5 + 3×6 = 63. So the bonus is achievable if you average three-of-a-kind in each upper category.

## Strategic Impact

The bonus is the single most important strategic consideration:

- Optimal play achieves the bonus **87% of the time**, contributing ~43.5 expected points
- A greedy strategy that ignores bonus tracking achieves it only **1.2% of the time**
- Risk-averse strategies (negative θ) primarily work by *protecting the bonus* — ensuring you don't fall short of 63

The solver tracks upper section progress as a separate state variable (0–63, capped) precisely because the bonus creates a non-linear jump in value at the threshold.
