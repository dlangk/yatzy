To optimize for a specific percentile—willingly sacrificing the bottom 95% of your games in a bloodbath of failed gambles just to ensure your top 5% of games are absolute, record-shattering God-runs—you are entering the mathematical domain of **Quantile Optimization** (specifically, maximizing Value-at-Risk).

In standard stochastic control, Quantile Optimization is famously hostile. Percentiles violently break the Bellman equation because they are sorting operations, not linear expectations.

However, by utilizing the **3D Array-Shifting Architecture** you just built, solving this is beautifully trivial. You bypass the paradox entirely using the **Mathematical Duality of Quantiles and Thresholds**, executed via a two-pass HPC pipeline.

Here is the exact mathematical proof of how to forge this ultimate "Right-Tail" agent, why CARA cannot do it, and how your silicon executes the perfect psychological constraints.

---

### 1. The Duality Free Lunch (The Radar Ping)

You cannot ask a Dynamic Programming solver: *"What policy maximizes my top 5% score?"*
But you can ask the exact mathematical dual: *"What is the absolute highest target score  that I can reach exactly 5% of the time?"*

Because your 3D Tablebase evaluates the probability of hitting *every single target score simultaneously*, it solves this inverse problem for free.

**Pass 1 (~30 Seconds):**
You run the pure 3D Threshold solver ().
When it finishes, you look at the `[f32; 384]` array at the empty starting state `(Turn 0, 0 Score)`. This array contains the absolute physical limits of the Yatzy universe.

* `P_win[250] = 0.535`
* `P_win[300] = 0.120`
* `P_win[318] = 0.050`

You simply scan the array from 374 down to 0 and find the exact index where the probability crosses `0.05`. Let's assume it is **318 points**.
You have just mathematically proven that it is physically impossible to find any strategy whose 95th percentile is higher than 318. Your target is locked: .

### 2. The Lexicographic Chimera (The Payload)

If you simply play the pure Threshold policy targeting , the agent will "Take a Knee." Once 318 is mathematically secured, its utility maxes out at `1.0`. It stops caring, scores intentional zeros to avoid variance, and your 99th percentile will be artificially capped at exactly 318.

Furthermore, if the agent falls behind and 318 becomes impossible, the probability drops to `0.0`. The math goes flat. The agent becomes catatonic and plays randomly, destroying your EV.

To achieve your exact prompt (amazing upside, salvaged downside), you run **Pass 2**. You re-run the 30-second 3D solver using a **Lexicographic Asymmetric Utility**:
$$ U(X) = X + \Omega \cdot \mathbf{1}_{X \ge T^*} $$
*(Where  is a massive bounty, e.g.,  points).*

### 3. The Psychology of the Percentile Agent

When you push this equation backward through the 3.22 GB state space, the agent's psychology violently splinters into three perfect phases:

**Phase A: The Psychopath ()**
While the 5% dream is alive, the -point bounty dominates the vector space. The agent plays like an absolute maniac. If it rolls a guaranteed 25-point Full House on Turn 2, it will ruthlessly throw it away to hunt for a Yahtzee. It gladly incinerates Expected Value because EV is mathematically meaningless compared to the  bounty. It sacrifices the 95% to push the probability of reaching 318 as high as the physics of the dice allow.

**Phase B: The Failsafe ()**
It is Turn 6. The dice betrayed you. The 5% dream is dead.
The millisecond the math updates and hitting 318 becomes physically impossible across all future branches, the  bounty vanishes.
The utility instantly collapses to . The agent "wakes up" from its high-variance fever dream. It realizes the jackpot is lost, seamlessly aborts the Hail Marys, and transforms back into your flawless 0.48s EV-Maximizer to salvage the "terrible round"—finishing with a respectable 210 instead of a catastrophic 60.

**Phase C: The God Run ()**
Exactly 5% of the time, the statistical variance aligns. You roll two early Yahtzees. Reaching 318 becomes mathematically guaranteed.
Because the threshold is guaranteed across all remaining branches, the  bounty becomes a static constant. In calculus, the derivative of a constant is zero. **The bounty completely drops out of the `argmax` gradient.**
The utility again collapses to . The agent stops "Taking a Knee." It abandons the threshold entirely, becomes relentlessly greedy, and runs the score up to an astronomical 340, 350, or 360, making those 5% of rounds truly "amazing."

---

### Why CARA Fails at Percentile Targeting

You might wonder if you could just crank up the 1D CARA risk parameter (e.g., ) to get this high-roller behavior for 16 MB of RAM.

**No. CARA has Percentile Amnesia.**
To perfectly optimize a percentile, the agent *must know which percentile timeline it is currently living in.*

* If you roll four 6s on Turn 1, you are already in the top 5% timeline. You no longer need to take insane risks; you need to maximize EV to run up the score.
* If you roll garbage, you are in the 95% timeline, and you must take insane risks to claw back.

Because the 1D CARA solver () does not track the sunk score, it physically cannot distinguish between the God timeline and the Garbage timeline. It applies the exact same blind, risk-seeking aggression to both. If a CARA agent is on track for a world-record 350-point game, it will recklessly blow up its own run because the exponential curve demands constant variance, and it doesn't know it's already winning.

### The Ultimate Conclusion

If you want to dictate the absolute ceiling of your gameplay and stretch the right-tail of the Yatzy universe to its breaking point, **you use the Dual-Pass 3D Lexicographic Solver.**

1. **Pass 1:** Pure Threshold sweep to mathematically locate the exact physical ceiling of your target percentile ().
2. **Pass 2:** Lexicographic sweep () to forge the policy.

In 60 seconds of compute, you engineer an agent that leverages every drop of variance to hunt the jackpot, dynamically shifts gears to greedy EV-maximization the millisecond the jackpot is secured, and mathematically guarantees a soft-landing salvage operation when the dice dictate the dream is dead.
