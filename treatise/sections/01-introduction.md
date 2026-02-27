:::section{#introduction}

## Yatzy: The Family Game Balancing Skill and Luck 🎲

I loved playing it as a kid, and I still do. Yatzy, for me, is closely associated with having fun as a family. In particular, I associate it with my wonderful dad. He is the most fun person I know to play Yatzy with! ❤️

Yatzy has some really interesting properties that make it fun to both play, and pick apart. To understand the game, let's start by taking a look at what happens if an optimal Yatzy players sits down and plays thousands of games:

:::html
<div class="chart-container" id="chart-score-spray">
  <div class="chart-controls">
    <button class="chart-btn" id="spray-replay-btn">&#9654; Replay</button>
  </div>
  <div id="chart-score-spray-wrap" style="position: relative;"></div>
  <div class="spray-legend" id="spray-legend"></div>
  <div class="spray-stats" id="spray-stats"></div>
  <div id="spray-scorecard-popup" class="spray-popup hidden"></div>
</div>
:::

As you can see, scores vary wildly even for an optimal player. I think that explains why it's fun for families: Yatzy has room for skill, but... it's mostly luck! 😅 That means that the best player will win slightly more, but anyone can get lucky and beat the strongest player by a wide margin.

Most family games require some mixture of skill and luck. The more skill, the less likely it is that your entire family will enjoy it together. For example, playing chess isn't really fun unless both players are about as good. At the same time, too much luck and you don't feel like you earned your win anymore. There needs to be decisions that matter.

Yatzy, I think, strikes a perfect balance. When you win, you feel skilled. When you loose, you had bad luck.

Many years ago, I got it into my head to figure out: exactly how much skill is involved in Yatzy? Or put differently: how good can you be at Yatzy?

This site is the result of that investigation. Along the way, it also turned out that Yatzy was a good workbench for testing AI technologies. A few of the things that makes Yatzy interesting are:

- **The Bonus Cliff.** The hardest thing about Yatzy is that you get a bonus based on your upper score. This makes reinforcement learning much harder to use successfully.

- **The Significant Inherent Randomness.**  Five dices with six sides that are rolled over and over again means a whole lot of randomness.

- **The Huge State Space.** There is something like ~1.7 × 10^170 possible Yatzy games. That's significantly larger than chess. And the Universe. There are about 10^80 atoms in the universe.

- **It Can Still Be Solved.** Clever mathematicians have figured out how to reduce the state space drastically, and as a result, Yatzy can be solved. I've implemented my own solver based on this math, and used AI to push that solver to extreme performance.

This work is the grand finale of many years of nerding out about Yatzy. I was finally able to publish everything with the help of Agentic Coding.
