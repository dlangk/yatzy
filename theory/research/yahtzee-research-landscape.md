# Solving Yahtzee: from backward induction to reinforcement learning

**Solitaire Yahtzee has been exactly solved since 1999, yielding an optimal expected score of 254.59 points** — a number independently confirmed by at least three research groups and ultimately pinned down to an exact rational fraction with a 139-digit numerator by Liese and Kelly in 2017. The solution rests on modeling Yahtzee as a Markov Decision Process and sweeping backward through roughly half a billion game states. Yet despite being "solved," Yahtzee continues to generate research interest: tournament-optimal strategies diverge sharply from score-maximizing ones, multiplayer Yahtzee remains computationally intractable, and the best reinforcement learning agents still fall **5% short** of the dynamic programming optimum — stuck on a credit-assignment problem that illuminates fundamental RL challenges.

---

## The game as a Markov Decision Process

Tom Verhoeff of Eindhoven University of Technology was the first to compute the optimal solitaire Yahtzee strategy, beginning in **1999**. James Glenn of Loyola University Maryland solved the problem independently around the same time, and Phil Woodward published the result in the statistics journal *Chance* in 2003. All three arrived at the same answer. The key insight shared across all approaches is modeling Yahtzee as a finite-horizon MDP with a bipartite, acyclic state graph.

Verhoeff's formulation splits the state space into **Roll states** (where dice are about to be thrown) and **Choice states** (where the player selects which dice to keep or which category to score). These alternate in a directed acyclic graph. Each 13-category turn passes through six layers: roll all five dice (252 multiset outcomes), choose keepers (462 possible subsets), reroll, choose keepers again, reroll a final time, then select a scoring category. The backward induction recurrence is clean: at Roll states, the optimal expected additional score **Ê** equals the probability-weighted average over outcomes; at Choice states, it equals the maximum over available actions of the immediate reward plus the successor's Ê value. Terminal states (all categories filled) have Ê = 0.

The computational trick that makes this tractable is a **two-level dynamic programming** architecture. Between turns, the game state is characterized by three components: the subset of unfilled categories (**2^13 = 8,192** possibilities), progress toward the upper-section bonus (roughly **64** values tracking points still needed to reach the 63-point threshold), and whether a Yahtzee has already been scored (**2** values for bonus eligibility). This yields **786,432** total between-turn states, of which only **536,448** are reachable from the initial position. Each between-turn state requires solving an internal within-turn MDP spanning **1,681 states** — but these are recomputed on the fly rather than stored, keeping the main DP table at a manageable **~6 MB**. Glenn describes the full position graph as containing roughly **half a billion nodes**.

The symmetry that makes the problem tractable is treating dice as **multisets** rather than ordered tuples. Five ordered dice produce 6^5 = 7,776 outcomes; as unordered multisets, this collapses to just **252**. Similarly, the set of possible "keeper" selections (choosing 0–5 dice from faces {1,...,6}) reduces to **462** multisets. Without this reduction, the state space would be orders of magnitude larger. Glenn's 2006 technical report (*An Optimal Strategy for Yahtzee*, CS-TR-0002, Loyola College) and his 2007 IEEE CIG paper (*Computer Strategies for Solitaire Yahtzee*) detail further pruning of unreachable states and methods for computing not just expected values but full score distributions and variances.

For context on the raw combinatorial scale: the number of possible complete Yahtzee games is approximately **1.7 × 10^170**, and the number of deterministic strategies is roughly 10^(10^100). Dynamic programming collapses this to a problem solvable in minutes on modern hardware — Jakub Pawlewicz of the University of Warsaw achieved a **~30 second** computation time with aggressive optimizations.

---

## The exact optimal expected score and what it means

The optimal expected score for solitaire Yahtzee under the full American rules (including the Extra Yahtzee Bonus and Joker rules) is **254.5896 ± 59.61** (mean ± standard deviation). Without the Extra Yahtzee Bonus and Joker rules, it drops to approximately **245.87 ± 39.82**. For comparison, random play scores about **46 points**, a simple greedy strategy reaches **~219**, and expert human players typically score in the **220–240** range.

The exact rational value was computed on 12 June 2017 by **Jeffrey Liese and Katy Kelly** using Mathematica 11, running for approximately five days on a desktop PC. The numerator of the fraction spans 139 digits, the denominator 137 digits — a testament to the combinatorial complexity lurking beneath a seemingly simple parlor game.

The score distribution under optimal play is notably right-skewed. The **median is 248** — six points below the mean, pulled up by rare high-scoring games involving multiple Yahtzees. Key percentiles: the 5th percentile sits at **180**, the 25th at **218**, the 75th at **273**, and the 99th at **474**. The minimum possible score under optimal play (worst-case dice luck) is just **12 points**; under a minimax/paranoid strategy designed to guarantee the best worst case, the floor rises to **18**.

Per-category statistics reveal that optimal play **fails to score a Yahtzee in about 66% of games**, obtains one Yahtzee roughly 34% of the time, and earns the Extra Yahtzee Bonus (requiring two or more five-of-a-kinds) in only about **8%** of games. The upper-section bonus of 35 points is missed in roughly **32%** of games even under perfect play. Verhoeff's website at TU Eindhoven provides full per-category expected values, variances, and even the complete covariance matrix between scoring categories.

---

## Beyond expected score: risk, tournaments, and winning

Maximizing expected score is the natural objective for solitaire Yahtzee, but it is not always the right one. Several researchers have explored alternative formulations that reveal strikingly different optimal strategies.

**Cremers (2002)** developed the earliest alternative: a "MaxProb" strategy that maximizes the probability of exceeding a fixed target score, computed as part of his master's thesis at Eindhoven under Verhoeff's supervision. For each game state and threshold *s*, the approach propagates Pr(total ≥ s) backward through the game tree, producing strategies that sacrifice expected score for reduced downside variance when the target is modest, or increased upside variance when the target is ambitious.

**Pawlewicz (2009–2011)** generalized this by computing full probability distributions over final scores at every game state. These distributions became building blocks for multiplayer analysis. In interactive multiplayer Yahtzee, the optimal strategy adapts to opponents' visible scores: play aggressively when trailing, conservatively when leading. Pawlewicz showed this heuristic approach is experimentally "indistinguishable from optimal" for practical purposes.

The most dramatic divergence from expected-score maximization appears in **Theodore Perkins' 2019 AAMAS paper** on blind Yahtzee tournaments (the format used in the popular Yahtzee with Buddies app). In a tournament of *N* independent players where only the highest score wins, the optimal strategy becomes progressively riskier as *N* grows. Against 1 opponent, the tournament-optimal strategy scores **241.6 ± 43** (sacrificing 13 expected points for a 2.67% higher win probability). Against 10,000 opponents, it scores just **198.3 ± 61** — abandoning 56 expected points — yet wins **70% more often** than the score-maximizing strategy. These strategies essentially gamble on multiple Yahtzees, accepting near-certain mediocre scores for a small chance at the 400+ point games needed to top a large field. Perkins found that winning scores in real Yahtzee with Buddies tournaments average ~349 for 15 players and ~563 for 10,000+ players, far exceeding the 254.59 mean of optimal solitaire play.

No study has explicitly constructed a **mean–standard deviation Pareto frontier** for Yahtzee strategies, though Perkins' results implicitly trace one: as strategies shift from score-maximizing to tournament-optimal, the (mean, σ) pairs move from (254.6, 59.6) through (241.6, 43) down to (198.3, 61), mapping out a frontier of risk-return tradeoffs. The full covariance structure published by Verhoeff would enable such analysis, representing an open research opportunity.

---

## Reinforcement learning: 95% of the way there

Multiple research groups have applied RL to Yahtzee, and the results converge on a consistent finding: the best agents reach roughly **95% of optimal play** but cannot close the remaining gap.

The most rigorous academic study is **Pape (2025)**, published as arXiv:2601.00007, which systematically compared REINFORCE, Advantage Actor-Critic (A2C), and Proximal Policy Optimization (PPO) using multi-headed networks with shared trunks trained on NVIDIA A40 GPUs. A2C proved the clear winner, achieving a median score of **241.78 points** — within 5.0% of the 254.59 optimum. REINFORCE and PPO proved significantly more sensitive to hyperparameters and failed to match A2C's performance under equivalent training budgets.

**Dion Häfner's Yahtzotron (2021)** independently demonstrated A2C's effectiveness for Yahtzee using JAX and DeepMind's Haiku library. With a pre-training curriculum (greedy lookup table → advantage lookup table → full A2C), the agent reached ~240–242 average points in roughly **two hours on a single CPU** — remarkably efficient. Häfner also tried genetic optimization as a baseline, which plateaued at a meager ~130 points, illustrating why gradient-based methods dominate.

DQN-based approaches show more variable results. **Dutschke's** MLP-based Q-learning agent achieved **241.6 average** after 8,000 training games, while **Yuan's (2023)** DQN for two-player Yahtzee managed only **159 points** with a 6% win rate against the optimal agent — highlighting that the two-player setting dramatically compounds difficulty. **Kang and Schroeder (2018)** at Stanford tried hierarchical RL with MAXQ decomposition, achieving 99% win rate against random opponents but only 68% against greedy agents.

The persistent ~5% gap between RL and exact DP traces to three specific challenges:

- **The upper-section bonus problem.** All RL models struggle to learn the strategy of accumulating upper-section points to earn the 35-point bonus. This requires coordinating decisions across multiple turns — a classic long-horizon credit-assignment problem. Pape (2025) found that RL agents systematically overindex on four-of-a-kind plays at the expense of strategically building upper-section totals.

- **Heterogeneous action spaces.** Each turn involves two qualitatively different decision types — dice-keeping (32 subsets) and category selection (up to 13 choices) — with the action space shrinking as categories are consumed. Multi-headed networks and action masking (setting invalid action logits to −∞) partially address this but add architectural complexity.

- **Stochastic credit assignment.** Dice randomness injects high variance into outcomes, making it difficult to attribute results to decisions. In multiplayer settings, this problem is acute: as Kang and Schroeder noted, "wins are attributed mostly to the opponent's bad luck rather than any good decision on the agent's part."

No **AlphaZero/MuZero-style** approach (MCTS + neural networks) has been published for Yahtzee. Häfner explicitly considered and rejected MCTS because the heavy stochastic element (chance nodes for every dice roll) makes tree search expensive, and the solitaire version is already exactly solved, reducing the motivation. The absence represents both a practical gap and a reflection that Yahtzee's single-player, stochastic nature sits outside AlphaZero's two-player, deterministic sweet spot.

---

## American Yahtzee versus Scandinavian Yatzy

The two major variants differ substantially in rules, scoring, and computational complexity. Yatzy uses **15 categories** (versus Yahtzee's 13), adding One Pair and Two Pairs. Three-of-a-Kind and Four-of-a-Kind in Yatzy score only the matching dice rather than all five. Straights are fixed sequences (1-2-3-4-5 = 15 points; 2-3-4-5-6 = 20 points) rather than Yahtzee's more flexible and higher-valued definitions (Small Straight = any 4-sequence for 30 points; Large Straight = any 5-sequence for 40 points). Crucially, **Yatzy has no Yahtzee Bonus or Joker rule**, eliminating the state-tracking complexity those rules create.

The additional two categories expand the category-subset state space from **2^13 = 8,192** to **2^15 = 32,768** — a fourfold increase. However, the absence of the Yahtzee Bonus/Joker system removes a dimension from the between-turn state, partially offsetting this expansion. **Larsson and Sjöberg (KTH, 2012)** were the first to compute the optimal Yatzy strategy, arriving at an expected score of **248.63 points** — about 6 points below Yahtzee's 254.59, primarily because Yatzy lacks bonus Yahtzee points (worth ~9.58 on average) and has lower straight values. The maximum possible Yatzy score is **374** versus Yahtzee's theoretical maximum of **1,575** (with 13 consecutive Yahtzees).

Other notable variants include **Yacht** (the 1938 predecessor with 12 categories and no upper-section bonus), **Maxi Yatzy** (a Swedish variant with 6 dice and 20 categories), **Forced Yatzy** (categories must be scored in order), and **Kniffel** (the German variant, essentially identical to American Yahtzee).

---

## Open-source solvers and practical tools

The research community has produced several high-quality open-source implementations:

- **Verhoeff and Scheffers' Optimal Solitaire Yahtzee Player** (TU Eindhoven, 1999–present) remains the gold standard — an online tool that plays Yahtzee optimally and provides move-by-move analysis. The precomputed strategy table is ~8 MB and takes ~10 minutes to generate.
- **timpalpant/yahtzee** (Go) is a full optimal player with a web server interface, deployed on a Raspberry Pi. Expected-value tables consume 5.7 MB; full score distributions require 1.8 GB.
- **greeness/yahtzee-optimal-strategy** (Python) directly implements Glenn's 2006 paper for the version without Yahtzee bonus.
- **dpmerrell/yahtzee** (Python) provides a mathematically guaranteed optimal strategy via MDP modeling (without upper-section bonus tracking, yielding 229.6 average).
- **Felix Holderied's yahtzee** (C, web demo at yahtzee.holderied.de) implements three strategy modes — maximize expected value, reach a threshold, and beat an opponent — with precomputation requiring 5–20 hours and 1.6 GB.
- **dionhaefner/yahtzotron** (Python/JAX) is the leading RL implementation, achieving ~240 average via A2C.
- **philvasseur/Yahtzee-DQN-Thesis** accompanies the Vasseur and Glenn (2019) paper on deep Q-learning strategy ladders.

---

## Conclusion

Yahtzee's deceptive simplicity conceals a rich computational problem. The solitaire game is exactly solved — the **254.5896** expected score under optimal play is now known to 139-digit precision — yet the research frontier remains active in three directions. First, **multiplayer Yahtzee** is computationally intractable: even two-player optimal play pushes the limits of today's largest computing clusters, and Pawlewicz proved that three or more players are beyond reach. Second, **risk-sensitive and tournament-optimal** strategies represent a largely unexplored design space; Perkins' 2019 work showed that the optimal tournament strategy can sacrifice over 50 expected points to dramatically increase win probability, and no one has yet mapped the full Pareto frontier of risk-return tradeoffs. Third, **reinforcement learning** has converged at ~95% of optimal — close enough to be practically useful but far enough to illuminate fundamental challenges in long-horizon credit assignment and stochastic environments. The stubborn 5% gap, driven primarily by the difficulty of learning the upper-section bonus strategy, serves as a compact benchmark problem for advancing RL methods on structured combinatorial tasks with delayed rewards.