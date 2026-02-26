:::section{#conclusion}

:::part-title
Conclusion
:::

## What a Dice Game Teaches

A children's dice game contains a 1.43-million-state MDP that admits exact
solution in 1.1 seconds. The optimal policy scores 248.4 with standard
deviation 38.5. The gap between casual play and computed perfection is 82
points --an entire category's worth of score hidden in plain sight.

The optimal policy is compressible but not fully: 100 English rules recover
227.5 EV (59.7% of the gap closed), while 40.3% resists compression into
human-readable form. The residual concentrates near the upper bonus threshold,
where value is discontinuous. This is the boundary between what can be taught
and what must be computed.

Risk-sensitive extensions reveal a rich mean-variance frontier. Sweeping
&theta; from &minus;3 to +3 traces clear phase transitions, and the frontier
is empirically tight --no adaptive policy beats a fixed-&theta; solution,
because the Bellman equation already encodes state-dependent behavior within
a constant risk attitude.

The 82-point gap is a window into bounded rationality. Cognitive profiling
with four parameters (&theta;, &beta;, &gamma;, <var>d</var>) decomposes it
into identifiable dimensions. Human-level play requires 6,000–15,000
decision tree parameters; the oracle needs 2 million entries. That ratio
quantifies the compression biological cognition performs and the price it pays.

Future directions include asymmetric utility functions (prospect theory),
transformer credit assignment for identifying which early decisions most
influence outcomes, and multiplayer Nash equilibria that transform Yatzy from
single-agent optimization into genuine game theory. Each inherits the
infrastructure built here: fast solver, simulation engine, profiling system,
and exact distributions as ground truth.

Scandinavian Yatzy is small enough to solve and large enough to exhibit the
full complexity of sequential decision-making under uncertainty --a
microcosm where optimality, engineering, compression, and human judgment
converge on five dice and a scorecard.

:::depth-2

The key quantitative results span four levels of analysis:

:::equation
Solver: 248.4 EV, 1.1s solve, 37 &theta; sweep<br>
Compression: 227.5 EV from 100 rules (59.7% gap closed)<br>
Semantic rerolls: 184.4 EV, +15.7 over bitmask (168.7)<br>
Human model: 4 params (&theta;, &beta;, &gamma;, d), 30 scenarios, 648-combo grid
:::

The incompressible residual of 20.9 EV establishes a lower bound on the
complexity of Yatzy strategy under sequential decision list representation.
Whether alternative representations (gradient-boosted trees, attention
mechanisms, or hybrid rule-plus-lookup architectures) can reduce this residual
remains an open question. The scaling law --marginal value peaking at
0.74 EV/rule in the 26–40 band, then decaying to 0.15 EV/rule by
rule 100 --suggests that the greedy covering algorithm has largely
exhausted the expressible structure.

:::

:::depth-3

Reproducing the full pipeline from source:

```bash
# Complete pipeline: solve, simulate, analyze, profile
just setup              # Build solver + install analytics
just precompute         # theta=0 strategy table (1.1s)
just sweep              # All 37 theta values (resumable)
just simulate           # 1M Monte Carlo games
just pipeline           # compute + plot + categories + efficiency
just density            # Exact forward-DP PMFs
just profile-generate   # 30 quiz scenarios + Q-grids
just profile-validate   # Monte Carlo parameter recovery
```

:::

:::
