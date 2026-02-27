# Resource Rationality

**Resource rationality** is the idea that human decision-making is optimal *given cognitive constraints*. Rather than being irrational, people make the best decisions they can with limited memory, attention, and computation time.

## Herbert Simon's Insight

Herbert Simon introduced **bounded rationality** in the 1950s: real decision-makers cannot evaluate all options and must use simplified strategies. He called this **satisficing** -- choosing an option that is good enough rather than searching for the best.

Resource rationality, developed more formally by researchers like Tom Griffiths and others, refines this by asking not just "what shortcuts do people use?" but "are those shortcuts optimal given the cost of thinking?" A chess grandmaster who spends 3 minutes on a move is not irrational for failing to find the engine's best line. They are making an excellent decision given that deeper search costs time on the clock.

## The (theta, beta, gamma, d) Model

In Yatzy, four parameters capture different types of cognitive constraint:

- **theta (risk preference)**: how the player weighs upside vs. downside. Not a constraint per se, but a preference that shapes which outcomes the player cares about.
- **beta (precision)**: how noisy the player's action selection is. High beta means near-deterministic choice of the best-evaluated option; low beta means frequent "mistakes" where suboptimal actions are chosen. This captures attention and evaluation noise.
- **gamma (myopia)**: how far ahead the player looks. gamma = 1 means full consideration of all future consequences; gamma = 0 means the player ignores all future states and plays greedily for immediate score. Values in between geometrically discount future rounds.
- **d (depth)**: how many reroll steps the player actually evaluates. d = 0 means the player only considers which category to score, not which dice to keep. d = 1 means one level of keep evaluation. d = 2 means full evaluation of both rerolls.

## Cognitive Cost Interpretation

Each parameter maps to a specific cognitive limitation:

- Low beta: limited working memory or attention -- the player cannot reliably identify the best action among several close alternatives
- Low gamma: inability or unwillingness to plan multiple turns ahead -- "what should I do in round 12 given what I'll need in round 15?" is too complex
- Low d: unwillingness to enumerate dice-keeping options -- evaluating 462 keep possibilities per roll is beyond human capacity

A resource-rational agent with gamma = 0.7 is not "bad at Yatzy." They are playing optimally for someone who finds multi-turn planning cognitively expensive and who rationally chooses to allocate limited mental effort elsewhere.

## Expected Performance

Each parameter combination produces a different expected score. The solver can simulate any (theta, beta, gamma, d) agent by modifying how the value function is computed:

- theta changes the objective (EV vs. certainty equivalent)
- gamma discounts future state values geometrically
- d truncates the reroll evaluation depth
- beta adds softmax noise to action selection

The full 4D parameter grid (6 theta x 6 gamma x 3 d x several beta) produces 648 distinct agent types. Simulating 10,000 games per agent gives a complete performance map, showing how score degrades as each cognitive parameter deviates from optimal.

## Profiling

By observing a player's decisions across 30 quiz scenarios and fitting the (theta, beta, gamma, d) parameters via maximum likelihood, we can construct a **cognitive profile** that explains their play style.

This goes beyond "good or bad" to identify *which specific cognitive simplifications* the player is making. Two players with the same mean score might have very different profiles: one might be precise but myopic (high beta, low gamma), while another is far-sighted but noisy (low beta, high gamma). The profiles suggest different improvement strategies: the first player should learn to think ahead; the second should slow down and evaluate more carefully.

## Historical Context

The resource rationality framework connects Yatzy analysis to a long tradition in cognitive science. Simon's original work on bounded rationality won the Nobel Prize in Economics (1978). The computational formalization by Griffiths, Lieder, and others in the 2010s provided the mathematical tools (rational metareasoning, cost-benefit analysis of computation) that make the (theta, beta, gamma, d) model principled rather than ad hoc.

The key insight is that "irrational" behavior often reflects rational adaptation to cognitive limits. A player who ignores future consequences (low gamma) is not failing -- they are allocating their finite attention to the most immediately relevant features of the decision.

## Practical Applications

The resource-rationality framework has practical implications beyond academic interest:

- **Targeted coaching**: knowing that a player has low gamma (myopia) suggests teaching them to track upper-section progress, which is the single highest-value planning skill
- **Difficulty calibration**: quiz scenarios can be selected to target specific cognitive parameters, creating personalized training sequences
- **AI design**: artificial agents with bounded computation budgets can use the (theta, beta, gamma, d) framework to choose which simplifications cost the least EV per unit of computation saved
- **Game design**: understanding how cognitive constraints affect play quality can inform rule modifications that make games more or less forgiving of bounded rationality
