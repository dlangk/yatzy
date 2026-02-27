# Sequential Decision List

A **sequential decision list** is an ordered sequence of if-then rules evaluated from top to bottom. The first rule whose condition matches determines the action. It is one of the simplest interpretable classifiers.

## Structure

```
Rule 1:  IF has_yatzy AND yatzy_open         THEN score Yatzy
Rule 2:  IF has_large_straight AND ls_open    THEN score Large Straight
Rule 3:  IF upper_gap <= 3 AND sixes >= 15    THEN score Sixes
Rule 4:  IF has_full_house AND fh_open        THEN score Full House
...
Rule N:  ELSE score lowest-value open category
```

The order matters critically. Rule 1 is checked first; if it matches, rules 2 through N are never evaluated. This creates an implicit priority system where earlier rules override later ones.

## Why Order Matters

Consider two rules:
- "If you have 4 sixes, score Sixes"
- "If you have 4 of a kind, score Four of a Kind"

When you have four sixes, *both* rules match. The ordering determines which one fires. In this case, the correct priority depends on upper-section progress: if you need sixes for the bonus, the first rule should be higher; if the bonus is already secured, Four of a Kind is worth more.

Finding the optimal ordering is itself a combinatorial optimization problem. With 100 candidate rules, there are 100! possible orderings -- far too many to enumerate. In practice, rules are ordered by a combination of domain knowledge and greedy search.

## Performance in Yatzy

A decision list for **category selection** (which category to score given final dice) achieves:

- **100 rules**: mean score of ~227.5 (vs. 248 optimal)
- **200 rules**: marginal improvement, with diminishing returns

The ~20-point gap from optimal comes primarily from two sources:

1. The inability to condition on upper-section progress with sufficient granularity. The bonus threshold at 63 creates a complex dependency that requires many specialized rules to approximate.
2. Interactions between open categories. With 10 categories remaining, the best action depends on which *specific* categories are open, creating 2^10 = 1,024 contexts that a linear rule list struggles to cover.

## Strengths

Decision lists are fully interpretable -- a human can read and follow them during a real game. They require no matrix operations, no floating-point arithmetic beyond comparisons, and no precomputed tables. A player with a printed decision list can play at ~92% of optimal efficiency.

They are also easy to explain: "I scored Sixes because it was the first matching rule." This transparency is valuable for teaching Yatzy strategy.

## Limitations

The main limitation is expressiveness. Each rule is a conjunction of simple conditions (AND logic), and the sequential evaluation means interactions between features are hard to capture. A rule cannot easily express "score Fours if upper_gap <= 7 AND no threes-of-a-kind are available AND the pair bonus opportunity is weak."

Decision trees overcome this by branching on features hierarchically, achieving better accuracy at the cost of less linear readability. A depth-15 decision tree (81K parameters) scores 239 versus the decision list's 227.5, a meaningful improvement from the richer representational capacity.

## Learning Decision Lists

Constructing an optimal decision list is NP-hard in general, but practical heuristics work well:

1. **Greedy construction**: at each step, find the rule that correctly classifies the most remaining examples, add it to the list, and remove the covered examples. Repeat until all examples are covered or a size limit is reached.
2. **Iterative refinement**: start with a greedy list, then swap rule orderings and evaluate the resulting mean score via simulation. Keep improvements.
3. **Beam search**: maintain several candidate orderings in parallel, expanding each with different next-rule choices.

The greedy approach produces reasonable lists quickly. Iterative refinement typically improves the mean score by 3--5 points over the greedy baseline, with diminishing returns after ~100 swap iterations.

## Historical Connections

Decision lists were studied by Ronald Rivest in 1987 as a learning model intermediate between single rules and full decision trees. They correspond to the class of "disjunctions of conjunctions" in Boolean logic and have well-understood PAC-learning guarantees. In the Yatzy context, they serve as the simplest structured policy representation above raw heuristics.
