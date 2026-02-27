# Threshold Policy

A **threshold policy** is a decision rule of the form "if some quantity exceeds (or falls below) a threshold, take action A; otherwise take action B." These rules are compact, interpretable, and often capture the essence of optimal behavior.

## Structure

A threshold policy replaces a full lookup table with a small set of conditional rules:

- "If upper_gap <= 7, score sixes even if they're mediocre"
- "If you have >= 3 of a face and upper_total < 42, keep them for the upper section"
- "If only one category remains and upper_total >= 63, the bonus is secured -- score greedily"
- "If upper_gap <= 3 and you have two fives, keep them"

Each rule depends on a state variable crossing a critical value. The optimal action changes discontinuously at that point -- one point of upper progress can flip the best category from a lower-section play to an upper-section play.

## Why Thresholds Emerge

The upper bonus creates a sharp non-linearity at upper_total = 63. Near this boundary, the marginal value of one more upper-section point jumps from ~1 point (its face value) to ~51 points (face value plus the 50-point bonus). This cliff in the value function naturally produces threshold behavior: there is a critical gap below which it is worth sacrificing lower-section opportunities to secure the bonus.

The threshold is not fixed at 63. It shifts depending on how many upper categories remain. With 3 upper categories left and a total of 45, the relevant question is: "Can I reach 63 with three more upper scores?" The effective threshold depends on the expected value of the remaining upper categories.

## Endgame Thresholds

Threshold policies are most accurate in the **endgame** (few categories remaining) where the decision space is small and the bonus boundary dominates strategy. With 2--3 categories left, a handful of threshold rules can replicate optimal play almost exactly.

The endgame is where thresholds shine because:

1. Few categories means few possible actions -- threshold rules can cover all of them
2. The bonus boundary creates a clear binary structure (will you make 63 or not?)
3. Category interactions are minimal with few categories remaining

## Early-Game Limitations

In the early game, thresholds are weaker approximations. With 10+ categories open, the interaction between many remaining categories creates a richer decision landscape. The optimal action depends on subtle combinations of state variables (which specific categories are open, the exact upper progress, the dice showing) that resist compression into simple threshold rules.

## Compact Representation

A threshold policy for category selection might need only 50--100 parameters (thresholds and associated actions) compared to the full DP table's 16 MB. The tradeoff is a small EV loss -- typically 2--5 points versus optimal -- concentrated in mid-game situations where the thresholds oversimplify.

This makes threshold policies useful as human-playable heuristics: a player who memorizes a few key thresholds (especially around upper bonus tracking) can capture most of the value of optimal play without any computation aids.

## Relationship to Decision Trees

Threshold policies and decision trees are closely related. A depth-1 decision tree is exactly a single threshold rule. A depth-N decision tree is a hierarchical composition of N threshold rules, where later thresholds can depend on the outcomes of earlier ones.

The key difference is structure: a threshold policy applies rules independently (or in a fixed sequence), while a decision tree applies them conditionally. This gives trees more expressiveness -- they can represent "if upper_gap <= 7 AND fives_count >= 3, then score Fives; otherwise if upper_gap <= 7 AND sixes_count >= 3, score Sixes" as a natural branching structure.

In practice, the most effective human strategies combine both: a short list of threshold rules for common situations, with occasional tree-like branching for the bonus-boundary region where decisions are most consequential.

## Discovering Thresholds

The DP solution implicitly contains all threshold information. By examining where the optimal action changes as a single state variable varies (holding all others fixed), we can extract the threshold values:

1. Fix the scored-categories mask and dice outcome
2. Sweep upper_total from 0 to 63
3. Record where the optimal category switches

This produces a "phase diagram" showing, for each game situation, the critical upper-total values where strategy changes. These phase boundaries are the thresholds. In the endgame, the boundaries are clean vertical lines (sharp thresholds). In the mid-game, they become more complex -- diagonal or curved boundaries that depend on multiple state variables simultaneously.

Extracting and codifying these thresholds is one path to human-playable strategy guides: instead of a 16 MB lookup table, a player memorizes 20--30 critical threshold values.
