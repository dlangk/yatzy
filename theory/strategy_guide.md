# Scandinavian Yatzy Strategy Guide

Based on gap analysis of 100,000 games comparing a pattern-matching heuristic (mean 166) against the EV-optimal solver (mean 248). The 82-point gap breaks down as: category selection (38.4 pts), first reroll (24.0 pts), second reroll (20.2 pts).

## Tier 1: The One Rule That Matters Most

**The upper bonus is worth 50 points and drives nearly half of all EV loss.** You need 63 points across Ones through Sixes (averaging 3-of-each-face). The optimal solver gets the bonus 87% of the time; the heuristic gets it 1.2%. Every reroll and category decision should be filtered through: "does this help or hurt my bonus chances?"

## Tier 2: The Five Biggest Mistakes

### Mistake 1: Scoring Lower-Section Patterns When Upper Categories Need the Dice (~14 pts/game)

The heuristic's #1 category error is taking a lower-section pattern when the optimal play is to bank those dice toward the upper bonus. Examples:

- **Three 6s** → heuristic picks Three of a Kind (18 pts). Optimal picks Sixes (18 pts, same score but advances the bonus).
- **Four 5s** → heuristic picks Four of a Kind (20 pts). Optimal picks Fives (20 pts, same score but advances the bonus).
- **Three 1s** → heuristic picks Three of a Kind (3 pts). Optimal picks Ones (3 pts for bonus progress).

**Rule: When you roll N-of-a-kind and the matching upper category is open, always take the upper category if the score is the same or close.** The upper bonus is a 50-point swing that dwarfs most lower-section advantages.

### Mistake 2: Chasing Yatzy Instead of Banking Upper Scores (~3 pts/game)

The heuristic always chases Yatzy when it sees 3-of-a-kind, keeping just the three and rerolling everything else. The optimal solver often keeps a pair alongside the three (for full house backup) or abandons the Yatzy chase entirely when upper section needs are pressing.

**Rule: Don't chase Yatzy from 3-of-a-kind unless the upper bonus is already secured or nearly there.** The expected Yatzy hit rate from 3-of-a-kind with 2 rerolls is only ~4.6%.

### Mistake 3: Wrong Reroll Patterns — Keeping Too Many or Too Few Dice (~27 pts/game combined)

The two biggest reroll mistakes:

- **Keeping only 1 die when optimal keeps 2**: The heuristic keeps only a highest singleton when it should keep a pair. Example: `[1,5,5,6,6]` — heuristic keeps just the 6 (for Chance/upper); optimal keeps the pair of 6s (for upper bonus + pair/full house potential).

- **Keeping 4 dice when optimal keeps 2**: The heuristic over-commits to a near-straight or two-pair hand when optimal would reroll more aggressively for upper section targets.

**Rule: Pairs are almost always worth keeping.** A pair of high dice (4s, 5s, 6s) gives you a foundation for both upper bonus and lower section patterns.

### Mistake 4: Premature Upper Section Dumps (~12 pts/game)

The heuristic scores upper categories "at par" (e.g., three 4s = 12 for Fours) even when a small straight or other lower-section play would be significantly better. The biggest offenders:

- Taking Fours/Fives/Threes when Small Straight scores 15 (3.2, 3.0, 2.9 pts/game each)
- Taking Sixes/Twos when Small Straight would score (2.2, 1.4 pts/game)

**Rule: Don't auto-bank upper categories. Compare against what lower-section patterns are available.** A small straight (15 pts) beats three 3s (9 pts) or three 4s (12 pts). Only favor the upper category when it's above par AND you need the bonus progress.

### Mistake 5: Not Keeping All Dice When Already Done (~2.8 pts/game)

On the second reroll, the heuristic sometimes continues rerolling when optimal would keep all five dice. This happens when the current hand already scores well and any reroll risks making it worse (e.g., breaking up a scoring combination).

**Rule: On reroll 2, if your current dice score well in an open category, seriously consider keeping all five.**

## Tier 3: Quick Reference

### Upper Bonus Targets

| Category | Target (3x) | Surplus from 4x | Surplus from 5x |
|----------|-------------|-----------------|-----------------|
| Ones     | 3           | +1              | +2              |
| Twos     | 6           | +2              | +4              |
| Threes   | 9           | +3              | +6              |
| Fours    | 12          | +4              | +8              |
| Fives    | 15          | +5              | +10             |
| Sixes    | 18          | +6              | +12             |
| **Total**| **63**      |                 |                 |

You need exactly 63 across all six categories. Surpluses in high categories (4x Sixes = 24, surplus +6) offset shortfalls in low categories (2x Ones = 2, deficit -1).

### Decision Flowchart

1. **Do I have a completed pattern?** (Yatzy, straight, full house) → Score it, unless the same dice give equal or better value in an open upper category.

2. **Can I score at-or-above par in an upper category?** → Strong candidate, but compare against available lower-section alternatives. Prefer upper when bonus is still reachable.

3. **Do I have a high pair or better?** → Keep it. Reroll the rest targeting upper bonus faces or pattern completion.

4. **Nothing special?** → Keep your highest die(s) that match open upper categories. Reroll the rest.

### When to Abandon the Bonus Chase

If you're past turn 10 and more than 15 points below the bonus pace (i.e., you'd need 4x or better in all remaining upper categories), shift to maximizing raw score: take the highest-scoring available category each turn, favor Chance for high-sum hands, and dump low-value categories.

### Category Priority When Dumping

When you must score 0 somewhere, dump in this order (least to most valuable):
1. Ones (max 5, contributes least to bonus)
2. Twos (max 10)
3. Yatzy (only 4.6% hit rate — if you haven't scored it by late game, dump it)
4. Small Straight (15 pts but hard to get late)
5. Large Straight (20 pts but very hard to get late)

### EV Loss Breakdown Summary

| Source | EV Loss/Game | % of Gap |
|--------|-------------|----------|
| Category selection mistakes | 38.4 | 47% |
| First reroll mistakes | 24.0 | 29% |
| Second reroll mistakes | 20.2 | 24% |
| **Total** | **82.6** | **100%** |

The category mistakes alone account for nearly half the gap. A human who fixes only their category selection — always preferring upper categories when scores match, comparing against lower-section alternatives — would recover ~38 points and jump from ~166 to ~204 mean.
