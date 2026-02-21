# Scandinavian Yatzy Strategy Guide: Dice → Decision

## 1. **Slam the Naturals** (Rules 1, 2, 5, 7, 10, 16, 17, 18, 19, 20, 21, 25, 27, 31)

**When:** You roll a perfect category match (Large Straight, Yatzy, etc.)

**What to do:** Immediately lock in the natural:
- Large Straight (2-3-4-5-6) → Take it for 20 points
- Five of a kind → Take Yatzy for 50 points (with tactical exceptions below)

**Why:** These are maximum-value hands with massive regret if missed. Large Straight prevents 17.8 points of regret vs. dumping in Chance. Yatzy prevents 20-30+ points of regret depending on alternatives.

**Tactical Yatzy exceptions:**
- If you have 5 Ones and Yatzy is open → Take Yatzy (28.9 point advantage)
- If you're ahead on bonus pace (≥1.24) with 5 Fives → Consider Yatzy over Fives
- If Twos/Fives are already burned → Always take Yatzy with those dice
- Early game (turn ≤ 9) with 5 Fives → Lean toward Yatzy (13.5 point advantage)

*Coverage: 19,000+ situations | Mean regret: 0.02*

## 2. **Four-of-a-Kind Upper Pivot** (Rules 4, 6, 8, 22, 23, 32, 33)

**When:** You roll exactly 4 matching dice in the upper section

**What to do:** Take the upper category instead of Four of a Kind when:
- 4 Ones → Take Ones (prevents 8.0 regret)
- 4 Twos (after turn 2) → Take Twos (prevents 8.7 regret)
- 4 Threes (early game) → Take Threes (prevents 8.9 regret)
- 4 Fours (when alternatives ≥ 3) → Take Fours (prevents 5.8 regret)
- 4 Fives (behind bonus pace or early) → Take Fives (prevents 5.1-5.8 regret)

**Why:** The solver recognizes that upper section progress compounds through the bonus. Taking 4 matching upper dice early preserves Four of a Kind as a flexible late-game sink while advancing toward the 63-point threshold.

*Coverage: 27,000+ situations | Mean regret: 0.008*

## 3. **Triple Upper Protection** (Rules 9, 11, 13, 34, 36, 40, 43, 45, 46, 48)

**When:** You roll exactly 3 matching upper dice

**What to do:** Prioritize the upper category when:
- 3 Threes with any 2 → Take Threes (prevents 7.8 regret)
- 3 Threes with Full House burned → Take Threes (prevents 7.7 regret)
- 3 Fours (upper score ≤ 42) → Take Fours (prevents 9.8 regret)
- 3 Fives with 1 Two → Take Fives (prevents 2.5 regret)
- 3 Sixes → Usually take Sixes (prevents 5.5 regret)

**Why:** Three matching upper dice represent critical bonus progress. The solver routes these to upper categories to maintain bonus trajectory, using lower categories as variance sinks later.

*Coverage: 85,000+ situations | Mean regret: 0.15*

## 4. **Small Straight Timing** (Rules 14, 24, 26, 28)

**When:** You roll 1-2-3-4-5 with specific game states

**What to do:** Take Small Straight when:
- Upper categories mostly filled (≤3 left) → Prevents 14.3 regret
- Twos already burned → Prevents 13.9 regret
- Many dump categories remain (≥4 zeros) → Prevents 15.8 regret
- Best alternative ≤ 12 points → Prevents 7.3 regret

**Why:** Small Straight's fixed 15 points becomes more valuable as flexibility decreases. The solver times it to avoid burning it cheaply while preserving higher-variance categories.

*Coverage: 9,000+ situations | Mean regret: 0.003*

## 5. **Full House Threshold** (Rules 35, 42)

**When:** You roll three of a kind + a pair

**What to do:**
- Upper score ≤ 42 → Take Full House (prevents 7.1 regret)
- Ahead on bonus with 3 Fives → Consider Full House despite -8.8 regret

**Why:** Full House offers high guaranteed points (up to 28). The solver uses it as a commitment device when behind on bonus, but may sacrifice it when ahead.

*Coverage: 17,000+ situations | Mean regret: 0.25*

## 6. **Late Game Cleanup** (Rules 3, 41, 47, 49, 50)

**When:** Categories are running out or specific options are burned

**What to do:**
- Final turn → Take Four of a Kind if available
- Threes burned + 4 Threes → Route to Four of a Kind
- High sum (≥26) with pairs → Take Two Pairs
- Two Pairs burned → Take any two pair as One Pair
- Only 2 Sixes → Take One Pair if Two Pairs gone

**Why:** The solver prevents stranded points by routing hands to their best remaining home. This "garbage collection" phase minimizes waste.

*Coverage: 53,000+ situations | Mean regret: 0.25*

## 7. **Default Play**

**When:** No specific rule applies

**What to do:** Take the highest-scoring available category, considering:
1. Upper bonus progress (current pace vs. 63 target)
2. Category flexibility remaining
3. Probability of improving the category later

**Why:** The uncovered situations have lower regret differentials, making simple score maximization nearly optimal.

---

*Total coverage: 280,000+ situations analyzed | Overall mean regret: 0.12 | Maximum prevention: 31.4 points (Yatzy decisions)*