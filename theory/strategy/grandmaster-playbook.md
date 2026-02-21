# Scandinavian Yatzy Strategy Heuristics

## 1. **Slam the Straights** 
*Source: Rules 1, 14, 26*  
**When:** You roll a Large Straight (2-3-4-5-6) or Small Straight (1-2-3-4-5)  
**Action:** Lock it in immediately - Large Straight for 20 points, Small Straight for 15  
**Why:** These fixed-value patterns prevent massive regret. Rule 1 shows taking Large Straight avoids losing 17.8 points to the runner-up (Chance). The solver prioritizes guaranteed points over speculative upper section builds.  
**Coverage:** 10,212 + 3,312 decisions (13.5% of analyzed positions)

## 2. **Yatzy Supremacy**
*Source: Rules 2, 5, 7, 10, 16-21, 25, 27, 31*  
**When:** You roll five-of-a-kind (any face)  
**Action:** Take Yatzy for 50 points, unless very specific conditions apply:
- Only exception: When you need exactly that upper number for bonus AND you're behind pace
**Why:** Yatzy's 50 points typically crushes alternatives. Rule 2 shows taking Yatzy with five 1s prevents 28.9 points of regret vs taking Ones. Even five 6s (30 points in Sixes) usually can't compete with Yatzy's value.  
**Coverage:** 10,000+ decisions across all Yatzy rules

## 3. **Upper Section Quad Protocol**
*Source: Rules 4, 6, 8, 11, 22-23, 32-33, 43*  
**When:** You roll four-of-a-kind in an upper section number (1s through 6s)  
**Action:** Usually take the matching upper category IF:
- You're behind bonus pace (need to catch up to 63)
- It's early/mid game with flexibility remaining
- The quad is in 4s, 5s, or 6s (higher value)
**Why:** Four dice in an upper category often outscores the generic Four of a Kind category. Rule 23 shows four 4s in Fours (16 points) beats taking them as Four of a Kind, preventing 5.8 points of regret. This builds toward the crucial 50-point bonus.  
**Coverage:** ~25,000 decisions  

## 4. **Triple Management**
*Source: Rules 9, 13, 30, 34, 36-37, 40, 45-46, 48*  
**When:** You roll three-of-a-kind  
**Action:** Priority order:
1. Sixes (18 points) - almost always take when available (Rule 37)
2. Fives (15 points) - strong unless you have better options  
3. Fours (12 points) - decent early, weaker late
4. Threes (9 points) - only if behind on bonus or no better plays
5. Twos/Ones (6/3 points) - usually better in Three of a Kind category
**Why:** Higher triples provide superior upper section progress. Rule 37 shows three 6s prevents 5.5 points of regret. The solver recognizes that upper section progress compounds through bonus potential.  
**Coverage:** 40,000+ triple decisions

## 5. **Full House Timing**
*Source: Rules 35, 42*  
**When:** You have three-of-a-kind plus a pair  
**Action:** Take Full House (sum of all dice) when:
- Upper score ≤ 42 (still building toward bonus)
- Late game when you need the guaranteed points
- Avoid if ahead on bonus pace and the triple is 5s or 6s
**Why:** Full House provides solid guaranteed points but conflicts with upper section building. Rule 35 shows taking Full House early prevents 7.1 points of regret vs alternatives.  
**Coverage:** 15,615 decisions

## 6. **Pair Dynamics**
*Source: Rules 38, 47, 49-53, 55-59, 61-62, 66*  
**When:** You have one or two pairs  
**Action:**
- Two Pairs: Take when sum ≥ 26 (high pairs) or alternatives are weak
- One Pair: Prioritize 6s (12 points), especially late game
- Dump low pairs in their upper categories early if behind on bonus
**Why:** Pairs provide consistent scoring but lower ceiling than building uppers. Rule 49 shows taking a pair of 6s late prevents only 0.7 regret—it's often the best remaining option.  
**Coverage:** 50,000+ pair decisions

## 7. **Chance as Safety Valve**
*Source: Rules 62, 65-67, 86*  
**When:** High dice sum (≥20) with no strong patterns  
**Action:** Take Chance when:
- Dice sum ≥ 22 and you have flexibility
- Very early game (turn 1-2) to preserve options
- Late game when category scores converge
**Why:** Chance preserves flexibility while capturing value from high rolls. Rule 67 shows taking 20+ in Chance on turn 1 prevents 2.2 regret by avoiding premature category burns.  
**Coverage:** 5,000+ decisions

## 8. **Garbage Routing**
*Source: Rules 63, 68-77, 87-100*  
**When:** Poor roll with no natural fits  
**Action:** Burn categories in this order:
1. Ones/Twos (if bonus secured or hopeless)
2. Small Straight (if you have Large Straight)
3. Large Straight (if bonus behind and late game)
4. Four of a Kind (only if very late with no quads possible)
**Why:** Minimize opportunity cost by burning low-impact categories. Rules 87-88 show burning Ones after securing bonus prevents negative regret—you're removing a liability.  
**Coverage:** 15,000+ garbage decisions

## 9. **Endgame Forcing**
*Source: Rules 3, 78-82, 84-85*  
**When:** Final 1-3 turns with limited categories  
**Action:** Take whatever's available, but if Four of a Kind is open:
- High sums (≥23) go here even without four-of-a-kind
- Acts as a "second Chance" category
**Why:** Flexibility vanishes in endgame. Rule 3 shows the final turn forces any roll into the last category with 0.0 regret—there's no alternative.  
**Coverage:** 20,000+ endgame positions

## Default Play
**When no specific rule applies:**  
Take the highest available score, with tiebreakers:
1. Prefer upper section if behind bonus pace
2. Prefer lower section if ahead on bonus  
3. Prefer categories that preserve future flexibility
4. When truly stuck, burn Ones or Twos

The unmatched positions show mean regret around 2-3 points, suggesting the 100 rules capture most high-stakes decisions. The solver's philosophy: secure guaranteed points, protect bonus pace, and route garbage intelligently.