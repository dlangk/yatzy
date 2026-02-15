//! Heuristic (human-like) Yatzy strategy.
//!
//! Implements pattern-matching reroll decisions and greedy category selection
//! that approximate how a competent human plays Yatzy. No state-value tables
//! are needed — all decisions use local pattern recognition.

use crate::constants::*;
use crate::dice_mechanics::count_faces;
use crate::game_mechanics::{calculate_category_score, update_upper_score};

/// Compute a heuristic reroll mask for the given dice and game state.
///
/// Returns a 5-bit mask where bit `i` set means "reroll die at position i".
/// The dice must be sorted ascending. The `face_count` array is 7-element
/// with index 0 unused (faces are 1-indexed).
///
/// Priority cascade (first match wins):
/// 1. Yatzy chase: ≥3 of a kind and Yatzy open → keep all of that face
/// 2. Large straight: 4 of 5 present and open → keep the 4
/// 3. Small straight: 4 of 5 present and open → keep the 4
/// 4. Full house: 3+2 → keep all; 3+1+1 → keep the 3
/// 5. Four-of-a-kind: ≥4 → keep the 4
/// 6. Three-of-a-kind: ≥3 → keep the 3
/// 7. Two pairs: two distinct pairs → keep all 4
/// 8. One pair: highest pair → keep the 2
/// 9. Fallback: keep the highest die, reroll rest
pub fn heuristic_reroll_mask(
    dice: &[i32; 5],
    face_count: &[i32; 7],
    scored: i32,
    _upper_score: i32,
) -> i32 {
    // 1. Yatzy chase: if any face has count ≥ 3 and Yatzy is open
    if !is_category_scored(scored, CATEGORY_YATZY) {
        for f in (1..=6).rev() {
            if face_count[f] >= 3 {
                return mask_keeping_face(dice, f as i32);
            }
        }
    }

    // 2. Large straight [2,3,4,5,6]
    if !is_category_scored(scored, CATEGORY_LARGE_STRAIGHT) {
        let mask = straight_chase_mask(dice, face_count, &[2, 3, 4, 5, 6]);
        if mask >= 0 {
            return mask;
        }
    }

    // 3. Small straight [1,2,3,4,5]
    if !is_category_scored(scored, CATEGORY_SMALL_STRAIGHT) {
        let mask = straight_chase_mask(dice, face_count, &[1, 2, 3, 4, 5]);
        if mask >= 0 {
            return mask;
        }
    }

    // 4. Full house
    if !is_category_scored(scored, CATEGORY_FULL_HOUSE) {
        let three_face = (1..=6).rev().find(|&f| face_count[f] == 3);
        let pair_face = (1..=6).rev().find(|&f| face_count[f] == 2);
        if let Some(_tf) = three_face {
            if pair_face.is_some() {
                // Already a full house — keep all
                return 0;
            }
            // 3-of-a-kind + 2 singletons → keep the 3
            return mask_keeping_face(dice, _tf as i32);
        }
    }

    // 5. Four-of-a-kind
    if !is_category_scored(scored, CATEGORY_FOUR_OF_A_KIND) {
        for f in (1..=6).rev() {
            if face_count[f] >= 4 {
                return mask_keeping_n_of_face(dice, f as i32, 4);
            }
        }
    }

    // 6. Three-of-a-kind
    if !is_category_scored(scored, CATEGORY_THREE_OF_A_KIND) {
        for f in (1..=6).rev() {
            if face_count[f] >= 3 {
                return mask_keeping_face(dice, f as i32);
            }
        }
    }

    // 7. Two pairs
    if !is_category_scored(scored, CATEGORY_TWO_PAIRS) {
        let pairs: Vec<usize> = (1..=6).rev().filter(|&f| face_count[f] >= 2).collect();
        if pairs.len() >= 2 {
            // Keep all dice matching the two highest pairs
            return mask_keeping_two_faces(dice, pairs[0] as i32, pairs[1] as i32);
        }
    }

    // 8. One pair: highest pair
    if !is_category_scored(scored, CATEGORY_ONE_PAIR) {
        for f in (1..=6).rev() {
            if face_count[f] >= 2 {
                return mask_keeping_n_of_face(dice, f as i32, 2);
            }
        }
    }

    // 9. Fallback: keep the single highest die, reroll rest
    // (biased toward Chance / upper section high dice)
    mask_keeping_highest(dice)
}

/// Choose the best open category for the given final dice.
///
/// Priority cascade (greedy, no lookahead):
/// 1. Yatzy = 50
/// 2. Large straight = 20
/// 3. Small straight = 15
/// 4. Full house > 0
/// 5. Four-of-a-kind > 0
/// 6. Three-of-a-kind > 0
/// 7. Two pairs > 0
/// 8. One pair ≥ 8
/// 9. Upper section bonus chase (score ≥ face × 3)
/// 10. Chance ≥ 20
/// 11. Any upper category with score > 0
/// 12. Chance (any score)
/// 13. Dump: score 0 on least valuable open category
pub fn heuristic_pick_category(
    dice: &[i32; 5],
    _face_count: &[i32; 7],
    scored: i32,
    upper_score: i32,
) -> usize {
    // Helper: check if open and scores the given value
    let open = |c: usize| !is_category_scored(scored, c);
    let score = |c: usize| calculate_category_score(dice, c);

    // 1. Yatzy = 50
    if open(CATEGORY_YATZY) && score(CATEGORY_YATZY) == 50 {
        return CATEGORY_YATZY;
    }

    // 2. Large straight = 20
    if open(CATEGORY_LARGE_STRAIGHT) && score(CATEGORY_LARGE_STRAIGHT) == 20 {
        return CATEGORY_LARGE_STRAIGHT;
    }

    // 3. Small straight = 15
    if open(CATEGORY_SMALL_STRAIGHT) && score(CATEGORY_SMALL_STRAIGHT) == 15 {
        return CATEGORY_SMALL_STRAIGHT;
    }

    // 4. Full house > 0
    if open(CATEGORY_FULL_HOUSE) && score(CATEGORY_FULL_HOUSE) > 0 {
        return CATEGORY_FULL_HOUSE;
    }

    // 5. Four-of-a-kind > 0
    if open(CATEGORY_FOUR_OF_A_KIND) && score(CATEGORY_FOUR_OF_A_KIND) > 0 {
        return CATEGORY_FOUR_OF_A_KIND;
    }

    // 6. Three-of-a-kind > 0
    if open(CATEGORY_THREE_OF_A_KIND) && score(CATEGORY_THREE_OF_A_KIND) > 0 {
        return CATEGORY_THREE_OF_A_KIND;
    }

    // 7. Two pairs > 0
    if open(CATEGORY_TWO_PAIRS) && score(CATEGORY_TWO_PAIRS) > 0 {
        return CATEGORY_TWO_PAIRS;
    }

    // 8. One pair ≥ 8 (decent pair: 4+4 or higher)
    if open(CATEGORY_ONE_PAIR) && score(CATEGORY_ONE_PAIR) >= 8 {
        return CATEGORY_ONE_PAIR;
    }

    // 9. Upper section bonus chase: if upper_score < 63, find best upper
    //    category where score ≥ face × 3 (the "expected" threshold)
    if upper_score < 63 {
        let mut best_upper: Option<usize> = None;
        let mut best_upper_score = 0;
        for c in 0..6 {
            if open(c) {
                let s = score(c);
                let face = (c + 1) as i32;
                let threshold = face * 3;
                if s >= threshold && s > best_upper_score {
                    best_upper = Some(c);
                    best_upper_score = s;
                }
            }
        }
        if let Some(c) = best_upper {
            return c;
        }
    }

    // 10. Chance ≥ 20 (decent catch-all)
    if open(CATEGORY_CHANCE) && score(CATEGORY_CHANCE) >= 20 {
        return CATEGORY_CHANCE;
    }

    // 11. Any upper category with score > 0 (pick the one closest to par)
    {
        let mut best_upper: Option<usize> = None;
        let mut best_ratio = f32::NEG_INFINITY;
        for c in 0..6 {
            if open(c) {
                let s = score(c);
                if s > 0 {
                    let face = (c + 1) as i32;
                    let ratio = s as f32 / (face * 3) as f32;
                    if ratio > best_ratio {
                        best_ratio = ratio;
                        best_upper = Some(c);
                    }
                }
            }
        }
        if let Some(c) = best_upper {
            return c;
        }
    }

    // 12. Chance as fallback (any score)
    if open(CATEGORY_CHANCE) {
        return CATEGORY_CHANCE;
    }

    // 13. One pair (any score)
    if open(CATEGORY_ONE_PAIR) && score(CATEGORY_ONE_PAIR) > 0 {
        return CATEGORY_ONE_PAIR;
    }

    // 14. Dump: score 0 on least valuable open category
    // Prefer dumping in order: Ones, Twos, Threes, Yatzy, Small Straight,
    // Large Straight, Full House, Two Pairs, Three-of-a-Kind, Four-of-a-Kind,
    // One Pair, Fours, Fives, Sixes, Chance
    let dump_order = [
        CATEGORY_ONES,
        CATEGORY_TWOS,
        CATEGORY_THREES,
        CATEGORY_YATZY,
        CATEGORY_SMALL_STRAIGHT,
        CATEGORY_LARGE_STRAIGHT,
        CATEGORY_FULL_HOUSE,
        CATEGORY_TWO_PAIRS,
        CATEGORY_THREE_OF_A_KIND,
        CATEGORY_FOUR_OF_A_KIND,
        CATEGORY_ONE_PAIR,
        CATEGORY_FOURS,
        CATEGORY_FIVES,
        CATEGORY_SIXES,
        CATEGORY_CHANCE,
    ];
    for &c in &dump_order {
        if open(c) {
            return c;
        }
    }

    // Should never reach here if game state is valid (at least 1 category open)
    unreachable!("No open category found — invalid game state");
}

// ── Mask helpers ─────────────────────────────────────────────────────────

/// Build a reroll mask that keeps ALL dice matching `face`, rerolls the rest.
fn mask_keeping_face(dice: &[i32; 5], face: i32) -> i32 {
    let mut mask = 0;
    for i in 0..5 {
        if dice[i] != face {
            mask |= 1 << i;
        }
    }
    mask
}

/// Build a reroll mask that keeps exactly `n` dice matching `face`.
fn mask_keeping_n_of_face(dice: &[i32; 5], face: i32, n: i32) -> i32 {
    let mut mask = 0;
    let mut kept = 0;
    // Keep the first n matching dice (dice are sorted, so this is deterministic)
    for i in 0..5 {
        if dice[i] == face && kept < n {
            kept += 1;
        } else {
            mask |= 1 << i;
        }
    }
    mask
}

/// Build a reroll mask that keeps all dice matching either of two faces.
fn mask_keeping_two_faces(dice: &[i32; 5], face1: i32, face2: i32) -> i32 {
    let mut mask = 0;
    let mut kept1 = 0;
    let mut kept2 = 0;
    for i in 0..5 {
        if dice[i] == face1 && kept1 < 2 {
            kept1 += 1;
        } else if dice[i] == face2 && kept2 < 2 {
            kept2 += 1;
        } else {
            mask |= 1 << i;
        }
    }
    mask
}

/// Build a reroll mask keeping only the highest single die.
fn mask_keeping_highest(dice: &[i32; 5]) -> i32 {
    // dice is sorted ascending, so dice[4] is the highest
    let mut mask = 0;
    for i in 0..4 {
        mask |= 1 << i;
    }
    // Keep position 4 (highest die)
    let _ = dice; // suppress unused warning — we rely on sorted order
    mask
}

/// Check if a straight chase is viable: at least 4 of the 5 target faces present.
/// Returns the reroll mask (keeping the matching 4+) or -1 if not viable.
fn straight_chase_mask(dice: &[i32; 5], face_count: &[i32; 7], target: &[i32; 5]) -> i32 {
    let mut present = 0;
    let mut missing_face = 0;
    for &f in target {
        if face_count[f as usize] >= 1 {
            present += 1;
        } else {
            missing_face = f;
        }
    }
    if present < 4 {
        return -1;
    }

    if present == 5 {
        // Already have the straight — keep all
        return 0;
    }

    // Have 4 of 5: keep the 4 target faces, reroll the odd die
    // Find which die to reroll: it's the one not in the target set
    let mut mask = 0;
    let mut target_remaining = [0i32; 5];
    target_remaining.copy_from_slice(target);
    // Remove the missing face from targets
    let mut active_targets: Vec<i32> = target_remaining
        .iter()
        .copied()
        .filter(|&f| f != missing_face)
        .collect();

    for i in 0..5 {
        // Try to match this die to an active target
        if let Some(pos) = active_targets.iter().position(|&t| t == dice[i]) {
            active_targets.remove(pos);
        } else {
            mask |= 1 << i;
        }
    }
    mask
}

/// Pick the best category and compute the score for given final dice.
///
/// Convenience wrapper around `heuristic_pick_category` that also computes
/// the score and updated upper score.
/// Returns (category, score, new_upper_score).
pub fn heuristic_score_dice(
    dice: &[i32; 5],
    scored: i32,
    upper_score: i32,
) -> (usize, i32, i32) {
    let face_count = count_faces(dice);
    let cat = heuristic_pick_category(dice, &face_count, scored, upper_score);
    let scr = calculate_category_score(dice, cat);
    let new_upper = update_upper_score(upper_score, cat, scr);
    (cat, scr, new_upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yatzy_chase() {
        // Three 6s with Yatzy open → keep the 6s, reroll the rest
        let dice = [2, 3, 6, 6, 6];
        let fc = count_faces(&dice);
        let mask = heuristic_reroll_mask(&dice, &fc, 0, 0);
        // Should reroll positions 0, 1 (dice[0]=2, dice[1]=3)
        assert_eq!(mask, 0b00011); // bits 0 and 1
    }

    #[test]
    fn test_yatzy_scored_skip_to_next() {
        // Three 6s but Yatzy already scored → should fall through
        let dice = [2, 3, 6, 6, 6];
        let fc = count_faces(&dice);
        let scored = 1 << CATEGORY_YATZY;
        let mask = heuristic_reroll_mask(&dice, &fc, scored, 0);
        // Should NOT chase Yatzy. Next applicable rule depends on what's open.
        // With only Yatzy scored, three-of-a-kind is open → keep the 6s
        assert_eq!(mask, 0b00011);
    }

    #[test]
    fn test_large_straight_chase() {
        // Have [2,3,4,5,5] → missing 6, have 4 of [2,3,4,5,6]
        let dice = [2, 3, 4, 5, 5];
        let fc = count_faces(&dice);
        let mask = heuristic_reroll_mask(&dice, &fc, 0, 0);
        // Should keep 2,3,4,5 and reroll the extra 5
        // After Yatzy check (no ≥3 of a kind), tries large straight
        assert_ne!(mask, 0); // should reroll something
        // The extra 5 at position 4 (or 3) should be rerolled
        // Positions: [2,3,4,5,5] → keep 2(pos0), 3(pos1), 4(pos2), 5(pos3), reroll 5(pos4)
        assert_eq!(mask, 0b10000);
    }

    #[test]
    fn test_full_house_keep_all() {
        let dice = [2, 2, 5, 5, 5];
        let fc = count_faces(&dice);
        let mask = heuristic_reroll_mask(&dice, &fc, 0, 0);
        // Yatzy check: 3 fives, Yatzy open → chase Yatzy (keep 5s, reroll 2s)
        // Actually, Yatzy chase takes priority!
        assert_eq!(mask, 0b00011); // reroll the two 2s
    }

    #[test]
    fn test_full_house_when_yatzy_scored() {
        let dice = [2, 2, 5, 5, 5];
        let fc = count_faces(&dice);
        let scored = 1 << CATEGORY_YATZY; // Yatzy already scored
        let mask = heuristic_reroll_mask(&dice, &fc, scored, 0);
        // Now should see full house: 3+2 → keep all
        assert_eq!(mask, 0);
    }

    #[test]
    fn test_one_pair_keep_highest() {
        // [1, 2, 3, 5, 5] — pair of 5s
        let dice = [1, 2, 3, 5, 5];
        let fc = count_faces(&dice);
        // Score out Yatzy, straights, full house, 4oak, 3oak, 2pairs
        let scored = (1 << CATEGORY_YATZY)
            | (1 << CATEGORY_LARGE_STRAIGHT)
            | (1 << CATEGORY_SMALL_STRAIGHT)
            | (1 << CATEGORY_FULL_HOUSE)
            | (1 << CATEGORY_FOUR_OF_A_KIND)
            | (1 << CATEGORY_THREE_OF_A_KIND)
            | (1 << CATEGORY_TWO_PAIRS);
        let mask = heuristic_reroll_mask(&dice, &fc, scored, 0);
        // Should keep the pair of 5s (positions 3,4), reroll rest
        assert_eq!(mask, 0b00111);
    }

    #[test]
    fn test_fallback_keep_highest() {
        let dice = [1, 2, 3, 4, 6];
        let fc = count_faces(&dice);
        // Score out everything except Chance
        let scored = (1 << CATEGORY_YATZY)
            | (1 << CATEGORY_LARGE_STRAIGHT)
            | (1 << CATEGORY_SMALL_STRAIGHT)
            | (1 << CATEGORY_FULL_HOUSE)
            | (1 << CATEGORY_FOUR_OF_A_KIND)
            | (1 << CATEGORY_THREE_OF_A_KIND)
            | (1 << CATEGORY_TWO_PAIRS)
            | (1 << CATEGORY_ONE_PAIR);
        let mask = heuristic_reroll_mask(&dice, &fc, scored, 0);
        // Fallback: keep highest die (position 4 = 6), reroll rest
        assert_eq!(mask, 0b01111);
    }

    #[test]
    fn test_pick_yatzy() {
        let dice = [3, 3, 3, 3, 3];
        let fc = count_faces(&dice);
        assert_eq!(heuristic_pick_category(&dice, &fc, 0, 0), CATEGORY_YATZY);
    }

    #[test]
    fn test_pick_large_straight() {
        let dice = [2, 3, 4, 5, 6];
        let fc = count_faces(&dice);
        assert_eq!(
            heuristic_pick_category(&dice, &fc, 0, 0),
            CATEGORY_LARGE_STRAIGHT
        );
    }

    #[test]
    fn test_pick_full_house() {
        let dice = [2, 2, 5, 5, 5];
        let fc = count_faces(&dice);
        let scored = (1 << CATEGORY_YATZY)
            | (1 << CATEGORY_LARGE_STRAIGHT)
            | (1 << CATEGORY_SMALL_STRAIGHT);
        assert_eq!(
            heuristic_pick_category(&dice, &fc, scored, 0),
            CATEGORY_FULL_HOUSE
        );
    }

    #[test]
    fn test_dump_order() {
        // Dice [1,2,3,4,6] with almost everything scored → dump on least valuable
        let dice = [1, 2, 3, 4, 6];
        let fc = count_faces(&dice);
        // Score out everything except Ones and Twos
        let mut scored = 0;
        for c in 2..CATEGORY_COUNT {
            scored |= 1 << c;
        }
        // Ones scores 1, Twos scores 2. Both open.
        // Upper bonus chase: Ones=1 < 1*3=3 and Twos=2 < 2*3=6 → neither meets threshold
        // No Chance open. No One Pair open.
        // Falls through to "any upper with score > 0": Ones ratio=1/3, Twos ratio=2/6=1/3 (tie)
        // Actually both have ratio 1/3, but Ones is first in the loop (c=0 before c=1)
        let cat = heuristic_pick_category(&dice, &fc, scored, 0);
        assert!(cat == CATEGORY_ONES || cat == CATEGORY_TWOS);
    }

    #[test]
    fn test_upper_bonus_chase() {
        // Dice [5,5,5,1,2] → Fives score = 15 ≥ 5*3=15 → take Fives
        let dice = [1, 2, 5, 5, 5];
        let fc = count_faces(&dice);
        // Score out all lower categories
        let scored = (1 << CATEGORY_ONE_PAIR)
            | (1 << CATEGORY_TWO_PAIRS)
            | (1 << CATEGORY_THREE_OF_A_KIND)
            | (1 << CATEGORY_FOUR_OF_A_KIND)
            | (1 << CATEGORY_SMALL_STRAIGHT)
            | (1 << CATEGORY_LARGE_STRAIGHT)
            | (1 << CATEGORY_FULL_HOUSE)
            | (1 << CATEGORY_CHANCE)
            | (1 << CATEGORY_YATZY);
        let cat = heuristic_pick_category(&dice, &fc, scored, 0);
        assert_eq!(cat, CATEGORY_FIVES);
    }
}
