//! Property-based tests for core game mechanics.

use proptest::prelude::*;

use yatzy::constants::*;
use yatzy::dice_mechanics::{count_faces, sort_dice_set};
use yatzy::game_mechanics::{calculate_category_score, update_upper_score};

/// Strategy: generate a valid dice array (each die 1-6).
fn dice_strategy() -> impl Strategy<Value = [i32; 5]> {
    prop::array::uniform5(1..=6i32)
}

/// Strategy: generate a valid category index (0-14).
fn category_strategy() -> impl Strategy<Value = usize> {
    0..CATEGORY_COUNT
}

proptest! {
    // 1. Scores are always non-negative
    #[test]
    fn score_non_negative(dice in dice_strategy(), cat in category_strategy()) {
        let score = calculate_category_score(&dice, cat);
        prop_assert!(score >= 0, "score={score} for dice={dice:?} cat={cat}");
    }

    // 2. Scoring is deterministic
    #[test]
    fn score_deterministic(dice in dice_strategy(), cat in category_strategy()) {
        let s1 = calculate_category_score(&dice, cat);
        let s2 = calculate_category_score(&dice, cat);
        prop_assert_eq!(s1, s2);
    }

    // 3. sort_dice_set is idempotent
    #[test]
    fn sort_idempotent(dice in dice_strategy()) {
        let mut once = dice;
        sort_dice_set(&mut once);
        let mut twice = once;
        sort_dice_set(&mut twice);
        prop_assert_eq!(once, twice);
    }

    // 4. state_index produces unique indices for all valid (up, scored) pairs
    //    We test pairs of distinct inputs produce distinct outputs.
    #[test]
    fn state_index_unique(
        up1 in 0..64usize, sc1 in 0..(1u32 << 15) as usize,
        up2 in 0..64usize, sc2 in 0..(1u32 << 15) as usize,
    ) {
        if (up1, sc1) != (up2, sc2) {
            prop_assert_ne!(state_index(up1, sc1), state_index(up2, sc2));
        }
    }

    // 5. update_upper_score never exceeds 63
    #[test]
    fn upper_score_capped(
        up in 0..=63i32,
        cat in category_strategy(),
        score in 0..=50i32,
    ) {
        let result = update_upper_score(up, cat, score);
        prop_assert!(result <= 63, "result={result}");
        prop_assert!(result >= 0, "result={result}");
    }

    // 6. Five identical dice always score 50 for Yatzy
    #[test]
    fn yatzy_five_of_a_kind(face in 1..=6i32) {
        let dice = [face; 5];
        let score = calculate_category_score(&dice, CATEGORY_YATZY);
        prop_assert_eq!(score, 50);
    }

    // 7. count_faces always sums to 5
    #[test]
    fn count_faces_sums_to_5(dice in dice_strategy()) {
        let counts = count_faces(&dice);
        let total: i32 = counts.iter().sum();
        prop_assert_eq!(total, 5);
    }
}

// 8. Keep-table probability rows sum to ~1.0 (non-proptest, needs YatzyContext)
#[test]
fn keep_table_rows_sum_to_one() {
    let mut ctx = yatzy::types::YatzyContext::new_boxed();
    yatzy::phase0_tables::precompute_lookup_tables(&mut ctx);

    let kt = &ctx.keep_table;
    for keep_id in 0..NUM_KEEP_MULTISETS {
        let start = kt.row_start[keep_id] as usize;
        let end = kt.row_start[keep_id + 1] as usize;
        if start == end {
            continue; // empty row (keep-all produces exactly one outcome)
        }
        let sum: f32 = kt.vals[start..end].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "keep_id={keep_id} row sum={sum}, expected ~1.0"
        );
    }
}
