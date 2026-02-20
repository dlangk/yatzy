//! Semantic feature DSL for human-interpretable game state descriptions.
//!
//! Computes ~40 features from (game state, dice, rerolls_remaining) that are
//! meaningful to a human player: dice topology, bonus tracking, category availability,
//! and immediate scores.

use crate::constants::*;
use crate::dice_mechanics::count_faces;
use crate::game_mechanics::calculate_category_score;
use crate::types::YatzyContext;

/// Human-interpretable features computed from game state + dice.
#[derive(Debug, Clone)]
pub struct SemanticFeatures {
    // Game progress
    pub turn: u8,
    pub categories_left: u8,
    pub rerolls_remaining: u8,

    // Upper bonus tracking
    pub upper_score: u8,
    pub bonus_secured: bool,
    pub bonus_pace: f32,
    pub upper_cats_left: u8,

    // Dice topology
    pub face_counts: [u8; 6],
    pub max_count: u8,
    pub num_distinct: u8,
    pub dice_sum: u8,
    pub has_pair: bool,
    pub has_two_pair: bool,
    pub has_three_of_kind: bool,
    pub has_four_of_kind: bool,
    pub has_full_house: bool,
    pub has_small_straight: bool,
    pub has_large_straight: bool,
    pub has_yatzy: bool,

    // Category availability
    pub cat_available: [bool; CATEGORY_COUNT],

    // Scores if chosen now
    pub cat_scores: [i32; CATEGORY_COUNT],

    // Safety valves
    pub zeros_available: u8,
    pub best_available_score: i32,
}

/// Expected upper-section score per turn, assuming average dice.
/// E[face × count] for 5 dice = face × 5/6 ≈ 0.833 × face.
/// Cumulative targets at each turn: 3.5 × 5/6 × turns_used ≈ 2.917/turn.
/// But upper categories are scored opportunistically, so pace is relative to
/// a simple linear schedule: 63 points / 6 upper categories = 10.5 per cat.
const PACE_PER_UPPER_CAT_SCORED: f32 = 10.5;

/// Compute semantic features from game state and dice.
pub fn compute_features(
    turn: usize,
    upper_score: i32,
    scored: i32,
    dice: &[i32; 5],
    rerolls_remaining: u8,
) -> SemanticFeatures {
    let face_count_i32 = count_faces(dice);
    let mut face_counts = [0u8; 6];
    for i in 0..6 {
        face_counts[i] = face_count_i32[i + 1] as u8;
    }

    let max_count = *face_counts.iter().max().unwrap();
    let num_distinct = face_counts.iter().filter(|&&c| c > 0).count() as u8;
    let dice_sum: u8 = dice.iter().map(|&d| d as u8).sum();

    // Dice topology booleans
    let pair_count = face_counts.iter().filter(|&&c| c >= 2).count();
    let has_pair = pair_count >= 1;
    let has_two_pair = pair_count >= 2;
    let has_three_of_kind = face_counts.iter().any(|&c| c >= 3);
    let has_four_of_kind = face_counts.iter().any(|&c| c >= 4);
    let has_yatzy = max_count == 5;

    // Full house: exactly one triple and one pair (different faces)
    let has_triple = face_counts.iter().any(|&c| c == 3);
    let has_exact_pair = face_counts.iter().any(|&c| c == 2);
    let has_full_house = has_triple && has_exact_pair;

    // Straights (Scandinavian: small=1-5, large=2-6)
    let has_small_straight = face_count_i32[1] == 1
        && face_count_i32[2] == 1
        && face_count_i32[3] == 1
        && face_count_i32[4] == 1
        && face_count_i32[5] == 1;
    let has_large_straight = face_count_i32[2] == 1
        && face_count_i32[3] == 1
        && face_count_i32[4] == 1
        && face_count_i32[5] == 1
        && face_count_i32[6] == 1;

    // Category availability and scoring
    let mut cat_available = [false; CATEGORY_COUNT];
    let mut cat_scores = [0i32; CATEGORY_COUNT];
    let mut categories_left: u8 = 0;
    let mut upper_cats_left: u8 = 0;
    let mut zeros_available: u8 = 0;
    let mut best_available_score: i32 = i32::MIN;

    for c in 0..CATEGORY_COUNT {
        if !is_category_scored(scored, c) {
            cat_available[c] = true;
            categories_left += 1;
            if c < 6 {
                upper_cats_left += 1;
            }
            let scr = calculate_category_score(dice, c);
            cat_scores[c] = scr;
            if scr == 0 {
                zeros_available += 1;
            }
            if scr > best_available_score {
                best_available_score = scr;
            }
        }
    }

    if best_available_score == i32::MIN {
        best_available_score = 0;
    }

    // Upper bonus tracking
    let bonus_secured = upper_score >= 63;
    let upper_cats_scored = 6 - upper_cats_left;
    let bonus_pace = if upper_cats_scored == 0 {
        1.0 // no data yet, assume on pace
    } else {
        let expected = upper_cats_scored as f32 * PACE_PER_UPPER_CAT_SCORED;
        if expected > 0.0 {
            upper_score as f32 / expected
        } else {
            1.0
        }
    };

    SemanticFeatures {
        turn: turn as u8,
        categories_left,
        rerolls_remaining,
        upper_score: upper_score as u8,
        bonus_secured,
        bonus_pace,
        upper_cats_left,
        face_counts,
        max_count,
        num_distinct,
        dice_sum,
        has_pair,
        has_two_pair,
        has_three_of_kind,
        has_four_of_kind,
        has_full_house,
        has_small_straight,
        has_large_straight,
        has_yatzy,
        cat_available,
        cat_scores,
        zeros_available,
        best_available_score,
    }
}

/// Serialize features into a flat f32 vector for binary export.
/// Raw (unnormalized) values — matches get_feature_value() in policy.rs.
/// Layout (56 floats):
///   [0]  turn (0-14)
///   [1]  categories_left (1-15)
///   [2]  rerolls_remaining (0-2)
///   [3]  upper_score (0-63)
///   [4]  bonus_secured (0/1)
///   [5]  bonus_pace (float)
///   [6]  upper_cats_left (0-6)
///   [7..13]  face_counts[0..6] (0-5)
///   [13] max_count (1-5)
///   [14] num_distinct (1-6)
///   [15] dice_sum (5-30)
///   [16] has_pair (0/1)
///   [17] has_two_pair (0/1)
///   [18] has_three_of_kind (0/1)
///   [19] has_four_of_kind (0/1)
///   [20] has_full_house (0/1)
///   [21] has_small_straight (0/1)
///   [22] has_large_straight (0/1)
///   [23] has_yatzy (0/1)
///   [24..39] cat_available[0..15] (0/1)
///   [39..54] cat_scores[0..15] (raw integer scores)
///   [54] zeros_available (0-15)
///   [55] best_available_score (raw integer)
pub const NUM_SEMANTIC_FEATURES: usize = 56;

pub fn features_to_f32(f: &SemanticFeatures) -> [f32; NUM_SEMANTIC_FEATURES] {
    let mut out = [0.0f32; NUM_SEMANTIC_FEATURES];
    let b = |v: bool| -> f32 {
        if v {
            1.0
        } else {
            0.0
        }
    };

    out[0] = f.turn as f32;
    out[1] = f.categories_left as f32;
    out[2] = f.rerolls_remaining as f32;
    out[3] = f.upper_score as f32;
    out[4] = b(f.bonus_secured);
    out[5] = f.bonus_pace;
    out[6] = f.upper_cats_left as f32;
    for i in 0..6 {
        out[7 + i] = f.face_counts[i] as f32;
    }
    out[13] = f.max_count as f32;
    out[14] = f.num_distinct as f32;
    out[15] = f.dice_sum as f32;
    out[16] = b(f.has_pair);
    out[17] = b(f.has_two_pair);
    out[18] = b(f.has_three_of_kind);
    out[19] = b(f.has_four_of_kind);
    out[20] = b(f.has_full_house);
    out[21] = b(f.has_small_straight);
    out[22] = b(f.has_large_straight);
    out[23] = b(f.has_yatzy);
    for i in 0..CATEGORY_COUNT {
        out[24 + i] = b(f.cat_available[i]);
    }
    for i in 0..CATEGORY_COUNT {
        out[39 + i] = f.cat_scores[i] as f32;
    }
    out[54] = f.zeros_available as f32;
    out[55] = f.best_available_score as f32;

    out
}

/// Feature names for the flat f32 vector (for Python consumption).
pub const FEATURE_NAMES: [&str; NUM_SEMANTIC_FEATURES] = [
    "turn",
    "categories_left",
    "rerolls_remaining",
    "upper_score",
    "bonus_secured",
    "bonus_pace",
    "upper_cats_left",
    "face_count_1",
    "face_count_2",
    "face_count_3",
    "face_count_4",
    "face_count_5",
    "face_count_6",
    "max_count",
    "num_distinct",
    "dice_sum",
    "has_pair",
    "has_two_pair",
    "has_three_of_kind",
    "has_four_of_kind",
    "has_full_house",
    "has_small_straight",
    "has_large_straight",
    "has_yatzy",
    "cat_avail_ones",
    "cat_avail_twos",
    "cat_avail_threes",
    "cat_avail_fours",
    "cat_avail_fives",
    "cat_avail_sixes",
    "cat_avail_one_pair",
    "cat_avail_two_pairs",
    "cat_avail_three_of_kind",
    "cat_avail_four_of_kind",
    "cat_avail_small_straight",
    "cat_avail_large_straight",
    "cat_avail_full_house",
    "cat_avail_chance",
    "cat_avail_yatzy",
    "cat_score_ones",
    "cat_score_twos",
    "cat_score_threes",
    "cat_score_fours",
    "cat_score_fives",
    "cat_score_sixes",
    "cat_score_one_pair",
    "cat_score_two_pairs",
    "cat_score_three_of_kind",
    "cat_score_four_of_kind",
    "cat_score_small_straight",
    "cat_score_large_straight",
    "cat_score_full_house",
    "cat_score_chance",
    "cat_score_yatzy",
    "zeros_available",
    "best_available_score",
];

/// Compute category Q-values: Q(state, dice, category) for all available categories.
/// Returns (q_values[15], best_category, best_q) where unavailable categories get NEG_INFINITY.
pub fn compute_category_q_values(
    ctx: &YatzyContext,
    sv: &[f32],
    up_score: i32,
    scored: i32,
    ds_index: usize,
    is_last_turn: bool,
) -> ([f32; CATEGORY_COUNT], usize, f32) {
    use crate::game_mechanics::update_upper_score;
    let mut q = [f32::NEG_INFINITY; CATEGORY_COUNT];
    let mut best_q = f32::NEG_INFINITY;
    let mut best_cat = 0usize;

    if is_last_turn {
        for c in 0..CATEGORY_COUNT {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let bonus = if c < 6 {
                    let new_up = update_upper_score(up_score, c, scr);
                    if new_up >= 63 && up_score < 63 {
                        50
                    } else {
                        0
                    }
                } else {
                    0
                };
                q[c] = (scr + bonus) as f32;
                if q[c] > best_q {
                    best_q = q[c];
                    best_cat = c;
                }
            }
        }
    } else {
        for c in 0..6 {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let new_up = update_upper_score(up_score, c, scr);
                let new_scored = scored | (1 << c);
                q[c] = scr as f32
                    + unsafe {
                        *sv.get_unchecked(state_index(new_up as usize, new_scored as usize))
                    };
                if q[c] > best_q {
                    best_q = q[c];
                    best_cat = c;
                }
            }
        }
        for c in 6..CATEGORY_COUNT {
            if !is_category_scored(scored, c) {
                let scr = ctx.precomputed_scores[ds_index][c];
                let new_scored = scored | (1 << c);
                q[c] = scr as f32
                    + unsafe {
                        *sv.get_unchecked(state_index(up_score as usize, new_scored as usize))
                    };
                if q[c] > best_q {
                    best_q = q[c];
                    best_cat = c;
                }
            }
        }
    }

    (q, best_cat, best_q)
}

/// Compute reroll Q-values: Q(state, dice, keep_mask) for all valid keeps.
/// Returns (q_values, best_mask, best_q) where q_values maps mask -> EV.
/// Index 0 = keep all (mask=0), indices 1..=unique_count = unique keep multisets.
pub fn compute_reroll_q_values(
    ctx: &YatzyContext,
    e_ds: &[f32; 252],
    dice: &[i32; 5],
) -> (Vec<(i32, f32)>, i32, f32) {
    use crate::dice_mechanics::find_dice_set_index;
    let ds_index = find_dice_set_index(ctx, dice);
    let kt = &ctx.keep_table;

    let mut actions: Vec<(i32, f32)> = Vec::with_capacity(32);

    // mask=0: keep all dice
    let keep_all_ev = e_ds[ds_index];
    actions.push((0, keep_all_ev));
    let mut best_q = keep_all_ev;
    let mut best_mask = 0i32;

    for j in 0..kt.unique_count[ds_index] as usize {
        let kid = kt.unique_keep_ids[ds_index][j] as usize;
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;
        let mut ev: f32 = 0.0;
        for k in start..end {
            unsafe {
                ev += (*kt.vals.get_unchecked(k) as f32)
                    * e_ds.get_unchecked(*kt.cols.get_unchecked(k) as usize);
            }
        }
        let mask = kt.keep_to_mask[ds_index * 32 + j];
        actions.push((mask, ev));
        if ev > best_q {
            best_q = ev;
            best_mask = mask;
        }
    }

    (actions, best_mask, best_q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_features_basic() {
        let dice = [1, 2, 3, 4, 5];
        let f = compute_features(0, 0, 0, &dice, 2);
        assert_eq!(f.turn, 0);
        assert_eq!(f.categories_left, 15);
        assert_eq!(f.rerolls_remaining, 2);
        assert_eq!(f.upper_score, 0);
        assert!(!f.bonus_secured);
        assert_eq!(f.dice_sum, 15);
        assert_eq!(f.num_distinct, 5);
        assert_eq!(f.max_count, 1);
        assert!(f.has_small_straight);
        assert!(!f.has_large_straight);
        assert!(!f.has_yatzy);
        assert!(f.cat_available[0]); // ones available
        assert_eq!(f.cat_scores[CATEGORY_SMALL_STRAIGHT], 15);
    }

    #[test]
    fn test_compute_features_yatzy() {
        let dice = [6, 6, 6, 6, 6];
        let f = compute_features(5, 30, 0b111, &dice, 0);
        assert!(f.has_yatzy);
        assert!(f.has_four_of_kind);
        assert!(f.has_three_of_kind);
        assert!(f.has_pair);
        assert_eq!(f.max_count, 5);
        assert_eq!(f.num_distinct, 1);
        assert_eq!(f.dice_sum, 30);
        assert_eq!(f.cat_scores[CATEGORY_YATZY], 50);
        assert_eq!(f.cat_scores[CATEGORY_SIXES], 30);
        // first 3 categories scored
        assert!(!f.cat_available[0]);
        assert!(!f.cat_available[1]);
        assert!(!f.cat_available[2]);
        assert!(f.cat_available[3]); // fours still available
    }

    #[test]
    fn test_features_to_f32_length() {
        let dice = [1, 2, 3, 4, 5];
        let f = compute_features(0, 0, 0, &dice, 2);
        let v = features_to_f32(&f);
        assert_eq!(v.len(), NUM_SEMANTIC_FEATURES);
        assert_eq!(FEATURE_NAMES.len(), NUM_SEMANTIC_FEATURES);
    }

    #[test]
    fn test_bonus_pace() {
        // Scored 3 upper cats (ones, twos, threes) with 20 pts → pace = 20/(3*10.5) ≈ 0.635
        let dice = [4, 4, 5, 5, 6];
        let scored = 0b000_000_000_000_111; // ones, twos, threes
        let f = compute_features(3, 20, scored, &dice, 2);
        assert_eq!(f.upper_cats_left, 3);
        let expected_pace = 20.0 / (3.0 * 10.5);
        assert!((f.bonus_pace - expected_pace).abs() < 0.01);
    }
}
