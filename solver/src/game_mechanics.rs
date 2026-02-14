//! Yatzy scoring rules: s(S, r, c) and upper-score successor function.
//!
//! Implements the pseudocode's scoring function s(S, r, c) for all 15 Scandinavian
//! Yatzy categories, and the upper-score update u(S, r, c) used to compute
//! successor states n(S, r, c).

use crate::constants::*;
use crate::dice_mechanics::count_faces;

/// Compute s(S, r, c): the score for placing a 5-dice roll in the given category.
///
/// Categories 0–5 are the upper section (Ones through Sixes): score = face_value × count.
/// Categories 6–14 are the lower section (One Pair through Yatzy) with Scandinavian rules.
pub fn calculate_category_score(dice: &[i32; 5], category: usize) -> i32 {
    let face_count = count_faces(dice);
    let sum_all: i32 = dice.iter().sum();

    match category {
        CATEGORY_ONES | CATEGORY_TWOS | CATEGORY_THREES | CATEGORY_FOURS | CATEGORY_FIVES
        | CATEGORY_SIXES => {
            let face = (category + 1) as i32;
            face_count[face as usize] * face
        }
        CATEGORY_ONE_PAIR => {
            for f in (1..=6).rev() {
                if face_count[f] >= 2 {
                    return 2 * f as i32;
                }
            }
            0
        }
        CATEGORY_TWO_PAIRS => {
            let mut pairs = [0i32; 2];
            let mut pcount = 0;
            for f in (1..=6).rev() {
                if face_count[f] >= 2 {
                    pairs[pcount] = f as i32;
                    pcount += 1;
                    if pcount == 2 {
                        break;
                    }
                }
            }
            if pcount == 2 {
                2 * pairs[0] + 2 * pairs[1]
            } else {
                0
            }
        }
        CATEGORY_THREE_OF_A_KIND => n_of_a_kind_score(&face_count, 3),
        CATEGORY_FOUR_OF_A_KIND => n_of_a_kind_score(&face_count, 4),
        CATEGORY_SMALL_STRAIGHT => {
            if face_count[1] == 1
                && face_count[2] == 1
                && face_count[3] == 1
                && face_count[4] == 1
                && face_count[5] == 1
            {
                15
            } else {
                0
            }
        }
        CATEGORY_LARGE_STRAIGHT => {
            if face_count[2] == 1
                && face_count[3] == 1
                && face_count[4] == 1
                && face_count[5] == 1
                && face_count[6] == 1
            {
                20
            } else {
                0
            }
        }
        CATEGORY_FULL_HOUSE => {
            let mut three_face = 0;
            let mut pair_face = 0;
            for f in 1..=6 {
                if face_count[f] == 3 {
                    three_face = f;
                } else if face_count[f] == 2 {
                    pair_face = f;
                }
            }
            if three_face != 0 && pair_face != 0 {
                sum_all
            } else {
                0
            }
        }
        CATEGORY_CHANCE => sum_all,
        CATEGORY_YATZY => {
            for f in 1..=6 {
                if face_count[f] == 5 {
                    return 50;
                }
            }
            0
        }
        _ => 0,
    }
}

/// Scoring helper for N-of-a-kind categories.
/// Returns highest_face * n if any face appears >= n times, else 0.
fn n_of_a_kind_score(face_count: &[i32; 7], n: i32) -> i32 {
    for face in (1..=6).rev() {
        if face_count[face] >= n {
            return face as i32 * n;
        }
    }
    0
}

/// Compute successor upper score: m' = min(m + u(S,r,c), 63).
///
/// Pseudocode: u(S, r, c) = s(S,r,c) for upper categories (c ∈ {0..5}), 0 otherwise.
/// The result is capped at 63 because the upper bonus threshold is 63 points.
pub fn update_upper_score(upper_score: i32, category: usize, score: i32) -> i32 {
    if category < 6 {
        let new_upper_score = upper_score + score;
        if new_upper_score > 63 {
            63
        } else {
            new_upper_score
        }
    } else {
        upper_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upper_section() {
        assert_eq!(calculate_category_score(&[1, 1, 1, 1, 1], CATEGORY_ONES), 5);
        assert_eq!(
            calculate_category_score(&[6, 6, 6, 6, 6], CATEGORY_SIXES),
            30
        );
        assert_eq!(calculate_category_score(&[1, 2, 3, 4, 5], CATEGORY_ONES), 1);
        assert_eq!(
            calculate_category_score(&[3, 3, 4, 5, 6], CATEGORY_THREES),
            6
        );
        assert_eq!(calculate_category_score(&[1, 2, 3, 4, 5], CATEGORY_TWOS), 2);
        assert_eq!(
            calculate_category_score(&[4, 4, 4, 4, 4], CATEGORY_FOURS),
            20
        );
        assert_eq!(
            calculate_category_score(&[5, 5, 5, 1, 2], CATEGORY_FIVES),
            15
        );
    }

    #[test]
    fn test_one_pair() {
        assert_eq!(
            calculate_category_score(&[3, 3, 4, 5, 6], CATEGORY_ONE_PAIR),
            6
        );
        assert_eq!(
            calculate_category_score(&[6, 6, 5, 5, 1], CATEGORY_ONE_PAIR),
            12
        );
        assert_eq!(
            calculate_category_score(&[1, 2, 3, 4, 6], CATEGORY_ONE_PAIR),
            0
        );
    }

    #[test]
    fn test_two_pairs() {
        assert_eq!(
            calculate_category_score(&[3, 3, 5, 5, 6], CATEGORY_TWO_PAIRS),
            16
        );
        assert_eq!(
            calculate_category_score(&[1, 1, 2, 3, 4], CATEGORY_TWO_PAIRS),
            0
        );
    }

    #[test]
    fn test_n_of_a_kind() {
        assert_eq!(
            calculate_category_score(&[2, 2, 2, 4, 5], CATEGORY_THREE_OF_A_KIND),
            6
        );
        assert_eq!(
            calculate_category_score(&[4, 4, 4, 4, 2], CATEGORY_FOUR_OF_A_KIND),
            16
        );
        assert_eq!(
            calculate_category_score(&[1, 2, 3, 4, 5], CATEGORY_THREE_OF_A_KIND),
            0
        );
        assert_eq!(
            calculate_category_score(&[3, 3, 3, 4, 5], CATEGORY_FOUR_OF_A_KIND),
            0
        );
    }

    #[test]
    fn test_straights() {
        assert_eq!(
            calculate_category_score(&[1, 2, 3, 4, 5], CATEGORY_SMALL_STRAIGHT),
            15
        );
        assert_eq!(
            calculate_category_score(&[2, 3, 4, 5, 6], CATEGORY_LARGE_STRAIGHT),
            20
        );
        assert_eq!(
            calculate_category_score(&[1, 2, 3, 4, 6], CATEGORY_SMALL_STRAIGHT),
            0
        );
        assert_eq!(
            calculate_category_score(&[1, 2, 3, 4, 5], CATEGORY_LARGE_STRAIGHT),
            0
        );
        assert_eq!(
            calculate_category_score(&[2, 3, 4, 5, 6], CATEGORY_SMALL_STRAIGHT),
            0
        );
    }

    #[test]
    fn test_full_house() {
        assert_eq!(
            calculate_category_score(&[2, 2, 3, 3, 3], CATEGORY_FULL_HOUSE),
            13
        );
        assert_eq!(
            calculate_category_score(&[1, 2, 3, 4, 6], CATEGORY_FULL_HOUSE),
            0
        );
        assert_eq!(
            calculate_category_score(&[5, 5, 5, 5, 5], CATEGORY_FULL_HOUSE),
            0
        );
    }

    #[test]
    fn test_chance() {
        assert_eq!(
            calculate_category_score(&[3, 4, 1, 5, 6], CATEGORY_CHANCE),
            19
        );
        assert_eq!(
            calculate_category_score(&[1, 1, 1, 1, 1], CATEGORY_CHANCE),
            5
        );
    }

    #[test]
    fn test_yatzy() {
        assert_eq!(
            calculate_category_score(&[6, 6, 6, 6, 6], CATEGORY_YATZY),
            50
        );
        assert_eq!(
            calculate_category_score(&[6, 6, 6, 6, 5], CATEGORY_YATZY),
            0
        );
        assert_eq!(
            calculate_category_score(&[1, 1, 1, 1, 1], CATEGORY_YATZY),
            50
        );
    }

    #[test]
    fn test_update_upper_score() {
        assert_eq!(update_upper_score(0, CATEGORY_ONES, 5), 5);
        assert_eq!(update_upper_score(10, CATEGORY_SIXES, 30), 40);
        assert_eq!(update_upper_score(60, CATEGORY_FIVES, 30), 63);
        assert_eq!(update_upper_score(10, CATEGORY_ONE_PAIR, 12), 10);
        assert_eq!(update_upper_score(50, CATEGORY_YATZY, 50), 50);
        assert_eq!(update_upper_score(63, CATEGORY_ONES, 5), 63);
    }
}
