use crate::types::YatzyContext;

/// Count occurrences of each face (1-6) in a 5-dice set.
/// face_count[0] is unused; face_count[f] = count of face f.
pub fn count_faces(dice: &[i32; 5]) -> [i32; 7] {
    let mut face_count = [0i32; 7];
    for &d in dice {
        face_count[d as usize] += 1;
    }
    face_count
}

/// Normalize dice to canonical sorted form (ascending).
pub fn sort_dice_set(arr: &mut [i32; 5]) {
    for i in 0..4 {
        for j in (i + 1)..5 {
            if arr[j] < arr[i] {
                arr.swap(i, j);
            }
        }
    }
}

/// Map a sorted dice set to its index in R_{5,6} (0-251).
#[inline(always)]
pub fn find_dice_set_index(ctx: &YatzyContext, dice: &[i32; 5]) -> usize {
    ctx.index_lookup[(dice[0] - 1) as usize][(dice[1] - 1) as usize][(dice[2] - 1) as usize]
        [(dice[3] - 1) as usize][(dice[4] - 1) as usize] as usize
}

/// Compute P(empty -> r): probability of rolling this exact sorted dice set
/// from 5 fresh dice.
///
/// Formula: multinomial(5; face_counts) / 6^5
///        = 5! / (n1! * n2! * ... * n6!) / 6^5
pub fn compute_probability_of_dice_set(ctx: &YatzyContext, dice: &[i32; 5]) -> f64 {
    let mut face_count = [0i32; 7];
    for &d in dice {
        face_count[d as usize] += 1;
    }

    let numerator = ctx.factorial[5] as f64;
    let mut denominator = 1.0;
    for f in 1..=6 {
        if face_count[f] > 1 {
            let mut sub_fact = 1;
            for x in 2..=face_count[f] {
                sub_fact *= x;
            }
            denominator *= sub_fact as f64;
        }
    }

    let permutations = numerator / denominator;
    let total_outcomes = 6.0f64.powi(5);
    permutations / total_outcomes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0_tables;

    fn make_ctx() -> Box<YatzyContext> {
        let mut ctx = Box::new(YatzyContext::new());
        phase0_tables::precompute_lookup_tables(&mut ctx);
        ctx
    }

    #[test]
    fn test_sort_dice_set() {
        let mut d1 = [5, 3, 1, 4, 2];
        sort_dice_set(&mut d1);
        assert_eq!(d1, [1, 2, 3, 4, 5]);

        let mut d2 = [1, 2, 3, 4, 5];
        sort_dice_set(&mut d2);
        assert_eq!(d2, [1, 2, 3, 4, 5]);

        let mut d3 = [6, 5, 4, 3, 2];
        sort_dice_set(&mut d3);
        assert_eq!(d3, [2, 3, 4, 5, 6]);

        let mut d4 = [3, 3, 3, 3, 3];
        sort_dice_set(&mut d4);
        assert_eq!(d4, [3, 3, 3, 3, 3]);
    }

    #[test]
    fn test_find_dice_set_index() {
        let ctx = make_ctx();
        assert_eq!(find_dice_set_index(&ctx, &[1, 1, 1, 1, 1]), 0);
        assert_eq!(find_dice_set_index(&ctx, &[6, 6, 6, 6, 6]), 251);

        // Round-trip
        for i in 0..252 {
            assert_eq!(find_dice_set_index(&ctx, &ctx.all_dice_sets[i]), i);
        }
    }

    #[test]
    fn test_count_faces() {
        let fc = count_faces(&[1, 1, 2, 3, 3]);
        assert_eq!(fc[1], 2);
        assert_eq!(fc[2], 1);
        assert_eq!(fc[3], 2);
        assert_eq!(fc[4], 0);
        assert_eq!(fc[5], 0);
        assert_eq!(fc[6], 0);

        let fc2 = count_faces(&[6, 6, 6, 6, 6]);
        assert_eq!(fc2[6], 5);
        assert_eq!(fc2[1], 0);
    }

    #[test]
    fn test_probability() {
        let ctx = make_ctx();

        let p1 = compute_probability_of_dice_set(&ctx, &[1, 1, 1, 1, 1]);
        assert!((p1 - 1.0 / 7776.0).abs() < 1e-12);

        let p2 = compute_probability_of_dice_set(&ctx, &[1, 1, 1, 1, 2]);
        assert!((p2 - 5.0 / 7776.0).abs() < 1e-12);

        let sum: f64 = ctx.dice_set_probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }
}
