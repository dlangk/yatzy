//! API-facing computation wrappers.
//!
//! The main entry point is [`compute_roll_response`], which returns everything
//! the frontend needs for a single roll: mask EVs, category scores/EVs, and optimal
//! actions. One backend call per roll, zero for dice toggling or category selection.
//!
//! All internal computation uses f32; return types are f64 for JSON compatibility.

use crate::constants::*;
use crate::dice_mechanics::{find_dice_set_index, sort_dice_set};
use crate::game_mechanics::update_upper_score;
use crate::types::YatzyContext;
use crate::widget_solver::*;

/// Per-category info returned in the fat response.
#[derive(Clone)]
pub struct CategoryInfo {
    pub id: usize,
    pub name: &'static str,
    pub score: i32,
    pub available: bool,
    pub ev_if_scored: f64,
}

/// Everything the frontend needs until the next roll.
pub struct RollResponse {
    /// EV for each of the 32 reroll masks (None when rerolls_remaining == 0).
    pub mask_evs: Option<[f64; 32]>,
    /// Best reroll mask (None when rerolls_remaining == 0).
    pub optimal_mask: Option<i32>,
    /// EV of the optimal mask (None when rerolls_remaining == 0).
    pub optimal_mask_ev: Option<f64>,
    /// All 15 categories with scores and EVs.
    pub categories: [CategoryInfo; CATEGORY_COUNT],
    /// Best category to score (argmax of ev_if_scored among available).
    pub optimal_category: i32,
    /// EV of the optimal category.
    pub optimal_category_ev: f64,
    /// Overall state EV (for reference).
    pub state_ev: f64,
}

/// Compute everything the frontend needs for a single roll.
///
/// 1. Group 6: E(S, r, 0) for all 252 dice sets + per-category info for the given dice
/// 2. If rerolls > 0: propagate through Groups 5/3 to get e_ds at the right reroll level
/// 3. Compute all 32 mask EVs for the specific dice set
/// 4. Find optimal mask and optimal category
pub fn compute_roll_response(
    ctx: &YatzyContext,
    upper_score: i32,
    scored_categories: i32,
    dice: &[i32; 5],
    rerolls_remaining: i32,
) -> RollResponse {
    let mut sorted_dice = *dice;
    sort_dice_set(&mut sorted_dice);
    let ds_index = find_dice_set_index(ctx, &sorted_dice);
    let sv = ctx.state_values.as_slice();

    // --- Build categories info ---
    let categories: [CategoryInfo; CATEGORY_COUNT] = std::array::from_fn(|c| {
        let scr = ctx.precomputed_scores[ds_index][c];
        let available = !is_category_scored(scored_categories, c);
        let ev_if_scored = if available {
            let new_up = update_upper_score(upper_score, c, scr);
            let new_scored = scored_categories | (1 << c);
            (scr as f32 + sv[state_index(new_up as usize, new_scored as usize)]) as f64
        } else {
            f64::NEG_INFINITY
        };
        CategoryInfo {
            id: c,
            name: CATEGORY_NAMES[c],
            score: scr,
            available,
            ev_if_scored,
        }
    });

    // Find optimal category (argmax ev_if_scored among available with score > 0)
    let mut optimal_category = -1i32;
    let mut optimal_category_ev = f64::NEG_INFINITY;
    for cat in &categories {
        if cat.available && cat.score > 0 && cat.ev_if_scored > optimal_category_ev {
            optimal_category_ev = cat.ev_if_scored;
            optimal_category = cat.id as i32;
        }
    }

    // --- Build Group 6: E(S, r, 0) for all 252 dice sets ---
    let mut e_ds_0 = [0.0f32; 252];
    for ds_i in 0..252 {
        let mut best_val = f32::NEG_INFINITY;
        for c in 0..CATEGORY_COUNT {
            if !is_category_scored(scored_categories, c) {
                let scr = ctx.precomputed_scores[ds_i][c];
                let new_up = update_upper_score(upper_score, c, scr);
                let new_scored = scored_categories | (1 << c);
                let val = scr as f32 + sv[state_index(new_up as usize, new_scored as usize)];
                if val > best_val {
                    best_val = val;
                }
            }
        }
        e_ds_0[ds_i] = best_val;
    }

    // --- If rerolls remaining, compute mask EVs ---
    if rerolls_remaining > 0 {
        // Build reroll levels
        let e_ds_for_masks = if rerolls_remaining == 1 {
            e_ds_0
        } else {
            // rerolls == 2: need one level of propagation
            let mut e_ds_1 = [0.0f32; 252];
            let mut dummy_mask = [0i32; 252];
            compute_expected_values_for_n_rerolls(ctx, &e_ds_0, &mut e_ds_1, &mut dummy_mask);
            e_ds_1
        };

        // Compute all 32 mask EVs
        let mut mask_evs = [0.0f64; 32];
        for mask in 0..32 {
            mask_evs[mask] =
                compute_expected_value_for_reroll_mask(ctx, ds_index, &e_ds_for_masks, mask as i32);
        }

        // Find optimal mask
        let mut best_mask = 0i32;
        let mut best_mask_ev = mask_evs[0];
        for mask in 1..32 {
            if mask_evs[mask] > best_mask_ev {
                best_mask_ev = mask_evs[mask];
                best_mask = mask as i32;
            }
        }

        // Compare optimal mask vs optimal category to set overall state_ev
        let effective_state_ev = best_mask_ev.max(optimal_category_ev);

        RollResponse {
            mask_evs: Some(mask_evs),
            optimal_mask: Some(best_mask),
            optimal_mask_ev: Some(best_mask_ev),
            categories,
            optimal_category,
            optimal_category_ev,
            state_ev: effective_state_ev,
        }
    } else {
        // No rerolls: only categories matter
        RollResponse {
            mask_evs: None,
            optimal_mask: None,
            optimal_mask_ev: None,
            categories,
            optimal_category,
            optimal_category_ev,
            state_ev: optimal_category_ev,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0_tables;

    fn make_ctx() -> Box<YatzyContext> {
        let mut ctx = YatzyContext::new_boxed();
        phase0_tables::precompute_lookup_tables(&mut ctx);
        ctx
    }

    #[test]
    fn test_fat_response_with_rerolls() {
        let ctx = make_ctx();
        let dice = [1, 2, 3, 4, 5];
        let resp = compute_roll_response(&ctx, 0, 0, &dice, 2);

        // Should have mask EVs since rerolls > 0
        assert!(resp.mask_evs.is_some());
        assert!(resp.optimal_mask.is_some());
        assert!(resp.optimal_mask_ev.is_some());

        // All 15 categories should be available
        for cat in &resp.categories {
            assert!(cat.available);
        }

        // mask_evs[0] (keep all) should equal the best category EV
        let mask_evs = resp.mask_evs.unwrap();
        assert!(mask_evs[0].is_finite());

        // Optimal mask EV should be >= keep-all EV
        assert!(resp.optimal_mask_ev.unwrap() >= mask_evs[0] - 1e-9);
    }

    #[test]
    fn test_fat_response_no_rerolls() {
        let ctx = make_ctx();
        let dice = [6, 6, 6, 6, 6];
        let resp = compute_roll_response(&ctx, 0, 0, &dice, 0);

        // Should NOT have mask EVs
        assert!(resp.mask_evs.is_none());
        assert!(resp.optimal_mask.is_none());
        assert!(resp.optimal_mask_ev.is_none());

        // All categories available, Yatzy should score 50
        let yatzy = &resp.categories[CATEGORY_YATZY];
        assert!(yatzy.available);
        assert_eq!(yatzy.score, 50);
    }

    #[test]
    fn test_fat_response_scored_categories() {
        let ctx = make_ctx();
        let scored = (1 << CATEGORY_ONES) | (1 << CATEGORY_TWOS);
        let dice = [3, 3, 4, 5, 6];
        let resp = compute_roll_response(&ctx, 5, scored, &dice, 1);

        // Ones and Twos should be unavailable
        assert!(!resp.categories[CATEGORY_ONES].available);
        assert!(!resp.categories[CATEGORY_TWOS].available);
        assert!(resp.categories[CATEGORY_ONES].ev_if_scored.is_infinite());

        // Remaining categories should be available
        for c in 2..CATEGORY_COUNT {
            assert!(resp.categories[c].available);
        }
    }

    #[test]
    fn test_fat_response_category_scores() {
        let ctx = make_ctx();
        let dice = [1, 2, 3, 4, 5];
        let resp = compute_roll_response(&ctx, 0, 0, &dice, 0);

        // Verify known scores
        assert_eq!(resp.categories[CATEGORY_ONES].score, 1);
        assert_eq!(resp.categories[CATEGORY_TWOS].score, 2);
        assert_eq!(resp.categories[CATEGORY_SMALL_STRAIGHT].score, 15);
        assert_eq!(resp.categories[CATEGORY_LARGE_STRAIGHT].score, 0);
        assert_eq!(resp.categories[CATEGORY_YATZY].score, 0);
        assert_eq!(resp.categories[CATEGORY_CHANCE].score, 15);
    }

    #[test]
    fn test_fat_response_optimal_category_yatzy() {
        let ctx = make_ctx();
        // Only Yatzy + Chance left, roll all sixes
        let all_scored = (1 << CATEGORY_COUNT) - 1;
        let scored = all_scored ^ (1 << CATEGORY_YATZY) ^ (1 << CATEGORY_CHANCE);
        let dice = [6, 6, 6, 6, 6];
        let resp = compute_roll_response(&ctx, 0, scored, &dice, 0);

        assert_eq!(resp.optimal_category, CATEGORY_YATZY as i32);
    }
}
