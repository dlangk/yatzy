//! Validate precomputed state values from bin files.
//!
//! Loads the full DP table from data/all_states.bin and checks that
//! optimal actions and expected values are sensible for hand-verifiable
//! game states.
//!
//! Requires: YATZY_BASE_PATH set to the backend directory (or run from there).

use yatzy::api_computations::*;
use yatzy::constants::*;
use yatzy::dice_mechanics::sort_dice_set;
use yatzy::phase0_tables;
use yatzy::storage::{file_exists, load_all_state_values};
use yatzy::types::{YatzyContext, YatzyState};
use yatzy::widget_solver::compute_best_scoring_value_for_dice_set;

const ALL_SCORED: i32 = (1 << CATEGORY_COUNT) - 1;

fn only(cat: usize) -> i32 {
    ALL_SCORED ^ (1 << cat)
}

fn ev(ctx: &YatzyContext, up: i32, scored: i32) -> f64 {
    ctx.state_values.as_slice()[state_index(up as usize, scored as usize)] as f64
}

fn setup() -> Option<Box<YatzyContext>> {
    // Set working directory
    let base_path = std::env::var("YATZY_BASE_PATH").unwrap_or_else(|_| ".".to_string());
    let _ = std::env::set_current_dir(&base_path);

    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);

    if !load_all_state_values(&mut ctx, "data/all_states.bin") {
        return None;
    }
    Some(ctx)
}

#[test]
fn test_terminal_states() {
    let ctx = match setup() {
        Some(c) => c,
        None => {
            eprintln!("Skipping test: all_states.bin not found");
            return;
        }
    };

    assert!((ev(&ctx, 63, ALL_SCORED) - 50.0).abs() < 1e-9);
    assert!((ev(&ctx, 42, ALL_SCORED)).abs() < 1e-9);
    assert!((ev(&ctx, 0, ALL_SCORED)).abs() < 1e-9);
    assert!((ev(&ctx, 62, ALL_SCORED)).abs() < 1e-9);

    for up in 0..63 {
        assert!(ev(&ctx, up, ALL_SCORED).abs() < 1e-9);
    }
}

#[test]
fn test_game_start_ev() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    let e = ev(&ctx, 0, 0);
    assert!(e > 230.0 && e < 290.0, "EV(0,0) = {}", e);
    println!("EV(0,0) = {:.4}", e);
}

#[test]
fn test_one_category_remaining() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    assert!(ev(&ctx, 0, only(CATEGORY_ONES)) > 1.0 && ev(&ctx, 0, only(CATEGORY_ONES)) < 5.0);
    assert!(ev(&ctx, 0, only(CATEGORY_TWOS)) > 2.0 && ev(&ctx, 0, only(CATEGORY_TWOS)) < 10.0);
    assert!(ev(&ctx, 0, only(CATEGORY_THREES)) > 3.0 && ev(&ctx, 0, only(CATEGORY_THREES)) < 15.0);
    assert!(ev(&ctx, 0, only(CATEGORY_FOURS)) > 5.0 && ev(&ctx, 0, only(CATEGORY_FOURS)) < 20.0);
    assert!(ev(&ctx, 0, only(CATEGORY_FIVES)) > 6.0 && ev(&ctx, 0, only(CATEGORY_FIVES)) < 25.0);
    assert!(ev(&ctx, 0, only(CATEGORY_SIXES)) > 8.0 && ev(&ctx, 0, only(CATEGORY_SIXES)) < 20.0);
    assert!(
        ev(&ctx, 0, only(CATEGORY_ONE_PAIR)) > 6.0 && ev(&ctx, 0, only(CATEGORY_ONE_PAIR)) < 12.0
    );
    assert!(
        ev(&ctx, 0, only(CATEGORY_TWO_PAIRS)) > 10.0
            && ev(&ctx, 0, only(CATEGORY_TWO_PAIRS)) < 22.0
    );
    assert!(
        ev(&ctx, 0, only(CATEGORY_THREE_OF_A_KIND)) > 5.0
            && ev(&ctx, 0, only(CATEGORY_THREE_OF_A_KIND)) < 18.0
    );
    assert!(
        ev(&ctx, 0, only(CATEGORY_FOUR_OF_A_KIND)) > 3.0
            && ev(&ctx, 0, only(CATEGORY_FOUR_OF_A_KIND)) < 18.0
    );
    assert!(
        ev(&ctx, 0, only(CATEGORY_SMALL_STRAIGHT)) > 1.0
            && ev(&ctx, 0, only(CATEGORY_SMALL_STRAIGHT)) < 8.0
    );
    assert!(
        ev(&ctx, 0, only(CATEGORY_LARGE_STRAIGHT)) > 1.0
            && ev(&ctx, 0, only(CATEGORY_LARGE_STRAIGHT)) < 8.0
    );
    assert!(
        ev(&ctx, 0, only(CATEGORY_FULL_HOUSE)) > 5.0
            && ev(&ctx, 0, only(CATEGORY_FULL_HOUSE)) < 22.0
    );
    assert!(ev(&ctx, 0, only(CATEGORY_CHANCE)) > 20.0 && ev(&ctx, 0, only(CATEGORY_CHANCE)) < 30.0);
    assert!(ev(&ctx, 0, only(CATEGORY_YATZY)) > 1.0 && ev(&ctx, 0, only(CATEGORY_YATZY)) < 5.0);
}

#[test]
fn test_single_category_ordering() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    // Upper category EVs should increase with face value
    for cat in CATEGORY_ONES..CATEGORY_SIXES {
        let ev_lo = ev(&ctx, 0, only(cat));
        let ev_hi = ev(&ctx, 0, only(cat + 1));
        assert!(
            ev_lo < ev_hi,
            "cat {} EV={} >= cat {} EV={}",
            cat,
            ev_lo,
            cat + 1,
            ev_hi
        );
    }

    // Chance > Sixes
    assert!(ev(&ctx, 0, only(CATEGORY_CHANCE)) > ev(&ctx, 0, only(CATEGORY_SIXES)));

    // Yatzy < Sixes
    assert!(ev(&ctx, 0, only(CATEGORY_YATZY)) < ev(&ctx, 0, only(CATEGORY_SIXES)));

    // Chance > straights
    let ev_chance = ev(&ctx, 0, only(CATEGORY_CHANCE));
    assert!(ev_chance > ev(&ctx, 0, only(CATEGORY_SMALL_STRAIGHT)));
    assert!(ev_chance > ev(&ctx, 0, only(CATEGORY_LARGE_STRAIGHT)));

    // One Pair > Small Straight
    assert!(ev(&ctx, 0, only(CATEGORY_ONE_PAIR)) > ev(&ctx, 0, only(CATEGORY_SMALL_STRAIGHT)));
}

#[test]
fn test_optimal_category_choice() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    // [6,6,6,6,6] with Yatzy + Chance + Sixes open -> pick Yatzy
    {
        let scored =
            ALL_SCORED ^ (1 << CATEGORY_YATZY) ^ (1 << CATEGORY_CHANCE) ^ (1 << CATEGORY_SIXES);
        let dice = [6, 6, 6, 6, 6];
        let mut best_ev = 0.0;
        let cat = choose_best_category_no_rerolls(&ctx, 0, scored, &dice, &mut best_ev);
        assert_eq!(cat, CATEGORY_YATZY as i32);
    }

    // [1,1,1,1,1] with Yatzy + Ones + Chance open -> pick Yatzy
    {
        let scored =
            ALL_SCORED ^ (1 << CATEGORY_YATZY) ^ (1 << CATEGORY_ONES) ^ (1 << CATEGORY_CHANCE);
        let dice = [1, 1, 1, 1, 1];
        let mut best_ev = 0.0;
        let cat = choose_best_category_no_rerolls(&ctx, 0, scored, &dice, &mut best_ev);
        assert_eq!(cat, CATEGORY_YATZY as i32);
    }

    // [1,2,3,4,5] with Small Straight + Ones open -> pick Small Straight
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_SMALL_STRAIGHT) ^ (1 << CATEGORY_ONES);
        let dice = [1, 2, 3, 4, 5];
        let mut best_ev = 0.0;
        let cat = choose_best_category_no_rerolls(&ctx, 0, scored, &dice, &mut best_ev);
        assert_eq!(cat, CATEGORY_SMALL_STRAIGHT as i32);
    }

    // [2,3,4,5,6] with Large Straight + Small Straight open -> pick Large
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_LARGE_STRAIGHT) ^ (1 << CATEGORY_SMALL_STRAIGHT);
        let dice = [2, 3, 4, 5, 6];
        let mut best_ev = 0.0;
        let cat = choose_best_category_no_rerolls(&ctx, 0, scored, &dice, &mut best_ev);
        assert_eq!(cat, CATEGORY_LARGE_STRAIGHT as i32);
    }

    // [3,3,3,4,4] with Full House + Three of Kind open -> pick Full House
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_FULL_HOUSE) ^ (1 << CATEGORY_THREE_OF_A_KIND);
        let mut dice = [3, 3, 3, 4, 4];
        sort_dice_set(&mut dice);
        let mut best_ev = 0.0;
        let cat = choose_best_category_no_rerolls(&ctx, 0, scored, &dice, &mut best_ev);
        assert_eq!(cat, CATEGORY_FULL_HOUSE as i32);
    }

    // [1,5,5,6,6] with Two Pairs + One Pair open -> pick Two Pairs
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_TWO_PAIRS) ^ (1 << CATEGORY_ONE_PAIR);
        let dice = [1, 5, 5, 6, 6];
        let mut best_ev = 0.0;
        let cat = choose_best_category_no_rerolls(&ctx, 0, scored, &dice, &mut best_ev);
        assert_eq!(cat, CATEGORY_TWO_PAIRS as i32);
    }

    // [5,6,6,6,6] with 4oaK + 3oaK open -> pick 4oaK
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_FOUR_OF_A_KIND) ^ (1 << CATEGORY_THREE_OF_A_KIND);
        let dice = [5, 6, 6, 6, 6];
        let mut best_ev = 0.0;
        let cat = choose_best_category_no_rerolls(&ctx, 0, scored, &dice, &mut best_ev);
        assert_eq!(cat, CATEGORY_FOUR_OF_A_KIND as i32);
    }

    // [2,3,4,5,6] with Large Straight + Chance open -> pick Large Straight
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_LARGE_STRAIGHT) ^ (1 << CATEGORY_CHANCE);
        let dice = [2, 3, 4, 5, 6];
        let mut best_ev = 0.0;
        let cat = choose_best_category_no_rerolls(&ctx, 0, scored, &dice, &mut best_ev);
        assert_eq!(cat, CATEGORY_LARGE_STRAIGHT as i32);
    }
}

#[test]
fn test_optimal_reroll() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    // [1,6,6,6,6] with only Yatzy left -> reroll the 1 (mask=1)
    {
        let dice = [1, 6, 6, 6, 6];
        let mut best_mask = 0;
        let mut best_ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            0,
            only(CATEGORY_YATZY),
            &dice,
            2,
            &mut best_mask,
            &mut best_ev,
        );
        assert_eq!(best_mask, 1);
    }

    // [6,6,6,6,6] with Yatzy left -> keep all
    {
        let dice = [6, 6, 6, 6, 6];
        let mut best_mask = 0;
        let mut best_ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            0,
            only(CATEGORY_YATZY),
            &dice,
            2,
            &mut best_mask,
            &mut best_ev,
        );
        assert_eq!(best_mask, 0);
    }

    // [1,1,1,1,1] with only Sixes left -> reroll all (mask=31)
    {
        let dice = [1, 1, 1, 1, 1];
        let mut best_mask = 0;
        let mut best_ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            0,
            only(CATEGORY_SIXES),
            &dice,
            2,
            &mut best_mask,
            &mut best_ev,
        );
        assert_eq!(best_mask, 31);
    }

    // [1,2,3,4,5] with only Small Straight left -> keep all
    {
        let dice = [1, 2, 3, 4, 5];
        let mut best_mask = 0;
        let mut best_ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            0,
            only(CATEGORY_SMALL_STRAIGHT),
            &dice,
            2,
            &mut best_mask,
            &mut best_ev,
        );
        assert_eq!(best_mask, 0);
    }

    // [2,3,4,5,6] with only Large Straight left -> keep all
    {
        let dice = [2, 3, 4, 5, 6];
        let mut best_mask = 0;
        let mut best_ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            0,
            only(CATEGORY_LARGE_STRAIGHT),
            &dice,
            2,
            &mut best_mask,
            &mut best_ev,
        );
        assert_eq!(best_mask, 0);
    }

    // [1,1,1,1,2] with only Ones left -> reroll the 2 (mask=16)
    {
        let dice = [1, 1, 1, 1, 2];
        let mut best_mask = 0;
        let mut best_ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            0,
            only(CATEGORY_ONES),
            &dice,
            2,
            &mut best_mask,
            &mut best_ev,
        );
        assert_eq!(best_mask, 16);
    }

    // [4,5,6,6,6] with only Sixes left -> reroll 4,5 (mask=3)
    {
        let dice = [4, 5, 6, 6, 6];
        let mut best_mask = 0;
        let mut best_ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            0,
            only(CATEGORY_SIXES),
            &dice,
            2,
            &mut best_mask,
            &mut best_ev,
        );
        assert_eq!(best_mask, 3);
    }

    // [3,3,3,4,4] with only Full House left -> keep all
    {
        let dice = [3, 3, 3, 4, 4];
        let mut best_mask = 0;
        let mut best_ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            0,
            only(CATEGORY_FULL_HOUSE),
            &dice,
            2,
            &mut best_mask,
            &mut best_ev,
        );
        assert_eq!(best_mask, 0);
    }

    // [5,5,5,5,5] with only Yatzy left -> keep all
    {
        let dice = [5, 5, 5, 5, 5];
        let mut best_mask = 0;
        let mut best_ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            0,
            only(CATEGORY_YATZY),
            &dice,
            2,
            &mut best_mask,
            &mut best_ev,
        );
        assert_eq!(best_mask, 0);
    }

    // [1,2,3,4,6] with only Small Straight left -> reroll the 6 (mask=16)
    {
        let dice = [1, 2, 3, 4, 6];
        let mut best_mask = 0;
        let mut best_ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            0,
            only(CATEGORY_SMALL_STRAIGHT),
            &dice,
            2,
            &mut best_mask,
            &mut best_ev,
        );
        assert_eq!(best_mask, 16);
    }

    // [1,3,4,5,6] with only Large Straight left -> reroll 1 (mask=1)
    {
        let dice = [1, 3, 4, 5, 6];
        let mut best_mask = 0;
        let mut best_ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            0,
            only(CATEGORY_LARGE_STRAIGHT),
            &dice,
            2,
            &mut best_mask,
            &mut best_ev,
        );
        assert_eq!(best_mask, 1);
    }
}

#[test]
fn test_reroll_ev_bounds() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    // [6,6,6,6,6] only Yatzy -> EV = 50
    {
        let dice = [6, 6, 6, 6, 6];
        let mut best_mask = 0;
        let mut best_ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            0,
            only(CATEGORY_YATZY),
            &dice,
            2,
            &mut best_mask,
            &mut best_ev,
        );
        assert!((best_ev - 50.0).abs() < 1e-9);
    }

    // [1,2,3,4,5] only Sm Straight -> EV = 15
    {
        let dice = [1, 2, 3, 4, 5];
        let mut best_mask = 0;
        let mut best_ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            0,
            only(CATEGORY_SMALL_STRAIGHT),
            &dice,
            2,
            &mut best_mask,
            &mut best_ev,
        );
        assert!((best_ev - 15.0).abs() < 1e-9);
    }

    // [2,3,4,5,6] only Lg Straight -> EV = 20
    {
        let dice = [2, 3, 4, 5, 6];
        let mut best_mask = 0;
        let mut best_ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            0,
            only(CATEGORY_LARGE_STRAIGHT),
            &dice,
            2,
            &mut best_mask,
            &mut best_ev,
        );
        assert!((best_ev - 20.0).abs() < 1e-9);
    }

    // [6,6,6,6,6] only Chance -> EV = 30
    {
        let dice = [6, 6, 6, 6, 6];
        let mut best_mask = 0;
        let mut best_ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            0,
            only(CATEGORY_CHANCE),
            &dice,
            2,
            &mut best_mask,
            &mut best_ev,
        );
        assert!((best_ev - 30.0).abs() < 1e-9);
    }

    // [6,6,6,6,6] only Sixes -> EV = 30
    {
        let dice = [6, 6, 6, 6, 6];
        let mut best_mask = 0;
        let mut best_ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            0,
            only(CATEGORY_SIXES),
            &dice,
            2,
            &mut best_mask,
            &mut best_ev,
        );
        assert!((best_ev - 30.0).abs() < 1e-9);
    }

    // [5,5,6,6,6] only Full House -> EV = 28
    {
        let dice = [5, 5, 6, 6, 6];
        let mut best_mask = 0;
        let mut best_ev = 0.0;
        compute_best_reroll_strategy(
            &ctx,
            0,
            only(CATEGORY_FULL_HOUSE),
            &dice,
            2,
            &mut best_mask,
            &mut best_ev,
        );
        assert!((best_ev - 28.0).abs() < 1e-9);
    }
}

#[test]
fn test_monotonicity() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    let ev_0 = ev(&ctx, 0, 0);
    let ev_1 = ev(&ctx, 0, 1);
    let ev_2 = ev(&ctx, 0, 3);
    assert!(ev_0 > ev_1);
    assert!(ev_1 > ev_2);

    // Upper score proximity
    let sixes_scored = 0x20;
    assert!(ev(&ctx, 18, sixes_scored) > ev(&ctx, 0, sixes_scored));

    let all_upper = 0x3F;
    let ev_63 = ev(&ctx, 63, all_upper);
    let ev_0_all = ev(&ctx, 0, all_upper);
    assert!(ev_63 > ev_0_all);
    let bonus_diff = ev_63 - ev_0_all;
    assert!(bonus_diff > 5.0 && bonus_diff <= 50.0);
}

#[test]
fn test_monotonicity_chain() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    let mut scored = 0i32;
    let mut prev_ev = ev(&ctx, 0, 0);
    for cat in 0..CATEGORY_COUNT {
        scored |= 1 << cat;
        let e = ev(&ctx, 0, scored);
        assert!(
            e < prev_ev,
            "Monotonicity break at cat {}: {} >= {}",
            cat,
            e,
            prev_ev
        );
        prev_ev = e;
    }
}

#[test]
fn test_two_categories() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    let yatzy_chance = ALL_SCORED ^ (1 << CATEGORY_YATZY) ^ (1 << CATEGORY_CHANCE);
    let ev_both = ev(&ctx, 0, yatzy_chance);
    let ev_y = ev(&ctx, 0, only(CATEGORY_YATZY));
    let ev_c = ev(&ctx, 0, only(CATEGORY_CHANCE));
    assert!(ev_both > ev_y && ev_both > ev_c);
    assert!((ev_both - (ev_y + ev_c)).abs() < 5.0);

    // Full House + Two Pairs
    let fh_2p = ALL_SCORED ^ (1 << CATEGORY_FULL_HOUSE) ^ (1 << CATEGORY_TWO_PAIRS);
    assert!(ev(&ctx, 0, fh_2p) > ev(&ctx, 0, only(CATEGORY_FULL_HOUSE)));
    assert!(ev(&ctx, 0, fh_2p) > ev(&ctx, 0, only(CATEGORY_TWO_PAIRS)));

    // 3oaK + 4oaK
    let oak_34 = ALL_SCORED ^ (1 << CATEGORY_THREE_OF_A_KIND) ^ (1 << CATEGORY_FOUR_OF_A_KIND);
    assert!(ev(&ctx, 0, oak_34) > ev(&ctx, 0, only(CATEGORY_THREE_OF_A_KIND)));
    assert!(ev(&ctx, 0, oak_34) > ev(&ctx, 0, only(CATEGORY_FOUR_OF_A_KIND)));

    // Small Straight + Large Straight
    let str_both = ALL_SCORED ^ (1 << CATEGORY_SMALL_STRAIGHT) ^ (1 << CATEGORY_LARGE_STRAIGHT);
    assert!(ev(&ctx, 0, str_both) > ev(&ctx, 0, only(CATEGORY_SMALL_STRAIGHT)));
    assert!(ev(&ctx, 0, str_both) > ev(&ctx, 0, only(CATEGORY_LARGE_STRAIGHT)));
}

#[test]
fn test_ev_positivity() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    for scored in 0..(1u32 << CATEGORY_COUNT) {
        let upper_mask = (scored & 0x3F) as usize;
        for up in 0..=63 {
            if !ctx.reachable[upper_mask][up] {
                continue;
            }
            let e = ev(&ctx, up as i32, scored as i32);
            assert!(
                e >= -1e-9,
                "Negative EV at up={} scored=0x{:x}: {}",
                up,
                scored,
                e
            );
        }
    }
}

#[test]
fn test_unreachable_states_zero() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    for up in 1..=63 {
        assert!(
            ev(&ctx, up, 0).abs() < 1e-9,
            "Unreachable state (up={}, scored=0) has EV={}",
            up,
            ev(&ctx, up, 0)
        );
    }

    for up in 6..=63 {
        assert!(ev(&ctx, up, 0x01).abs() < 1e-9);
    }

    for up in 1..=63 {
        if up % 6 == 0 && up <= 30 {
            continue;
        }
        assert!(ev(&ctx, up, 0x20).abs() < 1e-9);
    }
}

#[test]
fn test_reroll_ev_consistency() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    let scored = (1 << CATEGORY_ONES) | (1 << CATEGORY_TWOS);
    let mut dice = [3, 3, 4, 5, 6];
    sort_dice_set(&mut dice);

    let mut best_mask = 0;
    let mut best_ev = 0.0;
    compute_best_reroll_strategy(&ctx, 5, scored, &dice, 2, &mut best_mask, &mut best_ev);

    let state = YatzyState {
        upper_score: 5,
        scored_categories: scored,
    };
    let keep_all_ev = compute_best_scoring_value_for_dice_set(&ctx, &state, &dice);
    assert!(best_ev >= keep_all_ev - 1e-9);
}

#[test]
fn test_upper_bonus_cliff() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    let all_upper = 0x3F;
    let cliff = ev(&ctx, 63, all_upper) - ev(&ctx, 0, all_upper);
    assert!(cliff > 5.0);

    let almost_all = ALL_SCORED ^ (1 << CATEGORY_YATZY);
    let diff = ev(&ctx, 63, almost_all) - ev(&ctx, 0, almost_all);
    assert!((diff - 50.0).abs() < 1e-4);
}

#[test]
fn test_exact_terminal_ev() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    assert!(ev(&ctx, 0, only(CATEGORY_CHANCE)) >= 5.0);
    assert!(ev(&ctx, 0, only(CATEGORY_FULL_HOUSE)) >= 0.0);

    let diff = ev(&ctx, 63, only(CATEGORY_YATZY)) - ev(&ctx, 0, only(CATEGORY_YATZY));
    assert!((diff - 50.0).abs() < 1e-4);
}

#[test]
fn test_score_category_consistency() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    // [6,6,6,6,6] only Yatzy -> score = 50
    {
        let dice = [6, 6, 6, 6, 6];
        let mut best_ev = 0.0;
        choose_best_category_no_rerolls(&ctx, 0, only(CATEGORY_YATZY), &dice, &mut best_ev);
        assert!((best_ev - 50.0).abs() < 1e-9);
    }

    // [1,2,3,4,6] only Small Straight -> no valid scoring
    {
        let dice = [1, 2, 3, 4, 6];
        let mut best_ev = 0.0;
        let cat = choose_best_category_no_rerolls(
            &ctx,
            0,
            only(CATEGORY_SMALL_STRAIGHT),
            &dice,
            &mut best_ev,
        );
        assert_eq!(cat, -1);
    }

    // [1,1,1,1,1] only Sixes -> no valid scoring
    {
        let dice = [1, 1, 1, 1, 1];
        let mut best_ev = 0.0;
        let cat =
            choose_best_category_no_rerolls(&ctx, 0, only(CATEGORY_SIXES), &dice, &mut best_ev);
        assert_eq!(cat, -1);
    }

    // [4,4,4,4,4] only Fours -> score = 20
    {
        let dice = [4, 4, 4, 4, 4];
        let mut best_ev = 0.0;
        choose_best_category_no_rerolls(&ctx, 0, only(CATEGORY_FOURS), &dice, &mut best_ev);
        assert!((best_ev - 20.0).abs() < 1e-9);
    }
}

#[test]
fn test_file_format() {
    let _ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    assert!(file_exists("data/all_states.bin"));

    let data = std::fs::read("data/all_states.bin").unwrap();
    assert!(data.len() >= 16);

    // Check header
    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let total_states = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let reserved = u32::from_le_bytes(data[12..16].try_into().unwrap());

    assert_eq!(magic, STATE_FILE_MAGIC);
    assert_eq!(version, STATE_FILE_VERSION);
    assert_eq!(total_states, NUM_STATES as u32);
    assert_eq!(reserved, 0);

    let expected_size = 16 + NUM_STATES * 4;
    assert_eq!(data.len(), expected_size);
}

#[test]
fn test_lower_category_upper_independence() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    let scored = 0x3F | (1 << CATEGORY_ONE_PAIR) | (1 << CATEGORY_THREE_OF_A_KIND);
    let ev_ref = ev(&ctx, 0, scored);
    for up in 1..63 {
        let e = ev(&ctx, up, scored);
        assert!(
            (e - ev_ref).abs() < 1e-4,
            "EV(up={}) = {} != EV(up=0) = {}",
            up,
            e,
            ev_ref
        );
    }
    let ev_63 = ev(&ctx, 63, scored);
    assert!((ev_63 - ev_ref - 50.0).abs() < 1e-4);
}

#[test]
fn test_lower_additivity() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    let ev_chance = ev(&ctx, 0, ALL_SCORED ^ (1 << CATEGORY_CHANCE));
    let ev_yatzy = ev(&ctx, 0, ALL_SCORED ^ (1 << CATEGORY_YATZY));
    let both = ALL_SCORED ^ (1 << CATEGORY_CHANCE) ^ (1 << CATEGORY_YATZY);
    let ev_both = ev(&ctx, 0, both);
    assert!((ev_both - (ev_chance + ev_yatzy)).abs() < 5.0);
}

#[test]
fn test_early_vs_late_game() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    let ten_scored = (1 << CATEGORY_ONES)
        | (1 << CATEGORY_TWOS)
        | (1 << CATEGORY_THREES)
        | (1 << CATEGORY_FOURS)
        | (1 << CATEGORY_FIVES)
        | (1 << CATEGORY_SIXES)
        | (1 << CATEGORY_ONE_PAIR)
        | (1 << CATEGORY_TWO_PAIRS)
        | (1 << CATEGORY_THREE_OF_A_KIND)
        | (1 << CATEGORY_FOUR_OF_A_KIND);

    let ev_start = ev(&ctx, 0, 0);
    let ev_late = ev(&ctx, 0, ten_scored);
    assert!(ev_start > ev_late);
    assert!(ev_late > 0.0);
    assert!(ev_late > 30.0 && ev_late < 80.0);
}
