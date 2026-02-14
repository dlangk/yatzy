//! Validate precomputed state values from bin files.
//!
//! Loads the full DP table from data/all_states.bin and checks that
//! optimal actions and expected values are sensible for hand-verifiable
//! game states.
//!
//! Requires: YATZY_BASE_PATH set to the backend directory (or run from there).

use yatzy::api_computations::*;
use yatzy::constants::*;
use yatzy::phase0_tables;
use yatzy::storage::{file_exists, load_all_state_values};
use yatzy::types::YatzyContext;

const ALL_SCORED: i32 = (1 << CATEGORY_COUNT) - 1;

fn only(cat: usize) -> i32 {
    ALL_SCORED ^ (1 << cat)
}

fn ev(ctx: &YatzyContext, up: i32, scored: i32) -> f64 {
    ctx.state_values.as_slice()[state_index(up as usize, scored as usize)] as f64
}

fn setup() -> Option<Box<YatzyContext>> {
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
        let resp = compute_roll_response(&ctx, 0, scored, &dice, 0);
        assert_eq!(resp.optimal_category, CATEGORY_YATZY as i32);
    }

    // [1,1,1,1,1] with Yatzy + Ones + Chance open -> pick Yatzy
    {
        let scored =
            ALL_SCORED ^ (1 << CATEGORY_YATZY) ^ (1 << CATEGORY_ONES) ^ (1 << CATEGORY_CHANCE);
        let dice = [1, 1, 1, 1, 1];
        let resp = compute_roll_response(&ctx, 0, scored, &dice, 0);
        assert_eq!(resp.optimal_category, CATEGORY_YATZY as i32);
    }

    // [1,2,3,4,5] with Small Straight + Ones open -> pick Small Straight
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_SMALL_STRAIGHT) ^ (1 << CATEGORY_ONES);
        let dice = [1, 2, 3, 4, 5];
        let resp = compute_roll_response(&ctx, 0, scored, &dice, 0);
        assert_eq!(resp.optimal_category, CATEGORY_SMALL_STRAIGHT as i32);
    }

    // [2,3,4,5,6] with Large Straight + Small Straight open -> pick Large
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_LARGE_STRAIGHT) ^ (1 << CATEGORY_SMALL_STRAIGHT);
        let dice = [2, 3, 4, 5, 6];
        let resp = compute_roll_response(&ctx, 0, scored, &dice, 0);
        assert_eq!(resp.optimal_category, CATEGORY_LARGE_STRAIGHT as i32);
    }

    // [3,3,3,4,4] with Full House + Three of Kind open -> pick Full House
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_FULL_HOUSE) ^ (1 << CATEGORY_THREE_OF_A_KIND);
        let dice = [3, 3, 3, 4, 4];
        let resp = compute_roll_response(&ctx, 0, scored, &dice, 0);
        assert_eq!(resp.optimal_category, CATEGORY_FULL_HOUSE as i32);
    }

    // [1,5,5,6,6] with Two Pairs + One Pair open -> pick Two Pairs
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_TWO_PAIRS) ^ (1 << CATEGORY_ONE_PAIR);
        let dice = [1, 5, 5, 6, 6];
        let resp = compute_roll_response(&ctx, 0, scored, &dice, 0);
        assert_eq!(resp.optimal_category, CATEGORY_TWO_PAIRS as i32);
    }

    // [5,6,6,6,6] with 4oaK + 3oaK open -> pick 4oaK
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_FOUR_OF_A_KIND) ^ (1 << CATEGORY_THREE_OF_A_KIND);
        let dice = [5, 6, 6, 6, 6];
        let resp = compute_roll_response(&ctx, 0, scored, &dice, 0);
        assert_eq!(resp.optimal_category, CATEGORY_FOUR_OF_A_KIND as i32);
    }

    // [2,3,4,5,6] with Large Straight + Chance open -> pick Large Straight
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_LARGE_STRAIGHT) ^ (1 << CATEGORY_CHANCE);
        let dice = [2, 3, 4, 5, 6];
        let resp = compute_roll_response(&ctx, 0, scored, &dice, 0);
        assert_eq!(resp.optimal_category, CATEGORY_LARGE_STRAIGHT as i32);
    }

    // Score-0 optimal: wasting a low-value category to preserve Chance for future rounds.
    // Chance has very high future EV (~23), while straights and Yatzy have low future EV (~3-5).

    // [1,2,3,4,6] Chance + SmStr open -> waste SmStr (score=0), preserve Chance
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_SMALL_STRAIGHT) ^ (1 << CATEGORY_CHANCE);
        let dice = [1, 2, 3, 4, 6];
        let resp = compute_roll_response(&ctx, 0, scored, &dice, 0);
        assert_eq!(resp.optimal_category, CATEGORY_SMALL_STRAIGHT as i32);
        assert_eq!(resp.categories[CATEGORY_SMALL_STRAIGHT].score, 0);
    }

    // [1,2,3,4,5] Chance + LgStr open -> waste LgStr (score=0), preserve Chance
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_LARGE_STRAIGHT) ^ (1 << CATEGORY_CHANCE);
        let dice = [1, 2, 3, 4, 5];
        let resp = compute_roll_response(&ctx, 0, scored, &dice, 0);
        assert_eq!(resp.optimal_category, CATEGORY_LARGE_STRAIGHT as i32);
        assert_eq!(resp.categories[CATEGORY_LARGE_STRAIGHT].score, 0);
    }

    // [1,1,1,1,2] Chance + Yatzy open -> waste Yatzy (score=0), preserve Chance
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_YATZY) ^ (1 << CATEGORY_CHANCE);
        let dice = [1, 1, 1, 1, 2];
        let resp = compute_roll_response(&ctx, 0, scored, &dice, 0);
        assert_eq!(resp.optimal_category, CATEGORY_YATZY as i32);
        assert_eq!(resp.categories[CATEGORY_YATZY].score, 0);
    }

    // [2,2,2,2,3] Chance + Yatzy open -> waste Yatzy (score=0), preserve Chance
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_YATZY) ^ (1 << CATEGORY_CHANCE);
        let dice = [2, 2, 2, 2, 3];
        let resp = compute_roll_response(&ctx, 0, scored, &dice, 0);
        assert_eq!(resp.optimal_category, CATEGORY_YATZY as i32);
        assert_eq!(resp.categories[CATEGORY_YATZY].score, 0);
    }

    // [1,1,1,1,1] Chance + LgStr open -> waste LgStr (score=0), preserve Chance
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_LARGE_STRAIGHT) ^ (1 << CATEGORY_CHANCE);
        let dice = [1, 1, 1, 1, 1];
        let resp = compute_roll_response(&ctx, 0, scored, &dice, 0);
        assert_eq!(resp.optimal_category, CATEGORY_LARGE_STRAIGHT as i32);
        assert_eq!(resp.categories[CATEGORY_LARGE_STRAIGHT].score, 0);
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
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_YATZY), &dice, 2);
        assert_eq!(resp.optimal_mask.unwrap(), 1);
    }

    // [6,6,6,6,6] with Yatzy left -> keep all
    {
        let dice = [6, 6, 6, 6, 6];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_YATZY), &dice, 2);
        assert_eq!(resp.optimal_mask.unwrap(), 0);
    }

    // [1,1,1,1,1] with only Sixes left -> reroll all (mask=31)
    {
        let dice = [1, 1, 1, 1, 1];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_SIXES), &dice, 2);
        assert_eq!(resp.optimal_mask.unwrap(), 31);
    }

    // [1,2,3,4,5] with only Small Straight left -> keep all
    {
        let dice = [1, 2, 3, 4, 5];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_SMALL_STRAIGHT), &dice, 2);
        assert_eq!(resp.optimal_mask.unwrap(), 0);
    }

    // [2,3,4,5,6] with only Large Straight left -> keep all
    {
        let dice = [2, 3, 4, 5, 6];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_LARGE_STRAIGHT), &dice, 2);
        assert_eq!(resp.optimal_mask.unwrap(), 0);
    }

    // [1,1,1,1,2] with only Ones left -> reroll the 2 (mask=16)
    {
        let dice = [1, 1, 1, 1, 2];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_ONES), &dice, 2);
        assert_eq!(resp.optimal_mask.unwrap(), 16);
    }

    // [4,5,6,6,6] with only Sixes left -> reroll 4,5 (mask=3)
    {
        let dice = [4, 5, 6, 6, 6];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_SIXES), &dice, 2);
        assert_eq!(resp.optimal_mask.unwrap(), 3);
    }

    // [3,3,3,4,4] with only Full House left -> keep all
    {
        let dice = [3, 3, 3, 4, 4];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_FULL_HOUSE), &dice, 2);
        assert_eq!(resp.optimal_mask.unwrap(), 0);
    }

    // [5,5,5,5,5] with only Yatzy left -> keep all
    {
        let dice = [5, 5, 5, 5, 5];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_YATZY), &dice, 2);
        assert_eq!(resp.optimal_mask.unwrap(), 0);
    }

    // [1,2,3,4,6] with only Small Straight left -> reroll the 6 (mask=16)
    {
        let dice = [1, 2, 3, 4, 6];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_SMALL_STRAIGHT), &dice, 2);
        assert_eq!(resp.optimal_mask.unwrap(), 16);
    }

    // [1,3,4,5,6] with only Large Straight left -> reroll 1 (mask=1)
    {
        let dice = [1, 3, 4, 5, 6];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_LARGE_STRAIGHT), &dice, 2);
        assert_eq!(resp.optimal_mask.unwrap(), 1);
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
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_YATZY), &dice, 2);
        assert!((resp.optimal_mask_ev.unwrap() - 50.0).abs() < 1e-9);
    }

    // [1,2,3,4,5] only Sm Straight -> EV = 15
    {
        let dice = [1, 2, 3, 4, 5];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_SMALL_STRAIGHT), &dice, 2);
        assert!((resp.optimal_mask_ev.unwrap() - 15.0).abs() < 1e-9);
    }

    // [2,3,4,5,6] only Lg Straight -> EV = 20
    {
        let dice = [2, 3, 4, 5, 6];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_LARGE_STRAIGHT), &dice, 2);
        assert!((resp.optimal_mask_ev.unwrap() - 20.0).abs() < 1e-9);
    }

    // [6,6,6,6,6] only Chance -> EV = 30
    {
        let dice = [6, 6, 6, 6, 6];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_CHANCE), &dice, 2);
        assert!((resp.optimal_mask_ev.unwrap() - 30.0).abs() < 1e-9);
    }

    // [6,6,6,6,6] only Sixes -> EV = 30
    {
        let dice = [6, 6, 6, 6, 6];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_SIXES), &dice, 2);
        assert!((resp.optimal_mask_ev.unwrap() - 30.0).abs() < 1e-9);
    }

    // [5,5,6,6,6] only Full House -> EV = 28
    {
        let dice = [5, 5, 6, 6, 6];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_FULL_HOUSE), &dice, 2);
        assert!((resp.optimal_mask_ev.unwrap() - 28.0).abs() < 1e-9);
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
    let dice = [3, 3, 4, 5, 6];

    let resp = compute_roll_response(&ctx, 5, scored, &dice, 2);

    // Optimal mask EV should be >= keep-all EV (mask=0)
    let mask_evs = resp.mask_evs.unwrap();
    let keep_all_ev = mask_evs[0];
    assert!(resp.optimal_mask_ev.unwrap() >= keep_all_ev - 1e-9);
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

    // [6,6,6,6,6] only Yatzy -> optimal category EV = 50
    {
        let dice = [6, 6, 6, 6, 6];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_YATZY), &dice, 0);
        assert!((resp.optimal_category_ev - 50.0).abs() < 1e-9);
    }

    // [1,2,3,4,6] only Small Straight -> score=0 but still the only available category
    {
        let dice = [1, 2, 3, 4, 6];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_SMALL_STRAIGHT), &dice, 0);
        assert_eq!(resp.optimal_category, CATEGORY_SMALL_STRAIGHT as i32);
    }

    // [1,1,1,1,1] only Sixes -> score=0 but still the only available category
    {
        let dice = [1, 1, 1, 1, 1];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_SIXES), &dice, 0);
        assert_eq!(resp.optimal_category, CATEGORY_SIXES as i32);
    }

    // [4,4,4,4,4] only Fours -> score = 20
    {
        let dice = [4, 4, 4, 4, 4];
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_FOURS), &dice, 0);
        assert!((resp.optimal_category_ev - 20.0).abs() < 1e-9);
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

#[test]
fn test_fat_response_all_categories_info() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    // Game start with [1,2,3,4,5]
    let dice = [1, 2, 3, 4, 5];
    let resp = compute_roll_response(&ctx, 0, 0, &dice, 2);

    // All 15 categories should be available
    for cat in &resp.categories {
        assert!(cat.available);
        assert!(cat.ev_if_scored.is_finite());
    }

    // Small straight should score 15
    assert_eq!(resp.categories[CATEGORY_SMALL_STRAIGHT].score, 15);
    // Large straight should score 0
    assert_eq!(resp.categories[CATEGORY_LARGE_STRAIGHT].score, 0);

    // Should have mask EVs
    assert!(resp.mask_evs.is_some());
    let mask_evs = resp.mask_evs.unwrap();
    // All mask EVs should be finite
    for m in 0..32 {
        assert!(mask_evs[m].is_finite());
    }
}

#[test]
fn test_fat_response_mask_ev_consistency() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    // For perfect dice (all sixes, only Yatzy left), keep-all should be optimal
    let dice = [6, 6, 6, 6, 6];
    let resp = compute_roll_response(&ctx, 0, only(CATEGORY_YATZY), &dice, 2);
    let mask_evs = resp.mask_evs.unwrap();

    // mask=0 (keep all) should have highest EV
    for mask in 1..32 {
        assert!(
            mask_evs[0] >= mask_evs[mask] - 1e-9,
            "mask {} has higher EV than keep-all: {} > {}",
            mask,
            mask_evs[mask],
            mask_evs[0]
        );
    }
}

/// Verify that optimal_category == argmax(ev_if_scored) among available categories
/// across a variety of game states and dice combinations.
#[test]
fn test_optimal_category_matches_max_ev() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    // Helper: assert the invariant for a given response
    fn assert_optimal_is_argmax(resp: &RollResponse, label: &str) {
        let best = resp
            .categories
            .iter()
            .filter(|c| c.available)
            .max_by(|a, b| a.ev_if_scored.partial_cmp(&b.ev_if_scored).unwrap())
            .unwrap();
        assert_eq!(
            resp.optimal_category, best.id as i32,
            "{}: optimal_category={} but argmax(ev_if_scored) is {} (ev={})",
            label, resp.optimal_category, best.id, best.ev_if_scored
        );
        assert!(
            (resp.optimal_category_ev - best.ev_if_scored).abs() < 1e-9,
            "{}: optimal_category_ev={} but max ev_if_scored={}",
            label,
            resp.optimal_category_ev,
            best.ev_if_scored
        );
    }

    // Case 1: All categories open, various dice, rerolls=0
    let test_dice: &[[i32; 5]] = &[
        [1, 2, 3, 4, 5],
        [1, 1, 1, 1, 1],
        [6, 6, 6, 6, 6],
        [1, 2, 4, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 3, 3, 4, 4],
        [1, 1, 2, 2, 3],
        [5, 5, 5, 5, 6],
    ];
    for dice in test_dice {
        let resp = compute_roll_response(&ctx, 0, 0, dice, 0);
        assert_optimal_is_argmax(&resp, &format!("all open, dice={:?}", dice));
    }

    // Case 2: Single category open, score=0 cases
    {
        // [1,2,3,4,6] only Small Straight open (score=0)
        let resp =
            compute_roll_response(&ctx, 0, only(CATEGORY_SMALL_STRAIGHT), &[1, 2, 3, 4, 6], 0);
        assert_optimal_is_argmax(&resp, "only SmStr, score=0");
    }
    {
        // [1,1,1,1,1] only Sixes open (score=0)
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_SIXES), &[1, 1, 1, 1, 1], 0);
        assert_optimal_is_argmax(&resp, "only Sixes, score=0");
    }
    {
        // [1,2,3,4,5] only Large Straight open (score=0)
        let resp =
            compute_roll_response(&ctx, 0, only(CATEGORY_LARGE_STRAIGHT), &[1, 2, 3, 4, 5], 0);
        assert_optimal_is_argmax(&resp, "only LgStr, score=0");
    }

    // Case 3: Two categories open, various upper scores
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_ONES) ^ (1 << CATEGORY_CHANCE);
        let resp = compute_roll_response(&ctx, 20, scored, &[1, 1, 3, 4, 5], 0);
        assert_optimal_is_argmax(&resp, "Ones+Chance open, up=20");
    }
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_YATZY) ^ (1 << CATEGORY_FULL_HOUSE);
        let resp = compute_roll_response(&ctx, 0, scored, &[3, 3, 3, 5, 5], 0);
        assert_optimal_is_argmax(&resp, "Yatzy+FH open");
    }

    // Case 4: Many categories open, mid-game
    {
        let scored = (1 << CATEGORY_ONES)
            | (1 << CATEGORY_TWOS)
            | (1 << CATEGORY_THREES)
            | (1 << CATEGORY_FOURS);
        let resp = compute_roll_response(&ctx, 12, scored, &[5, 5, 6, 6, 6], 0);
        assert_optimal_is_argmax(&resp, "mid-game, 11 cats open");
    }
}

/// For each of the 15 categories, test 3 dice+state combinations where that category
/// is the optimal choice (rerolls=0). Also tests score=0 optimal cases.
#[test]
fn test_optimal_category_per_category() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    // Helper: assert optimal category equals expected
    fn assert_optimal(resp: &RollResponse, expected: usize, label: &str) {
        assert_eq!(
            resp.optimal_category, expected as i32,
            "{}: expected {} but got {}",
            label, expected, resp.optimal_category
        );
    }

    // --- Ones (0) ---
    {
        // Only Ones open, dice score in Ones
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_ONES), &[1, 1, 1, 1, 2], 0);
        assert_optimal(&resp, CATEGORY_ONES, "Ones: only open");
    }
    {
        // Ones vs Twos: [1,1,1,2,3] -> Ones=3, Twos=2
        let scored = ALL_SCORED ^ (1 << CATEGORY_ONES) ^ (1 << CATEGORY_TWOS);
        let resp = compute_roll_response(&ctx, 0, scored, &[1, 1, 1, 2, 3], 0);
        // Whichever the DP says is optimal, it must match argmax(ev_if_scored)
        let best = resp
            .categories
            .iter()
            .filter(|c| c.available)
            .max_by(|a, b| a.ev_if_scored.partial_cmp(&b.ev_if_scored).unwrap())
            .unwrap();
        assert_eq!(resp.optimal_category, best.id as i32, "Ones vs Twos");
    }
    {
        // Only Ones open, score=0 dice
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_ONES), &[2, 3, 4, 5, 6], 0);
        assert_optimal(&resp, CATEGORY_ONES, "Ones: score=0, only open");
    }

    // --- Twos (1) ---
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_TWOS), &[2, 2, 2, 2, 3], 0);
        assert_optimal(&resp, CATEGORY_TWOS, "Twos: only open");
    }
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_TWOS) ^ (1 << CATEGORY_ONES);
        let resp = compute_roll_response(&ctx, 0, scored, &[2, 2, 2, 2, 1], 0);
        let best = resp
            .categories
            .iter()
            .filter(|c| c.available)
            .max_by(|a, b| a.ev_if_scored.partial_cmp(&b.ev_if_scored).unwrap())
            .unwrap();
        assert_eq!(resp.optimal_category, best.id as i32, "Twos vs Ones");
    }
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_TWOS), &[1, 3, 4, 5, 6], 0);
        assert_optimal(&resp, CATEGORY_TWOS, "Twos: score=0, only open");
    }

    // --- Threes (2) ---
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_THREES), &[3, 3, 3, 3, 4], 0);
        assert_optimal(&resp, CATEGORY_THREES, "Threes: only open");
    }
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_THREES) ^ (1 << CATEGORY_TWOS);
        let resp = compute_roll_response(&ctx, 0, scored, &[3, 3, 3, 2, 1], 0);
        let best = resp
            .categories
            .iter()
            .filter(|c| c.available)
            .max_by(|a, b| a.ev_if_scored.partial_cmp(&b.ev_if_scored).unwrap())
            .unwrap();
        assert_eq!(resp.optimal_category, best.id as i32, "Threes vs Twos");
    }
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_THREES), &[1, 2, 4, 5, 6], 0);
        assert_optimal(&resp, CATEGORY_THREES, "Threes: score=0, only open");
    }

    // --- Fours (3) ---
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_FOURS), &[4, 4, 4, 4, 5], 0);
        assert_optimal(&resp, CATEGORY_FOURS, "Fours: only open");
    }
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_FOURS) ^ (1 << CATEGORY_THREES);
        let resp = compute_roll_response(&ctx, 0, scored, &[4, 4, 4, 3, 1], 0);
        let best = resp
            .categories
            .iter()
            .filter(|c| c.available)
            .max_by(|a, b| a.ev_if_scored.partial_cmp(&b.ev_if_scored).unwrap())
            .unwrap();
        assert_eq!(resp.optimal_category, best.id as i32, "Fours vs Threes");
    }
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_FOURS), &[1, 2, 3, 5, 6], 0);
        assert_optimal(&resp, CATEGORY_FOURS, "Fours: score=0, only open");
    }

    // --- Fives (4) ---
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_FIVES), &[5, 5, 5, 5, 6], 0);
        assert_optimal(&resp, CATEGORY_FIVES, "Fives: only open");
    }
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_FIVES) ^ (1 << CATEGORY_FOURS);
        let resp = compute_roll_response(&ctx, 0, scored, &[5, 5, 5, 4, 1], 0);
        let best = resp
            .categories
            .iter()
            .filter(|c| c.available)
            .max_by(|a, b| a.ev_if_scored.partial_cmp(&b.ev_if_scored).unwrap())
            .unwrap();
        assert_eq!(resp.optimal_category, best.id as i32, "Fives vs Fours");
    }
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_FIVES), &[1, 2, 3, 4, 6], 0);
        assert_optimal(&resp, CATEGORY_FIVES, "Fives: score=0, only open");
    }

    // --- Sixes (5) ---
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_SIXES), &[6, 6, 6, 6, 5], 0);
        assert_optimal(&resp, CATEGORY_SIXES, "Sixes: only open");
    }
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_SIXES) ^ (1 << CATEGORY_FIVES);
        let resp = compute_roll_response(&ctx, 0, scored, &[6, 6, 6, 5, 1], 0);
        let best = resp
            .categories
            .iter()
            .filter(|c| c.available)
            .max_by(|a, b| a.ev_if_scored.partial_cmp(&b.ev_if_scored).unwrap())
            .unwrap();
        assert_eq!(resp.optimal_category, best.id as i32, "Sixes vs Fives");
    }
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_SIXES), &[1, 1, 1, 1, 1], 0);
        assert_optimal(&resp, CATEGORY_SIXES, "Sixes: score=0, only open");
    }

    // --- One Pair (6) ---
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_ONE_PAIR), &[6, 6, 1, 2, 3], 0);
        assert_optimal(&resp, CATEGORY_ONE_PAIR, "OnePair: only open");
    }
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_ONE_PAIR) ^ (1 << CATEGORY_ONES);
        let resp = compute_roll_response(&ctx, 0, scored, &[6, 6, 1, 2, 3], 0);
        let best = resp
            .categories
            .iter()
            .filter(|c| c.available)
            .max_by(|a, b| a.ev_if_scored.partial_cmp(&b.ev_if_scored).unwrap())
            .unwrap();
        assert_eq!(resp.optimal_category, best.id as i32, "OnePair vs Ones");
    }
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_ONE_PAIR), &[1, 2, 3, 4, 6], 0);
        assert_optimal(&resp, CATEGORY_ONE_PAIR, "OnePair: score=0, only open");
    }

    // --- Two Pairs (7) ---
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_TWO_PAIRS), &[5, 5, 6, 6, 1], 0);
        assert_optimal(&resp, CATEGORY_TWO_PAIRS, "TwoPairs: only open");
    }
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_TWO_PAIRS) ^ (1 << CATEGORY_ONE_PAIR);
        let resp = compute_roll_response(&ctx, 0, scored, &[5, 5, 6, 6, 1], 0);
        assert_eq!(
            resp.optimal_category, CATEGORY_TWO_PAIRS as i32,
            "TwoPairs vs OnePair"
        );
    }
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_TWO_PAIRS), &[1, 2, 3, 4, 6], 0);
        assert_optimal(&resp, CATEGORY_TWO_PAIRS, "TwoPairs: score=0, only open");
    }

    // --- Three of a Kind (8) ---
    {
        let resp =
            compute_roll_response(&ctx, 0, only(CATEGORY_THREE_OF_A_KIND), &[6, 6, 6, 1, 2], 0);
        assert_optimal(&resp, CATEGORY_THREE_OF_A_KIND, "3oaK: only open");
    }
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_THREE_OF_A_KIND) ^ (1 << CATEGORY_ONE_PAIR);
        let resp = compute_roll_response(&ctx, 0, scored, &[6, 6, 6, 1, 2], 0);
        let best = resp
            .categories
            .iter()
            .filter(|c| c.available)
            .max_by(|a, b| a.ev_if_scored.partial_cmp(&b.ev_if_scored).unwrap())
            .unwrap();
        assert_eq!(resp.optimal_category, best.id as i32, "3oaK vs OnePair");
    }
    {
        let resp =
            compute_roll_response(&ctx, 0, only(CATEGORY_THREE_OF_A_KIND), &[1, 2, 3, 4, 6], 0);
        assert_optimal(&resp, CATEGORY_THREE_OF_A_KIND, "3oaK: score=0, only open");
    }

    // --- Four of a Kind (9) ---
    {
        let resp =
            compute_roll_response(&ctx, 0, only(CATEGORY_FOUR_OF_A_KIND), &[6, 6, 6, 6, 1], 0);
        assert_optimal(&resp, CATEGORY_FOUR_OF_A_KIND, "4oaK: only open");
    }
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_FOUR_OF_A_KIND) ^ (1 << CATEGORY_THREE_OF_A_KIND);
        let resp = compute_roll_response(&ctx, 0, scored, &[6, 6, 6, 6, 1], 0);
        assert_eq!(
            resp.optimal_category, CATEGORY_FOUR_OF_A_KIND as i32,
            "4oaK vs 3oaK"
        );
    }
    {
        let resp =
            compute_roll_response(&ctx, 0, only(CATEGORY_FOUR_OF_A_KIND), &[1, 2, 3, 4, 6], 0);
        assert_optimal(&resp, CATEGORY_FOUR_OF_A_KIND, "4oaK: score=0, only open");
    }

    // --- Small Straight (10) ---
    {
        let resp =
            compute_roll_response(&ctx, 0, only(CATEGORY_SMALL_STRAIGHT), &[1, 2, 3, 4, 5], 0);
        assert_optimal(&resp, CATEGORY_SMALL_STRAIGHT, "SmStr: only open");
    }
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_SMALL_STRAIGHT) ^ (1 << CATEGORY_ONES);
        let resp = compute_roll_response(&ctx, 0, scored, &[1, 2, 3, 4, 5], 0);
        assert_eq!(
            resp.optimal_category, CATEGORY_SMALL_STRAIGHT as i32,
            "SmStr vs Ones"
        );
    }
    {
        // Score=0 case: missing 5
        let resp =
            compute_roll_response(&ctx, 0, only(CATEGORY_SMALL_STRAIGHT), &[1, 2, 3, 4, 6], 0);
        assert_optimal(&resp, CATEGORY_SMALL_STRAIGHT, "SmStr: score=0, only open");
    }

    // --- Large Straight (11) ---
    {
        let resp =
            compute_roll_response(&ctx, 0, only(CATEGORY_LARGE_STRAIGHT), &[2, 3, 4, 5, 6], 0);
        assert_optimal(&resp, CATEGORY_LARGE_STRAIGHT, "LgStr: only open");
    }
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_LARGE_STRAIGHT) ^ (1 << CATEGORY_SMALL_STRAIGHT);
        let resp = compute_roll_response(&ctx, 0, scored, &[2, 3, 4, 5, 6], 0);
        assert_eq!(
            resp.optimal_category, CATEGORY_LARGE_STRAIGHT as i32,
            "LgStr vs SmStr"
        );
    }
    {
        // Score=0 case
        let resp =
            compute_roll_response(&ctx, 0, only(CATEGORY_LARGE_STRAIGHT), &[1, 2, 3, 4, 5], 0);
        assert_optimal(&resp, CATEGORY_LARGE_STRAIGHT, "LgStr: score=0, only open");
    }

    // --- Full House (12) ---
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_FULL_HOUSE), &[5, 5, 6, 6, 6], 0);
        assert_optimal(&resp, CATEGORY_FULL_HOUSE, "FH: only open");
    }
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_FULL_HOUSE) ^ (1 << CATEGORY_THREE_OF_A_KIND);
        let resp = compute_roll_response(&ctx, 0, scored, &[5, 5, 6, 6, 6], 0);
        assert_eq!(
            resp.optimal_category, CATEGORY_FULL_HOUSE as i32,
            "FH vs 3oaK"
        );
    }
    {
        // Score=0 case
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_FULL_HOUSE), &[1, 2, 3, 4, 5], 0);
        assert_optimal(&resp, CATEGORY_FULL_HOUSE, "FH: score=0, only open");
    }

    // --- Chance (13) ---
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_CHANCE), &[6, 6, 6, 6, 6], 0);
        assert_optimal(&resp, CATEGORY_CHANCE, "Chance: only open");
    }
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_CHANCE) ^ (1 << CATEGORY_SIXES);
        let resp = compute_roll_response(&ctx, 0, scored, &[6, 6, 6, 6, 6], 0);
        let best = resp
            .categories
            .iter()
            .filter(|c| c.available)
            .max_by(|a, b| a.ev_if_scored.partial_cmp(&b.ev_if_scored).unwrap())
            .unwrap();
        assert_eq!(resp.optimal_category, best.id as i32, "Chance vs Sixes");
    }
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_CHANCE), &[1, 1, 1, 1, 1], 0);
        assert_optimal(&resp, CATEGORY_CHANCE, "Chance: low dice, only open");
    }

    // --- Yatzy (14) ---
    {
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_YATZY), &[6, 6, 6, 6, 6], 0);
        assert_optimal(&resp, CATEGORY_YATZY, "Yatzy: only open");
    }
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_YATZY) ^ (1 << CATEGORY_CHANCE);
        let resp = compute_roll_response(&ctx, 0, scored, &[6, 6, 6, 6, 6], 0);
        assert_eq!(
            resp.optimal_category, CATEGORY_YATZY as i32,
            "Yatzy vs Chance"
        );
    }
    {
        // Score=0 case: no Yatzy
        let resp = compute_roll_response(&ctx, 0, only(CATEGORY_YATZY), &[1, 2, 3, 4, 5], 0);
        assert_optimal(&resp, CATEGORY_YATZY, "Yatzy: score=0, only open");
    }

    // --- All categories open, score=0 optimal in some ---
    {
        let dice = [1, 2, 4, 4, 5];
        let resp = compute_roll_response(&ctx, 0, 0, &dice, 0);
        let best = resp
            .categories
            .iter()
            .filter(|c| c.available)
            .max_by(|a, b| a.ev_if_scored.partial_cmp(&b.ev_if_scored).unwrap())
            .unwrap();
        assert_eq!(
            resp.optimal_category, best.id as i32,
            "[1,2,4,4,5] all open"
        );
    }
}

/// Print ev_if_scored for interesting game states, and verify structural properties.
/// Run with `-- --nocapture` for visibility.
#[test]
fn test_ev_if_scored_values() {
    let ctx = match setup() {
        Some(c) => c,
        None => return,
    };

    // Late game: Chance + SmStr, dice that miss SmStr
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_SMALL_STRAIGHT) ^ (1 << CATEGORY_CHANCE);
        let resp = compute_roll_response(&ctx, 0, scored, &[1, 2, 3, 4, 6], 0);
        let ev_smstr = resp.categories[CATEGORY_SMALL_STRAIGHT].ev_if_scored;
        let ev_chance = resp.categories[CATEGORY_CHANCE].ev_if_scored;
        println!(
            "Chance+SmStr, [1,2,3,4,6]: SmStr ev={:.2} (score=0), Chance ev={:.2} (score=16)",
            ev_smstr, ev_chance
        );
        // SmStr (score=0) should have higher ev_if_scored than Chance (score=16)
        assert!(
            ev_smstr > ev_chance,
            "SmStr ev={} should beat Chance ev={}",
            ev_smstr,
            ev_chance
        );
    }

    // Late game: Chance + Yatzy, dice with no Yatzy
    {
        let scored = ALL_SCORED ^ (1 << CATEGORY_YATZY) ^ (1 << CATEGORY_CHANCE);
        let resp = compute_roll_response(&ctx, 0, scored, &[1, 1, 1, 1, 2], 0);
        let ev_yatzy = resp.categories[CATEGORY_YATZY].ev_if_scored;
        let ev_chance = resp.categories[CATEGORY_CHANCE].ev_if_scored;
        println!(
            "Chance+Yatzy, [1,1,1,1,2]: Yatzy ev={:.2} (score=0), Chance ev={:.2} (score=6)",
            ev_yatzy, ev_chance
        );
        assert!(
            ev_yatzy > ev_chance,
            "Yatzy ev={} should beat Chance ev={}",
            ev_yatzy,
            ev_chance
        );
    }

    // All categories open, various dice — print top 3 for reference
    for dice in &[
        [1, 2, 3, 4, 5],
        [1, 2, 4, 4, 5],
        [6, 6, 6, 6, 6],
        [3, 3, 3, 4, 4],
    ] {
        let resp = compute_roll_response(&ctx, 0, 0, dice, 0);
        let mut cats: Vec<_> = resp.categories.iter().filter(|c| c.available).collect();
        cats.sort_by(|a, b| b.ev_if_scored.partial_cmp(&a.ev_if_scored).unwrap());
        println!(
            "\nAll open, {:?}: optimal={} ({})",
            dice, CATEGORY_NAMES[resp.optimal_category as usize], resp.optimal_category
        );
        for c in cats.iter().take(3) {
            println!(
                "  {:20} score={:3}  ev={:.4}",
                c.name, c.score, c.ev_if_scored
            );
        }
        // Invariant: optimal matches argmax
        assert_eq!(resp.optimal_category, cats[0].id as i32);
    }
}
