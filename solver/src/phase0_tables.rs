//! Phase 0: Precompute all static lookup tables.
//!
//! Implements pseudocode `PRECOMPUTE_ROLLS_AND_PROBABILITIES` and `COMPUTE_REACHABILITY`.
//!
//! The orchestrator [`precompute_lookup_tables`] runs 8 sub-steps in dependency order:
//!
//! 1. **Factorials** — 0!..5! for multinomial coefficients
//! 2. **Dice combinations** — enumerate R_{5,6} (252 sorted 5-dice multisets) + reverse lookup
//! 3. **Category scores** — precompute s(S, r, c) for all (r, c) pairs
//! 4. **Keep-multiset table** — sparse CSR transition matrix P(r'→r) with dedup mappings
//! 5. **Dice set probabilities** — P(⊥→r) for each r ∈ R_{5,6}
//! 6. **Scored category counts** — popcount cache for bitmask → |C|
//! 7. **Terminal states** — Phase 2 base case: E(S) = 50 if m≥63, else 0
//! 8. **Reachability pruning** — Phase 1: which (upper_mask, m) pairs are achievable

use std::time::Instant;

use crate::constants::*;
use crate::dice_mechanics::compute_probability_of_dice_set;
use crate::game_mechanics::calculate_category_score;
use crate::types::YatzyContext;

/// Precompute factorials 0!..5! for multinomial coefficient calculations.
pub fn precompute_factorials(ctx: &mut YatzyContext) {
    ctx.factorial[0] = 1;
    for i in 1..=5 {
        ctx.factorial[i] = ctx.factorial[i - 1] * i as i32;
    }
}

/// Enumerate all C(10,5) = 252 sorted 5-dice multisets R_{5,6} and build
/// a 5D reverse lookup table: `index_lookup[d1-1][d2-1][d3-1][d4-1][d5-1] = index`.
pub fn build_all_dice_combinations(ctx: &mut YatzyContext) {
    ctx.num_combinations = 0;
    for a in 1..=6i32 {
        for b in a..=6 {
            for c in b..=6 {
                for d in c..=6 {
                    for e in d..=6 {
                        let idx = ctx.num_combinations;
                        ctx.all_dice_sets[idx] = [a, b, c, d, e];
                        ctx.index_lookup[(a - 1) as usize][(b - 1) as usize][(c - 1) as usize]
                            [(d - 1) as usize][(e - 1) as usize] = idx as i32;
                        ctx.num_combinations += 1;
                    }
                }
            }
        }
    }
}

/// Precompute s(S, r, c) for all r in R_{5,6} and all 15 categories.
pub fn precompute_category_scores(ctx: &mut YatzyContext) {
    for i in 0..252 {
        let dice = ctx.all_dice_sets[i];
        for cat in 0..CATEGORY_COUNT {
            ctx.precomputed_scores[i][cat] = calculate_category_score(&dice, cat);
        }
    }
}

/// Build the keep-multiset transition table (the key optimization).
///
/// This is the most important Phase 0 step. It precomputes P(r'→r) from the
/// pseudocode and organizes it in sparse CSR format with per-dice-set dedup.
///
/// Three sub-steps:
///
/// **3a.** Enumerate all 462 keep-multisets R_k (0–5 dice from {1..6}) as
/// frequency vectors [f1..f6]. Build a reverse lookup from frequency vector → index.
///
/// **3b.** For each keep K ∈ R_k and target T ∈ R_{5,6}, compute the transition
/// probability P(K→T) via the multinomial formula:
///   P(K→T) = n! / (d1!·d2!·...·d6!) / 6^n
/// where n = 5 - |K| (dice rerolled) and di = tf[i] - kf[i] (rerolled dice per face).
/// Results stored in CSR format: vals[]/cols[] with row_start[] boundaries.
///
/// **3c.** For each dice set ds and reroll mask (1–31), compute which keep-multiset
/// the mask produces, then deduplicate: multiple masks can yield the same keep.
/// Builds `unique_keep_ids`, `unique_count`, `mask_to_keep`, `keep_to_mask`.
pub fn precompute_keep_table(ctx: &mut YatzyContext) {
    let kt = &mut ctx.keep_table;

    // 3a: Enumerate all 462 keep-multisets as frequency vectors [f1..f6].
    let mut keep_freq = [[0i32; 6]; NUM_KEEP_MULTISETS];
    let mut keep_size = [0i32; NUM_KEEP_MULTISETS];
    let mut num_keeps = 0usize;

    // Reverse lookup: freq vector -> keep index.
    let mut keep_lookup = vec![-1i32; 46656]; // 6^6

    for f1 in 0..=5i32 {
        for f2 in 0..=(5 - f1) {
            for f3 in 0..=(5 - f1 - f2) {
                for f4 in 0..=(5 - f1 - f2 - f3) {
                    for f5 in 0..=(5 - f1 - f2 - f3 - f4) {
                        for f6 in 0..=(5 - f1 - f2 - f3 - f4 - f5) {
                            let idx = num_keeps;
                            keep_freq[idx] = [f1, f2, f3, f4, f5, f6];
                            keep_size[idx] = f1 + f2 + f3 + f4 + f5 + f6;
                            let lookup_key =
                                ((((f1 * 6 + f2) * 6 + f3) * 6 + f4) * 6 + f5) * 6 + f6;
                            keep_lookup[lookup_key as usize] = idx as i32;
                            num_keeps += 1;
                        }
                    }
                }
            }
        }
    }

    // 3b: Compute P(K->T) for each keep K and target T.
    let pow6: [i32; 6] = [1, 6, 36, 216, 1296, 7776];

    kt.vals.clear();
    kt.cols.clear();

    for ki in 0..num_keeps {
        kt.row_start[ki] = kt.vals.len() as i32;
        let n = 5 - keep_size[ki]; // dice rerolled

        if n == 0 {
            // Keep all 5: deterministic transition to self
            let mut dice = [0i32; 5];
            let mut d = 0;
            for face in 0..6 {
                for _ in 0..keep_freq[ki][face] {
                    dice[d] = face as i32 + 1;
                    d += 1;
                }
            }
            if d == 5 {
                let ti = ctx.index_lookup[(dice[0] - 1) as usize][(dice[1] - 1) as usize]
                    [(dice[2] - 1) as usize][(dice[3] - 1) as usize][(dice[4] - 1) as usize];
                kt.vals.push(1.0f32);
                kt.cols.push(ti);
            }
            continue;
        }

        let inv_pow6n = 1.0 / pow6[n as usize] as f64;
        let fact_n = ctx.factorial[n as usize];

        for ti in 0..252usize {
            // Get target frequency vector
            let mut tf = [0i32; 6];
            let td = &ctx.all_dice_sets[ti];
            for j in 0..5 {
                tf[(td[j] - 1) as usize] += 1;
            }

            // Check subset: ki <= ti for all faces
            let mut valid = true;
            let mut denom = 1i32;
            for f in 0..6 {
                if keep_freq[ki][f] > tf[f] {
                    valid = false;
                    break;
                }
                denom *= ctx.factorial[(tf[f] - keep_freq[ki][f]) as usize];
            }
            if !valid {
                continue;
            }

            kt.vals
                .push((fact_n as f64 / denom as f64 * inv_pow6n) as f32);
            kt.cols.push(ti as i32);
        }
    }
    kt.row_start[num_keeps] = kt.vals.len() as i32;

    // 3c: For each (ds, mask), compute kept-dice frequency vector,
    //     look up keep index, build dedup and reverse mappings.
    kt.unique_count = [0; NUM_DICE_SETS];
    kt.mask_to_keep = vec![-1; NUM_DICE_SETS * 32];

    let mut total_unique = 0;
    for ds in 0..252usize {
        let dice = &ctx.all_dice_sets[ds];
        let mut seen = [0i32; NUM_KEEP_MULTISETS];
        let mut n_unique = 0usize;

        for mask in 1..32i32 {
            // Compute frequency vector of kept dice (bits NOT set in mask)
            let mut kf = [0i32; 6];
            for i in 0..5 {
                if (mask & (1 << i)) == 0 {
                    kf[(dice[i] - 1) as usize] += 1;
                }
            }
            let lookup_key =
                ((((kf[0] * 6 + kf[1]) * 6 + kf[2]) * 6 + kf[3]) * 6 + kf[4]) * 6 + kf[5];
            let kid = keep_lookup[lookup_key as usize];
            kt.mask_to_keep[ds * 32 + mask as usize] = kid;

            // Dedup: check if we've already seen this keep for this ds
            let mut found = false;
            for j in 0..n_unique {
                if seen[j] == kid {
                    found = true;
                    break;
                }
            }
            if !found {
                seen[n_unique] = kid;
                kt.unique_keep_ids[ds][n_unique] = kid;
                kt.keep_to_mask[ds * 32 + n_unique] = mask;
                n_unique += 1;
            }
        }

        kt.unique_count[ds] = n_unique as i32;
        total_unique += n_unique;

        // mask=0: keep all -> identity
        let mut kf_all = [0i32; 6];
        for i in 0..5 {
            kf_all[(dice[i] - 1) as usize] += 1;
        }
        let lookup_key =
            ((((kf_all[0] * 6 + kf_all[1]) * 6 + kf_all[2]) * 6 + kf_all[3]) * 6 + kf_all[4]) * 6
                + kf_all[5];
        kt.mask_to_keep[ds * 32] = keep_lookup[lookup_key as usize];
    }

    let nnz = kt.vals.len();
    println!(
        "    Keep-multiset table: {} keeps, {} nnz, avg {:.1} unique/ds, {:.1} KB",
        num_keeps,
        nnz,
        total_unique as f64 / 252.0,
        (nnz * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>())) as f64 / 1024.0
    );
}

/// Popcount cache: maps scored-category bitmask -> |C|.
pub fn precompute_scored_category_counts(ctx: &mut YatzyContext) {
    for scored in 0..(1 << CATEGORY_COUNT) {
        ctx.scored_category_count_cache[scored] = (scored as u32).count_ones() as i32;
    }
}

/// Precompute P(empty -> r) for all r in R_{5,6}.
pub fn precompute_dice_set_probabilities(ctx: &mut YatzyContext) {
    for ds_i in 0..252 {
        ctx.dice_set_probabilities[ds_i] =
            compute_probability_of_dice_set(ctx, &ctx.all_dice_sets[ds_i]);
    }
}

/// Phase 2 base case (terminal states, |C| = 15).
///
/// Pseudocode: E(C, m, f) = 35 if m ≥ 63, else 0.
/// Scandinavian Yatzy uses 50-point upper bonus instead of 35.
///
/// Three modes:
/// - θ = 0 (EV): `E(S) = bonus`
/// - θ ≠ 0, utility domain (|θ| ≤ 0.15): `U(S) = e^(θ·bonus)`
/// - θ ≠ 0, log domain (|θ| > 0.15): `L(S) = θ·bonus`
pub fn initialize_final_states(ctx: &mut YatzyContext) {
    let all_scored_mask = (1 << CATEGORY_COUNT) - 1;
    let theta = ctx.theta;
    let use_utility = theta != 0.0 && theta.abs() <= 0.15;
    let state_values = ctx.state_values.as_mut_slice();
    for up in 0..=63usize {
        let bonus = if up >= 63 { 50.0f32 } else { 0.0f32 };
        let final_val = if theta == 0.0 {
            bonus
        } else if use_utility {
            (theta * bonus).exp()
        } else {
            theta * bonus
        };
        state_values[state_index(up, all_scored_mask)] = final_val;
    }
    // Fill topological padding for terminal states: indices 64..127 = value at 63
    let capped_val = state_values[state_index(63, all_scored_mask)];
    let base = state_index(0, all_scored_mask);
    for pad in 64..STATE_STRIDE {
        state_values[base + pad] = capped_val;
    }
}

/// Phase 1: Reachability pruning (pseudocode `COMPUTE_REACHABILITY`).
///
/// Determines which (upper_mask, upper_score) pairs are reachable by dynamic
/// programming over the 6 upper-section categories. A pair (mask, n) is reachable
/// if there exists a valid assignment of upper category scores that sums to exactly n
/// using the categories indicated by mask.
///
/// Result: `ctx.reachable[mask][n]` = true if reachable. Upper score 63 means "≥63"
/// (all exact values 63..105 are OR'd together).
/// Eliminates ~31.8% of (mask, score) pairs, reducing the Phase 2 workload.
pub fn precompute_reachability(ctx: &mut YatzyContext) {
    // R_exact[n][mask]: n in [0,105], mask in [0,63]
    let mut r = vec![vec![false; 64]; 106];
    r[0][0] = true;

    // Build R_exact bottom-up: add one face at a time
    for face in 1..=6usize {
        let bit = 1 << (face - 1);
        for n in (0..=105).rev() {
            for mask in 0..64usize {
                if (mask & bit) == 0 {
                    continue;
                }
                if r[n][mask] {
                    continue;
                }
                let prev_mask = mask ^ bit;
                for k in 0..=5usize {
                    let contrib = k * face;
                    if contrib > n {
                        break;
                    }
                    if r[n - contrib][prev_mask] {
                        r[n][mask] = true;
                        break;
                    }
                }
            }
        }
    }

    // Collapse to reachable[mask][upper_score] with 63-cap handling
    ctx.reachable = [[false; 64]; 64];
    for mask in 0..64usize {
        for n in 0..63usize {
            ctx.reachable[mask][n] = r[n][mask];
        }
        // upper_score=63 means ">=63": OR together all exact values 63..105
        for n in 63..=105usize {
            if r[n][mask] {
                ctx.reachable[mask][63] = true;
                break;
            }
        }
    }

    // Diagnostics
    let mut reachable_count = 0;
    let mut total_count = 0;
    for mask in 0..64 {
        for up in 0..=63 {
            total_count += 1;
            if ctx.reachable[mask][up] {
                reachable_count += 1;
            }
        }
    }
    println!(
        "    Reachable upper pairs: {} / {} ({:.1}% pruned)",
        reachable_count,
        total_count,
        100.0 * (1.0 - reachable_count as f64 / total_count as f64)
    );
}

/// Phase 0 orchestrator: build all static lookup tables in dependency order.
pub fn precompute_lookup_tables(ctx: &mut YatzyContext) {
    println!("=== Phase 0: Precompute Lookup Tables ===");
    let phase0_start = Instant::now();

    macro_rules! timed {
        ($label:expr, $body:expr) => {{
            let t0 = Instant::now();
            $body;
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            println!("  {:<42} {:>8.3} ms", $label, dt);
        }};
    }

    timed!("Factorials", precompute_factorials(ctx));
    timed!("Dice combinations (252)", build_all_dice_combinations(ctx));
    timed!("Category scores", precompute_category_scores(ctx));
    timed!("Keep-multiset table", precompute_keep_table(ctx));
    timed!(
        "Dice set probabilities",
        precompute_dice_set_probabilities(ctx)
    );
    timed!(
        "Scored category counts",
        precompute_scored_category_counts(ctx)
    );
    timed!("Terminal states", initialize_final_states(ctx));
    timed!("Reachability pruning", precompute_reachability(ctx));

    let total = phase0_start.elapsed().as_secs_f64() * 1000.0;
    println!("  {:<42} {:>8.3} ms", "TOTAL Phase 0", total);
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx() -> Box<YatzyContext> {
        let mut ctx = YatzyContext::new_boxed();
        precompute_lookup_tables(&mut ctx);
        ctx
    }

    #[test]
    fn test_context_creation() {
        let ctx = make_ctx();
        assert_eq!(ctx.factorial[0], 1);
        assert_eq!(ctx.factorial[1], 1);
        assert_eq!(ctx.factorial[5], 120);
        assert_eq!(ctx.num_combinations, 252);
        assert_eq!(ctx.all_dice_sets[0], [1, 1, 1, 1, 1]);
        assert_eq!(ctx.all_dice_sets[251], [6, 6, 6, 6, 6]);
        assert_eq!(ctx.precomputed_scores[0][CATEGORY_ONES], 5);
        assert_eq!(ctx.precomputed_scores[0][CATEGORY_TWOS], 0);
        assert_eq!(ctx.precomputed_scores[0][CATEGORY_YATZY], 50);
        assert_eq!(ctx.precomputed_scores[251][CATEGORY_SIXES], 30);
        assert_eq!(ctx.precomputed_scores[251][CATEGORY_YATZY], 50);
        assert_eq!(ctx.scored_category_count_cache[0], 0);
        assert_eq!(
            ctx.scored_category_count_cache[(1 << CATEGORY_COUNT) - 1],
            15
        );
        assert_eq!(ctx.scored_category_count_cache[1], 1);

        let prob_sum: f64 = ctx.dice_set_probabilities.iter().sum();
        assert!((prob_sum - 1.0).abs() < 1e-9);

        // Final states
        let all_scored = (1 << CATEGORY_COUNT) - 1;
        assert!((ctx.get_state_value(63, all_scored as i32) - 50.0).abs() < 1e-9);
        assert!((ctx.get_state_value(0, all_scored as i32)).abs() < 1e-9);
        assert!((ctx.get_state_value(62, all_scored as i32)).abs() < 1e-9);
    }

    #[test]
    fn test_252_dice_sets() {
        let ctx = make_ctx();
        assert_eq!(ctx.num_combinations, 252);

        // All sorted
        for i in 0..252 {
            for j in 0..4 {
                assert!(ctx.all_dice_sets[i][j] <= ctx.all_dice_sets[i][j + 1]);
            }
        }

        // All unique
        for i in 0..252 {
            for j in (i + 1)..252 {
                assert_ne!(ctx.all_dice_sets[i], ctx.all_dice_sets[j]);
            }
        }
    }

    #[test]
    fn test_keep_table_row_sums() {
        let ctx = make_ctx();
        let kt = &ctx.keep_table;
        for ki in 0..NUM_KEEP_MULTISETS {
            let start = kt.row_start[ki] as usize;
            let end = kt.row_start[ki + 1] as usize;
            if start == end {
                continue;
            }
            let sum: f64 = (start..end).map(|k| kt.vals[k] as f64).sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "keep_table row {} sums to {}",
                ki,
                sum
            );
        }
    }

    #[test]
    fn test_keep_table_enumeration() {
        let ctx = make_ctx();
        let kt = &ctx.keep_table;

        // [1,1,1,1,1] has 5 unique keeps
        assert_eq!(kt.unique_count[0], 5);
        // [6,6,6,6,6] has 5 unique keeps
        assert_eq!(kt.unique_count[251], 5);

        // [1,2,3,4,5] has 31 unique keeps
        let ds_12345 = ctx.index_lookup[0][1][2][3][4] as usize;
        assert_eq!(kt.unique_count[ds_12345], 31);

        // Average
        let total: i32 = kt.unique_count.iter().sum();
        let avg = total as f64 / 252.0;
        assert!(avg > 15.0 && avg < 25.0);
    }

    #[test]
    fn test_keep_table_reroll_all() {
        let ctx = make_ctx();
        let kt = &ctx.keep_table;

        // All dice sets with mask=31 should map to the same keep index
        let empty_kid = kt.mask_to_keep[0 * 32 + 31];
        for ds in 1..252 {
            assert_eq!(kt.mask_to_keep[ds * 32 + 31], empty_kid);
        }

        // Empty keep probs should match dice_set_probabilities
        let mut empty_probs = [0.0f64; 252];
        let start = kt.row_start[empty_kid as usize] as usize;
        let end = kt.row_start[empty_kid as usize + 1] as usize;
        for k in start..end {
            empty_probs[kt.cols[k] as usize] = kt.vals[k] as f64;
        }
        for t in 0..252 {
            assert!(
                (empty_probs[t] - ctx.dice_set_probabilities[t]).abs() < 1e-6,
                "empty keep prob[{}]={} != dice_set_prob={}",
                t,
                empty_probs[t],
                ctx.dice_set_probabilities[t]
            );
        }
    }

    #[test]
    fn test_precomputed_scores_match() {
        let ctx = make_ctx();
        use crate::game_mechanics::calculate_category_score;
        for i in 0..252 {
            for c in 0..CATEGORY_COUNT {
                let precomp = ctx.precomputed_scores[i][c];
                let direct = calculate_category_score(&ctx.all_dice_sets[i], c);
                assert_eq!(precomp, direct, "Mismatch at ds={} cat={}", i, c);
            }
        }
    }

    #[test]
    fn test_popcount_cache() {
        let ctx = make_ctx();
        for i in 0..(1 << 15) {
            assert_eq!(
                ctx.scored_category_count_cache[i],
                (i as u32).count_ones() as i32
            );
        }
    }

    #[test]
    fn test_reachability() {
        let ctx = make_ctx();
        assert!(ctx.reachable[0][0]);
        assert!(!ctx.reachable[0][1]);
        assert!(!ctx.reachable[0][63]);
        assert!(ctx.reachable[1][0]);
        assert!(ctx.reachable[1][5]);
        assert!(!ctx.reachable[1][6]);
        assert!(ctx.reachable[0x20][0]);
        assert!(ctx.reachable[0x20][6]);
        assert!(ctx.reachable[0x20][30]);
        assert!(!ctx.reachable[0x20][1]);
        assert!(!ctx.reachable[0x20][7]);
        assert!(ctx.reachable[0x3F][63]);
    }
}
