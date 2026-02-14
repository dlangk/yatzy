/*
 * phase0_tables.c — Phase 0: Precompute lookup tables
 *
 * Builds all static lookup tables before the DP pass:
 *   - s(S, r, c) scores for all (dice_set, category) pairs
 *   - Factorials for multinomial coefficient calculations
 *   - R_{5,6} enumeration (252 sorted 5-dice multisets)
 *   - P(r' → r) reroll transition probabilities
 *   - P(⊥ → r) initial roll probabilities
 *   - Popcount cache for scored-category bitmasks
 *   - Terminal state values (Phase 2 base case)
 *
 * See: theory/optimal_yahtzee_pseudocode.md, Phase 0.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "phase0_tables.h"
#include "dice_mechanics.h"
#include "game_mechanics.h"

/* Precompute s(S, r, c) for all r ∈ R_{5,6} and all 15 categories.
 * Stored in ctx->precomputed_scores[dice_set_index][category]. */
void PrecomputeCategoryScores(YatzyContext *ctx) {
    for (int i = 0; i < 252; i++) {
        int *dice = ctx->all_dice_sets[i];
        for (int cat = 0; cat < CATEGORY_COUNT; cat++) {
            ctx->precomputed_scores[i][cat] = CalculateCategoryScore(dice, cat);
        }
    }
}

/* Precompute factorials 0!..5! for multinomial coefficient calculations.
 * Used by ComputeProbabilityOfDiceSet to compute P(⊥ → r). */
void PrecomputeFactorials(YatzyContext *ctx) {
    ctx->factorial[0] = 1;
    for (int i = 1; i <= 5; i++) ctx->factorial[i] = ctx->factorial[i - 1] * i;
}

/* Enumerate all 252 sorted 5-dice multisets R_{5,6} and build reverse lookup.
 * Populates ctx->all_dice_sets[252][5] and ctx->index_lookup[6][6][6][6][6].
 *
 * See pseudocode Phase 0: "Enumerate all 252 distinct 5-dice multisets -> R_5,6" */
void BuildAllDiceCombinations(YatzyContext *ctx) {
    ctx->num_combinations = 0;
    for (int a = 1; a <= 6; a++) {
        for (int b = a; b <= 6; b++) {
            for (int c = b; c <= 6; c++) {
                for (int d = c; d <= 6; d++) {
                    for (int e = d; e <= 6; e++) {
                        int idx = ctx->num_combinations++;
                        ctx->all_dice_sets[idx][0] = a;
                        ctx->all_dice_sets[idx][1] = b;
                        ctx->all_dice_sets[idx][2] = c;
                        ctx->all_dice_sets[idx][3] = d;
                        ctx->all_dice_sets[idx][4] = e;
                        ctx->index_lookup[a - 1][b - 1][c - 1][d - 1][e - 1] = idx;
                    }
                }
            }
        }
    }
}

/*
 * PrecomputeKeepTable — Build dense keep-multiset transition table.
 *
 * Three sub-steps:
 *   3a. Enumerate all 462 keep-multisets (0-5 dice from {1..6}).
 *   3b. For each keep K and target T, compute P(K→T) via multinomial formula.
 *   3c. For each (ds, mask), map to keep index; build dedup and reverse mappings.
 */
void PrecomputeKeepTable(YatzyContext *ctx) {
    KeepTable *kt = &ctx->keep_table;

    /*
     * 3a: Enumerate all 462 keep-multisets as frequency vectors [f1..f6].
     *
     * A keep-multiset is defined by how many of each face {1..6} are kept.
     * For 0 kept dice: [0,0,0,0,0,0] (1 multiset)
     * For k kept dice: all ways to partition k into 6 bins (C(k+5,5) multisets)
     * Total: 1 + 6 + 21 + 56 + 126 + 252 = 462
     */
    int keep_freq[NUM_KEEP_MULTISETS][6]; /* frequency vector per keep */
    int keep_size[NUM_KEEP_MULTISETS];    /* number of dice kept */
    int num_keeps = 0;

    /* Reverse lookup: freq vector -> keep index.
     * Index: f1*6^5 + f2*6^4 + f3*6^3 + f4*6^2 + f5*6 + f6
     * but frequencies are 0-5, so max index = 5*6^5+... = 46655. Use 6^6=46656 array. */
    int keep_lookup[46656];
    memset(keep_lookup, -1, sizeof(keep_lookup));

    for (int f1 = 0; f1 <= 5; f1++)
    for (int f2 = 0; f2 <= 5 - f1; f2++)
    for (int f3 = 0; f3 <= 5 - f1 - f2; f3++)
    for (int f4 = 0; f4 <= 5 - f1 - f2 - f3; f4++)
    for (int f5 = 0; f5 <= 5 - f1 - f2 - f3 - f4; f5++) {
        int f6 = 0; /* f6 can be 0..5-sum, but we enumerate all valid totals */
        for (f6 = 0; f6 <= 5 - f1 - f2 - f3 - f4 - f5; f6++) {
            int idx = num_keeps++;
            keep_freq[idx][0] = f1;
            keep_freq[idx][1] = f2;
            keep_freq[idx][2] = f3;
            keep_freq[idx][3] = f4;
            keep_freq[idx][4] = f5;
            keep_freq[idx][5] = f6;
            keep_size[idx] = f1 + f2 + f3 + f4 + f5 + f6;
            int lookup_key = ((((f1 * 6 + f2) * 6 + f3) * 6 + f4) * 6 + f5) * 6 + f6;
            keep_lookup[lookup_key] = idx;
        }
    }

    /*
     * 3b: Compute P(K→T) for each keep K and target T.
     * Store in sparse per-row format: only non-zero entries.
     *
     * For keep K (freq [k1..k6]) and target T (freq [t1..t6]):
     *   P(K→T) = n! / prod((ti-ki)!) / 6^n   if ki <= ti for all i
     *          = 0                              otherwise
     * where n = 5 - |K| (number of rerolled dice).
     */
    int pow6[6] = {1, 6, 36, 216, 1296, 7776};
    int nnz = 0;

    for (int ki = 0; ki < num_keeps; ki++) {
        kt->row_start[ki] = nnz;
        int n = 5 - keep_size[ki]; /* dice rerolled */

        if (n == 0) {
            /* Keep all 5: deterministic transition to self */
            int dice[5], d = 0;
            for (int face = 0; face < 6; face++)
                for (int c = 0; c < keep_freq[ki][face]; c++)
                    dice[d++] = face + 1;
            if (d == 5) {
                int ti = ctx->index_lookup[dice[0]-1][dice[1]-1][dice[2]-1][dice[3]-1][dice[4]-1];
                kt->vals[nnz] = 1.0;
                kt->cols[nnz] = ti;
                nnz++;
            }
            continue;
        }

        double inv_pow6n = 1.0 / pow6[n];
        int fact_n = ctx->factorial[n];

        for (int ti = 0; ti < 252; ti++) {
            /* Get target frequency vector */
            int tf[6] = {0};
            const int *td = ctx->all_dice_sets[ti];
            for (int j = 0; j < 5; j++) tf[td[j] - 1]++;

            /* Check subset: ki <= ti for all faces */
            int valid = 1;
            int denom = 1;
            for (int f = 0; f < 6; f++) {
                if (keep_freq[ki][f] > tf[f]) { valid = 0; break; }
                denom *= ctx->factorial[tf[f] - keep_freq[ki][f]];
            }
            if (!valid) continue;

            kt->vals[nnz] = (double)fact_n / denom * inv_pow6n;
            kt->cols[nnz] = ti;
            nnz++;
        }
    }
    kt->row_start[num_keeps] = nnz;

    /*
     * 3c: For each (ds, mask), compute kept-dice frequency vector,
     *     look up keep index, build dedup and reverse mappings.
     */
    memset(kt->unique_count, 0, sizeof(kt->unique_count));
    memset(kt->mask_to_keep, -1, sizeof(kt->mask_to_keep));

    int total_unique = 0;
    for (int ds = 0; ds < 252; ds++) {
        const int *dice = ctx->all_dice_sets[ds];
        int seen[NUM_KEEP_MULTISETS]; /* track which keep ids we've seen for this ds */
        int n_unique = 0;

        for (int mask = 1; mask < 32; mask++) {
            /* Compute frequency vector of kept dice (bits NOT set in mask) */
            int kf[6] = {0};
            for (int i = 0; i < 5; i++) {
                if (!(mask & (1 << i))) {
                    kf[dice[i] - 1]++;
                }
            }
            int lookup_key = ((((kf[0] * 6 + kf[1]) * 6 + kf[2]) * 6 + kf[3]) * 6 + kf[4]) * 6 + kf[5];
            int kid = keep_lookup[lookup_key];
            kt->mask_to_keep[ds * 32 + mask] = kid;

            /* Dedup: check if we've already seen this keep for this ds */
            int found = 0;
            for (int j = 0; j < n_unique; j++) {
                if (seen[j] == kid) { found = 1; break; }
            }
            if (!found) {
                seen[n_unique] = kid;
                kt->unique_keep_ids[ds][n_unique] = kid;
                kt->keep_to_mask[ds * 32 + n_unique] = mask;
                n_unique++;
            }
        }

        kt->unique_count[ds] = n_unique;
        total_unique += n_unique;

        /* mask=0: keep all → identity, not stored in unique_keep_ids */
        /* Map mask=0 to the keep index for all 5 dice */
        int kf_all[6] = {0};
        for (int i = 0; i < 5; i++) kf_all[dice[i] - 1]++;
        int lookup_key = ((((kf_all[0] * 6 + kf_all[1]) * 6 + kf_all[2]) * 6 + kf_all[3]) * 6 + kf_all[4]) * 6 + kf_all[5];
        kt->mask_to_keep[ds * 32 + 0] = keep_lookup[lookup_key];
    }

    printf("    Keep-multiset table: %d keeps, %d nnz, avg %.1f unique/ds, %.1f KB\n",
           num_keeps, nnz, (double)total_unique / 252,
           (nnz * (sizeof(double) + sizeof(int))) / 1024.0);
}

/* Popcount cache: maps scored-category bitmask → |C| (number of categories scored).
 * Avoids repeated __builtin_popcount calls during the DP. */
void PrecomputeScoredCategoryCounts(YatzyContext *ctx) {
    for (int scored = 0; scored < (1 << CATEGORY_COUNT); scored++) {
        int count = 0;
        int temp = scored;
        for (int i = 0; i < CATEGORY_COUNT; i++) {
            count += (temp & 1);
            temp >>= 1;
        }
        ctx->scored_category_count_cache[scored] = count;
    }
}

/* Precompute P(⊥ → r) for all r ∈ R_{5,6}.
 * Stored in ctx->dice_set_probabilities[252]. */
void PrecomputeDiceSetProbabilities(YatzyContext *ctx) {
    for (int ds_i = 0; ds_i < 252; ds_i++) {
        ctx->dice_set_probabilities[ds_i] = ComputeProbabilityOfDiceSet(ctx, ctx->all_dice_sets[ds_i]);
    }
}

/* Phase 2 base case (terminal states, |C| = 15).
 * E(S) = 50 if m >= 63 (upper bonus), else 0.
 *
 * Note: Scandinavian Yatzy awards a 50-point upper bonus (not 35 as in
 * standard Yahtzee). See pseudocode Phase 2, Step 1. */
void InitializeFinalStates(YatzyContext *ctx) {
    int all_scored_mask = (1 << CATEGORY_COUNT) - 1;
    for (int up = 0; up <= 63; up++) {
        double final_val = (up >= 63) ? 50.0 : 0.0;
        ctx->state_values[STATE_INDEX(up, all_scored_mask)] = final_val;
    }
}

/*
 * Phase 1: Reachability pruning.
 *
 * Determines which (upper_mask, upper_score) pairs are reachable.
 * upper_mask is the 6-bit mask of which upper categories (Ones..Sixes)
 * have been scored. upper_score is the capped sum m ∈ [0,63].
 *
 * DP: R_exact[n][mask] = true if exact upper total n is achievable
 *     by scoring exactly the categories in mask.
 *
 * R_exact[0][0] = true (base case: no cats scored, total = 0)
 * R_exact[n][mask] = exists face x in mask, exists k in 0..5:
 *     k*x <= n AND R_exact[n - k*x][mask \ {x}]
 *
 * upper_score=63 means ">=63", so reachable[mask][63] = any(R_exact[k][mask] for k=63..105).
 * Max possible: 5*1 + 5*2 + ... + 5*6 = 105.
 */
void PrecomputeReachability(YatzyContext *ctx) {
    /* R_exact[n][mask]: n ∈ [0,105], mask ∈ [0,63] */
    uint8_t R[106][64];
    memset(R, 0, sizeof(R));
    R[0][0] = 1;

    /* Build R_exact bottom-up: add one face at a time */
    for (int face = 1; face <= 6; face++) {
        int bit = 1 << (face - 1);
        /* Process in reverse order of n to avoid using newly-set values
         * within the same face iteration. Actually we need to iterate
         * over masks that include this face and derive from masks without it. */
        for (int n = 105; n >= 0; n--) {
            for (int mask = 0; mask < 64; mask++) {
                if (!(mask & bit)) continue;  /* face not in mask */
                if (R[n][mask]) continue;      /* already reachable */
                int prev_mask = mask ^ bit;    /* mask without this face */
                for (int k = 0; k <= 5; k++) {
                    int contrib = k * face;
                    if (contrib > n) break;
                    if (R[n - contrib][prev_mask]) {
                        R[n][mask] = 1;
                        break;
                    }
                }
            }
        }
    }

    /* Collapse to reachable[mask][upper_score] with 63-cap handling */
    memset(ctx->reachable, 0, sizeof(ctx->reachable));
    for (int mask = 0; mask < 64; mask++) {
        for (int n = 0; n < 63; n++) {
            ctx->reachable[mask][n] = R[n][mask];
        }
        /* upper_score=63 means ">=63": OR together all exact values 63..105 */
        for (int n = 63; n <= 105; n++) {
            if (R[n][mask]) {
                ctx->reachable[mask][63] = 1;
                break;
            }
        }
    }

    /* Diagnostics: count reachable vs total */
    int reachable_count = 0;
    int total_count = 0;
    for (int mask = 0; mask < 64; mask++) {
        for (int up = 0; up <= 63; up++) {
            total_count++;
            if (ctx->reachable[mask][up]) reachable_count++;
        }
    }
    printf("    Reachable upper pairs: %d / %d (%.1f%% pruned)\n",
           reachable_count, total_count,
           100.0 * (1.0 - (double)reachable_count / total_count));
}
