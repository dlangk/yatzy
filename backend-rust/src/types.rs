//! Core data structures: game context, keep-multiset table, and state values.
//!
//! The central type is [`YatzyContext`], which holds all precomputed lookup tables
//! and the DP result array E_table[S]. It is built once by Phase 0
//! ([`crate::phase0_tables::precompute_lookup_tables`]) and then shared immutably
//! across threads during Phase 2 computation and API serving.

use crate::constants::*;

/// Keep-multiset transition table with sparse CSR (Compressed Sparse Row) storage.
///
/// A "keep-multiset" is the sorted multiset of dice retained before rerolling.
/// Multiple 5-bit reroll masks can produce the same keep (e.g., dice [1,1,2,3,4]
/// with masks 0b00001 and 0b00010 both keep {1,2,3,4}). This table collapses
/// those duplicates and stores transition probabilities P(r'→r) in sparse format.
///
/// Pseudocode equivalent: the P(r'→r) lookup in SOLVE_WIDGET groups 3 and 5.
///
/// Layout:
/// - `vals[row_start[ki]..row_start[ki+1]]` — probabilities for keep ki
/// - `cols[row_start[ki]..row_start[ki+1]]` — target dice-set indices
/// - `unique_keep_ids[ds][0..unique_count[ds]]` — deduplicated keep indices per dice set
/// - `mask_to_keep[ds*32 + mask]` — reroll mask → keep index (API input translation)
/// - `keep_to_mask[ds*32 + j]` — unique keep j → representative mask (API output)
pub struct KeepTable {
    /// Sparse probability values for each keep row.
    pub vals: Vec<f64>,
    /// Column indices corresponding to vals entries.
    pub cols: Vec<i32>,
    /// Row boundaries: row_start[ki]..row_start[ki+1] gives range in vals/cols.
    pub row_start: [i32; NUM_KEEP_MULTISETS + 1],
    /// Per dice set: how many unique keep-multisets (masks 1-31).
    pub unique_count: [i32; NUM_DICE_SETS],
    /// Keep indices for each unique keep per dice set.
    pub unique_keep_ids: [[i32; 31]; NUM_DICE_SETS],
    /// Reverse mapping: mask_to_keep[ds*32 + mask] = keep index.
    pub mask_to_keep: Vec<i32>,
    /// Representative mask: keep_to_mask[ds*32 + j] = first mask for unique keep j.
    pub keep_to_mask: Vec<i32>,
}

impl Default for KeepTable {
    fn default() -> Self {
        Self::new()
    }
}

impl KeepTable {
    pub fn new() -> Self {
        Self {
            vals: Vec::with_capacity(MAX_KEEP_NNZ_TOTAL),
            cols: Vec::with_capacity(MAX_KEEP_NNZ_TOTAL),
            row_start: [0; NUM_KEEP_MULTISETS + 1],
            unique_count: [0; NUM_DICE_SETS],
            unique_keep_ids: [[0; 31]; NUM_DICE_SETS],
            mask_to_keep: vec![-1; NUM_DICE_SETS * 32],
            keep_to_mask: vec![0; NUM_DICE_SETS * 32],
        }
    }
}

/// Turn-start state S = (m, C) from the pseudocode.
///
/// - `upper_score` (m): capped upper-section total, 0 ≤ m ≤ 63
/// - `scored_categories` (C): 15-bit bitmask where bit i is set if category i has been scored
///
/// The pseudocode also includes a Yahtzee bonus flag `f`, which is not used in
/// Scandinavian Yatzy.
#[derive(Clone, Copy, Debug)]
pub struct YatzyState {
    pub upper_score: i32,
    pub scored_categories: i32,
}

/// E_table[S] storage: either owned (during computation) or memory-mapped (loaded from disk).
///
/// - `Owned(Vec<f32>)`: allocated during precomputation, written via unsafe raw pointers
///   from parallel rayon workers.
/// - `Mmap`: zero-copy memory-mapped file, loaded in <1ms. The mmap includes a 16-byte
///   header (Storage v3 format), so `as_slice()` skips the first 16 bytes.
pub enum StateValues {
    Owned(Vec<f32>),
    Mmap { mmap: memmap2::Mmap },
}

impl StateValues {
    pub fn as_slice(&self) -> &[f32] {
        match self {
            StateValues::Owned(v) => v.as_slice(),
            StateValues::Mmap { mmap } => {
                let data_ptr = unsafe { mmap.as_ptr().add(16) as *const f32 };
                unsafe { std::slice::from_raw_parts(data_ptr, NUM_STATES) }
            }
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        match self {
            StateValues::Owned(v) => v.as_mut_slice(),
            StateValues::Mmap { .. } => panic!("Cannot mutably access mmap'd state values"),
        }
    }
}

/// Core context: all precomputed tables (Phase 0) and the DP result array (Phase 2).
///
/// This struct is ~10 MB and must be heap-allocated (use [`YatzyContext::new_boxed`]).
/// After Phase 0, it is shared immutably (`Arc<YatzyContext>`) across:
/// - Rayon workers during Phase 2 backward induction
/// - Axum HTTP handlers during API serving
///
/// Fields map to pseudocode concepts:
/// - `all_dice_sets` → R_{5,6} (252 sorted 5-dice multisets)
/// - `precomputed_scores` → s(S, r, c) (score for dice set r in category c)
/// - `dice_set_probabilities` → P(⊥ → r) (probability of rolling r from scratch)
/// - `keep_table` → P(r' → r) transition probabilities (sparse CSR format)
/// - `state_values` → E_table[S] (expected game value for each turn-start state)
/// - `reachable` → Phase 1 pruning result
pub struct YatzyContext {
    /// R_{5,6}: all 252 distinct sorted 5-dice multisets.
    pub all_dice_sets: [[i32; 5]; NUM_DICE_SETS],
    /// |R_{5,6}| (always 252).
    pub num_combinations: usize,
    /// Reverse lookup: sorted dice values -> index in R_{5,6}.
    pub index_lookup: [[[[[i32; 6]; 6]; 6]; 6]; 6],
    /// precomputed_scores[r][c] = s(S, r, c) for dice set r and category c.
    pub precomputed_scores: [[i32; CATEGORY_COUNT]; NUM_DICE_SETS],
    /// Popcount cache: scored_category_count_cache[C] = |C|.
    pub scored_category_count_cache: Vec<i32>,
    /// factorial[n] for n in 0..=5.
    pub factorial: [i32; 6],

    /// E_table[S]: DP result — expected game value for state S = (m, C).
    pub state_values: StateValues,
    /// P(empty -> r): probability of rolling each r in R_{5,6} from 5 fresh dice.
    pub dice_set_probabilities: [f64; NUM_DICE_SETS],
    /// Keep-multiset transition table.
    pub keep_table: KeepTable,

    /// Phase 1 reachability: reachable[upper_mask][upper_score] = true if reachable.
    pub reachable: [[bool; 64]; 64],
}

impl Default for YatzyContext {
    fn default() -> Self {
        Self::new()
    }
}

impl YatzyContext {
    pub fn new() -> Self {
        Self {
            all_dice_sets: [[0; 5]; NUM_DICE_SETS],
            num_combinations: 0,
            index_lookup: [[[[[0; 6]; 6]; 6]; 6]; 6],
            precomputed_scores: [[0; CATEGORY_COUNT]; NUM_DICE_SETS],
            scored_category_count_cache: vec![0; 1 << CATEGORY_COUNT],
            factorial: [0; 6],
            state_values: StateValues::Owned(vec![0.0f32; NUM_STATES]),
            dice_set_probabilities: [0.0; NUM_DICE_SETS],
            keep_table: KeepTable::new(),
            reachable: [[false; 64]; 64],
        }
    }

    /// Allocate on the heap directly (avoids stack overflow in debug builds).
    pub fn new_boxed() -> Box<Self> {
        Box::new(Self::new())
    }

    /// Look up E_table[S] for state (upper_score, scored_categories).
    ///
    /// Returns f64 for computation precision, even though storage is f32.
    #[inline(always)]
    pub fn get_state_value(&self, upper_score: i32, scored: i32) -> f64 {
        self.state_values.as_slice()[state_index(upper_score as usize, scored as usize)] as f64
    }
}
