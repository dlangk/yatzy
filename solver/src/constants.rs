//! Game constants and state-indexing functions.
//!
//! Maps pseudocode notation to concrete values:
//! - |𝒞| = [`CATEGORY_COUNT`] = 15 (Scandinavian Yatzy)
//! - |R_{5,6}| = [`NUM_DICE_SETS`] = 252
//! - |R_k| = [`NUM_KEEP_MULTISETS`] = 462
//! - STATE_INDEX(m, C) = [`state_index`]`(m, C)` = C * STATE_STRIDE + m
//!
//! The index layout groups all upper-score variants of the same scored-categories
//! mask into a contiguous region (STATE_STRIDE × f32 = 512 bytes). Indices 64..127
//! are padded with the capped value (index 63), enabling branchless upper-category
//! scoring via `sv[base + up + scr]` without `min(up + scr, 63)`.

/// Number of scoring categories in Scandinavian Yatzy (Ones through Yatzy).
/// Pseudocode uses |𝒞| = 13 (standard Yahtzee); we use 15.
pub const CATEGORY_COUNT: usize = 15;

/// Stride per scored-categories mask in the state array.
///
/// Padded from 64 to 128: indices 0..63 hold actual upper-score values,
/// indices 64..127 are filled with copies of index 63 (the capped value).
/// This enables branchless upper-category scoring: `sv[base + up + scr]`
/// always reads a valid, correct value without `min(up + scr, 63)`.
pub const STATE_STRIDE: usize = 128;

/// Total number of slots in the state array: STATE_STRIDE * 2^15.
/// With padding, this is 4,194,304 slots (~16 MB).
pub const NUM_STATES: usize = STATE_STRIDE * (1 << 15);

/// Number of distinct sorted 5-dice multisets from {1..6}: C(10,5) = 252.
pub const NUM_DICE_SETS: usize = 252;

/// Number of unique keep-multisets for 0-5 dice from {1..6}: 1+6+21+56+126+252 = 462.
pub const NUM_KEEP_MULTISETS: usize = 462;

/// Upper bound on total non-zero entries across all 462 keep rows.
pub const MAX_KEEP_NNZ_TOTAL: usize = 60000;

/// Storage format magic number: "STZY" in hex.
pub const STATE_FILE_MAGIC: u32 = 0x59545A53;

/// Storage format version (v6: scored*128+up layout with topological padding).
pub const STATE_FILE_VERSION: u32 = 6;

/// Storage format version v7: scored*128+up with θ (risk parameter) in header.
pub const STATE_FILE_VERSION_V5: u32 = 7;

/// Scandinavian Yatzy upper bonus: 50 points if upper score >= 63.
pub const UPPER_BONUS: f64 = 50.0;

/// Upper score cap.
pub const UPPER_SCORE_CAP: i32 = 63;

/// Category indices — used as bit positions in the scored_categories bitmask.
pub const CATEGORY_ONES: usize = 0;
pub const CATEGORY_TWOS: usize = 1;
pub const CATEGORY_THREES: usize = 2;
pub const CATEGORY_FOURS: usize = 3;
pub const CATEGORY_FIVES: usize = 4;
pub const CATEGORY_SIXES: usize = 5;
pub const CATEGORY_ONE_PAIR: usize = 6;
pub const CATEGORY_TWO_PAIRS: usize = 7;
pub const CATEGORY_THREE_OF_A_KIND: usize = 8;
pub const CATEGORY_FOUR_OF_A_KIND: usize = 9;
pub const CATEGORY_SMALL_STRAIGHT: usize = 10;
pub const CATEGORY_LARGE_STRAIGHT: usize = 11;
pub const CATEGORY_FULL_HOUSE: usize = 12;
pub const CATEGORY_CHANCE: usize = 13;
pub const CATEGORY_YATZY: usize = 14;

/// Human-readable category names.
pub const CATEGORY_NAMES: [&str; CATEGORY_COUNT] = [
    "Ones",
    "Twos",
    "Threes",
    "Fours",
    "Fives",
    "Sixes",
    "One Pair",
    "Two Pairs",
    "Three of a Kind",
    "Four of a Kind",
    "Small Straight",
    "Large Straight",
    "Full House",
    "Chance",
    "Yatzy",
];

/// Map state S = (upper_score, scored_categories) to flat array index.
///
/// Layout: `scored_categories * STATE_STRIDE + upper_score`. With STATE_STRIDE=128,
/// each scored mask occupies a 512-byte region (128 × f32). Indices 0..63 hold
/// actual values; indices 64..127 are padded with copies of index 63 (capped value).
/// This enables branchless upper-category scoring via `sv[base + up + scr]`.
#[inline(always)]
pub fn state_index(upper_score: usize, scored_categories: usize) -> usize {
    scored_categories * STATE_STRIDE + upper_score
}

/// Test whether category `cat` has been scored (bit `cat` is set).
#[inline(always)]
pub fn is_category_scored(scored: i32, cat: usize) -> bool {
    (scored & (1 << cat)) != 0
}
