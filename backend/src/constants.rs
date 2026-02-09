//! Game constants and state-indexing functions.
//!
//! Maps pseudocode notation to concrete values:
//! - |𝒞| = [`CATEGORY_COUNT`] = 15 (Scandinavian Yatzy)
//! - |R_{5,6}| = [`NUM_DICE_SETS`] = 252
//! - |R_k| = [`NUM_KEEP_MULTISETS`] = 462
//! - STATE_INDEX(m, C) = [`state_index`]`(m, C)` = C * 64 + m
//!
//! The index layout groups all 64 upper-score variants of the same scored-categories
//! mask into a contiguous 256-byte region, enabling L1 cache hits during Group 6
//! successor lookups (which vary upper_score but share scored_categories).

/// Number of scoring categories in Scandinavian Yatzy (Ones through Yatzy).
/// Pseudocode uses |𝒞| = 13 (standard Yahtzee); we use 15.
pub const CATEGORY_COUNT: usize = 15;

/// Total number of game states: 64 possible upper-section scores * 2^15 scored-category bitmasks.
pub const NUM_STATES: usize = 64 * (1 << 15);

/// Number of distinct sorted 5-dice multisets from {1..6}: C(10,5) = 252.
pub const NUM_DICE_SETS: usize = 252;

/// Number of unique keep-multisets for 0-5 dice from {1..6}: 1+6+21+56+126+252 = 462.
pub const NUM_KEEP_MULTISETS: usize = 462;

/// Upper bound on total non-zero entries across all 462 keep rows.
pub const MAX_KEEP_NNZ_TOTAL: usize = 60000;

/// Storage format magic number: "STZY" in hex.
pub const STATE_FILE_MAGIC: u32 = 0x59545A53;

/// Storage format version (v4: scored*64+up index layout for cache locality).
pub const STATE_FILE_VERSION: u32 = 4;

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
/// Layout: `scored_categories * 64 + upper_score`. This groups all upper-score
/// variants of the same scored mask into a contiguous 256-byte region (64 × f32),
/// giving L1 cache hits when Group 6 iterates successor states that share the
/// same scored mask but differ in upper_score.
#[inline(always)]
pub fn state_index(upper_score: usize, scored_categories: usize) -> usize {
    scored_categories * 64 + upper_score
}

/// Test whether category `cat` has been scored (bit `cat` is set).
#[inline(always)]
pub fn is_category_scored(scored: i32, cat: usize) -> bool {
    (scored & (1 << cat)) != 0
}
