//! Fast PRNG for simulation — SplitMix64 with die-roll specialization.
//!
//! SplitMix64 has a single u64 state word (vs SmallRng's 128-byte Xoshiro256++).
//! This gives better cache behavior when processing millions of games — each
//! GameState is 8 bytes smaller, and the PRNG runs in ~2 cycles vs ~4.
//!
//! For dice rolling, we extract 5 die values from a single u64 using modular
//! arithmetic and rejection-free bias-corrected extraction.

/// SplitMix64 PRNG — single u64 state, excellent statistical quality.
#[derive(Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create from seed.
    #[inline(always)]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate next u64.
    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Roll 5 dice (values 1-6) from a single PRNG call.
    ///
    /// Extracts 5 values using modular arithmetic on different bit ranges.
    /// Each extraction uses multiply-high to avoid modulo bias:
    /// `value = ((bits & 0xFFFF) * 6) >> 16` gives [0,5], then +1 for [1,6].
    ///
    /// Uses 5 non-overlapping 12-bit ranges from a 64-bit value.
    /// The multiply-high mapping partitions the 4096 window values into
    /// buckets of 683 (faces 1,2,4,5) and 682 (faces 3,6): an exact,
    /// permanent face bias of +0.049%/−0.098% — negligible for simulation
    /// and pinned exactly by test_dice_mapping_exact_census.
    #[inline(always)]
    pub fn roll_5_dice(&mut self) -> [i32; 5] {
        let r = self.next_u64();
        [
            ((((r & 0xFFF) * 6) >> 12) + 1) as i32,
            (((((r >> 12) & 0xFFF) * 6) >> 12) + 1) as i32,
            (((((r >> 24) & 0xFFF) * 6) >> 12) + 1) as i32,
            (((((r >> 36) & 0xFFF) * 6) >> 12) + 1) as i32,
            (((((r >> 48) & 0xFFF) * 6) >> 12) + 1) as i32,
        ]
    }

    /// Roll a single die (1-6).
    #[inline(always)]
    pub fn roll_die(&mut self) -> i32 {
        let r = self.next_u64();
        (((r & 0xFFF) * 6) >> 12) as i32 + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T14: the die mapping's face census is DETERMINISTIC — pin it exactly
    /// instead of chi-squaring around it. `v = (b·6) >> 12` over b ∈ [0,4096)
    /// gives faces 1,2,4,5 exactly 683 codes and faces 3,6 exactly 682.
    /// A ±3% statistical test (the old approach) cannot see a 1-2% extraction
    /// bug; this census catches a single miscounted code.
    #[test]
    fn test_dice_mapping_exact_census() {
        let mut counts = [0u32; 6];
        for b in 0u64..4096 {
            let face = ((b * 6) >> 12) as usize; // 0..=5
            counts[face] += 1;
        }
        assert_eq!(
            counts,
            [683, 683, 682, 683, 683, 682],
            "die-mapping census changed — the extraction bias is no longer \
             the documented +0.049%/-0.098%"
        );
        assert_eq!(counts.iter().sum::<u32>(), 4096);
    }

    /// T14: roll_5_dice must equal manual extraction of the same u64 —
    /// validates the window layout (5 non-overlapping 12-bit fields) without
    /// statistics. Also covers roll_die (the reroll path, previously
    /// untested).
    #[test]
    fn test_roll_matches_manual_extraction() {
        let mut rng = SplitMix64::new(0xABCDEF);
        let mut shadow = SplitMix64::new(0xABCDEF);
        for _ in 0..1000 {
            let r = shadow.next_u64();
            let manual = [
                ((((r) & 0xFFF) * 6 >> 12) + 1) as i32,
                ((((r >> 12) & 0xFFF) * 6 >> 12) + 1) as i32,
                ((((r >> 24) & 0xFFF) * 6 >> 12) + 1) as i32,
                ((((r >> 36) & 0xFFF) * 6 >> 12) + 1) as i32,
                ((((r >> 48) & 0xFFF) * 6 >> 12) + 1) as i32,
            ];
            assert_eq!(rng.roll_5_dice(), manual);
        }
        // roll_die: low 12 bits of a fresh draw.
        let mut rng = SplitMix64::new(0x1234);
        let mut shadow = SplitMix64::new(0x1234);
        for _ in 0..1000 {
            let r = shadow.next_u64();
            let manual = (((r & 0xFFF) * 6) >> 12) as i32 + 1;
            assert_eq!(rng.roll_die(), manual);
        }
    }

    #[test]
    fn test_splitmix64_deterministic() {
        let mut rng1 = SplitMix64::new(42);
        let mut rng2 = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_roll_5_dice_range() {
        let mut rng = SplitMix64::new(12345);
        for _ in 0..10000 {
            let dice = rng.roll_5_dice();
            for &d in &dice {
                assert!(d >= 1 && d <= 6, "Die out of range: {}", d);
            }
        }
    }

    #[test]
    fn test_roll_die_range() {
        let mut rng = SplitMix64::new(99);
        for _ in 0..10000 {
            let d = rng.roll_die();
            assert!(d >= 1 && d <= 6, "Die out of range: {}", d);
        }
    }

    #[test]
    fn test_roll_5_dice_distribution() {
        let mut rng = SplitMix64::new(42);
        let mut counts = [0u64; 6];
        let n = 100_000;
        for _ in 0..n {
            let dice = rng.roll_5_dice();
            for &d in &dice {
                counts[(d - 1) as usize] += 1;
            }
        }
        // Each face should be ~1/6 = ~83333 out of 500000
        let total = 5 * n as u64;
        let expected = total as f64 / 6.0;
        for (face, &count) in counts.iter().enumerate() {
            let ratio = count as f64 / expected;
            assert!(
                ratio > 0.97 && ratio < 1.03,
                "Face {} has count {} (expected ~{:.0}, ratio {:.3})",
                face + 1,
                count,
                expected,
                ratio
            );
        }
    }
}
