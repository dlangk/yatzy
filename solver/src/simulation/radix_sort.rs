//! Radix sort for lockstep simulation — sort game indices by state_index.
//!
//! Uses a 2-pass counting sort on the 21-bit state_index (STATE_STRIDE * 32768):
//! - Pass 1: sort by bits 0-10 (2048 buckets)
//! - Pass 2: sort by bits 11-20 (1024 buckets)
//!
//! Total: O(2N) work, no hashing, produces contiguous groups.
//! For 10M games: ~80M memory accesses, ~1s at memory bandwidth limit.

/// Produced by radix sort: contiguous groups of game indices sharing a state.
pub struct SortedGroups {
    /// Game indices sorted by their state_index key.
    pub indices: Vec<u32>,
    /// (state_index, start, count) for each non-empty group.
    pub groups: Vec<(u32, u32, u32)>,
}

/// Sort game indices by state_index using 2-pass counting sort.
///
/// Returns sorted indices plus group boundaries for efficient iteration.
pub fn radix_sort_by_state(keys: &[u32], n: usize) -> SortedGroups {
    debug_assert_eq!(keys.len(), n);

    // Pass 1: sort by low 11 bits (2048 buckets)
    let mut src: Vec<u32> = (0..n as u32).collect();
    let mut dst = vec![0u32; n];

    const BITS1: u32 = 11;
    const MASK1: u32 = (1 << BITS1) - 1; // 0x7FF
    const BUCKETS1: usize = 1 << BITS1; // 2048

    // Histogram
    let mut hist = [0u32; BUCKETS1];
    for &k in keys.iter() {
        hist[(k & MASK1) as usize] += 1;
    }

    // Prefix sum
    let mut sum = 0u32;
    for h in hist.iter_mut() {
        let count = *h;
        *h = sum;
        sum += count;
    }

    // Scatter
    for &idx in src.iter() {
        let bucket = (keys[idx as usize] & MASK1) as usize;
        dst[hist[bucket] as usize] = idx;
        hist[bucket] += 1;
    }

    // Pass 2: sort by bits 11-20 (1024 buckets — max state_index is ~4M for stride 128)
    std::mem::swap(&mut src, &mut dst);

    const BITS2: u32 = 11;
    const SHIFT2: u32 = BITS1;
    const MASK2: u32 = (1 << BITS2) - 1;
    const BUCKETS2: usize = 1 << BITS2; // 2048

    let mut hist2 = [0u32; BUCKETS2];
    for &idx in src.iter() {
        hist2[((keys[idx as usize] >> SHIFT2) & MASK2) as usize] += 1;
    }

    let mut sum = 0u32;
    for h in hist2.iter_mut() {
        let count = *h;
        *h = sum;
        sum += count;
    }

    for &idx in src.iter() {
        let bucket = ((keys[idx as usize] >> SHIFT2) & MASK2) as usize;
        dst[hist2[bucket] as usize] = idx;
        hist2[bucket] += 1;
    }

    // Build groups from sorted order
    let mut groups: Vec<(u32, u32, u32)> = Vec::new();
    if n > 0 {
        let mut start = 0u32;
        let mut current_key = keys[dst[0] as usize];
        for i in 1..n {
            let k = keys[dst[i] as usize];
            if k != current_key {
                groups.push((current_key, start, i as u32 - start));
                current_key = k;
                start = i as u32;
            }
        }
        groups.push((current_key, start, n as u32 - start));
    }

    SortedGroups {
        indices: dst,
        groups,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radix_sort_basic() {
        let keys = vec![5, 3, 8, 3, 1, 5, 8, 1];
        let result = radix_sort_by_state(&keys, 8);

        // Check sorted order
        for i in 1..result.indices.len() {
            let ka = keys[result.indices[i - 1] as usize];
            let kb = keys[result.indices[i] as usize];
            assert!(ka <= kb, "Not sorted at {}: {} > {}", i, ka, kb);
        }

        // Check groups
        assert_eq!(result.groups.len(), 4); // keys 1, 3, 5, 8
        for &(key, start, count) in &result.groups {
            for j in start..(start + count) {
                assert_eq!(keys[result.indices[j as usize] as usize], key);
            }
        }
    }

    #[test]
    fn test_radix_sort_single_group() {
        let keys = vec![42, 42, 42, 42];
        let result = radix_sort_by_state(&keys, 4);
        assert_eq!(result.groups.len(), 1);
        assert_eq!(result.groups[0], (42, 0, 4));
    }

    #[test]
    fn test_radix_sort_large_keys() {
        // STATE_STRIDE = 128, max scored = 32767 → max key ~4M
        let keys = vec![0, 4_000_000, 128, 256, 4_000_000, 128];
        let result = radix_sort_by_state(&keys, 6);

        for i in 1..result.indices.len() {
            let ka = keys[result.indices[i - 1] as usize];
            let kb = keys[result.indices[i] as usize];
            assert!(ka <= kb);
        }
        assert_eq!(result.groups.len(), 4);
    }

    #[test]
    fn test_radix_sort_empty() {
        let keys: Vec<u32> = vec![];
        let result = radix_sort_by_state(&keys, 0);
        assert_eq!(result.indices.len(), 0);
        assert_eq!(result.groups.len(), 0);
    }
}
