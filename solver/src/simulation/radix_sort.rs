//! Radix sort for lockstep simulation — sort game indices by state_index.
//!
//! Uses a 2-pass counting sort on the 22-bit state_index (STATE_STRIDE * 32768):
//! - Pass 1: sort by bits 0-10 (2048 buckets)
//! - Pass 2: sort by bits 11-21 (2048 buckets)
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

/// Reusable buffers for [`radix_sort_by_state_into`]. Hoist one of these
/// outside a per-turn loop to avoid three N-sized allocations per call.
pub struct RadixScratch {
    /// Pass-1 output (intermediate ordering).
    buf: Vec<u32>,
    /// Pass-2 output: game indices sorted by key.
    pub indices: Vec<u32>,
    /// (state_index, start, count) for each non-empty group.
    pub groups: Vec<(u32, u32, u32)>,
    /// keys[indices[i]] materialized during the pass-2 scatter, so group
    /// boundary detection scans sequentially instead of gathering.
    sorted_keys: Vec<u32>,
    /// Per-(chunk, bucket) histograms/cursors for the parallel path.
    chunk_hists: Vec<u32>,
}

impl RadixScratch {
    pub fn new() -> Self {
        RadixScratch {
            buf: Vec::new(),
            indices: Vec::new(),
            groups: Vec::new(),
            sorted_keys: Vec::new(),
            chunk_hists: Vec::new(),
        }
    }
}

impl Default for RadixScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Sort game indices by state_index using 2-pass counting sort, writing the
/// result into `scratch` (buffers are reused across calls; contents are
/// fully overwritten).
pub fn radix_sort_by_state_into(keys: &[u32], n: usize, scratch: &mut RadixScratch) {
    debug_assert_eq!(keys.len(), n);

    scratch.buf.resize(n, 0);
    scratch.indices.resize(n, 0);
    scratch.groups.clear();
    let mid = &mut scratch.buf;
    let dst = &mut scratch.indices;

    const BITS1: u32 = 11;
    const MASK1: u32 = (1 << BITS1) - 1; // 0x7FF
    const BUCKETS1: usize = 1 << BITS1; // 2048

    // Pass 1: sort by low 11 bits (2048 buckets).
    // Phase 1: Histogram — count elements per bucket
    let mut hist = [0u32; BUCKETS1];
    for &k in keys.iter() {
        hist[(k & MASK1) as usize] += 1;
    }

    // Phase 2: Prefix sum — convert counts to offsets
    let mut sum = 0u32;
    for h in hist.iter_mut() {
        let count = *h;
        *h = sum;
        sum += count;
    }

    // Phase 3: Scatter — the input ordering is the identity, so iterate
    // 0..n directly instead of materializing an index array.
    for idx in 0..n as u32 {
        let bucket = (keys[idx as usize] & MASK1) as usize;
        mid[hist[bucket] as usize] = idx;
        hist[bucket] += 1;
    }

    // Pass 2: sort by bits 11-21 (2048 buckets — max state_index is ~4M for stride 128)
    const BITS2: u32 = 11;
    const SHIFT2: u32 = BITS1;
    const MASK2: u32 = (1 << BITS2) - 1;
    const BUCKETS2: usize = 1 << BITS2; // 2048

    // Phase 1: Histogram — count elements per bucket
    let mut hist2 = [0u32; BUCKETS2];
    for &idx in mid.iter() {
        hist2[((keys[idx as usize] >> SHIFT2) & MASK2) as usize] += 1;
    }

    // Phase 2: Prefix sum — convert counts to offsets
    let mut sum = 0u32;
    for h in hist2.iter_mut() {
        let count = *h;
        *h = sum;
        sum += count;
    }

    // Phase 3: Scatter — place elements in sorted order
    for &idx in mid.iter() {
        let bucket = ((keys[idx as usize] >> SHIFT2) & MASK2) as usize;
        dst[hist2[bucket] as usize] = idx;
        hist2[bucket] += 1;
    }

    // Build groups from sorted order
    if n > 0 {
        let mut start = 0u32;
        let mut current_key = keys[dst[0] as usize];
        for i in 1..n {
            let k = keys[dst[i] as usize];
            if k != current_key {
                scratch.groups.push((current_key, start, i as u32 - start));
                current_key = k;
                start = i as u32;
            }
        }
        scratch.groups.push((current_key, start, n as u32 - start));
    }
}

/// Sort game indices by state_index using 2-pass counting sort.
///
/// Returns sorted indices plus group boundaries for efficient iteration.
/// Allocates fresh buffers; use [`radix_sort_by_state_into`] with a hoisted
/// [`RadixScratch`] when calling in a loop.
pub fn radix_sort_by_state(keys: &[u32], n: usize) -> SortedGroups {
    let mut scratch = RadixScratch::new();
    radix_sort_by_state_into(keys, n, &mut scratch);
    SortedGroups {
        indices: scratch.indices,
        groups: scratch.groups,
    }
}

/// Below this input size the serial sort wins (rayon fork/join overhead).
const PAR_SORT_THRESHOLD: usize = 200_000;

/// Stable parallel counting sort, output identical to
/// [`radix_sort_by_state_into`].
///
/// Each pass: per-chunk histograms (parallel), a bucket-major/chunk-minor
/// exclusive prefix scan (serial, 2048×chunks adds), then an in-order
/// per-chunk scatter (parallel). Chunk-minor ordering preserves the global
/// input order within every bucket, i.e. the sort is stable, so
/// indices/groups are bit-identical to the serial result.
pub fn radix_sort_by_state_into_parallel(keys: &[u32], n: usize, scratch: &mut RadixScratch) {
    use rayon::prelude::*;

    if n < PAR_SORT_THRESHOLD {
        radix_sort_by_state_into(keys, n, scratch);
        return;
    }
    debug_assert_eq!(keys.len(), n);

    const BITS: u32 = 11;
    const MASK: u32 = (1 << BITS) - 1;
    const BUCKETS: usize = 1 << BITS;

    scratch.buf.resize(n, 0);
    scratch.indices.resize(n, 0);
    scratch.sorted_keys.resize(n, 0);
    scratch.groups.clear();

    let n_chunks = rayon::current_num_threads().clamp(1, 64);
    let chunk = n.div_ceil(n_chunks);
    scratch.chunk_hists.resize(n_chunks * BUCKETS, 0);

    // ── Pass 1: bucket by low 11 bits; input order is the identity ──
    let hists = &mut scratch.chunk_hists;
    hists
        .par_chunks_mut(BUCKETS)
        .enumerate()
        .for_each(|(c, h)| {
            h.fill(0);
            let lo = c * chunk;
            let hi = ((c + 1) * chunk).min(n);
            for key in &keys[lo..hi] {
                h[(key & MASK) as usize] += 1;
            }
        });
    // Exclusive prefix: bucket-major, chunk-minor → per-chunk write bases.
    let mut sum = 0u32;
    for b in 0..BUCKETS {
        for c in 0..n_chunks {
            let idx = c * BUCKETS + b;
            let count = hists[idx];
            hists[idx] = sum;
            sum += count;
        }
    }
    // Scatter: each chunk owns disjoint per-(bucket, chunk) segments of mid.
    let mid_addr = scratch.buf.as_mut_ptr() as usize;
    hists
        .par_chunks_mut(BUCKETS)
        .enumerate()
        .for_each(|(c, cursors)| {
            let mid = mid_addr as *mut u32;
            let lo = c * chunk;
            let hi = ((c + 1) * chunk).min(n);
            for i in lo..hi {
                let b = (keys[i] & MASK) as usize;
                // SAFETY: cursor ranges are disjoint across chunks by the
                // prefix-scan construction; every write lands in-bounds
                // (total counts sum to n).
                unsafe { *mid.add(cursors[b] as usize) = i as u32 };
                cursors[b] += 1;
            }
        });

    // ── Pass 2: bucket by bits 11-21, reading mid; also materialize keys ──
    let mid_ref = &scratch.buf;
    let hists = &mut scratch.chunk_hists;
    hists
        .par_chunks_mut(BUCKETS)
        .enumerate()
        .for_each(|(c, h)| {
            h.fill(0);
            let lo = c * chunk;
            let hi = ((c + 1) * chunk).min(n);
            for &idx in &mid_ref[lo..hi] {
                h[((keys[idx as usize] >> BITS) & MASK) as usize] += 1;
            }
        });
    let mut sum = 0u32;
    for b in 0..BUCKETS {
        for c in 0..n_chunks {
            let idx = c * BUCKETS + b;
            let count = hists[idx];
            hists[idx] = sum;
            sum += count;
        }
    }
    let dst_addr = scratch.indices.as_mut_ptr() as usize;
    let sk_addr = scratch.sorted_keys.as_mut_ptr() as usize;
    hists
        .par_chunks_mut(BUCKETS)
        .enumerate()
        .for_each(|(c, cursors)| {
            let dst = dst_addr as *mut u32;
            let sk = sk_addr as *mut u32;
            let lo = c * chunk;
            let hi = ((c + 1) * chunk).min(n);
            for &idx in &mid_ref[lo..hi] {
                let key = keys[idx as usize];
                let b = ((key >> BITS) & MASK) as usize;
                let pos = cursors[b] as usize;
                // SAFETY: same disjoint-segment argument as pass 1.
                unsafe {
                    *dst.add(pos) = idx;
                    *sk.add(pos) = key;
                }
                cursors[b] += 1;
            }
        });

    // ── Group boundaries: sequential scan over the materialized keys ──
    let sk = &scratch.sorted_keys;
    let mut start = 0u32;
    let mut current_key = sk[0];
    for (i, &k) in sk.iter().enumerate().skip(1) {
        if k != current_key {
            scratch.groups.push((current_key, start, i as u32 - start));
            current_key = k;
            start = i as u32;
        }
    }
    scratch.groups.push((current_key, start, n as u32 - start));
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

    #[test]
    fn test_parallel_matches_serial() {
        // The parallel sort must be stable, i.e. produce bit-identical
        // indices AND groups vs the serial sort, above and below the
        // parallel threshold.
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32) % 4_194_304
        };
        for &n in &[
            1usize,
            1000,
            PAR_SORT_THRESHOLD - 1,
            PAR_SORT_THRESHOLD + 12_345,
            300_001,
        ] {
            let keys: Vec<u32> = (0..n).map(|_| next()).collect();
            let mut serial = RadixScratch::new();
            radix_sort_by_state_into(&keys, n, &mut serial);
            let mut parallel = RadixScratch::new();
            radix_sort_by_state_into_parallel(&keys, n, &mut parallel);
            assert_eq!(
                serial.indices, parallel.indices,
                "indices differ at n={}",
                n
            );
            assert_eq!(serial.groups, parallel.groups, "groups differ at n={}", n);
        }
    }

    #[test]
    fn test_scratch_reuse_across_calls() {
        // Reusing one scratch across calls with different keys and sizes
        // must produce the same result as fresh allocations.
        let mut scratch = RadixScratch::new();
        let inputs: Vec<Vec<u32>> = vec![
            vec![5, 3, 8, 3, 1, 5, 8, 1],
            vec![42, 42, 42],
            vec![0, 4_000_000, 128, 256, 4_000_000, 128, 7, 7, 7, 7],
            vec![],
            vec![2048, 0, 2048, 4096, 1],
        ];
        for keys in &inputs {
            radix_sort_by_state_into(keys, keys.len(), &mut scratch);
            let fresh = radix_sort_by_state(keys, keys.len());
            assert_eq!(scratch.indices, fresh.indices);
            assert_eq!(scratch.groups, fresh.groups);
            // Sorted order + group coverage
            for i in 1..scratch.indices.len() {
                assert!(keys[scratch.indices[i - 1] as usize] <= keys[scratch.indices[i] as usize]);
            }
            let total: u32 = scratch.groups.iter().map(|&(_, _, c)| c).sum();
            assert_eq!(total as usize, keys.len());
        }
    }
}
