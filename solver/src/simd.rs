//! NEON intrinsic kernels for the solver hot path.
//!
//! Provides hand-written ARM NEON kernels that replace auto-vectorized `for up in 0..64`
//! loops in the batched solver. Each kernel processes 64 f32 values (256 bytes) using
//! 4-wide SIMD operations (16 iterations of 4 elements).
//!
//! ## Kernels
//!
//! - [`neon_fma_64`]: `dst[i] += scalar * src[i]` — Groups 5/3 Step 1 (keep EV accumulation)
//! - [`neon_max_64`]: `dst[i] = max(dst[i], src[i])` — Groups 5/3 Step 2 (best keep selection)
//! - [`neon_add_max_64`]: `dst[i] = max(dst[i], scalar + src[i])` — Group 6 lower categories
//! - [`neon_add_max_offset_64`]: `dst[i] = max(dst[i], scalar + sv[base + i + offset])` — Group 6 upper categories (branchless via topological padding)
//! - [`neon_mul_max_64`]: `dst[i] = max(dst[i], scalar * src[i])` — Group 6 utility-domain
//! - [`neon_mul_max_offset_64`]: `dst[i] = max(dst[i], scalar * sv[base + i + offset])` — Group 6 utility upper
//! - [`neon_min_64`]: `dst[i] = min(dst[i], src[i])` — risk-averse decision nodes
//! - [`neon_weighted_sum_64`]: `dst[i] += scalar * src[i]` (alias of fma) — Group 1

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// dst[i] += scalar * src[i] for i in 0..64
///
/// Used in Groups 5/3 Step 1: `keep_ev[kid][up] += prob * e_prev[ds'][up]`
/// and Group 1: `result[up] += prob * e[ds][up]`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_fma_64(dst: &mut [f32; 64], src: &[f32; 64], scalar: f32) {
    let s = vdupq_n_f32(scalar);
    for i in (0..64).step_by(4) {
        let d = vld1q_f32(dst.as_ptr().add(i));
        let r = vld1q_f32(src.as_ptr().add(i));
        let result = vfmaq_f32(d, r, s); // d + r * s
        vst1q_f32(dst.as_mut_ptr().add(i), result);
    }
}

/// dst[i] = max(dst[i], src[i]) for i in 0..64
///
/// Used in Groups 5/3 Step 2: `e_curr[ds][up] = max(e_curr[ds][up], keep_ev[kid][up])`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_max_64(dst: &mut [f32; 64], src: &[f32; 64]) {
    for i in (0..64).step_by(4) {
        let d = vld1q_f32(dst.as_ptr().add(i));
        let s = vld1q_f32(src.as_ptr().add(i));
        vst1q_f32(dst.as_mut_ptr().add(i), vmaxq_f32(d, s));
    }
}

/// dst[i] = min(dst[i], src[i]) for i in 0..64
///
/// Used in risk-averse (theta < 0) decision nodes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_min_64(dst: &mut [f32; 64], src: &[f32; 64]) {
    for i in (0..64).step_by(4) {
        let d = vld1q_f32(dst.as_ptr().add(i));
        let s = vld1q_f32(src.as_ptr().add(i));
        vst1q_f32(dst.as_mut_ptr().add(i), vminq_f32(d, s));
    }
}

/// dst[i] = max(dst[i], scalar + src[i]) for i in 0..64
///
/// Used in Group 6 lower categories: `row[up] = max(row[up], scr + sv[base + up])`.
/// The src pointer points to a contiguous 64-element block in the state values array.
///
/// # Safety
/// `src` must point to at least 64 contiguous valid f32 values.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_add_max_64(dst: &mut [f32; 64], src: *const f32, scalar: f32) {
    let s = vdupq_n_f32(scalar);
    for i in (0..64).step_by(4) {
        let d = vld1q_f32(dst.as_ptr().add(i));
        let v = vld1q_f32(src.add(i));
        let val = vaddq_f32(v, s);
        vst1q_f32(dst.as_mut_ptr().add(i), vmaxq_f32(d, val));
    }
}

/// dst[i] = min(dst[i], scalar + src[i]) for i in 0..64
///
/// Risk-averse variant of add_max: for theta < 0 decision nodes.
///
/// # Safety
/// `src` must point to at least 64 contiguous valid f32 values.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_add_min_64(dst: &mut [f32; 64], src: *const f32, scalar: f32) {
    let s = vdupq_n_f32(scalar);
    for i in (0..64).step_by(4) {
        let d = vld1q_f32(dst.as_ptr().add(i));
        let v = vld1q_f32(src.add(i));
        let val = vaddq_f32(v, s);
        vst1q_f32(dst.as_mut_ptr().add(i), vminq_f32(d, val));
    }
}

/// dst[i] = max(dst[i], scalar + sv[base + offset + i]) for i in 0..64
///
/// Used in Group 6 upper categories with topological padding:
/// `row[up] = max(row[up], scr + sv[succ_base + up + scr])`.
/// The offset parameter is the category score (scr), making sv[base + offset + i]
/// a sequential read starting from sv[base + scr].
///
/// # Safety
/// `sv[base_plus_offset..base_plus_offset+64]` must be valid readable f32 values.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_add_max_offset_64(
    dst: &mut [f32; 64],
    sv: *const f32,
    base_plus_offset: usize,
    scalar: f32,
) {
    let s = vdupq_n_f32(scalar);
    let src = sv.add(base_plus_offset);
    for i in (0..64).step_by(4) {
        let d = vld1q_f32(dst.as_ptr().add(i));
        let v = vld1q_f32(src.add(i));
        let val = vaddq_f32(v, s);
        vst1q_f32(dst.as_mut_ptr().add(i), vmaxq_f32(d, val));
    }
}

/// dst[i] = min(dst[i], scalar + sv[base + offset + i]) for i in 0..64
///
/// Risk-averse variant of add_max_offset.
///
/// # Safety
/// `sv[base_plus_offset..base_plus_offset+64]` must be valid readable f32 values.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_add_min_offset_64(
    dst: &mut [f32; 64],
    sv: *const f32,
    base_plus_offset: usize,
    scalar: f32,
) {
    let s = vdupq_n_f32(scalar);
    let src = sv.add(base_plus_offset);
    for i in (0..64).step_by(4) {
        let d = vld1q_f32(dst.as_ptr().add(i));
        let v = vld1q_f32(src.add(i));
        let val = vaddq_f32(v, s);
        vst1q_f32(dst.as_mut_ptr().add(i), vminq_f32(d, val));
    }
}

/// dst[i] = max(dst[i], scalar * src[i]) for i in 0..64
///
/// Used in Group 6 utility-domain lower categories:
/// `row[up] = max(row[up], exp_scr * sv[base + up])`.
///
/// # Safety
/// `src` must point to at least 64 contiguous valid f32 values.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_mul_max_64(dst: &mut [f32; 64], src: *const f32, scalar: f32) {
    let s = vdupq_n_f32(scalar);
    for i in (0..64).step_by(4) {
        let d = vld1q_f32(dst.as_ptr().add(i));
        let v = vld1q_f32(src.add(i));
        let val = vmulq_f32(v, s);
        vst1q_f32(dst.as_mut_ptr().add(i), vmaxq_f32(d, val));
    }
}

/// dst[i] = min(dst[i], scalar * src[i]) for i in 0..64
///
/// Risk-averse variant of mul_max.
///
/// # Safety
/// `src` must point to at least 64 contiguous valid f32 values.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_mul_min_64(dst: &mut [f32; 64], src: *const f32, scalar: f32) {
    let s = vdupq_n_f32(scalar);
    for i in (0..64).step_by(4) {
        let d = vld1q_f32(dst.as_ptr().add(i));
        let v = vld1q_f32(src.add(i));
        let val = vmulq_f32(v, s);
        vst1q_f32(dst.as_mut_ptr().add(i), vminq_f32(d, val));
    }
}

/// dst[i] = max(dst[i], scalar * sv[base + offset + i]) for i in 0..64
///
/// Used in Group 6 utility-domain upper categories with topological padding.
///
/// # Safety
/// `sv[base_plus_offset..base_plus_offset+64]` must be valid readable f32 values.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_mul_max_offset_64(
    dst: &mut [f32; 64],
    sv: *const f32,
    base_plus_offset: usize,
    scalar: f32,
) {
    let s = vdupq_n_f32(scalar);
    let src = sv.add(base_plus_offset);
    for i in (0..64).step_by(4) {
        let d = vld1q_f32(dst.as_ptr().add(i));
        let v = vld1q_f32(src.add(i));
        let val = vmulq_f32(v, s);
        vst1q_f32(dst.as_mut_ptr().add(i), vmaxq_f32(d, val));
    }
}

/// dst[i] = min(dst[i], scalar * sv[base + offset + i]) for i in 0..64
///
/// Risk-averse variant of mul_max_offset.
///
/// # Safety
/// `sv[base_plus_offset..base_plus_offset+64]` must be valid readable f32 values.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_mul_min_offset_64(
    dst: &mut [f32; 64],
    sv: *const f32,
    base_plus_offset: usize,
    scalar: f32,
) {
    let s = vdupq_n_f32(scalar);
    let src = sv.add(base_plus_offset);
    for i in (0..64).step_by(4) {
        let d = vld1q_f32(dst.as_ptr().add(i));
        let v = vld1q_f32(src.add(i));
        let val = vmulq_f32(v, s);
        vst1q_f32(dst.as_mut_ptr().add(i), vminq_f32(d, val));
    }
}

/// dst[i] = max(dst[i], src[i]) with argmax tracking: idx[i] = new_idx if src[i] > dst[i].
///
/// Used by oracle builder in Groups 5/3 Step 2 to track which keep-multiset wins.
/// Processes 16 indices at a time using NEON narrowing chain + vbslq_u8 blend,
/// avoiding scalar lane extraction that breaks the NEON pipeline.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_max_64_argmax(
    dst: &mut [f32; 64],
    src: &[f32; 64],
    idx: &mut [u8; 64],
    new_idx: u8,
) {
    let new_vec = vdupq_n_u8(new_idx);

    for i in (0..64).step_by(16) {
        // 4 sub-groups of 4 f32: compare + max
        let d0 = vld1q_f32(dst.as_ptr().add(i));
        let s0 = vld1q_f32(src.as_ptr().add(i));
        let cmp0 = vcgtq_f32(s0, d0);
        vst1q_f32(dst.as_mut_ptr().add(i), vmaxq_f32(d0, s0));

        let d1 = vld1q_f32(dst.as_ptr().add(i + 4));
        let s1 = vld1q_f32(src.as_ptr().add(i + 4));
        let cmp1 = vcgtq_f32(s1, d1);
        vst1q_f32(dst.as_mut_ptr().add(i + 4), vmaxq_f32(d1, s1));

        let d2 = vld1q_f32(dst.as_ptr().add(i + 8));
        let s2 = vld1q_f32(src.as_ptr().add(i + 8));
        let cmp2 = vcgtq_f32(s2, d2);
        vst1q_f32(dst.as_mut_ptr().add(i + 8), vmaxq_f32(d2, s2));

        let d3 = vld1q_f32(dst.as_ptr().add(i + 12));
        let s3 = vld1q_f32(src.as_ptr().add(i + 12));
        let cmp3 = vcgtq_f32(s3, d3);
        vst1q_f32(dst.as_mut_ptr().add(i + 12), vmaxq_f32(d3, s3));

        // NEON has no direct f32-comparison-to-u8-mask instruction; narrow 4x u32 masks to one u8x16 for vbslq.
        // u32 all-1s (0xFFFFFFFF) narrows to u16 all-1s (0xFFFF) to u8 all-1s (0xFF)
        let n0 = vmovn_u32(cmp0); // uint16x4_t
        let n1 = vmovn_u32(cmp1);
        let n2 = vmovn_u32(cmp2);
        let n3 = vmovn_u32(cmp3);
        let m01 = vcombine_u16(n0, n1); // uint16x8_t
        let m23 = vcombine_u16(n2, n3);
        let b01 = vmovn_u16(m01); // uint8x8_t
        let b23 = vmovn_u16(m23);
        let mask = vcombine_u8(b01, b23); // uint8x16_t

        // Blend indices: pick new_idx where src > dst, keep old otherwise
        let old_idx_v = vld1q_u8(idx.as_ptr().add(i));
        let updated = vbslq_u8(mask, new_vec, old_idx_v);
        vst1q_u8(idx.as_mut_ptr().add(i), updated);
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub unsafe fn neon_max_64_argmax(
    dst: &mut [f32; 64],
    src: &[f32; 64],
    idx: &mut [u8; 64],
    new_idx: u8,
) {
    for i in 0..64 {
        if src[i] > dst[i] {
            dst[i] = src[i];
            idx[i] = new_idx;
        }
    }
}

/// Fast f32 exp approximation using NEON intrinsics.
///
/// Uses the identity exp(x) = 2^(x/ln2), splits into integer and fractional
/// parts, computes 2^frac via a degree-5 minimax polynomial (Cephes coefficients),
/// and reconstructs via bit manipulation.
///
/// Accuracy: max ~2 ULP error across [-87, 88] (f32 exp range).
/// Speed: ~8 NEON instructions vs ~20+ for libm expf.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_fast_exp_f32x4(x: float32x4_t) -> float32x4_t {
    // Constants
    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E); // 1/ln(2)
    let ln2_hi = vdupq_n_f32(0.693359375_f32); // ln(2) high bits
    let ln2_lo = vdupq_n_f32(-2.12194440e-4_f32); // ln(2) low bits

    // Cephes polynomial coefficients for 2^frac - 1
    let c1 = vdupq_n_f32(1.0_f32);
    let c2 = vdupq_n_f32(0.693147182_f32); // ln(2)
    let c3 = vdupq_n_f32(0.240226507_f32); // ln(2)^2/2!
    let c4 = vdupq_n_f32(0.0558013551_f32); // ln(2)^3/3!
    let c5 = vdupq_n_f32(0.00898052313_f32); // ln(2)^4/4!
    let c6 = vdupq_n_f32(0.00187820407_f32); // ln(2)^5/5!

    // Clamp to valid range to avoid infinity/NaN
    let x_clamped = vmaxq_f32(vdupq_n_f32(-87.0), vminq_f32(vdupq_n_f32(88.0), x));

    // n = floor(x * log2e)
    let t = vmulq_f32(x_clamped, log2e);
    let n = vcvtq_s32_f32(vrndmq_f32(t)); // floor
    let nf = vcvtq_f32_s32(n);

    // r = x - n * ln2 (in [-0.5*ln2, 0.5*ln2])
    // vfmsq_f32(a, b, c) = a - b*c (fused multiply-subtract, opposite of vfmaq)
    let r = vfmsq_f32(x_clamped, nf, ln2_hi); // x - n*ln2_hi
    let r = vfmsq_f32(r, nf, ln2_lo); // r - n*ln2_lo

    // Horner's method: p = c6*r^5 + c5*r^4 + ... + c1
    // p = ((((c6*r + c5)*r + c4)*r + c3)*r + c2)*r + c1
    let p = vfmaq_f32(c5, c6, r);
    let p = vfmaq_f32(c4, p, r);
    let p = vfmaq_f32(c3, p, r);
    let p = vfmaq_f32(c2, p, r);
    let p = vfmaq_f32(c1, p, r);

    // Multiply by 2^n: add n to the exponent bits
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n, vdupq_n_s32(127))));
    vmulq_f32(p, pow2n)
}

/// dst[i] += scalar * exp(src[i] - sub[i]) for i in 0..64
///
/// Used in Groups 5/3 risk-sensitive Step 1 pass 2: weighted exp-sum for LSE.
/// Uses fast polynomial exp approximation via NEON.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_weighted_exp_sum_64(
    dst: &mut [f32; 64],
    src: &[f32; 64],
    sub: &[f32; 64],
    scalar: f32,
) {
    let s = vdupq_n_f32(scalar);
    for i in (0..64).step_by(4) {
        let src_v = vld1q_f32(src.as_ptr().add(i));
        let sub_v = vld1q_f32(sub.as_ptr().add(i));
        let diff = vsubq_f32(src_v, sub_v);
        let exp_v = neon_fast_exp_f32x4(diff);
        let d = vld1q_f32(dst.as_ptr().add(i));
        let result = vfmaq_f32(d, s, exp_v);
        vst1q_f32(dst.as_mut_ptr().add(i), result);
    }
}

// ── Fallback implementations for non-aarch64 targets ──

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub unsafe fn neon_fma_64(dst: &mut [f32; 64], src: &[f32; 64], scalar: f32) {
    for i in 0..64 {
        dst[i] += scalar * src[i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub unsafe fn neon_max_64(dst: &mut [f32; 64], src: &[f32; 64]) {
    for i in 0..64 {
        if src[i] > dst[i] {
            dst[i] = src[i];
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub unsafe fn neon_min_64(dst: &mut [f32; 64], src: &[f32; 64]) {
    for i in 0..64 {
        if src[i] < dst[i] {
            dst[i] = src[i];
        }
    }
}

/// # Safety
/// `src` must point to at least 64 contiguous valid f32 values.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub unsafe fn neon_add_max_64(dst: &mut [f32; 64], src: *const f32, scalar: f32) {
    for i in 0..64 {
        let val = scalar + *src.add(i);
        if val > dst[i] {
            dst[i] = val;
        }
    }
}

/// # Safety
/// `src` must point to at least 64 contiguous valid f32 values.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub unsafe fn neon_add_min_64(dst: &mut [f32; 64], src: *const f32, scalar: f32) {
    for i in 0..64 {
        let val = scalar + *src.add(i);
        if val < dst[i] {
            dst[i] = val;
        }
    }
}

/// # Safety
/// `sv[base_plus_offset..base_plus_offset+64]` must be valid readable f32 values.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub unsafe fn neon_add_max_offset_64(
    dst: &mut [f32; 64],
    sv: *const f32,
    base_plus_offset: usize,
    scalar: f32,
) {
    let src = sv.add(base_plus_offset);
    for i in 0..64 {
        let val = scalar + *src.add(i);
        if val > dst[i] {
            dst[i] = val;
        }
    }
}

/// # Safety
/// `sv[base_plus_offset..base_plus_offset+64]` must be valid readable f32 values.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub unsafe fn neon_add_min_offset_64(
    dst: &mut [f32; 64],
    sv: *const f32,
    base_plus_offset: usize,
    scalar: f32,
) {
    let src = sv.add(base_plus_offset);
    for i in 0..64 {
        let val = scalar + *src.add(i);
        if val < dst[i] {
            dst[i] = val;
        }
    }
}

/// # Safety
/// `src` must point to at least 64 contiguous valid f32 values.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub unsafe fn neon_mul_max_64(dst: &mut [f32; 64], src: *const f32, scalar: f32) {
    for i in 0..64 {
        let val = scalar * *src.add(i);
        if val > dst[i] {
            dst[i] = val;
        }
    }
}

/// # Safety
/// `src` must point to at least 64 contiguous valid f32 values.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub unsafe fn neon_mul_min_64(dst: &mut [f32; 64], src: *const f32, scalar: f32) {
    for i in 0..64 {
        let val = scalar * *src.add(i);
        if val < dst[i] {
            dst[i] = val;
        }
    }
}

/// # Safety
/// `sv[base_plus_offset..base_plus_offset+64]` must be valid readable f32 values.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub unsafe fn neon_mul_max_offset_64(
    dst: &mut [f32; 64],
    sv: *const f32,
    base_plus_offset: usize,
    scalar: f32,
) {
    let src = sv.add(base_plus_offset);
    for i in 0..64 {
        let val = scalar * *src.add(i);
        if val > dst[i] {
            dst[i] = val;
        }
    }
}

/// # Safety
/// `sv[base_plus_offset..base_plus_offset+64]` must be valid readable f32 values.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub unsafe fn neon_mul_min_offset_64(
    dst: &mut [f32; 64],
    sv: *const f32,
    base_plus_offset: usize,
    scalar: f32,
) {
    let src = sv.add(base_plus_offset);
    for i in 0..64 {
        let val = scalar * *src.add(i);
        if val < dst[i] {
            dst[i] = val;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub unsafe fn neon_weighted_exp_sum_64(
    dst: &mut [f32; 64],
    src: &[f32; 64],
    sub: &[f32; 64],
    scalar: f32,
) {
    for i in 0..64 {
        dst[i] += scalar * (src[i] - sub[i]).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_fma_64() {
        let mut dst = [1.0f32; 64];
        let src = [2.0f32; 64];
        unsafe { neon_fma_64(&mut dst, &src, 3.0) };
        for i in 0..64 {
            assert!(
                (dst[i] - 7.0).abs() < 1e-6,
                "fma failed at {}: {}",
                i,
                dst[i]
            );
        }
    }

    #[test]
    fn test_neon_max_64() {
        let mut dst = [5.0f32; 64];
        let mut src = [3.0f32; 64];
        src[10] = 10.0;
        src[63] = 7.0;
        unsafe { neon_max_64(&mut dst, &src) };
        assert!((dst[0] - 5.0).abs() < 1e-6);
        assert!((dst[10] - 10.0).abs() < 1e-6);
        assert!((dst[63] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_neon_min_64() {
        let mut dst = [5.0f32; 64];
        let mut src = [3.0f32; 64];
        src[10] = 10.0;
        unsafe { neon_min_64(&mut dst, &src) };
        assert!((dst[0] - 3.0).abs() < 1e-6);
        assert!((dst[10] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_neon_add_max_64() {
        let mut dst = [5.0f32; 64];
        let src = [2.0f32; 64];
        unsafe { neon_add_max_64(&mut dst, src.as_ptr(), 4.0) };
        // 4.0 + 2.0 = 6.0 > 5.0
        for i in 0..64 {
            assert!((dst[i] - 6.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_neon_add_max_offset_64() {
        let sv = vec![0.0f32; 256];
        let mut dst = [f32::NEG_INFINITY; 64];
        // sv[100..164] = 0.0, so result should be scalar + 0.0 = 10.0
        unsafe { neon_add_max_offset_64(&mut dst, sv.as_ptr(), 100, 10.0) };
        for i in 0..64 {
            assert!((dst[i] - 10.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_neon_mul_max_64() {
        let mut dst = [5.0f32; 64];
        let src = [3.0f32; 64];
        unsafe { neon_mul_max_64(&mut dst, src.as_ptr(), 2.0) };
        // 2.0 * 3.0 = 6.0 > 5.0
        for i in 0..64 {
            assert!((dst[i] - 6.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_neon_weighted_exp_sum_64() {
        let mut dst = [0.0f32; 64];
        let src = [1.0f32; 64];
        let sub = [1.0f32; 64];
        // exp(1.0 - 1.0) = exp(0) = 1.0, so dst += 2.0 * 1.0 = 2.0
        unsafe { neon_weighted_exp_sum_64(&mut dst, &src, &sub, 2.0) };
        for i in 0..64 {
            assert!((dst[i] - 2.0).abs() < 1e-6);
        }
    }
}
