#![allow(dead_code)]

pub fn eq_padded_bytes_simd(a: &[u8], b: &[u8]) -> bool {
    assert_eq!(a.len(), b.len(), "Slices must have the same length");

    #[cfg(target_arch = "aarch64")]
    {
        log::debug!("eq_padded_bytes_simd: using NEON (aarch64)");
        unsafe { arm::eq_padded_bytes_neon(a, b) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        x86::eq_padded_bytes_x86(a, b)
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        debug!("eq_padded_bytes_simd: using scalar fallback (other arch)");
        a == b
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use core::arch::aarch64::*;

    /// NEON implementation: 16 bytes per iteration + scalar tail.
    /// # Safety
    /// Assumes a.len() == b.len().
    #[inline]
    pub unsafe fn eq_padded_bytes_neon(a: &[u8], b: &[u8]) -> bool {
        let len = a.len();
        let mut i = 0;

        // 16-byte blocks
        while i + 16 <= len {
            let pa = unsafe { a.as_ptr().add(i) };
            let pb = unsafe { b.as_ptr().add(i) };

            let va = unsafe { vld1q_u8(pa) };
            let vb = unsafe { vld1q_u8(pb) };

            // 0xFF equal, 0x00 where not
            let eq_mask = unsafe { vceqq_u8(va, vb) };

            // Take the minimum across all lanes; if any lane is 0x00, min == 0x00
            let min_val: u8 = unsafe { vminvq_u8(eq_mask) };
            if min_val != 0xFF {
                return false;
            }

            i += 16;
        }

        // Scalar tail (for non-multiple-of-16 lengths)
        while i < len {
            if unsafe { *a.get_unchecked(i) } != unsafe { *b.get_unchecked(i) } {
                return false;
            }
            i += 1;
        }

        true
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::*;
    use log::debug;

    #[inline]
    pub fn eq_padded_bytes_x86(a: &[u8], b: &[u8]) -> bool {
        assert_eq!(a.len(), b.len(), "Slices must have the same length");

        #[cfg(target_feature = "avx512bw")]
        {
            debug!("eq_padded_bytes_simd: using AVX-512BW implementation");
            return unsafe { eq_padded_bytes_avx512(a, b) };
        }

        #[cfg(all(not(target_feature = "avx512bw"), target_feature = "avx2"))]
        {
            debug!("eq_padded_bytes_simd: using AVX2 implementation");
            return unsafe { eq_padded_bytes_avx2(a, b) };
        }

        #[cfg(all(
            not(target_feature = "avx512bw"),
            not(target_feature = "avx2"),
            target_feature = "sse4.1"
        ))]
        {
            debug!("eq_padded_bytes_simd: using SSE4.1 implementation");
            return unsafe { eq_padded_bytes_sse41(a, b) };
        }

        #[cfg(not(any(
            target_feature = "avx512bw",
            target_feature = "avx2",
            target_feature = "sse4.1"
        )))]
        {
            debug!("eq_padded_bytes_simd: using scalar fallback (no SIMD features enabled)");
            return a == b;
        }
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn eq_padded_bytes_sse41(a: &[u8], b: &[u8]) -> bool {
        let len = a.len();
        let mut i = 0;

        while i + 16 <= len {
            let pa = unsafe { a.as_ptr().add(i) as *const __m128i };
            let pb = unsafe { b.as_ptr().add(i) as *const __m128i };

            let va = unsafe { _mm_loadu_si128(pa) };
            let vb = unsafe { _mm_loadu_si128(pb) };

            let cmp = _mm_cmpeq_epi8(va, vb); // 0xFF where equal
            let mask = _mm_movemask_epi8(cmp); // 16 bits

            if mask != 0xFFFF {
                return false;
            }

            i += 16;
        }

        while i < len {
            if unsafe { *a.get_unchecked(i) != *b.get_unchecked(i) } {
                return false;
            }
            i += 1;
        }

        true
    }

    #[target_feature(enable = "avx2")]
    unsafe fn eq_padded_bytes_avx2(a: &[u8], b: &[u8]) -> bool {
        let len = a.len();
        let mut i = 0;

        while i + 32 <= len {
            let pa = unsafe { a.as_ptr().add(i) as *const __m256i };
            let pb = unsafe { b.as_ptr().add(i) as *const __m256i };

            let va = unsafe { _mm256_loadu_si256(pa) };
            let vb = unsafe { _mm256_loadu_si256(pb) };

            let cmp = _mm256_cmpeq_epi8(va, vb);
            let mask = _mm256_movemask_epi8(cmp);

            if mask != -1i32 {
                return false;
            }

            i += 32;
        }

        // 16-byte SSE tail
        while i + 16 <= len {
            let pa = unsafe { a.as_ptr().add(i) as *const __m128i };
            let pb = unsafe { b.as_ptr().add(i) as *const __m128i };

            let va = unsafe { _mm_loadu_si128(pa) };
            let vb = unsafe { _mm_loadu_si128(pb) };

            let cmp = _mm_cmpeq_epi8(va, vb);
            let mask = _mm_movemask_epi8(cmp);

            if mask != 0xFFFF {
                return false;
            }

            i += 16;
        }

        // Scalar tail
        while i < len {
            if unsafe { *a.get_unchecked(i) != *b.get_unchecked(i) } {
                return false;
            }
            i += 1;
        }

        true
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn eq_padded_bytes_avx512(a: &[u8], b: &[u8]) -> bool {
        let len = a.len();
        let mut i = 0;

        while i + 64 <= len {
            let pa = unsafe { a.as_ptr().add(i) as *const __m512i };
            let pb = unsafe { b.as_ptr().add(i) as *const __m512i };

            let va = unsafe { _mm512_loadu_si512(pa) };
            let vb = unsafe { _mm512_loadu_si512(pb) };

            let mask: __mmask64 = _mm512_cmpeq_epi8_mask(va, vb);

            if mask != !0u64 {
                return false;
            }

            i += 64;
        }

        while i < len {
            if unsafe { *a.get_unchecked(i) != *b.get_unchecked(i) } {
                return false;
            }
            i += 1;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pad_to_multiple_of_16(bytes: &[u8], pad: u8) -> Vec<u8> {
        let mut out = bytes.to_vec();
        let rem = out.len() % 16;
        if rem != 0 {
            out.extend(core::iter::repeat(pad).take(16 - rem));
        }
        out
    }

    #[test]
    fn equal_strings_same_padding() {
        let a = pad_to_multiple_of_16(b"hello", 0);
        let b = pad_to_multiple_of_16(b"hello", 0);

        assert!(eq_padded_bytes_simd(&a, &b));
    }

    #[test]
    fn equal_strings_different_content_fail() {
        let a = pad_to_multiple_of_16(b"hello", 0);
        let b = pad_to_multiple_of_16(b"hellp", 0);

        assert!(!eq_padded_bytes_simd(&a, &b));
    }

    #[test]
    fn equal_up_to_padding() {
        // Same logical string, different padding byte, but we pad both with the same
        // value to keep the comparison simple and “padding-free” as requested.
        let base = b"neon is nice";

        let a = pad_to_multiple_of_16(base, 0x00);
        let b = pad_to_multiple_of_16(base, 0x00);
        assert!(eq_padded_bytes_simd(&a, &b));
    }

    #[test]
    #[should_panic]
    fn mismatched_length_panics() {
        let a = pad_to_multiple_of_16(b"foo", 0);
        let mut b = pad_to_multiple_of_16(b"foo", 0);
        b.extend_from_slice(&[0u8; 16]); // make b longer

        let _ = eq_padded_bytes_simd(&a, &b);
    }

    fn pad_to_multiple_of(block: usize, bytes: &[u8], pad: u8) -> Vec<u8> {
        let mut out = bytes.to_vec();
        let rem = out.len() % block;
        if rem != 0 {
            out.extend(core::iter::repeat(pad).take(block - rem));
        }
        out
    }

    #[test]
    fn eq_padded_bytes_simd_equal() {
        let a = pad_to_multiple_of(64, b"hello neon/x86", 0);
        let b = pad_to_multiple_of(64, b"hello neon/x86", 0);
        assert!(eq_padded_bytes_simd(&a, &b));
    }

    #[test]
    fn eq_padded_bytes_simd_diff() {
        let a = pad_to_multiple_of(64, b"hello neon/x86", 0);
        let b = pad_to_multiple_of(64, b"hello neom/x86", 0);
        assert!(!eq_padded_bytes_simd(&a, &b));
    }
}
