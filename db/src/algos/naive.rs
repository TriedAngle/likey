use crate::like::{LiteralAlgorithm, RowLiteralSearch};
use crate::storage::utf8::{Utf8Column, Utf8Row};

use super::utf8_shared::{
    ByteNeedle, byte_index_symbols, byte_literal_len, compile_byte_literal, matches_at_bytes,
    utf8_row_len,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct Naive;
#[derive(Debug, Clone, Copy, Default)]
pub struct NaiveScalar;
#[derive(Debug, Clone, Copy, Default)]
pub struct NaiveVectorized;
#[derive(Debug, Clone, Copy, Default)]
pub struct NaiveVectorizedV2;
#[derive(Debug, Clone, Copy, Default)]
pub struct NaiveMixed;

macro_rules! impl_naive_literal_algorithm {
    ($ty:ty) => {
        impl LiteralAlgorithm for $ty {
            type Needle = ByteNeedle;
            type State = ();

            const SUPPORTS_UNDERSCORE: bool = false;

            #[inline]
            fn compile_literal(src: &str) -> Option<Self::Needle> {
                compile_byte_literal(src)
            }

            #[inline]
            fn build_state(_needle: &Self::Needle) -> Self::State {
                ()
            }

            #[inline]
            fn literal_len(needle: &Self::Needle) -> u32 {
                byte_literal_len(needle)
            }

            #[inline]
            fn index_symbols(needle: &Self::Needle) -> Option<Box<[u8]>> {
                byte_index_symbols(needle)
            }
        }
    };
}

impl_naive_literal_algorithm!(Naive);
impl_naive_literal_algorithm!(NaiveScalar);
impl_naive_literal_algorithm!(NaiveVectorized);
impl_naive_literal_algorithm!(NaiveVectorizedV2);
impl_naive_literal_algorithm!(NaiveMixed);

macro_rules! impl_naive_row_search {
    ($ty:ty, $find:path) => {
        impl<'db> RowLiteralSearch<Utf8Column<'db>> for $ty {
            #[inline]
            fn row_len<'r>(row: &Utf8Row<'r>) -> u32 {
                utf8_row_len(row)
            }

            #[inline(always)]
            fn matches_at<'r>(
                row: &Utf8Row<'r>,
                pos: u32,
                needle: &Self::Needle,
                _state: &Self::State,
            ) -> bool {
                matches_at_bytes(row, pos, needle)
            }

            #[inline]
            fn find_from<'r>(
                row: &Utf8Row<'r>,
                from: u32,
                needle: &Self::Needle,
                _state: &Self::State,
            ) -> Option<u32> {
                let text = row.bytes();
                let pat = needle.bytes();
                let from = from as usize;

                if from > text.len() {
                    return None;
                }
                $find(&text[from..], pat).map(|pos| (pos + from) as u32)
            }
        }
    };
}

impl_naive_row_search!(Naive, naive_find);
impl_naive_row_search!(NaiveScalar, naive_find_scalar);
impl_naive_row_search!(NaiveVectorized, naive_find_vectorized);
impl_naive_row_search!(NaiveVectorizedV2, naive_find_vectorized_v2);
impl_naive_row_search!(NaiveMixed, naive_find_mixed);

#[inline]
pub fn naive_find_scalar(text: &[u8], pattern: &[u8]) -> Option<usize> {
    let n = text.len();
    let m = pattern.len();

    if m == 0 {
        return Some(0);
    }
    if m > n {
        return None;
    }

    for i in 0..=n - m {
        let mut matched = true;
        for j in 0..m {
            if text[i + j] != pattern[j] {
                matched = false;
                break;
            }
        }
        if matched {
            return Some(i);
        }
    }

    None
}

#[inline]
pub fn naive_find_mixed(text: &[u8], pattern: &[u8]) -> Option<usize> {
    let m = pattern.len();
    let n = text.len();

    if m <= 3 || n < 64 {
        return naive_find_scalar(text, pattern);
    }

    naive_find_vectorized_v2(text, pattern)
}

#[inline]
pub fn naive_find_vectorized(text: &[u8], pattern: &[u8]) -> Option<usize> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: guarded by cfg for aarch64+neon.
        return unsafe { neon::naive_find_neon(text, pattern) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is baseline on x86_64.
        return unsafe { x86::naive_find_sse2(text, pattern) };
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        target_arch = "x86_64"
    )))]
    {
        naive_find_scalar(text, pattern)
    }
}

#[inline]
pub fn naive_find_vectorized_v2(text: &[u8], pattern: &[u8]) -> Option<usize> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: guarded by cfg for aarch64+neon.
        return unsafe { neon::naive_find_neon_v2(text, pattern) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is baseline on x86_64.
        return unsafe { x86::naive_find_sse2_v2(text, pattern) };
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        target_arch = "x86_64"
    )))]
    {
        naive_find_scalar(text, pattern)
    }
}

/// Default naive search. Keeps the old behavior: NEON on aarch64 when enabled,
/// otherwise scalar. Use `NaiveMixed` or `NaiveVectorizedV2` to force the
/// first/last-byte vector prefilter variants.
#[inline]
pub fn naive_find(text: &[u8], pattern: &[u8]) -> Option<usize> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: guarded by cfg for aarch64+neon.
        return unsafe { neon::naive_find_neon(text, pattern) };
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        naive_find_scalar(text, pattern)
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod neon {
    use core::arch::aarch64::*;

    /// NEON-accelerated naive string search.
    ///
    /// # Safety
    /// Requires aarch64 NEON support; guarded by cfg at call sites.
    pub unsafe fn naive_find_neon(text: &[u8], pattern: &[u8]) -> Option<usize> {
        let n = text.len();
        let m = pattern.len();

        if m == 0 {
            return Some(0);
        }
        if m > n {
            return None;
        }

        let first = pattern[0];
        let first_vec = unsafe { vdupq_n_u8(first) };
        let chunk_size = 16usize;

        let mut i = 0usize;
        while i + chunk_size <= n {
            let ptr = unsafe { text.as_ptr().add(i) };
            let chunk = unsafe { vld1q_u8(ptr) };
            let cmp = unsafe { vceqq_u8(chunk, first_vec) };

            let mut lanes = [0u8; 16];
            unsafe { vst1q_u8(lanes.as_mut_ptr(), cmp) };

            for (lane, &value) in lanes.iter().enumerate() {
                if value == 0xFF {
                    let cand = i + lane;
                    if cand + m <= n && &text[cand..cand + m] == pattern {
                        return Some(cand);
                    }
                }
            }

            i += chunk_size;
        }

        while i + m <= n {
            if &text[i..i + m] == pattern {
                return Some(i);
            }
            i += 1;
        }

        None
    }

    /// NEON-accelerated naive string search with first/last-byte prefilter.
    ///
    /// # Safety
    /// Requires aarch64 NEON support; guarded by cfg at call sites.
    pub unsafe fn naive_find_neon_v2(text: &[u8], pattern: &[u8]) -> Option<usize> {
        let n = text.len();
        let m = pattern.len();

        if m == 0 {
            return Some(0);
        }
        if m > n {
            return None;
        }
        if m == 1 {
            return text.iter().position(|&b| b == pattern[0]);
        }

        let first_vec = unsafe { vdupq_n_u8(pattern[0]) };
        let last_vec = unsafe { vdupq_n_u8(pattern[m - 1]) };
        let chunk_size = 16usize;
        let last_off = m - 1;

        let mut i = 0usize;
        while i + chunk_size + last_off <= n {
            let ptr_first = unsafe { text.as_ptr().add(i) };
            let ptr_last = unsafe { text.as_ptr().add(i + last_off) };

            let chunk_first = unsafe { vld1q_u8(ptr_first) };
            let chunk_last = unsafe { vld1q_u8(ptr_last) };

            let cmp_first = unsafe { vceqq_u8(chunk_first, first_vec) };
            let cmp_last = unsafe { vceqq_u8(chunk_last, last_vec) };
            let cmp = unsafe { vandq_u8(cmp_first, cmp_last) };

            let mut lanes = [0u8; 16];
            unsafe { vst1q_u8(lanes.as_mut_ptr(), cmp) };

            for (lane, &value) in lanes.iter().enumerate() {
                if value == 0xFF {
                    let cand = i + lane;
                    if &text[cand..cand + m] == pattern {
                        return Some(cand);
                    }
                }
            }

            i += chunk_size;
        }

        while i + m <= n {
            if text[i] == pattern[0]
                && text[i + m - 1] == pattern[m - 1]
                && &text[i..i + m] == pattern
            {
                return Some(i);
            }
            i += 1;
        }

        None
    }
}

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use core::arch::x86_64::*;

    /// SSE2-accelerated naive string search.
    ///
    /// # Safety
    /// Requires x86_64 SSE2 support, which is baseline for x86_64.
    #[target_feature(enable = "sse2")]
    pub unsafe fn naive_find_sse2(text: &[u8], pattern: &[u8]) -> Option<usize> {
        let n = text.len();
        let m = pattern.len();

        if m == 0 {
            return Some(0);
        }
        if m > n {
            return None;
        }

        let first_vec = _mm_set1_epi8(pattern[0] as i8);
        let chunk_size = 16usize;

        let mut i = 0usize;
        while i + chunk_size <= n {
            let ptr = unsafe { text.as_ptr().add(i).cast::<__m128i>() };
            let chunk = unsafe { _mm_loadu_si128(ptr) };
            let cmp = _mm_cmpeq_epi8(chunk, first_vec);
            let mut mask = _mm_movemask_epi8(cmp) as u32;

            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                let cand = i + lane;
                if cand + m <= n && &text[cand..cand + m] == pattern {
                    return Some(cand);
                }
                mask &= mask - 1;
            }

            i += chunk_size;
        }

        while i + m <= n {
            if &text[i..i + m] == pattern {
                return Some(i);
            }
            i += 1;
        }

        None
    }

    /// SSE2-accelerated naive search with first/last-byte prefilter.
    ///
    /// # Safety
    /// Requires x86_64 SSE2 support, which is baseline for x86_64.
    #[target_feature(enable = "sse2")]
    pub unsafe fn naive_find_sse2_v2(text: &[u8], pattern: &[u8]) -> Option<usize> {
        let n = text.len();
        let m = pattern.len();

        if m == 0 {
            return Some(0);
        }
        if m > n {
            return None;
        }
        if m == 1 {
            return text.iter().position(|&b| b == pattern[0]);
        }

        let first_vec = _mm_set1_epi8(pattern[0] as i8);
        let last_vec = _mm_set1_epi8(pattern[m - 1] as i8);
        let chunk_size = 16usize;
        let last_off = m - 1;

        let mut i = 0usize;
        while i + chunk_size + last_off <= n {
            let ptr_first = unsafe { text.as_ptr().add(i).cast::<__m128i>() };
            let ptr_last = unsafe { text.as_ptr().add(i + last_off).cast::<__m128i>() };

            let chunk_first = unsafe { _mm_loadu_si128(ptr_first) };
            let chunk_last = unsafe { _mm_loadu_si128(ptr_last) };
            let cmp_first = _mm_cmpeq_epi8(chunk_first, first_vec);
            let cmp_last = _mm_cmpeq_epi8(chunk_last, last_vec);

            let mask_first = _mm_movemask_epi8(cmp_first) as u32;
            let mask_last = _mm_movemask_epi8(cmp_last) as u32;
            let mut mask = mask_first & mask_last;

            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                let cand = i + lane;
                if &text[cand..cand + m] == pattern {
                    return Some(cand);
                }
                mask &= mask - 1;
            }

            i += chunk_size;
        }

        while i + m <= n {
            if text[i] == pattern[0]
                && text[i + m - 1] == pattern[m - 1]
                && &text[i..i + m] == pattern
            {
                return Some(i);
            }
            i += 1;
        }

        None
    }
}
