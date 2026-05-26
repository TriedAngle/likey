use crate::like::{LiteralAlgorithm, RowLiteralSearch};
use crate::storage::utf8::{Utf8Column, Utf8Row};

use super::utf8_shared::{
    ByteNeedle, ByteWildcardNeedle, ByteWildcardState, byte_index_symbols, byte_literal_len,
    byte_wildcard_index_symbols, byte_wildcard_literal_len, bytes_match_wildcard_same_len,
    compile_byte_literal, compile_byte_wildcard_literal, matches_at_bytes,
    matches_at_bytes_wildcard, utf8_row_len,
};

#[derive(Debug, Clone, Copy, Default)]
/// Default byte-wise naive literal search marker.
pub struct Naive;
#[derive(Debug, Clone, Copy, Default)]
/// Scalar byte-wise naive literal search marker.
pub struct NaiveScalar;
#[derive(Debug, Clone, Copy, Default)]
/// Platform vectorized byte-wise naive literal search marker.
pub struct NaiveVectorized;
#[derive(Debug, Clone, Copy, Default)]
/// Platform vectorized byte-wise search using first/last-byte prefilters.
pub struct NaiveVectorizedV2;
#[derive(Debug, Clone, Copy, Default)]
/// AVX2 byte-wise search marker with runtime fallback when AVX2 is unavailable.
pub struct NaiveAvx2;
#[derive(Debug, Clone, Copy, Default)]
/// AVX2 byte-wise search marker using first/last-byte prefilters.
pub struct NaiveAvx2V2;
#[derive(Debug, Clone, Copy, Default)]
/// AVX-512 byte-wise search marker with runtime fallback when unavailable.
pub struct NaiveAvx512;
#[derive(Debug, Clone, Copy, Default)]
/// AVX-512 byte-wise search marker using first/last-byte prefilters.
pub struct NaiveAvx512V2;
#[derive(Debug, Clone, Copy, Default)]
/// Runtime-selected byte-wise search marker.
pub struct NaiveAuto;
#[derive(Debug, Clone, Copy, Default)]
/// Hybrid search marker that uses scalar code for small cases and auto selection otherwise.
pub struct NaiveMixed;

#[derive(Debug, Clone, Copy, Default)]
/// Default byte-wise search marker that treats `_` inside literals as a wildcard byte.
pub struct NaiveWildcard;
#[derive(Debug, Clone, Copy, Default)]
/// Scalar wildcard-aware byte-wise search marker.
pub struct NaiveScalarWildcard;
#[derive(Debug, Clone, Copy, Default)]
/// Platform vectorized wildcard-aware byte-wise search marker.
pub struct NaiveVectorizedWildcard;
#[derive(Debug, Clone, Copy, Default)]
/// Vectorized wildcard-aware search marker using first/last fixed-byte anchors.
pub struct NaiveVectorizedV2Wildcard;
#[derive(Debug, Clone, Copy, Default)]
/// AVX2 wildcard-aware search marker with runtime fallback when AVX2 is unavailable.
pub struct NaiveAvx2Wildcard;
#[derive(Debug, Clone, Copy, Default)]
/// AVX2 wildcard-aware search marker using first/last fixed-byte anchors.
pub struct NaiveAvx2V2Wildcard;
#[derive(Debug, Clone, Copy, Default)]
/// AVX-512 wildcard-aware search marker with runtime fallback when unavailable.
pub struct NaiveAvx512Wildcard;
#[derive(Debug, Clone, Copy, Default)]
/// AVX-512 wildcard-aware search marker using first/last fixed-byte anchors.
pub struct NaiveAvx512V2Wildcard;
#[derive(Debug, Clone, Copy, Default)]
/// Runtime-selected wildcard-aware byte-wise search marker.
pub struct NaiveAutoWildcard;
#[derive(Debug, Clone, Copy, Default)]
/// Hybrid wildcard-aware marker that uses scalar code for small cases and auto selection otherwise.
pub struct NaiveMixedWildcard;

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
impl_naive_literal_algorithm!(NaiveAvx2);
impl_naive_literal_algorithm!(NaiveAvx2V2);
impl_naive_literal_algorithm!(NaiveAvx512);
impl_naive_literal_algorithm!(NaiveAvx512V2);
impl_naive_literal_algorithm!(NaiveAuto);
impl_naive_literal_algorithm!(NaiveMixed);

macro_rules! impl_naive_wildcard_literal_algorithm {
    ($ty:ty) => {
        impl LiteralAlgorithm for $ty {
            type Needle = ByteWildcardNeedle;
            type State = ByteWildcardState;

            const SUPPORTS_UNDERSCORE: bool = true;

            #[inline]
            fn compile_literal(src: &str) -> Option<Self::Needle> {
                compile_byte_wildcard_literal(src)
            }

            #[inline]
            fn build_state(needle: &Self::Needle) -> Self::State {
                ByteWildcardState::build(needle.bytes())
            }

            #[inline]
            fn literal_len(needle: &Self::Needle) -> u32 {
                byte_wildcard_literal_len(needle)
            }

            #[inline]
            fn index_symbols(needle: &Self::Needle) -> Option<Box<[u8]>> {
                byte_wildcard_index_symbols(needle)
            }
        }
    };
}

impl_naive_wildcard_literal_algorithm!(NaiveWildcard);
impl_naive_wildcard_literal_algorithm!(NaiveScalarWildcard);
impl_naive_wildcard_literal_algorithm!(NaiveVectorizedWildcard);
impl_naive_wildcard_literal_algorithm!(NaiveVectorizedV2Wildcard);
impl_naive_wildcard_literal_algorithm!(NaiveAvx2Wildcard);
impl_naive_wildcard_literal_algorithm!(NaiveAvx2V2Wildcard);
impl_naive_wildcard_literal_algorithm!(NaiveAvx512Wildcard);
impl_naive_wildcard_literal_algorithm!(NaiveAvx512V2Wildcard);
impl_naive_wildcard_literal_algorithm!(NaiveAutoWildcard);
impl_naive_wildcard_literal_algorithm!(NaiveMixedWildcard);

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
impl_naive_row_search!(NaiveAvx2, naive_find_avx2);
impl_naive_row_search!(NaiveAvx2V2, naive_find_avx2_v2);
impl_naive_row_search!(NaiveAvx512, naive_find_avx512);
impl_naive_row_search!(NaiveAvx512V2, naive_find_avx512_v2);
impl_naive_row_search!(NaiveAuto, naive_find_auto);
impl_naive_row_search!(NaiveMixed, naive_find_mixed);

macro_rules! impl_naive_wildcard_row_search {
    ($ty:ty, $exact_find:path, $wild_find:path) => {
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
                matches_at_bytes_wildcard(row, pos, needle)
            }

            #[inline]
            fn find_from<'r>(
                row: &Utf8Row<'r>,
                from: u32,
                needle: &Self::Needle,
                state: &Self::State,
            ) -> Option<u32> {
                let text = row.bytes();
                let pat = needle.bytes();
                let from = from as usize;
                if from > text.len() {
                    return None;
                }
                if needle.has_wildcard() {
                    $wild_find(&text[from..], pat, state).map(|pos| (pos + from) as u32)
                } else {
                    $exact_find(&text[from..], pat).map(|pos| (pos + from) as u32)
                }
            }
        }
    };
}

impl_naive_wildcard_row_search!(NaiveWildcard, naive_find, naive_find_wildcard);
impl_naive_wildcard_row_search!(
    NaiveScalarWildcard,
    naive_find_scalar,
    naive_find_wildcard_scalar
);
impl_naive_wildcard_row_search!(
    NaiveVectorizedWildcard,
    naive_find_vectorized,
    naive_find_wildcard_vectorized
);
impl_naive_wildcard_row_search!(
    NaiveVectorizedV2Wildcard,
    naive_find_vectorized_v2,
    naive_find_wildcard_vectorized_v2
);
impl_naive_wildcard_row_search!(NaiveAvx2Wildcard, naive_find_avx2, naive_find_wildcard_avx2);
impl_naive_wildcard_row_search!(
    NaiveAvx2V2Wildcard,
    naive_find_avx2_v2,
    naive_find_wildcard_avx2_v2
);
impl_naive_wildcard_row_search!(
    NaiveAvx512Wildcard,
    naive_find_avx512,
    naive_find_wildcard_avx512
);
impl_naive_wildcard_row_search!(
    NaiveAvx512V2Wildcard,
    naive_find_avx512_v2,
    naive_find_wildcard_avx512_v2
);
impl_naive_wildcard_row_search!(NaiveAutoWildcard, naive_find_auto, naive_find_wildcard_auto);
impl_naive_wildcard_row_search!(
    NaiveMixedWildcard,
    naive_find_mixed,
    naive_find_wildcard_mixed
);

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
        naive_find_scalar(text, pattern)
    } else {
        naive_find_auto(text, pattern)
    }
}

#[inline]
pub fn naive_find_auto(text: &[u8], pattern: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw")
            {
                return unsafe { x86::naive_find_avx512_v2(text, pattern) };
            }
        }
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { x86::naive_find_avx2_v2(text, pattern) };
        }
        return unsafe { x86::naive_find_sse2_v2(text, pattern) };
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { neon::naive_find_neon_v2(text, pattern) };
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        naive_find_scalar(text, pattern)
    }
}

#[inline]
pub fn naive_find_vectorized(text: &[u8], pattern: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        return unsafe { x86::naive_find_sse2(text, pattern) };
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { neon::naive_find_neon(text, pattern) };
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        naive_find_scalar(text, pattern)
    }
}

#[inline]
pub fn naive_find_vectorized_v2(text: &[u8], pattern: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        return unsafe { x86::naive_find_sse2_v2(text, pattern) };
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { neon::naive_find_neon_v2(text, pattern) };
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        naive_find_scalar(text, pattern)
    }
}

#[inline]
pub fn naive_find_avx2(text: &[u8], pattern: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { x86::naive_find_avx2(text, pattern) };
        }
        return unsafe { x86::naive_find_sse2(text, pattern) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        naive_find_vectorized(text, pattern)
    }
}

#[inline]
pub fn naive_find_avx2_v2(text: &[u8], pattern: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { x86::naive_find_avx2_v2(text, pattern) };
        }
        return unsafe { x86::naive_find_sse2_v2(text, pattern) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        naive_find_vectorized_v2(text, pattern)
    }
}

#[inline]
pub fn naive_find_avx512(text: &[u8], pattern: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw")
            {
                return unsafe { x86::naive_find_avx512(text, pattern) };
            }
        }
        naive_find_avx2(text, pattern)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        naive_find_vectorized(text, pattern)
    }
}

#[inline]
pub fn naive_find_avx512_v2(text: &[u8], pattern: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw")
            {
                return unsafe { x86::naive_find_avx512_v2(text, pattern) };
            }
        }
        naive_find_avx2_v2(text, pattern)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        naive_find_vectorized_v2(text, pattern)
    }
}

#[inline]
pub fn naive_find(text: &[u8], pattern: &[u8]) -> Option<usize> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { neon::naive_find_neon(text, pattern) };
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        naive_find_scalar(text, pattern)
    }
}

// Wildcard variants. These keep `_` in the literal and treat it as one byte.
#[inline]
pub fn naive_find_wildcard(
    text: &[u8],
    pattern: &[u8],
    state: &ByteWildcardState,
) -> Option<usize> {
    naive_find_wildcard_scalar(text, pattern, state)
}

#[inline]
pub fn naive_find_wildcard_scalar(
    text: &[u8],
    pattern: &[u8],
    state: &ByteWildcardState,
) -> Option<usize> {
    wildcard_find_scalar_impl(text, pattern, state, true)
}

#[inline]
pub fn naive_find_wildcard_mixed(
    text: &[u8],
    pattern: &[u8],
    state: &ByteWildcardState,
) -> Option<usize> {
    if pattern.len() <= 3 || text.len() < 64 {
        naive_find_wildcard_scalar(text, pattern, state)
    } else {
        naive_find_wildcard_auto(text, pattern, state)
    }
}

#[inline]
pub fn naive_find_wildcard_auto(
    text: &[u8],
    pattern: &[u8],
    state: &ByteWildcardState,
) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw")
            {
                return unsafe { x86_wildcard::find_avx512(text, pattern, state, true) };
            }
        }
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { x86_wildcard::find_avx2(text, pattern, state, true) };
        }
        return unsafe { x86_wildcard::find_sse2(text, pattern, state, true) };
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { neon_wildcard::find_neon(text, pattern, state, true) };
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        wildcard_find_scalar_impl(text, pattern, state, true)
    }
}

#[inline]
pub fn naive_find_wildcard_vectorized(
    text: &[u8],
    pattern: &[u8],
    state: &ByteWildcardState,
) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        return unsafe { x86_wildcard::find_sse2(text, pattern, state, false) };
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { neon_wildcard::find_neon(text, pattern, state, false) };
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        wildcard_find_scalar_impl(text, pattern, state, false)
    }
}

#[inline]
pub fn naive_find_wildcard_vectorized_v2(
    text: &[u8],
    pattern: &[u8],
    state: &ByteWildcardState,
) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        return unsafe { x86_wildcard::find_sse2(text, pattern, state, true) };
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { neon_wildcard::find_neon(text, pattern, state, true) };
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        wildcard_find_scalar_impl(text, pattern, state, true)
    }
}

#[inline]
pub fn naive_find_wildcard_avx2(
    text: &[u8],
    pattern: &[u8],
    state: &ByteWildcardState,
) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { x86_wildcard::find_avx2(text, pattern, state, false) };
        }
        return unsafe { x86_wildcard::find_sse2(text, pattern, state, false) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        naive_find_wildcard_vectorized(text, pattern, state)
    }
}

#[inline]
pub fn naive_find_wildcard_avx2_v2(
    text: &[u8],
    pattern: &[u8],
    state: &ByteWildcardState,
) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { x86_wildcard::find_avx2(text, pattern, state, true) };
        }
        return unsafe { x86_wildcard::find_sse2(text, pattern, state, true) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        naive_find_wildcard_vectorized_v2(text, pattern, state)
    }
}

#[inline]
pub fn naive_find_wildcard_avx512(
    text: &[u8],
    pattern: &[u8],
    state: &ByteWildcardState,
) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw")
            {
                return unsafe { x86_wildcard::find_avx512(text, pattern, state, false) };
            }
        }
        naive_find_wildcard_avx2(text, pattern, state)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        naive_find_wildcard_vectorized(text, pattern, state)
    }
}

#[inline]
pub fn naive_find_wildcard_avx512_v2(
    text: &[u8],
    pattern: &[u8],
    state: &ByteWildcardState,
) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw")
            {
                return unsafe { x86_wildcard::find_avx512(text, pattern, state, true) };
            }
        }
        naive_find_wildcard_avx2_v2(text, pattern, state)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        naive_find_wildcard_vectorized_v2(text, pattern, state)
    }
}

#[inline]
fn wildcard_find_scalar_impl(
    text: &[u8],
    pattern: &[u8],
    state: &ByteWildcardState,
    use_last: bool,
) -> Option<usize> {
    let n = text.len();
    let m = pattern.len();
    if m == 0 {
        return Some(0);
    }
    if m > n {
        return None;
    }
    if state.first_fixed().is_none() {
        return Some(0);
    }

    let last_start = n - m;
    let mut pos = 0usize;
    while pos <= last_start {
        if byte_wildcard_prefilter_at(text, pos, state, use_last)
            && bytes_match_wildcard_same_len(&text[pos..pos + m], pattern)
        {
            return Some(pos);
        }
        pos += 1;
    }
    None
}

#[inline(always)]
fn byte_wildcard_prefilter_at(
    text: &[u8],
    pos: usize,
    state: &ByteWildcardState,
    use_last: bool,
) -> bool {
    if let Some((off, want)) = state.first_fixed() {
        if unsafe { *text.get_unchecked(pos + off) } != want {
            return false;
        }
    }
    if use_last {
        if let Some((off, want)) = state.last_fixed() {
            if Some((off, want)) != state.first_fixed()
                && unsafe { *text.get_unchecked(pos + off) } != want
            {
                return false;
            }
        }
    }
    true
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod neon {
    use core::arch::aarch64::*;

    pub unsafe fn naive_find_neon(text: &[u8], pattern: &[u8]) -> Option<usize> {
        let n = text.len();
        let m = pattern.len();
        if m == 0 {
            return Some(0);
        }
        if m > n {
            return None;
        }
        let first_vec = unsafe { vdupq_n_u8(pattern[0]) };
        let chunk_size = 16usize;
        let mut i = 0usize;
        while i + chunk_size <= n {
            let chunk = unsafe { vld1q_u8(text.as_ptr().add(i)) };
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
            let chunk_first = unsafe { vld1q_u8(text.as_ptr().add(i)) };
            let chunk_last = unsafe { vld1q_u8(text.as_ptr().add(i + last_off)) };
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
        let mut i = 0usize;
        while i + 16 <= n {
            let chunk = unsafe { _mm_loadu_si128(text.as_ptr().add(i).cast::<__m128i>()) };
            let mut mask = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, first_vec)) as u32;
            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                let cand = i + lane;
                if cand + m <= n && &text[cand..cand + m] == pattern {
                    return Some(cand);
                }
                mask &= mask - 1;
            }
            i += 16;
        }
        while i + m <= n {
            if &text[i..i + m] == pattern {
                return Some(i);
            }
            i += 1;
        }
        None
    }

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
        let last_off = m - 1;
        let mut i = 0usize;
        while i + 16 + last_off <= n {
            let chunk_first = unsafe { _mm_loadu_si128(text.as_ptr().add(i).cast::<__m128i>()) };
            let chunk_last =
                unsafe { _mm_loadu_si128(text.as_ptr().add(i + last_off).cast::<__m128i>()) };
            let mut mask = (_mm_movemask_epi8(_mm_cmpeq_epi8(chunk_first, first_vec))
                & _mm_movemask_epi8(_mm_cmpeq_epi8(chunk_last, last_vec)))
                as u32;
            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                let cand = i + lane;
                if &text[cand..cand + m] == pattern {
                    return Some(cand);
                }
                mask &= mask - 1;
            }
            i += 16;
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

    #[target_feature(enable = "avx2")]
    pub unsafe fn naive_find_avx2(text: &[u8], pattern: &[u8]) -> Option<usize> {
        let n = text.len();
        let m = pattern.len();
        if m == 0 {
            return Some(0);
        }
        if m > n {
            return None;
        }
        let first_vec = _mm256_set1_epi8(pattern[0] as i8);
        let mut i = 0usize;
        while i + 32 <= n {
            let chunk = unsafe { _mm256_loadu_si256(text.as_ptr().add(i).cast::<__m256i>()) };
            let mut mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, first_vec)) as u32;
            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                let cand = i + lane;
                if cand + m <= n && &text[cand..cand + m] == pattern {
                    return Some(cand);
                }
                mask &= mask - 1;
            }
            i += 32;
        }
        while i + m <= n {
            if &text[i..i + m] == pattern {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn naive_find_avx2_v2(text: &[u8], pattern: &[u8]) -> Option<usize> {
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
        let first_vec = _mm256_set1_epi8(pattern[0] as i8);
        let last_vec = _mm256_set1_epi8(pattern[m - 1] as i8);
        let last_off = m - 1;
        let mut i = 0usize;
        while i + 32 + last_off <= n {
            let chunk_first = unsafe { _mm256_loadu_si256(text.as_ptr().add(i).cast::<__m256i>()) };
            let chunk_last =
                unsafe { _mm256_loadu_si256(text.as_ptr().add(i + last_off).cast::<__m256i>()) };
            let mut mask = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk_first, first_vec))
                & _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk_last, last_vec)))
                as u32;
            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                let cand = i + lane;
                if &text[cand..cand + m] == pattern {
                    return Some(cand);
                }
                mask &= mask - 1;
            }
            i += 32;
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

    #[cfg(feature = "avx512")]
    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn naive_find_avx512(text: &[u8], pattern: &[u8]) -> Option<usize> {
        let n = text.len();
        let m = pattern.len();
        if m == 0 {
            return Some(0);
        }
        if m > n {
            return None;
        }
        let first_vec = _mm512_set1_epi8(pattern[0] as i8);
        let mut i = 0usize;
        while i + 64 <= n {
            let chunk = unsafe { _mm512_loadu_si512(text.as_ptr().add(i).cast()) };
            let mut mask = _mm512_cmpeq_epi8_mask(chunk, first_vec) as u64;
            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                let cand = i + lane;
                if cand + m <= n && &text[cand..cand + m] == pattern {
                    return Some(cand);
                }
                mask &= mask - 1;
            }
            i += 64;
        }
        while i + m <= n {
            if &text[i..i + m] == pattern {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    #[cfg(feature = "avx512")]
    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn naive_find_avx512_v2(text: &[u8], pattern: &[u8]) -> Option<usize> {
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
        let first_vec = _mm512_set1_epi8(pattern[0] as i8);
        let last_vec = _mm512_set1_epi8(pattern[m - 1] as i8);
        let last_off = m - 1;
        let mut i = 0usize;
        while i + 64 + last_off <= n {
            let chunk_first = unsafe { _mm512_loadu_si512(text.as_ptr().add(i).cast()) };
            let chunk_last = unsafe { _mm512_loadu_si512(text.as_ptr().add(i + last_off).cast()) };
            let mut mask = (_mm512_cmpeq_epi8_mask(chunk_first, first_vec) as u64)
                & (_mm512_cmpeq_epi8_mask(chunk_last, last_vec) as u64);
            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                let cand = i + lane;
                if &text[cand..cand + m] == pattern {
                    return Some(cand);
                }
                mask &= mask - 1;
            }
            i += 64;
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

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon_wildcard {
    use super::*;
    use core::arch::aarch64::*;

    pub unsafe fn find_neon(
        text: &[u8],
        pattern: &[u8],
        state: &ByteWildcardState,
        use_last: bool,
    ) -> Option<usize> {
        let n = text.len();
        let m = pattern.len();
        if m == 0 {
            return Some(0);
        }
        if m > n {
            return None;
        }
        let Some((first_off, first_byte)) = state.first_fixed() else {
            return Some(0);
        };
        let first_vec = unsafe { vdupq_n_u8(first_byte) };
        let last_filter = if use_last {
            match state.last_fixed() {
                Some(last) if Some(last) != state.first_fixed() => {
                    Some((last.0, unsafe { vdupq_n_u8(last.1) }))
                }
                _ => None,
            }
        } else {
            None
        };
        let candidate_count = n - m + 1;
        let mut pos = 0usize;
        while pos + 16 <= candidate_count {
            let first_chunk = unsafe { vld1q_u8(text.as_ptr().add(pos + first_off)) };
            let mut cmp = unsafe { vceqq_u8(first_chunk, first_vec) };
            if let Some((last_off, last_vec)) = last_filter {
                let last_chunk = unsafe { vld1q_u8(text.as_ptr().add(pos + last_off)) };
                cmp = unsafe { vandq_u8(cmp, vceqq_u8(last_chunk, last_vec)) };
            }
            let mut lanes = [0u8; 16];
            unsafe { vst1q_u8(lanes.as_mut_ptr(), cmp) };
            for (lane, &value) in lanes.iter().enumerate() {
                if value == 0xFF {
                    let cand = pos + lane;
                    if bytes_match_wildcard_same_len(&text[cand..cand + m], pattern) {
                        return Some(cand);
                    }
                }
            }
            pos += 16;
        }
        scalar_tail(text, pattern, state, use_last, pos, candidate_count)
    }

    fn scalar_tail(
        text: &[u8],
        pattern: &[u8],
        state: &ByteWildcardState,
        use_last: bool,
        mut pos: usize,
        candidate_count: usize,
    ) -> Option<usize> {
        if candidate_count == 0 {
            return None;
        }
        let last_start = candidate_count - 1;
        while pos <= last_start {
            if byte_wildcard_prefilter_at(text, pos, state, use_last)
                && bytes_match_wildcard_same_len(&text[pos..pos + pattern.len()], pattern)
            {
                return Some(pos);
            }
            pos += 1;
        }
        None
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_wildcard {
    use super::*;
    use core::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    pub unsafe fn find_sse2(
        text: &[u8],
        pattern: &[u8],
        state: &ByteWildcardState,
        use_last: bool,
    ) -> Option<usize> {
        let n = text.len();
        let m = pattern.len();
        if m == 0 {
            return Some(0);
        }
        if m > n {
            return None;
        }
        let Some((first_off, first_byte)) = state.first_fixed() else {
            return Some(0);
        };
        let first_vec = _mm_set1_epi8(first_byte as i8);
        let last_filter = if use_last {
            match state.last_fixed() {
                Some(last) if Some(last) != state.first_fixed() => {
                    Some((last.0, _mm_set1_epi8(last.1 as i8)))
                }
                _ => None,
            }
        } else {
            None
        };
        let candidate_count = n - m + 1;
        let mut pos = 0usize;
        while pos + 16 <= candidate_count {
            let first_chunk =
                unsafe { _mm_loadu_si128(text.as_ptr().add(pos + first_off).cast::<__m128i>()) };
            let mut mask = _mm_movemask_epi8(_mm_cmpeq_epi8(first_chunk, first_vec)) as u32;
            if let Some((last_off, last_vec)) = last_filter {
                let last_chunk =
                    unsafe { _mm_loadu_si128(text.as_ptr().add(pos + last_off).cast::<__m128i>()) };
                mask &= _mm_movemask_epi8(_mm_cmpeq_epi8(last_chunk, last_vec)) as u32;
            }
            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                let cand = pos + lane;
                if bytes_match_wildcard_same_len(&text[cand..cand + m], pattern) {
                    return Some(cand);
                }
                mask &= mask - 1;
            }
            pos += 16;
        }
        scalar_tail(text, pattern, state, use_last, pos, candidate_count)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn find_avx2(
        text: &[u8],
        pattern: &[u8],
        state: &ByteWildcardState,
        use_last: bool,
    ) -> Option<usize> {
        let n = text.len();
        let m = pattern.len();
        if m == 0 {
            return Some(0);
        }
        if m > n {
            return None;
        }
        let Some((first_off, first_byte)) = state.first_fixed() else {
            return Some(0);
        };
        let first_vec = _mm256_set1_epi8(first_byte as i8);
        let last_filter = if use_last {
            match state.last_fixed() {
                Some(last) if Some(last) != state.first_fixed() => {
                    Some((last.0, _mm256_set1_epi8(last.1 as i8)))
                }
                _ => None,
            }
        } else {
            None
        };
        let candidate_count = n - m + 1;
        let mut pos = 0usize;
        while pos + 32 <= candidate_count {
            let first_chunk =
                unsafe { _mm256_loadu_si256(text.as_ptr().add(pos + first_off).cast::<__m256i>()) };
            let mut mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(first_chunk, first_vec)) as u32;
            if let Some((last_off, last_vec)) = last_filter {
                let last_chunk = unsafe {
                    _mm256_loadu_si256(text.as_ptr().add(pos + last_off).cast::<__m256i>())
                };
                mask &= _mm256_movemask_epi8(_mm256_cmpeq_epi8(last_chunk, last_vec)) as u32;
            }
            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                let cand = pos + lane;
                if bytes_match_wildcard_same_len(&text[cand..cand + m], pattern) {
                    return Some(cand);
                }
                mask &= mask - 1;
            }
            pos += 32;
        }
        scalar_tail(text, pattern, state, use_last, pos, candidate_count)
    }

    #[cfg(feature = "avx512")]
    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn find_avx512(
        text: &[u8],
        pattern: &[u8],
        state: &ByteWildcardState,
        use_last: bool,
    ) -> Option<usize> {
        let n = text.len();
        let m = pattern.len();
        if m == 0 {
            return Some(0);
        }
        if m > n {
            return None;
        }
        let Some((first_off, first_byte)) = state.first_fixed() else {
            return Some(0);
        };
        let first_vec = _mm512_set1_epi8(first_byte as i8);
        let last_filter = if use_last {
            match state.last_fixed() {
                Some(last) if Some(last) != state.first_fixed() => {
                    Some((last.0, _mm512_set1_epi8(last.1 as i8)))
                }
                _ => None,
            }
        } else {
            None
        };
        let candidate_count = n - m + 1;
        let mut pos = 0usize;
        while pos + 64 <= candidate_count {
            let first_chunk =
                unsafe { _mm512_loadu_si512(text.as_ptr().add(pos + first_off).cast()) };
            let mut mask = _mm512_cmpeq_epi8_mask(first_chunk, first_vec) as u64;
            if let Some((last_off, last_vec)) = last_filter {
                let last_chunk =
                    unsafe { _mm512_loadu_si512(text.as_ptr().add(pos + last_off).cast()) };
                mask &= _mm512_cmpeq_epi8_mask(last_chunk, last_vec) as u64;
            }
            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                let cand = pos + lane;
                if bytes_match_wildcard_same_len(&text[cand..cand + m], pattern) {
                    return Some(cand);
                }
                mask &= mask - 1;
            }
            pos += 64;
        }
        scalar_tail(text, pattern, state, use_last, pos, candidate_count)
    }

    #[inline]
    fn scalar_tail(
        text: &[u8],
        pattern: &[u8],
        state: &ByteWildcardState,
        use_last: bool,
        mut pos: usize,
        candidate_count: usize,
    ) -> Option<usize> {
        if candidate_count == 0 {
            return None;
        }
        let last_start = candidate_count - 1;
        while pos <= last_start {
            if byte_wildcard_prefilter_at(text, pos, state, use_last)
                && bytes_match_wildcard_same_len(&text[pos..pos + pattern.len()], pattern)
            {
                return Some(pos);
            }
            pos += 1;
        }
        None
    }
}
