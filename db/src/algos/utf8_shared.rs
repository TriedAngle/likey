//! Shared UTF-8/byte literal helpers used by several literal-search algorithms.
//!
//! The database's `Utf8Column` currently uses byte-wise LIKE semantics, so a
//! literal fragment is represented as exact bytes. Algorithms differ in how
//! they implement unanchored search (`find_from`), but anchored checks
//! (`matches_at`) can all share the same fixed-position equality primitive.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteNeedle {
    bytes: Box<[u8]>,
}

impl ByteNeedle {
    #[inline]
    pub fn new(bytes: Box<[u8]>) -> Self {
        Self { bytes }
    }

    #[inline]
    pub fn from_str(src: &str) -> Self {
        Self {
            bytes: src.as_bytes().into(),
        }
    }

    #[inline]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Byte literal that may contain `_` as an algorithm-level one-byte wildcard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteWildcardNeedle {
    bytes: Box<[u8]>,
    has_wildcard: bool,
}

impl ByteWildcardNeedle {
    #[inline]
    pub fn new(bytes: Box<[u8]>, has_wildcard: bool) -> Self {
        Self {
            bytes,
            has_wildcard,
        }
    }

    #[inline]
    pub fn from_str(src: &str) -> Self {
        let bytes: Box<[u8]> = src.as_bytes().into();
        let has_wildcard = bytes.iter().any(|&b| b == b'_');
        Self {
            bytes,
            has_wildcard,
        }
    }

    #[inline]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[inline]
    pub fn has_wildcard(&self) -> bool {
        self.has_wildcard
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ByteWildcardState {
    pub(crate) first_fixed: Option<(usize, u8)>,
    pub(crate) last_fixed: Option<(usize, u8)>,
}

impl ByteWildcardState {
    #[inline]
    pub fn build(pattern: &[u8]) -> Self {
        let first_fixed = pattern
            .iter()
            .enumerate()
            .find_map(|(idx, &b)| (b != b'_').then_some((idx, b)));
        let last_fixed = pattern
            .iter()
            .enumerate()
            .rev()
            .find_map(|(idx, &b)| (b != b'_').then_some((idx, b)));
        Self {
            first_fixed,
            last_fixed,
        }
    }

    #[inline]
    pub fn first_fixed(self) -> Option<(usize, u8)> {
        self.first_fixed
    }

    #[inline]
    pub fn last_fixed(self) -> Option<(usize, u8)> {
        self.last_fixed
    }
}

#[inline(always)]
pub fn compile_byte_wildcard_literal(src: &str) -> Option<ByteWildcardNeedle> {
    Some(ByteWildcardNeedle::from_str(src))
}

#[inline(always)]
pub fn byte_wildcard_literal_len(needle: &ByteWildcardNeedle) -> u32 {
    needle.bytes().len() as u32
}

#[inline(always)]
pub fn byte_wildcard_index_symbols(needle: &ByteWildcardNeedle) -> Option<Box<[u8]>> {
    if needle.has_wildcard() {
        None
    } else {
        Some(needle.bytes().into())
    }
}

/// Anchored byte-wildcard equality at `pos`; `_` in the needle matches one byte.
#[inline(always)]
pub fn matches_at_bytes_wildcard(
    row: &crate::storage::utf8::Utf8Row<'_>,
    pos: u32,
    needle: &ByteWildcardNeedle,
) -> bool {
    if !needle.has_wildcard() {
        return eq_at_bytes(row.bytes(), pos, needle.bytes());
    }

    let pos = pos as usize;
    let len = needle.bytes().len();
    let Some(end) = pos.checked_add(len) else {
        return false;
    };
    let Some(candidate) = row.bytes().get(pos..end) else {
        return false;
    };
    bytes_match_wildcard_same_len(candidate, needle.bytes())
}

#[inline(always)]
pub fn bytes_match_wildcard_same_len(text: &[u8], pattern: &[u8]) -> bool {
    debug_assert_eq!(text.len(), pattern.len());
    let mut i = 0usize;
    while i < pattern.len() {
        let p = unsafe { *pattern.get_unchecked(i) };
        if p != b'_' && unsafe { *text.get_unchecked(i) } != p {
            return false;
        }
        i += 1;
    }
    true
}

#[inline(always)]
pub fn compile_byte_literal(src: &str) -> Option<ByteNeedle> {
    Some(ByteNeedle::from_str(src))
}

#[inline(always)]
pub fn byte_literal_len(needle: &ByteNeedle) -> u32 {
    needle.bytes().len() as u32
}

#[inline(always)]
pub fn byte_index_symbols(needle: &ByteNeedle) -> Option<Box<[u8]>> {
    Some(needle.bytes().into())
}

#[inline(always)]
pub fn utf8_row_len(row: &crate::storage::utf8::Utf8Row<'_>) -> u32 {
    row.logical_len()
}

/// Anchored byte equality at `pos`.
///
/// This is the shared `matches_at` implementation for exact UTF-8-byte
/// literals. It deliberately does not use KMP/BM/Two-Way/etc.; those algorithms
/// are for unanchored search. At a known position, equality is the right kernel.
#[inline(always)]
pub fn matches_at_bytes(
    row: &crate::storage::utf8::Utf8Row<'_>,
    pos: u32,
    needle: &ByteNeedle,
) -> bool {
    eq_at_bytes(row.bytes(), pos, needle.bytes())
}

#[inline(always)]
pub fn eq_at_bytes(haystack: &[u8], pos: u32, needle: &[u8]) -> bool {
    let pos = pos as usize;
    let len = needle.len();
    let Some(end) = pos.checked_add(len) else {
        return false;
    };
    let Some(candidate) = haystack.get(pos..end) else {
        return false;
    };
    bytes_eq_same_len(candidate, needle)
}

#[inline(always)]
pub fn bytes_eq_same_len(a: &[u8], b: &[u8]) -> bool {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    if len < 8 {
        return bytes_eq_small(a, b);
    }
    if len < 16 {
        return bytes_eq_8_to_15(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if len >= 64
                && std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
            {
                // SAFETY: guarded by runtime feature detection.
                return unsafe { bytes_eq_avx512(a, b) };
            }
        }
        if len >= 32 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { bytes_eq_avx2(a, b) };
        }
        // SAFETY: SSE2 is baseline on x86_64.
        return unsafe { bytes_eq_sse2(a, b) };
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: guarded by cfg for aarch64+neon.
        return unsafe { bytes_eq_neon(a, b) };
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        bytes_eq_u64_chunks(a, b)
    }
}

#[inline(always)]
fn bytes_eq_small(a: &[u8], b: &[u8]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(a.len() < 8);

    let mut diff = 0u8;
    for i in 0..a.len() {
        // SAFETY: both slices have equal length and `i < a.len()`.
        diff |= unsafe { *a.get_unchecked(i) ^ *b.get_unchecked(i) };
    }
    diff == 0
}

#[inline(always)]
fn bytes_eq_8_to_15(a: &[u8], b: &[u8]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(a.len() >= 8 && a.len() < 16);

    let len = a.len();
    let last = len - 8;

    unsafe {
        read_u64_unaligned(a.as_ptr()) == read_u64_unaligned(b.as_ptr())
            && read_u64_unaligned(a.as_ptr().add(last)) == read_u64_unaligned(b.as_ptr().add(last))
    }
}

#[inline(always)]
pub fn bytes_eq_u64_chunks(a: &[u8], b: &[u8]) -> bool {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let mut i = 0usize;

    unsafe {
        while i + 8 <= len {
            let aa = read_u64_unaligned(a.as_ptr().add(i));
            let bb = read_u64_unaligned(b.as_ptr().add(i));
            if aa != bb {
                return false;
            }
            i += 8;
        }
    }

    bytes_eq_small(&a[i..], &b[i..])
}

#[inline(always)]
unsafe fn read_u64_unaligned(ptr: *const u8) -> u64 {
    // SAFETY: caller ensures the pointer is valid for reading 8 bytes. The read
    // is explicitly unaligned.
    unsafe { core::ptr::read_unaligned(ptr.cast::<u64>()) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn bytes_eq_sse2(a: &[u8], b: &[u8]) -> bool {
    use core::arch::x86_64::*;

    debug_assert_eq!(a.len(), b.len());
    debug_assert!(a.len() >= 16);

    let len = a.len();
    let mut i = 0usize;

    while i + 16 <= len {
        // SAFETY: loop condition ensures 16 bytes are in-bounds for both loads.
        let aa = unsafe { _mm_loadu_si128(a.as_ptr().add(i).cast::<__m128i>()) };
        let bb = unsafe { _mm_loadu_si128(b.as_ptr().add(i).cast::<__m128i>()) };
        let cmp = _mm_cmpeq_epi8(aa, bb);
        let mask = _mm_movemask_epi8(cmp);
        if mask != 0xFFFF {
            return false;
        }
        i += 16;
    }

    bytes_eq_u64_chunks(&a[i..], &b[i..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bytes_eq_avx2(a: &[u8], b: &[u8]) -> bool {
    use core::arch::x86_64::*;

    debug_assert_eq!(a.len(), b.len());
    debug_assert!(a.len() >= 32);

    let len = a.len();
    let mut i = 0usize;

    while i + 32 <= len {
        // SAFETY: loop condition ensures 32 bytes are in-bounds for both loads.
        let aa = unsafe { _mm256_loadu_si256(a.as_ptr().add(i).cast::<__m256i>()) };
        let bb = unsafe { _mm256_loadu_si256(b.as_ptr().add(i).cast::<__m256i>()) };
        let cmp = _mm256_cmpeq_epi8(aa, bb);
        let mask = _mm256_movemask_epi8(cmp);
        if mask != -1 {
            return false;
        }
        i += 32;
    }

    bytes_eq_u64_chunks(&a[i..], &b[i..])
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
unsafe fn bytes_eq_avx512(a: &[u8], b: &[u8]) -> bool {
    use core::arch::x86_64::*;

    debug_assert_eq!(a.len(), b.len());
    debug_assert!(a.len() >= 64);

    let len = a.len();
    let mut i = 0usize;

    while i + 64 <= len {
        // SAFETY: loop condition ensures 64 bytes are in-bounds for both loads.
        let aa = unsafe { _mm512_loadu_si512(a.as_ptr().add(i).cast()) };
        let bb = unsafe { _mm512_loadu_si512(b.as_ptr().add(i).cast()) };
        let mask = _mm512_cmpeq_epi8_mask(aa, bb) as u64;
        if mask != u64::MAX {
            return false;
        }
        i += 64;
    }

    bytes_eq_u64_chunks(&a[i..], &b[i..])
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn bytes_eq_neon(a: &[u8], b: &[u8]) -> bool {
    use core::arch::aarch64::*;

    debug_assert_eq!(a.len(), b.len());
    debug_assert!(a.len() >= 16);

    let len = a.len();
    let mut i = 0usize;

    while i + 16 <= len {
        // SAFETY: loop condition ensures 16 bytes are in-bounds for both loads.
        let aa = unsafe { vld1q_u8(a.as_ptr().add(i)) };
        let bb = unsafe { vld1q_u8(b.as_ptr().add(i)) };
        let cmp = unsafe { vceqq_u8(aa, bb) };
        let mut lanes = [0u8; 16];
        unsafe { vst1q_u8(lanes.as_mut_ptr(), cmp) };
        if lanes.iter().any(|&x| x != 0xFF) {
            return false;
        }
        i += 16;
    }

    bytes_eq_u64_chunks(&a[i..], &b[i..])
}

#[cfg(test)]
pub(crate) fn expected_find_from(text: &[u8], pattern: &[u8], from: usize) -> Option<usize> {
    if from > text.len() {
        return None;
    }
    if pattern.is_empty() {
        return Some(from);
    }
    if pattern.len() > text.len().saturating_sub(from) {
        return None;
    }
    text[from..]
        .windows(pattern.len())
        .position(|w| w == pattern)
        .map(|p| p + from)
}
