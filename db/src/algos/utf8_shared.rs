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
        if len >= 32 && std::is_x86_feature_detected!("avx2") {
            return unsafe { bytes_eq_avx2(a, b) };
        }
        return unsafe { bytes_eq_sse2(a, b) };
    }

    #[cfg(not(target_arch = "x86_64"))]
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
