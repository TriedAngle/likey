//! Pair-Horspool literal search adapted from glibc's `memmem` strategy.
//!
//! This is a Rust reimplementation of the modified Horspool path described in
//! glibc `string/memmem.c`, with the single-byte path delegated to libc
//! `memchr` on libc targets.
//!
//! References:
//! - https://sourceware.org/git/?p=glibc.git;a=blob;f=string/memmem.c;hb=glibc-2.39
//! - https://sourceware.org/git/?p=glibc.git;a=blob;f=string/str-two-way.h;hb=glibc-2.39

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "android"))]
use core::ffi::{c_int, c_void};

use crate::like::{LiteralAlgorithm, RowLiteralSearch};
use crate::storage::utf8::{Utf8Column, Utf8Row};

use super::two_way2::{TwoWay2, TwoWay2State, two_way2_find};
use super::utf8_shared::{
    ByteNeedle, byte_index_symbols, byte_literal_len, compile_byte_literal, matches_at_bytes,
    utf8_row_len,
};

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "android"))]
unsafe extern "C" {
    fn memchr(s: *const c_void, c: c_int, n: usize) -> *mut c_void;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PairHorspool;

#[derive(Debug, Clone, Copy)]
pub struct PairHorspoolState {
    shift: [u8; 256],
    shift1: usize,
    long_state: Option<TwoWay2State>,
}

impl LiteralAlgorithm for PairHorspool {
    type Needle = ByteNeedle;
    type State = PairHorspoolState;

    const SUPPORTS_UNDERSCORE: bool = false;

    #[inline]
    fn compile_literal(src: &str) -> Option<Self::Needle> {
        compile_byte_literal(src)
    }

    #[inline]
    fn build_state(needle: &Self::Needle) -> Self::State {
        PairHorspoolState::build(needle)
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

impl PairHorspoolState {
    #[inline]
    fn build(needle: &ByteNeedle) -> Self {
        let pattern = needle.bytes();
        let mut shift = [0u8; 256];
        let mut shift1 = 1usize;

        if (3..=256).contains(&pattern.len()) {
            let m1 = pattern.len() - 1;
            for i in 1..m1 {
                shift[hash_pair(pattern[i - 1], pattern[i]) as usize] = i as u8;
            }

            let last_hash = hash_pair(pattern[m1 - 1], pattern[m1]) as usize;
            shift1 = m1 - shift[last_hash] as usize;
            shift[last_hash] = m1 as u8;
        }

        Self {
            shift,
            shift1,
            long_state: (pattern.len() > 256).then(|| TwoWay2::build_state(needle)),
        }
    }
}

impl<'db> RowLiteralSearch<Utf8Column<'db>> for PairHorspool {
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
        state: &Self::State,
    ) -> Option<u32> {
        let text = row.bytes();
        let pat = needle.bytes();
        let from = from as usize;

        if from > text.len() {
            return None;
        }
        pair_horspool_find(&text[from..], pat, state).map(|pos| (pos + from) as u32)
    }
}

#[inline(always)]
fn hash_pair(prev: u8, curr: u8) -> u8 {
    curr.wrapping_sub(prev.wrapping_shl(3))
}

#[inline]
pub fn pair_horspool_find(text: &[u8], pattern: &[u8], state: &PairHorspoolState) -> Option<usize> {
    let n = text.len();
    let m = pattern.len();

    if m == 0 {
        return Some(0);
    }
    if m > n {
        return None;
    }
    if m == 1 {
        return find_one(text, pattern[0]);
    }
    if m == 2 {
        return find_two(text, pattern);
    }
    if m > 256 {
        let long_state = state
            .long_state
            .as_ref()
            .expect("PairHorspool long-needle state should be built");
        return two_way2_find(text, pattern, long_state);
    }

    pair_horspool_find_medium(text, pattern, state)
}

#[inline]
fn find_one(text: &[u8], byte: u8) -> Option<usize> {
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "android"))]
    {
        return find_one_memchr(text, byte);
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "android")))]
    {
        text.iter().position(|&b| b == byte)
    }
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "android"))]
#[inline]
fn find_one_memchr(text: &[u8], byte: u8) -> Option<usize> {
    // SAFETY: `text.as_ptr()` is valid for `text.len()` bytes. `memchr` does
    // not write through the pointer and returns either null or a pointer into
    // the searched range.
    let found = unsafe { memchr(text.as_ptr().cast::<c_void>(), byte as c_int, text.len()) };
    if found.is_null() {
        None
    } else {
        Some(found as usize - text.as_ptr() as usize)
    }
}

#[inline]
fn find_two(text: &[u8], pattern: &[u8]) -> Option<usize> {
    debug_assert_eq!(pattern.len(), 2);

    let needle = ((pattern[0] as u32) << 16) | pattern[1] as u32;
    let end = text.len() - 2;
    let mut i = 0usize;
    let ptr = text.as_ptr();

    // SAFETY: caller has already checked `pattern.len() == 2` and
    // `pattern.len() <= text.len()`, so bytes 0 and 1 are valid. The loop only
    // reads `i + 1 <= end + 1 == text.len() - 1`.
    let mut window = unsafe { ((*ptr as u32) << 16) | *ptr.add(1) as u32 };

    while i < end && window != needle {
        i += 1;
        // SAFETY: see bounds argument above.
        window = unsafe { (window << 16) | *ptr.add(i + 1) as u32 };
    }

    (window == needle).then_some(i)
}

#[inline]
fn pair_horspool_find_medium(
    text: &[u8],
    pattern: &[u8],
    state: &PairHorspoolState,
) -> Option<usize> {
    debug_assert!((3..=256).contains(&pattern.len()));
    debug_assert!(pattern.len() <= text.len());

    let m = pattern.len();
    let m1 = m - 1;
    // SAFETY: preconditions above guarantee `3 <= m <= 256` and `m <= n`.
    // The offset loop mirrors glibc's pointer loop: a candidate start is valid
    // while `pos <= max_start`, and the pair hash reads at most `pos + m1`,
    // which is then bounded by `text.len() - 1`.
    unsafe { pair_horspool_find_medium_unchecked(text, pattern, state, m1) }
}

#[inline(always)]
unsafe fn pair_horspool_find_medium_unchecked(
    text: &[u8],
    pattern: &[u8],
    state: &PairHorspoolState,
    m1: usize,
) -> Option<usize> {
    let text_ptr = text.as_ptr();
    let pat_ptr = pattern.as_ptr();
    let shift_ptr = state.shift.as_ptr();
    let max_start = text.len() - (m1 + 1);
    let mut offset = 0usize;
    let mut pos = 0usize;

    while pos <= max_start {
        let mut end = pos + m1;
        let mut shift;

        loop {
            // SAFETY: `end <= max_start + m1 == text.len() - 1`, and m1 >= 2.
            let hash = unsafe { hash_pair(*text_ptr.add(end - 1), *text_ptr.add(end)) as usize };
            // SAFETY: hash_pair returns u8, so the table index is in 0..256.
            shift = unsafe { *shift_ptr.add(hash) as usize };
            if shift != 0 || end > max_start {
                break;
            }
            end += m1;
        }

        pos = end.wrapping_sub(shift);
        if shift < m1 {
            continue;
        }

        if m1 < 15
            // SAFETY: when m1 >= 15, glibc's offset update keeps offset within
            // `0..=m1 - 8`, so both 8-byte reads are in-bounds.
            || unsafe { read_u64_unaligned(text_ptr.add(pos + offset))
                == read_u64_unaligned(pat_ptr.add(offset)) }
        {
            // SAFETY: `pos <= max_start`, so `pos + m1 <= text.len() - 1`.
            if unsafe { bytes_eq_ptr(text_ptr.add(pos), pat_ptr, m1) } {
                return Some(pos);
            }

            offset = if offset >= 8 {
                offset - 8
            } else {
                m1.wrapping_sub(8)
            };
        }

        pos += state.shift1;
    }

    None
}

#[inline(always)]
unsafe fn bytes_eq_ptr(a: *const u8, b: *const u8, len: usize) -> bool {
    let mut i = 0usize;
    while i + 8 <= len {
        // SAFETY: caller guarantees both pointers are valid for `len` bytes and
        // loop condition keeps the unaligned u64 reads in-bounds.
        if unsafe { read_u64_unaligned(a.add(i)) != read_u64_unaligned(b.add(i)) } {
            return false;
        }
        i += 8;
    }
    while i < len {
        // SAFETY: caller guarantees both pointers are valid for `len` bytes.
        if unsafe { *a.add(i) != *b.add(i) } {
            return false;
        }
        i += 1;
    }
    true
}

#[inline(always)]
unsafe fn read_u64_unaligned(ptr: *const u8) -> u64 {
    // SAFETY: caller ensures the pointer is valid for reading 8 bytes.
    unsafe { core::ptr::read_unaligned(ptr.cast::<u64>()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find(text: &[u8], pattern: &[u8]) -> Option<usize> {
        let needle = ByteNeedle::new(pattern.into());
        let state = PairHorspoolState::build(&needle);
        pair_horspool_find(text, pattern, &state)
    }

    fn expected_find(text: &[u8], pattern: &[u8]) -> Option<usize> {
        if pattern.is_empty() {
            return Some(0);
        }
        if pattern.len() > text.len() {
            return None;
        }
        text.windows(pattern.len()).position(|w| w == pattern)
    }

    #[test]
    fn pair_horspool_handles_short_needles() {
        assert_eq!(find(b"abc", b""), Some(0));
        assert_eq!(find(b"abc", b"b"), Some(1));
        assert_eq!(find(b"abc", b"bc"), Some(1));
        assert_eq!(find(b"abc", b"bd"), None);
    }

    #[test]
    fn pair_horspool_find_one_matches_scalar() {
        for len in 0..300 {
            let mut text = vec![b'a'; len];
            assert_eq!(find_one(&text, b'z'), None);

            for pos in 0..len {
                text.fill(b'a');
                text[pos] = b'z';
                assert_eq!(find_one(&text, b'z'), Some(pos));
            }
        }
    }

    #[test]
    fn pair_horspool_handles_medium_edges() {
        let cases: &[(&[u8], &[u8])] = &[
            (b"ababcabcabababd", b"ababd"),
            (b"aaaaaaaaaaaaaaaaab", b"aaab"),
            (b"zzzzzzzzabc", b"abc"),
            (b"abczzzzzzzz", b"abc"),
            (b"ACGTACGTACGT", b"CGTA"),
            (b"abababababab", b"baba"),
            (b"the quick brown fox", b"brown"),
        ];

        for &(text, pattern) in cases {
            assert_eq!(find(text, pattern), expected_find(text, pattern));
        }
    }

    #[test]
    fn pair_horspool_falls_back_for_long_needles() {
        let mut text = vec![b'a'; 400];
        text.extend_from_slice(&vec![b'b'; 300]);
        let pattern = vec![b'b'; 300];

        assert_eq!(find(&text, &pattern), Some(400));
    }
}
