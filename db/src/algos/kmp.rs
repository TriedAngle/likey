use crate::like::{LiteralAlgorithm, RowLiteralSearch};
use crate::storage::utf8::{Utf8Column, Utf8Row};

use super::utf8_shared::{
    ByteNeedle, byte_index_symbols, byte_literal_len, compile_byte_literal, matches_at_bytes,
    utf8_row_len,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct Utf8Kmp;

impl LiteralAlgorithm for Utf8Kmp {
    type Needle = ByteNeedle;
    type State = Box<[usize]>;

    const SUPPORTS_UNDERSCORE: bool = false;

    #[inline]
    fn compile_literal(src: &str) -> Option<Self::Needle> {
        compile_byte_literal(src)
    }

    #[inline]
    fn build_state(needle: &Self::Needle) -> Self::State {
        build_kmp_lps(needle.bytes()).into_boxed_slice()
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

impl<'db> RowLiteralSearch<Utf8Column<'db>> for Utf8Kmp {
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
        if pat.is_empty() {
            return Some(from as u32);
        }
        if pat.len() > text.len().saturating_sub(from) {
            return None;
        }

        let mut j = 0usize;
        for (i, &b) in text.iter().enumerate().skip(from) {
            while j > 0 && b != pat[j] {
                j = state[j - 1];
            }
            if b == pat[j] {
                j += 1;
                if j == pat.len() {
                    return Some((i + 1 - pat.len()) as u32);
                }
            }
        }
        None
    }
}

fn build_kmp_lps(pat: &[u8]) -> Vec<usize> {
    let mut lps = vec![0usize; pat.len()];
    let mut len = 0usize;

    for i in 1..pat.len() {
        while len > 0 && pat[i] != pat[len] {
            len = lps[len - 1];
        }
        if pat[i] == pat[len] {
            len += 1;
            lps[i] = len;
        }
    }

    lps
}
