use std::cmp::max;

use crate::like::{LiteralAlgorithm, RowLiteralSearch};
use crate::storage::utf8::{Utf8Column, Utf8Row};

use super::utf8_shared::{
    ByteNeedle, byte_index_symbols, byte_literal_len, compile_byte_literal, matches_at_bytes,
    utf8_row_len,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct BM;

#[derive(Debug, Clone)]
pub struct BMState {
    bad_char: [isize; 256],
    good_suffix: Vec<usize>,
}

impl LiteralAlgorithm for BM {
    type Needle = ByteNeedle;
    type State = BMState;

    const SUPPORTS_UNDERSCORE: bool = false;

    #[inline]
    fn compile_literal(src: &str) -> Option<Self::Needle> {
        compile_byte_literal(src)
    }

    #[inline]
    fn build_state(needle: &Self::Needle) -> Self::State {
        let pattern = needle.bytes();
        BMState {
            bad_char: build_bad_char_table(pattern),
            good_suffix: build_good_suffix_table(pattern),
        }
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

impl<'db> RowLiteralSearch<Utf8Column<'db>> for BM {
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
        bm_find(&text[from..], pat, state).map(|pos| (pos + from) as u32)
    }
}

fn build_bad_char_table(pattern: &[u8]) -> [isize; 256] {
    let mut table = [-1isize; 256];
    for (i, &b) in pattern.iter().enumerate() {
        table[b as usize] = i as isize;
    }
    table
}

fn build_good_suffix_table(pattern: &[u8]) -> Vec<usize> {
    let m = pattern.len();
    let mut shift = vec![0usize; m + 1];
    let mut border_pos = vec![0usize; m + 1];

    let mut i = m;
    let mut j = m + 1;
    border_pos[i] = j;

    while i > 0 {
        while j <= m && pattern[i - 1] != pattern[j - 1] {
            if shift[j] == 0 {
                shift[j] = j - i;
            }
            j = border_pos[j];
        }
        i -= 1;
        j -= 1;
        border_pos[i] = j;
    }

    j = border_pos[0];
    for i in 0..=m {
        if shift[i] == 0 {
            shift[i] = j;
        }
        if i == j {
            j = border_pos[j];
        }
    }

    shift
}

pub fn bm_find(text: &[u8], pattern: &[u8], state: &BMState) -> Option<usize> {
    let n = text.len();
    let m = pattern.len();

    if m == 0 {
        return Some(0);
    }
    if m > n {
        return None;
    }

    let mut i = 0usize;

    while i <= n - m {
        let mut j = (m - 1) as isize;

        while j >= 0 && pattern[j as usize] == text[i + j as usize] {
            j -= 1;
        }

        if j < 0 {
            return Some(i);
        }

        let mismatch_index = j as usize;
        let bad_byte = text[i + mismatch_index];

        let last_occurrence = state.bad_char[bad_byte as usize];
        let bc_shift = mismatch_index as isize - last_occurrence;
        let bc_shift = if bc_shift > 0 { bc_shift as usize } else { 1 };

        let gs_shift = state.good_suffix[mismatch_index + 1];

        i += max(bc_shift, gs_shift);
    }

    None
}
