//! FFTSTR variant with a TwoWay2-style fixed-byte prefilter.
//!
//! This intentionally reuses the long-pattern `FftStr1` backend and only adds a
//! cheap necessary-condition scan before invoking the FFT verifier.

use crate::like::{LiteralAlgorithm, RowLiteralSearch};
use crate::storage::utf8::{Utf8Column, Utf8Row};

use super::fftstr::{FftNeedle, FftState1, FftStr1};
use super::utf8_shared::utf8_row_len;

const FFT_WILDCARD: u8 = b'_';

#[derive(Debug, Clone, Copy, Default)]
pub struct FftstrV2;

#[derive(Debug)]
pub struct FftstrV2State {
    inner: FftState1,
    prefilter: FftPrefilter,
}

impl LiteralAlgorithm for FftstrV2 {
    type Needle = FftNeedle;
    type State = FftstrV2State;

    const SUPPORTS_UNDERSCORE: bool = true;

    #[inline]
    fn compile_literal(src: &str) -> Option<Self::Needle> {
        FftStr1::compile_literal(src)
    }

    #[inline]
    fn build_state(needle: &Self::Needle) -> Self::State {
        Self::State {
            inner: FftStr1::build_state(needle),
            prefilter: FftPrefilter::build(needle.bytes()),
        }
    }

    #[inline]
    fn literal_len(needle: &Self::Needle) -> u32 {
        FftStr1::literal_len(needle)
    }

    #[inline]
    fn index_symbols(needle: &Self::Needle) -> Option<Box<[u8]>> {
        FftStr1::index_symbols(needle)
    }
}

impl<'db> RowLiteralSearch<Utf8Column<'db>> for FftstrV2 {
    #[inline]
    fn row_len<'r>(row: &Utf8Row<'r>) -> u32 {
        utf8_row_len(row)
    }

    #[inline]
    fn matches_at<'r>(
        row: &Utf8Row<'r>,
        pos: u32,
        needle: &Self::Needle,
        state: &Self::State,
    ) -> bool {
        FftStr1::matches_at(row, pos, needle, &state.inner)
    }

    #[inline]
    fn find_from<'r>(
        row: &Utf8Row<'r>,
        from: u32,
        needle: &Self::Needle,
        state: &Self::State,
    ) -> Option<u32> {
        let text = row.bytes();
        let from_usize = from as usize;
        if from_usize > text.len() {
            return None;
        }
        if needle.bytes().is_empty() {
            return Some(from);
        }
        if needle.bytes().len() > text.len().saturating_sub(from_usize) {
            return None;
        }

        let suffix = &text[from_usize..];
        let skip = state.prefilter.first_candidate(suffix, needle.bytes())?;
        let adjusted = from.checked_add(skip as u32)?;
        FftStr1::find_from(row, adjusted, needle, &state.inner)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FftPrefilter {
    active: bool,
    byteset: u64,
    pair_index1: usize,
    pair_index2: usize,
    pair_byte1: u8,
    pair_byte2: u8,
    byteset_skip_offset: usize,
    byteset_skip_len: usize,
}

impl FftPrefilter {
    fn build(pattern: &[u8]) -> Self {
        let Some((pair_index1, pair_index2)) = pick_fixed_pair(pattern) else {
            return Self {
                active: false,
                byteset: 0,
                pair_index1: 0,
                pair_index2: 0,
                pair_byte1: 0,
                pair_byte2: 0,
                byteset_skip_offset: 0,
                byteset_skip_len: 0,
            };
        };

        let (fixed_run_end, fixed_run_len) = longest_fixed_run(pattern);
        let fixed_run_start = fixed_run_end + 1 - fixed_run_len;
        let byteset = pattern[fixed_run_start..=fixed_run_end]
            .iter()
            .fold(0u64, |acc, &b| acc | (1u64 << ((b & 0x3f) as usize)));

        Self {
            active: true,
            byteset,
            pair_index1,
            pair_index2,
            pair_byte1: pattern[pair_index1],
            pair_byte2: pattern[pair_index2],
            byteset_skip_offset: fixed_run_end,
            byteset_skip_len: fixed_run_len,
        }
    }

    fn first_candidate(self, text: &[u8], pattern: &[u8]) -> Option<usize> {
        if !self.active {
            return Some(0);
        }

        let m = pattern.len();
        if m == 0 {
            return Some(0);
        }
        if text.len() < m {
            return None;
        }

        let last_start = text.len() - m;
        let mut pos = 0usize;

        while pos <= last_start {
            if self.byteset_skip_len != 0
                && !byteset_contains(self, text[pos + self.byteset_skip_offset])
            {
                pos += self.byteset_skip_len;
                continue;
            }

            let chunk = (last_start - pos + 1).min(16);
            match prefilter_delta(text, self, pos, chunk) {
                Some(0) => return Some(pos),
                Some(delta) => pos += delta,
                None => pos += chunk,
            }
        }

        None
    }
}

fn longest_fixed_run(pattern: &[u8]) -> (usize, usize) {
    let mut best_start = 0usize;
    let mut best_len = 0usize;
    let mut cur_start = 0usize;
    let mut cur_len = 0usize;

    for (idx, &byte) in pattern.iter().enumerate() {
        if byte == FFT_WILDCARD {
            if cur_len >= best_len {
                best_start = cur_start;
                best_len = cur_len;
            }
            cur_start = idx + 1;
            cur_len = 0;
        } else {
            cur_len += 1;
        }
    }

    if cur_len >= best_len {
        best_start = cur_start;
        best_len = cur_len;
    }

    if best_len == 0 {
        (0, 0)
    } else {
        (best_start + best_len - 1, best_len)
    }
}

fn pick_fixed_pair(pattern: &[u8]) -> Option<(usize, usize)> {
    let (first_idx, _) = pattern
        .iter()
        .enumerate()
        .find(|&(_, &b)| b != FFT_WILDCARD)?;
    let (last_idx, _) = pattern
        .iter()
        .enumerate()
        .rev()
        .find(|&(_, &b)| b != FFT_WILDCARD)?;
    Some((first_idx, last_idx))
}

#[inline]
fn byteset_contains(state: FftPrefilter, byte: u8) -> bool {
    ((state.byteset >> ((byte & 0x3f) as usize)) & 1) != 0
}

#[inline(always)]
fn prefilter_delta(text: &[u8], state: FftPrefilter, pos: usize, chunk: usize) -> Option<usize> {
    debug_assert!(chunk > 0);
    debug_assert!(chunk <= 16);

    #[cfg(target_arch = "x86_64")]
    if chunk == 16 {
        // SAFETY: caller ensures at least 16 candidate starts are in-bounds.
        unsafe {
            return x86::prefilter_delta_sse2(
                text,
                pos,
                state.pair_index1,
                state.pair_index2,
                state.pair_byte1,
                state.pair_byte2,
            );
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    if chunk == 16 {
        // SAFETY: caller ensures at least 16 candidate starts are in-bounds.
        unsafe {
            return neon::prefilter_delta_neon(
                text,
                pos,
                state.pair_index1,
                state.pair_index2,
                state.pair_byte1,
                state.pair_byte2,
            );
        }
    }

    prefilter_delta_scalar(text, state, pos, chunk)
}

#[inline(always)]
fn prefilter_delta_scalar(
    text: &[u8],
    state: FftPrefilter,
    pos: usize,
    chunk: usize,
) -> Option<usize> {
    for lane in 0..chunk {
        let cand = pos + lane;
        if text[cand + state.pair_index1] == state.pair_byte1
            && text[cand + state.pair_index2] == state.pair_byte2
        {
            return Some(lane);
        }
    }
    None
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    pub unsafe fn prefilter_delta_sse2(
        text: &[u8],
        pos: usize,
        pair_index1: usize,
        pair_index2: usize,
        pair_byte1: u8,
        pair_byte2: u8,
    ) -> Option<usize> {
        let v1 = _mm_set1_epi8(pair_byte1 as i8);
        let v2 = _mm_set1_epi8(pair_byte2 as i8);
        let chunk1 =
            unsafe { _mm_loadu_si128(text.as_ptr().add(pos + pair_index1).cast::<__m128i>()) };
        let chunk2 =
            unsafe { _mm_loadu_si128(text.as_ptr().add(pos + pair_index2).cast::<__m128i>()) };
        let eq = _mm_and_si128(_mm_cmpeq_epi8(chunk1, v1), _mm_cmpeq_epi8(chunk2, v2));
        let mask = _mm_movemask_epi8(eq) as u32;

        if mask == 0 {
            None
        } else {
            Some(mask.trailing_zeros() as usize)
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    use core::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub unsafe fn prefilter_delta_neon(
        text: &[u8],
        pos: usize,
        pair_index1: usize,
        pair_index2: usize,
        pair_byte1: u8,
        pair_byte2: u8,
    ) -> Option<usize> {
        let v1 = vdupq_n_u8(pair_byte1);
        let v2 = vdupq_n_u8(pair_byte2);
        let chunk1 = unsafe { vld1q_u8(text.as_ptr().add(pos + pair_index1)) };
        let chunk2 = unsafe { vld1q_u8(text.as_ptr().add(pos + pair_index2)) };
        let eq1 = vceqq_u8(chunk1, v1);
        let eq2 = vceqq_u8(chunk2, v2);
        let eq = vandq_u8(eq1, eq2);

        if vmaxvq_u8(eq) == 0 {
            return None;
        }

        let mut lanes = [0u8; 16];
        unsafe { vst1q_u8(lanes.as_mut_ptr(), eq) };
        for (lane, &value) in lanes.iter().enumerate() {
            if value == 0xFF {
                return Some(lane);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::DbBuilder;
    use crate::like::{LiteralAlgorithm, RowLiteralSearch};
    use crate::storage::utf8::Utf8TableBuilder;

    fn one_row(text: &str) -> (crate::Db, crate::TableId) {
        let mut t = Utf8TableBuilder::new("t");
        t.push_str(text);
        let mut dbb = DbBuilder::new();
        let id = dbb.add_utf8_table(t).unwrap();
        (dbb.freeze(), id)
    }

    fn assert_same_as_fftstr1(text: &str, pat: &str) {
        let (db, id) = one_row(text);
        let table = db.utf8_table(id).unwrap();
        let row = table.text().row_view(0);
        let needle1 = FftStr1::compile_literal(pat).unwrap();
        let state1 = FftStr1::build_state(&needle1);
        let needle2 = FftstrV2::compile_literal(pat).unwrap();
        let state2 = FftstrV2::build_state(&needle2);

        for from in 0..=text.len() as u32 + 1 {
            let got = FftstrV2::find_from(&row, from, &needle2, &state2);
            let expect = FftStr1::find_from(&row, from, &needle1, &state1);
            assert_eq!(got, expect, "text={text:?}, pat={pat:?}, from={from}");
        }
    }

    #[test]
    fn finds_same_as_fftstr1() {
        assert_same_as_fftstr1("abababababababab", "a_b");
        assert_same_as_fftstr1("abababababababab", "a__a");
        assert_same_as_fftstr1("zzabczz", "a_c");
        assert_same_as_fftstr1("zzabczz", "_bc");
        assert_same_as_fftstr1("zzabczz", "ab_");
        assert_same_as_fftstr1("zzxabczz", "_abc_");
        assert_same_as_fftstr1("zzzzazzzz", "__a__");
        assert_same_as_fftstr1("zzzzzz", "____");
        assert_same_as_fftstr1("hello world", "world");
    }

    #[test]
    fn byteset_skip_uses_longest_fixed_run() {
        let pattern = b"_abc_";
        let prefilter = FftPrefilter::build(pattern);

        assert_eq!(prefilter.pair_index1, 1);
        assert_eq!(prefilter.pair_index2, 3);
        assert_eq!(prefilter.byteset_skip_offset, 3);
        assert_eq!(prefilter.byteset_skip_len, 3);
        assert_eq!(prefilter.first_candidate(b"yyyxabcx", pattern), Some(3));
    }
}
