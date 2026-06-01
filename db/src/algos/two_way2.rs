use core::cmp::max;

use crate::like::{LiteralAlgorithm, RowLiteralSearch};
use crate::storage::utf8::{Utf8Column, Utf8Row};

use super::utf8_shared::{
    byte_index_symbols, byte_literal_len, compile_byte_literal, matches_at_bytes, utf8_row_len,
    ByteNeedle,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct TwoWay2;

#[derive(Clone, Copy, Debug)]
pub struct TwoWay2State {
    // crit = ell + 1, so crit == 0 corresponds to ell == -1.
    crit: usize,
    period: usize,
    is_periodic: bool,
    // Shift used in the non-periodic case.
    shift: usize,
    // Small byteset borrowed from stdlib's two-way implementation.
    byteset: u64,
    pair_index1: usize,
    pair_index2: usize,
    pair_byte1: u8,
    pair_byte2: u8,
}

impl LiteralAlgorithm for TwoWay2 {
    type Needle = ByteNeedle;
    type State = TwoWay2State;

    const SUPPORTS_UNDERSCORE: bool = false;

    #[inline]
    fn compile_literal(src: &str) -> Option<Self::Needle> {
        compile_byte_literal(src)
    }

    #[inline]
    fn build_state(needle: &Self::Needle) -> Self::State {
        build_state(needle.bytes())
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

impl<'db> RowLiteralSearch<Utf8Column<'db>> for TwoWay2 {
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
        two_way2_find(&text[from..], pat, state).map(|pos| (pos + from) as u32)
    }
}

#[inline(always)]
fn maximal_suffix(pattern: &[u8], reversed: bool) -> (isize, usize) {
    let m = pattern.len();
    let ptr = pattern.as_ptr();

    let mut ms: isize = -1;
    let mut j: usize = 0;
    let mut k: usize = 1;
    let mut p: usize = 1;

    while j + k < m {
        unsafe {
            let a = *ptr.add(j + k);
            let b = *ptr.add((ms + k as isize) as usize);

            if (!reversed && a < b) || (reversed && a > b) {
                j += k;
                k = 1;
                p = (j as isize - ms) as usize;
            } else if a == b {
                if k != p {
                    k += 1;
                } else {
                    j += p;
                    k = 1;
                }
            } else {
                ms = j as isize;
                j = (ms + 1) as usize;
                k = 1;
                p = 1;
            }
        }
    }

    (ms, p)
}

#[inline]
fn build_state(pattern: &[u8]) -> TwoWay2State {
    let m = pattern.len();
    let (pair_index1, pair_index2) = pick_pair(pattern, 0);
    let pair_byte1 = pattern.get(pair_index1).copied().unwrap_or(0);
    let pair_byte2 = pattern.get(pair_index2).copied().unwrap_or(0);

    if m == 0 {
        return TwoWay2State {
            crit: 0,
            period: 1,
            is_periodic: true,
            shift: 1,
            byteset: 0,
            pair_index1,
            pair_index2,
            pair_byte1,
            pair_byte2,
        };
    }
    if m == 1 {
        return TwoWay2State {
            crit: 0,
            period: 1,
            is_periodic: false,
            shift: 1,
            byteset: byteset_create(pattern),
            pair_index1,
            pair_index2,
            pair_byte1,
            pair_byte2,
        };
    }

    let (ms1, p1) = maximal_suffix(pattern, false);
    let (ms2, p2) = maximal_suffix(pattern, true);

    let (ell, period) = if ms1 > ms2 { (ms1, p1) } else { (ms2, p2) };
    let crit = (ell + 1) as usize;

    let is_periodic =
        period < m && crit <= (m - period) && pattern[..crit] == pattern[period..period + crit];

    let shift = max(crit, m - crit) + 1;
    let (pair_index1, pair_index2) = pick_pair(pattern, crit);

    TwoWay2State {
        crit,
        period,
        is_periodic,
        shift,
        byteset: byteset_create(pattern),
        pair_index1,
        pair_index2,
        pair_byte1: pattern[pair_index1],
        pair_byte2: pattern[pair_index2],
    }
}

#[inline]
fn byteset_create(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(0u64, |acc, &b| acc | (1u64 << ((b & 0x3f) as usize)))
}

#[inline]
fn byteset_contains(state: &TwoWay2State, byte: u8) -> bool {
    ((state.byteset >> ((byte & 0x3f) as usize)) & 1) != 0
}

#[inline]
fn pick_pair(pattern: &[u8], crit: usize) -> (usize, usize) {
    let m = pattern.len();
    if m <= 1 {
        return (0, 0);
    }

    let last = m - 1;
    if pattern[0] != pattern[last] {
        return (0, last);
    }
    if crit < last && pattern[crit] != pattern[0] {
        return (crit, last);
    }
    for idx in 1..last {
        if pattern[idx] != pattern[0] {
            return (0, idx);
        }
    }
    (0, last)
}

#[inline(always)]
fn prefilter_delta(text: &[u8], state: &TwoWay2State, pos: usize, chunk: usize) -> Option<usize> {
    debug_assert!(chunk > 0);
    debug_assert!(chunk <= 16);

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
    state: &TwoWay2State,
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

pub fn two_way2_find(text: &[u8], pattern: &[u8], state: &TwoWay2State) -> Option<usize> {
    let n = text.len();
    let m = pattern.len();

    if m == 0 {
        return Some(0);
    }
    if m > n {
        return None;
    }
    if m == 1 {
        let needle = pattern[0];
        return text.iter().position(|&b| b == needle);
    }
    if m == 2 {
        let a = pattern[0];
        let b = pattern[1];
        let mut i = 0usize;
        while i + 1 < n {
            if text[i] == a && text[i + 1] == b {
                return Some(i);
            }
            i += 1;
        }
        return None;
    }

    let crit = state.crit;
    let pat = pattern.as_ptr();
    let txt = text.as_ptr();
    let last_off = m - 1;

    let last_start = n - m;
    let mut pos = 0usize;

    unsafe {
        if state.is_periodic {
            let mut memory = 0usize;

            while pos <= last_start {
                if !byteset_contains(state, text[pos + last_off]) {
                    pos += m;
                    memory = 0;
                    continue;
                }

                let chunk = (last_start - pos + 1).min(16);
                match prefilter_delta(text, state, pos, chunk) {
                    Some(0) => {}
                    Some(delta) => {
                        pos += delta;
                        memory = 0;
                    }
                    None => {
                        pos += chunk;
                        memory = 0;
                        continue;
                    }
                }

                let mut i = max(crit, memory);

                while i < m && *pat.add(i) == *txt.add(pos + i) {
                    i += 1;
                }

                if i >= m {
                    let mut i1 = crit;
                    while i1 > memory && *pat.add(i1 - 1) == *txt.add(pos + i1 - 1) {
                        i1 -= 1;
                    }

                    if i1 <= memory {
                        return Some(pos);
                    }

                    pos += state.period;
                    memory = m - state.period;
                } else {
                    pos += i + 1 - crit;
                    memory = 0;
                }
            }
        } else {
            while pos <= last_start {
                if !byteset_contains(state, text[pos + last_off]) {
                    pos += m;
                    continue;
                }

                let chunk = (last_start - pos + 1).min(16);
                match prefilter_delta(text, state, pos, chunk) {
                    Some(0) => {}
                    Some(delta) => pos += delta,
                    None => {
                        pos += chunk;
                        continue;
                    }
                }

                let mut i = crit;

                while i < m && *pat.add(i) == *txt.add(pos + i) {
                    i += 1;
                }

                if i >= m {
                    let mut i1 = crit;
                    while i1 > 0 && *pat.add(i1 - 1) == *txt.add(pos + i1 - 1) {
                        i1 -= 1;
                    }

                    if i1 == 0 {
                        return Some(pos);
                    }

                    pos += state.shift;
                } else {
                    pos += i + 1 - crit;
                }
            }
        }
    }

    None
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
