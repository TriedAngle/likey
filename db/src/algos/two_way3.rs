use core::cmp::max;

use crate::like::{LiteralAlgorithm, RowLiteralSearch};
use crate::storage::utf8::{Utf8Column, Utf8Row};

use super::utf8_shared::{
    ByteNeedle, byte_literal_len, compile_byte_literal, matches_at_bytes, utf8_row_len,
};

const MAX_PREFILTER_ANCHORS: usize = 4;

#[derive(Debug, Clone, Copy, Default)]
pub struct TwoWay3;

#[derive(Clone, Copy, Debug)]
pub struct TwoWay3State {
    // crit = ell + 1, so crit == 0 corresponds to ell == -1.
    crit: usize,
    period: usize,
    is_periodic: bool,
    // Shift used in the non-periodic case.
    shift: usize,
    // Small byteset borrowed from stdlib's two-way implementation.
    byteset: u64,
    pattern_len: usize,
    anchors: [PrefilterAnchor; MAX_PREFILTER_ANCHORS],
    anchor_count: usize,
    simd: PrefilterSimd,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PrefilterAnchor {
    index: usize,
    byte: u8,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefilterSimd {
    Scalar,
    Sse2,
    Avx2,
    Avx512,
    Neon,
}

impl LiteralAlgorithm for TwoWay3 {
    type Needle = ByteNeedle;
    type State = TwoWay3State;

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
}

impl<'db> RowLiteralSearch<Utf8Column<'db>> for TwoWay3 {
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
        two_way3_find(&text[from..], pat, state).map(|pos| (pos + from) as u32)
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
fn build_state(pattern: &[u8]) -> TwoWay3State {
    let m = pattern.len();
    let simd = best_prefilter_simd();

    if m == 0 {
        return TwoWay3State {
            crit: 0,
            period: 1,
            is_periodic: true,
            shift: 1,
            byteset: 0,
            pattern_len: m,
            anchors: [PrefilterAnchor::default(); MAX_PREFILTER_ANCHORS],
            anchor_count: 0,
            simd,
        };
    }
    if m == 1 {
        let mut anchors = [PrefilterAnchor::default(); MAX_PREFILTER_ANCHORS];
        anchors[0] = PrefilterAnchor {
            index: 0,
            byte: pattern[0],
        };
        return TwoWay3State {
            crit: 0,
            period: 1,
            is_periodic: false,
            shift: 1,
            byteset: byteset_create(pattern),
            pattern_len: m,
            anchors,
            anchor_count: 1,
            simd,
        };
    }

    let (ms1, p1) = maximal_suffix(pattern, false);
    let (ms2, p2) = maximal_suffix(pattern, true);

    let (ell, period) = if ms1 > ms2 { (ms1, p1) } else { (ms2, p2) };
    let crit = (ell + 1) as usize;

    let is_periodic =
        period < m && crit <= (m - period) && pattern[..crit] == pattern[period..period + crit];

    let shift = max(crit, m - crit) + 1;
    let (anchors, anchor_count) = pick_anchors(pattern, crit);

    TwoWay3State {
        crit,
        period,
        is_periodic,
        shift,
        byteset: byteset_create(pattern),
        pattern_len: m,
        anchors,
        anchor_count,
        simd,
    }
}

#[inline]
fn best_prefilter_simd() -> PrefilterSimd {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx2")
        {
            return PrefilterSimd::Avx512;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return PrefilterSimd::Avx2;
        }
        return PrefilterSimd::Sse2;
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return PrefilterSimd::Neon;
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        PrefilterSimd::Scalar
    }
}

#[inline]
fn byteset_create(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(0u64, |acc, &b| acc | (1u64 << ((b & 0x3f) as usize)))
}

#[inline]
fn byteset_contains(state: &TwoWay3State, byte: u8) -> bool {
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

#[inline]
fn pick_anchors(pattern: &[u8], crit: usize) -> ([PrefilterAnchor; MAX_PREFILTER_ANCHORS], usize) {
    let mut anchors = [PrefilterAnchor::default(); MAX_PREFILTER_ANCHORS];
    let mut count = 0usize;
    let m = pattern.len();
    let last = m - 1;
    let (pair_index1, pair_index2) = pick_pair(pattern, crit);

    push_anchor(&mut anchors, &mut count, pattern, pair_index1, false);
    push_anchor(&mut anchors, &mut count, pattern, pair_index2, false);

    let candidates = [crit, m / 2, m / 3, (2 * m) / 3, m / 4, (3 * m) / 4, 0, last];

    for idx in candidates {
        if count == MAX_PREFILTER_ANCHORS {
            break;
        }
        push_anchor(&mut anchors, &mut count, pattern, idx, true);
    }

    for idx in candidates {
        if count == MAX_PREFILTER_ANCHORS {
            break;
        }
        push_anchor(&mut anchors, &mut count, pattern, idx, false);
    }

    let mut idx = 0usize;
    while count < MAX_PREFILTER_ANCHORS && idx < m {
        push_anchor(&mut anchors, &mut count, pattern, idx, false);
        idx += 1;
    }

    (anchors, count)
}

#[inline]
fn push_anchor(
    anchors: &mut [PrefilterAnchor; MAX_PREFILTER_ANCHORS],
    count: &mut usize,
    pattern: &[u8],
    index: usize,
    require_new_byte: bool,
) -> bool {
    if *count == MAX_PREFILTER_ANCHORS || index >= pattern.len() {
        return false;
    }
    if anchors[..*count].iter().any(|anchor| anchor.index == index) {
        return false;
    }

    let byte = pattern[index];
    if require_new_byte && anchors[..*count].iter().any(|anchor| anchor.byte == byte) {
        return false;
    }

    anchors[*count] = PrefilterAnchor { index, byte };
    *count += 1;
    true
}

#[inline(always)]
fn prefilter_chunk_len(state: &TwoWay3State, remaining: usize) -> usize {
    debug_assert!(remaining > 0);

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if matches!(state.simd, PrefilterSimd::Avx512) && remaining >= 64 {
        return 64;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if matches!(state.simd, PrefilterSimd::Avx2 | PrefilterSimd::Avx512) && remaining >= 32 {
            return 32;
        }
        if remaining >= 16 {
            return 16;
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    if matches!(state.simd, PrefilterSimd::Neon) && remaining >= 16 {
        return 16;
    }

    remaining.min(16)
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
fn avx2_anchor_count(state: &TwoWay3State) -> usize {
    let max = if state.pattern_len >= 32 { 3 } else { 2 };
    state.anchor_count.min(max)
}

#[inline(always)]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn avx512_anchor_count(state: &TwoWay3State) -> usize {
    let max = if state.pattern_len >= 64 {
        4
    } else if state.pattern_len >= 32 {
        3
    } else {
        2
    };
    state.anchor_count.min(max)
}

#[inline(always)]
fn prefilter_delta(text: &[u8], state: &TwoWay3State, pos: usize, chunk: usize) -> Option<usize> {
    debug_assert!(chunk > 0);
    debug_assert!(chunk <= 64);

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if chunk == 64 && matches!(state.simd, PrefilterSimd::Avx512) {
        // SAFETY: caller ensures at least 64 candidate starts are in-bounds and
        // runtime feature detection selected AVX-512.
        unsafe {
            return x86::prefilter_delta_avx512(
                text,
                pos,
                &state.anchors,
                avx512_anchor_count(state),
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if chunk == 32 && matches!(state.simd, PrefilterSimd::Avx2 | PrefilterSimd::Avx512) {
            // SAFETY: caller ensures at least 32 candidate starts are in-bounds and
            // runtime feature detection selected AVX2 or better.
            unsafe {
                return x86::prefilter_delta_avx2(
                    text,
                    pos,
                    &state.anchors,
                    avx2_anchor_count(state),
                );
            }
        }
        if chunk == 16 {
            // SAFETY: caller ensures at least 16 candidate starts are in-bounds.
            unsafe {
                return x86::prefilter_delta_sse2(
                    text,
                    pos,
                    &state.anchors,
                    state.anchor_count.min(2),
                );
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    if chunk == 16 && matches!(state.simd, PrefilterSimd::Neon) {
        // SAFETY: caller ensures at least 16 candidate starts are in-bounds.
        unsafe {
            return neon::prefilter_delta_neon(
                text,
                pos,
                &state.anchors,
                state.anchor_count.min(2),
            );
        }
    }

    prefilter_delta_scalar(text, state, pos, chunk, state.anchor_count.min(2))
}

#[inline(always)]
fn prefilter_delta_scalar(
    text: &[u8],
    state: &TwoWay3State,
    pos: usize,
    chunk: usize,
    anchor_count: usize,
) -> Option<usize> {
    for lane in 0..chunk {
        let cand = pos + lane;
        let mut matched = true;
        for anchor in &state.anchors[..anchor_count] {
            if text[cand + anchor.index] != anchor.byte {
                matched = false;
                break;
            }
        }
        if matched {
            return Some(lane);
        }
    }
    None
}

pub fn two_way3_find(text: &[u8], pattern: &[u8], state: &TwoWay3State) -> Option<usize> {
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

                let chunk = prefilter_chunk_len(state, last_start - pos + 1);
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

                let chunk = prefilter_chunk_len(state, last_start - pos + 1);
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

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use core::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    pub unsafe fn prefilter_delta_sse2(
        text: &[u8],
        pos: usize,
        anchors: &[PrefilterAnchor; MAX_PREFILTER_ANCHORS],
        anchor_count: usize,
    ) -> Option<usize> {
        debug_assert!(anchor_count > 0);
        debug_assert!(anchor_count <= 2);

        let first = anchors[0];
        let first_vec = _mm_set1_epi8(first.byte as i8);
        let first_chunk =
            unsafe { _mm_loadu_si128(text.as_ptr().add(pos + first.index).cast::<__m128i>()) };
        let mut mask = _mm_movemask_epi8(_mm_cmpeq_epi8(first_chunk, first_vec)) as u32;
        if mask == 0 {
            return None;
        }

        if anchor_count >= 2 {
            let anchor = anchors[1];
            let needle = _mm_set1_epi8(anchor.byte as i8);
            let chunk =
                unsafe { _mm_loadu_si128(text.as_ptr().add(pos + anchor.index).cast::<__m128i>()) };
            mask &= _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, needle)) as u32;
        }

        if mask == 0 {
            None
        } else {
            Some(mask.trailing_zeros() as usize)
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn prefilter_delta_avx2(
        text: &[u8],
        pos: usize,
        anchors: &[PrefilterAnchor; MAX_PREFILTER_ANCHORS],
        anchor_count: usize,
    ) -> Option<usize> {
        debug_assert!(anchor_count > 0);
        debug_assert!(anchor_count <= 3);

        let first = anchors[0];
        let first_vec = _mm256_set1_epi8(first.byte as i8);
        let first_chunk =
            unsafe { _mm256_loadu_si256(text.as_ptr().add(pos + first.index).cast::<__m256i>()) };
        let mut mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(first_chunk, first_vec)) as u32;
        if mask == 0 {
            return None;
        }

        if anchor_count >= 2 {
            let anchor = anchors[1];
            let needle = _mm256_set1_epi8(anchor.byte as i8);
            let chunk = unsafe {
                _mm256_loadu_si256(text.as_ptr().add(pos + anchor.index).cast::<__m256i>())
            };
            mask &= _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, needle)) as u32;
            if mask == 0 {
                return None;
            }
        }
        if anchor_count >= 3 {
            let anchor = anchors[2];
            let needle = _mm256_set1_epi8(anchor.byte as i8);
            let chunk = unsafe {
                _mm256_loadu_si256(text.as_ptr().add(pos + anchor.index).cast::<__m256i>())
            };
            mask &= _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, needle)) as u32;
        }

        if mask == 0 {
            None
        } else {
            Some(mask.trailing_zeros() as usize)
        }
    }

    #[cfg(feature = "avx512")]
    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn prefilter_delta_avx512(
        text: &[u8],
        pos: usize,
        anchors: &[PrefilterAnchor; MAX_PREFILTER_ANCHORS],
        anchor_count: usize,
    ) -> Option<usize> {
        debug_assert!(anchor_count > 0);
        debug_assert!(anchor_count <= 4);

        let first = anchors[0];
        let first_vec = _mm512_set1_epi8(first.byte as i8);
        let first_chunk =
            unsafe { _mm512_loadu_si512(text.as_ptr().add(pos + first.index).cast()) };
        let mut mask = _mm512_cmpeq_epi8_mask(first_chunk, first_vec) as u64;
        if mask == 0 {
            return None;
        }

        if anchor_count >= 2 {
            let anchor = anchors[1];
            let needle = _mm512_set1_epi8(anchor.byte as i8);
            let chunk = unsafe { _mm512_loadu_si512(text.as_ptr().add(pos + anchor.index).cast()) };
            mask &= _mm512_cmpeq_epi8_mask(chunk, needle) as u64;
            if mask == 0 {
                return None;
            }
        }
        if anchor_count >= 3 {
            let anchor = anchors[2];
            let needle = _mm512_set1_epi8(anchor.byte as i8);
            let chunk = unsafe { _mm512_loadu_si512(text.as_ptr().add(pos + anchor.index).cast()) };
            mask &= _mm512_cmpeq_epi8_mask(chunk, needle) as u64;
            if mask == 0 {
                return None;
            }
        }
        if anchor_count >= 4 {
            let anchor = anchors[3];
            let needle = _mm512_set1_epi8(anchor.byte as i8);
            let chunk = unsafe { _mm512_loadu_si512(text.as_ptr().add(pos + anchor.index).cast()) };
            mask &= _mm512_cmpeq_epi8_mask(chunk, needle) as u64;
        }

        if mask == 0 {
            None
        } else {
            Some(mask.trailing_zeros() as usize)
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    use super::*;
    use core::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub unsafe fn prefilter_delta_neon(
        text: &[u8],
        pos: usize,
        anchors: &[PrefilterAnchor; MAX_PREFILTER_ANCHORS],
        anchor_count: usize,
    ) -> Option<usize> {
        debug_assert!(anchor_count > 0);
        debug_assert!(anchor_count <= 2);

        let first = anchors[0];
        let v1 = vdupq_n_u8(first.byte);
        let chunk1 = unsafe { vld1q_u8(text.as_ptr().add(pos + first.index)) };
        let mut eq = vceqq_u8(chunk1, v1);

        if anchor_count >= 2 {
            let anchor = anchors[1];
            let needle = vdupq_n_u8(anchor.byte);
            let chunk = unsafe { vld1q_u8(text.as_ptr().add(pos + anchor.index)) };
            eq = vandq_u8(eq, vceqq_u8(chunk, needle));
        }

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

    #[test]
    fn avx_anchor_selection_prefers_distinct_positions() {
        let (anchors, count) = pick_anchors(b"abcdefgh", 3);

        assert_eq!(count, 4);
        assert_eq!(anchors[0].index, 0);
        assert_eq!(anchors[1].index, 7);
        assert!(anchors[..count].iter().any(|anchor| anchor.index == 3));
        assert!(anchors[..count].iter().any(|anchor| anchor.index == 4));
    }
}
