#![allow(dead_code, unused_variables)]
use crate::StringSearch;

pub struct LutShort<'a>(std::marker::PhantomData<&'a ()>);

#[derive(Clone, Debug)]
pub struct LutShortState {
    pattern: [u8; 8],
    len: usize,
    sig: u8,
    sig_index: usize,
    lut_lo: [u8; 16],
    lut_hi: [u8; 16],
}

impl<'a> StringSearch for LutShort<'a> {
    type Config = &'a [u8];
    type State = LutShortState;

    fn build(config: &Self::Config) -> Self::State {
        build_state(config)
    }

    fn find_bytes(config: &Self::Config, state: &Self::State, text: &[u8]) -> Option<usize> {
        #[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
        unsafe {
            return x86::find_ssse3(state, text, config);
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            return neon::find_neon(state, text, config);
        }

        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "ssse3"),
            all(target_arch = "aarch64", target_feature = "neon")
        )))]
        {
            let _ = state;
            let _ = text;
            unimplemented!("lut-short requires SSSE3 or NEON");
        }
    }
}

fn build_state(pattern: &[u8]) -> LutShortState {
    let len = pattern.len();
    let mut buf = [0u8; 8];
    if len > 0 {
        let copy_len = len.min(8);
        buf[..copy_len].copy_from_slice(&pattern[..copy_len]);
    }

    let (sig, sig_index) = rarest_byte(pattern);
    let mut lut_lo = [0u8; 16];
    let mut lut_hi = [0u8; 16];
    let lo = (sig & 0x0f) as usize;
    let hi = (sig >> 4) as usize;
    lut_lo[lo] = 0xff;
    lut_hi[hi] = 0xff;

    LutShortState {
        pattern: buf,
        len,
        sig,
        sig_index,
        lut_lo,
        lut_hi,
    }
}

fn rarest_byte(pattern: &[u8]) -> (u8, usize) {
    if pattern.is_empty() {
        return (0, 0);
    }

    let mut counts = [0u8; 256];
    for &b in pattern {
        counts[b as usize] = counts[b as usize].saturating_add(1);
    }

    let mut best = pattern[0];
    let mut best_idx = 0usize;
    let mut best_count = counts[best as usize];

    for (idx, &b) in pattern.iter().enumerate() {
        let count = counts[b as usize];
        if count < best_count {
            best = b;
            best_idx = idx;
            best_count = count;
        }
    }

    (best, best_idx)
}

fn matches_at(state: &LutShortState, text: &[u8], pos: usize) -> bool {
    let m = state.len;
    if pos + m > text.len() {
        return false;
    }
    let pat = &state.pattern[..m];
    &text[pos..pos + m] == pat
}

#[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
mod x86 {
    use super::{matches_at, LutShortState};
    use core::arch::x86_64::*;

    #[target_feature(enable = "ssse3,sse2")]
    pub unsafe fn find_ssse3(state: &LutShortState, text: &[u8], pattern: &[u8]) -> Option<usize> {
        let m = state.len;
        let n = text.len();
        if m == 0 {
            return Some(0);
        }
        if m > 8 {
            return crate::naive::naive_find_scalar(text, pattern);
        }
        if m > n {
            return None;
        }
        if m == 1 {
            let target = state.pattern[0];
            for (i, &b) in text.iter().enumerate() {
                if b == target {
                    return Some(i);
                }
            }
            return None;
        }

        let lut_lo = unsafe { _mm_loadu_si128(state.lut_lo.as_ptr() as *const __m128i) };
        let lut_hi = unsafe { _mm_loadu_si128(state.lut_hi.as_ptr() as *const __m128i) };
        let mask_0f = _mm_set1_epi8(0x0f);
        let sig_index = state.sig_index;

        let mut i = 0usize;
        while i + 64 <= n {
            for block in 0..4 {
                let base = i + block * 16;
                let ptr = unsafe { text.as_ptr().add(base) as *const __m128i };
                let chunk = unsafe { _mm_loadu_si128(ptr) };
                let lo = _mm_and_si128(chunk, mask_0f);
                let hi = _mm_and_si128(_mm_srli_epi16(chunk, 4), mask_0f);
                let lo_mask = _mm_shuffle_epi8(lut_lo, lo);
                let hi_mask = _mm_shuffle_epi8(lut_hi, hi);
                let eq = _mm_and_si128(lo_mask, hi_mask);
                let mut mask = _mm_movemask_epi8(eq) as u32;

                while mask != 0 {
                    let bit = mask.trailing_zeros() as usize;
                    mask &= mask - 1;
                    let cand = base + bit;
                    if cand < sig_index {
                        continue;
                    }
                    let start = cand - sig_index;
                    if start + m <= n && matches_at(state, text, start) {
                        return Some(start);
                    }
                }
            }
            i += 64;
        }

        while i + 16 <= n {
            if let Some(pos) = unsafe { scan_block(state, text, i, 16) } {
                return Some(pos);
            }
            i += 16;
        }

        if i < n {
            let rem = n - i;
            if let Some(pos) = unsafe { scan_tail(state, text, i, rem) } {
                return Some(pos);
            }
        }

        None
    }

    #[target_feature(enable = "ssse3,sse2")]
    unsafe fn scan_block(
        state: &LutShortState,
        text: &[u8],
        base: usize,
        limit: usize,
    ) -> Option<usize> {
        let m = state.len;
        let n = text.len();
        let lut_lo = unsafe { _mm_loadu_si128(state.lut_lo.as_ptr() as *const __m128i) };
        let lut_hi = unsafe { _mm_loadu_si128(state.lut_hi.as_ptr() as *const __m128i) };
        let mask_0f = _mm_set1_epi8(0x0f);
        let sig_index = state.sig_index;

        let ptr = unsafe { text.as_ptr().add(base) as *const __m128i };
        let chunk = unsafe { _mm_loadu_si128(ptr) };
        let lo = _mm_and_si128(chunk, mask_0f);
        let hi = _mm_and_si128(_mm_srli_epi16(chunk, 4), mask_0f);
        let lo_mask = _mm_shuffle_epi8(lut_lo, lo);
        let hi_mask = _mm_shuffle_epi8(lut_hi, hi);
        let eq = _mm_and_si128(lo_mask, hi_mask);
        let mut mask = _mm_movemask_epi8(eq) as u32;
        if limit < 16 {
            let limit_mask = if limit == 0 { 0 } else { (1u32 << limit) - 1 };
            mask &= limit_mask;
        }

        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let cand = base + bit;
            if cand < sig_index {
                continue;
            }
            let start = cand - sig_index;
            if start + m <= n && matches_at(state, text, start) {
                return Some(start);
            }
        }

        None
    }

    #[target_feature(enable = "ssse3,sse2")]
    unsafe fn scan_tail(
        state: &LutShortState,
        text: &[u8],
        base: usize,
        rem: usize,
    ) -> Option<usize> {
        let fill = state.sig.wrapping_add(1);
        let mut tmp = [fill; 16];
        tmp[..rem].copy_from_slice(&text[base..base + rem]);

        let lut_lo = unsafe { _mm_loadu_si128(state.lut_lo.as_ptr() as *const __m128i) };
        let lut_hi = unsafe { _mm_loadu_si128(state.lut_hi.as_ptr() as *const __m128i) };
        let mask_0f = _mm_set1_epi8(0x0f);
        let sig_index = state.sig_index;
        let m = state.len;
        let n = text.len();

        let chunk = unsafe { _mm_loadu_si128(tmp.as_ptr() as *const __m128i) };
        let lo = _mm_and_si128(chunk, mask_0f);
        let hi = _mm_and_si128(_mm_srli_epi16(chunk, 4), mask_0f);
        let lo_mask = _mm_shuffle_epi8(lut_lo, lo);
        let hi_mask = _mm_shuffle_epi8(lut_hi, hi);
        let eq = _mm_and_si128(lo_mask, hi_mask);
        let mut mask = _mm_movemask_epi8(eq) as u32;
        let limit_mask = if rem == 0 { 0 } else { (1u32 << rem) - 1 };
        mask &= limit_mask;

        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let cand = base + bit;
            if cand < sig_index {
                continue;
            }
            let start = cand - sig_index;
            if start + m <= n && matches_at(state, text, start) {
                return Some(start);
            }
        }

        None
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    use super::{matches_at, LutShortState};
    use core::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub unsafe fn find_neon(state: &LutShortState, text: &[u8], pattern: &[u8]) -> Option<usize> {
        let m = state.len;
        let n = text.len();
        if m == 0 {
            return Some(0);
        }
        if m > 8 {
            return crate::naive::naive_find_scalar(text, pattern);
        }
        if m > n {
            return None;
        }
        if m == 1 {
            let target = state.pattern[0];
            for (i, &b) in text.iter().enumerate() {
                if b == target {
                    return Some(i);
                }
            }
            return None;
        }

        let lut_lo = unsafe { vld1q_u8(state.lut_lo.as_ptr()) };
        let lut_hi = unsafe { vld1q_u8(state.lut_hi.as_ptr()) };
        let mask_0f = unsafe { vdupq_n_u8(0x0f) };
        let sig_index = state.sig_index;

        let mut i = 0usize;
        while i + 64 <= n {
            for block in 0..4 {
                let base = i + block * 16;
                if let Some(pos) =
                    unsafe { scan_block(state, text, base, 16, lut_lo, lut_hi, mask_0f) }
                {
                    return Some(pos);
                }
            }
            i += 64;
        }

        while i + 16 <= n {
            if let Some(pos) = unsafe { scan_block(state, text, i, 16, lut_lo, lut_hi, mask_0f) } {
                return Some(pos);
            }
            i += 16;
        }

        if i < n {
            let rem = n - i;
            let fill = state.sig.wrapping_add(1);
            let mut tmp = [fill; 16];
            tmp[..rem].copy_from_slice(&text[i..i + rem]);
            let chunk = unsafe { vld1q_u8(tmp.as_ptr()) };
            if let Some(pos) =
                unsafe { scan_chunk(state, text, i, rem, chunk, lut_lo, lut_hi, mask_0f) }
            {
                return Some(pos);
            }
        }

        None
    }

    #[target_feature(enable = "neon")]
    unsafe fn scan_block(
        state: &LutShortState,
        text: &[u8],
        base: usize,
        limit: usize,
        lut_lo: uint8x16_t,
        lut_hi: uint8x16_t,
        mask_0f: uint8x16_t,
    ) -> Option<usize> {
        let ptr = unsafe { text.as_ptr().add(base) };
        let chunk = unsafe { vld1q_u8(ptr) };
        unsafe { scan_chunk(state, text, base, limit, chunk, lut_lo, lut_hi, mask_0f) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn scan_chunk(
        state: &LutShortState,
        text: &[u8],
        base: usize,
        limit: usize,
        chunk: uint8x16_t,
        lut_lo: uint8x16_t,
        lut_hi: uint8x16_t,
        mask_0f: uint8x16_t,
    ) -> Option<usize> {
        let m = state.len;
        let n = text.len();
        let sig_index = state.sig_index;

        let lo = unsafe { vandq_u8(chunk, mask_0f) };
        let hi = unsafe { vandq_u8(vshrq_n_u8(chunk, 4), mask_0f) };
        let lo_mask = unsafe { vqtbl1q_u8(lut_lo, lo) };
        let hi_mask = unsafe { vqtbl1q_u8(lut_hi, hi) };
        let eq = unsafe { vandq_u8(lo_mask, hi_mask) };

        let mut lanes = [0u8; 16];
        unsafe { vst1q_u8(lanes.as_mut_ptr(), eq) };

        for lane in 0..limit {
            if lanes[lane] == 0xff {
                let cand = base + lane;
                if cand < sig_index {
                    continue;
                }
                let start = cand - sig_index;
                if start + m <= n && matches_at(state, text, start) {
                    return Some(start);
                }
            }
        }

        None
    }
}

#[cfg(all(
    test,
    any(
        all(target_arch = "x86_64", target_feature = "ssse3"),
        all(target_arch = "aarch64", target_feature = "neon")
    )
))]
mod tests {
    use super::LutShort;
    use crate::StringSearch;

    #[test]
    fn test_short_matches() {
        let text = b"xxabcxxabcdxx";
        let patterns = [b"a", b"ab", b"abc", b"abcd", b"bc", b"x", b"xxa"];

        for pat in patterns.iter() {
            let config = &pat[..];
            let state = LutShort::build(&config);
            let found = LutShort::find_bytes(&config, &state, text);
            let expected = text.windows(pat.len()).position(|w| w == *pat);
            assert_eq!(found, expected, "pattern {:?}", pat);
        }
    }

    #[test]
    fn test_no_match() {
        let text = b"abcdefg";
        let pat = b"hij";
        let config = &pat[..];
        let state = LutShort::build(&config);
        let found = LutShort::find_bytes(&config, &state, text);
        assert_eq!(found, None);
    }

    #[test]
    fn test_too_long_pattern() {
        let text = b"abcdefg";
        let pat = b"abcdefghi";
        let config = &pat[..];
        let state = LutShort::build(&config);
        let found = LutShort::find_bytes(&config, &state, text);
        assert_eq!(found, None);
    }
}
