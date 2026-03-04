use crate::StringSearch;
use std::marker::PhantomData;

pub struct Naive<'a>(PhantomData<&'a ()>);
pub struct NaiveScalar<'a>(PhantomData<&'a ()>);
pub struct NaiveVectorized<'a>(PhantomData<&'a ()>);
pub struct NaiveVectorizedV2<'a>(PhantomData<&'a ()>);
pub struct NaiveMixed<'a>(PhantomData<&'a ()>);

impl<'a> StringSearch for Naive<'a> {
    type Config = &'a [u8];
    type State = ();
    fn build(_config: &Self::Config) -> Self::State {
        ()
    }
    fn find_bytes(config: &Self::Config, _state: &Self::State, text: &[u8]) -> Option<usize> {
        naive_find(text, config)
    }
}

impl<'a> StringSearch for NaiveScalar<'a> {
    type Config = &'a [u8];
    type State = ();
    fn build(_config: &Self::Config) -> Self::State {
        ()
    }
    fn find_bytes(config: &Self::Config, _state: &Self::State, text: &[u8]) -> Option<usize> {
        naive_find_scalar(text, config)
    }
}

impl<'a> StringSearch for NaiveVectorized<'a> {
    type Config = &'a [u8];
    type State = ();
    fn build(_config: &Self::Config) -> Self::State {
        ()
    }
    fn find_bytes(config: &Self::Config, _state: &Self::State, text: &[u8]) -> Option<usize> {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            neon::naive_find_neon(text, config)
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        unsafe {
            x86::naive_find_sse2(text, config)
        }

        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "x86_64", target_feature = "sse2")
        )))]
        {
            let _ = config;
            let _ = text;
            unimplemented!("no vector backend")
        }
    }
}

impl<'a> StringSearch for NaiveVectorizedV2<'a> {
    type Config = &'a [u8];
    type State = ();
    fn build(_config: &Self::Config) -> Self::State {
        ()
    }
    fn find_bytes(config: &Self::Config, _state: &Self::State, text: &[u8]) -> Option<usize> {
        naive_find_vectorized_v2(text, config)
    }
}

impl<'a> StringSearch for NaiveMixed<'a> {
    type Config = &'a [u8];
    type State = ();
    fn build(_config: &Self::Config) -> Self::State {
        ()
    }
    fn find_bytes(config: &Self::Config, _state: &Self::State, text: &[u8]) -> Option<usize> {
        naive_find_mixed(text, config)
    }
}

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

pub fn naive_find_mixed(text: &[u8], pattern: &[u8]) -> Option<usize> {
    let m = pattern.len();
    let n = text.len();

    if m <= 3 || n < 64 {
        return naive_find_scalar(text, pattern);
    }

    naive_find_vectorized_v2(text, pattern)
}

pub fn naive_find_vectorized_v2(text: &[u8], pattern: &[u8]) -> Option<usize> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // Safety: guarded by cfg for aarch64+neon.
        return unsafe { neon::naive_find_neon_v2(text, pattern) };
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        // Safety: guarded by cfg for x86_64+sse2.
        return unsafe { x86::naive_find_sse2_v2(text, pattern) };
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    {
        naive_find_scalar(text, pattern)
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod neon {
    use core::arch::aarch64::*;

    /// neon accellerated naive string search
    /// # Safety
    /// a device with neon support is required
    pub unsafe fn naive_find_neon(text: &[u8], pattern: &[u8]) -> Option<usize> {
        let n = text.len();
        let m = pattern.len();

        if m == 0 {
            return Some(0);
        }
        if m > n {
            return None;
        }

        let first = pattern[0];
        let first_vec = unsafe { vdupq_n_u8(first) };
        let chunk_size = 16;

        let mut i = 0;

        // Vectorized scanning for the first byte of the pattern.
        while i + chunk_size <= n {
            let ptr = unsafe { text.as_ptr().add(i) };
            let chunk = unsafe { vld1q_u8(ptr) };
            let cmp = unsafe { vceqq_u8(chunk, first_vec) };

            let mut lanes = [0; 16];
            unsafe { vst1q_u8(lanes.as_mut_ptr(), cmp) };

            for (lane, &value) in lanes.iter().enumerate().take(chunk_size) {
                if value == 0xFF {
                    let cand = i + lane;
                    if cand + m <= n && &text[cand..cand + m] == pattern {
                        return Some(cand);
                    }
                }
            }

            i += chunk_size;
        }

        // Scalar tail
        while i + m <= n {
            if &text[i..i + m] == pattern {
                return Some(i);
            }
            i += 1;
        }

        None
    }

    /// NEON accelerated naive string search (v2)
    /// Prefilters with first and last pattern byte.
    /// # Safety
    /// a device with neon support is required
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
            let ptr_first = unsafe { text.as_ptr().add(i) };
            let ptr_last = unsafe { text.as_ptr().add(i + last_off) };

            let chunk_first = unsafe { vld1q_u8(ptr_first) };
            let chunk_last = unsafe { vld1q_u8(ptr_last) };

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

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
pub mod x86 {
    use core::arch::x86_64::*;

    /// SSE2 accelerated naive string search
    /// # Safety
    /// Requires x86_64 with SSE2 support.
    pub unsafe fn naive_find_sse2(text: &[u8], pattern: &[u8]) -> Option<usize> {
        let n = text.len();
        let m = pattern.len();

        if m == 0 {
            return Some(0);
        }
        if m > n {
            return None;
        }

        let first = pattern[0] as i8;
        let first_vec = unsafe { _mm_set1_epi8(first) };
        let chunk_size = 16;

        let mut i = 0usize;
        while i + chunk_size <= n {
            let ptr = unsafe { text.as_ptr().add(i) as *const __m128i };
            let chunk = unsafe { _mm_loadu_si128(ptr) };
            let cmp = unsafe { _mm_cmpeq_epi8(chunk, first_vec) };
            let mut mask = unsafe { _mm_movemask_epi8(cmp) } as u32;

            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                let cand = i + lane;
                if cand + m <= n && &text[cand..cand + m] == pattern {
                    return Some(cand);
                }
                mask &= mask - 1;
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

    /// SSE2 accelerated naive string search (v2)
    /// Prefilters with first and last pattern byte.
    /// # Safety
    /// Requires x86_64 with SSE2 support.
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

        let first_vec = unsafe { _mm_set1_epi8(pattern[0] as i8) };
        let last_vec = unsafe { _mm_set1_epi8(pattern[m - 1] as i8) };
        let chunk_size = 16usize;
        let last_off = m - 1;

        let mut i = 0usize;
        while i + chunk_size + last_off <= n {
            let ptr_first = unsafe { text.as_ptr().add(i) as *const __m128i };
            let ptr_last = unsafe { text.as_ptr().add(i + last_off) as *const __m128i };

            let chunk_first = unsafe { _mm_loadu_si128(ptr_first) };
            let chunk_last = unsafe { _mm_loadu_si128(ptr_last) };
            let cmp_first = unsafe { _mm_cmpeq_epi8(chunk_first, first_vec) };
            let cmp_last = unsafe { _mm_cmpeq_epi8(chunk_last, last_vec) };

            let mask_first = unsafe { _mm_movemask_epi8(cmp_first) } as u32;
            let mask_last = unsafe { _mm_movemask_epi8(cmp_last) } as u32;
            let mut mask = mask_first & mask_last;

            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                let cand = i + lane;
                if &text[cand..cand + m] == pattern {
                    return Some(cand);
                }
                mask &= mask - 1;
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

pub fn naive_find(text: &[u8], pattern: &[u8]) -> Option<usize> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // Safety: guarded by cfg for aarch64+neon.
        unsafe { neon::naive_find_neon(text, pattern) }
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        naive_find_scalar(text, pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naive_scalar_basic() {
        let hay = b"ababcabcabababd";
        let pat: &[u8] = b"ababd";
        assert_eq!(
            NaiveScalar::find(&pat, std::str::from_utf8(hay).unwrap()),
            Some(10)
        );
    }

    #[test]
    fn test_naive_auto_basic() {
        let hay = b"ababcabcabababd";
        let pat: &[u8] = b"ababd";
        assert_eq!(
            Naive::find(&pat, std::str::from_utf8(hay).unwrap()),
            Some(10)
        );
    }

    #[test]
    fn test_naive_not_found() {
        let hay = "hello world";
        let pat: &[u8] = b"rust";
        assert_eq!(Naive::find(&pat, hay), None);
    }

    #[test]
    fn test_naive_empty_pattern() {
        let hay = "abc";
        let pat: &[u8] = b"";
        assert_eq!(Naive::find(&pat, hay), Some(0));
    }

    #[test]
    fn test_naive_vectorized_v2_basic() {
        let hay = b"ababcabcabababd";
        let pat: &[u8] = b"ababd";
        assert_eq!(NaiveVectorizedV2::find(&pat, "ababcabcabababd"), Some(10));
        assert_eq!(naive_find_vectorized_v2(hay, pat), Some(10));
    }

    #[test]
    fn test_naive_mixed_basic() {
        let pat: &[u8] = b"hello";
        assert_eq!(NaiveMixed::find(&pat, "xxhellozz"), Some(2));
    }

    #[test]
    fn test_naive_vectorized_v2_matches_scalar() {
        let text = b"aaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaa";
        let pattern = b"aaab";
        assert_eq!(
            naive_find_vectorized_v2(text, pattern),
            naive_find_scalar(text, pattern)
        );
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn test_naive_neon_explicit() {
        let hay = b"ababcabcabababd";
        let pat: &[u8] = b"ababd";
        assert_eq!(
            NaiveVectorized::find(&pat, std::str::from_utf8(hay).unwrap()),
            Some(10)
        );
    }
}
