use crate::StringSearch;
use std::marker::PhantomData;

pub struct Naive<'a>(PhantomData<&'a ()>);
pub struct NaiveScalar<'a>(PhantomData<&'a ()>);
pub struct NaiveVectorized<'a>(PhantomData<&'a ()>);

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
        unsafe { neon::naive_find_neon(text, config) }
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
