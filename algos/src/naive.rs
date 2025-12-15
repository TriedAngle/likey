use crate::StringSearch;

pub struct Naive;
pub struct NaiveScalar;
pub struct NaiveVectorized;

impl StringSearch for Naive {
    type Config = ();
    type State = ();
    fn find_bytes(_state: Self::State, text: &[u8], pattern: &[u8]) -> Option<usize> {
        naive_find(text, pattern)
    }

    fn find_all_bytes(_state: Self::State, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        naive_find_all(text, pattern)
    }
}

impl StringSearch for NaiveScalar {
    type Config = ();
    type State = ();
    fn find_bytes(_state: Self::State, text: &[u8], pattern: &[u8]) -> Option<usize> {
        naive_find_scalar(text, pattern)
    }

    fn find_all_bytes(_state: Self::State, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        naive_find_all_scalar(text, pattern)
    }
}

impl StringSearch for NaiveVectorized {
    type Config = ();
    type State = ();
    fn find_bytes(_state: Self::State, text: &[u8], pattern: &[u8]) -> Option<usize> {
        unsafe { neon::naive_find_neon(text, pattern) }
    }

    fn find_all_bytes(_state: Self::State, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        unsafe { neon::naive_find_all_neon(text, pattern) }
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

pub fn naive_find_all_scalar(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    let n = text.len();
    let m = pattern.len();
    let mut result = Vec::new();

    if m == 0 {
        for i in 0..=n {
            result.push(i);
        }
        return result;
    }
    if m > n {
        return result;
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
            result.push(i);
        }
    }

    result
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

    /// neon accellerated naive string search to yield all findings
    /// # Safety
    /// a device with neon support is required
    pub unsafe fn naive_find_all_neon(text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len();
        let m = pattern.len();
        let mut result = Vec::new();

        if m == 0 {
            // Same convention as scalar: match at every position.
            for i in 0..=n {
                result.push(i);
            }
            return result;
        }
        if m > n {
            return result;
        }

        let first = pattern[0];
        let first_vec = unsafe { vdupq_n_u8(first) };
        let chunk_size = 16;

        let mut i = 0;

        // Vectorized scanning, but collect *all* matches (overlapping allowed).
        while i + chunk_size <= n {
            let ptr = unsafe { text.as_ptr().add(i) };
            let chunk = unsafe { vld1q_u8(ptr) };
            let cmp = unsafe { vceqq_u8(chunk, first_vec) };

            let mut lanes = [0u8; 16];
            unsafe { vst1q_u8(lanes.as_mut_ptr(), cmp) };

            for (lane, &value) in lanes.iter().enumerate().take(chunk_size) {
                if value == 0xFF {
                    let cand = i + lane;
                    if cand + m <= n && &text[cand..cand + m] == pattern {
                        result.push(cand);
                    }
                }
            }

            i += chunk_size;
        }

        // Scalar tail for remaining positions
        while i + m <= n {
            if &text[i..i + m] == pattern {
                result.push(i);
            }
            i += 1;
        }

        result
    }
}

pub fn naive_find(text: &[u8], pattern: &[u8]) -> Option<usize> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        log::debug!("naive_find: using NEON implementation");
        // Safety: guarded by cfg for aarch64+neon.
        unsafe { neon::naive_find_neon(text, pattern) }
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        log::debug!("naive_find: using scalar implementation");
        naive_find_scalar(text, pattern)
    }
}

pub fn naive_find_all(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        log::debug!("naive_find_all: using NEON implementation");
        // Safety: guarded by cfg for aarch64+neon.
        unsafe { neon::naive_find_all_neon(text, pattern) }
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        debug!("naive_find_all: using scalar implementation");
        naive_find_all_scalar(text, pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(test)]
    mod tests {
        use super::*;

        fn run_shared_tests<F, G>(mut find: F, mut find_all: G)
        where
            F: FnMut(&[u8], &[u8]) -> Option<usize>,
            G: FnMut(&[u8], &[u8]) -> Vec<usize>,
        {
            // basic
            let hay = b"ababcabcabababd";
            let pat = b"ababd";
            assert_eq!(find(hay, pat), Some(10));

            let hay = b"hello world";
            let pat = b"rust";
            assert_eq!(find(hay, pat), None);

            let hay = b"abc";
            let pat: &[u8] = b"";
            assert_eq!(find(hay, pat), Some(0));

            let hay = b"aaaa";
            let pat = b"aa";
            assert_eq!(find_all(hay, pat), vec![0, 1, 2]);

            let hay = "🌍hello🌍hello".as_bytes();
            let pat = "🌍hello".as_bytes();

            assert_eq!("🌍".len(), 4);
            assert_eq!("hello".len(), 5);
            assert_eq!("🌍hello".len(), 9);
            assert_eq!("🌍hello🌍hello".len(), 18);

            assert_eq!(find(hay, pat), Some(0));
            assert_eq!(find_all(hay, pat), vec![0, "🌍hello".len()]);
        }

        #[test]
        fn scalar_impl_is_correct() {
            run_shared_tests(naive_find_scalar, naive_find_all_scalar);
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        #[test]
        fn neon_impl_is_correct() {
            run_shared_tests(
                |h, n| unsafe { super::neon::naive_find_neon(h, n) },
                |h, n| unsafe { super::neon::naive_find_all_neon(h, n) },
            );
        }

        #[test]
        fn public_api_behaves_correctly() {
            let hay = b"ababcabcabababd";
            let pat = b"ababd";
            assert_eq!(naive_find(hay, pat), Some(10));

            let hay = b"aaaa";
            let pat = b"aa";
            assert_eq!(naive_find_all(hay, pat), vec![0, 1, 2]);
        }
    }
}
