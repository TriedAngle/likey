use crate::StringSearch;
use std::marker::PhantomData;

pub struct KMP<'a>(PhantomData<&'a ()>);

impl<'a> StringSearch for KMP<'a> {
    type Config = &'a [u8];
    type State = Vec<usize>;

    fn build(config: &Self::Config) -> Self::State {
        build_lps(config)
    }

    fn find_bytes(config: &Self::Config, state: &Self::State, text: &[u8]) -> Option<usize> {
        kmp_search(text, config, state)
    }
}

fn build_lps(pattern: &[u8]) -> Vec<usize> {
    let m = pattern.len();
    let mut lps = vec![0; m];

    let mut len = 0;
    let mut i = 1;

    while i < m {
        if unsafe { pattern.get_unchecked(i) == pattern.get_unchecked(len) } {
            len += 1;
            lps[i] = len;
            i += 1;
        } else if len != 0 {
            len = lps[len - 1];
        } else {
            lps[i] = 0;
            i += 1;
        }
    }

    lps
}

fn kmp_search(text: &[u8], pattern: &[u8], lps: &[usize]) -> Option<usize> {
    let n = text.len();
    let m = pattern.len();

    if m == 0 {
        return Some(0);
    }
    if m > n {
        return None;
    }

    let mut i = 0;
    let mut j = 0;

    while i < n {
        let t = unsafe { *text.get_unchecked(i) };
        let p = unsafe { *pattern.get_unchecked(j) };
        if t == p {
            i += 1;
            j += 1;

            if j == m {
                return Some(i - j);
            }
        } else if j != 0 {
            j = lps[j - 1];
        } else {
            i += 1;
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmp_basic() {
        let hay = b"ababcabcabababd";
        let pat: &[u8] = b"ababd";
        assert_eq!(KMP::find(&pat, std::str::from_utf8(hay).unwrap()), Some(10));
    }

    #[test]
    fn test_kmp_not_found() {
        let hay = b"hello world";
        let pat: &[u8] = b"rust";
        assert_eq!(KMP::find(&pat, std::str::from_utf8(hay).unwrap()), None);
    }

    #[test]
    fn test_kmp_empty_pattern() {
        let hay = b"abc";
        let pat: &[u8] = b"";
        assert_eq!(KMP::find(&pat, std::str::from_utf8(hay).unwrap()), Some(0));
    }

    #[test]
    fn test_kmp_utf8() {
        let hay = "🌍hello🌍hello";
        let pat = "🌍hello";
        assert_eq!(KMP::find(&pat.as_bytes(), hay), Some(0));
    }
}
