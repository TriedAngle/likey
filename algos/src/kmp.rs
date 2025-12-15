use crate::StringSearch;

pub struct KMP;

impl StringSearch for KMP {
    type Config = ();
    type State = ();
    fn find_bytes(_state: Self::State, text: &[u8], pattern: &[u8]) -> Option<usize> {
        kmp_find(text, pattern)
    }

    fn find_all_bytes(_state: Self::State, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        kmp_find_all(text, pattern)
    }
}

/// Build the "longest proper prefix which is also suffix" (LPS) table
fn build_lps(pattern: &[u8]) -> Vec<usize> {
    let m = pattern.len();
    let mut lps = vec![0; m];

    let mut len = 0;
    let mut i = 1;

    while i < m {
        if pattern[i] == pattern[len] {
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

pub fn kmp_find(text: &[u8], pattern: &[u8]) -> Option<usize> {
    let n = text.len();
    let m = pattern.len();

    if m == 0 {
        return Some(0); // convention: empty pattern matches at 0
    }

    if m > n {
        return None;
    }

    let lps = build_lps(pattern);

    let mut i = 0;
    let mut j = 0;

    while i < n {
        let t = unsafe { *text.get_unchecked(i) };
        let p = unsafe { *pattern.get_unchecked(j) };
        if t == p {
            i += 1;
            j += 1;

            if j == m {
                // full match ending at i-1
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

pub fn kmp_find_all(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    let n = text.len();
    let m = pattern.len();

    if m == 0 {
        // Convention: match at every index, or just return empty.
        // Here we choose "every index including n":
        return (0..=n).collect();
    }
    if m > n {
        return Vec::new();
    }

    let lps = build_lps(pattern);
    let mut result = Vec::new();

    let mut i = 0usize; // index in haystack
    let mut j = 0usize; // index in needle

    while i < n {
        let t = unsafe { *text.get_unchecked(i) };
        let p = unsafe { *pattern.get_unchecked(j) };
        if t == p {
            i += 1;
            j += 1;

            if j == m {
                result.push(i - j);
                j = lps[j - 1];
            }
        } else if j != 0 {
            j = lps[j - 1];
        } else {
            i += 1;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmp_basic() {
        let hay = b"ababcabcabababd";
        let pat = b"ababd";
        assert_eq!(kmp_find(hay, pat), Some(10));
    }

    #[test]
    fn test_kmp_not_found() {
        let hay = b"hello world";
        let pat = b"rust";
        assert_eq!(kmp_find(hay, pat), None);
    }

    #[test]
    fn test_kmp_empty_pattern() {
        let hay = b"abc";
        let pat: &[u8] = b"";
        assert_eq!(kmp_find(hay, pat), Some(0));
    }

    #[test]
    fn test_kmp_find_all_overlapping() {
        let hay = b"aaaa";
        let pat = b"aa";
        assert_eq!(kmp_find_all(hay, pat), vec![0, 1, 2]);
    }

    #[test]
    fn test_kmp_utf8() {
        let hay = "🌍hello🌍hello".as_bytes();
        let pat = "🌍hello".as_bytes();

        assert_eq!("🌍".len(), 4);
        assert_eq!("hello".len(), 5);
        assert_eq!("🌍hello".len(), 9);
        assert_eq!("🌍hello🌍hello".len(), 18);

        assert_eq!(kmp_find(hay, pat), Some(0));
        assert_eq!(kmp_find_all(hay, pat), vec![0, "🌍hello".len()]);
    }
}
