use crate::StringSearch;

pub struct BM;

impl StringSearch for BM {
    type Config = ();
    type State = ();

    fn find_bytes(_state: Self::State, text: &[u8], pattern: &[u8]) -> Option<usize> {
        bm_find(text, pattern)
    }

    fn find_all_bytes(_state: Self::State, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        bm_find_all(text, pattern)
    }
}

/// Build the bad-character shift table for Boyer–Moore.
fn build_bad_char_table(pattern: &[u8]) -> [isize; 256] {
    let mut table = [-1isize; 256];
    for (i, &b) in pattern.iter().enumerate() {
        table[b as usize] = i as isize;
    }
    table
}

/// Build the good-suffix shift table for Boyer–Moore.
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
    for (i, value) in shift.iter_mut().enumerate().take(m + 1) {
        if *value == 0 {
            *value = j;
        }
        if i == j {
            j = border_pos[j];
        }
    }

    shift
}

/// Find the first occurrence of `pattern` in `text` using Boyer–Moore.
/// Returns Some(start_index) if found, None otherwise.
///
/// Operates on raw bytes; UTF-8 is fine but not required.
pub fn bm_find(text: &[u8], pattern: &[u8]) -> Option<usize> {
    let n = text.len();
    let m = pattern.len();

    if m == 0 {
        return Some(0);
    }
    if m > n {
        return None;
    }

    let bad_char = build_bad_char_table(pattern);
    let good_suffix = build_good_suffix_table(pattern);

    let mut i = 0usize; // index in text where the current pattern alignment starts

    while i <= n - m {
        let mut j  = (m - 1) as isize;

        while j >= 0 && pattern[j as usize] == text[i + j as usize] {
            j -= 1;
        }

        if j < 0 {
            // full match
            return Some(i);
        } else {
            let mismatch_index = j as usize;
            let bad_byte = text[i + mismatch_index];

            // Bad-character shift
            let last_occurrence = bad_char[bad_byte as usize]; // isize
            let bc_shift = mismatch_index as isize - last_occurrence;
            let bc_shift = if bc_shift > 0 { bc_shift as usize } else { 1 };

            // Good-suffix shift (note +1 indexing)
            let gs_shift = good_suffix[mismatch_index + 1];

            i += bc_shift.max(gs_shift);
        }
    }

    None
}

/// Find all (possibly overlapping) occurrences of `pattern` in `text`
/// using Boyer–Moore. Returns a vector of starting indices.
pub fn bm_find_all(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    let n = text.len();
    let m = pattern.len();

    if m == 0 {
        // Convention: match at every index (including at the end)
        return (0..=n).collect();
    }
    if m > n {
        return Vec::new();
    }

    let bad_char = build_bad_char_table(pattern);
    let good_suffix = build_good_suffix_table(pattern);

    let mut res = Vec::new();
    let mut i = 0usize;

    while i <= n - m {
        let mut j = (m - 1) as isize;

        while j >= 0 && pattern[j as usize] == text[i + j as usize] {
            j -= 1;
        }

        if j < 0 {
            // full match
            res.push(i);
            // Shift by the full-match good-suffix entry
            i += good_suffix[0];
        } else {
            let mismatch_index = j as usize;
            let bad_byte = text[i + mismatch_index];

            let last_occurrence = bad_char[bad_byte as usize];
            let bc_shift = mismatch_index as isize - last_occurrence;
            let bc_shift = if bc_shift > 0 { bc_shift as usize } else { 1 };

            let gs_shift = good_suffix[mismatch_index + 1];

            i += bc_shift.max(gs_shift);
        }
    }

    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm_basic() {
        let hay = b"ababcabcabababd";
        let pat = b"ababd";
        assert_eq!(bm_find(hay, pat), Some(10));
    }

    #[test]
    fn test_bm_not_found() {
        let hay = b"hello world";
        let pat = b"rust";
        assert_eq!(bm_find(hay, pat), None);
    }

    #[test]
    fn test_bm_empty_pattern() {
        let hay = b"abc";
        let pat: &[u8] = b"";
        assert_eq!(bm_find(hay, pat), Some(0));
        assert_eq!(bm_find_all(hay, pat), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_bm_find_all_overlapping() {
        let hay = b"aaaa";
        let pat = b"aa";
        assert_eq!(bm_find_all(hay, pat), vec![0, 1, 2]);
    }

    #[test]
    fn test_bm_find_all_cut() {
        let hay = b"aabaa";
        let pat = b"aa";
        assert_eq!(bm_find_all(hay, pat), vec![0, 3]);
    }

    #[test]
    fn test_bm_utf8() {
        let hay_s = "🌍hello🌍hello";
        let pat_s = "🌍hello";
        let hay = hay_s.as_bytes();
        let pat = pat_s.as_bytes();

        assert_eq!(pat_s.len(), 9);
        assert_eq!(hay_s.len(), 18);

        assert_eq!(bm_find(hay, pat), Some(0));
        assert_eq!(bm_find_all(hay, pat), vec![0, pat_s.len()]);
    }
}

