use crate::StringSearch;
use std::cmp::max;
use std::marker::PhantomData;

pub struct BM<'a>(PhantomData<&'a ()>);

pub struct BMState {
    bad_char: [isize; 256],
    good_suffix: Vec<usize>,
}

impl<'a> StringSearch for BM<'a> {
    type Config = &'a [u8];
    type State = BMState;

    fn build(pattern: &Self::Config) -> Self::State {
        BMState {
            bad_char: build_bad_char_table(pattern),
            good_suffix: build_good_suffix_table(pattern),
        }
    }

    fn find_bytes(pattern: &Self::Config, state: &Self::State, text: &[u8]) -> Option<usize> {
        bm_find(text, pattern, state)
    }
}

fn build_bad_char_table(pattern: &[u8]) -> [isize; 256] {
    let mut table = [-1isize; 256];
    for (i, &b) in pattern.iter().enumerate() {
        table[b as usize] = i as isize;
    }
    table
}

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

fn bm_find(text: &[u8], pattern: &[u8], state: &BMState) -> Option<usize> {
    let n = text.len();
    let m = pattern.len();

    if m == 0 {
        return Some(0);
    }
    if m > n {
        return None;
    }

    let mut i = 0usize;

    while i <= n - m {
        let mut j = (m - 1) as isize;

        while j >= 0 && pattern[j as usize] == text[i + j as usize] {
            j -= 1;
        }

        if j < 0 {
            return Some(i);
        } else {
            let mismatch_index = j as usize;
            let bad_byte = text[i + mismatch_index];

            // Bad-character shift
            let last_occurrence = state.bad_char[bad_byte as usize]; // isize
            let bc_shift = mismatch_index as isize - last_occurrence;
            let bc_shift = if bc_shift > 0 { bc_shift as usize } else { 1 };

            // Good-suffix shift (note +1 indexing)
            let gs_shift = state.good_suffix[mismatch_index + 1];

            i += max(bc_shift, gs_shift);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm_basic() {
        let hay = b"ababcabcabababd";
        let pat: &[u8] = b"ababd";

        // Convenience method
        assert_eq!(BM::find(&pat, std::str::from_utf8(hay).unwrap()), Some(10));
    }

    #[test]
    fn test_bm_not_found() {
        let hay = b"hello world";
        let pat: &[u8] = b"rust";
        assert_eq!(BM::find(&pat, std::str::from_utf8(hay).unwrap()), None);
    }

    #[test]
    fn test_bm_empty_pattern() {
        let hay = b"abc";
        let pat: &[u8] = b"";
        assert_eq!(BM::find(&pat, std::str::from_utf8(hay).unwrap()), Some(0));
    }

    #[test]
    fn test_bm_utf8() {
        let hay_s = "🌍hello🌍hello";
        let pat_s = "🌍hello";
        let pat_bytes = pat_s.as_bytes();

        let state = BM::build(&pat_bytes);
        assert_eq!(BM::find_str(&pat_bytes, &state, hay_s), Some(0));
    }
}
