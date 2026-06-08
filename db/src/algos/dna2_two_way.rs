use core::cmp::max;

use crate::like::{LiteralAlgorithm, RowLiteralSearch};
use crate::storage::dna2::{Dna2Column, Dna2Row, DnaBase};

use super::dna2::DNA_N;

#[derive(Debug, Clone, Copy, Default)]
pub struct Dna2TwoWay;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dna2TwoWayNeedle {
    symbols: Box<[u8]>,
}

impl Dna2TwoWayNeedle {
    #[inline]
    pub fn symbols(&self) -> &[u8] {
        &self.symbols
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Dna2TwoWayState {
    // crit = ell + 1, so crit == 0 corresponds to ell == -1.
    crit: usize,
    period: usize,
    is_periodic: bool,
    // Shift used in the non-periodic case.
    shift: usize,
}

impl LiteralAlgorithm for Dna2TwoWay {
    type Needle = Dna2TwoWayNeedle;
    type State = Dna2TwoWayState;

    const SUPPORTS_UNDERSCORE: bool = false;

    #[inline]
    fn compile_literal(src: &str) -> Option<Self::Needle> {
        let mut symbols = Vec::with_capacity(src.len());
        for &b in src.as_bytes() {
            if matches!(b, b'N' | b'n') {
                symbols.push(DNA_N);
            } else {
                symbols.push(DnaBase::from_ascii(b).ok()?.code());
            }
        }
        Some(Dna2TwoWayNeedle {
            symbols: symbols.into_boxed_slice(),
        })
    }

    #[inline]
    fn build_state(needle: &Self::Needle) -> Self::State {
        build_state(needle.symbols())
    }

    #[inline]
    fn literal_len(needle: &Self::Needle) -> u32 {
        needle.symbols().len() as u32
    }
}

impl<'db> RowLiteralSearch<Dna2Column<'db>> for Dna2TwoWay {
    #[inline]
    fn row_len<'r>(row: &Dna2Row<'r>) -> u32 {
        row.len_bases()
    }

    #[inline]
    fn matches_at<'r>(
        row: &Dna2Row<'r>,
        pos: u32,
        needle: &Self::Needle,
        _state: &Self::State,
    ) -> bool {
        dna2_exact_matches_at(row, pos, needle.symbols())
    }

    #[inline]
    fn find_from<'r>(
        row: &Dna2Row<'r>,
        from: u32,
        needle: &Self::Needle,
        state: &Self::State,
    ) -> Option<u32> {
        dna2_two_way_find(row, from, needle.symbols(), state)
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
fn build_state(pattern: &[u8]) -> Dna2TwoWayState {
    let m = pattern.len();

    if m == 0 {
        return Dna2TwoWayState {
            crit: 0,
            period: 1,
            is_periodic: true,
            shift: 1,
        };
    }
    if m == 1 {
        return Dna2TwoWayState {
            crit: 0,
            period: 1,
            is_periodic: false,
            shift: 1,
        };
    }

    let (ms1, p1) = maximal_suffix(pattern, false);
    let (ms2, p2) = maximal_suffix(pattern, true);

    let (ell, period) = if ms1 > ms2 { (ms1, p1) } else { (ms2, p2) };
    let crit = (ell + 1) as usize;

    let is_periodic =
        period < m && crit <= (m - period) && pattern[..crit] == pattern[period..period + crit];

    let shift = max(crit, m - crit) + 1;

    Dna2TwoWayState {
        crit,
        period,
        is_periodic,
        shift,
    }
}

#[inline]
pub fn dna2_exact_matches_at(row: &Dna2Row<'_>, pos: u32, pattern: &[u8]) -> bool {
    let len = pattern.len() as u32;
    let Some(end) = pos.checked_add(len) else {
        return false;
    };
    if end > row.len_bases() {
        return false;
    }
    if !row.has_n() {
        for (idx, &want) in pattern.iter().enumerate() {
            if want == DNA_N || row.base_code_at(pos + idx as u32) != want {
                return false;
            }
        }
    } else {
        for (idx, &want) in pattern.iter().enumerate() {
            let row_pos = pos + idx as u32;
            if want == DNA_N {
                if !row.is_n_at(row_pos) {
                    return false;
                }
            } else if row.is_n_at(row_pos) || row.base_code_at(row_pos) != want {
                return false;
            }
        }
    }

    true
}

pub fn dna2_two_way_find(
    row: &Dna2Row<'_>,
    from: u32,
    pattern: &[u8],
    state: &Dna2TwoWayState,
) -> Option<u32> {
    let n = row.len_bases() as usize;
    let m = pattern.len();
    let from = from as usize;

    if from > n {
        return None;
    }
    if m == 0 {
        return Some(from as u32);
    }
    if m > n.saturating_sub(from) {
        return None;
    }
    if row.has_n() {
        return dna2_exact_find_slow(row, from as u32, pattern);
    }
    if pattern_has_n(pattern) {
        return None;
    }
    if m == 1 {
        let needle = pattern[0];
        let mut pos = from;
        while pos < n {
            if row.base_code_at(pos as u32) == needle {
                return Some(pos as u32);
            }
            pos += 1;
        }
        return None;
    }
    if m == 2 {
        let a = pattern[0];
        let b = pattern[1];
        let mut pos = from;
        while pos + 1 < n {
            if row.base_code_at(pos as u32) == a && row.base_code_at((pos + 1) as u32) == b {
                return Some(pos as u32);
            }
            pos += 1;
        }
        return None;
    }

    let crit = state.crit;
    let pat = pattern.as_ptr();

    let last_start = n - m;
    let mut pos = from;

    unsafe {
        if state.is_periodic {
            let mut memory = 0usize;

            while pos <= last_start {
                let mut i = max(crit, memory);

                while i < m && *pat.add(i) == row.base_code_at((pos + i) as u32) {
                    i += 1;
                }

                if i >= m {
                    let mut i1 = crit;
                    while i1 > memory && *pat.add(i1 - 1) == row.base_code_at((pos + i1 - 1) as u32)
                    {
                        i1 -= 1;
                    }

                    if i1 <= memory {
                        return Some(pos as u32);
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
                let mut i = crit;

                while i < m && *pat.add(i) == row.base_code_at((pos + i) as u32) {
                    i += 1;
                }

                if i >= m {
                    let mut i1 = crit;
                    while i1 > 0 && *pat.add(i1 - 1) == row.base_code_at((pos + i1 - 1) as u32) {
                        i1 -= 1;
                    }

                    if i1 == 0 {
                        return Some(pos as u32);
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

#[inline]
fn pattern_has_n(pattern: &[u8]) -> bool {
    pattern.contains(&DNA_N)
}

fn dna2_exact_find_slow(row: &Dna2Row<'_>, from: u32, pattern: &[u8]) -> Option<u32> {
    let n = row.len_bases();
    let m = pattern.len() as u32;
    if from > n {
        return None;
    }
    if m == 0 {
        return Some(from);
    }
    if m > n.saturating_sub(from) {
        return None;
    }

    let last_start = n - m;
    let mut pos = from;
    while pos <= last_start {
        if dna2_exact_matches_at(row, pos, pattern) {
            return Some(pos);
        }
        if pos == last_start {
            break;
        }
        pos += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::like::{LikePattern, RowLiteralSearch};
    use crate::storage::Column;
    use crate::{DbBuilder, FullScan, RowId, execute_like};

    fn one_row(seq: &str) -> (crate::Db, crate::TableId) {
        let mut table = crate::Dna2TableBuilder::new("dna");
        table.push_str(seq).unwrap();
        let mut dbb = DbBuilder::new();
        let id = dbb.add_dna2_table(table).unwrap();
        (dbb.freeze(), id)
    }

    fn reference_find(row: &Dna2Row<'_>, from: u32, pattern: &[u8]) -> Option<u32> {
        dna2_exact_find_slow(row, from, pattern)
    }

    #[test]
    fn literal_search_matches_reference() {
        let cases = [
            ("", ""),
            ("ACGT", ""),
            ("", "A"),
            ("ACGT", "A"),
            ("ACGT", "CG"),
            ("ACGTACGT", "GTAC"),
            ("AAAAAA", "AAA"),
            ("AAAAAAAAAAAAATAAAAA", "AAAAT"),
            ("ACACACACACAC", "ACAC"),
            ("TTTTACGTAAAA", "ACGT"),
            ("ACNNNT", "AAA"),
            ("ACNNNT", "AC"),
            ("ACNNNT", "N"),
            ("ACNNNT", "NNN"),
            ("ACNNNT", "CN"),
            ("ACNNNT", "AN"),
            ("AAAA", "N"),
            ("NNNN", "AN"),
        ];

        for (text, pat) in cases {
            let (db, id) = one_row(text);
            let table = db.dna2_table(id).unwrap();
            let col = table.sequence();
            let row = col.row_view(0);
            let needle = Dna2TwoWay::compile_literal(pat).unwrap();
            let state = Dna2TwoWay::build_state(&needle);

            for from in 0..=(row.len_bases() + 1) {
                let got = Dna2TwoWay::find_from(&row, from, &needle, &state);
                let expect = reference_find(&row, from, needle.symbols());
                assert_eq!(got, expect, "text={text:?}, pat={pat:?}, from={from}");
            }

            for pos in 0..=(row.len_bases() + 1) {
                let got = Dna2TwoWay::matches_at(&row, pos, &needle, &state);
                let expect = dna2_exact_matches_at(&row, pos, needle.symbols());
                assert_eq!(got, expect, "text={text:?}, pat={pat:?}, pos={pos}");
            }
        }
    }

    #[test]
    fn underscore_is_lowered_by_like_compiler() {
        assert!(Dna2TwoWay::compile_literal("A_G").is_none());

        let mut reads = crate::Dna2TableBuilder::new("reads");
        reads.push_str("ACGT").unwrap();
        reads.push_str("AGGT").unwrap();
        reads.push_str("ATTT").unwrap();

        let mut dbb = DbBuilder::new();
        let id = dbb.add_dna2_table(reads).unwrap();
        let db = dbb.freeze();
        let table = db.dna2_table(id).unwrap();
        let col = table.sequence();

        let like = LikePattern::<Dna2TwoWay>::compile("A_G%").unwrap();
        let mut scan = FullScan::new(col.row_count(), 16);
        let mut matches = Vec::<RowId>::new();
        execute_like(&col, &mut scan, &like, &mut matches);
        assert_eq!(matches, vec![0, 1]);
    }

    #[test]
    fn n_is_exact_logical_symbol() {
        let mut reads = crate::Dna2TableBuilder::new("reads");
        reads.push_str("ACGT").unwrap();
        reads.push_str("ANNT").unwrap();
        reads.push_str("AAAT").unwrap();
        reads.push_str("NNNN").unwrap();

        let mut dbb = DbBuilder::new();
        let id = dbb.add_dna2_table(reads).unwrap();
        let db = dbb.freeze();
        let table = db.dna2_table(id).unwrap();
        let col = table.sequence();

        let like = LikePattern::<Dna2TwoWay>::compile("%N%").unwrap();
        let mut scan = FullScan::new(col.row_count(), 16);
        let mut matches = Vec::<RowId>::new();
        execute_like(&col, &mut scan, &like, &mut matches);
        assert_eq!(matches, vec![1, 3]);

        let like = LikePattern::<Dna2TwoWay>::compile("AAAT").unwrap();
        let mut scan = FullScan::new(col.row_count(), 16);
        let mut matches = Vec::<RowId>::new();
        execute_like(&col, &mut scan, &like, &mut matches);
        assert_eq!(matches, vec![2]);
    }
}
