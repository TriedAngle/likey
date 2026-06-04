//! Generic trigram index over logical `u8` symbols.
//!
//! This index intentionally does not specialize by storage type. It asks the
//! column for its logical symbol stream and stores 24-bit trigram keys in a
//! sparse hash map. DNA2 therefore uses the same default index as UTF-8 and FSST;
//! the old fixed-64 DNA2 layout remains available manually in `dna2_trigram`.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use crate::RowId;
use crate::like::{LikePattern, LiteralAlgorithm};
use crate::query::{CandidateBatch, CandidateProvider, CandidateScratch};
use crate::storage::Column;

#[inline(always)]
pub fn trigram_key(a: u8, b: u8, c: u8) -> u32 {
    ((a as u32) << 16) | ((b as u32) << 8) | (c as u32)
}

pub fn trigram_keys(symbols: &[u8]) -> Vec<u32> {
    if symbols.len() < 3 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(symbols.len() - 2);
    for window in symbols.windows(3) {
        out.push(trigram_key(window[0], window[1], window[2]));
    }
    out
}

#[derive(Clone)]
pub struct TrigramIndex<C>
where
    C: Column<Symbol = u8>,
{
    row_count: RowId,
    postings: HashMap<u32, Vec<RowId>>,
    _marker: PhantomData<fn(&C)>,
}

impl<C> std::fmt::Debug for TrigramIndex<C>
where
    C: Column<Symbol = u8>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrigramIndex")
            .field("row_count", &self.row_count)
            .field("keys", &self.postings.len())
            .finish()
    }
}

impl<C> TrigramIndex<C>
where
    C: Column<Symbol = u8>,
{
    pub fn build(column: &C) -> Self {
        let row_count = column.row_count();
        let mut postings = HashMap::<u32, Vec<RowId>>::new();

        for row in 0..row_count {
            for_each_unique_row_trigram(column, row, |key| {
                postings.entry(key).or_default().push(row);
            });
        }

        Self {
            row_count,
            postings,
            _marker: PhantomData,
        }
    }

    pub fn row_count(&self) -> RowId {
        self.row_count
    }

    pub fn postings_for_key(&self, key: u32) -> Option<&[RowId]> {
        self.postings.get(&key).map(Vec::as_slice)
    }

    pub fn postings_for_gram(&self, gram: [u8; 3]) -> Option<&[RowId]> {
        self.postings_for_key(trigram_key(gram[0], gram[1], gram[2]))
    }

    /// Return candidate rows for a literal by intersecting all of its trigram
    /// postings. `None` means the literal is too short.
    pub fn search_literal(&self, literal: &[u8]) -> Option<Vec<RowId>> {
        let mut grams = trigram_keys(literal);
        if grams.is_empty() {
            return None;
        }

        grams.sort_unstable();
        grams.dedup();

        let mut lists = Vec::<&[RowId]>::with_capacity(grams.len());
        for key in grams {
            let Some(list) = self.postings_for_key(key) else {
                return Some(Vec::new());
            };
            lists.push(list);
        }

        lists.sort_by_key(|list| list.len());
        let mut result = lists[0].to_vec();
        for list in lists.into_iter().skip(1) {
            intersect_sorted_in_place(&mut result, list);
            if result.is_empty() {
                break;
            }
        }

        Some(result)
    }

    pub fn probe_literal(&self, literal: &[u8], batch_rows: usize) -> Option<TrigramProbe> {
        let rows = self.search_literal(literal)?;
        Some(TrigramProbe::new(rows, batch_rows))
    }

    pub fn probe(&self, gram: [u8; 3]) -> TrigramProbe {
        self.probe_with_batch(gram, 4096)
    }

    pub fn probe_with_batch(&self, gram: [u8; 3], batch_rows: usize) -> TrigramProbe {
        self.probe_literal(&gram, batch_rows)
            .unwrap_or_else(|| TrigramProbe::new(Vec::new(), batch_rows))
    }

    pub fn probe_longest_like_literal<A>(
        &self,
        pattern: &LikePattern<A>,
        batch_rows: usize,
    ) -> Option<TrigramProbe>
    where
        A: LiteralAlgorithm,
    {
        let literal = pattern.longest_indexable_literal()?;
        self.probe_literal(literal, batch_rows)
    }
}

fn for_each_unique_row_trigram<C, F>(column: &C, row: RowId, mut f: F)
where
    C: Column<Symbol = u8>,
    F: FnMut(u32),
{
    let symbols = column.symbols(row).collect::<Vec<_>>();
    if symbols.len() < 3 {
        return;
    }

    let mut seen = HashSet::<u32>::new();
    for window in symbols.windows(3) {
        let key = trigram_key(window[0], window[1], window[2]);
        if seen.insert(key) {
            f(key);
        }
    }
}

pub(crate) fn intersect_sorted_in_place(left: &mut Vec<RowId>, right: &[RowId]) {
    let mut out = 0usize;
    let mut i = 0usize;
    let mut j = 0usize;

    while i < left.len() && j < right.len() {
        let a = left[i];
        let b = right[j];
        if a == b {
            left[out] = a;
            out += 1;
            i += 1;
            j += 1;
        } else if a < b {
            i += 1;
        } else {
            j += 1;
        }
    }

    left.truncate(out);
}

#[derive(Debug, Clone)]
pub struct TrigramProbe {
    rows: Vec<RowId>,
    cursor: usize,
    batch_rows: usize,
}

impl TrigramProbe {
    pub fn new(rows: Vec<RowId>, batch_rows: usize) -> Self {
        Self {
            rows,
            cursor: 0,
            batch_rows: batch_rows.max(1),
        }
    }

    pub fn rows(&self) -> &[RowId] {
        &self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

impl CandidateProvider for TrigramProbe {
    fn reset(&mut self) {
        self.cursor = 0;
    }

    fn next_batch<'a>(
        &'a mut self,
        _scratch: &'a mut CandidateScratch,
    ) -> Option<CandidateBatch<'a>> {
        if self.cursor >= self.rows.len() {
            return None;
        }
        let start = self.cursor;
        let end = (start + self.batch_rows).min(self.rows.len());
        self.cursor = end;
        Some(CandidateBatch::SortedRows(&self.rows[start..end]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::DbBuilder;
    use crate::storage::dna2::Dna2TableBuilder;
    use crate::storage::utf8::Utf8TableBuilder;

    #[test]
    fn posting_trigram_intersects_all_utf8_grams() {
        let mut docs = Utf8TableBuilder::new("docs");
        for row in [
            "apple",
            "applet",
            "pineapple",
            "application",
            "banana",
            "bandana",
        ] {
            docs.push_str(row);
        }

        let mut dbb = DbBuilder::new();
        let id = dbb.add_utf8_table(docs).unwrap();
        let db = dbb.freeze();
        let table = db.utf8_table(id).unwrap();
        let col = table.text();

        let idx = TrigramIndex::build(&col);
        assert_eq!(idx.search_literal(b"appl").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(idx.search_literal(b"ana").unwrap(), vec![4, 5]);
        assert_eq!(idx.search_literal(b"pine").unwrap(), vec![2]);
        assert!(idx.search_literal(b"an").is_none());
    }

    #[test]
    fn generic_trigram_indexes_dna2_symbols() {
        let mut reads = Dna2TableBuilder::new("reads");
        reads.push_str("ACGTACGT").unwrap();
        reads.push_str("TTTACGTT").unwrap();
        reads.push_str("GGGGGGGG").unwrap();
        reads.push_str("AACGAAAA").unwrap();

        let mut dbb = DbBuilder::new();
        let id = dbb.add_dna2_table(reads).unwrap();
        let db = dbb.freeze();
        let table = db.dna2_table(id).unwrap();
        let col = table.sequence();

        let idx = TrigramIndex::build(&col);
        assert_eq!(idx.search_literal(&[0, 1, 2]).unwrap(), vec![0, 1, 3]);
        assert_eq!(idx.search_literal(&[2, 2, 2]).unwrap(), vec![2]);
        assert!(idx.search_literal(&[0, 1]).is_none());
        assert_eq!(idx.search_literal(&[0, 9, 2]).unwrap(), Vec::<RowId>::new());
    }
}
