//! Generic q-gram index over row index bytes.
//!
//! The q value is intentionally fixed at compile time for now. With `Q = 15`,
//! keys are packed exactly into the low 120 bits of a `u128`, so lookups are
//! collision-free without storing the source bytes for each gram.

use std::marker::PhantomData;

use crate::like::{LikePattern, LiteralAlgorithm};
use crate::query::{CandidateBatch, CandidateProvider};
use crate::storage::Column;
use crate::{BuildIndex, RowId};

use super::trigram::intersect_sorted_in_place;

pub const QGRAM_Q: usize = 15;

const QGRAM_KEY_BITS: usize = QGRAM_Q * 8;
const QGRAM_KEY_MASK: u128 = (1u128 << QGRAM_KEY_BITS) - 1;
const DEFAULT_ROW_POSTING_DIVISOR: usize = 4;
const DEFAULT_MIN_BROAD_POSTING: usize = 1024;

#[inline(always)]
pub fn qgram_key(gram: &[u8]) -> Option<u128> {
    (gram.len() == QGRAM_Q).then(|| pack_qgram_key(gram))
}

pub fn qgram_keys(symbols: &[u8]) -> Vec<u128> {
    if symbols.len() < QGRAM_Q {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(symbols.len() - QGRAM_Q + 1);
    let mut key = pack_qgram_key(&symbols[..QGRAM_Q]);
    out.push(key);

    for &byte in &symbols[QGRAM_Q..] {
        key = roll_qgram_key(key, byte);
        out.push(key);
    }

    out
}

#[derive(Clone)]
pub struct QgramIndex<C>
where
    C: Column<Symbol = u8>,
{
    row_count: RowId,
    keys: Vec<u128>,
    offsets: Vec<usize>,
    rows: Vec<RowId>,
    _marker: PhantomData<fn(&C)>,
}

#[derive(Debug, Clone)]
pub enum QgramProbeOutcome {
    Probe(QgramProbe),
    TooBroad,
}

impl<C> std::fmt::Debug for QgramIndex<C>
where
    C: Column<Symbol = u8>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QgramIndex")
            .field("q", &QGRAM_Q)
            .field("row_count", &self.row_count)
            .field("keys", &self.keys.len())
            .field("postings", &self.rows.len())
            .finish()
    }
}

impl<C> BuildIndex<C> for QgramIndex<C>
where
    C: Column<Symbol = u8>,
{
    fn build(column: &C) -> Self {
        Self::build(column)
    }
}

impl<C> QgramIndex<C>
where
    C: Column<Symbol = u8>,
{
    pub fn build(column: &C) -> Self {
        let row_count = column.row_count();
        let mut entries = Vec::<(u128, RowId)>::new();

        for row in 0..row_count {
            for_each_unique_row_qgram(column, row, |key| entries.push((key, row)));
        }

        entries.sort_unstable();
        entries.dedup();

        let mut keys = Vec::new();
        let mut offsets = Vec::new();
        let mut rows = Vec::with_capacity(entries.len());
        let mut idx = 0usize;

        while idx < entries.len() {
            let key = entries[idx].0;
            keys.push(key);
            offsets.push(rows.len());

            while idx < entries.len() && entries[idx].0 == key {
                rows.push(entries[idx].1);
                idx += 1;
            }
        }

        offsets.push(rows.len());

        Self {
            row_count,
            keys,
            offsets,
            rows,
            _marker: PhantomData,
        }
    }

    pub fn row_count(&self) -> RowId {
        self.row_count
    }

    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    pub fn posting_count(&self) -> usize {
        self.rows.len()
    }

    pub fn postings_for_key(&self, key: u128) -> Option<&[RowId]> {
        let idx = self.keys.binary_search(&key).ok()?;
        Some(&self.rows[self.offsets[idx]..self.offsets[idx + 1]])
    }

    pub fn postings_for_gram(&self, gram: &[u8]) -> Option<&[RowId]> {
        self.postings_for_key(qgram_key(gram)?)
    }

    /// Return candidate rows for a literal by intersecting all of its q-gram
    /// postings. `None` means the literal is shorter than `QGRAM_Q`.
    pub fn search_literal(&self, literal: &[u8]) -> Option<Vec<RowId>> {
        let lists = self.posting_lists_for_literal(literal)?;
        Some(intersect_posting_lists(lists))
    }

    pub fn search_literal_selective(&self, literal: &[u8]) -> Option<QgramProbeOutcome> {
        let lists = self.posting_lists_for_literal(literal)?;

        if lists
            .first()
            .is_some_and(|list| list.len() > self.broad_posting_limit())
        {
            return Some(QgramProbeOutcome::TooBroad);
        }

        let result = intersect_posting_lists(lists);

        if result.len() > self.broad_posting_limit() {
            return Some(QgramProbeOutcome::TooBroad);
        }

        Some(QgramProbeOutcome::Probe(QgramProbe::new(result, 1)))
    }

    pub fn probe_literal(&self, literal: &[u8], batch_rows: usize) -> Option<QgramProbe> {
        let rows = self.search_literal(literal)?;
        Some(QgramProbe::new(rows, batch_rows))
    }

    pub fn probe_literal_selective(
        &self,
        literal: &[u8],
        batch_rows: usize,
    ) -> Option<QgramProbeOutcome> {
        match self.search_literal_selective(literal)? {
            QgramProbeOutcome::Probe(probe) => Some(QgramProbeOutcome::Probe(QgramProbe::new(
                probe.into_rows(),
                batch_rows,
            ))),
            QgramProbeOutcome::TooBroad => Some(QgramProbeOutcome::TooBroad),
        }
    }

    pub fn probe_longest_like_literal<A>(
        &self,
        pattern: &LikePattern<A>,
        batch_rows: usize,
    ) -> Option<QgramProbe>
    where
        A: LiteralAlgorithm,
    {
        let literal = pattern.longest_fixed_source_fragment()?;
        self.probe_literal(literal.as_bytes(), batch_rows)
    }

    pub fn probe_selective_longest_like_literal<A>(
        &self,
        pattern: &LikePattern<A>,
        batch_rows: usize,
    ) -> Option<QgramProbeOutcome>
    where
        A: LiteralAlgorithm,
    {
        let literal = pattern.longest_fixed_source_fragment()?;
        self.probe_literal_selective(literal.as_bytes(), batch_rows)
    }

    pub fn broad_posting_limit(&self) -> usize {
        let row_count = usize::try_from(self.row_count).unwrap_or(usize::MAX);
        (row_count / DEFAULT_ROW_POSTING_DIVISOR)
            .max(DEFAULT_MIN_BROAD_POSTING)
            .max(1)
    }

    fn posting_lists_for_literal(&self, literal: &[u8]) -> Option<Vec<&[RowId]>> {
        let mut grams = qgram_keys(literal);
        if grams.is_empty() {
            return None;
        }

        grams.sort_unstable();
        grams.dedup();

        let mut lists = Vec::<&[RowId]>::with_capacity(grams.len());
        for key in grams {
            let Some(list) = self.postings_for_key(key) else {
                return Some(vec![&[]]);
            };
            lists.push(list);
        }

        lists.sort_by_key(|list| list.len());
        Some(lists)
    }
}

#[inline(always)]
fn pack_qgram_key(gram: &[u8]) -> u128 {
    let mut key = 0u128;
    for &byte in gram {
        key = (key << 8) | u128::from(byte);
    }
    key
}

#[inline(always)]
fn roll_qgram_key(key: u128, byte: u8) -> u128 {
    ((key << 8) & QGRAM_KEY_MASK) | u128::from(byte)
}

fn for_each_unique_row_qgram<C, F>(column: &C, row: RowId, mut f: F)
where
    C: Column<Symbol = u8>,
    F: FnMut(u128),
{
    let symbols = column.symbols(row).collect::<Vec<_>>();
    let mut keys = qgram_keys(&symbols);
    if keys.is_empty() {
        return;
    }

    keys.sort_unstable();
    keys.dedup();

    for key in keys {
        f(key);
    }
}

fn intersect_posting_lists(lists: Vec<&[RowId]>) -> Vec<RowId> {
    let mut result = lists[0].to_vec();
    for list in lists.into_iter().skip(1) {
        intersect_sorted_in_place(&mut result, list);
        if result.is_empty() {
            break;
        }
    }

    result
}

#[derive(Debug, Clone)]
pub struct QgramProbe {
    rows: Vec<RowId>,
    cursor: usize,
    batch_rows: usize,
}

impl QgramProbe {
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

    pub fn into_rows(self) -> Vec<RowId> {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

impl CandidateProvider for QgramProbe {
    fn reset(&mut self) {
        self.cursor = 0;
    }

    fn next_batch(&mut self) -> Option<CandidateBatch<'_>> {
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
    fn qgram_keys_are_exact_15_byte_windows() {
        let keys = qgram_keys(b"abcdefghijklmnop");

        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0], qgram_key(b"abcdefghijklmno").unwrap());
        assert_eq!(keys[1], qgram_key(b"bcdefghijklmnop").unwrap());
        assert_eq!(qgram_key(b"abcdefghijklmn"), None);
        assert!(qgram_keys(b"abcdefghijklmn").is_empty());
    }

    #[test]
    fn posting_qgram_intersects_all_utf8_grams() {
        let mut docs = Utf8TableBuilder::new("docs");
        for row in [
            "abcdefghijklmnop",
            "xxabcdefghijklmnozz",
            "bcdefghijklmnop",
            "abcdefghijklmn",
            "missing missing missing",
        ] {
            docs.push_str(row);
        }

        let mut dbb = DbBuilder::new();
        let id = dbb.add_utf8_table(docs).unwrap();
        let db = dbb.freeze();
        let table = db.utf8_table(id).unwrap();
        let col = table.text();

        let idx = QgramIndex::build(&col);
        assert_eq!(idx.search_literal(b"abcdefghijklmno").unwrap(), vec![0, 1]);
        assert_eq!(idx.search_literal(b"bcdefghijklmnop").unwrap(), vec![0, 2]);
        assert_eq!(idx.search_literal(b"abcdefghijklmnop").unwrap(), vec![0]);
        assert_eq!(
            idx.search_literal(b"zzzzzzzzzzzzzzz").unwrap(),
            Vec::<RowId>::new()
        );
        assert!(idx.search_literal(b"abcdefghijklmn").is_none());
    }

    #[test]
    fn selective_probe_rejects_broad_qgram_postings() {
        let mut docs = Utf8TableBuilder::new("docs");
        for _ in 0..2000 {
            docs.push_str("aaaaaaaaaaaaaaaa");
        }

        let mut dbb = DbBuilder::new();
        let id = dbb.add_utf8_table(docs).unwrap();
        let db = dbb.freeze();
        let table = db.utf8_table(id).unwrap();
        let col = table.text();

        let idx = QgramIndex::build(&col);
        assert!(matches!(
            idx.probe_literal_selective(b"aaaaaaaaaaaaaaa", 64),
            Some(QgramProbeOutcome::TooBroad)
        ));
        assert!(matches!(
            idx.probe_literal_selective(b"zzzzzzzzzzzzzzz", 64),
            Some(QgramProbeOutcome::Probe(probe)) if probe.is_empty()
        ));
    }

    #[test]
    fn generic_qgram_indexes_dna2_ascii_bytes() {
        let mut reads = Dna2TableBuilder::new("reads");
        reads.push_str("ACGTACGTACGTACG").unwrap();
        reads.push_str("TTACGTACGTACGTACG").unwrap();
        reads.push_str("GGGGGGGGGGGGGGG").unwrap();
        reads.push_str("ACGTACGTACGTAC").unwrap();
        reads.push_str("AANTACGTACGTACG").unwrap();

        let mut dbb = DbBuilder::new();
        let id = dbb.add_dna2_table(reads).unwrap();
        let db = dbb.freeze();
        let table = db.dna2_table(id).unwrap();
        let col = table.sequence();

        let idx = QgramIndex::build(&col);
        assert_eq!(idx.search_literal(b"ACGTACGTACGTACG").unwrap(), vec![0, 1]);
        assert_eq!(idx.search_literal(b"GGGGGGGGGGGGGGG").unwrap(), vec![2]);
        assert_eq!(idx.search_literal(b"AANTACGTACGTACG").unwrap(), vec![4]);
        assert!(idx.search_literal(b"ACGTACGTACGTAC").is_none());
        assert_eq!(
            idx.search_literal(b"A?NTACGTACGTACG").unwrap(),
            Vec::<RowId>::new()
        );
    }
}
