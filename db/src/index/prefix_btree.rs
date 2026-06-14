//! Prefix B-tree index over full logical row values.
//!
//! The index stores each complete row value as a B-tree key. A LIKE pattern with
//! a fixed leading prefix can probe the lexicographic range
//! `[prefix, prefix_successor(prefix))` and let the normal verifier apply the
//! remaining LIKE semantics.

use std::collections::BTreeMap;
use std::marker::PhantomData;

use crate::like::{LikePattern, LiteralAlgorithm};
use crate::query::{CandidateBatch, CandidateProvider};
use crate::storage::Column;
use crate::{BuildIndex, RowId};

#[derive(Clone)]
pub struct PrefixBtreeIndex<C>
where
    C: Column<Symbol = u8>,
{
    row_count: RowId,
    rows_by_value: BTreeMap<Box<[u8]>, Vec<RowId>>,
    _marker: PhantomData<fn(&C)>,
}

impl<C> std::fmt::Debug for PrefixBtreeIndex<C>
where
    C: Column<Symbol = u8>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixBtreeIndex")
            .field("row_count", &self.row_count)
            .field("distinct_values", &self.rows_by_value.len())
            .finish()
    }
}

impl<C> BuildIndex<C> for PrefixBtreeIndex<C>
where
    C: Column<Symbol = u8>,
{
    fn build(column: &C) -> Self {
        Self::build(column)
    }
}

impl<C> PrefixBtreeIndex<C>
where
    C: Column<Symbol = u8>,
{
    pub fn build(column: &C) -> Self {
        let row_count = column.row_count();
        let mut rows_by_value = BTreeMap::<Box<[u8]>, Vec<RowId>>::new();

        for row in 0..row_count {
            let key = column.symbols(row).collect::<Vec<_>>().into_boxed_slice();
            rows_by_value.entry(key).or_default().push(row);
        }

        Self {
            row_count,
            rows_by_value,
            _marker: PhantomData,
        }
    }

    pub fn row_count(&self) -> RowId {
        self.row_count
    }

    pub fn distinct_value_count(&self) -> usize {
        self.rows_by_value.len()
    }

    /// Approximate retained in-memory size of the prefix B-tree in bytes.
    ///
    /// This counts the index struct, stored keys, posting vectors, and a simple
    /// per-entry payload estimate. It does not include exact B-tree node or
    /// allocator overhead.
    pub fn estimated_size_bytes(&self) -> usize {
        self.rows_by_value
            .iter()
            .fold(std::mem::size_of::<Self>(), |bytes, (key, rows)| {
                bytes
                    .saturating_add(std::mem::size_of::<(Box<[u8]>, Vec<RowId>)>())
                    .saturating_add(key.len())
                    .saturating_add(rows.capacity().saturating_mul(std::mem::size_of::<RowId>()))
            })
    }

    /// Return sorted candidate rows whose full value starts with `prefix`.
    ///
    /// `None` means the prefix is empty and therefore not selective. Callers
    /// should fall back to a full scan or another index in that case.
    pub fn search_prefix(&self, prefix: &[u8]) -> Option<Vec<RowId>> {
        if prefix.is_empty() {
            return None;
        }

        let lower = prefix.to_vec().into_boxed_slice();
        let mut rows = Vec::new();

        if let Some(upper) = prefix_successor(prefix) {
            for (_value, row_ids) in self.rows_by_value.range(lower..upper) {
                rows.extend_from_slice(row_ids);
            }
        } else {
            for (_value, row_ids) in self.rows_by_value.range(lower..) {
                rows.extend_from_slice(row_ids);
            }
        }

        rows.sort_unstable();
        rows.dedup();
        Some(rows)
    }

    /// Return sorted rows whose full value exactly equals `value`.
    pub fn search_exact(&self, value: &[u8]) -> Vec<RowId> {
        self.rows_by_value.get(value).cloned().unwrap_or_default()
    }

    pub fn probe_exact(&self, value: &[u8], batch_rows: usize) -> PrefixBtreeProbe {
        PrefixBtreeProbe::new(self.search_exact(value), batch_rows)
    }

    pub fn probe_prefix(&self, prefix: &[u8], batch_rows: usize) -> Option<PrefixBtreeProbe> {
        let rows = self.search_prefix(prefix)?;
        Some(PrefixBtreeProbe::new(rows, batch_rows))
    }

    pub fn probe_like_prefix<A>(
        &self,
        pattern: &LikePattern<A>,
        batch_rows: usize,
    ) -> Option<PrefixBtreeProbe>
    where
        A: LiteralAlgorithm,
    {
        if let Some(exact_source) = pattern.exact_source() {
            return Some(self.probe_exact(exact_source.as_bytes(), batch_rows));
        }

        let prefix_source = pattern.leading_fixed_source_prefix()?;
        self.probe_prefix(prefix_source.as_bytes(), batch_rows)
    }
}

fn prefix_successor(prefix: &[u8]) -> Option<Box<[u8]>> {
    let mut end = prefix.to_vec();
    for idx in (0..end.len()).rev() {
        if end[idx] != u8::MAX {
            end[idx] += 1;
            end.truncate(idx + 1);
            return Some(end.into_boxed_slice());
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct PrefixBtreeProbe {
    rows: Vec<RowId>,
    cursor: usize,
    batch_rows: usize,
}

impl PrefixBtreeProbe {
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

impl CandidateProvider for PrefixBtreeProbe {
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
    use crate::query::{FullScan, execute_like};
    use crate::storage::dna2::Dna2TableBuilder;
    use crate::storage::utf8::Utf8TableBuilder;
    use crate::{DbBuilder, Dna2, LikePattern, NaiveWildcard, StdSearch};

    #[test]
    fn prefix_btree_searches_utf8_prefixes() {
        let mut docs = Utf8TableBuilder::new("docs");
        for row in [
            "apple",
            "applet",
            "pineapple",
            "application",
            "banana",
            "app",
            "",
        ] {
            docs.push_str(row);
        }

        let mut dbb = DbBuilder::new();
        let id = dbb.add_utf8_table(docs).unwrap();
        let db = dbb.freeze();
        let table = db.utf8_table(id).unwrap();
        let col = table.text();

        let idx = PrefixBtreeIndex::build(&col);
        assert_eq!(idx.search_prefix(b"app").unwrap(), vec![0, 1, 3, 5]);
        assert_eq!(idx.search_prefix(b"apple").unwrap(), vec![0, 1]);
        assert_eq!(idx.search_prefix(b"missing").unwrap(), Vec::<RowId>::new());
        assert!(idx.search_prefix(b"").is_none());
    }

    #[test]
    fn prefix_btree_candidates_verify_larger_like_pattern() {
        let mut docs = Utf8TableBuilder::new("docs");
        for row in ["application", "apple", "app relation", "z application"] {
            docs.push_str(row);
        }

        let mut dbb = DbBuilder::new();
        let id = dbb.add_utf8_table(docs).unwrap();
        let db = dbb.freeze();
        let table = db.utf8_table(id).unwrap();
        let col = table.text();

        let like = LikePattern::<StdSearch>::compile("app%tion").unwrap();
        let prefix = like.leading_fixed_source_prefix().unwrap().as_bytes();
        assert_eq!(prefix, b"app");

        let idx = PrefixBtreeIndex::build(&col);
        let mut probe = idx.probe_prefix(prefix, 2).unwrap();
        let mut indexed = Vec::<RowId>::new();
        execute_like(&col, &mut probe, &like, &mut indexed);

        let mut scan = FullScan::new(col.row_count(), 2);
        let mut expected = Vec::<RowId>::new();
        execute_like(&col, &mut scan, &like, &mut expected);

        assert_eq!(indexed, expected);
        assert_eq!(indexed, vec![0, 2]);
    }

    #[test]
    fn prefix_btree_exact_probe_does_not_return_longer_prefix_rows() {
        let mut docs = Utf8TableBuilder::new("docs");
        for row in ["app", "apple", "applet", "application", "banana"] {
            docs.push_str(row);
        }

        let mut dbb = DbBuilder::new();
        let id = dbb.add_utf8_table(docs).unwrap();
        let db = dbb.freeze();
        let table = db.utf8_table(id).unwrap();
        let col = table.text();
        let idx = PrefixBtreeIndex::build(&col);
        let like = LikePattern::<StdSearch>::compile("app").unwrap();

        let probe = idx.probe_like_prefix(&like, 2).unwrap();
        assert_eq!(probe.rows(), &[0]);
    }

    #[test]
    fn prefix_btree_exact_probe_preserves_algorithm_level_underscore() {
        let mut docs = Utf8TableBuilder::new("docs");
        for row in ["apple", "app_e", "applet", "banana"] {
            docs.push_str(row);
        }

        let mut dbb = DbBuilder::new();
        let id = dbb.add_utf8_table(docs).unwrap();
        let db = dbb.freeze();
        let table = db.utf8_table(id).unwrap();
        let col = table.text();
        let idx = PrefixBtreeIndex::build(&col);
        let like = LikePattern::<NaiveWildcard>::compile("app_e").unwrap();

        let mut indexed_probe = idx.probe_like_prefix(&like, 2).unwrap();
        let mut indexed = Vec::<RowId>::new();
        execute_like(&col, &mut indexed_probe, &like, &mut indexed);

        let mut scan = FullScan::new(col.row_count(), 2);
        let mut expected = Vec::<RowId>::new();
        execute_like(&col, &mut scan, &like, &mut expected);

        assert_eq!(indexed, expected);
        assert_eq!(indexed, vec![0, 1]);
    }

    #[test]
    fn leading_prefix_stops_before_algorithm_level_underscore() {
        let like = LikePattern::<NaiveWildcard>::compile("app_e%").unwrap();
        let prefix = like.leading_fixed_source_prefix().unwrap();
        assert_eq!(prefix, "app");
        assert_eq!(prefix.as_bytes(), b"app");
    }

    #[test]
    fn leading_prefix_uses_logical_dna2_symbols() {
        let mut reads = Dna2TableBuilder::new("reads");
        reads.push_str("ACGT").unwrap();
        reads.push_str("ACGA").unwrap();
        reads.push_str("TCGT").unwrap();

        let mut dbb = DbBuilder::new();
        let id = dbb.add_dna2_table(reads).unwrap();
        let db = dbb.freeze();
        let table = db.dna2_table(id).unwrap();
        let col = table.sequence();

        let like = LikePattern::<Dna2>::compile("AC_T%").unwrap();
        let prefix = like.leading_fixed_source_prefix().unwrap().as_bytes();
        assert_eq!(prefix, b"AC");

        let idx = PrefixBtreeIndex::build(&col);
        assert_eq!(idx.search_prefix(prefix).unwrap(), vec![0, 1]);
    }
}
