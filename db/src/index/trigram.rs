//! Trigram indexes over logical `u8` symbols.
//!
//! The public [`TrigramIndex`] is generic over the searched column. It chooses a
//! storage-specific [`TrigramDomain`] through [`HasTrigramIndex`]:
//!
//! - UTF-8 byte columns use 24-bit byte trigrams stored in a `HashMap`.
//! - FSST columns decode rows during index construction and use the same byte
//!   trigram domain as UTF-8 columns.
//! - DNA2 columns use 6-bit DNA trigrams stored in a fixed `[Vec<RowId>; 64]`.
//!
//! The outer query API is the same for both: probe a literal, get a
//! [`CandidateProvider`], and let the LIKE verifier remain the correctness gate.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::marker::PhantomData;

use crate::RowId;
use crate::like::{LikePattern, LiteralAlgorithm};
use crate::query::{CandidateBatch, CandidateProvider, CandidateScratch};
use crate::storage::Column;
use crate::storage::dna2::{Dna2Column, Dna2Row};
use crate::storage::fsst::FsstColumn;
use crate::storage::utf8::Utf8Column;

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

#[inline(always)]
pub fn dna2_trigram_key(a: u8, b: u8, c: u8) -> u8 {
    debug_assert!(a < 4);
    debug_assert!(b < 4);
    debug_assert!(c < 4);
    (a << 4) | (b << 2) | c
}

/// Posting-list storage used by a trigram domain.
pub trait TrigramPostingStore<K: Copy> {
    fn insert(&mut self, key: K, row: RowId);
    fn finish(&mut self) {}
    fn postings(&self, key: K) -> Option<&[RowId]>;
}

#[derive(Debug, Clone, Default)]
pub struct HashMapPostingStore<K>
where
    K: Copy + Eq + Hash,
{
    postings: HashMap<K, Vec<RowId>>,
}

impl<K> HashMapPostingStore<K>
where
    K: Copy + Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.postings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.postings.is_empty()
    }
}

impl<K> TrigramPostingStore<K> for HashMapPostingStore<K>
where
    K: Copy + Eq + Hash,
{
    fn insert(&mut self, key: K, row: RowId) {
        self.postings.entry(key).or_default().push(row);
    }

    fn postings(&self, key: K) -> Option<&[RowId]> {
        self.postings.get(&key).map(Vec::as_slice)
    }
}

/// Fixed posting-list store for the 64 possible DNA2 trigrams.
#[derive(Debug, Clone)]
pub struct Fixed64PostingStore {
    postings: [Vec<RowId>; 64],
}

impl Fixed64PostingStore {
    pub fn new() -> Self {
        Self {
            postings: std::array::from_fn(|_| Vec::new()),
        }
    }

    pub fn postings_by_index(&self, key: usize) -> Option<&[RowId]> {
        let rows = self.postings.get(key)?;
        if rows.is_empty() {
            None
        } else {
            Some(rows.as_slice())
        }
    }
}

impl Default for Fixed64PostingStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TrigramPostingStore<u8> for Fixed64PostingStore {
    fn insert(&mut self, key: u8, row: RowId) {
        debug_assert!(key < 64);
        self.postings[key as usize].push(row);
    }

    fn postings(&self, key: u8) -> Option<&[RowId]> {
        if key >= 64 {
            return None;
        }
        self.postings_by_index(key as usize)
    }
}

/// Storage-specific trigram extraction and key encoding.
pub trait TrigramDomain<C>
where
    C: Column<Symbol = u8>,
{
    type Key: Copy + Eq + Ord + Hash;
    type Store: TrigramPostingStore<Self::Key>;

    fn new_store(row_count: RowId) -> Self::Store;

    /// Emit each distinct trigram key in `row` at most once.
    fn for_each_unique_row_trigram<F>(column: &C, row: RowId, f: F)
    where
        F: FnMut(Self::Key);

    /// Convert an exact literal symbol sequence into trigram keys.
    ///
    /// `None` means the literal is too short or cannot be represented in this
    /// domain.
    fn literal_trigrams(symbols: &[u8]) -> Option<Vec<Self::Key>>;
}

/// Default domain for byte-oriented UTF-8 columns.
#[derive(Debug, Clone, Copy, Default)]
pub struct Utf8ByteTrigramDomain;

impl<'db> TrigramDomain<Utf8Column<'db>> for Utf8ByteTrigramDomain {
    type Key = u32;
    type Store = HashMapPostingStore<u32>;

    fn new_store(_row_count: RowId) -> Self::Store {
        HashMapPostingStore::new()
    }

    fn for_each_unique_row_trigram<F>(column: &Utf8Column<'db>, row: RowId, mut f: F)
    where
        F: FnMut(Self::Key),
    {
        let bytes = column.row_bytes(row);
        if bytes.len() < 3 {
            return;
        }

        let mut seen = HashSet::<u32>::new();
        for window in bytes.windows(3) {
            let key = trigram_key(window[0], window[1], window[2]);
            if seen.insert(key) {
                f(key);
            }
        }
    }

    fn literal_trigrams(symbols: &[u8]) -> Option<Vec<Self::Key>> {
        let keys = trigram_keys(symbols);
        if keys.is_empty() { None } else { Some(keys) }
    }
}

/// Default domain for FSST-compressed byte columns.
///
/// The current FSST trigram domain decodes one row at index-build time and then
/// emits byte trigrams over the decompressed bytes. This keeps trigram semantics
/// identical to `Utf8ByteTrigramDomain` while preserving compressed storage for
/// the table itself.
#[derive(Debug, Clone, Copy, Default)]
pub struct FsstDecodedTrigramDomain;

impl<'db> TrigramDomain<FsstColumn<'db>> for FsstDecodedTrigramDomain {
    type Key = u32;
    type Store = HashMapPostingStore<u32>;

    fn new_store(_row_count: RowId) -> Self::Store {
        HashMapPostingStore::new()
    }

    fn for_each_unique_row_trigram<F>(column: &FsstColumn<'db>, row: RowId, mut f: F)
    where
        F: FnMut(Self::Key),
    {
        let bytes = column.row_bytes_decoded(row);
        if bytes.len() < 3 {
            return;
        }

        let mut seen = HashSet::<u32>::new();
        for window in bytes.windows(3) {
            let key = trigram_key(window[0], window[1], window[2]);
            if seen.insert(key) {
                f(key);
            }
        }
    }

    fn literal_trigrams(symbols: &[u8]) -> Option<Vec<Self::Key>> {
        let keys = trigram_keys(symbols);
        if keys.is_empty() { None } else { Some(keys) }
    }
}

/// Specialized domain for packed DNA2 columns.
///
/// The key space is fixed at 64 entries: `A/C/G/T` are encoded as `0..=3`, and
/// `abc` is encoded as `(a << 4) | (b << 2) | c`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Dna2TrigramDomain;

impl<'db> TrigramDomain<Dna2Column<'db>> for Dna2TrigramDomain {
    type Key = u8;
    type Store = Fixed64PostingStore;

    fn new_store(_row_count: RowId) -> Self::Store {
        Fixed64PostingStore::new()
    }

    fn for_each_unique_row_trigram<F>(column: &Dna2Column<'db>, row: RowId, f: F)
    where
        F: FnMut(Self::Key),
    {
        let row = column.row_view(row);
        row_for_each_unique_dna2_trigram(row, f);
    }

    fn literal_trigrams(symbols: &[u8]) -> Option<Vec<Self::Key>> {
        if symbols.len() < 3 {
            return None;
        }

        let mut out = Vec::with_capacity(symbols.len() - 2);
        for window in symbols.windows(3) {
            let a = window[0];
            let b = window[1];
            let c = window[2];
            if a >= 4 || b >= 4 || c >= 4 {
                return None;
            }
            out.push(dna2_trigram_key(a, b, c));
        }

        Some(out)
    }
}

#[inline]
fn row_for_each_unique_dna2_trigram<F>(row: Dna2Row<'_>, mut f: F)
where
    F: FnMut(u8),
{
    let len = row.logical_len();
    if len < 3 {
        return;
    }

    let mut seen = 0u64;
    let mut a = row.base_code_at(0);
    let mut b = row.base_code_at(1);

    for i in 2..len {
        let c = row.base_code_at(i);
        let key = dna2_trigram_key(a, b, c);
        let bit = 1u64 << key;
        if seen & bit == 0 {
            seen |= bit;
            f(key);
        }
        a = b;
        b = c;
    }
}

/// Columns that have a preferred specialized trigram domain.
pub trait HasTrigramIndex: Column<Symbol = u8> + Sized {
    type TrigramDomain: crate::index::trigram::TrigramDomain<Self>;

    fn build_trigram_index(&self) -> TrigramIndex<Self> {
        TrigramIndex::build(self)
    }
}

impl<'db> HasTrigramIndex for Utf8Column<'db> {
    type TrigramDomain = Utf8ByteTrigramDomain;
}

impl<'db> HasTrigramIndex for FsstColumn<'db> {
    type TrigramDomain = FsstDecodedTrigramDomain;
}

impl<'db> HasTrigramIndex for Dna2Column<'db> {
    type TrigramDomain = Dna2TrigramDomain;
}

/// Generic implementation. Most users should use [`TrigramIndex<C>`], which
/// automatically selects `C`'s preferred domain.
#[derive(Clone)]
pub struct TypedTrigramIndex<C, D>
where
    C: Column<Symbol = u8>,
    D: TrigramDomain<C>,
{
    row_count: RowId,
    store: D::Store,
    _marker: PhantomData<fn(&C, D)>,
}

impl<C, D> std::fmt::Debug for TypedTrigramIndex<C, D>
where
    C: Column<Symbol = u8>,
    D: TrigramDomain<C>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypedTrigramIndex")
            .field("row_count", &self.row_count)
            .finish_non_exhaustive()
    }
}

impl<C, D> TypedTrigramIndex<C, D>
where
    C: Column<Symbol = u8>,
    D: TrigramDomain<C>,
{
    pub fn build_with_domain(column: &C) -> Self {
        let row_count = column.row_count();
        let mut store = D::new_store(row_count);

        for row in 0..row_count {
            D::for_each_unique_row_trigram(column, row, |key| {
                store.insert(key, row);
            });
        }

        store.finish();

        Self {
            row_count,
            store,
            _marker: PhantomData,
        }
    }

    pub fn row_count(&self) -> RowId {
        self.row_count
    }

    pub fn postings_for_key(&self, key: D::Key) -> Option<&[RowId]> {
        self.store.postings(key)
    }

    /// Return candidate rows for a literal by intersecting all of its trigram
    /// postings. `None` means the literal is too short or not representable for
    /// this domain.
    pub fn search_literal(&self, literal: &[u8]) -> Option<Vec<RowId>> {
        let mut grams = D::literal_trigrams(literal)?;
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

/// Preferred trigram index for column `C`.
pub struct TrigramIndex<C>
where
    C: HasTrigramIndex,
{
    inner: TypedTrigramIndex<C, <C as HasTrigramIndex>::TrigramDomain>,
}

impl<C> std::fmt::Debug for TrigramIndex<C>
where
    C: HasTrigramIndex,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrigramIndex")
            .field("row_count", &self.row_count())
            .finish_non_exhaustive()
    }
}

impl<C> TrigramIndex<C>
where
    C: HasTrigramIndex,
{
    pub fn build(column: &C) -> Self {
        Self {
            inner: TypedTrigramIndex::<C, <C as HasTrigramIndex>::TrigramDomain>::build_with_domain(
                column,
            ),
        }
    }

    pub fn row_count(&self) -> RowId {
        self.inner.row_count()
    }

    pub fn search_literal(&self, literal: &[u8]) -> Option<Vec<RowId>> {
        self.inner.search_literal(literal)
    }

    pub fn probe_literal(&self, literal: &[u8], batch_rows: usize) -> Option<TrigramProbe> {
        self.inner.probe_literal(literal, batch_rows)
    }

    /// Compatibility helper for probing one trigram directly.
    ///
    /// For UTF-8 columns the gram is three bytes. For DNA2 columns it is three
    /// logical base codes (`A=0, C=1, G=2, T=3`). Missing or invalid grams yield
    /// an empty probe.
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
        self.inner.probe_longest_like_literal(pattern, batch_rows)
    }
}

impl<'db> TrigramIndex<Utf8Column<'db>> {
    pub fn postings_for_key(&self, key: u32) -> Option<&[RowId]> {
        self.inner.postings_for_key(key)
    }

    pub fn postings_for_gram(&self, gram: [u8; 3]) -> Option<&[RowId]> {
        self.postings_for_key(trigram_key(gram[0], gram[1], gram[2]))
    }
}

impl<'db> TrigramIndex<FsstColumn<'db>> {
    pub fn postings_for_key(&self, key: u32) -> Option<&[RowId]> {
        self.inner.postings_for_key(key)
    }

    pub fn postings_for_gram(&self, gram: [u8; 3]) -> Option<&[RowId]> {
        self.postings_for_key(trigram_key(gram[0], gram[1], gram[2]))
    }
}

impl<'db> TrigramIndex<Dna2Column<'db>> {
    pub fn postings_for_dna2_key(&self, key: u8) -> Option<&[RowId]> {
        self.inner.postings_for_key(key)
    }

    pub fn postings_for_dna2_gram(&self, gram: [u8; 3]) -> Option<&[RowId]> {
        if gram[0] >= 4 || gram[1] >= 4 || gram[2] >= 4 {
            return None;
        }
        self.postings_for_dna2_key(dna2_trigram_key(gram[0], gram[1], gram[2]))
    }
}

fn intersect_sorted_in_place(left: &mut Vec<RowId>, right: &[RowId]) {
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
    fn posting_trigram_uses_fixed_dna2_domain() {
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
        assert!(idx.search_literal(&[0, 9, 2]).is_none());
    }
}
