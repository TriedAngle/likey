//! Query execution shell.
//!
//! `execute_like` deliberately does not know LIKE syntax. Compiled patterns
//! such as [`LikePattern`](crate::LikePattern) implement [`RowVerifier`], while
//! scans and indexes provide candidate rows. The verifier remains the
//! correctness gate for every candidate.

use crate::storage::Column;
use crate::{LenConstraint, RowId};

#[derive(Debug, Default)]
pub struct QueryScratch {
    pub candidates: CandidateScratch,
    pub verify: VerifyScratch,
}

#[derive(Debug, Default)]
pub struct CandidateScratch {
    /// Generic row-id buffer for posting-list intersections or FM occurrence
    /// row deduplication.
    pub row_ids: Vec<RowId>,

    /// Generic bitmap buffer for dense candidate sets.
    pub bitmap_words: Vec<u64>,
}

impl CandidateScratch {
    pub fn clear(&mut self) {
        self.row_ids.clear();
        self.bitmap_words.clear();
    }
}

#[derive(Debug, Default)]
pub struct VerifyScratch {
    /// Generic byte buffer for streaming decode, encoded literal lowering, etc.
    pub bytes: Vec<u8>,

    /// Generic position buffer for algorithms that keep occurrence offsets.
    pub positions: Vec<u32>,
}

impl VerifyScratch {
    pub fn clear(&mut self) {
        self.bytes.clear();
        self.positions.clear();
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CandidateBatch<'a> {
    /// Consecutive physical row IDs `[start, start + len)`.
    RowRange { start: RowId, len: u64 },

    /// Sorted physical row IDs. Indexes should deduplicate before returning.
    SortedRows(&'a [RowId]),

    /// Dense bitmap block. `words[0]` corresponds to rows `base..base+64`.
    BitmapBlock { base: RowId, words: &'a [u64] },
}

impl CandidateBatch<'_> {
    pub fn is_empty(&self) -> bool {
        match self {
            CandidateBatch::RowRange { len, .. } => *len == 0,
            CandidateBatch::SortedRows(rows) => rows.is_empty(),
            CandidateBatch::BitmapBlock { words, .. } => words.iter().all(|&w| w == 0),
        }
    }
}

/// Producer of candidate row IDs.
///
/// Full scans, trigram postings, FM-index probes, equality indexes, and custom
/// benchmark fixtures can all implement this.
pub trait CandidateProvider {
    fn reset(&mut self);

    fn next_batch<'a>(
        &'a mut self,
        scratch: &'a mut CandidateScratch,
    ) -> Option<CandidateBatch<'a>>;
}

#[derive(Debug, Clone)]
pub struct FullScan {
    start: RowId,
    end: RowId,
    cursor: RowId,
    batch_rows: u64,
}

impl FullScan {
    pub fn new(row_count: RowId, batch_rows: u64) -> Self {
        Self::range(0, row_count, batch_rows)
    }

    pub fn range(start: RowId, len: u64, batch_rows: u64) -> Self {
        let batch_rows = batch_rows.max(1);
        Self {
            start,
            end: start.checked_add(len).expect("row range overflow"),
            cursor: start,
            batch_rows,
        }
    }
}

impl CandidateProvider for FullScan {
    fn reset(&mut self) {
        self.cursor = self.start;
    }

    fn next_batch<'a>(
        &'a mut self,
        _scratch: &'a mut CandidateScratch,
    ) -> Option<CandidateBatch<'a>> {
        if self.cursor >= self.end {
            return None;
        }
        let start = self.cursor;
        let len = (self.end - self.cursor).min(self.batch_rows);
        self.cursor += len;
        Some(CandidateBatch::RowRange { start, len })
    }
}

/// Candidate provider over a sorted row-id slice.
#[derive(Debug, Clone)]
pub struct SortedRowsProbe<'a> {
    rows: &'a [RowId],
    cursor: usize,
    batch_rows: usize,
}

impl<'a> SortedRowsProbe<'a> {
    pub fn new(rows: &'a [RowId], batch_rows: usize) -> Self {
        Self {
            rows,
            cursor: 0,
            batch_rows: batch_rows.max(1),
        }
    }

    pub fn rows(&self) -> &'a [RowId] {
        self.rows
    }
}

impl<'p> CandidateProvider for SortedRowsProbe<'p> {
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

/// Verifier for one candidate row.
///
/// Keep concrete verifier types monomorphized in hot paths. Avoid boxing this
/// trait unless benchmark overhead is irrelevant.
pub trait RowVerifier<C: Column> {
    fn len_constraint(&self) -> LenConstraint {
        LenConstraint::any()
    }

    fn verify(&self, column: &C, row: RowId, scratch: &mut VerifyScratch) -> bool;
}

/// A verifier useful for plumbing tests and examples.
#[derive(Debug, Clone, Copy, Default)]
pub struct AcceptAll {
    pub len_constraint: LenConstraint,
}

impl AcceptAll {
    pub fn new(len_constraint: LenConstraint) -> Self {
        Self { len_constraint }
    }
}

impl<C: Column> RowVerifier<C> for AcceptAll {
    fn len_constraint(&self) -> LenConstraint {
        self.len_constraint
    }

    fn verify(&self, _column: &C, _row: RowId, _scratch: &mut VerifyScratch) -> bool {
        true
    }
}

pub trait ResultSink {
    fn push(&mut self, row: RowId);
}

impl ResultSink for Vec<RowId> {
    fn push(&mut self, row: RowId) {
        Vec::push(self, row)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CountSink {
    pub count: u64,
}

impl ResultSink for CountSink {
    fn push(&mut self, _row: RowId) {
        self.count += 1;
    }
}

#[derive(Debug, Clone, Default)]
pub struct BitmapSink {
    words: Vec<u64>,
}

impl BitmapSink {
    pub fn with_row_capacity(row_count: RowId) -> Self {
        let words = ((row_count + 63) / 64) as usize;
        Self {
            words: vec![0; words],
        }
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn clear(&mut self) {
        self.words.fill(0);
    }

    pub fn contains(&self, row: RowId) -> bool {
        let idx = (row / 64) as usize;
        let bit = row % 64;
        self.words
            .get(idx)
            .map_or(false, |word| (word & (1u64 << bit)) != 0)
    }
}

impl ResultSink for BitmapSink {
    fn push(&mut self, row: RowId) {
        let idx = (row / 64) as usize;
        let bit = row % 64;
        if idx >= self.words.len() {
            self.words.resize(idx + 1, 0);
        }
        self.words[idx] |= 1u64 << bit;
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct QueryStats {
    pub candidate_rows_seen: u64,
    pub rows_after_len_filter: u64,
    pub rows_matched: u64,
}

pub fn execute_like<C, P, V, S>(
    column: &C,
    candidates: &mut P,
    verifier: &V,
    scratch: &mut QueryScratch,
    sink: &mut S,
) -> QueryStats
where
    C: Column,
    P: CandidateProvider,
    V: RowVerifier<C>,
    S: ResultSink,
{
    let len_constraint = verifier.len_constraint();
    let mut stats = QueryStats::default();

    candidates.reset();

    loop {
        let Some(batch) = candidates.next_batch(&mut scratch.candidates) else {
            break;
        };

        match batch {
            CandidateBatch::RowRange { start, len } => {
                for row in start..start + len {
                    verify_one(
                        column,
                        row,
                        len_constraint,
                        verifier,
                        &mut scratch.verify,
                        sink,
                        &mut stats,
                    );
                }
            }
            CandidateBatch::SortedRows(rows) => {
                for &row in rows {
                    verify_one(
                        column,
                        row,
                        len_constraint,
                        verifier,
                        &mut scratch.verify,
                        sink,
                        &mut stats,
                    );
                }
            }
            CandidateBatch::BitmapBlock { base, words } => {
                for (word_idx, &word) in words.iter().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as u64;
                        let row = base + word_idx as u64 * 64 + bit;
                        verify_one(
                            column,
                            row,
                            len_constraint,
                            verifier,
                            &mut scratch.verify,
                            sink,
                            &mut stats,
                        );
                        bits &= bits - 1;
                    }
                }
            }
        }
    }

    stats
}

#[inline]
fn verify_one<C, V, S>(
    column: &C,
    row: RowId,
    len_constraint: LenConstraint,
    verifier: &V,
    scratch: &mut VerifyScratch,
    sink: &mut S,
    stats: &mut QueryStats,
) where
    C: Column,
    V: RowVerifier<C>,
    S: ResultSink,
{
    debug_assert!(row < column.row_count(), "candidate row out of bounds");
    stats.candidate_rows_seen += 1;

    let len = column.logical_len(row);
    if !len_constraint.matches(len) {
        return;
    }
    stats.rows_after_len_filter += 1;

    if verifier.verify(column, row, scratch) {
        stats.rows_matched += 1;
        sink.push(row);
    }
}
