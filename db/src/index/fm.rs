//! Baseline FM-index over dense logical-symbol columns.
//!
//! This index builds over any `Column<Symbol = u8>`. UTF-8 columns
//! contribute byte symbols; DNA2 columns contribute base-code symbols `0..=3`.
//! The FM-index uses private internal symbols for row separators and the final
//! sentinel, so literal matches cannot cross row boundaries.

use crate::like::{LikePattern, LiteralAlgorithm};
use crate::query::{CandidateBatch, CandidateProvider, CandidateScratch};
use crate::storage::Column;
use crate::{BuildIndex, RowId};

const SENTINEL: u16 = 0;
const SEPARATOR: u16 = 1;
const DATA_BASE: u16 = 2;
const NO_ROW: RowId = RowId::MAX;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors returned while building an FM-index.
pub enum FmIndexError {
    RowCountTooLarge,
    TextTooLarge,
    CheckpointIsZero,
}

impl std::fmt::Display for FmIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FmIndexError::RowCountTooLarge => write!(f, "row count is too large for this FM-index"),
            FmIndexError::TextTooLarge => write!(f, "indexed text is too large for this FM-index"),
            FmIndexError::CheckpointIsZero => write!(f, "FM-index checkpoint must be non-zero"),
        }
    }
}

impl std::error::Error for FmIndexError {}

#[derive(Debug, Clone)]
/// Baseline FM-index over a dense logical-symbol column.
///
/// The index concatenates every row's logical symbols, inserts private row
/// separators between rows, and appends a final sentinel. Searches return row
/// IDs whose rows contain the searched literal; matches cannot cross row
/// boundaries. The implementation is intended as a correctness-oriented
/// benchmark index rather than a compressed production index.
pub struct FmIndex {
    text_len: usize,
    sa: Box<[usize]>,
    bwt_ranks: Box<[u16]>,
    c: Box<[usize]>,
    counts: Box<[usize]>,
    occ: Box<[u32]>,
    checkpoint: usize,
    sigma: usize,
    symbol_to_rank: Box<[i16]>,
    data_ranks: Box<[usize]>,
    pos_to_row: Box<[RowId]>,
    row_count: RowId,
}

impl<C> BuildIndex<C> for FmIndex
where
    C: Column<Symbol = u8>,
{
    fn build(column: &C) -> Self {
        FmIndex::build(column).expect("FM-index build failed")
    }
}

impl FmIndex {
    /// Default occurrence checkpoint spacing.
    pub const DEFAULT_CHECKPOINT: usize = 128;

    /// Build an FM-index using [`DEFAULT_CHECKPOINT`](Self::DEFAULT_CHECKPOINT).
    pub fn build<C>(column: &C) -> Result<Self, FmIndexError>
    where
        C: Column<Symbol = u8>,
    {
        Self::build_with_checkpoint(column, Self::DEFAULT_CHECKPOINT)
    }

    /// Build an FM-index with a custom occurrence checkpoint spacing.
    pub fn build_with_checkpoint<C>(column: &C, checkpoint: usize) -> Result<Self, FmIndexError>
    where
        C: Column<Symbol = u8>,
    {
        if checkpoint == 0 {
            return Err(FmIndexError::CheckpointIsZero);
        }

        let row_count = column.row_count();
        if row_count == NO_ROW {
            return Err(FmIndexError::RowCountTooLarge);
        }

        let row_count_usize =
            usize::try_from(row_count).map_err(|_| FmIndexError::RowCountTooLarge)?;
        let mut total_symbols = 1usize; // final sentinel
        total_symbols = total_symbols
            .checked_add(row_count_usize)
            .ok_or(FmIndexError::TextTooLarge)?; // row separators
        for row in 0..row_count {
            total_symbols = total_symbols
                .checked_add(column.logical_len(row) as usize)
                .ok_or(FmIndexError::TextTooLarge)?;
        }

        let mut text = Vec::with_capacity(total_symbols);
        let mut pos_to_row = Vec::with_capacity(total_symbols);

        for row in 0..row_count {
            for sym in column.symbols(row) {
                text.push(encode_data_symbol(sym));
                pos_to_row.push(row);
            }
            text.push(SEPARATOR);
            pos_to_row.push(NO_ROW);
        }

        text.push(SENTINEL);
        pos_to_row.push(NO_ROW);

        debug_assert_eq!(text.len(), total_symbols);
        debug_assert_eq!(pos_to_row.len(), text.len());

        let text_len = text.len();
        let sa = build_suffix_array(&text);
        let bwt_symbols = build_bwt(&text, &sa);
        let alphabet = build_alphabet(&text);
        let bwt_ranks = remap_bwt(&bwt_symbols, &alphabet.symbol_to_rank);
        let c = build_c(&alphabet.counts);
        let occ = build_occ(&bwt_ranks, alphabet.counts.len(), checkpoint);
        let data_ranks = alphabet
            .rank_to_symbol
            .iter()
            .enumerate()
            .filter_map(|(rank, &symbol)| (symbol >= DATA_BASE).then_some(rank))
            .collect::<Vec<_>>();

        Ok(Self {
            text_len,
            sa: sa.into_boxed_slice(),
            bwt_ranks: bwt_ranks.into_boxed_slice(),
            c: c.into_boxed_slice(),
            counts: alphabet.counts.into_boxed_slice(),
            occ: occ.into_boxed_slice(),
            checkpoint,
            sigma: alphabet.rank_to_symbol.len(),
            symbol_to_rank: alphabet.symbol_to_rank.into_boxed_slice(),
            data_ranks: data_ranks.into_boxed_slice(),
            pos_to_row: pos_to_row.into_boxed_slice(),
            row_count,
        })
    }

    /// Number of source rows indexed.
    pub fn row_count(&self) -> RowId {
        self.row_count
    }

    /// Total indexed symbols, including row separators and final sentinel.
    pub fn text_len(&self) -> usize {
        self.text_len
    }

    /// Suffix array over the internal encoded text.
    pub fn suffix_array(&self) -> &[usize] {
        &self.sa
    }

    /// Number of internal alphabet ranks.
    pub fn alphabet_size(&self) -> usize {
        self.sigma
    }

    /// Occurrence checkpoint spacing.
    pub fn checkpoint(&self) -> usize {
        self.checkpoint
    }

    /// Return the suffix-array interval for an exact literal.
    pub fn backward_search(&self, needle: &[u8]) -> Option<(usize, usize)> {
        if needle.is_empty() {
            return Some((0, self.text_len));
        }

        let mut top = 0usize;
        let mut bottom = self.text_len;

        for &symbol in needle.iter().rev() {
            let rank = self.rank_for_external_symbol(symbol)?;
            if self.counts[rank] == 0 {
                return None;
            }

            top = self.c[rank] + self.occ_at(rank, top);
            bottom = self.c[rank] + self.occ_at(rank, bottom);

            if top >= bottom {
                return None;
            }
        }

        Some((top, bottom))
    }

    /// Return sorted, deduplicated rows containing an exact literal.
    pub fn search_rows(&self, needle: &[u8]) -> Vec<RowId> {
        if needle.is_empty() {
            return (0..self.row_count).collect();
        }

        let Some((top, bottom)) = self.backward_search(needle) else {
            return Vec::new();
        };

        self.rows_from_interval(top, bottom)
    }

    /// Search a fixed-length pattern where `wildcard` matches any one data
    /// symbol. It never matches row separators or the final sentinel.
    pub fn search_rows_with_wildcard(&self, pattern: &[u8], wildcard: u8) -> Vec<RowId> {
        if pattern.is_empty() {
            return (0..self.row_count).collect();
        }

        let mut rows = Vec::new();
        self.wildcard_rec(
            pattern,
            wildcard,
            pattern.len() as isize - 1,
            0,
            self.text_len,
            &mut rows,
        );
        rows.sort_unstable();
        rows.dedup();
        rows
    }

    /// Build a candidate probe for an exact literal.
    pub fn probe(&self, needle: &[u8], batch_rows: usize) -> FmProbe {
        FmProbe::new(self.search_rows(needle), batch_rows)
    }

    /// Build a candidate probe for a fixed-length wildcard literal.
    pub fn probe_with_wildcard(&self, pattern: &[u8], wildcard: u8, batch_rows: usize) -> FmProbe {
        FmProbe::new(
            self.search_rows_with_wildcard(pattern, wildcard),
            batch_rows,
        )
    }

    /// Use the longest exact literal exported by a compiled LIKE pattern.
    ///
    /// If the pattern has no exact literal fragment, return `None`; call a full
    /// scan or another index in that case.
    pub fn probe_longest_like_literal<A>(
        &self,
        pattern: &LikePattern<A>,
        batch_rows: usize,
    ) -> Option<FmProbe>
    where
        A: LiteralAlgorithm,
    {
        pattern
            .longest_indexable_literal()
            .map(|lit| self.probe(lit, batch_rows))
    }

    fn rows_from_interval(&self, top: usize, bottom: usize) -> Vec<RowId> {
        let mut rows = Vec::with_capacity(bottom.saturating_sub(top).min(1024));
        for &pos in &self.sa[top..bottom] {
            if let Some(row) = self.row_for_text_pos(pos) {
                rows.push(row);
            }
        }
        rows.sort_unstable();
        rows.dedup();
        rows
    }

    fn wildcard_rec(
        &self,
        pattern: &[u8],
        wildcard: u8,
        idx: isize,
        top: usize,
        bottom: usize,
        rows: &mut Vec<RowId>,
    ) {
        if top >= bottom {
            return;
        }

        if idx < 0 {
            for &pos in &self.sa[top..bottom] {
                if let Some(row) = self.row_for_text_pos(pos) {
                    rows.push(row);
                }
            }
            return;
        }

        let ch = pattern[idx as usize];
        if ch == wildcard {
            for &rank in &self.data_ranks {
                let new_top = self.c[rank] + self.occ_at(rank, top);
                let new_bottom = self.c[rank] + self.occ_at(rank, bottom);
                if new_top < new_bottom {
                    self.wildcard_rec(pattern, wildcard, idx - 1, new_top, new_bottom, rows);
                }
            }
        } else if let Some(rank) = self.rank_for_external_symbol(ch) {
            let new_top = self.c[rank] + self.occ_at(rank, top);
            let new_bottom = self.c[rank] + self.occ_at(rank, bottom);
            if new_top < new_bottom {
                self.wildcard_rec(pattern, wildcard, idx - 1, new_top, new_bottom, rows);
            }
        }
    }

    #[inline]
    fn row_for_text_pos(&self, pos: usize) -> Option<RowId> {
        let row = *self.pos_to_row.get(pos)?;
        (row != NO_ROW).then_some(row)
    }

    #[inline]
    fn rank_for_external_symbol(&self, symbol: u8) -> Option<usize> {
        rank_for_symbol(&self.symbol_to_rank, encode_data_symbol(symbol))
    }

    #[inline]
    fn occ_at(&self, rank: usize, index: usize) -> usize {
        let capped = index.min(self.text_len);
        let base_idx = capped / self.checkpoint;
        let base_pos = base_idx * self.checkpoint;
        let mut count = self.occ[base_idx * self.sigma + rank] as usize;

        for &r in &self.bwt_ranks[base_pos..capped] {
            if r as usize == rank {
                count += 1;
            }
        }

        count
    }
}

#[derive(Debug, Clone)]
/// Candidate provider backed by FM-index search results.
pub struct FmProbe {
    rows: Vec<RowId>,
    cursor: usize,
    batch_rows: usize,
}

impl FmProbe {
    /// Create a probe from sorted, deduplicated row IDs.
    pub fn new(rows: Vec<RowId>, batch_rows: usize) -> Self {
        Self {
            rows,
            cursor: 0,
            batch_rows: batch_rows.max(1),
        }
    }

    /// Candidate rows returned by the probe.
    pub fn rows(&self) -> &[RowId] {
        &self.rows
    }

    /// Consume the probe and return its candidate rows.
    pub fn into_rows(self) -> Vec<RowId> {
        self.rows
    }

    /// Whether the probe has no candidate rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

impl CandidateProvider for FmProbe {
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

struct Alphabet {
    symbol_to_rank: Vec<i16>,
    rank_to_symbol: Vec<u16>,
    counts: Vec<usize>,
}

#[inline]
fn encode_data_symbol(symbol: u8) -> u16 {
    DATA_BASE + u16::from(symbol)
}

#[inline]
fn rank_for_symbol(symbol_to_rank: &[i16], symbol: u16) -> Option<usize> {
    let rank = *symbol_to_rank.get(symbol as usize)?;
    (rank >= 0).then_some(rank as usize)
}

fn build_suffix_array(text: &[u16]) -> Vec<usize> {
    let n = text.len();
    let mut sa = (0..n).collect::<Vec<_>>();
    if n <= 1 {
        return sa;
    }

    let mut rank = text.iter().map(|&s| i32::from(s)).collect::<Vec<_>>();
    let mut next_rank = vec![0i32; n];
    let mut k = 1usize;

    loop {
        sa.sort_unstable_by(|&a, &b| match rank[a].cmp(&rank[b]) {
            std::cmp::Ordering::Equal => rank_at(&rank, a, k).cmp(&rank_at(&rank, b, k)),
            other => other,
        });

        next_rank[sa[0]] = 0;
        for i in 1..n {
            let prev = sa[i - 1];
            let curr = sa[i];
            let prev_key = (rank[prev], rank_at(&rank, prev, k));
            let curr_key = (rank[curr], rank_at(&rank, curr, k));
            next_rank[curr] = next_rank[prev] + if prev_key != curr_key { 1 } else { 0 };
        }

        rank.clone_from(&next_rank);
        if rank[sa[n - 1]] as usize == n - 1 || k >= n {
            break;
        }
        k = k.saturating_mul(2);
    }

    sa
}

#[inline]
fn rank_at(rank: &[i32], pos: usize, width: usize) -> i32 {
    match pos.checked_add(width) {
        Some(next) if next < rank.len() => rank[next],
        _ => -1,
    }
}

fn build_bwt(text: &[u16], sa: &[usize]) -> Vec<u16> {
    let n = text.len();
    let mut bwt = Vec::with_capacity(n);
    for &pos in sa {
        if pos == 0 {
            bwt.push(text[n - 1]);
        } else {
            bwt.push(text[pos - 1]);
        }
    }
    bwt
}

fn build_alphabet(text: &[u16]) -> Alphabet {
    let max_symbol = text.iter().copied().max().unwrap_or(0) as usize;
    let mut raw_counts = vec![0usize; max_symbol + 1];
    for &symbol in text {
        raw_counts[symbol as usize] += 1;
    }

    let mut symbol_to_rank = vec![-1i16; raw_counts.len()];
    let mut rank_to_symbol = Vec::new();
    let mut counts = Vec::new();

    for (symbol, &count) in raw_counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let rank = rank_to_symbol.len();
        symbol_to_rank[symbol] = rank as i16;
        rank_to_symbol.push(symbol as u16);
        counts.push(count);
    }

    Alphabet {
        symbol_to_rank,
        rank_to_symbol,
        counts,
    }
}

fn remap_bwt(bwt_symbols: &[u16], symbol_to_rank: &[i16]) -> Vec<u16> {
    bwt_symbols
        .iter()
        .map(|&symbol| {
            rank_for_symbol(symbol_to_rank, symbol).expect("BWT symbol must be in alphabet") as u16
        })
        .collect()
}

fn build_c(counts: &[usize]) -> Vec<usize> {
    let mut c = Vec::with_capacity(counts.len());
    let mut total = 0usize;
    for &count in counts {
        c.push(total);
        total += count;
    }
    c
}

fn build_occ(bwt_ranks: &[u16], sigma: usize, checkpoint: usize) -> Vec<u32> {
    let checkpoint_count = bwt_ranks.len() / checkpoint + 1;
    let mut occ = vec![0u32; checkpoint_count * sigma];
    let mut counts = vec![0u32; sigma];

    for (idx, &rank) in bwt_ranks.iter().enumerate() {
        counts[rank as usize] = counts[rank as usize]
            .checked_add(1)
            .expect("FM-index occurrence count exceeds u32");
        if (idx + 1) % checkpoint == 0 {
            let checkpoint_idx = (idx + 1) / checkpoint;
            let dst = checkpoint_idx * sigma;
            occ[dst..dst + sigma].copy_from_slice(&counts);
        }
    }

    occ
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DbBuilder, Dna2TableBuilder, Utf8TableBuilder};

    #[test]
    fn utf8_exact_rows() {
        let mut docs = Utf8TableBuilder::new("docs");
        docs.push_str("banana");
        docs.push_str("bandana");
        docs.push_str("apple");

        let mut dbb = DbBuilder::new();
        let id = dbb.add_utf8_table(docs).unwrap();
        let db = dbb.freeze();
        let col = db.utf8_table(id).unwrap().text();

        let fm = FmIndex::build(&col).unwrap();
        assert_eq!(fm.search_rows(b"ana"), vec![0, 1]);
        assert_eq!(fm.search_rows(b"apple"), vec![2]);
        assert!(fm.search_rows(b"orange").is_empty());
    }

    #[test]
    fn dna2_exact_and_wildcard_rows() {
        let mut reads = Dna2TableBuilder::new("reads");
        reads.push_str("AACGT").unwrap();
        reads.push_str("TTACG").unwrap();
        reads.push_str("AGGT").unwrap();
        reads.push_str("CCCCC").unwrap();

        let mut dbb = DbBuilder::new();
        let id = dbb.add_dna2_table(reads).unwrap();
        let db = dbb.freeze();
        let col = db.dna2_table(id).unwrap().sequence();

        let fm = FmIndex::build(&col).unwrap();
        assert_eq!(fm.search_rows(&[0, 1, 2]), vec![0, 1]);
        assert_eq!(
            fm.search_rows_with_wildcard(&[0, 255, 2], 255),
            vec![0, 1, 2]
        );
    }
}
