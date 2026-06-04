//! Manual fixed-64 trigram index for packed DNA2 columns.
//!
//! This is intentionally not the default `TrigramIndex` used by the DB/runner.
//! The generic hash-map trigram index is simpler and works for all
//! `Column<Symbol = u8>` implementations. This fixed DNA2 layout is kept so it
//! can be measured or used manually when index build time matters.
//!
//! Measurement on a GRCh38.p14 FASTA subset downloaded with
//! `scripts/download_bio_benchmarks.py`, loading the first 100,000,000 valid
//! A/C/G/T bases from `data/fasta/dna_benchmark.fna`:
//!
//! ```text
//! loaded rows=8 bases=100000000 payload_bytes=25000000 load=647.939ms
//!
//! domain       build      probe      total_candidates
//! fixed64      53.185ms   32.641ms   1600000
//! sparse_hash  461.854ms  38.254ms   1600000
//! ```

use crate::RowId;
use crate::like::{LikePattern, LiteralAlgorithm};
use crate::storage::Column;
use crate::storage::dna2::{Dna2Column, Dna2Row};

use super::trigram::{TrigramProbe, intersect_sorted_in_place};

#[inline(always)]
pub fn dna2_trigram_key(a: u8, b: u8, c: u8) -> u8 {
    debug_assert!(a < 4);
    debug_assert!(b < 4);
    debug_assert!(c < 4);
    (a << 4) | (b << 2) | c
}

#[derive(Debug, Clone)]
pub struct Dna2FixedTrigramIndex {
    row_count: RowId,
    postings: [Vec<RowId>; 64],
}

impl Dna2FixedTrigramIndex {
    pub fn build(column: &Dna2Column<'_>) -> Self {
        let row_count = column.row_count();
        let mut postings: [Vec<RowId>; 64] = std::array::from_fn(|_| Vec::new());

        for row in 0..row_count {
            let row_view = column.row_view(row);
            row_for_each_unique_trigram(row_view, |key| postings[key as usize].push(row));
        }

        Self {
            row_count,
            postings,
        }
    }

    pub fn row_count(&self) -> RowId {
        self.row_count
    }

    pub fn postings_for_key(&self, key: u8) -> Option<&[RowId]> {
        if key >= 64 {
            return None;
        }
        let rows = &self.postings[key as usize];
        if rows.is_empty() {
            None
        } else {
            Some(rows.as_slice())
        }
    }

    pub fn postings_for_gram(&self, gram: [u8; 3]) -> Option<&[RowId]> {
        if gram[0] >= 4 || gram[1] >= 4 || gram[2] >= 4 {
            return None;
        }
        self.postings_for_key(dna2_trigram_key(gram[0], gram[1], gram[2]))
    }

    pub fn search_literal(&self, literal: &[u8]) -> Option<Vec<RowId>> {
        let mut grams = literal_trigrams(literal)?;
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

fn literal_trigrams(literal: &[u8]) -> Option<Vec<u8>> {
    if literal.len() < 3 {
        return None;
    }

    let mut out = Vec::with_capacity(literal.len() - 2);
    for window in literal.windows(3) {
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

fn row_for_each_unique_trigram<F>(row: Dna2Row<'_>, mut f: F)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::DbBuilder;
    use crate::like::LikePattern;
    use crate::query::{QueryScratch, execute_like};
    use crate::storage::Column;
    use crate::storage::dna2::Dna2TableBuilder;
    use crate::{Dna2, FullScan};

    #[test]
    fn fixed_dna2_trigram_is_manually_callable() {
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

        let idx = Dna2FixedTrigramIndex::build(&col);
        assert_eq!(idx.search_literal(&[0, 1, 2]).unwrap(), vec![0, 1, 3]);
        assert_eq!(idx.search_literal(&[2, 2, 2]).unwrap(), vec![2]);
        assert!(idx.search_literal(&[0, 1]).is_none());
        assert!(idx.search_literal(&[0, 9, 2]).is_none());
    }

    #[test]
    fn fixed_dna2_trigram_candidates_match_full_scan_results() {
        let mut reads = Dna2TableBuilder::new("reads");
        for row in ["ACGTACGT", "TTTACGTT", "GGGGGGGG", "AACGAAAA"] {
            reads.push_str(row).unwrap();
        }

        let mut dbb = DbBuilder::new();
        let id = dbb.add_dna2_table(reads).unwrap();
        let db = dbb.freeze();
        let table = db.dna2_table(id).unwrap();
        let col = table.sequence();
        let like = LikePattern::<Dna2>::compile("%ACG%").unwrap();

        let mut scratch = QueryScratch::default();
        let mut expected = Vec::new();
        let mut scan = FullScan::new(col.row_count(), 2);
        execute_like(&col, &mut scan, &like, &mut scratch, &mut expected);

        let idx = Dna2FixedTrigramIndex::build(&col);
        let mut probe = idx.probe_longest_like_literal(&like, 2).unwrap();
        scratch.candidates.clear();
        scratch.verify.clear();
        let mut got = Vec::new();
        execute_like(&col, &mut probe, &like, &mut scratch, &mut got);

        assert_eq!(got, expected);
    }
}
