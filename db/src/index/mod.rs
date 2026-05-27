//! Index extension traits and small baseline index implementations.
//!
//! The database itself does not need to know about concrete indexes. The seam is
//! [`CandidateProvider`](crate::CandidateProvider): a trigram index, FM-index,
//! equality index, or benchmark fixture exposes a probe object that yields row
//! candidates for [`execute_like`](crate::execute_like).

pub mod fm;
pub mod trigram;

pub use fm::{FmIndex, FmIndexError, FmProbe};
pub use trigram::{
    Dna2TrigramDomain, Fixed64PostingStore, FsstDecodedTrigramDomain, HasTrigramIndex,
    HashMapPostingStore, TrigramDomain, TrigramIndex, TrigramPostingStore, TrigramProbe,
    TypedTrigramIndex, Utf8ByteTrigramDomain, dna2_trigram_key, trigram_key, trigram_keys,
};

use crate::RowId;
use crate::query::CandidateProvider;
use crate::storage::Column;

/// Optional build hook for reusable index types.
///
/// You do not need this trait to use `execute_like`; it is only a convention.
pub trait BuildIndex<C: Column>: Sized {
    fn build(column: &C) -> Self;
}

impl<C> BuildIndex<C> for TrigramIndex<C>
where
    C: trigram::HasTrigramIndex,
{
    fn build(column: &C) -> Self {
        TrigramIndex::build(column)
    }
}

/// Marker trait for index probes.
///
/// Implementing [`CandidateProvider`] is the important part. This marker trait is
/// useful when naming APIs such as `fn probe(...) -> impl IndexProbe`.
pub trait IndexProbe: CandidateProvider {}

impl<T: CandidateProvider> IndexProbe for T {}

/// Intersect two sorted, deduplicated row-id lists into `out`.
///
/// This is handy for row-list trigram indexes where a LIKE pattern has several
/// required grams. The output is also sorted and deduplicated if the inputs are.
pub fn intersect_sorted_rowids(a: &[RowId], b: &[RowId], out: &mut Vec<RowId>) {
    out.clear();
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::CandidateProvider;
    use crate::storage::Column;
    use crate::storage::dna2::{Dna2Column, Dna2TableBuilder};
    use crate::storage::utf8::{Utf8Column, Utf8TableBuilder};
    use crate::{
        DbBuilder, Dna2, FullScan, LikePattern, QueryScratch, RowLiteralSearch, StdSearch,
        execute_like,
    };

    macro_rules! index_suites {
        ($($test_name:ident => $suite:ident, $index:ty),* $(,)?) => {
            $(
                #[test]
                fn $test_name() {
                    $suite::<$index>();
                }
            )*
        };
    }

    index_suites! {
        fm_utf8_like_suite => utf8_like_index_suite, FmIndexUnderTest,
        trigram_utf8_like_suite => utf8_like_index_suite, TrigramIndexUnderTest,
        fm_dna2_like_suite => dna2_like_index_suite, FmIndexUnderTest,
        trigram_dna2_like_suite => dna2_like_index_suite, TrigramIndexUnderTest,
    }

    trait IndexUnderTest<C, A>
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
    {
        fn name() -> &'static str;
        fn literal_candidates(column: &C, literal: &[u8]) -> Option<Vec<RowId>>;
        fn like_matches(column: &C, pattern: &LikePattern<A>) -> Option<Vec<RowId>>;

        fn literal_candidates_are_exact() -> bool {
            false
        }
    }

    struct FmIndexUnderTest;

    impl<C, A> IndexUnderTest<C, A> for FmIndexUnderTest
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
    {
        fn name() -> &'static str {
            "fm"
        }

        fn literal_candidates(column: &C, literal: &[u8]) -> Option<Vec<RowId>> {
            let index = FmIndex::build(column).expect("FM-index should build");
            Some(index.search_rows(literal))
        }

        fn like_matches(column: &C, pattern: &LikePattern<A>) -> Option<Vec<RowId>> {
            let index = FmIndex::build(column).expect("FM-index should build");
            let probe = index.probe_longest_like_literal(pattern, 2)?;
            Some(execute_with_probe(column, probe, pattern))
        }

        fn literal_candidates_are_exact() -> bool {
            true
        }
    }

    struct TrigramIndexUnderTest;

    impl<C, A> IndexUnderTest<C, A> for TrigramIndexUnderTest
    where
        C: HasTrigramIndex,
        A: RowLiteralSearch<C>,
    {
        fn name() -> &'static str {
            "trigram"
        }

        fn literal_candidates(column: &C, literal: &[u8]) -> Option<Vec<RowId>> {
            let index = TrigramIndex::build(column);
            index.search_literal(literal)
        }

        fn like_matches(column: &C, pattern: &LikePattern<A>) -> Option<Vec<RowId>> {
            let index = TrigramIndex::build(column);
            let probe = index.probe_longest_like_literal(pattern, 2)?;
            Some(execute_with_probe(column, probe, pattern))
        }
    }

    fn utf8_like_index_suite<I>()
    where
        for<'db> I: IndexUnderTest<Utf8Column<'db>, StdSearch>,
    {
        let (db, id) = utf8_db();
        let table = db.utf8_table(id).unwrap();
        let column = table.text();

        for literal in ["ana", "abcd", "café", "rés", "🍌ban", "missing"] {
            assert_literal_candidates_cover::<_, StdSearch, I>(&column, literal.as_bytes());
        }

        for pattern in [
            "%ana%",
            "ban%",
            "%ana",
            "%abcd%",
            "%café",
            "rés%",
            "%🍌ban%",
            "___ana%",
        ] {
            assert_index_matches_full_scan::<_, StdSearch, I>(&column, pattern);
        }
    }

    fn dna2_like_index_suite<I>()
    where
        for<'db> I: IndexUnderTest<Dna2Column<'db>, Dna2>,
    {
        let (db, id) = dna2_db();
        let table = db.dna2_table(id).unwrap();
        let column = table.sequence();

        for literal in [&[0, 1, 2][..], &[0, 1, 2, 3], &[2, 2, 2], &[3, 0, 1, 2]] {
            assert_literal_candidates_cover::<_, Dna2, I>(&column, literal);
        }

        for pattern in [
            "%ACG%", "ACG%", "%CGT", "%GGG%", "%ACGT%", "T%ACG%", "A_G%ACG%",
        ] {
            assert_index_matches_full_scan::<_, Dna2, I>(&column, pattern);
        }
    }

    fn assert_literal_candidates_cover<C, A, I>(column: &C, literal: &[u8])
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
        I: IndexUnderTest<C, A>,
    {
        let expected = exact_literal_rows(column, literal);
        let candidates = I::literal_candidates(column, literal).unwrap_or_else(|| {
            panic!(
                "{} did not produce candidates for literal {literal:?}",
                I::name()
            )
        });

        assert_sorted_unique(&candidates, I::name(), literal);

        if I::literal_candidates_are_exact() {
            assert_eq!(
                candidates,
                expected,
                "{} exact literal candidates mismatch for literal {literal:?}",
                I::name()
            );
        } else {
            for row in expected {
                assert!(
                    candidates.binary_search(&row).is_ok(),
                    "{} missed row {row} for literal {literal:?}; candidates={candidates:?}",
                    I::name()
                );
            }
        }
    }

    fn assert_index_matches_full_scan<C, A, I>(column: &C, pattern: &str)
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
        I: IndexUnderTest<C, A>,
    {
        let like = LikePattern::<A>::compile(pattern).expect("LIKE pattern should compile");
        let expected = execute_full_scan(column, &like);
        let indexed = I::like_matches(column, &like).unwrap_or_else(|| {
            panic!(
                "{} did not produce a LIKE probe for pattern {pattern:?}",
                I::name()
            )
        });

        assert_eq!(
            indexed,
            expected,
            "{} LIKE results mismatch for pattern {pattern:?}",
            I::name()
        );
    }

    fn execute_full_scan<C, A>(column: &C, pattern: &LikePattern<A>) -> Vec<RowId>
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
    {
        let scan = FullScan::new(column.row_count(), 2);
        execute_with_probe(column, scan, pattern)
    }

    fn execute_with_probe<C, A, P>(column: &C, mut probe: P, pattern: &LikePattern<A>) -> Vec<RowId>
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
        P: CandidateProvider,
    {
        let mut scratch = QueryScratch::default();
        let mut matches = Vec::<RowId>::new();
        execute_like(column, &mut probe, pattern, &mut scratch, &mut matches);
        matches
    }

    fn exact_literal_rows<C>(column: &C, literal: &[u8]) -> Vec<RowId>
    where
        C: Column<Symbol = u8>,
    {
        (0..column.row_count())
            .filter(|&row| row_contains_literal(column, row, literal))
            .collect()
    }

    fn row_contains_literal<C>(column: &C, row: RowId, literal: &[u8]) -> bool
    where
        C: Column<Symbol = u8>,
    {
        if literal.is_empty() {
            return true;
        }

        let symbols = column.symbols(row).collect::<Vec<_>>();
        symbols
            .windows(literal.len())
            .any(|window| window == literal)
    }

    fn assert_sorted_unique(rows: &[RowId], index_name: &str, literal: &[u8]) {
        for window in rows.windows(2) {
            assert!(
                window[0] < window[1],
                "{index_name} candidates are not sorted and unique for literal {literal:?}: {rows:?}"
            );
        }
    }

    fn utf8_db() -> (crate::Db, crate::TableId) {
        let mut docs = Utf8TableBuilder::new("docs");
        for row in [
            "banana",
            "bandana",
            "cabana",
            "anagram",
            "résumé banana",
            "naïve café",
            "🍌banana",
            "",
            "xxana",
            "ban",
            "abcXbcd",
            "xxabcdyy",
        ] {
            docs.push_str(row);
        }

        let mut dbb = DbBuilder::new();
        let id = dbb.add_utf8_table(docs).unwrap();
        (dbb.freeze(), id)
    }

    fn dna2_db() -> (crate::Db, crate::TableId) {
        let mut reads = Dna2TableBuilder::new("reads");
        for row in [
            "ACGTACGT", "TTTACGTT", "GGGGGGGG", "AACGAAAA", "TACG", "ACG", "CGTAC", "ATATAT", "",
            "CCACGTAA", "ACGCGT",
        ] {
            reads.push_str(row).unwrap();
        }

        let mut dbb = DbBuilder::new();
        let id = dbb.add_dna2_table(reads).unwrap();
        (dbb.freeze(), id)
    }
}
