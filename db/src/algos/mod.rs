//! Literal-search algorithms that plug into `LikePattern<A>`.
//!
//! These are not separate LIKE matchers. They are literal backends used by the
//! generic LIKE verifier. All exact UTF-8 byte algorithms share the same
//! anchored equality primitive in `utf8_shared`; they differ only in
//! unanchored `find_from`.

pub mod utf8_shared;

pub mod bm;
pub mod dna2_naive;
pub mod kmp;
pub mod libc_find;
pub mod naive;
pub mod std_search;
pub mod two_way;
pub mod two_way2;

pub use bm::{BM, BMState, bm_find};
pub use dna2_naive::{Dna2NaiveWildcard, Dna2Needle};
pub use kmp::Utf8Kmp;
pub use libc_find::{LibcMemmem, memmem_find};
pub use naive::{
    Naive, NaiveMixed, NaiveScalar, NaiveVectorized, NaiveVectorizedV2, naive_find,
    naive_find_mixed, naive_find_scalar, naive_find_vectorized, naive_find_vectorized_v2,
};
pub use std_search::StdSearch;
pub use two_way::{TwoWay, TwoWayState, two_way_find};
pub use two_way2::{TwoWay2, TwoWay2State, two_way2_find};
pub use utf8_shared::{ByteNeedle, bytes_eq_same_len, eq_at_bytes, matches_at_bytes};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Column;
    use crate::DbBuilder;
    use crate::RowId;
    use crate::{FullScan, QueryScratch, execute_like};
    use crate::{LikeCompileOptions, LikePattern, LiteralAlgorithm, RowLiteralSearch};
    use crate::{Utf8Column, Utf8TableBuilder};

    use super::utf8_shared::expected_find_from;

    macro_rules! utf8_algo_suites {
        ($($test_name:ident => $algo:ty),* $(,)?) => {
            $(
                #[test]
                fn $test_name() {
                    literal_search_suite::<$algo>();
                    like_integration_suite::<$algo>();
                }
            )*
        };
    }

    utf8_algo_suites! {
        utf8_std_search_suite => StdSearch,
        utf8_kmp_suite => Utf8Kmp,
        utf8_naive_suite => Naive,
        utf8_naive_scalar_suite => NaiveScalar,
        utf8_naive_vectorized_suite => NaiveVectorized,
        utf8_naive_vectorized_v2_suite => NaiveVectorizedV2,
        utf8_naive_mixed_suite => NaiveMixed,
        utf8_bm_suite => BM,
        utf8_two_way_suite => TwoWay,
        utf8_two_way2_suite => TwoWay2,
        utf8_libc_memmem_suite => LibcMemmem,
    }

    fn one_row_column(text: &str) -> (crate::db::Db, crate::TableId) {
        let mut table = Utf8TableBuilder::new("t");
        table.push_str(text);
        let mut dbb = DbBuilder::new();
        let id = dbb.add_utf8_table(table).expect("table should build");
        (dbb.freeze(), id)
    }

    fn literal_search_suite<A>()
    where
        A: LiteralAlgorithm<Needle = ByteNeedle>,
        for<'db> A: RowLiteralSearch<Utf8Column<'db>>,
    {
        let cases: &[(&str, &str)] = &[
            ("", ""),
            ("abc", ""),
            ("", "a"),
            ("abc", "a"),
            ("abc", "b"),
            ("abc", "c"),
            ("abc", "d"),
            ("ababcabcabababd", "ababd"),
            ("hello world", "world"),
            ("hello world", "rust"),
            ("aaaaaa", "aaa"),
            ("aaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaa", "aaab"),
            ("ACGTACGTACGT", "CGTA"),
            ("🌍hello🌍hello", "🌍hello"),
            ("prefix-middle-suffix", "middle"),
        ];

        for &(text, pat) in cases {
            let (db, id) = one_row_column(text);
            let table = db.utf8_table(id).unwrap();
            let col = table.text();
            let row = col.row(0);
            let needle = A::compile_literal(pat).expect("valid literal");
            let state = A::build_state(&needle);

            for from in 0..=text.len() + 1 {
                let got = A::find_from(&row, from as u32, &needle, &state).map(|x| x as usize);
                let expect = expected_find_from(text.as_bytes(), pat.as_bytes(), from);
                assert_eq!(
                    got,
                    expect,
                    "find_from mismatch: algo={}, text={text:?}, pat={pat:?}, from={from}",
                    core::any::type_name::<A>()
                );
            }

            for pos in 0..=text.len() + 1 {
                let got = A::matches_at(&row, pos as u32, &needle, &state);
                let expect = text
                    .as_bytes()
                    .get(pos..pos.saturating_add(pat.len()))
                    .map_or(false, |s| s == pat.as_bytes());
                assert_eq!(
                    got,
                    expect,
                    "matches_at mismatch: algo={}, text={text:?}, pat={pat:?}, pos={pos}",
                    core::any::type_name::<A>()
                );
            }
        }

        random_literal_search_suite::<A>();
    }

    fn random_literal_search_suite<A>()
    where
        A: LiteralAlgorithm<Needle = ByteNeedle>,
        for<'db> A: RowLiteralSearch<Utf8Column<'db>>,
    {
        fn next(seed: &mut u64) -> u64 {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *seed
        }

        let mut seed = 1u64;
        for case_idx in 0..5_000 {
            let text_len = (next(&mut seed) % 192) as usize;
            let pat_len = (next(&mut seed) % 48) as usize;
            let from = if text_len == 0 {
                0usize
            } else {
                (next(&mut seed) % (text_len as u64 + 2)) as usize
            };

            let mut text = Vec::with_capacity(text_len);
            let mut pat = Vec::with_capacity(pat_len);
            for _ in 0..text_len {
                text.push(b'a' + (next(&mut seed) % 26) as u8);
            }
            for _ in 0..pat_len {
                pat.push(b'a' + (next(&mut seed) % 26) as u8);
            }

            let text_s = std::str::from_utf8(&text).unwrap();
            let pat_s = std::str::from_utf8(&pat).unwrap();

            let (db, id) = one_row_column(text_s);
            let table = db.utf8_table(id).unwrap();
            let col = table.text();
            let row = col.row(0);
            let needle = A::compile_literal(pat_s).expect("valid literal");
            let state = A::build_state(&needle);

            let got = A::find_from(&row, from as u32, &needle, &state).map(|x| x as usize);
            let expect = expected_find_from(&text, &pat, from);
            assert_eq!(
                got,
                expect,
                "random find mismatch: algo={}, case={case_idx}, text={text_s:?}, pat={pat_s:?}, from={from}",
                core::any::type_name::<A>()
            );
        }
    }

    fn like_integration_suite<A>()
    where
        A: LiteralAlgorithm<Needle = ByteNeedle>,
        for<'db> A: RowLiteralSearch<Utf8Column<'db>>,
    {
        let rows = [
            "hello",
            "hello world",
            "world hello",
            "hxllo",
            "heLLo",
            "banana",
            "bandana",
            "",
        ];

        let mut table = Utf8TableBuilder::new("docs");
        for row in rows {
            table.push_str(row);
        }
        let mut dbb = DbBuilder::new();
        let id = dbb.add_utf8_table(table).unwrap();
        let db = dbb.freeze();
        let table = db.utf8_table(id).unwrap();
        let col = table.text();

        let patterns_and_expected: &[(&str, &[RowId])] = &[
            ("hello", &[0]),
            ("hello%", &[0, 1]),
            ("%hello", &[0, 2]),
            ("%ell%", &[0, 1, 2]),
            ("h_llo", &[0, 3]),
            ("%", &[0, 1, 2, 3, 4, 5, 6, 7]),
            ("", &[7]),
            ("%ana%", &[5, 6]),
            ("b%a", &[5, 6]),
        ];

        for &(pattern, expected) in patterns_and_expected {
            let like = LikePattern::<A>::compile_with_options(
                pattern,
                LikeCompileOptions {
                    pass_underscore_to_algorithm: false,
                },
            )
            .expect("pattern should compile");

            let mut scan = FullScan::new(col.row_count(), 16);
            let mut scratch = QueryScratch::default();
            let mut matches = Vec::<RowId>::new();
            execute_like(&col, &mut scan, &like, &mut scratch, &mut matches);

            assert_eq!(
                matches.as_slice(),
                expected,
                "LIKE mismatch: algo={}, pattern={pattern:?}",
                core::any::type_name::<A>()
            );
        }
    }

    #[test]
    fn dna2_wildcard_like_suite() {
        use crate::storage::dna2::Dna2TableBuilder;

        let mut reads = Dna2TableBuilder::new("reads");
        reads.push_str("ACGT").unwrap();
        reads.push_str("AGGT").unwrap();
        reads.push_str("TTTT").unwrap();
        reads.push_str("AACGAAAA").unwrap();

        let mut dbb = DbBuilder::new();
        let id = dbb.add_dna2_table(reads).unwrap();
        let db = dbb.freeze();
        let table = db.dna2_table(id).unwrap();
        let col = table.sequence();

        let like = LikePattern::<Dna2NaiveWildcard>::compile("A_G%").unwrap();
        let mut scan = FullScan::new(col.row_count(), 16);
        let mut scratch = QueryScratch::default();
        let mut matches = Vec::<RowId>::new();
        execute_like(&col, &mut scan, &like, &mut scratch, &mut matches);
        assert_eq!(matches, vec![0, 1]);
    }
}
