//! Decoded-row literal-search adapters for FSST columns.
//!
//! `FsstColumn::row()` returns an owned decoded row. These impls then reuse the
//! same byte-search kernels used by `Utf8Column`. This keeps FSST storage
//! compatible with the existing LIKE verifier and indexes without pretending to
//! be compressed-domain search.

use crate::like::RowLiteralSearch;
use crate::storage::fsst::{FsstColumn, FsstRow};

use super::bm::{BM, bm_find};
use super::kmp::{Utf8Kmp, kmp_find_from};
use super::libc_find::{LibcMemmem, memmem_find};
use super::naive::{
    Naive, NaiveAuto, NaiveAutoWildcard, NaiveAvx2, NaiveAvx2V2, NaiveAvx2V2Wildcard,
    NaiveAvx2Wildcard, NaiveAvx512, NaiveAvx512V2, NaiveAvx512V2Wildcard, NaiveAvx512Wildcard,
    NaiveMixed, NaiveMixedWildcard, NaiveScalar, NaiveScalarWildcard, NaiveVectorized,
    NaiveVectorizedV2, NaiveVectorizedV2Wildcard, NaiveVectorizedWildcard, NaiveWildcard,
    naive_find, naive_find_auto, naive_find_avx2, naive_find_avx2_v2, naive_find_avx512,
    naive_find_avx512_v2, naive_find_mixed, naive_find_scalar, naive_find_vectorized,
    naive_find_vectorized_v2, naive_find_wildcard, naive_find_wildcard_auto,
    naive_find_wildcard_avx2, naive_find_wildcard_avx2_v2, naive_find_wildcard_avx512,
    naive_find_wildcard_avx512_v2, naive_find_wildcard_mixed, naive_find_wildcard_scalar,
    naive_find_wildcard_vectorized, naive_find_wildcard_vectorized_v2,
};
use super::std_search::StdSearch;
use super::two_way::{TwoWay, two_way_find};
use super::two_way2::{TwoWay2, two_way2_find};
use super::utf8_shared::{
    ByteNeedle, ByteWildcardNeedle, bytes_match_wildcard_same_len, eq_at_bytes,
};

#[inline(always)]
fn fsst_row_len(row: &FsstRow) -> u32 {
    row.logical_len()
}

#[inline(always)]
fn matches_at_decoded_bytes(row: &FsstRow, pos: u32, needle: &ByteNeedle) -> bool {
    eq_at_bytes(row.bytes(), pos, needle.bytes())
}

#[inline(always)]
fn matches_at_decoded_wildcard(row: &FsstRow, pos: u32, needle: &ByteWildcardNeedle) -> bool {
    if !needle.has_wildcard() {
        return eq_at_bytes(row.bytes(), pos, needle.bytes());
    }

    let pos = pos as usize;
    let len = needle.bytes().len();
    let Some(end) = pos.checked_add(len) else {
        return false;
    };
    let Some(candidate) = row.bytes().get(pos..end) else {
        return false;
    };
    bytes_match_wildcard_same_len(candidate, needle.bytes())
}

macro_rules! impl_fsst_exact_no_state {
    ($ty:ty, $find:path) => {
        impl<'db> RowLiteralSearch<FsstColumn<'db>> for $ty {
            #[inline]
            fn row_len<'r>(row: &FsstRow) -> u32 {
                fsst_row_len(row)
            }

            #[inline(always)]
            fn matches_at<'r>(
                row: &FsstRow,
                pos: u32,
                needle: &Self::Needle,
                _state: &Self::State,
            ) -> bool {
                matches_at_decoded_bytes(row, pos, needle)
            }

            #[inline]
            fn find_from<'r>(
                row: &FsstRow,
                from: u32,
                needle: &Self::Needle,
                _state: &Self::State,
            ) -> Option<u32> {
                let text = row.bytes();
                let from = from as usize;
                if from > text.len() {
                    return None;
                }
                $find(&text[from..], needle.bytes()).map(|pos| (pos + from) as u32)
            }
        }
    };
}

macro_rules! impl_fsst_exact_with_state {
    ($ty:ty, $find:path) => {
        impl<'db> RowLiteralSearch<FsstColumn<'db>> for $ty {
            #[inline]
            fn row_len<'r>(row: &FsstRow) -> u32 {
                fsst_row_len(row)
            }

            #[inline(always)]
            fn matches_at<'r>(
                row: &FsstRow,
                pos: u32,
                needle: &Self::Needle,
                _state: &Self::State,
            ) -> bool {
                matches_at_decoded_bytes(row, pos, needle)
            }

            #[inline]
            fn find_from<'r>(
                row: &FsstRow,
                from: u32,
                needle: &Self::Needle,
                state: &Self::State,
            ) -> Option<u32> {
                let text = row.bytes();
                let from = from as usize;
                if from > text.len() {
                    return None;
                }
                $find(&text[from..], needle.bytes(), state).map(|pos| (pos + from) as u32)
            }
        }
    };
}

macro_rules! impl_fsst_wildcard {
    ($ty:ty, $exact_find:path, $wild_find:path) => {
        impl<'db> RowLiteralSearch<FsstColumn<'db>> for $ty {
            #[inline]
            fn row_len<'r>(row: &FsstRow) -> u32 {
                fsst_row_len(row)
            }

            #[inline(always)]
            fn matches_at<'r>(
                row: &FsstRow,
                pos: u32,
                needle: &Self::Needle,
                _state: &Self::State,
            ) -> bool {
                matches_at_decoded_wildcard(row, pos, needle)
            }

            #[inline]
            fn find_from<'r>(
                row: &FsstRow,
                from: u32,
                needle: &Self::Needle,
                state: &Self::State,
            ) -> Option<u32> {
                let text = row.bytes();
                let from = from as usize;
                if from > text.len() {
                    return None;
                }
                if needle.has_wildcard() {
                    $wild_find(&text[from..], needle.bytes(), state).map(|pos| (pos + from) as u32)
                } else {
                    $exact_find(&text[from..], needle.bytes()).map(|pos| (pos + from) as u32)
                }
            }
        }
    };
}

impl<'db> RowLiteralSearch<FsstColumn<'db>> for Utf8Kmp {
    #[inline]
    fn row_len<'r>(row: &FsstRow) -> u32 {
        fsst_row_len(row)
    }

    #[inline(always)]
    fn matches_at<'r>(
        row: &FsstRow,
        pos: u32,
        needle: &Self::Needle,
        _state: &Self::State,
    ) -> bool {
        matches_at_decoded_bytes(row, pos, needle)
    }

    #[inline]
    fn find_from<'r>(
        row: &FsstRow,
        from: u32,
        needle: &Self::Needle,
        state: &Self::State,
    ) -> Option<u32> {
        kmp_find_from(row.bytes(), needle.bytes(), state, from as usize).map(|pos| pos as u32)
    }
}

impl<'db> RowLiteralSearch<FsstColumn<'db>> for StdSearch {
    #[inline]
    fn row_len<'r>(row: &FsstRow) -> u32 {
        fsst_row_len(row)
    }

    #[inline(always)]
    fn matches_at<'r>(
        row: &FsstRow,
        pos: u32,
        needle: &Self::Needle,
        _state: &Self::State,
    ) -> bool {
        matches_at_decoded_bytes(row, pos, needle)
    }

    #[inline]
    fn find_from<'r>(
        row: &FsstRow,
        from: u32,
        needle: &Self::Needle,
        _state: &Self::State,
    ) -> Option<u32> {
        let bytes = row.bytes();
        let pat = needle.bytes();
        let from = from as usize;
        if from > bytes.len() {
            return None;
        }
        if pat.is_empty() {
            return Some(from as u32);
        }
        if pat.len() > bytes.len().saturating_sub(from) {
            return None;
        }

        let Ok(text) = std::str::from_utf8(bytes) else {
            return naive_find_scalar(&bytes[from..], pat).map(|pos| (pos + from) as u32);
        };

        if text.is_char_boundary(from) {
            let needle_str = unsafe { std::str::from_utf8_unchecked(pat) };
            text[from..].find(needle_str).map(|pos| (pos + from) as u32)
        } else {
            naive_find_scalar(&bytes[from..], pat).map(|pos| (pos + from) as u32)
        }
    }
}

impl_fsst_exact_no_state!(Naive, naive_find);
impl_fsst_exact_no_state!(NaiveScalar, naive_find_scalar);
impl_fsst_exact_no_state!(NaiveVectorized, naive_find_vectorized);
impl_fsst_exact_no_state!(NaiveVectorizedV2, naive_find_vectorized_v2);
impl_fsst_exact_no_state!(NaiveAvx2, naive_find_avx2);
impl_fsst_exact_no_state!(NaiveAvx2V2, naive_find_avx2_v2);
impl_fsst_exact_no_state!(NaiveAvx512, naive_find_avx512);
impl_fsst_exact_no_state!(NaiveAvx512V2, naive_find_avx512_v2);
impl_fsst_exact_no_state!(NaiveAuto, naive_find_auto);
impl_fsst_exact_no_state!(NaiveMixed, naive_find_mixed);

impl_fsst_exact_with_state!(BM, bm_find);
impl_fsst_exact_with_state!(TwoWay, two_way_find);
impl_fsst_exact_with_state!(TwoWay2, two_way2_find);
impl_fsst_exact_no_state!(LibcMemmem, memmem_find);

impl_fsst_wildcard!(NaiveWildcard, naive_find, naive_find_wildcard);
impl_fsst_wildcard!(
    NaiveScalarWildcard,
    naive_find_scalar,
    naive_find_wildcard_scalar
);
impl_fsst_wildcard!(
    NaiveVectorizedWildcard,
    naive_find_vectorized,
    naive_find_wildcard_vectorized
);
impl_fsst_wildcard!(
    NaiveVectorizedV2Wildcard,
    naive_find_vectorized_v2,
    naive_find_wildcard_vectorized_v2
);
impl_fsst_wildcard!(NaiveAvx2Wildcard, naive_find_avx2, naive_find_wildcard_avx2);
impl_fsst_wildcard!(
    NaiveAvx2V2Wildcard,
    naive_find_avx2_v2,
    naive_find_wildcard_avx2_v2
);
impl_fsst_wildcard!(
    NaiveAvx512Wildcard,
    naive_find_avx512,
    naive_find_wildcard_avx512
);
impl_fsst_wildcard!(
    NaiveAvx512V2Wildcard,
    naive_find_avx512_v2,
    naive_find_wildcard_avx512_v2
);
impl_fsst_wildcard!(NaiveAutoWildcard, naive_find_auto, naive_find_wildcard_auto);
impl_fsst_wildcard!(
    NaiveMixedWildcard,
    naive_find_mixed,
    naive_find_wildcard_mixed
);
