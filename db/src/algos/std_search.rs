use crate::like::{LiteralAlgorithm, RowLiteralSearch};
use crate::storage::utf8::{Utf8Column, Utf8Row};

use super::utf8_shared::{
    byte_index_symbols, byte_literal_len, compile_byte_literal, matches_at_bytes, utf8_row_len,
    ByteNeedle,
};

/// Uses Rust's standard `str::find` on unchecked UTF-8 row slices.
///
/// The benchmark data is prepared as valid UTF-8, so this implementation avoids
/// per-query UTF-8 validation and boundary checks.
#[derive(Debug, Clone, Copy, Default)]
pub struct StdSearch;

impl LiteralAlgorithm for StdSearch {
    type Needle = ByteNeedle;
    type State = ();

    const SUPPORTS_UNDERSCORE: bool = false;

    #[inline]
    fn compile_literal(src: &str) -> Option<Self::Needle> {
        compile_byte_literal(src)
    }

    #[inline]
    fn build_state(_needle: &Self::Needle) -> Self::State {
        ()
    }

    #[inline]
    fn literal_len(needle: &Self::Needle) -> u32 {
        byte_literal_len(needle)
    }

    #[inline]
    fn index_symbols(needle: &Self::Needle) -> Option<Box<[u8]>> {
        byte_index_symbols(needle)
    }
}

impl<'db> RowLiteralSearch<Utf8Column<'db>> for StdSearch {
    #[inline]
    fn row_len<'r>(row: &Utf8Row<'r>) -> u32 {
        utf8_row_len(row)
    }

    #[inline(always)]
    fn matches_at<'r>(
        row: &Utf8Row<'r>,
        pos: u32,
        needle: &Self::Needle,
        _state: &Self::State,
    ) -> bool {
        matches_at_bytes(row, pos, needle)
    }

    #[inline]
    fn find_from<'r>(
        row: &Utf8Row<'r>,
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

        // SAFETY: benchmark UTF-8 storage and search offsets are valid UTF-8
        // boundaries in the workloads using StdSearch.
        let text = unsafe { std::str::from_utf8_unchecked(bytes) };
        // SAFETY: `from` is a valid UTF-8 boundary under the benchmark
        // invariant above, and `needle` came from a Rust `&str` in
        // `compile_literal`.
        let suffix = unsafe { text.get_unchecked(from..) };
        let needle_str = unsafe { std::str::from_utf8_unchecked(pat) };
        suffix.find(needle_str).map(|pos| (pos + from) as u32)
    }
}
