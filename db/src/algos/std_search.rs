use crate::like::{LiteralAlgorithm, RowLiteralSearch};
use crate::storage::utf8::{Utf8Column, Utf8Row};

use super::naive::naive_find_scalar;
use super::utf8_shared::{
    ByteNeedle, byte_index_symbols, byte_literal_len, compile_byte_literal, matches_at_bytes,
    utf8_row_len,
};

/// Uses Rust's standard `str::find` when the row and start offset allow it.
///
/// `Utf8Column` has byte-wise LIKE semantics in this crate. Therefore this
/// implementation falls back to byte search when the row bytes are not valid
/// UTF-8 or when `from` is not a UTF-8 character boundary. This keeps the
/// algorithm correct for byte-oriented LIKE while still benchmarking `str::find`
/// in the common valid-UTF-8 path.
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

        let Ok(text) = std::str::from_utf8(bytes) else {
            return naive_find_scalar(&bytes[from..], pat).map(|pos| (pos + from) as u32);
        };

        if text.is_char_boundary(from) {
            // SAFETY: `from` is a checked UTF-8 boundary and `needle` came from
            // a Rust `&str` in `compile_literal`, so it is valid UTF-8.
            let needle_str = unsafe { std::str::from_utf8_unchecked(pat) };
            text[from..].find(needle_str).map(|pos| (pos + from) as u32)
        } else {
            // For byte-wise LIKE, a search from the middle of a UTF-8 character
            // is still meaningful. `str::find` cannot start there, so use the
            // byte-correct fallback.
            naive_find_scalar(&bytes[from..], pat).map(|pos| (pos + from) as u32)
        }
    }
}
