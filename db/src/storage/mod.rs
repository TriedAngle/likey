//! Storage abstractions and concrete dense column implementations.
//!
//! The central abstraction exposes concrete row views for matchers plus a plain
//! byte stream for generic candidate indexes. UTF-8 columns expose row bytes,
//! FSST columns expose decoded bytes, and DNA2 columns expose ASCII DNA bytes
//! (`A/C/G/T/N`) rather than packed base codes.

use crate::RowId;

pub mod dna2;
pub mod fsst;
pub mod utf8;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// Retained byte accounting for dense column storage.
///
/// This counts the main owned arrays that make up the frozen column. It does
/// not include allocator bookkeeping or opaque third-party state that cannot be
/// inspected directly.
pub struct ColumnStorageSize {
    pub offsets_bytes: usize,
    pub logical_lens_bytes: usize,
    pub payload_bytes: usize,
    pub codec_bytes: usize,
}

impl ColumnStorageSize {
    #[inline]
    pub const fn total_bytes(self) -> usize {
        self.offsets_bytes + self.logical_lens_bytes + self.payload_bytes + self.codec_bytes
    }
}

/// Dense logical-symbol column.
///
/// This trait deliberately does not expose `get_string(row) -> String`. A row
/// can be viewed in the concrete representation, and generic indexes can stream
/// logical symbols.
pub trait Column {
    type Row<'r>
    where
        Self: 'r;

    type Symbol: Copy + Eq + Ord + std::hash::Hash + 'static;

    type SymbolIter<'r>: Iterator<Item = Self::Symbol>
    where
        Self: 'r;

    fn row_count(&self) -> RowId;

    /// Logical length in this column's semantics.
    ///
    /// UTF-8 byte column: bytes. FSST column: decoded bytes. DNA2 column: bases.
    ///
    /// Internal callers are expected to pass valid row IDs. Implementations may
    /// use debug assertions to catch invalid IDs during testing while keeping the
    /// release hot path unchecked.
    fn logical_len(&self, row: RowId) -> u32;

    /// Borrow a row in its concrete representation.
    ///
    /// Internal callers are expected to pass valid row IDs. Implementations may
    /// use debug assertions to catch invalid IDs during testing while keeping the
    /// release hot path unchecked.
    fn row(&self, row: RowId) -> Self::Row<'_>;

    /// Iterate index bytes of one row.
    ///
    /// This is intended for generic candidate-index construction. Optimized
    /// algorithms should use the concrete row/column methods instead.
    fn symbols(&self, row: RowId) -> Self::SymbolIter<'_>;

    fn is_empty(&self) -> bool {
        self.row_count() == 0
    }
}
