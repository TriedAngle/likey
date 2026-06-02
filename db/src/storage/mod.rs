//! Storage abstractions and concrete dense column implementations.
//!
//! The central abstraction is a dense logical-symbol column. UTF-8 byte columns
//! expose symbols as bytes. FSST columns expose decoded bytes. DNA2 columns
//! expose symbols as base codes 0..=3.
//! Generic indexes can be written over `Column<Symbol = u8>`.

use crate::RowId;

pub mod dna2;
pub mod fsst;
pub mod utf8;

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

    /// Iterate logical symbols of one row.
    ///
    /// This is intended for generic index construction. Optimized algorithms can
    /// use the concrete row/column methods instead.
    fn symbols(&self, row: RowId) -> Self::SymbolIter<'_>;

    fn is_empty(&self) -> bool {
        self.row_count() == 0
    }
}
