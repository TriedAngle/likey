//! Dense FSST-compressed byte string table.
//!
//! Rows are stored compressed with one FSST symbol table for the whole column.
//! Logical symbols are still uncompressed bytes, so generic indexes such as the
//! FM-index and trigram index remain correct: they decode rows during index
//! construction. The row-level LIKE fallback also decodes one candidate row into
//! an owned `FsstRow` view before applying a byte literal algorithm.
//!
//! This is intentionally a correctness-first compressed storage type. It does
//! not yet perform compressed-domain LIKE matching.

use std::fmt;

use fsst::{Compressor, CompressorBuilder};

use crate::arena::{ArenaBuilder, FrozenArena, RelSlice};
use crate::storage::Column;
use crate::RowId;

#[derive(Clone)]
/// FSST compressor plus a compact symbol-table snapshot.
///
/// The compressor is used for row decompression. `symbol_words` and
/// `symbol_lens` expose the trained symbol table for diagnostics or future
/// persistence work.
pub struct FsstCodec {
    compressor: Compressor,
    symbol_words: Box<[u64]>,
    symbol_lens: Box<[u8]>,
}

impl FsstCodec {
    /// Capture an FSST compressor and its symbol table.
    pub fn from_compressor(compressor: Compressor) -> Self {
        let symbol_words = compressor
            .symbol_table()
            .iter()
            .map(|symbol| symbol.to_u64())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let symbol_lens = compressor.symbol_lengths().to_vec().into_boxed_slice();
        Self {
            compressor,
            symbol_words,
            symbol_lens,
        }
    }

    #[inline]
    /// Compressor used to decompress stored rows.
    pub fn compressor(&self) -> &Compressor {
        &self.compressor
    }

    #[inline]
    /// Symbol words from the compressor's trained symbol table.
    pub fn symbol_words(&self) -> &[u64] {
        &self.symbol_words
    }

    #[inline]
    /// Byte length of each symbol in the trained symbol table.
    pub fn symbol_lens(&self) -> &[u8] {
        &self.symbol_lens
    }

    #[inline]
    /// Number of symbols in the trained symbol table.
    pub fn symbol_count(&self) -> usize {
        self.symbol_lens.len()
    }
}

impl fmt::Debug for FsstCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FsstCodec")
            .field("symbol_count", &self.symbol_count())
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
/// Frozen FSST table descriptor stored inside [`Db`](crate::Db).
pub struct FsstTableDesc {
    pub name: String,
    pub text: FsstColumnDesc,
    pub codec: FsstCodec,
}

impl fmt::Debug for FsstTableDesc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FsstTableDesc")
            .field("name", &self.name)
            .field("text", &self.text)
            .field("codec", &self.codec)
            .finish()
    }
}

#[derive(Debug, Clone)]
/// Frozen descriptor for one FSST-compressed byte column.
pub struct FsstColumnDesc {
    pub row_count: RowId,
    /// `row_count + 1` byte offsets into the compressed payload.
    pub offsets: RelSlice<u64>,
    /// Uncompressed byte length per row. Stored separately for length filters.
    pub logical_lens: RelSlice<u32>,
    /// Concatenated FSST-compressed row payload.
    pub compressed_payload: RelSlice<u8>,
    /// Sum of uncompressed row lengths.
    pub uncompressed_bytes: u64,
}

#[derive(Clone, Copy)]
/// Borrowed view of an FSST-compressed table.
pub struct FsstTable<'a> {
    arena: &'a FrozenArena,
    desc: &'a FsstTableDesc,
}

impl<'a> FsstTable<'a> {
    pub(crate) fn new(arena: &'a FrozenArena, desc: &'a FsstTableDesc) -> Self {
        Self { arena, desc }
    }

    #[inline]
    /// Table name.
    pub fn name(&self) -> &str {
        &self.desc.name
    }

    #[inline]
    /// Number of rows in the table.
    pub fn row_count(&self) -> RowId {
        self.desc.text.row_count
    }

    #[inline]
    /// The table's single compressed byte column.
    pub fn text(&self) -> FsstColumn<'a> {
        FsstColumn {
            arena: self.arena,
            desc: &self.desc.text,
            codec: &self.desc.codec,
        }
    }

    /// Number of logical columns in this table.
    pub fn column_count(&self) -> usize {
        1
    }

    /// Borrow and decode one row.
    pub fn row(&self, row: RowId) -> FsstRowEntry {
        FsstRowEntry {
            id: row,
            text: self.text().row_view(row),
        }
    }
}

impl fmt::Debug for FsstTable<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FsstTable")
            .field("name", &self.name())
            .field("row_count", &self.row_count())
            .field("compressed_bytes", &self.text().compressed_payload().len())
            .field("uncompressed_bytes", &self.text().uncompressed_bytes())
            .finish()
    }
}

#[derive(Clone, Copy)]
/// Borrowed view of one FSST-compressed byte column.
///
/// Row access decodes into an owned [`FsstRow`]. Generic symbol iteration also
/// decodes, so indexes see the same byte stream as UTF-8/byte storage.
pub struct FsstColumn<'a> {
    arena: &'a FrozenArena,
    desc: &'a FsstColumnDesc,
    codec: &'a FsstCodec,
}

impl<'a> FsstColumn<'a> {
    #[inline]
    /// Frozen column descriptor.
    pub fn desc(&self) -> &'a FsstColumnDesc {
        self.desc
    }

    #[inline]
    /// Codec used to decode compressed rows.
    pub fn codec(&self) -> &'a FsstCodec {
        self.codec
    }

    #[inline]
    /// Row-start offsets into the compressed payload.
    pub fn offsets(&self) -> &'a [u64] {
        self.arena.slice(self.desc.offsets)
    }

    #[inline]
    /// Decoded byte length per row.
    pub fn logical_lens(&self) -> &'a [u32] {
        self.arena.slice(self.desc.logical_lens)
    }

    #[inline]
    /// Concatenated compressed row payload.
    pub fn compressed_payload(&self) -> &'a [u8] {
        self.arena.bytes(self.desc.compressed_payload)
    }

    #[inline]
    /// Total decoded bytes across all rows.
    pub fn uncompressed_bytes(&self) -> u64 {
        self.desc.uncompressed_bytes
    }

    #[inline]
    /// Total compressed payload bytes.
    pub fn compressed_bytes(&self) -> usize {
        self.compressed_payload().len()
    }

    #[inline]
    /// Compressed-size to decoded-size ratio, or `None` for an empty column.
    pub fn compression_ratio(&self) -> Option<f64> {
        if self.desc.uncompressed_bytes == 0 {
            None
        } else {
            Some(self.compressed_bytes() as f64 / self.desc.uncompressed_bytes as f64)
        }
    }

    #[inline]
    /// Compressed bytes for one row.
    pub fn row_compressed_bytes(&self, row: RowId) -> &'a [u8] {
        assert!(row < self.desc.row_count, "row out of bounds");
        let offsets = self.offsets();
        let payload = self.compressed_payload();
        let start = offsets[row as usize] as usize;
        let end = offsets[row as usize + 1] as usize;
        &payload[start..end]
    }

    /// Decode one row into a new byte vector.
    pub fn row_bytes_decoded(&self, row: RowId) -> Vec<u8> {
        let compressed = self.row_compressed_bytes(row);
        let expected_len = self.logical_len(row) as usize;
        if expected_len == 0 {
            return Vec::new();
        }

        let decompressor = self.codec.compressor().decompressor();
        let capacity = decompressor
            .max_decompression_capacity(compressed)
            .max(expected_len.saturating_add(8));
        let mut decoded = Vec::<u8>::with_capacity(capacity);
        let len = decompressor.decompress_into(compressed, decoded.spare_capacity_mut());
        unsafe { decoded.set_len(len) };
        debug_assert_eq!(decoded.len(), expected_len);
        decoded
    }

    /// Decode one row into a caller-provided buffer.
    pub fn copy_row_decoded_to(&self, row: RowId, out: &mut Vec<u8>) {
        out.clear();
        let compressed = self.row_compressed_bytes(row);
        let expected_len = self.logical_len(row) as usize;
        if expected_len == 0 {
            return;
        }

        let decompressor = self.codec.compressor().decompressor();
        let capacity = decompressor
            .max_decompression_capacity(compressed)
            .max(expected_len.saturating_add(8));
        if out.capacity() < capacity {
            out.reserve(capacity.saturating_sub(out.len()));
        }
        let len = decompressor.decompress_into(compressed, out.spare_capacity_mut());
        unsafe { out.set_len(len) };
        debug_assert_eq!(out.len(), expected_len);
    }

    #[inline]
    /// Decode one row into an owned row view.
    pub fn row_view(&self, row: RowId) -> FsstRow {
        FsstRow {
            bytes: self.row_bytes_decoded(row).into_boxed_slice(),
        }
    }
}

impl fmt::Debug for FsstColumn<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FsstColumn")
            .field("row_count", &self.row_count())
            .field("compressed_bytes", &self.compressed_bytes())
            .field("uncompressed_bytes", &self.uncompressed_bytes())
            .field("compression_ratio", &self.compression_ratio())
            .finish()
    }
}

#[derive(Debug, Clone)]
/// Decoded row entry with its physical row ID.
pub struct FsstRowEntry {
    pub id: RowId,
    pub text: FsstRow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Owned decoded FSST row.
pub struct FsstRow {
    bytes: Box<[u8]>,
}

impl FsstRow {
    #[inline]
    /// Decoded row bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[inline]
    /// Interpret the decoded bytes as UTF-8.
    pub fn as_str(&self) -> Result<&str, std::str::Utf8Error> {
        std::str::from_utf8(&self.bytes)
    }

    #[inline]
    /// Decoded byte length.
    pub fn logical_len(&self) -> u32 {
        self.bytes.len() as u32
    }

    #[inline]
    /// Whether the decoded row is empty.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    #[inline]
    /// Consume the row and return decoded bytes.
    pub fn into_bytes(self) -> Box<[u8]> {
        self.bytes
    }
}

impl<'a> Column for FsstColumn<'a> {
    type Row<'r>
        = FsstRow
    where
        Self: 'r;
    type Symbol = u8;
    type SymbolIter<'r>
        = std::vec::IntoIter<u8>
    where
        Self: 'r;

    #[inline]
    fn row_count(&self) -> RowId {
        self.desc.row_count
    }

    #[inline]
    fn logical_len(&self, row: RowId) -> u32 {
        assert!(row < self.desc.row_count, "row out of bounds");
        self.logical_lens()[row as usize]
    }

    #[inline]
    fn row(&self, row: RowId) -> Self::Row<'_> {
        self.row_view(row)
    }

    #[inline]
    fn symbols(&self, row: RowId) -> Self::SymbolIter<'_> {
        self.row_bytes_decoded(row).into_iter()
    }
}

#[derive(Debug, Default)]
/// Builder for one FSST-compressed byte column.
///
/// Rows are buffered uncompressed until `finish`, when a single FSST compressor
/// is trained for the whole column and all rows are compressed into the arena.
pub struct FsstColumnBuilder {
    rows: Vec<Box<[u8]>>,
    uncompressed_bytes: usize,
}

impl FsstColumnBuilder {
    /// Create an empty FSST column builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an FSST column builder with row capacity.
    pub fn with_capacity(rows: usize, _uncompressed_bytes: usize) -> Self {
        Self {
            rows: Vec::with_capacity(rows),
            uncompressed_bytes: 0,
        }
    }

    /// Append a UTF-8 string row.
    pub fn push_str(&mut self, value: &str) {
        self.push_bytes(value.as_bytes())
    }

    /// Append a byte row.
    pub fn push_bytes(&mut self, value: &[u8]) {
        let _ = u32::try_from(value.len()).expect("row exceeds u32 logical length");
        self.uncompressed_bytes = self
            .uncompressed_bytes
            .checked_add(value.len())
            .expect("total FSST uncompressed byte length overflow");
        self.rows.push(value.into());
    }

    /// Number of buffered rows.
    pub fn row_count(&self) -> RowId {
        self.rows.len() as RowId
    }

    /// Total buffered uncompressed bytes.
    pub fn uncompressed_bytes(&self) -> usize {
        self.uncompressed_bytes
    }

    /// Whether no rows have been buffered.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Compress rows into the arena and return the frozen descriptor and codec.
    pub fn finish(self, arena: &mut ArenaBuilder) -> (FsstColumnDesc, FsstCodec) {
        let row_count = self.row_count();
        let compressor = if self.rows.is_empty() || self.uncompressed_bytes == 0 {
            CompressorBuilder::new().build()
        } else {
            let samples: Vec<&[u8]> = self.rows.iter().map(|row| row.as_ref()).collect();
            Compressor::train(&samples)
        };

        let mut offsets = Vec::<u64>::with_capacity(self.rows.len().saturating_add(1));
        let mut logical_lens = Vec::<u32>::with_capacity(self.rows.len());
        let mut compressed_payload = Vec::<u8>::new();
        offsets.push(0);

        for row in &self.rows {
            logical_lens.push(u32::try_from(row.len()).expect("row exceeds u32 logical length"));
            let compressed = compressor.compress(row.as_ref());
            compressed_payload.extend_from_slice(&compressed);
            offsets.push(compressed_payload.len() as u64);
        }

        let offsets = arena.append_slice(&offsets);
        let logical_lens = arena.append_slice(&logical_lens);
        let compressed_payload = arena.append_bytes_aligned(&compressed_payload, 64);
        let codec = FsstCodec::from_compressor(compressor);

        (
            FsstColumnDesc {
                row_count,
                offsets,
                logical_lens,
                compressed_payload,
                uncompressed_bytes: self.uncompressed_bytes as u64,
            },
            codec,
        )
    }
}

#[derive(Debug)]
/// Builder for a single-column FSST table.
pub struct FsstTableBuilder {
    name: String,
    text: FsstColumnBuilder,
}

impl FsstTableBuilder {
    /// Create an empty FSST table builder.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            text: FsstColumnBuilder::new(),
        }
    }

    /// Create an FSST table builder with row capacity.
    pub fn with_capacity(name: impl Into<String>, rows: usize, uncompressed_bytes: usize) -> Self {
        Self {
            name: name.into(),
            text: FsstColumnBuilder::with_capacity(rows, uncompressed_bytes),
        }
    }

    /// Table name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Append a UTF-8 string row to the table's text column.
    pub fn push_str(&mut self, value: &str) {
        self.text.push_str(value)
    }

    /// Append a byte row to the table's text column.
    pub fn push_bytes(&mut self, value: &[u8]) {
        self.text.push_bytes(value)
    }

    /// Number of buffered rows.
    pub fn row_count(&self) -> RowId {
        self.text.row_count()
    }

    /// Total buffered uncompressed bytes.
    pub fn uncompressed_bytes(&self) -> usize {
        self.text.uncompressed_bytes()
    }

    /// Compress rows into the arena and return the frozen table descriptor.
    pub fn finish(self, arena: &mut ArenaBuilder) -> FsstTableDesc {
        let (text, codec) = self.text.finish(arena);
        FsstTableDesc {
            name: self.name,
            text,
            codec,
        }
    }
}
