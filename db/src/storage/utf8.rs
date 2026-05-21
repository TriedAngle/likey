//! Dense UTF-8/byte string table.
//!
//! LIKE semantics for this type are byte semantics: `_` means one byte and `%`
//! means any byte sequence. If you want Unicode scalar or grapheme semantics,
//! add a separate storage type so the benchmark remains explicit.

use crate::RowId;
use crate::arena::{ArenaBuilder, FrozenArena, RelSlice};
use crate::storage::Column;

#[derive(Debug, Clone)]
pub struct Utf8TableDesc {
    pub name: String,
    pub text: Utf8ColumnDesc,
}

#[derive(Debug, Clone)]
pub struct Utf8ColumnDesc {
    pub row_count: RowId,
    /// `row_count + 1` byte offsets into `payload`.
    pub offsets: RelSlice<u64>,
    /// Byte length per row. Stored separately to make length filters cheap.
    pub logical_lens: RelSlice<u32>,
    /// Concatenated UTF-8/byte payload.
    pub payload: RelSlice<u8>,
}

#[derive(Clone, Copy)]
pub struct Utf8Table<'a> {
    arena: &'a FrozenArena,
    desc: &'a Utf8TableDesc,
}

impl<'a> Utf8Table<'a> {
    pub(crate) fn new(arena: &'a FrozenArena, desc: &'a Utf8TableDesc) -> Self {
        Self { arena, desc }
    }

    pub fn name(&self) -> &str {
        &self.desc.name
    }

    pub fn row_count(&self) -> RowId {
        self.desc.text.row_count
    }

    pub fn text(&self) -> Utf8Column<'a> {
        Utf8Column {
            arena: self.arena,
            desc: &self.desc.text,
        }
    }

    pub fn column_count(&self) -> usize {
        1
    }

    pub fn row(&self, row: RowId) -> Utf8RowEntry<'a> {
        Utf8RowEntry {
            id: row,
            text: self.text().row_view(row),
        }
    }
}

impl std::fmt::Debug for Utf8Table<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Utf8Table")
            .field("name", &self.name())
            .field("row_count", &self.row_count())
            .finish()
    }
}

#[derive(Clone, Copy)]
pub struct Utf8Column<'a> {
    arena: &'a FrozenArena,
    desc: &'a Utf8ColumnDesc,
}

impl<'a> Utf8Column<'a> {
    pub fn desc(&self) -> &'a Utf8ColumnDesc {
        self.desc
    }

    #[inline]
    pub fn offsets(&self) -> &'a [u64] {
        self.arena.slice(self.desc.offsets)
    }

    #[inline]
    pub fn logical_lens(&self) -> &'a [u32] {
        self.arena.slice(self.desc.logical_lens)
    }

    #[inline]
    pub fn payload(&self) -> &'a [u8] {
        self.arena.bytes(self.desc.payload)
    }

    #[inline]
    pub fn row_bytes(&self, row: RowId) -> &'a [u8] {
        assert!(row < self.desc.row_count, "row out of bounds");
        let offsets = self.offsets();
        let payload = self.payload();
        let start = offsets[row as usize] as usize;
        let end = offsets[row as usize + 1] as usize;
        &payload[start..end]
    }

    #[inline]
    pub fn row_str(&self, row: RowId) -> Result<&'a str, std::str::Utf8Error> {
        std::str::from_utf8(self.row_bytes(row))
    }

    #[inline]
    pub fn row_view(&self, row: RowId) -> Utf8Row<'a> {
        Utf8Row {
            bytes: self.row_bytes(row),
        }
    }
}

impl std::fmt::Debug for Utf8Column<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Utf8Column")
            .field("row_count", &self.row_count())
            .field("payload_bytes", &self.payload().len())
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Utf8RowEntry<'a> {
    pub id: RowId,
    pub text: Utf8Row<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct Utf8Row<'a> {
    bytes: &'a [u8],
}

impl<'a> Utf8Row<'a> {
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub fn as_str(&self) -> Result<&'a str, std::str::Utf8Error> {
        std::str::from_utf8(self.bytes)
    }

    pub fn logical_len(&self) -> u32 {
        self.bytes.len() as u32
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl<'a> Column for Utf8Column<'a> {
    type Row<'r>
        = Utf8Row<'r>
    where
        Self: 'r;
    type Symbol = u8;
    type SymbolIter<'r>
        = std::iter::Copied<std::slice::Iter<'r, u8>>
    where
        Self: 'r;

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
        self.row_bytes(row).iter().copied()
    }
}

#[derive(Debug, Default)]
pub struct Utf8ColumnBuilder {
    offsets: Vec<u64>,
    logical_lens: Vec<u32>,
    payload: Vec<u8>,
}

impl Utf8ColumnBuilder {
    pub fn new() -> Self {
        Self {
            offsets: vec![0],
            logical_lens: Vec::new(),
            payload: Vec::new(),
        }
    }

    pub fn with_capacity(rows: usize, payload_bytes: usize) -> Self {
        let mut offsets = Vec::with_capacity(rows.saturating_add(1));
        offsets.push(0);
        Self {
            offsets,
            logical_lens: Vec::with_capacity(rows),
            payload: Vec::with_capacity(payload_bytes),
        }
    }

    pub fn push_str(&mut self, value: &str) {
        self.push_bytes(value.as_bytes())
    }

    pub fn push_bytes(&mut self, value: &[u8]) {
        let len = u32::try_from(value.len()).expect("row exceeds u32 logical length");
        self.logical_lens.push(len);
        self.payload.extend_from_slice(value);
        self.offsets.push(self.payload.len() as u64);
    }

    pub fn row_count(&self) -> RowId {
        self.logical_lens.len() as RowId
    }

    pub fn payload_len(&self) -> usize {
        self.payload.len()
    }

    pub fn is_empty(&self) -> bool {
        self.row_count() == 0
    }

    pub fn finish(self, arena: &mut ArenaBuilder) -> Utf8ColumnDesc {
        let row_count = self.row_count();
        debug_assert_eq!(self.offsets.len(), row_count as usize + 1);
        debug_assert_eq!(self.logical_lens.len(), row_count as usize);

        let offsets = arena.append_slice(&self.offsets);
        let logical_lens = arena.append_slice(&self.logical_lens);
        // 64-byte alignment is not required for correctness, but it gives SIMD
        // implementations a friendly starting point.
        let payload = arena.append_bytes_aligned(&self.payload, 64);

        Utf8ColumnDesc {
            row_count,
            offsets,
            logical_lens,
            payload,
        }
    }
}

#[derive(Debug)]
pub struct Utf8TableBuilder {
    name: String,
    text: Utf8ColumnBuilder,
}

impl Utf8TableBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            text: Utf8ColumnBuilder::new(),
        }
    }

    pub fn with_capacity(name: impl Into<String>, rows: usize, payload_bytes: usize) -> Self {
        Self {
            name: name.into(),
            text: Utf8ColumnBuilder::with_capacity(rows, payload_bytes),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn push_str(&mut self, value: &str) {
        self.text.push_str(value)
    }

    pub fn push_bytes(&mut self, value: &[u8]) {
        self.text.push_bytes(value)
    }

    pub fn push_row(&mut self, value: &str) -> RowId {
        let row = self.row_count();
        self.push_str(value);
        row
    }

    pub fn row_count(&self) -> RowId {
        self.text.row_count()
    }

    pub fn text_builder(&self) -> &Utf8ColumnBuilder {
        &self.text
    }

    pub fn text_builder_mut(&mut self) -> &mut Utf8ColumnBuilder {
        &mut self.text
    }

    pub fn finish(self, arena: &mut ArenaBuilder) -> Utf8TableDesc {
        Utf8TableDesc {
            name: self.name,
            text: self.text.finish(arena),
        }
    }
}
