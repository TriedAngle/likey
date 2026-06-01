//! Dense DNA 2-bit table.
//!
//! Logical symbols are base codes: A=0, C=1, G=2, T=3. Rows are packed into a
//! single continuous 2-bit stream, so there is no per-row byte padding.

use std::fmt;
use std::iter::FusedIterator;

use crate::arena::{ArenaBuilder, FrozenArena, RelSlice};
use crate::storage::Column;
use crate::RowId;

#[derive(Debug, Clone)]
pub struct Dna2TableDesc {
    pub name: String,
    pub sequence: Dna2ColumnDesc,
}

#[derive(Debug, Clone)]
pub struct Dna2ColumnDesc {
    pub row_count: RowId,
    pub total_bases: u64,
    /// `row_count + 1` base offsets into the global packed stream.
    pub base_offsets: RelSlice<u64>,
    /// Base length per row. Stored separately for cheap length filters.
    pub logical_lens: RelSlice<u32>,
    /// Continuous 2-bit stream. Four bases per byte, most significant pair first.
    pub payload: RelSlice<u8>,
}

#[derive(Clone, Copy)]
pub struct Dna2Table<'a> {
    arena: &'a FrozenArena,
    desc: &'a Dna2TableDesc,
}

impl<'a> Dna2Table<'a> {
    pub(crate) fn new(arena: &'a FrozenArena, desc: &'a Dna2TableDesc) -> Self {
        Self { arena, desc }
    }

    pub fn name(&self) -> &str {
        &self.desc.name
    }

    pub fn row_count(&self) -> RowId {
        self.desc.sequence.row_count
    }

    pub fn sequence(&self) -> Dna2Column<'a> {
        Dna2Column {
            arena: self.arena,
            desc: &self.desc.sequence,
        }
    }

    pub fn column_count(&self) -> usize {
        1
    }

    pub fn row(&self, row: RowId) -> Dna2RowEntry<'a> {
        Dna2RowEntry {
            id: row,
            sequence: self.sequence().row_view(row),
        }
    }
}

impl fmt::Debug for Dna2Table<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dna2Table")
            .field("name", &self.name())
            .field("row_count", &self.row_count())
            .finish()
    }
}

#[derive(Clone, Copy)]
pub struct Dna2Column<'a> {
    arena: &'a FrozenArena,
    desc: &'a Dna2ColumnDesc,
}

impl<'a> Dna2Column<'a> {
    pub fn desc(&self) -> &'a Dna2ColumnDesc {
        self.desc
    }

    #[inline]
    pub fn base_offsets(&self) -> &'a [u64] {
        self.arena.slice(self.desc.base_offsets)
    }

    #[inline]
    pub fn logical_lens(&self) -> &'a [u32] {
        self.arena.slice(self.desc.logical_lens)
    }

    #[inline]
    pub fn packed_payload(&self) -> &'a [u8] {
        self.arena.bytes(self.desc.payload)
    }

    pub fn total_bases(&self) -> u64 {
        self.desc.total_bases
    }

    #[inline]
    pub fn base_code_at(&self, row: RowId, local_base_idx: u32) -> u8 {
        self.row_view(row).base_code_at(local_base_idx)
    }

    #[inline]
    pub fn row_view(&self, row: RowId) -> Dna2Row<'a> {
        assert!(row < self.desc.row_count, "row out of bounds");
        let offsets = self.base_offsets();
        Dna2Row {
            payload: self.packed_payload(),
            start_base: offsets[row as usize],
            len: self.logical_lens()[row as usize],
        }
    }

    pub fn row_to_ascii_string(&self, row: RowId) -> String {
        self.row_view(row).to_ascii_string()
    }
}

impl fmt::Debug for Dna2Column<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dna2Column")
            .field("row_count", &self.row_count())
            .field("total_bases", &self.total_bases())
            .field("payload_bytes", &self.packed_payload().len())
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Dna2RowEntry<'a> {
    pub id: RowId,
    pub sequence: Dna2Row<'a>,
}

#[derive(Clone, Copy)]
pub struct Dna2Row<'a> {
    payload: &'a [u8],
    start_base: u64,
    len: u32,
}

impl<'a> Dna2Row<'a> {
    pub fn len_bases(&self) -> u32 {
        self.len
    }

    pub fn logical_len(&self) -> u32 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn start_base_offset(&self) -> u64 {
        self.start_base
    }

    #[inline]
    pub fn packed_payload(&self) -> &'a [u8] {
        self.payload
    }

    #[inline]
    pub fn base_code_at(&self, local_base_idx: u32) -> u8 {
        assert!(local_base_idx < self.len, "base index out of bounds");
        get_base_code(self.payload, self.start_base + u64::from(local_base_idx))
    }

    /// Load up to 32 DNA bases starting at `local_base_idx` as a compact 2-bit
    /// word. The first requested base becomes the highest base within the
    /// returned `2 * bases` low bits. For example, three bases `A,C,G` become
    /// binary `00_01_10`.
    #[inline]
    pub fn load_2bit_window(&self, local_base_idx: u32, bases: u32) -> Option<u64> {
        if bases > 32 {
            return None;
        }
        let end = local_base_idx.checked_add(bases)?;
        if end > self.len {
            return None;
        }
        Some(load_2bit_window_from_payload(
            self.payload,
            self.start_base + u64::from(local_base_idx),
            bases,
        ))
    }

    pub fn iter(&self) -> Dna2Iter<'a> {
        Dna2Iter {
            payload: self.payload,
            start_base: self.start_base,
            pos: 0,
            len: self.len,
        }
    }

    pub fn copy_ascii_to(&self, out: &mut Vec<u8>) {
        out.reserve(self.len as usize);
        for code in self.iter() {
            out.push(
                DnaBase::from_code(code)
                    .expect("stored DNA2 code must be valid")
                    .ascii(),
            );
        }
    }

    pub fn to_ascii_string(&self) -> String {
        let mut bytes = Vec::with_capacity(self.len as usize);
        self.copy_ascii_to(&mut bytes);
        String::from_utf8(bytes).expect("DNA ASCII is valid UTF-8")
    }
}

impl fmt::Debug for Dna2Row<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dna2Row")
            .field("start_base", &self.start_base)
            .field("len", &self.len)
            .finish()
    }
}

#[inline]
fn load_2bit_window_from_payload(payload: &[u8], absolute_base: u64, bases: u32) -> u64 {
    debug_assert!(bases <= 32);
    if bases == 0 {
        return 0;
    }

    let bit_start = absolute_base * 2;
    let byte_idx = (bit_start / 8) as usize;
    let bit_in_byte = (bit_start % 8) as usize;
    let wanted_bits = (bases as usize) * 2;
    let needed_bits = bit_in_byte + wanted_bits;
    let needed_bytes = needed_bits.div_ceil(8);

    debug_assert!(needed_bytes <= 9);
    debug_assert!(byte_idx + needed_bytes <= payload.len());

    let mut acc = 0u128;
    for i in 0..needed_bytes {
        acc = (acc << 8) | u128::from(payload[byte_idx + i]);
    }

    let total_bits = needed_bytes * 8;
    let shift = total_bits - bit_in_byte - wanted_bits;
    let mask = if wanted_bits == 64 {
        u128::from(u64::MAX)
    } else {
        (1u128 << wanted_bits) - 1
    };

    ((acc >> shift) & mask) as u64
}

pub struct Dna2Iter<'a> {
    payload: &'a [u8],
    start_base: u64,
    pos: u32,
    len: u32,
}

impl<'a> Iterator for Dna2Iter<'a> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        let code = get_base_code(self.payload, self.start_base + u64::from(self.pos));
        self.pos += 1;
        Some(code)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.len - self.pos) as usize;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for Dna2Iter<'a> {
    fn len(&self) -> usize {
        (self.len - self.pos) as usize
    }
}

impl<'a> FusedIterator for Dna2Iter<'a> {}

impl<'a> Column for Dna2Column<'a> {
    type Row<'r>
        = Dna2Row<'r>
    where
        Self: 'r;
    type Symbol = u8;
    type SymbolIter<'r>
        = Dna2Iter<'r>
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
        self.row_view(row).iter()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum DnaBase {
    A = 0,
    C = 1,
    G = 2,
    T = 3,
}

impl DnaBase {
    pub const fn code(self) -> u8 {
        self as u8
    }

    pub const fn ascii(self) -> u8 {
        match self {
            DnaBase::A => b'A',
            DnaBase::C => b'C',
            DnaBase::G => b'G',
            DnaBase::T => b'T',
        }
    }

    pub fn from_code(code: u8) -> Result<Self, DnaError> {
        match code {
            0 => Ok(DnaBase::A),
            1 => Ok(DnaBase::C),
            2 => Ok(DnaBase::G),
            3 => Ok(DnaBase::T),
            other => Err(DnaError::InvalidCode(other)),
        }
    }

    pub fn from_ascii(b: u8) -> Result<Self, DnaError> {
        match b {
            b'A' | b'a' => Ok(DnaBase::A),
            b'C' | b'c' => Ok(DnaBase::C),
            b'G' | b'g' => Ok(DnaBase::G),
            b'T' | b't' => Ok(DnaBase::T),
            other => Err(DnaError::InvalidBase(other)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DnaError {
    InvalidBase(u8),
    InvalidCode(u8),
}

impl fmt::Display for DnaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DnaError::InvalidBase(b) => write!(f, "invalid DNA base byte {b:?}"),
            DnaError::InvalidCode(c) => write!(f, "invalid DNA2 base code {c}"),
        }
    }
}

impl std::error::Error for DnaError {}

#[derive(Debug, Default)]
pub struct Dna2ColumnBuilder {
    base_offsets: Vec<u64>,
    logical_lens: Vec<u32>,
    payload: Vec<u8>,
    total_bases: u64,
}

impl Dna2ColumnBuilder {
    pub fn new() -> Self {
        Self {
            base_offsets: vec![0],
            logical_lens: Vec::new(),
            payload: Vec::new(),
            total_bases: 0,
        }
    }

    pub fn with_capacity(rows: usize, bases: usize) -> Self {
        let mut base_offsets = Vec::with_capacity(rows.saturating_add(1));
        base_offsets.push(0);
        Self {
            base_offsets,
            logical_lens: Vec::with_capacity(rows),
            payload: Vec::with_capacity((bases + 3) / 4),
            total_bases: 0,
        }
    }

    pub fn push_ascii(&mut self, seq: &[u8]) -> Result<(), DnaError> {
        let len = u32::try_from(seq.len()).expect("row exceeds u32 logical length");
        self.logical_lens.push(len);
        for &b in seq {
            let code = DnaBase::from_ascii(b)?.code();
            self.push_code_unchecked(code);
        }
        self.base_offsets.push(self.total_bases);
        Ok(())
    }

    pub fn push_str(&mut self, seq: &str) -> Result<(), DnaError> {
        self.push_ascii(seq.as_bytes())
    }

    pub fn push_codes(&mut self, codes: &[u8]) -> Result<(), DnaError> {
        let len = u32::try_from(codes.len()).expect("row exceeds u32 logical length");
        self.logical_lens.push(len);
        for &code in codes {
            DnaBase::from_code(code)?;
            self.push_code_unchecked(code);
        }
        self.base_offsets.push(self.total_bases);
        Ok(())
    }

    pub fn row_count(&self) -> RowId {
        self.logical_lens.len() as RowId
    }

    pub fn total_bases(&self) -> u64 {
        self.total_bases
    }

    pub fn payload_len(&self) -> usize {
        self.payload.len()
    }

    pub fn is_empty(&self) -> bool {
        self.row_count() == 0
    }

    pub fn finish(self, arena: &mut ArenaBuilder) -> Dna2ColumnDesc {
        let row_count = self.row_count();
        debug_assert_eq!(self.base_offsets.len(), row_count as usize + 1);
        debug_assert_eq!(self.logical_lens.len(), row_count as usize);
        debug_assert_eq!(self.payload.len(), ((self.total_bases + 3) / 4) as usize);

        let base_offsets = arena.append_slice(&self.base_offsets);
        let logical_lens = arena.append_slice(&self.logical_lens);
        let payload = arena.append_bytes_aligned(&self.payload, 64);

        Dna2ColumnDesc {
            row_count,
            total_bases: self.total_bases,
            base_offsets,
            logical_lens,
            payload,
        }
    }

    #[inline]
    fn push_code_unchecked(&mut self, code: u8) {
        debug_assert!(code < 4);
        let byte_idx = (self.total_bases / 4) as usize;
        if byte_idx == self.payload.len() {
            self.payload.push(0);
        }
        let slot = (self.total_bases % 4) as u8;
        let shift = 6 - 2 * slot;
        self.payload[byte_idx] |= code << shift;
        self.total_bases += 1;
    }
}

#[derive(Debug)]
pub struct Dna2TableBuilder {
    name: String,
    sequence: Dna2ColumnBuilder,
}

impl Dna2TableBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            sequence: Dna2ColumnBuilder::new(),
        }
    }

    pub fn with_capacity(name: impl Into<String>, rows: usize, bases: usize) -> Self {
        Self {
            name: name.into(),
            sequence: Dna2ColumnBuilder::with_capacity(rows, bases),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn push_str(&mut self, seq: &str) -> Result<RowId, DnaError> {
        let row = self.row_count();
        self.sequence.push_str(seq)?;
        Ok(row)
    }

    pub fn push_ascii(&mut self, seq: &[u8]) -> Result<RowId, DnaError> {
        let row = self.row_count();
        self.sequence.push_ascii(seq)?;
        Ok(row)
    }

    pub fn push_codes(&mut self, codes: &[u8]) -> Result<RowId, DnaError> {
        let row = self.row_count();
        self.sequence.push_codes(codes)?;
        Ok(row)
    }

    pub fn row_count(&self) -> RowId {
        self.sequence.row_count()
    }

    pub fn sequence_builder(&self) -> &Dna2ColumnBuilder {
        &self.sequence
    }

    pub fn sequence_builder_mut(&mut self) -> &mut Dna2ColumnBuilder {
        &mut self.sequence
    }

    pub fn finish(self, arena: &mut ArenaBuilder) -> Dna2TableDesc {
        Dna2TableDesc {
            name: self.name,
            sequence: self.sequence.finish(arena),
        }
    }
}

#[inline]
fn get_base_code(payload: &[u8], global_base_idx: u64) -> u8 {
    let byte_idx = (global_base_idx / 4) as usize;
    let slot = (global_base_idx % 4) as u8;
    let shift = 6 - 2 * slot;
    (payload[byte_idx] >> shift) & 0b11
}
