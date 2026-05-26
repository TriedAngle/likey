//! Database container and typed table dispatch.
//!
//! The arena is shared, but each table is homogeneous and dense. Query code
//! should match on [`TableRef`] once and then enter a monomorphized hot path.

use std::fmt;

use crate::arena::{ArenaBuilder, FrozenArena};
use crate::storage::dna2::{Dna2Table, Dna2TableBuilder, Dna2TableDesc};
use crate::storage::fsst::{FsstTable, FsstTableBuilder, FsstTableDesc};
use crate::storage::utf8::{Utf8Table, Utf8TableBuilder, Utf8TableDesc};
use crate::{RowId, TableId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableKind {
    Utf8,
    Fsst,
    Dna2,
}

#[derive(Debug, Clone)]
pub enum TableDesc {
    Utf8(Utf8TableDesc),
    Fsst(FsstTableDesc),
    Dna2(Dna2TableDesc),
}

impl TableDesc {
    pub fn kind(&self) -> TableKind {
        match self {
            TableDesc::Utf8(_) => TableKind::Utf8,
            TableDesc::Fsst(_) => TableKind::Fsst,
            TableDesc::Dna2(_) => TableKind::Dna2,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            TableDesc::Utf8(t) => &t.name,
            TableDesc::Fsst(t) => &t.name,
            TableDesc::Dna2(t) => &t.name,
        }
    }

    pub fn row_count(&self) -> RowId {
        match self {
            TableDesc::Utf8(t) => t.text.row_count,
            TableDesc::Fsst(t) => t.text.row_count,
            TableDesc::Dna2(t) => t.sequence.row_count,
        }
    }
}

#[derive(Debug)]
pub enum TableBuilder {
    Utf8(Utf8TableBuilder),
    Fsst(FsstTableBuilder),
    Dna2(Dna2TableBuilder),
}

impl TableBuilder {
    pub fn kind(&self) -> TableKind {
        match self {
            TableBuilder::Utf8(_) => TableKind::Utf8,
            TableBuilder::Fsst(_) => TableKind::Fsst,
            TableBuilder::Dna2(_) => TableKind::Dna2,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            TableBuilder::Utf8(t) => t.name(),
            TableBuilder::Fsst(t) => t.name(),
            TableBuilder::Dna2(t) => t.name(),
        }
    }

    pub fn row_count(&self) -> RowId {
        match self {
            TableBuilder::Utf8(t) => t.row_count(),
            TableBuilder::Fsst(t) => t.row_count(),
            TableBuilder::Dna2(t) => t.row_count(),
        }
    }

    fn finish(self, arena: &mut ArenaBuilder) -> TableDesc {
        match self {
            TableBuilder::Utf8(t) => TableDesc::Utf8(t.finish(arena)),
            TableBuilder::Fsst(t) => TableDesc::Fsst(t.finish(arena)),
            TableBuilder::Dna2(t) => TableDesc::Dna2(t.finish(arena)),
        }
    }
}

impl From<Utf8TableBuilder> for TableBuilder {
    fn from(value: Utf8TableBuilder) -> Self {
        TableBuilder::Utf8(value)
    }
}

impl From<FsstTableBuilder> for TableBuilder {
    fn from(value: FsstTableBuilder) -> Self {
        TableBuilder::Fsst(value)
    }
}

impl From<Dna2TableBuilder> for TableBuilder {
    fn from(value: Dna2TableBuilder) -> Self {
        TableBuilder::Dna2(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DbError {
    TableIdOutOfBounds(TableId),
    WrongTableKind {
        expected: TableKind,
        actual: TableKind,
    },
    DuplicateTableName(String),
}

impl fmt::Display for DbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DbError::TableIdOutOfBounds(id) => write!(f, "table id {:?} is out of bounds", id),
            DbError::WrongTableKind { expected, actual } => {
                write!(
                    f,
                    "wrong table kind: expected {:?}, got {:?}",
                    expected, actual
                )
            }
            DbError::DuplicateTableName(name) => write!(f, "duplicate table name {name:?}"),
        }
    }
}

impl std::error::Error for DbError {}

pub struct DbBuilder {
    arena: ArenaBuilder,
    tables: Vec<TableDesc>,
    check_duplicate_names: bool,
}

impl DbBuilder {
    pub fn new() -> Self {
        Self {
            arena: ArenaBuilder::new(),
            tables: Vec::new(),
            check_duplicate_names: true,
        }
    }

    pub fn with_arena_capacity(capacity: usize) -> Self {
        Self {
            arena: ArenaBuilder::with_capacity(capacity),
            tables: Vec::new(),
            check_duplicate_names: true,
        }
    }

    /// Disable duplicate-name checks. Useful for benchmark fixtures where names
    /// are irrelevant and construction overhead should be minimal.
    pub fn allow_duplicate_names(mut self) -> Self {
        self.check_duplicate_names = false;
        self
    }

    pub fn arena_len(&self) -> usize {
        self.arena.len()
    }

    pub fn table_count(&self) -> usize {
        self.tables.len()
    }

    pub fn table_descs(&self) -> &[TableDesc] {
        &self.tables
    }

    pub fn add_table<T>(&mut self, table: T) -> Result<TableId, DbError>
    where
        T: Into<TableBuilder>,
    {
        let table = table.into();
        self.check_name(table.name())?;
        let id = TableId(self.tables.len() as u32);
        let desc = table.finish(&mut self.arena);
        self.tables.push(desc);
        Ok(id)
    }

    pub fn add_utf8_table(&mut self, table: Utf8TableBuilder) -> Result<TableId, DbError> {
        self.add_table(table)
    }

    pub fn add_fsst_table(&mut self, table: FsstTableBuilder) -> Result<TableId, DbError> {
        self.add_table(table)
    }

    pub fn add_dna2_table(&mut self, table: Dna2TableBuilder) -> Result<TableId, DbError> {
        self.add_table(table)
    }

    pub fn freeze(self) -> Db {
        Db {
            arena: self.arena.freeze(),
            tables: self.tables.into_boxed_slice(),
        }
    }

    fn check_name(&self, name: &str) -> Result<(), DbError> {
        if self.check_duplicate_names && self.tables.iter().any(|t| t.name() == name) {
            return Err(DbError::DuplicateTableName(name.to_owned()));
        }
        Ok(())
    }
}

impl Default for DbBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Db {
    arena: FrozenArena,
    tables: Box<[TableDesc]>,
}

impl Db {
    pub fn arena(&self) -> &FrozenArena {
        &self.arena
    }

    pub fn table_count(&self) -> usize {
        self.tables.len()
    }

    pub fn table_descs(&self) -> &[TableDesc] {
        &self.tables
    }

    pub fn table_desc(&self, id: TableId) -> Option<&TableDesc> {
        self.tables.get(id.0 as usize)
    }

    pub fn table(&self, id: TableId) -> Option<TableRef<'_>> {
        let desc = self.table_desc(id)?;
        Some(match desc {
            TableDesc::Utf8(t) => TableRef::Utf8(Utf8Table::new(&self.arena, t)),
            TableDesc::Fsst(t) => TableRef::Fsst(FsstTable::new(&self.arena, t)),
            TableDesc::Dna2(t) => TableRef::Dna2(Dna2Table::new(&self.arena, t)),
        })
    }

    pub fn table_by_name(&self, name: &str) -> Option<(TableId, TableRef<'_>)> {
        let idx = self.tables.iter().position(|t| t.name() == name)?;
        let id = TableId(idx as u32);
        Some((
            id,
            self.table(id).expect("table id from position must exist"),
        ))
    }

    pub fn utf8_table(&self, id: TableId) -> Result<Utf8Table<'_>, DbError> {
        match self.table(id).ok_or(DbError::TableIdOutOfBounds(id))? {
            TableRef::Utf8(t) => Ok(t),
            other => Err(DbError::WrongTableKind {
                expected: TableKind::Utf8,
                actual: other.kind(),
            }),
        }
    }

    pub fn fsst_table(&self, id: TableId) -> Result<FsstTable<'_>, DbError> {
        match self.table(id).ok_or(DbError::TableIdOutOfBounds(id))? {
            TableRef::Fsst(t) => Ok(t),
            other => Err(DbError::WrongTableKind {
                expected: TableKind::Fsst,
                actual: other.kind(),
            }),
        }
    }

    pub fn dna2_table(&self, id: TableId) -> Result<Dna2Table<'_>, DbError> {
        match self.table(id).ok_or(DbError::TableIdOutOfBounds(id))? {
            TableRef::Dna2(t) => Ok(t),
            other => Err(DbError::WrongTableKind {
                expected: TableKind::Dna2,
                actual: other.kind(),
            }),
        }
    }

    pub fn utf8_table_by_name(&self, name: &str) -> Option<(TableId, Utf8Table<'_>)> {
        let (id, table) = self.table_by_name(name)?;
        match table {
            TableRef::Utf8(t) => Some((id, t)),
            TableRef::Fsst(_) | TableRef::Dna2(_) => None,
        }
    }

    pub fn fsst_table_by_name(&self, name: &str) -> Option<(TableId, FsstTable<'_>)> {
        let (id, table) = self.table_by_name(name)?;
        match table {
            TableRef::Fsst(t) => Some((id, t)),
            TableRef::Utf8(_) | TableRef::Dna2(_) => None,
        }
    }

    pub fn dna2_table_by_name(&self, name: &str) -> Option<(TableId, Dna2Table<'_>)> {
        let (id, table) = self.table_by_name(name)?;
        match table {
            TableRef::Dna2(t) => Some((id, t)),
            TableRef::Utf8(_) | TableRef::Fsst(_) => None,
        }
    }
}

impl fmt::Debug for Db {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Db")
            .field("arena", &self.arena)
            .field("tables", &self.tables)
            .finish()
    }
}

#[derive(Clone, Copy)]
pub enum TableRef<'a> {
    Utf8(Utf8Table<'a>),
    Fsst(FsstTable<'a>),
    Dna2(Dna2Table<'a>),
}

impl<'a> TableRef<'a> {
    pub fn kind(&self) -> TableKind {
        match self {
            TableRef::Utf8(_) => TableKind::Utf8,
            TableRef::Fsst(_) => TableKind::Fsst,
            TableRef::Dna2(_) => TableKind::Dna2,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            TableRef::Utf8(t) => t.name(),
            TableRef::Fsst(t) => t.name(),
            TableRef::Dna2(t) => t.name(),
        }
    }

    pub fn row_count(&self) -> RowId {
        match self {
            TableRef::Utf8(t) => t.row_count(),
            TableRef::Fsst(t) => t.row_count(),
            TableRef::Dna2(t) => t.row_count(),
        }
    }

    pub fn as_utf8(self) -> Option<Utf8Table<'a>> {
        match self {
            TableRef::Utf8(t) => Some(t),
            TableRef::Fsst(_) | TableRef::Dna2(_) => None,
        }
    }

    pub fn as_fsst(self) -> Option<FsstTable<'a>> {
        match self {
            TableRef::Fsst(t) => Some(t),
            TableRef::Utf8(_) | TableRef::Dna2(_) => None,
        }
    }

    pub fn as_dna2(self) -> Option<Dna2Table<'a>> {
        match self {
            TableRef::Dna2(t) => Some(t),
            TableRef::Utf8(_) | TableRef::Fsst(_) => None,
        }
    }
}

impl fmt::Debug for TableRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TableRef::Utf8(t) => t.fmt(f),
            TableRef::Fsst(t) => t.fmt(f),
            TableRef::Dna2(t) => t.fmt(f),
        }
    }
}
