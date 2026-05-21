//! Database catalog and table access.
//!
//! A `Db` owns one frozen arena plus an in-memory table catalog.  Each table is
//! columnar: every column is dense and indexed by the same `RowId`.

use std::collections::HashSet;

use crate::arena::{ArenaBuilder, FrozenArena};
use crate::{Db, Dna2Column, Dna2TableBuilder};
use crate::{Utf8Column, Utf8TableBuilder};
use crate::{ColumnBuilder, ColumnDesc, ColumnRef};
use crate::{ColumnId, DbError, EncodingKind, RowId, TableId};

#[derive(Debug, Clone)]
pub struct TableDesc {
    pub name: String,
    pub row_count: RowId,
    pub columns: Vec<ColumnDesc>,
}

impl TableDesc {
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    pub fn column_id_by_name(&self, name: &str) -> Option<ColumnId> {
        self.columns
            .iter()
            .position(|c| c.name() == name)
            .map(|i| ColumnId(i as u32))
    }
}

#[derive(Debug, Clone)]
pub struct TableBuilder {
    name: String,
    columns: Vec<ColumnBuilder>,
}

impl TableBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), columns: Vec::new() }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    pub fn add_column(&mut self, column: ColumnBuilder) -> Result<ColumnId, DbError> {
        if self.columns.iter().any(|c| c.name() == column.name()) {
            return Err(DbError::DuplicateColumnName {
                table: self.name.clone(),
                column: column.name().to_string(),
            });
        }

        if let Some(first) = self.columns.first() {
            let expected = first.row_count();
            let found = column.row_count();
            if expected != found {
                return Err(DbError::RowCountMismatch {
                    table: self.name.clone(),
                    expected,
                    found,
                    column: column.name().to_string(),
                });
            }
        }

        let id = ColumnId(self.columns.len() as u32);
        self.columns.push(column);
        Ok(id)
    }

    pub fn add_utf8_column(&mut self, column: crate::Utf8ColumnBuilder) -> Result<ColumnId, DbError> {
        self.add_column(ColumnBuilder::Utf8(column))
    }

    pub fn add_dna2_column(&mut self, column: crate::Dna2ColumnBuilder) -> Result<ColumnId, DbError> {
        self.add_column(ColumnBuilder::Dna2(column))
    }

    pub fn finish(self, arena: &mut ArenaBuilder) -> Result<TableDesc, DbError> {
        if self.columns.is_empty() {
            return Err(DbError::EmptyTable { table: self.name });
        }

        let mut names = HashSet::new();
        let row_count = self.columns[0].row_count();
        for c in &self.columns {
            if c.row_count() != row_count {
                return Err(DbError::RowCountMismatch {
                    table: self.name.clone(),
                    expected: row_count,
                    found: c.row_count(),
                    column: c.name().to_string(),
                });
            }
            if !names.insert(c.name().to_string()) {
                return Err(DbError::DuplicateColumnName {
                    table: self.name.clone(),
                    column: c.name().to_string(),
                });
            }
        }

        let columns = self.columns.into_iter().map(|c| c.finish(arena)).collect();
        Ok(TableDesc { name: self.name, row_count, columns })
    }
}

#[derive(Clone, Copy)]
pub struct Table<'a> {
    pub(crate) db: &'a Db,
    pub(crate) id: TableId,
    pub(crate) desc: &'a TableDesc,
}

impl<'a> Table<'a> {
    pub fn id(&self) -> TableId {
        self.id
    }

    pub fn name(&self) -> &'a str {
        &self.desc.name
    }

    pub fn row_count(&self) -> RowId {
        self.desc.row_count
    }

    pub fn column_count(&self) -> usize {
        self.desc.columns.len()
    }

    pub fn column_desc(&self, id: ColumnId) -> Option<&'a ColumnDesc> {
        self.desc.columns.get(id.0 as usize)
    }

    pub fn column(&self, id: ColumnId) -> Option<ColumnRef<'a>> {
        let desc = self.column_desc(id)?;
        Some(ColumnRef::new(&self.db.arena, desc))
    }

    pub fn column_by_name(&self, name: &str) -> Option<ColumnRef<'a>> {
        let id = self.desc.column_id_by_name(name)?;
        self.column(id)
    }

    pub fn utf8_column(&self, id: ColumnId) -> Result<Utf8Column<'a>, DbError> {
        let col = self.column(id).ok_or(DbError::ColumnNotFound(id))?;
        match col {
            ColumnRef::Utf8(c) => Ok(c),
            ColumnRef::Dna2(c) => Err(DbError::WrongColumnKind {
                expected: EncodingKind::Utf8Bytes,
                found: c.encoding(),
            }),
        }
    }

    pub fn dna2_column(&self, id: ColumnId) -> Result<Dna2Column<'a>, DbError> {
        let col = self.column(id).ok_or(DbError::ColumnNotFound(id))?;
        match col {
            ColumnRef::Dna2(c) => Ok(c),
            ColumnRef::Utf8(c) => Err(DbError::WrongColumnKind {
                expected: EncodingKind::Dna2Bits,
                found: c.encoding(),
            }),
        }
    }

    pub fn utf8_column_by_name(&self, name: &str) -> Result<Utf8Column<'a>, DbError> {
        let id = self
            .desc
            .column_id_by_name(name)
            .ok_or_else(|| DbError::ColumnNameNotFound(name.to_string()))?;
        self.utf8_column(id)
    }

    pub fn dna2_column_by_name(&self, name: &str) -> Result<Dna2Column<'a>, DbError> {
        let id = self
            .desc
            .column_id_by_name(name)
            .ok_or_else(|| DbError::ColumnNameNotFound(name.to_string()))?;
        self.dna2_column(id)
    }

    pub fn as_single_column_table(&self) -> Result<SingleColumnTable<'a>, DbError> {
        if self.column_count() != 1 {
            return Err(DbError::NotSingleColumnTable {
                table: self.name().to_string(),
                columns: self.column_count(),
            });
        }

        match self.column(ColumnId(0)).expect("checked column count") {
            ColumnRef::Utf8(c) => Ok(SingleColumnTable::Utf8(Utf8Table { table: *self, text: c })),
            ColumnRef::Dna2(c) => Ok(SingleColumnTable::Dna2(Dna2Table { table: *self, sequence: c })),
        }
    }
}

pub type TableRef<'a> = SingleColumnTable<'a>;

#[derive(Clone, Copy)]
pub enum SingleColumnTable<'a> {
    Utf8(Utf8Table<'a>),
    Dna2(Dna2Table<'a>),
}

impl<'a> SingleColumnTable<'a> {
    pub fn name(&self) -> &'a str {
        match self {
            SingleColumnTable::Utf8(t) => t.name(),
            SingleColumnTable::Dna2(t) => t.name(),
        }
    }

    pub fn row_count(&self) -> RowId {
        match self {
            SingleColumnTable::Utf8(t) => t.row_count(),
            SingleColumnTable::Dna2(t) => t.row_count(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Utf8Table<'a> {
    table: Table<'a>,
    text: Utf8Column<'a>,
}

impl<'a> Utf8Table<'a> {
    pub fn table(&self) -> Table<'a> {
        self.table
    }

    pub fn name(&self) -> &'a str {
        self.table.name()
    }

    pub fn row_count(&self) -> RowId {
        self.table.row_count()
    }

    pub fn text(&self) -> Utf8Column<'a> {
        self.text
    }
}

#[derive(Clone, Copy)]
pub struct Dna2Table<'a> {
    table: Table<'a>,
    sequence: Dna2Column<'a>,
}

impl<'a> Dna2Table<'a> {
    pub fn table(&self) -> Table<'a> {
        self.table
    }

    pub fn name(&self) -> &'a str {
        self.table.name()
    }

    pub fn row_count(&self) -> RowId {
        self.table.row_count()
    }

    pub fn sequence(&self) -> Dna2Column<'a> {
        self.sequence
    }
}
