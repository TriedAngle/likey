use std::{fs::File, path::Path};

use csv::{ReaderBuilder, Trim};

use crate::{
    dataset::{Row, Table},
    BumpArena,
};

#[derive(Debug, Clone)]
pub struct ColumnSpec {
    pub name: String,
    pub index: usize,
}

#[derive(Debug, Clone)]
pub struct DelimitedOptions {
    pub delimiter: u8,
    pub has_headers: bool,
    pub trim_fields: bool,
}

impl Default for DelimitedOptions {
    fn default() -> Self {
        Self {
            delimiter: b',',
            has_headers: true,
            trim_fields: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ByteLimit {
    pub max_bytes: usize,
    pub current: usize,
}

impl ByteLimit {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            current: 0,
        }
    }

    pub fn try_reserve(&mut self, bytes: usize) -> bool {
        if self.current.saturating_add(bytes) > self.max_bytes {
            return false;
        }
        self.current += bytes;
        true
    }
}

pub fn load_delimited_columns<'a>(
    arena: &'a BumpArena,
    path: &Path,
    options: &DelimitedOptions,
    columns: &[ColumnSpec],
    limit: &mut Option<ByteLimit>,
) -> Result<Vec<Table<'a>>, String> {
    let file = File::open(path)
        .map_err(|e| format!("Failed to read delimited file {}: {}", path.display(), e))?;

    let mut reader = ReaderBuilder::new()
        .delimiter(options.delimiter)
        .has_headers(options.has_headers)
        .flexible(true)
        .trim(if options.trim_fields {
            Trim::All
        } else {
            Trim::None
        })
        .from_reader(file);

    let mut rows_by_column: Vec<Vec<Row<'a>>> = vec![Vec::new(); columns.len()];

    for record_result in reader.records() {
        let record =
            record_result.map_err(|e| format!("CSV parse error in {}: {}", path.display(), e))?;

        if let Some(limit) = limit.as_mut() {
            let mut bytes = 0usize;
            for spec in columns.iter() {
                bytes += record.get(spec.index).unwrap_or("").as_bytes().len();
            }
            if !limit.try_reserve(bytes) {
                continue;
            }
        }

        for (idx, spec) in columns.iter().enumerate() {
            let value = record.get(spec.index).unwrap_or("");
            let data = arena.alloc_str(value);
            rows_by_column[idx].push(Row {
                id: "",
                desc: "",
                data,
            });
        }
    }

    let file_name = filename_from_path(path)?;
    let mut tables = Vec::with_capacity(columns.len());
    for (spec, rows) in columns.iter().zip(rows_by_column.into_iter()) {
        tables.push(Table {
            name: format!("{}.{}", file_name, spec.name),
            rows: rows.into_boxed_slice(),
        });
    }

    Ok(tables)
}

fn filename_from_path(path: &Path) -> Result<String, String> {
    path.file_name()
        .ok_or_else(|| format!("Missing filename for path {}", path.display()))
        .map(|name| name.to_string_lossy().to_string())
}
