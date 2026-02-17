use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::{fasta, BumpArena};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    Text,
    Fasta,
}

#[derive(Debug, Clone)]
pub struct Source {
    pub path: PathBuf,
    pub kind: SourceKind,
}

#[derive(Debug, Clone)]
pub struct Row<'a> {
    pub id: &'a str,
    pub desc: &'a str,
    pub data: &'a str,
}

#[derive(Debug, Clone)]
pub struct Table<'a> {
    pub name: String,
    pub rows: Box<[Row<'a>]>,
}

#[derive(Debug, Clone)]
pub struct DataSet<'a> {
    pub tables: Box<[Table<'a>]>,
}

pub fn load_text_table<'a>(arena: &'a BumpArena, path: &Path) -> Result<Table<'a>, String> {
    let raw_string = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read text file {}: {}", path.display(), e))?;

    let data = arena.alloc_str(&raw_string);
    let file_name = filename_from_path(path)?;
    let id = arena.alloc_str(&file_name);

    let row = Row { id, desc: "", data };

    Ok(Table {
        name: file_name,
        rows: Box::new([row]),
    })
}

pub fn load_fasta_table<'a>(arena: &'a BumpArena, path: &Path) -> Result<Table<'a>, String> {
    let raw_bytes = fs::read(path)
        .map_err(|e| format!("Failed to read FASTA file {}: {}", path.display(), e))?;

    let entries = fasta::parse_fasta_into_arena(arena, &raw_bytes)?;

    let rows: Vec<Row<'a>> = entries
        .iter()
        .map(|entry| Row {
            id: entry.id,
            desc: entry.desc,
            data: entry.data,
        })
        .collect();

    Ok(Table {
        name: filename_from_path(path)?,
        rows: rows.into_boxed_slice(),
    })
}

pub fn load_dataset<'a>(arena: &'a BumpArena, sources: &[Source]) -> Result<DataSet<'a>, String> {
    let mut tables = Vec::with_capacity(sources.len());

    for source in sources {
        let table = match source.kind {
            SourceKind::Text => load_text_table(arena, &source.path)?,
            SourceKind::Fasta => load_fasta_table(arena, &source.path)?,
        };

        tables.push(table);
    }

    Ok(DataSet {
        tables: tables.into_boxed_slice(),
    })
}

pub fn load_dataset_from_paths<'a>(
    arena: &'a BumpArena,
    paths: &[PathBuf],
) -> Result<DataSet<'a>, String> {
    let sources: Vec<Source> = paths
        .iter()
        .map(|path| Source {
            path: path.clone(),
            kind: infer_source_kind(path),
        })
        .collect();

    load_dataset(arena, &sources)
}

pub fn infer_source_kind(path: &Path) -> SourceKind {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("fasta") | Some("fa") | Some("fna") | Some("faa") | Some("fsa") => SourceKind::Fasta,
        _ => SourceKind::Text,
    }
}

fn filename_from_path(path: &Path) -> Result<String, String> {
    path.file_name()
        .ok_or_else(|| format!("Missing filename for path {}", path.display()))
        .map(|name| name.to_string_lossy().to_string())
}
