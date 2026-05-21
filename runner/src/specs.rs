use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::cli::{parse_boolish, AlgorithmKind, DataType, IndexKind, StorageKind};

#[derive(Debug, Clone)]
pub struct DataSpec {
    pub name: String,
    pub path: PathBuf,
    pub data_type: DataType,
    pub storages: Vec<StorageKind>,
}

#[derive(Debug, Clone)]
pub struct PatternSpec {
    pub name: String,
    pub pattern: String,
}

#[derive(Debug, Deserialize)]
struct DataRow {
    name: Option<String>,
    path: String,
    #[serde(rename = "type")]
    data_type: Option<String>,
    storage: Option<String>,
    enabled: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AlgorithmRow {
    algorithm: Option<String>,
    name: Option<String>,
    enabled: Option<String>,
}

#[derive(Debug, Deserialize)]
struct IndexRow {
    index: Option<String>,
    name: Option<String>,
    enabled: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PatternRow {
    name: Option<String>,
    pattern: String,
    enabled: Option<String>,
}

pub fn load_data_specs(path: &std::path::Path) -> Result<Vec<DataSpec>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("open data CSV {}", path.display()))?;
    let mut out = Vec::new();
    for (idx, row) in reader.deserialize::<DataRow>().enumerate() {
        let row = row.with_context(|| format!("parse data CSV row {}", idx + 2))?;
        if row.enabled.as_deref().is_some_and(|s| !parse_boolish(s)) {
            continue;
        }
        let data_type = row
            .data_type
            .as_deref()
            .unwrap_or("fasta")
            .parse::<DataType>()?;
        let storages = parse_storage_list(row.storage.as_deref().unwrap_or("both"))?;
        let path_buf = PathBuf::from(&row.path);
        let name = row.name.unwrap_or_else(|| {
            path_buf
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("dataset")
                .to_owned()
        });
        out.push(DataSpec {
            name,
            path: path_buf,
            data_type,
            storages,
        });
    }
    if out.is_empty() {
        bail!("data CSV {} produced no enabled datasets", path.display());
    }
    Ok(out)
}

pub fn load_algorithms(path: &std::path::Path) -> Result<Vec<AlgorithmKind>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("open algorithms CSV {}", path.display()))?;
    let mut out = Vec::new();
    for (idx, row) in reader.deserialize::<AlgorithmRow>().enumerate() {
        let row = row.with_context(|| format!("parse algorithms CSV row {}", idx + 2))?;
        if row.enabled.as_deref().is_some_and(|s| !parse_boolish(s)) {
            continue;
        }
        let raw = row.algorithm.or(row.name).with_context(|| {
            format!("algorithms CSV row {} needs algorithm or name", idx + 2)
        })?;
        out.push(AlgorithmKind::from_str(&raw)?);
    }
    if out.is_empty() {
        bail!("algorithms CSV {} produced no enabled algorithms", path.display());
    }
    out.sort_by_key(|a| a.as_str());
    out.dedup();
    Ok(out)
}

pub fn load_indexes(path: Option<&std::path::Path>) -> Result<Vec<IndexKind>> {
    let Some(path) = path else {
        return Ok(vec![IndexKind::FullScan]);
    };
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("open indexes CSV {}", path.display()))?;
    let mut out = Vec::new();
    for (idx, row) in reader.deserialize::<IndexRow>().enumerate() {
        let row = row.with_context(|| format!("parse indexes CSV row {}", idx + 2))?;
        if row.enabled.as_deref().is_some_and(|s| !parse_boolish(s)) {
            continue;
        }
        let raw = row.index.or(row.name).with_context(|| {
            format!("indexes CSV row {} needs index or name", idx + 2)
        })?;
        out.push(IndexKind::from_str(&raw)?);
    }
    if out.is_empty() {
        bail!("indexes CSV {} produced no enabled indexes", path.display());
    }
    out.sort_by_key(|i| i.as_str());
    out.dedup();
    Ok(out)
}

pub fn load_patterns(path: &std::path::Path) -> Result<Vec<PatternSpec>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("open patterns CSV {}", path.display()))?;
    let mut out = Vec::new();
    for (idx, row) in reader.deserialize::<PatternRow>().enumerate() {
        let row = row.with_context(|| format!("parse patterns CSV row {}", idx + 2))?;
        if row.enabled.as_deref().is_some_and(|s| !parse_boolish(s)) {
            continue;
        }
        let name = row.name.unwrap_or_else(|| format!("pattern_{}", idx + 1));
        out.push(PatternSpec {
            name,
            pattern: row.pattern,
        });
    }
    if out.is_empty() {
        bail!("patterns CSV {} produced no enabled patterns", path.display());
    }
    Ok(out)
}

fn parse_storage_list(s: &str) -> Result<Vec<StorageKind>> {
    let mut out = Vec::new();
    for part in s.split(|c| c == ',' || c == ';' || c == '|') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let lower = part.to_ascii_lowercase();
        if lower == "both" || lower == "all" {
            out.push(StorageKind::Utf8);
            out.push(StorageKind::Dna2);
        } else {
            out.push(StorageKind::from_str(part)?);
        }
    }
    if out.is_empty() {
        bail!("storage list cannot be empty");
    }
    out.sort_by_key(|s| s.as_str());
    out.dedup();
    Ok(out)
}
