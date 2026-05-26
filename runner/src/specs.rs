use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::cli::{AlgorithmKind, DataType, IndexKind, StorageKind, parse_boolish};

#[derive(Debug, Clone)]
pub struct DataSpec {
    pub name: String,
    pub path: PathBuf,
    pub data_type: DataType,
    pub storages: Vec<StorageKind>,
    pub column: String,
    pub key_column: Option<String>,
    pub value_column: Option<String>,
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
    column: Option<String>,
    key_column: Option<String>,
    value_column: Option<String>,
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
            .unwrap_or("dna-fasta")
            .parse::<DataType>()?;
        let storage_raw = row
            .storage
            .as_deref()
            .unwrap_or(data_type.default_storage_raw());
        let storages = parse_storage_list(storage_raw)?;
        if !data_type.allows_dna2() && storages.iter().any(|s| *s == StorageKind::Dna2) {
            bail!(
                "data CSV row {} requested dna2 storage for {}; only dna-fasta supports bitpacked DNA2 storage",
                idx + 2,
                data_type.as_str()
            );
        }

        let path_buf = PathBuf::from(&row.path);
        let file_stem = path_buf
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("dataset");
        let name = row.name.unwrap_or_else(|| file_stem.to_owned());
        let column = row.column.unwrap_or_else(|| match data_type {
            DataType::DnaFasta | DataType::ProteinFasta => "sequence".to_owned(),
            DataType::JobCsv => file_stem.to_owned(),
        });
        out.push(DataSpec {
            name,
            path: path_buf,
            data_type,
            storages,
            column,
            key_column: row.key_column,
            value_column: row.value_column,
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
        let raw = row
            .algorithm
            .or(row.name)
            .with_context(|| format!("algorithms CSV row {} needs algorithm or name", idx + 2))?;
        out.push(AlgorithmKind::from_str(&raw)?);
    }
    if out.is_empty() {
        bail!(
            "algorithms CSV {} produced no enabled algorithms",
            path.display()
        );
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
        let raw = row
            .index
            .or(row.name)
            .with_context(|| format!("indexes CSV row {} needs index or name", idx + 2))?;
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
        bail!(
            "patterns CSV {} produced no enabled patterns",
            path.display()
        );
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
