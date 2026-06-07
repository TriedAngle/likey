use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Validate runner CSV inputs with the runner's CSV parser settings")]
struct Args {
    /// Runner data CSV to validate.
    #[arg(long, default_value = "data/data_all.csv")]
    data_csv: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let data_base = args
        .data_csv
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let mut data_reader = csv::Reader::from_path(&args.data_csv)
        .with_context(|| format!("open data CSV {}", args.data_csv.display()))?;
    let headers = data_reader.headers()?.clone();
    let path_idx = required_header(&headers, "path")?;
    let type_idx = optional_header(&headers, "type");
    let enabled_idx = optional_header(&headers, "enabled");
    let key_column_idx = optional_header(&headers, "key_column");
    let value_column_idx = optional_header(&headers, "value_column");

    let mut files = 0usize;
    let mut rows = 0u64;
    for (idx, record) in data_reader.records().enumerate() {
        let record = record.with_context(|| format!("parse data CSV row {}", idx + 2))?;
        if enabled_idx
            .and_then(|idx| record.get(idx))
            .is_some_and(|enabled| !parse_boolish(enabled))
        {
            continue;
        }
        if !is_job_csv(type_idx.and_then(|idx| record.get(idx))) {
            continue;
        }

        let path = record
            .get(path_idx)
            .filter(|path| !path.is_empty())
            .with_context(|| format!("data CSV row {} needs path", idx + 2))?;
        let path = resolve_relative(&data_base, Path::new(path));
        let key_column = key_column_idx
            .and_then(|idx| record.get(idx))
            .filter(|s| !s.is_empty());
        let value_column = value_column_idx
            .and_then(|idx| record.get(idx))
            .filter(|s| !s.is_empty());
        let file_rows = validate_job_csv(&path, key_column, value_column)
            .with_context(|| format!("validate {}", path.display()))?;
        files += 1;
        rows += file_rows;
        println!("{}: {file_rows} rows", path.display());
    }

    println!("CSV validation passed: {files} job-csv files, {rows} rows");
    Ok(())
}

fn validate_job_csv(
    path: &Path,
    key_column: Option<&str>,
    value_column: Option<&str>,
) -> Result<u64> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("open CSV {}", path.display()))?;
    let headers = reader.headers()?.clone();
    let key_idx = select_column_index(&headers, key_column, &["key", "id", "row_id"], 0)?;
    let value_idx = select_column_index(&headers, value_column, &["value", "text", "data"], 1)?;

    let mut rows = 0u64;
    for (idx, record) in reader.records().enumerate() {
        let record = record.with_context(|| format!("parse CSV record {}", idx + 2))?;
        if record.len() != headers.len() {
            bail!(
                "record {} has {} fields, expected {} from headers {:?}",
                idx + 2,
                record.len(),
                headers.len(),
                headers
            );
        }
        if record.get(key_idx).is_none() || record.get(value_idx).is_none() {
            bail!("record {} is missing selected key/value fields", idx + 2);
        }
        rows += 1;
    }
    Ok(rows)
}

fn select_column_index(
    headers: &csv::StringRecord,
    requested: Option<&str>,
    fallback_names: &[&str],
    fallback_idx: usize,
) -> Result<usize> {
    if let Some(name) = requested {
        if let Some(idx) = headers.iter().position(|h| h == name) {
            return Ok(idx);
        }
        bail!("CSV column {name:?} not found; headers are {:?}", headers);
    }
    for name in fallback_names {
        if let Some(idx) = headers.iter().position(|h| h.eq_ignore_ascii_case(name)) {
            return Ok(idx);
        }
    }
    if fallback_idx < headers.len() {
        Ok(fallback_idx)
    } else {
        bail!("CSV has too few columns; headers are {:?}", headers)
    }
}

fn required_header(headers: &csv::StringRecord, name: &str) -> Result<usize> {
    optional_header(headers, name).with_context(|| format!("data CSV missing {name:?} column"))
}

fn optional_header(headers: &csv::StringRecord, name: &str) -> Option<usize> {
    headers.iter().position(|header| header == name)
}

fn resolve_relative(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    }
}

fn is_job_csv(value: Option<&str>) -> bool {
    matches!(
        normalize_name(value.unwrap_or("dna-fasta")).as_str(),
        "job-csv" | "job" | "csv" | "key-value" | "keyvalue" | "kv-csv"
    )
}

fn parse_boolish(s: &str) -> bool {
    matches!(
        normalize_name(s).as_str(),
        "1" | "true" | "yes" | "y" | "on" | "enabled"
    )
}

fn normalize_name(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '_'], "-")
}
