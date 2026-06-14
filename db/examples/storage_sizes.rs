use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use db::{ColumnStorageSize, DbBuilder, FsstTableBuilder, Utf8TableBuilder};

const MAX_TOTAL_BYTES: u64 = 100_000_000;
const MAX_ROW_BYTES: u64 = 50_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DataType {
    DnaFasta,
    JobCsv,
}

impl DataType {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "dna-fasta" | "dna" | "fasta" | "fa" | "fna" => Ok(Self::DnaFasta),
            "job-csv" | "job" | "csv" | "key-value" | "keyvalue" | "kv-csv" => Ok(Self::JobCsv),
            other => Err(format!("unsupported data type {other:?}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::DnaFasta => "dna-fasta",
            Self::JobCsv => "job-csv",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Storage {
    Utf8,
    Fsst,
}

impl Storage {
    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "utf8" | "utf-8" | "bytes" | "byte" => Some(Self::Utf8),
            "fsst" => Some(Self::Fsst),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Utf8 => "utf8",
            Self::Fsst => "fsst",
        }
    }
}

#[derive(Debug)]
struct DataSpec {
    name: String,
    path: PathBuf,
    data_type: DataType,
    storages: Vec<Storage>,
    column: String,
    value_column: Option<String>,
}

#[derive(Default)]
struct LoadStats {
    records_seen: u64,
    records_loaded: u64,
    records_skipped: u64,
    records_truncated: u64,
    total_input_sequence_bytes: u64,
    total_loaded_symbols: u64,
    stopped_by_max_total_bytes: bool,
}

struct SizeRow {
    dataset: String,
    column: String,
    data_path: String,
    data_type: &'static str,
    storage: &'static str,
    stats: LoadStats,
    size: ColumnStorageSize,
    logical_payload_bytes: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_csvs = env::args().skip(1).collect::<Vec<_>>();
    if data_csvs.is_empty() {
        return Err("usage: cargo run -p db --example storage_sizes -- <data-csv>...".into());
    }

    println!(
        "dataset,column,data_path,data_type,storage,row_count,total_input_sequence_bytes,total_loaded_symbols,records_seen,records_loaded,records_skipped,records_truncated,storage_total_bytes,offsets_bytes,logical_lens_bytes,payload_bytes,codec_bytes,logical_payload_bytes"
    );

    for data_csv in data_csvs {
        let data_csv = PathBuf::from(data_csv);
        let base = data_csv.parent().unwrap_or_else(|| Path::new("."));
        let specs = load_data_specs(&data_csv)?;
        for spec in specs {
            let path = resolve_relative(base, &spec.path);
            for storage in spec.storages.iter().copied() {
                let row = match spec.data_type {
                    DataType::DnaFasta => load_fasta(&path, &spec, storage)?,
                    DataType::JobCsv => load_job_csv(&path, &spec, storage)?,
                };
                print_size_row(&row);
            }
        }
    }

    Ok(())
}

fn print_size_row(row: &SizeRow) {
    println!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        csv_escape(&row.dataset),
        csv_escape(&row.column),
        csv_escape(&row.data_path),
        row.data_type,
        row.storage,
        row.stats.records_loaded,
        row.stats.total_input_sequence_bytes,
        row.stats.total_loaded_symbols,
        row.stats.records_seen,
        row.stats.records_loaded,
        row.stats.records_skipped,
        row.stats.records_truncated,
        row.size.total_bytes(),
        row.size.offsets_bytes,
        row.size.logical_lens_bytes,
        row.size.payload_bytes,
        row.size.codec_bytes,
        row.logical_payload_bytes,
    );
}

fn load_data_specs(path: &Path) -> Result<Vec<DataSpec>, Box<dyn std::error::Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let mut specs = Vec::new();

    for record in reader.records() {
        let record = record?;
        if field(&headers, &record, "enabled").is_some_and(|enabled| !parse_boolish(enabled)) {
            continue;
        }

        let path_raw = field(&headers, &record, "path").ok_or("data CSV row missing path")?;
        let path_buf = PathBuf::from(path_raw);
        let file_stem = path_buf
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("dataset");
        let data_type = field(&headers, &record, "type")
            .map(DataType::parse)
            .transpose()?
            .unwrap_or(DataType::DnaFasta);
        let storages = field(&headers, &record, "storage")
            .unwrap_or("utf8")
            .split(';')
            .filter_map(Storage::parse)
            .collect::<Vec<_>>();
        if storages.is_empty() {
            continue;
        }

        let name = field(&headers, &record, "name")
            .filter(|value| !value.is_empty())
            .unwrap_or(file_stem)
            .to_owned();
        let column = field(&headers, &record, "column")
            .filter(|value| !value.is_empty())
            .unwrap_or(match data_type {
                DataType::DnaFasta => "sequence",
                DataType::JobCsv => file_stem,
            })
            .to_owned();
        let value_column = field(&headers, &record, "value_column")
            .filter(|value| !value.is_empty())
            .map(str::to_owned);

        specs.push(DataSpec {
            name,
            path: path_buf,
            data_type,
            storages,
            column,
            value_column,
        });
    }

    Ok(specs)
}

fn load_fasta(
    path: &Path,
    spec: &DataSpec,
    storage: Storage,
) -> Result<SizeRow, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut stats = LoadStats::default();
    let mut builder = AnyBuilder::new(
        storage,
        format!("{}.{}.{}", spec.name, spec.column, storage.as_str()),
    );
    let mut seq = Vec::<u8>::new();
    let mut in_record = false;
    let mut original_len = 0u64;
    let mut over_row_limit = false;

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('>') {
            if in_record {
                append_record(
                    &mut builder,
                    &mut stats,
                    &mut seq,
                    original_len,
                    over_row_limit,
                );
            }
            seq.clear();
            in_record = true;
            original_len = 0;
            over_row_limit = false;
            continue;
        }

        if !in_record {
            return Err(format!("sequence before first FASTA header in {}", path.display()).into());
        }

        for mut byte in trimmed.bytes().filter(|byte| !byte.is_ascii_whitespace()) {
            original_len += 1;
            if original_len > MAX_ROW_BYTES {
                over_row_limit = true;
                continue;
            }
            byte.make_ascii_uppercase();
            seq.push(byte);
        }
    }

    if in_record {
        append_record(
            &mut builder,
            &mut stats,
            &mut seq,
            original_len,
            over_row_limit,
        );
    }

    Ok(builder.finish(spec, path, storage, stats))
}

fn load_job_csv(
    path: &Path,
    spec: &DataSpec,
    storage: Storage,
) -> Result<SizeRow, Box<dyn std::error::Error>> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_path(path)?;
    let headers = reader.headers()?.clone();
    let value_idx = select_value_column(&headers, spec.value_column.as_deref())?;
    let mut stats = LoadStats::default();
    let mut builder = AnyBuilder::new(
        storage,
        format!("{}.{}.{}", spec.name, spec.column, storage.as_str()),
    );

    for record in reader.records() {
        if stats.stopped_by_max_total_bytes {
            break;
        }
        let record = record?;
        stats.records_seen += 1;
        let value = record.get(value_idx).unwrap_or("");
        stats.total_input_sequence_bytes += value.len() as u64;

        let mut bytes = value.as_bytes().to_vec();
        if bytes.len() as u64 > MAX_ROW_BYTES {
            bytes.truncate(MAX_ROW_BYTES as usize);
            stats.records_truncated += 1;
        }
        truncate_to_total_limit(&mut bytes, &mut stats);
        if bytes.is_empty() && !value.is_empty() {
            stats.records_skipped += 1;
            continue;
        }

        stats.records_loaded += 1;
        stats.total_loaded_symbols += bytes.len() as u64;
        builder.push_bytes(&bytes);
    }

    Ok(builder.finish(spec, path, storage, stats))
}

fn append_record(
    builder: &mut AnyBuilder,
    stats: &mut LoadStats,
    seq: &mut Vec<u8>,
    original_len: u64,
    over_row_limit: bool,
) {
    if stats.stopped_by_max_total_bytes {
        return;
    }
    stats.records_seen += 1;
    stats.total_input_sequence_bytes += original_len;
    if over_row_limit {
        stats.records_truncated += 1;
    }
    truncate_to_total_limit(seq, stats);
    if seq.is_empty() && original_len > 0 {
        stats.records_skipped += 1;
        return;
    }
    stats.records_loaded += 1;
    stats.total_loaded_symbols += seq.len() as u64;
    builder.push_bytes(seq);
}

enum AnyBuilder {
    Utf8(Utf8TableBuilder),
    Fsst(FsstTableBuilder),
}

impl AnyBuilder {
    fn new(storage: Storage, name: String) -> Self {
        match storage {
            Storage::Utf8 => Self::Utf8(Utf8TableBuilder::new(name)),
            Storage::Fsst => Self::Fsst(FsstTableBuilder::new(name)),
        }
    }

    fn push_bytes(&mut self, bytes: &[u8]) {
        match self {
            Self::Utf8(builder) => builder.push_bytes(bytes),
            Self::Fsst(builder) => builder.push_bytes(bytes),
        }
    }

    fn finish(self, spec: &DataSpec, path: &Path, storage: Storage, stats: LoadStats) -> SizeRow {
        let (size, logical_payload_bytes) = match self {
            Self::Utf8(builder) => {
                let mut db = DbBuilder::new();
                let id = db.add_utf8_table(builder).expect("add UTF8 table");
                let db = db.freeze();
                let table = db.utf8_table(id).expect("get UTF8 table");
                (table.storage_size(), table.text().payload().len() as u64)
            }
            Self::Fsst(builder) => {
                let mut db = DbBuilder::new();
                let id = db.add_fsst_table(builder).expect("add FSST table");
                let db = db.freeze();
                let table = db.fsst_table(id).expect("get FSST table");
                (table.storage_size(), table.text().uncompressed_bytes())
            }
        };

        SizeRow {
            dataset: spec.name.clone(),
            column: spec.column.clone(),
            data_path: path.display().to_string(),
            data_type: spec.data_type.as_str(),
            storage: storage.as_str(),
            stats,
            size,
            logical_payload_bytes,
        }
    }
}

fn select_value_column(
    headers: &csv::StringRecord,
    requested: Option<&str>,
) -> Result<usize, String> {
    if let Some(name) = requested {
        if let Some(idx) = headers.iter().position(|header| header == name) {
            return Ok(idx);
        }
        return Err(format!("CSV column {name:?} not found"));
    }
    for name in ["value", "text", "data"] {
        if let Some(idx) = headers
            .iter()
            .position(|header| header.eq_ignore_ascii_case(name))
        {
            return Ok(idx);
        }
    }
    if headers.len() > 1 {
        Ok(1)
    } else {
        Err("CSV has too few columns".to_owned())
    }
}

fn field<'a>(
    headers: &csv::StringRecord,
    record: &'a csv::StringRecord,
    name: &str,
) -> Option<&'a str> {
    headers
        .iter()
        .position(|header| header == name)
        .and_then(|idx| record.get(idx))
}

fn truncate_to_total_limit(bytes: &mut Vec<u8>, stats: &mut LoadStats) {
    if stats.total_loaded_symbols >= MAX_TOTAL_BYTES {
        stats.stopped_by_max_total_bytes = true;
        bytes.clear();
        return;
    }
    let remaining = MAX_TOTAL_BYTES - stats.total_loaded_symbols;
    if bytes.len() as u64 > remaining {
        bytes.truncate(remaining as usize);
        stats.records_truncated += 1;
        stats.stopped_by_max_total_bytes = true;
    }
}

fn resolve_relative(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    }
}

fn parse_boolish(raw: &str) -> bool {
    matches!(
        raw.trim()
            .to_ascii_lowercase()
            .replace([' ', '_'], "-")
            .as_str(),
        "1" | "true" | "yes" | "y" | "on" | "enabled"
    )
}

fn csv_escape(value: &str) -> String {
    if value
        .chars()
        .any(|ch| matches!(ch, ',' | '"' | '\n' | '\r'))
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}
