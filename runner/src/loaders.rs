use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use db::{Db, DbBuilder, Dna2TableBuilder, DnaBase, FsstTableBuilder, TableId, Utf8TableBuilder};

use crate::cli::{InvalidDnaPolicy, RowOverflowPolicy, StorageKind};
use crate::specs::DataSpec;

#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub max_rows: Option<u64>,
    pub max_total_bytes: u64,
    pub max_row_bytes: u64,
    pub row_overflow_policy: RowOverflowPolicy,
    pub invalid_dna: InvalidDnaPolicy,
    pub uppercase_sequences: bool,
}

#[derive(Debug, Clone, Default)]
pub struct LoadStats {
    pub records_seen: u64,
    pub records_loaded: u64,
    pub records_skipped: u64,
    pub records_truncated: u64,
    pub records_invalid_dna: u64,
    pub total_input_sequence_bytes: u64,
    pub total_loaded_symbols: u64,
    pub stopped_by_max_rows: bool,
    pub stopped_by_max_total_bytes: bool,
}

#[derive(Debug, Clone)]
pub struct LoadedColumn {
    pub name: String,
    pub source_path: PathBuf,
    pub storage: StorageKind,
    pub table_id: TableId,
    pub stats: LoadStats,
    pub row_labels: Vec<String>,
}

pub struct LoadedDataset {
    pub db: Db,
    pub columns: Vec<LoadedColumn>,
}

#[derive(Debug, Default)]
struct RecordBuffer {
    header: Option<String>,
    seq: Vec<u8>,
    original_len: u64,
    over_row_limit: bool,
}

impl RecordBuffer {
    fn clear_for_header(&mut self, header: String) {
        self.header = Some(header);
        self.seq.clear();
        self.original_len = 0;
        self.over_row_limit = false;
    }

    fn push_sequence_line(&mut self, line: &[u8], options: &LoadOptions) -> Result<()> {
        let clean = line.iter().copied().filter(|b| !b.is_ascii_whitespace());
        for b in clean {
            self.original_len += 1;
            if self.original_len > options.max_row_bytes {
                self.over_row_limit = true;
                match options.row_overflow_policy {
                    RowOverflowPolicy::Truncate | RowOverflowPolicy::Skip => continue,
                    RowOverflowPolicy::Error => bail!(
                        "FASTA record exceeds --max-row-bytes ({})",
                        options.max_row_bytes
                    ),
                }
            }
            self.seq.push(b);
        }
        Ok(())
    }
}

pub fn load_fasta_column(
    path: &Path,
    dataset: &str,
    column_name: &str,
    storage: StorageKind,
    options: &LoadOptions,
) -> Result<LoadedDataset> {
    let file = File::open(path).with_context(|| format!("open FASTA file {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut ids = Utf8TableBuilder::new(format!("{dataset}.ids"));
    let mut metadata = Utf8TableBuilder::new(format!("{dataset}.metadata"));
    let mut data_utf8 = (storage == StorageKind::Utf8)
        .then(|| Utf8TableBuilder::new(format!("{dataset}.{column_name}.utf8")));
    let mut data_fsst = (storage == StorageKind::Fsst)
        .then(|| FsstTableBuilder::new(format!("{dataset}.{column_name}.fsst")));
    let mut data_dna2 = (storage == StorageKind::Dna2)
        .then(|| Dna2TableBuilder::new(format!("{dataset}.{column_name}.dna2")));

    let mut stats = LoadStats::default();
    let mut row_labels = Vec::new();
    let mut current = RecordBuffer::default();

    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read FASTA line {}", line_no + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(header) = trimmed.strip_prefix('>') {
            if let Some(prev_header) = current.header.take() {
                let stop = append_fasta_record(
                    &prev_header,
                    &current,
                    storage,
                    options,
                    &mut ids,
                    &mut metadata,
                    &mut data_utf8,
                    &mut data_fsst,
                    &mut data_dna2,
                    &mut row_labels,
                    &mut stats,
                )?;
                if stop {
                    break;
                }
            }
            current.clear_for_header(header.trim().to_owned());
        } else {
            if current.header.is_none() {
                bail!(
                    "sequence data appears before first FASTA header at line {}",
                    line_no + 1
                );
            }
            current.push_sequence_line(trimmed.as_bytes(), options)?;
        }
    }

    if let Some(prev_header) = current.header.take() {
        let _ = append_fasta_record(
            &prev_header,
            &current,
            storage,
            options,
            &mut ids,
            &mut metadata,
            &mut data_utf8,
            &mut data_fsst,
            &mut data_dna2,
            &mut row_labels,
            &mut stats,
        )?;
    }

    let mut dbb = DbBuilder::new();
    let _id_table = dbb.add_utf8_table(ids)?;
    let _metadata_table = dbb.add_utf8_table(metadata)?;
    let data_table = match storage {
        StorageKind::Utf8 => dbb.add_utf8_table(data_utf8.expect("utf8 data builder exists"))?,
        StorageKind::Fsst => dbb.add_fsst_table(data_fsst.expect("fsst data builder exists"))?,
        StorageKind::Dna2 => dbb.add_dna2_table(data_dna2.expect("dna2 data builder exists"))?,
    };

    Ok(LoadedDataset {
        db: dbb.freeze(),
        columns: vec![LoadedColumn {
            name: column_name.to_owned(),
            source_path: path.to_owned(),
            storage,
            table_id: data_table,
            stats,
            row_labels,
        }],
    })
}

#[allow(clippy::too_many_arguments)]
fn append_fasta_record(
    header: &str,
    record: &RecordBuffer,
    storage: StorageKind,
    options: &LoadOptions,
    ids: &mut Utf8TableBuilder,
    metadata: &mut Utf8TableBuilder,
    data_utf8: &mut Option<Utf8TableBuilder>,
    data_fsst: &mut Option<FsstTableBuilder>,
    data_dna2: &mut Option<Dna2TableBuilder>,
    row_labels: &mut Vec<String>,
    stats: &mut LoadStats,
) -> Result<bool> {
    if stats.stopped_by_max_rows || stats.stopped_by_max_total_bytes {
        return Ok(true);
    }

    stats.records_seen += 1;
    stats.total_input_sequence_bytes += record.original_len;

    if let Some(max_rows) = options.max_rows {
        if stats.records_loaded >= max_rows {
            stats.stopped_by_max_rows = true;
            return Ok(true);
        }
    }

    if record.over_row_limit && options.row_overflow_policy == RowOverflowPolicy::Skip {
        stats.records_skipped += 1;
        return Ok(false);
    }

    let (id, meta) = split_header(header);
    let mut seq = record.seq.clone();
    if options.uppercase_sequences {
        seq.make_ascii_uppercase();
    }
    if record.over_row_limit {
        stats.records_truncated += 1;
    }

    truncate_to_total_limit(&mut seq, stats, options.max_total_bytes);
    if seq.is_empty() && record.original_len > 0 {
        stats.records_skipped += 1;
        return Ok(stats.stopped_by_max_total_bytes);
    }

    match storage {
        StorageKind::Utf8 => {
            ids.push_str(id);
            metadata.push_str(meta);
            data_utf8
                .as_mut()
                .expect("utf8 data builder exists")
                .push_bytes(&seq);
            row_labels.push(id.to_owned());
            stats.records_loaded += 1;
            stats.total_loaded_symbols += seq.len() as u64;
        }
        StorageKind::Fsst => {
            ids.push_str(id);
            metadata.push_str(meta);
            data_fsst
                .as_mut()
                .expect("fsst data builder exists")
                .push_bytes(&seq);
            row_labels.push(id.to_owned());
            stats.records_loaded += 1;
            stats.total_loaded_symbols += seq.len() as u64;
        }
        StorageKind::Dna2 => {
            let Some(clean) = normalize_dna2_record(&seq, options.invalid_dna, stats)
                .with_context(|| format!("record {id:?} contains invalid DNA symbols"))?
            else {
                stats.records_skipped += 1;
                return Ok(stats.stopped_by_max_total_bytes);
            };

            ids.push_str(id);
            metadata.push_str(meta);
            data_dna2
                .as_mut()
                .expect("dna2 data builder exists")
                .push_ascii(&clean)?;
            row_labels.push(id.to_owned());
            stats.records_loaded += 1;
            stats.total_loaded_symbols += clean.len() as u64;
        }
    }

    Ok(stats.stopped_by_max_total_bytes)
}

pub fn load_job_csv_dataset(
    dataset: &str,
    specs: &[DataSpec],
    base: &Path,
    options: &LoadOptions,
) -> Result<LoadedDataset> {
    if specs.is_empty() {
        bail!("JOB CSV dataset {dataset:?} has no column specs");
    }

    let mut columns = Vec::<JobColumnData>::new();
    let mut reference_keys: Option<Vec<String>> = None;

    for spec in specs {
        let path = resolve_relative(base, &spec.path);
        let data = read_job_column(&path, spec, options).with_context(|| {
            format!(
                "load JOB CSV column {} from {}",
                spec.column,
                path.display()
            )
        })?;
        if let Some(keys) = reference_keys.as_ref() {
            ensure_same_keys(dataset, &spec.column, keys, &data.keys)?;
        } else {
            reference_keys = Some(data.keys.clone());
        }
        columns.push(data);
    }

    let keys = reference_keys.unwrap_or_default();
    let mut dbb = DbBuilder::new();
    let mut ids = Utf8TableBuilder::new(format!("{dataset}.ids"));
    for key in &keys {
        ids.push_str(key);
    }
    let _ids_table = dbb.add_utf8_table(ids)?;

    let mut loaded_columns = Vec::new();
    for data in columns {
        for storage in data.storages {
            let table_id = match storage {
                StorageKind::Utf8 => {
                    let mut builder =
                        Utf8TableBuilder::new(format!("{}.{}.utf8", dataset, data.column));
                    for row in &data.rows {
                        builder.push_bytes(row);
                    }
                    dbb.add_utf8_table(builder)?
                }
                StorageKind::Fsst => {
                    let mut builder =
                        FsstTableBuilder::new(format!("{}.{}.fsst", dataset, data.column));
                    for row in &data.rows {
                        builder.push_bytes(row);
                    }
                    dbb.add_fsst_table(builder)?
                }
                StorageKind::Dna2 => bail!("JOB CSV storage dna2 should be rejected earlier"),
            };
            loaded_columns.push(LoadedColumn {
                name: data.column.clone(),
                source_path: data.path.clone(),
                storage,
                table_id,
                stats: data.stats.clone(),
                row_labels: keys.clone(),
            });
        }
    }

    Ok(LoadedDataset {
        db: dbb.freeze(),
        columns: loaded_columns,
    })
}

struct JobColumnData {
    column: String,
    path: PathBuf,
    storages: Vec<StorageKind>,
    keys: Vec<String>,
    rows: Vec<Vec<u8>>,
    stats: LoadStats,
}

fn read_job_column(path: &Path, spec: &DataSpec, options: &LoadOptions) -> Result<JobColumnData> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("open CSV {}", path.display()))?;
    let headers = reader.headers()?.clone();
    let key_idx = select_column_index(
        &headers,
        spec.key_column.as_deref(),
        &["key", "id", "row_id"],
        0,
    )?;
    let value_idx = select_column_index(
        &headers,
        spec.value_column.as_deref(),
        &["value", "text", "data"],
        1,
    )?;

    let mut keys = Vec::new();
    let mut rows = Vec::new();
    let mut stats = LoadStats::default();

    for (idx, record) in reader.records().enumerate() {
        if stats.stopped_by_max_rows || stats.stopped_by_max_total_bytes {
            break;
        }
        let record = record.with_context(|| format!("parse CSV record {}", idx + 2))?;
        stats.records_seen += 1;

        if let Some(max_rows) = options.max_rows {
            if stats.records_loaded >= max_rows {
                stats.stopped_by_max_rows = true;
                break;
            }
        }

        let key = record.get(key_idx).unwrap_or("").to_owned();
        let value = record.get(value_idx).unwrap_or("");
        stats.total_input_sequence_bytes += value.len() as u64;

        let mut bytes = value.as_bytes().to_vec();
        if bytes.len() as u64 > options.max_row_bytes {
            match options.row_overflow_policy {
                RowOverflowPolicy::Truncate => {
                    bytes.truncate(options.max_row_bytes as usize);
                    stats.records_truncated += 1;
                }
                RowOverflowPolicy::Skip => {
                    stats.records_skipped += 1;
                    continue;
                }
                RowOverflowPolicy::Error => bail!(
                    "CSV row {} in {} exceeds --max-row-bytes ({})",
                    idx + 2,
                    path.display(),
                    options.max_row_bytes
                ),
            }
        }

        truncate_to_total_limit(&mut bytes, &mut stats, options.max_total_bytes);
        if bytes.is_empty() && !value.is_empty() {
            stats.records_skipped += 1;
            continue;
        }

        keys.push(key);
        stats.records_loaded += 1;
        stats.total_loaded_symbols += bytes.len() as u64;
        rows.push(bytes);
    }

    Ok(JobColumnData {
        column: spec.column.clone(),
        path: path.to_owned(),
        storages: spec.storages.clone(),
        keys,
        rows,
        stats,
    })
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

fn ensure_same_keys(
    dataset: &str,
    column: &str,
    expected: &[String],
    actual: &[String],
) -> Result<()> {
    if expected.len() != actual.len() {
        bail!(
            "JOB dataset {dataset:?} column {column:?} has {} rows, expected {}; key-aligned columns must have equal row counts",
            actual.len(),
            expected.len()
        );
    }
    for (idx, (a, b)) in expected.iter().zip(actual).enumerate() {
        if a != b {
            bail!(
                "JOB dataset {dataset:?} column {column:?} key mismatch at row {idx}: expected {a:?}, got {b:?}"
            );
        }
    }
    Ok(())
}

fn split_header(header: &str) -> (&str, &str) {
    let header = header.trim();
    match header.find(char::is_whitespace) {
        Some(idx) => (&header[..idx], header[idx..].trim_start()),
        None => (header, ""),
    }
}

fn normalize_dna2_record(
    seq: &[u8],
    policy: InvalidDnaPolicy,
    stats: &mut LoadStats,
) -> Result<Option<Vec<u8>>> {
    let mut out = Vec::with_capacity(seq.len());

    for &b in seq {
        match DnaBase::from_ascii(b) {
            Ok(base) => out.push(base.ascii()),
            Err(_) => {
                stats.records_invalid_dna += 1;
                match policy {
                    InvalidDnaPolicy::Error => {
                        return Err(anyhow!("invalid DNA byte {:?}", b as char));
                    }
                    InvalidDnaPolicy::SkipRecord => return Ok(None),
                    InvalidDnaPolicy::MapToA => out.push(b'A'),
                }
            }
        }
    }

    Ok(Some(out))
}

fn truncate_to_total_limit(bytes: &mut Vec<u8>, stats: &mut LoadStats, max_total: u64) {
    if stats.total_loaded_symbols >= max_total {
        stats.stopped_by_max_total_bytes = true;
        bytes.clear();
        return;
    }

    let remaining_total = max_total - stats.total_loaded_symbols;
    if bytes.len() as u64 > remaining_total {
        bytes.truncate(remaining_total as usize);
        stats.records_truncated += 1;
        stats.stopped_by_max_total_bytes = true;
    }
}

pub fn resolve_relative(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    }
}
