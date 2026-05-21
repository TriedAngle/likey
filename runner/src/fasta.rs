use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use db::{Db, DbBuilder, Dna2TableBuilder, DnaBase, TableId, Utf8TableBuilder};

use crate::cli::{InvalidDnaPolicy, RowOverflowPolicy, StorageKind};

#[derive(Debug, Clone)]
pub struct FastaLoadOptions {
    pub dataset: String,
    pub storage: StorageKind,
    pub max_rows: Option<u64>,
    pub max_total_bytes: u64,
    pub max_row_bytes: u64,
    pub row_overflow_policy: RowOverflowPolicy,
    pub invalid_dna: InvalidDnaPolicy,
    pub uppercase_sequences: bool,
}

#[derive(Debug, Clone, Default)]
pub struct FastaStats {
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

pub struct LoadedFasta {
    pub db: Db,
    pub id_table: TableId,
    pub metadata_table: TableId,
    pub data_table: TableId,
    pub stats: FastaStats,
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

    fn push_sequence_line(&mut self, line: &[u8], options: &FastaLoadOptions) -> Result<()> {
        let clean = line.iter().copied().filter(|b| !b.is_ascii_whitespace());
        for b in clean {
            self.original_len += 1;
            if self.original_len > options.max_row_bytes {
                self.over_row_limit = true;
                match options.row_overflow_policy {
                    RowOverflowPolicy::Truncate => continue,
                    RowOverflowPolicy::Skip => continue,
                    RowOverflowPolicy::Error => {
                        bail!(
                            "FASTA record exceeds --max-row-bytes ({})",
                            options.max_row_bytes
                        )
                    }
                }
            }
            self.seq.push(b);
        }
        Ok(())
    }
}

pub fn load_fasta(path: &Path, options: &FastaLoadOptions) -> Result<LoadedFasta> {
    let file = File::open(path).with_context(|| format!("open FASTA file {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut ids = Utf8TableBuilder::new(format!("{}.ids", options.dataset));
    let mut metadata = Utf8TableBuilder::new(format!("{}.metadata", options.dataset));
    let mut data_utf8 = (options.storage == StorageKind::Utf8)
        .then(|| Utf8TableBuilder::new(format!("{}.data.utf8", options.dataset)));
    let mut data_dna2 = (options.storage == StorageKind::Dna2)
        .then(|| Dna2TableBuilder::new(format!("{}.data.dna2", options.dataset)));

    let mut stats = FastaStats::default();
    let mut current = RecordBuffer::default();

    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read FASTA line {}", line_no + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(header) = trimmed.strip_prefix('>') {
            if let Some(prev_header) = current.header.take() {
                let stop = append_record(
                    &prev_header,
                    &current,
                    options,
                    &mut ids,
                    &mut metadata,
                    &mut data_utf8,
                    &mut data_dna2,
                    &mut stats,
                )?;
                if stop {
                    break;
                }
            }
            current.clear_for_header(header.trim().to_owned());
        } else {
            if current.header.is_none() {
                bail!("sequence data appears before first FASTA header at line {}", line_no + 1);
            }
            current.push_sequence_line(trimmed.as_bytes(), options)?;
        }
    }

    if let Some(prev_header) = current.header.take() {
        let _ = append_record(
            &prev_header,
            &current,
            options,
            &mut ids,
            &mut metadata,
            &mut data_utf8,
            &mut data_dna2,
            &mut stats,
        )?;
    }

    let mut dbb = DbBuilder::new();
    let id_table = dbb.add_utf8_table(ids)?;
    let metadata_table = dbb.add_utf8_table(metadata)?;
    let data_table = match options.storage {
        StorageKind::Utf8 => dbb.add_utf8_table(data_utf8.expect("utf8 data builder exists"))?,
        StorageKind::Dna2 => dbb.add_dna2_table(data_dna2.expect("dna2 data builder exists"))?,
    };

    Ok(LoadedFasta {
        db: dbb.freeze(),
        id_table,
        metadata_table,
        data_table,
        stats,
    })
}

#[allow(clippy::too_many_arguments)]
fn append_record(
    header: &str,
    record: &RecordBuffer,
    options: &FastaLoadOptions,
    ids: &mut Utf8TableBuilder,
    metadata: &mut Utf8TableBuilder,
    data_utf8: &mut Option<Utf8TableBuilder>,
    data_dna2: &mut Option<Dna2TableBuilder>,
    stats: &mut FastaStats,
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

    if stats.total_loaded_symbols >= options.max_total_bytes {
        stats.stopped_by_max_total_bytes = true;
        return Ok(true);
    }

    let remaining_total = options.max_total_bytes - stats.total_loaded_symbols;
    if seq.len() as u64 > remaining_total {
        seq.truncate(remaining_total as usize);
        stats.records_truncated += 1;
        stats.stopped_by_max_total_bytes = true;
    }

    if seq.is_empty() && record.original_len > 0 {
        stats.records_skipped += 1;
        return Ok(stats.stopped_by_max_total_bytes);
    }

    match options.storage {
        StorageKind::Utf8 => {
            ids.push_str(id);
            metadata.push_str(meta);
            data_utf8
                .as_mut()
                .expect("utf8 data builder exists")
                .push_bytes(&seq);
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
            stats.records_loaded += 1;
            stats.total_loaded_symbols += clean.len() as u64;
        }
    }

    Ok(stats.stopped_by_max_total_bytes)
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
    stats: &mut FastaStats,
) -> Result<Option<Vec<u8>>> {
    let mut out = Vec::with_capacity(seq.len());

    for &b in seq {
        match DnaBase::from_ascii(b) {
            Ok(base) => out.push(base.ascii()),
            Err(_) => {
                stats.records_invalid_dna += 1;
                match policy {
                    InvalidDnaPolicy::Error => return Err(anyhow!("invalid DNA byte {:?}", b as char)),
                    InvalidDnaPolicy::SkipRecord => return Ok(None),
                    InvalidDnaPolicy::MapToA => out.push(b'A'),
                }
            }
        }
    }

    Ok(Some(out))
}
