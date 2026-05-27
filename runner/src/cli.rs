use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};

#[derive(Debug, Parser)]
#[command(author, version, about)]
pub struct Args {
    /// CSV file describing data inputs. Columns:
    /// name,path,type,storage,column\[,key_column,value_column,enabled\].
    /// type supports dna-fasta, protein-fasta, and job-csv.
    #[arg(long)]
    pub data_csv: PathBuf,

    /// CSV file listing algorithms to run. Columns: algorithm\[,enabled\].
    #[arg(long)]
    pub algorithms_csv: PathBuf,

    /// CSV file listing LIKE patterns. Columns: name,pattern\[,enabled\].
    #[arg(long)]
    pub patterns_csv: PathBuf,

    /// Optional CSV file listing candidate sources/indexes. Columns: index\[,enabled\].
    /// If omitted, only full-scan is run.
    #[arg(long)]
    pub indexes_csv: Option<PathBuf>,

    /// Output CSV containing one row per measured query iteration.
    #[arg(long, default_value = "bench-raw.csv")]
    pub output_csv: PathBuf,

    /// Optional summary CSV grouped by dataset/column/storage/algorithm/index/pattern.
    #[arg(long)]
    pub summary_csv: Option<PathBuf>,

    /// Optional row-level profile CSV. This runs a separate profiling pass and
    /// intentionally does not affect normal query timings.
    #[arg(long)]
    pub row_profile_csv: Option<PathBuf>,

    /// Repetitions per row during row profiling. Higher values reduce timer noise
    /// but make row profiling much more expensive.
    #[arg(long, default_value_t = 1)]
    pub row_profile_repeats: usize,

    /// Maximum rows to profile per dataset column. Omit to profile every row.
    #[arg(long)]
    pub row_profile_max_rows: Option<u64>,

    /// Store at most this many bytes/chars of the row value in row_profile_csv.
    #[arg(long, default_value_t = 120)]
    pub row_profile_sample_bytes: usize,

    /// Number of untimed warmup executions per combination.
    #[arg(long, default_value_t = 1)]
    pub warmups: usize,

    /// Number of timed executions per combination.
    #[arg(long, default_value_t = 5)]
    pub iterations: usize,

    /// Candidate batch size in rows for full scan and row-list indexes.
    #[arg(long, default_value_t = 4096)]
    pub batch_rows: usize,

    /// Maximum loaded rows per dataset/storage. Omit for unbounded.
    #[arg(long)]
    pub max_rows: Option<u64>,

    /// Maximum loaded sequence/value bytes across all rows per dataset column.
    /// Accepts plain bytes or units like 1GB, 512MiB, 100mb.
    #[arg(long, value_parser = parse_bytes, default_value = "1GiB")]
    pub max_total_bytes: u64,

    /// Maximum loaded sequence/value bytes per row.
    /// Accepts plain bytes or units like 50MB, 16MiB.
    #[arg(long, value_parser = parse_bytes, default_value = "50MiB")]
    pub max_row_bytes: u64,

    /// What to do if one input row exceeds --max-row-bytes.
    #[arg(long, value_enum, default_value_t = RowOverflowPolicy::Truncate)]
    pub row_overflow_policy: RowOverflowPolicy,

    /// Policy for non-ACGT bases when loading DNA2. UTF-8 storage keeps bytes unchanged.
    #[arg(long, value_enum, default_value_t = InvalidDnaPolicy::SkipRecord)]
    pub invalid_dna: InvalidDnaPolicy,

    /// Uppercase FASTA sequence lines while loading. CSV values are left unchanged.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub uppercase_sequences: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    DnaFasta,
    ProteinFasta,
    JobCsv,
}

impl DataType {
    pub const fn as_str(self) -> &'static str {
        match self {
            DataType::DnaFasta => "dna-fasta",
            DataType::ProteinFasta => "protein-fasta",
            DataType::JobCsv => "job-csv",
        }
    }

    pub const fn default_storage_raw(self) -> &'static str {
        match self {
            DataType::DnaFasta => "both",
            DataType::ProteinFasta | DataType::JobCsv => "utf8",
        }
    }

    pub const fn allows_dna2(self) -> bool {
        matches!(self, DataType::DnaFasta)
    }
}

impl FromStr for DataType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match normalize_name(s).as_str() {
            "dna-fasta" | "dna" | "fasta" | "fa" | "fna" => Ok(Self::DnaFasta),
            "protein-fasta" | "protein" | "aa-fasta" | "faa" | "pep" => Ok(Self::ProteinFasta),
            "job-csv" | "job" | "csv" | "key-value" | "keyvalue" | "kv-csv" => Ok(Self::JobCsv),
            other => {
                bail!("unknown data type {other:?}; supported: dna-fasta, protein-fasta, job-csv")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageKind {
    Utf8,
    Dna2,
}

impl StorageKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            StorageKind::Utf8 => "utf8",
            StorageKind::Dna2 => "dna2",
        }
    }
}

impl FromStr for StorageKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match normalize_name(s).as_str() {
            "utf8" | "utf-8" | "bytes" | "byte" => Ok(Self::Utf8),
            "dna2" | "dna-2" | "bitpack" | "bitpacked" | "2bit" | "2-bit" => Ok(Self::Dna2),
            other => bail!("unknown storage {other:?}; supported: utf8, dna2, both"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexKind {
    FullScan,
    Fm,
    Trigram,
}

impl IndexKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            IndexKind::FullScan => "full-scan",
            IndexKind::Fm => "fm",
            IndexKind::Trigram => "trigram",
        }
    }
}

impl FromStr for IndexKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match normalize_name(s).as_str() {
            "none" | "scan" | "fullscan" | "full-scan" | "full_scan" => Ok(Self::FullScan),
            "fm" | "fmindex" | "fm-index" | "fm_index" => Ok(Self::Fm),
            "trigram" | "tri" => Ok(Self::Trigram),
            other => bail!("unknown index {other:?}; supported: none/full-scan, fm, trigram"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlgorithmKind {
    Std,
    Kmp,
    Naive,
    NaiveScalar,
    NaiveVectorized,
    NaiveVectorizedV2,
    NaiveAvx2,
    NaiveAvx2V2,
    NaiveAvx512,
    NaiveAvx512V2,
    NaiveAuto,
    NaiveMixed,
    NaiveWildcard,
    NaiveScalarWildcard,
    NaiveVectorizedWildcard,
    NaiveVectorizedV2Wildcard,
    NaiveAvx2Wildcard,
    NaiveAvx2V2Wildcard,
    NaiveAvx512Wildcard,
    NaiveAvx512V2Wildcard,
    NaiveAutoWildcard,
    NaiveMixedWildcard,
    Bm,
    TwoWay,
    TwoWay2,
    LibcMemmem,
    Fft0,
    Fft1,
    Dna2Naive,
    Dna2PackedScalar,
    Dna2PackedVectorized,
}

impl AlgorithmKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            AlgorithmKind::Std => "std",
            AlgorithmKind::Kmp => "kmp",
            AlgorithmKind::Naive => "naive",
            AlgorithmKind::NaiveScalar => "naive-scalar",
            AlgorithmKind::NaiveVectorized => "naive-vectorized",
            AlgorithmKind::NaiveVectorizedV2 => "naive-vectorized-v2",
            AlgorithmKind::NaiveAvx2 => "naive-avx2",
            AlgorithmKind::NaiveAvx2V2 => "naive-avx2-v2",
            AlgorithmKind::NaiveAvx512 => "naive-avx512",
            AlgorithmKind::NaiveAvx512V2 => "naive-avx512-v2",
            AlgorithmKind::NaiveAuto => "naive-auto",
            AlgorithmKind::NaiveMixed => "naive-mixed",
            AlgorithmKind::NaiveWildcard => "naive-wildcard",
            AlgorithmKind::NaiveScalarWildcard => "naive-scalar-wildcard",
            AlgorithmKind::NaiveVectorizedWildcard => "naive-vectorized-wildcard",
            AlgorithmKind::NaiveVectorizedV2Wildcard => "naive-vectorized-v2-wildcard",
            AlgorithmKind::NaiveAvx2Wildcard => "naive-avx2-wildcard",
            AlgorithmKind::NaiveAvx2V2Wildcard => "naive-avx2-v2-wildcard",
            AlgorithmKind::NaiveAvx512Wildcard => "naive-avx512-wildcard",
            AlgorithmKind::NaiveAvx512V2Wildcard => "naive-avx512-v2-wildcard",
            AlgorithmKind::NaiveAutoWildcard => "naive-auto-wildcard",
            AlgorithmKind::NaiveMixedWildcard => "naive-mixed-wildcard",
            AlgorithmKind::Bm => "bm",
            AlgorithmKind::TwoWay => "two-way",
            AlgorithmKind::TwoWay2 => "two-way2",
            AlgorithmKind::LibcMemmem => "libc-memmem",
            AlgorithmKind::Fft0 => "fft0",
            AlgorithmKind::Fft1 => "fft1",
            AlgorithmKind::Dna2Naive => "dna2-naive",
            AlgorithmKind::Dna2PackedScalar => "dna2-packed-scalar",
            AlgorithmKind::Dna2PackedVectorized => "dna2-packed-vectorized",
        }
    }

    pub const fn is_compatible(self, storage: StorageKind) -> bool {
        match storage {
            StorageKind::Utf8 => !matches!(
                self,
                AlgorithmKind::Dna2Naive
                    | AlgorithmKind::Dna2PackedScalar
                    | AlgorithmKind::Dna2PackedVectorized
            ),
            StorageKind::Dna2 => matches!(
                self,
                AlgorithmKind::Dna2Naive
                    | AlgorithmKind::Dna2PackedScalar
                    | AlgorithmKind::Dna2PackedVectorized
            ),
        }
    }
}

impl FromStr for AlgorithmKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match normalize_name(s).as_str() {
            "std" | "std-search" | "std_search" => Ok(Self::Std),
            "kmp" | "utf8-kmp" | "utf8_kmp" => Ok(Self::Kmp),
            "naive" => Ok(Self::Naive),
            "naive-scalar" | "naive_scalar" => Ok(Self::NaiveScalar),
            "naive-vectorized" | "naive_vectorized" => Ok(Self::NaiveVectorized),
            "naive-vectorized-v2" | "naive_vectorized_v2" | "naive-v2" | "naive_v2" => {
                Ok(Self::NaiveVectorizedV2)
            }
            "naive-avx2" | "naive_avx2" | "avx2" => Ok(Self::NaiveAvx2),
            "naive-avx2-v2" | "naive_avx2_v2" | "avx2-v2" | "avx2_v2" => Ok(Self::NaiveAvx2V2),
            "naive-avx512" | "naive_avx512" | "avx512" => Ok(Self::NaiveAvx512),
            "naive-avx512-v2" | "naive_avx512_v2" | "avx512-v2" | "avx512_v2" => {
                Ok(Self::NaiveAvx512V2)
            }
            "naive-auto" | "naive_auto" | "auto" => Ok(Self::NaiveAuto),
            "naive-mixed" | "naive_mixed" => Ok(Self::NaiveMixed),
            "naive-wildcard" | "naive_wildcard" => Ok(Self::NaiveWildcard),
            "naive-scalar-wildcard" | "naive_scalar_wildcard" => Ok(Self::NaiveScalarWildcard),
            "naive-vectorized-wildcard"
            | "naive_vectorized_wildcard"
            | "naive-wildcard-vectorized"
            | "naive_wildcard_vectorized" => Ok(Self::NaiveVectorizedWildcard),
            "naive-vectorized-v2-wildcard"
            | "naive_vectorized_v2_wildcard"
            | "naive-v2-wildcard"
            | "naive_v2_wildcard"
            | "naive-wildcard-vectorized-v2"
            | "naive_wildcard_vectorized_v2"
            | "naive-wildcard-v2"
            | "naive_wildcard_v2" => Ok(Self::NaiveVectorizedV2Wildcard),
            "naive-avx2-wildcard"
            | "naive_avx2_wildcard"
            | "naive-wildcard-avx2"
            | "naive_wildcard_avx2"
            | "avx2-wildcard"
            | "avx2_wildcard" => Ok(Self::NaiveAvx2Wildcard),
            "naive-avx2-v2-wildcard"
            | "naive_avx2_v2_wildcard"
            | "naive-wildcard-avx2-v2"
            | "naive_wildcard_avx2_v2"
            | "avx2-v2-wildcard"
            | "avx2_v2_wildcard" => Ok(Self::NaiveAvx2V2Wildcard),
            "naive-avx512-wildcard"
            | "naive_avx512_wildcard"
            | "naive-wildcard-avx512"
            | "naive_wildcard_avx512"
            | "avx512-wildcard"
            | "avx512_wildcard" => Ok(Self::NaiveAvx512Wildcard),
            "naive-avx512-v2-wildcard"
            | "naive_avx512_v2_wildcard"
            | "naive-wildcard-avx512-v2"
            | "naive_wildcard_avx512_v2"
            | "avx512-v2-wildcard"
            | "avx512_v2_wildcard" => Ok(Self::NaiveAvx512V2Wildcard),
            "naive-auto-wildcard"
            | "naive_auto_wildcard"
            | "naive-wildcard-auto"
            | "naive_wildcard_auto"
            | "auto-wildcard"
            | "auto_wildcard" => Ok(Self::NaiveAutoWildcard),
            "naive-mixed-wildcard"
            | "naive_mixed_wildcard"
            | "naive-wildcard-mixed"
            | "naive_wildcard_mixed" => Ok(Self::NaiveMixedWildcard),
            "bm" | "boyer-moore" | "boyer_moore" => Ok(Self::Bm),
            "two-way" | "two_way" | "twoway" => Ok(Self::TwoWay),
            "two-way2" | "two_way2" | "twoway2" => Ok(Self::TwoWay2),
            "libc-memmem" | "libc_memmem" | "memmem" => Ok(Self::LibcMemmem),
            "fft0" | "fft-str0" | "fft_str0" | "fftstr0" => Ok(Self::Fft0),
            "fft1" | "fft-str1" | "fft_str1" | "fftstr1" | "fft" => Ok(Self::Fft1),
            "dna2-naive" | "dna2_naive" | "dna" | "dna2" => Ok(Self::Dna2Naive),
            "dna2-packed-scalar" | "dna2_packed_scalar" => Ok(Self::Dna2PackedScalar),
            "dna2-packed-vectorized" | "dna2_packed_vectorized" => Ok(Self::Dna2PackedVectorized),
            other => bail!("unknown algorithm {other:?}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
pub enum InvalidDnaPolicy {
    Error,
    SkipRecord,
    MapToA,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
pub enum RowOverflowPolicy {
    Truncate,
    Skip,
    Error,
}

fn normalize_name(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '_'], "-")
}

pub fn parse_boolish(s: &str) -> bool {
    matches!(
        normalize_name(s).as_str(),
        "1" | "true" | "yes" | "y" | "on" | "enabled"
    )
}

pub fn parse_bytes(s: &str) -> Result<u64> {
    let s = s.trim();
    if s.is_empty() {
        bail!("byte value cannot be empty");
    }

    let mut split = 0usize;
    for (idx, ch) in s.char_indices() {
        if !(ch.is_ascii_digit() || ch == '.') {
            split = idx;
            break;
        }
    }
    if split == 0 && s.chars().all(|c| c.is_ascii_digit() || c == '.') {
        split = s.len();
    }

    let number = &s[..split];
    let unit = s[split..].trim().to_ascii_lowercase();
    let value: f64 = number.parse()?;
    if value < 0.0 || !value.is_finite() {
        bail!("invalid byte value {s:?}");
    }

    let multiplier = match unit.as_str() {
        "" | "b" | "byte" | "bytes" => 1.0,
        "k" | "kb" => 1000.0,
        "m" | "mb" => 1000.0 * 1000.0,
        "g" | "gb" => 1000.0 * 1000.0 * 1000.0,
        "t" | "tb" => 1000.0 * 1000.0 * 1000.0 * 1000.0,
        "ki" | "kib" => 1024.0,
        "mi" | "mib" => 1024.0 * 1024.0,
        "gi" | "gib" => 1024.0 * 1024.0 * 1024.0,
        "ti" | "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        other => bail!("unknown byte unit {other:?} in {s:?}"),
    };

    Ok((value * multiplier).floor() as u64)
}
