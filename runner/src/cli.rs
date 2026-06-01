use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{bail, Result};
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

    /// Print coarse FM-index build progress to stderr.
    #[arg(long)]
    pub fm_build_progress: bool,
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
            DataType::DnaFasta => "all",
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
    Fsst,
    Dna2,
}

impl StorageKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            StorageKind::Utf8 => "utf8",
            StorageKind::Fsst => "fsst",
            StorageKind::Dna2 => "dna2",
        }
    }
}

impl FromStr for StorageKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match normalize_name(s).as_str() {
            "utf8" | "utf-8" | "bytes" | "byte" => Ok(Self::Utf8),
            "fsst" => Ok(Self::Fsst),
            "dna2" | "dna-2" | "bitpack" | "bitpacked" | "2bit" | "2-bit" => Ok(Self::Dna2),
            other => bail!("unknown storage {other:?}; supported: utf8, fsst, dna2, all"),
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
    StdSearch,
    Utf8Kmp,
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
    BM,
    TwoWay,
    TwoWay2,
    LibcMemmem,
    FftStr0,
    FftStr1,
    Dna2,
    Dna2PackedScalar,
    Dna2PackedVectorized,
}

impl AlgorithmKind {
    pub const ALL: [Self; 31] = [
        Self::StdSearch,
        Self::Utf8Kmp,
        Self::Naive,
        Self::NaiveScalar,
        Self::NaiveVectorized,
        Self::NaiveVectorizedV2,
        Self::NaiveAvx2,
        Self::NaiveAvx2V2,
        Self::NaiveAvx512,
        Self::NaiveAvx512V2,
        Self::NaiveAuto,
        Self::NaiveMixed,
        Self::NaiveWildcard,
        Self::NaiveScalarWildcard,
        Self::NaiveVectorizedWildcard,
        Self::NaiveVectorizedV2Wildcard,
        Self::NaiveAvx2Wildcard,
        Self::NaiveAvx2V2Wildcard,
        Self::NaiveAvx512Wildcard,
        Self::NaiveAvx512V2Wildcard,
        Self::NaiveAutoWildcard,
        Self::NaiveMixedWildcard,
        Self::BM,
        Self::TwoWay,
        Self::TwoWay2,
        Self::LibcMemmem,
        Self::FftStr0,
        Self::FftStr1,
        Self::Dna2,
        Self::Dna2PackedScalar,
        Self::Dna2PackedVectorized,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            AlgorithmKind::StdSearch => "StdSearch",
            AlgorithmKind::Utf8Kmp => "Utf8Kmp",
            AlgorithmKind::Naive => "Naive",
            AlgorithmKind::NaiveScalar => "NaiveScalar",
            AlgorithmKind::NaiveVectorized => "NaiveVectorized",
            AlgorithmKind::NaiveVectorizedV2 => "NaiveVectorizedV2",
            AlgorithmKind::NaiveAvx2 => "NaiveAvx2",
            AlgorithmKind::NaiveAvx2V2 => "NaiveAvx2V2",
            AlgorithmKind::NaiveAvx512 => "NaiveAvx512",
            AlgorithmKind::NaiveAvx512V2 => "NaiveAvx512V2",
            AlgorithmKind::NaiveAuto => "NaiveAuto",
            AlgorithmKind::NaiveMixed => "NaiveMixed",
            AlgorithmKind::NaiveWildcard => "NaiveWildcard",
            AlgorithmKind::NaiveScalarWildcard => "NaiveScalarWildcard",
            AlgorithmKind::NaiveVectorizedWildcard => "NaiveVectorizedWildcard",
            AlgorithmKind::NaiveVectorizedV2Wildcard => "NaiveVectorizedV2Wildcard",
            AlgorithmKind::NaiveAvx2Wildcard => "NaiveAvx2Wildcard",
            AlgorithmKind::NaiveAvx2V2Wildcard => "NaiveAvx2V2Wildcard",
            AlgorithmKind::NaiveAvx512Wildcard => "NaiveAvx512Wildcard",
            AlgorithmKind::NaiveAvx512V2Wildcard => "NaiveAvx512V2Wildcard",
            AlgorithmKind::NaiveAutoWildcard => "NaiveAutoWildcard",
            AlgorithmKind::NaiveMixedWildcard => "NaiveMixedWildcard",
            AlgorithmKind::BM => "BM",
            AlgorithmKind::TwoWay => "TwoWay",
            AlgorithmKind::TwoWay2 => "TwoWay2",
            AlgorithmKind::LibcMemmem => "LibcMemmem",
            AlgorithmKind::FftStr0 => "FftStr0",
            AlgorithmKind::FftStr1 => "FftStr1",
            AlgorithmKind::Dna2 => "Dna2",
            AlgorithmKind::Dna2PackedScalar => "Dna2PackedScalar",
            AlgorithmKind::Dna2PackedVectorized => "Dna2PackedVectorized",
        }
    }

    pub const fn is_compatible(self, storage: StorageKind) -> bool {
        match storage {
            StorageKind::Utf8 => !matches!(
                self,
                AlgorithmKind::Dna2
                    | AlgorithmKind::Dna2PackedScalar
                    | AlgorithmKind::Dna2PackedVectorized
            ),
            StorageKind::Fsst => !matches!(
                self,
                AlgorithmKind::FftStr0
                    | AlgorithmKind::FftStr1
                    | AlgorithmKind::Dna2
                    | AlgorithmKind::Dna2PackedScalar
                    | AlgorithmKind::Dna2PackedVectorized
            ),
            StorageKind::Dna2 => matches!(
                self,
                AlgorithmKind::Dna2
                    | AlgorithmKind::Dna2PackedScalar
                    | AlgorithmKind::Dna2PackedVectorized
            ),
        }
    }
}

impl FromStr for AlgorithmKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match Self::ALL
            .into_iter()
            .find(|algorithm| algorithm.as_str() == s)
        {
            Some(algorithm) => Ok(algorithm),
            None => bail!("unknown algorithm {s:?}"),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn algorithm_kind_matches_exact_names_only() {
        assert_eq!(
            "StdSearch".parse::<AlgorithmKind>().unwrap(),
            AlgorithmKind::StdSearch
        );
        assert_eq!(
            "Utf8Kmp".parse::<AlgorithmKind>().unwrap(),
            AlgorithmKind::Utf8Kmp
        );
        assert_eq!("BM".parse::<AlgorithmKind>().unwrap(), AlgorithmKind::BM);
        assert_eq!(
            "Dna2PackedScalar".parse::<AlgorithmKind>().unwrap(),
            AlgorithmKind::Dna2PackedScalar
        );

        assert!("std".parse::<AlgorithmKind>().is_err());
        assert!("std-search".parse::<AlgorithmKind>().is_err());
        assert!("utf8-kmp".parse::<AlgorithmKind>().is_err());
        assert!("naive-avx2".parse::<AlgorithmKind>().is_err());
        assert!("Dna2PackedScalar ".parse::<AlgorithmKind>().is_err());
    }
}
