use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;

use runner::{Args, DataType, StorageKind, PatternSpec, AlgorithmKind, IndexKind, BenchRow};
use runner::{load_fasta, FastaLoadOptions};
use runner::{
    build_indexes, run_dna2_algorithm, run_utf8_algorithm, write_rows, write_summary, BenchConfig,
};
use runner::{load_algorithms, load_data_specs, load_indexes, load_patterns};

fn main() -> Result<()> {
    let args = Args::parse();

    if args.iterations == 0 {
        bail!("--iterations must be greater than zero");
    }
    if args.batch_rows == 0 {
        bail!("--batch-rows must be greater than zero");
    }
    if args.max_row_bytes == 0 {
        bail!("--max-row-bytes must be greater than zero");
    }
    if args.max_total_bytes == 0 {
        bail!("--max-total-bytes must be greater than zero");
    }

    ensure_parent_dir(&args.output_csv)?;
    if let Some(summary) = args.summary_csv.as_deref() {
        ensure_parent_dir(summary)?;
    }

    let data_specs = load_data_specs(&args.data_csv)?;
    let algorithms = load_algorithms(&args.algorithms_csv)?;
    let patterns = load_patterns(&args.patterns_csv)?;
    let indexes = load_indexes(args.indexes_csv.as_deref())?;

    eprintln!(
        "benchmark matrix: datasets={}, algorithms={}, patterns={}, indexes={}",
        data_specs.len(),
        algorithms.len(),
        patterns.len(),
        indexes.len()
    );

    let data_base = args.data_csv.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
    let mut rows = Vec::new();

    for dataset in data_specs {
        let data_path = resolve_relative(&data_base, &dataset.path);
        for storage in dataset.storages {
            match dataset.data_type {
                DataType::Fasta => {
                    run_fasta_dataset(&args, &dataset.name, &data_path, storage, &algorithms, &patterns, &indexes, &mut rows)
                        .with_context(|| {
                            format!(
                                "run dataset={} storage={} path={}",
                                dataset.name,
                                storage.as_str(),
                                data_path.display()
                            )
                        })?;
                }
            }
        }
    }

    if rows.is_empty() {
        bail!("no benchmark rows were produced; check storage/algorithm compatibility and CSV enabled flags");
    }

    write_rows(&args.output_csv, &rows)
        .with_context(|| format!("write result CSV {}", args.output_csv.display()))?;

    if let Some(summary_path) = args.summary_csv.as_deref() {
        write_summary(summary_path, &rows)
            .with_context(|| format!("write summary CSV {}", summary_path.display()))?;
    }

    eprintln!("wrote {} measured runs to {}", rows.len(), args.output_csv.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_fasta_dataset(
    args: &Args,
    dataset_name: &str,
    data_path: &Path,
    storage: StorageKind,
    algorithms: &[AlgorithmKind],
    patterns: &[PatternSpec],
    indexes: &[IndexKind],
    rows: &mut Vec<BenchRow>,
) -> Result<()> {
    let compatible_algorithms = algorithms
        .iter()
        .copied()
        .filter(|a| a.is_compatible(storage))
        .collect::<Vec<_>>();

    if compatible_algorithms.is_empty() {
        eprintln!(
            "skipping dataset={} storage={} because no listed algorithms are compatible",
            dataset_name,
            storage.as_str()
        );
        return Ok(());
    }

    let load_options = FastaLoadOptions {
        dataset: dataset_name.to_owned(),
        storage,
        max_rows: args.max_rows,
        max_total_bytes: args.max_total_bytes,
        max_row_bytes: args.max_row_bytes,
        row_overflow_policy: args.row_overflow_policy,
        invalid_dna: args.invalid_dna,
        uppercase_sequences: args.uppercase_sequences,
    };

    let load_start = Instant::now();
    let loaded = load_fasta(data_path, &load_options)?;
    let load_ns = load_start.elapsed().as_nanos();

    eprintln!(
        "loaded dataset={} storage={} seen={} loaded={} skipped={} truncated={} symbols={} load_ns={}",
        dataset_name,
        storage.as_str(),
        loaded.stats.records_seen,
        loaded.stats.records_loaded,
        loaded.stats.records_skipped,
        loaded.stats.records_truncated,
        loaded.stats.total_loaded_symbols,
        load_ns
    );

    match storage {
        StorageKind::Utf8 => {
            let table = loaded.db.utf8_table(loaded.data_table).context("data table is not UTF-8")?;
            let column = table.text();
            let built_indexes = build_indexes(&column, indexes)?;
            let config = BenchConfig {
                dataset: dataset_name.to_owned(),
                data_path: data_path.to_owned(),
                data_type: "fasta".to_owned(),
                storage,
                algorithms: compatible_algorithms.clone(),
                indexes: indexes.to_vec(),
                patterns: patterns.to_vec(),
                warmups: args.warmups,
                iterations: args.iterations,
                batch_rows: args.batch_rows,
                load_ns,
                fasta_stats: loaded.stats.clone(),
            };

            for algorithm in compatible_algorithms {
                run_utf8_algorithm(&column, algorithm, &built_indexes, &config, rows)
                    .with_context(|| format!("run UTF-8 algorithm {}", algorithm.as_str()))?;
            }
        }
        StorageKind::Dna2 => {
            let table = loaded.db.dna2_table(loaded.data_table).context("data table is not DNA2")?;
            let column = table.sequence();
            let built_indexes = build_indexes(&column, indexes)?;
            let config = BenchConfig {
                dataset: dataset_name.to_owned(),
                data_path: data_path.to_owned(),
                data_type: "fasta".to_owned(),
                storage,
                algorithms: compatible_algorithms.clone(),
                indexes: indexes.to_vec(),
                patterns: patterns.to_vec(),
                warmups: args.warmups,
                iterations: args.iterations,
                batch_rows: args.batch_rows,
                load_ns,
                fasta_stats: loaded.stats.clone(),
            };

            for algorithm in compatible_algorithms {
                run_dna2_algorithm(&column, algorithm, &built_indexes, &config, rows)
                    .with_context(|| format!("run DNA2 algorithm {}", algorithm.as_str()))?;
            }
        }
    }

    Ok(())
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create output directory {}", parent.display()))?;
        }
    }
    Ok(())
}

fn resolve_relative(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    }
}
