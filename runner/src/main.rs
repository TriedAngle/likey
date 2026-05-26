mod cli;
mod loaders;
mod runner;
mod specs;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;

use crate::cli::{Args, DataType, StorageKind};
use crate::loaders::{
    LoadOptions, LoadedColumn, LoadedDataset, load_fasta_column, load_job_csv_dataset,
    resolve_relative,
};
use crate::runner::{
    BenchConfig, BenchRow, RowProfileRow, build_indexes, run_dna2_algorithm, run_utf8_algorithm,
    write_row_profiles, write_rows, write_summary,
};
use crate::specs::{DataSpec, load_algorithms, load_data_specs, load_indexes, load_patterns};

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
    if args.row_profile_repeats == 0 {
        bail!("--row-profile-repeats must be greater than zero");
    }

    ensure_parent_dir(&args.output_csv)?;
    if let Some(summary) = args.summary_csv.as_deref() {
        ensure_parent_dir(summary)?;
    }
    if let Some(profile) = args.row_profile_csv.as_deref() {
        ensure_parent_dir(profile)?;
    }

    let data_specs = load_data_specs(&args.data_csv)?;
    let algorithms = load_algorithms(&args.algorithms_csv)?;
    let patterns = load_patterns(&args.patterns_csv)?;
    let indexes = load_indexes(args.indexes_csv.as_deref())?;

    eprintln!(
        "benchmark inputs: data_specs={}, algorithms={}, patterns={}, indexes={}",
        data_specs.len(),
        algorithms.len(),
        patterns.len(),
        indexes.len()
    );

    let data_base = args
        .data_csv
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let load_options = LoadOptions {
        max_rows: args.max_rows,
        max_total_bytes: args.max_total_bytes,
        max_row_bytes: args.max_row_bytes,
        row_overflow_policy: args.row_overflow_policy,
        invalid_dna: args.invalid_dna,
        uppercase_sequences: args.uppercase_sequences,
    };

    let mut bench_rows = Vec::<BenchRow>::new();
    let mut profile_rows = Vec::<RowProfileRow>::new();

    run_fasta_specs(
        &args,
        &data_base,
        &data_specs,
        &load_options,
        &algorithms,
        &patterns,
        &indexes,
        &mut bench_rows,
        &mut profile_rows,
    )?;

    run_job_specs(
        &args,
        &data_base,
        &data_specs,
        &load_options,
        &algorithms,
        &patterns,
        &indexes,
        &mut bench_rows,
        &mut profile_rows,
    )?;

    if bench_rows.is_empty() {
        bail!(
            "no benchmark rows were produced; check storage/algorithm compatibility and CSV enabled flags"
        );
    }

    write_rows(&args.output_csv, &bench_rows)
        .with_context(|| format!("write result CSV {}", args.output_csv.display()))?;

    if let Some(summary_path) = args.summary_csv.as_deref() {
        write_summary(summary_path, &bench_rows)
            .with_context(|| format!("write summary CSV {}", summary_path.display()))?;
    }

    if let Some(profile_path) = args.row_profile_csv.as_deref() {
        write_row_profiles(profile_path, &profile_rows)
            .with_context(|| format!("write row profile CSV {}", profile_path.display()))?;
    }

    eprintln!(
        "wrote {} measured runs to {}",
        bench_rows.len(),
        args.output_csv.display()
    );
    if let Some(path) = args.row_profile_csv.as_deref() {
        eprintln!(
            "wrote {} row profile rows to {}",
            profile_rows.len(),
            path.display()
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_fasta_specs(
    args: &Args,
    data_base: &Path,
    data_specs: &[DataSpec],
    load_options: &LoadOptions,
    algorithms: &[crate::cli::AlgorithmKind],
    patterns: &[crate::specs::PatternSpec],
    indexes: &[crate::cli::IndexKind],
    bench_rows: &mut Vec<BenchRow>,
    profile_rows: &mut Vec<RowProfileRow>,
) -> Result<()> {
    for spec in data_specs
        .iter()
        .filter(|s| matches!(s.data_type, DataType::DnaFasta | DataType::ProteinFasta))
    {
        let data_path = resolve_relative(data_base, &spec.path);
        for &storage in &spec.storages {
            let load_start = Instant::now();
            let loaded =
                load_fasta_column(&data_path, &spec.name, &spec.column, storage, load_options)?;
            let load_ns = load_start.elapsed().as_nanos();
            run_loaded_dataset(
                args,
                &spec.name,
                spec.data_type.as_str(),
                loaded,
                load_ns,
                algorithms,
                patterns,
                indexes,
                bench_rows,
                profile_rows,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_job_specs(
    args: &Args,
    data_base: &Path,
    data_specs: &[DataSpec],
    load_options: &LoadOptions,
    algorithms: &[crate::cli::AlgorithmKind],
    patterns: &[crate::specs::PatternSpec],
    indexes: &[crate::cli::IndexKind],
    bench_rows: &mut Vec<BenchRow>,
    profile_rows: &mut Vec<RowProfileRow>,
) -> Result<()> {
    let mut groups = BTreeMap::<String, Vec<DataSpec>>::new();
    for spec in data_specs
        .iter()
        .filter(|s| s.data_type == DataType::JobCsv)
    {
        groups
            .entry(spec.name.clone())
            .or_default()
            .push(spec.clone());
    }

    for (dataset, specs) in groups {
        let load_start = Instant::now();
        let loaded = load_job_csv_dataset(&dataset, &specs, data_base, load_options)?;
        let load_ns = load_start.elapsed().as_nanos();
        run_loaded_dataset(
            args,
            &dataset,
            DataType::JobCsv.as_str(),
            loaded,
            load_ns,
            algorithms,
            patterns,
            indexes,
            bench_rows,
            profile_rows,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_loaded_dataset(
    args: &Args,
    dataset_name: &str,
    data_type: &str,
    loaded: LoadedDataset,
    load_ns: u128,
    algorithms: &[crate::cli::AlgorithmKind],
    patterns: &[crate::specs::PatternSpec],
    indexes: &[crate::cli::IndexKind],
    bench_rows: &mut Vec<BenchRow>,
    profile_rows: &mut Vec<RowProfileRow>,
) -> Result<()> {
    for loaded_column in &loaded.columns {
        run_loaded_column(
            args,
            dataset_name,
            data_type,
            &loaded,
            loaded_column,
            load_ns,
            algorithms,
            patterns,
            indexes,
            bench_rows,
            profile_rows,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_loaded_column(
    args: &Args,
    dataset_name: &str,
    data_type: &str,
    loaded: &LoadedDataset,
    loaded_column: &LoadedColumn,
    load_ns: u128,
    algorithms: &[crate::cli::AlgorithmKind],
    patterns: &[crate::specs::PatternSpec],
    indexes: &[crate::cli::IndexKind],
    bench_rows: &mut Vec<BenchRow>,
    profile_rows: &mut Vec<RowProfileRow>,
) -> Result<()> {
    let storage = loaded_column.storage;
    let compatible_algorithms = algorithms
        .iter()
        .copied()
        .filter(|a| a.is_compatible(storage))
        .collect::<Vec<_>>();

    if compatible_algorithms.is_empty() {
        eprintln!(
            "skipping dataset={} column={} storage={} because no listed algorithms are compatible",
            dataset_name,
            loaded_column.name,
            storage.as_str()
        );
        return Ok(());
    }

    eprintln!(
        "loaded dataset={} column={} type={} storage={} loaded={} skipped={} truncated={} symbols={} load_ns={}",
        dataset_name,
        loaded_column.name,
        data_type,
        storage.as_str(),
        loaded_column.stats.records_loaded,
        loaded_column.stats.records_skipped,
        loaded_column.stats.records_truncated,
        loaded_column.stats.total_loaded_symbols,
        load_ns
    );

    match storage {
        StorageKind::Utf8 => {
            let table = loaded
                .db
                .utf8_table(loaded_column.table_id)
                .context("data table is not UTF-8")?;
            let column = table.text();
            let built_indexes = build_indexes(&column, indexes)?;
            let config = BenchConfig {
                dataset: dataset_name.to_owned(),
                column: loaded_column.name.clone(),
                data_path: loaded_column.source_path.clone(),
                data_type: data_type.to_owned(),
                storage,
                indexes: indexes.to_vec(),
                patterns: patterns.to_vec(),
                warmups: args.warmups,
                iterations: args.iterations,
                batch_rows: args.batch_rows,
                load_ns,
                load_stats: loaded_column.stats.clone(),
                row_labels: &loaded_column.row_labels,
                row_profile_enabled: args.row_profile_csv.is_some(),
                row_profile_repeats: args.row_profile_repeats,
                row_profile_max_rows: args.row_profile_max_rows,
                row_profile_sample_bytes: args.row_profile_sample_bytes,
            };

            for algorithm in compatible_algorithms {
                run_utf8_algorithm(
                    &column,
                    algorithm,
                    &built_indexes,
                    &config,
                    bench_rows,
                    args.row_profile_csv.as_ref().map(|_| &mut *profile_rows),
                )
                .with_context(|| format!("run UTF-8 algorithm {}", algorithm.as_str()))?;
            }
        }
        StorageKind::Dna2 => {
            let table = loaded
                .db
                .dna2_table(loaded_column.table_id)
                .context("data table is not DNA2")?;
            let column = table.sequence();
            let built_indexes = build_indexes(&column, indexes)?;
            let config = BenchConfig {
                dataset: dataset_name.to_owned(),
                column: loaded_column.name.clone(),
                data_path: loaded_column.source_path.clone(),
                data_type: data_type.to_owned(),
                storage,
                indexes: indexes.to_vec(),
                patterns: patterns.to_vec(),
                warmups: args.warmups,
                iterations: args.iterations,
                batch_rows: args.batch_rows,
                load_ns,
                load_stats: loaded_column.stats.clone(),
                row_labels: &loaded_column.row_labels,
                row_profile_enabled: args.row_profile_csv.is_some(),
                row_profile_repeats: args.row_profile_repeats,
                row_profile_max_rows: args.row_profile_max_rows,
                row_profile_sample_bytes: args.row_profile_sample_bytes,
            };

            for algorithm in compatible_algorithms {
                run_dna2_algorithm(
                    &column,
                    algorithm,
                    &built_indexes,
                    &config,
                    bench_rows,
                    args.row_profile_csv.as_ref().map(|_| &mut *profile_rows),
                )
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
