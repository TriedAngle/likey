use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::Serialize;
use db::{
    execute_like, BM, CountSink, Column, Dna2Column, Dna2NaiveWildcard, FmIndex, FullScan,
    LibcMemmem, LikePattern, Naive, NaiveMixed, NaiveScalar, NaiveVectorized,
    NaiveVectorizedV2, QueryScratch, QueryStats, RowId, RowLiteralSearch, RowVerifier,
    StdSearch, /* TrigramBitmapIndex, */ TwoWay, TwoWay2, Utf8Column, Utf8Kmp, VerifyScratch,
};

use crate::cli::{AlgorithmKind, IndexKind, StorageKind};
use crate::loaders::LoadStats;
use crate::specs::PatternSpec;

#[derive(Debug, Clone)]
pub struct BenchConfig<'a> {
    pub dataset: String,
    pub column: String,
    pub data_path: PathBuf,
    pub data_type: String,
    pub storage: StorageKind,
    pub indexes: Vec<IndexKind>,
    pub patterns: Vec<PatternSpec>,
    pub warmups: usize,
    pub iterations: usize,
    pub batch_rows: usize,
    pub load_ns: u128,
    pub load_stats: LoadStats,
    pub row_labels: &'a [String],
    pub row_profile_enabled: bool,
    pub row_profile_repeats: usize,
    pub row_profile_max_rows: Option<u64>,
    pub row_profile_sample_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchRow {
    pub dataset: String,
    pub column: String,
    pub data_path: String,
    pub data_type: String,
    pub storage: String,
    pub algorithm: String,
    pub requested_index: String,
    pub actual_index: String,
    pub pattern_name: String,
    pub pattern: String,
    pub iteration: usize,
    pub row_count: u64,
    pub total_input_sequence_bytes: u64,
    pub total_loaded_symbols: u64,
    pub records_seen: u64,
    pub records_loaded: u64,
    pub records_skipped: u64,
    pub records_truncated: u64,
    pub records_invalid_dna: u64,
    pub load_ns: u128,
    pub index_build_ns: u128,
    pub compile_ns: u128,
    pub candidate_prepare_ns: u128,
    pub execute_ns: u128,
    pub query_total_ns: u128,
    pub candidate_rows_seen: u64,
    pub rows_after_len_filter: u64,
    pub rows_matched: u64,
    pub ns_per_table_row: f64,
    pub ns_per_candidate_row: f64,
    pub ns_per_loaded_symbol: f64,
    pub fallback_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RowProfileRow {
    pub dataset: String,
    pub column: String,
    pub data_path: String,
    pub data_type: String,
    pub storage: String,
    pub algorithm: String,
    pub pattern_name: String,
    pub pattern: String,
    pub row_id: u64,
    pub row_label: String,
    pub row_len: u32,
    pub matched: bool,
    pub repeats: usize,
    pub verify_ns_total: u128,
    pub verify_ns_per_repeat: f64,
    pub row_sample: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SummaryRow {
    pub dataset: String,
    pub column: String,
    pub storage: String,
    pub algorithm: String,
    pub requested_index: String,
    pub actual_index: String,
    pub pattern_name: String,
    pub pattern: String,
    pub runs: usize,
    pub row_count: u64,
    pub rows_matched: u64,
    pub sum_query_total_ns: u128,
    pub min_query_total_ns: u128,
    pub median_query_total_ns: f64,
    pub mean_query_total_ns: f64,
    pub geomean_query_total_ns: f64,
    pub p90_query_total_ns: u128,
    pub max_query_total_ns: u128,
    pub median_execute_ns: f64,
    pub median_candidate_prepare_ns: f64,
    pub median_ns_per_table_row: f64,
    pub geomean_ns_per_table_row: f64,
    pub median_ns_per_candidate_row: f64,
}

#[derive(Debug)]
pub struct BuiltIndex<T> {
    pub index: T,
    pub build_ns: u128,
}

#[derive(Debug, Default)]
pub struct BuiltIndexes {
    pub fm: Option<BuiltIndex<FmIndex>>,
    // pub trigram: Option<BuiltIndex<TrigramBitmapIndex>>,
}

pub fn build_indexes<C>(column: &C, requested: &[IndexKind]) -> Result<BuiltIndexes>
where
    C: Column<Symbol = u8>,
{
    let mut out = BuiltIndexes::default();

    if requested.iter().any(|kind| *kind == IndexKind::Fm) {
        let start = Instant::now();
        let index = FmIndex::build(column)?;
        out.fm = Some(BuiltIndex {
            index,
            build_ns: start.elapsed().as_nanos(),
        });
    }

    // if requested.iter().any(|kind| *kind == IndexKind::Trigram) {
    //     let start = Instant::now();
    //     let index = TrigramBitmapIndex::build(column);
    //     out.trigram = Some(BuiltIndex {
    //         index,
    //         build_ns: start.elapsed().as_nanos(),
    //     });
    // }

    Ok(out)
}

pub fn run_utf8_algorithm<'db>(
    column: &Utf8Column<'db>,
    algorithm: AlgorithmKind,
    indexes: &BuiltIndexes,
    config: &BenchConfig<'_>,
    out: &mut Vec<BenchRow>,
    profile_out: Option<&mut Vec<RowProfileRow>>,
) -> Result<()> {
    match algorithm {
        AlgorithmKind::Std => run_algorithm::<Utf8Column<'db>, StdSearch, _>(column, algorithm, indexes, config, out, profile_out, sample_utf8_row),
        AlgorithmKind::Kmp => run_algorithm::<Utf8Column<'db>, Utf8Kmp, _>(column, algorithm, indexes, config, out, profile_out, sample_utf8_row),
        AlgorithmKind::Naive => run_algorithm::<Utf8Column<'db>, Naive, _>(column, algorithm, indexes, config, out, profile_out, sample_utf8_row),
        AlgorithmKind::NaiveScalar => run_algorithm::<Utf8Column<'db>, NaiveScalar, _>(column, algorithm, indexes, config, out, profile_out, sample_utf8_row),
        AlgorithmKind::NaiveVectorized => run_algorithm::<Utf8Column<'db>, NaiveVectorized, _>(column, algorithm, indexes, config, out, profile_out, sample_utf8_row),
        AlgorithmKind::NaiveVectorizedV2 => run_algorithm::<Utf8Column<'db>, NaiveVectorizedV2, _>(column, algorithm, indexes, config, out, profile_out, sample_utf8_row),
        AlgorithmKind::NaiveMixed => run_algorithm::<Utf8Column<'db>, NaiveMixed, _>(column, algorithm, indexes, config, out, profile_out, sample_utf8_row),
        AlgorithmKind::Bm => run_algorithm::<Utf8Column<'db>, BM, _>(column, algorithm, indexes, config, out, profile_out, sample_utf8_row),
        AlgorithmKind::TwoWay => run_algorithm::<Utf8Column<'db>, TwoWay, _>(column, algorithm, indexes, config, out, profile_out, sample_utf8_row),
        AlgorithmKind::TwoWay2 => run_algorithm::<Utf8Column<'db>, TwoWay2, _>(column, algorithm, indexes, config, out, profile_out, sample_utf8_row),
        AlgorithmKind::LibcMemmem => run_algorithm::<Utf8Column<'db>, LibcMemmem, _>(column, algorithm, indexes, config, out, profile_out, sample_utf8_row),
        AlgorithmKind::Dna2Naive => bail!("algorithm dna2-naive cannot run on UTF-8 storage"),
    }
}

pub fn run_dna2_algorithm<'db>(
    column: &Dna2Column<'db>,
    algorithm: AlgorithmKind,
    indexes: &BuiltIndexes,
    config: &BenchConfig<'_>,
    out: &mut Vec<BenchRow>,
    profile_out: Option<&mut Vec<RowProfileRow>>,
) -> Result<()> {
    match algorithm {
        AlgorithmKind::Dna2Naive => run_algorithm::<Dna2Column<'db>, Dna2NaiveWildcard, _>(column, algorithm, indexes, config, out, profile_out, sample_dna2_row),
        other => bail!("algorithm {} cannot run on DNA2 storage", other.as_str()),
    }
}

fn run_algorithm<C, A, F>(
    column: &C,
    algorithm: AlgorithmKind,
    indexes: &BuiltIndexes,
    config: &BenchConfig<'_>,
    out: &mut Vec<BenchRow>,
    mut profile_out: Option<&mut Vec<RowProfileRow>>,
    sample_row: F,
) -> Result<()>
where
    C: Column<Symbol = u8>,
    A: RowLiteralSearch<C>,
    F: Fn(&C, RowId, usize) -> String + Copy,
{
    for pattern_spec in &config.patterns {
        let compile_start = Instant::now();
        let pattern = LikePattern::<A>::compile(&pattern_spec.pattern)
            .with_context(|| format!("compile LIKE pattern {:?}", pattern_spec.pattern))?;
        let compile_ns = compile_start.elapsed().as_nanos();

        if config.row_profile_enabled {
            if let Some(rows) = profile_out.as_mut() {
                profile_algorithm_rows::<C, A, F>(column, algorithm, &pattern, pattern_spec, config, &mut **rows, sample_row);
            }
        }

        for &requested_index in &config.indexes {
            let index_build_ns = index_build_ns(indexes, requested_index);
            let mut scratch = QueryScratch::default();

            for _ in 0..config.warmups {
                scratch.candidates.clear();
                scratch.verify.clear();
                let exec = execute_once(column, &pattern, requested_index, indexes, config.batch_rows, &mut scratch);
                std::hint::black_box(exec.stats.rows_matched);
                std::hint::black_box(exec.count);
            }

            for iteration in 0..config.iterations {
                scratch.candidates.clear();
                scratch.verify.clear();

                let exec = execute_once(column, &pattern, requested_index, indexes, config.batch_rows, &mut scratch);

                std::hint::black_box(exec.stats.rows_matched);
                std::hint::black_box(exec.count);

                out.push(BenchRow {
                    dataset: config.dataset.clone(),
                    column: config.column.clone(),
                    data_path: config.data_path.display().to_string(),
                    data_type: config.data_type.clone(),
                    storage: config.storage.as_str().to_owned(),
                    algorithm: algorithm.as_str().to_owned(),
                    requested_index: requested_index.as_str().to_owned(),
                    actual_index: exec.actual_index.to_owned(),
                    pattern_name: pattern_spec.name.clone(),
                    pattern: pattern_spec.pattern.clone(),
                    iteration,
                    row_count: column.row_count(),
                    total_input_sequence_bytes: config.load_stats.total_input_sequence_bytes,
                    total_loaded_symbols: config.load_stats.total_loaded_symbols,
                    records_seen: config.load_stats.records_seen,
                    records_loaded: config.load_stats.records_loaded,
                    records_skipped: config.load_stats.records_skipped,
                    records_truncated: config.load_stats.records_truncated,
                    records_invalid_dna: config.load_stats.records_invalid_dna,
                    load_ns: config.load_ns,
                    index_build_ns,
                    compile_ns,
                    candidate_prepare_ns: exec.candidate_prepare_ns,
                    execute_ns: exec.execute_ns,
                    query_total_ns: exec.candidate_prepare_ns + exec.execute_ns,
                    candidate_rows_seen: exec.stats.candidate_rows_seen,
                    rows_after_len_filter: exec.stats.rows_after_len_filter,
                    rows_matched: exec.stats.rows_matched,
                    ns_per_table_row: ns_per(exec.candidate_prepare_ns + exec.execute_ns, column.row_count()),
                    ns_per_candidate_row: ns_per(exec.execute_ns, exec.stats.candidate_rows_seen),
                    ns_per_loaded_symbol: ns_per(exec.candidate_prepare_ns + exec.execute_ns, config.load_stats.total_loaded_symbols),
                    fallback_reason: exec.fallback_reason.to_owned(),
                });
            }
        }
    }

    Ok(())
}

fn profile_algorithm_rows<C, A, F>(
    column: &C,
    algorithm: AlgorithmKind,
    pattern: &LikePattern<A>,
    pattern_spec: &PatternSpec,
    config: &BenchConfig<'_>,
    out: &mut Vec<RowProfileRow>,
    sample_row: F,
) where
    C: Column<Symbol = u8>,
    A: RowLiteralSearch<C>,
    F: Fn(&C, RowId, usize) -> String + Copy,
{
    let rows_to_profile = config
        .row_profile_max_rows
        .unwrap_or_else(|| column.row_count())
        .min(column.row_count());
    let repeats = config.row_profile_repeats.max(1);
    let len_constraint = pattern.len_constraint();
    let mut scratch = VerifyScratch::default();

    for row in 0..rows_to_profile {
        let row_len = column.logical_len(row);
        let len_ok = len_constraint.matches(row_len);
        let mut matched = false;
        let start = Instant::now();
        for _ in 0..repeats {
            scratch.clear();
            matched = len_ok && pattern.verify(column, row, &mut scratch);
            std::hint::black_box(matched);
        }
        let ns_total = start.elapsed().as_nanos();
        let row_label = config
            .row_labels
            .get(row as usize)
            .cloned()
            .unwrap_or_else(|| row.to_string());
        out.push(RowProfileRow {
            dataset: config.dataset.clone(),
            column: config.column.clone(),
            data_path: config.data_path.display().to_string(),
            data_type: config.data_type.clone(),
            storage: config.storage.as_str().to_owned(),
            algorithm: algorithm.as_str().to_owned(),
            pattern_name: pattern_spec.name.clone(),
            pattern: pattern_spec.pattern.clone(),
            row_id: row,
            row_label,
            row_len,
            matched,
            repeats,
            verify_ns_total: ns_total,
            verify_ns_per_repeat: ns_total as f64 / repeats as f64,
            row_sample: sample_row(column, row, config.row_profile_sample_bytes),
        });
    }
}

#[derive(Debug, Clone, Copy)]
struct ExecuteOnceResult {
    stats: QueryStats,
    count: u64,
    actual_index: &'static str,
    fallback_reason: &'static str,
    candidate_prepare_ns: u128,
    execute_ns: u128,
}

fn execute_once<C, A>(
    column: &C,
    pattern: &LikePattern<A>,
    requested_index: IndexKind,
    indexes: &BuiltIndexes,
    batch_rows: usize,
    scratch: &mut QueryScratch,
) -> ExecuteOnceResult
where
    C: Column<Symbol = u8>,
    A: RowLiteralSearch<C>,
{
    match requested_index {
        IndexKind::FullScan => execute_full_scan(column, pattern, batch_rows, scratch, "", "full-scan"),
        IndexKind::Fm => {
            if let Some(fm) = indexes.fm.as_ref() {
                let prepare_start = Instant::now();
                let probe = fm.index.probe_longest_like_literal(pattern, batch_rows);
                let candidate_prepare_ns = prepare_start.elapsed().as_nanos();
                if let Some(mut probe) = probe {
                    let mut sink = CountSink::default();
                    let execute_start = Instant::now();
                    let stats = execute_like(column, &mut probe, pattern, scratch, &mut sink);
                    let execute_ns = execute_start.elapsed().as_nanos();
                    ExecuteOnceResult {
                        stats,
                        count: sink.count,
                        actual_index: "fm",
                        fallback_reason: "",
                        candidate_prepare_ns,
                        execute_ns,
                    }
                } else {
                    let mut res = execute_full_scan(
                        column,
                        pattern,
                        batch_rows,
                        scratch,
                        "no-indexable-literal",
                        "full-scan",
                    );
                    res.candidate_prepare_ns += candidate_prepare_ns;
                    res
                }
            } else {
                execute_full_scan(column, pattern, batch_rows, scratch, "fm-not-built", "full-scan")
            }
        }
        IndexKind::Trigram => {
            unimplemented!()
            // if let Some(trigram) = indexes.trigram.as_ref() {
            //     if let Some(literal) = pattern.longest_indexable_literal() {
            //         if literal.len() >= 3 {
            //             let gram = [literal[0], literal[1], literal[2]];
            //             let block_words = ((batch_rows.max(64) + 63) / 64).max(1);
            //             let prepare_start = Instant::now();
            //             let mut probe = trigram.index.probe(gram, block_words);
            //             let candidate_prepare_ns = prepare_start.elapsed().as_nanos();
            //             let mut sink = CountSink::default();
            //             let execute_start = Instant::now();
            //             let stats = execute_like(column, &mut probe, pattern, scratch, &mut sink);
            //             let execute_ns = execute_start.elapsed().as_nanos();
            //             ExecuteOnceResult {
            //                 stats,
            //                 count: sink.count,
            //                 actual_index: "trigram",
            //                 fallback_reason: "",
            //                 candidate_prepare_ns,
            //                 execute_ns,
            //             }
            //         } else {
            //             execute_full_scan(
            //                 column,
            //                 pattern,
            //                 batch_rows,
            //                 scratch,
            //                 "literal-shorter-than-trigram",
            //                 "full-scan",
            //             )
            //         }
            //     } else {
            //         execute_full_scan(
            //             column,
            //             pattern,
            //             batch_rows,
            //             scratch,
            //             "no-indexable-literal",
            //             "full-scan",
            //         )
            //     }
            // } else {
            //     execute_full_scan(column, pattern, batch_rows, scratch, "trigram-not-built", "full-scan")
            // }
        }
    }
}

fn execute_full_scan<C, A>(
    column: &C,
    pattern: &LikePattern<A>,
    batch_rows: usize,
    scratch: &mut QueryScratch,
    fallback_reason: &'static str,
    actual_index: &'static str,
) -> ExecuteOnceResult
where
    C: Column<Symbol = u8>,
    A: RowLiteralSearch<C>,
{
    let prepare_start = Instant::now();
    let mut probe = FullScan::new(column.row_count(), batch_rows as u64);
    let candidate_prepare_ns = prepare_start.elapsed().as_nanos();
    let mut sink = CountSink::default();
    let execute_start = Instant::now();
    let stats = execute_like(column, &mut probe, pattern, scratch, &mut sink);
    let execute_ns = execute_start.elapsed().as_nanos();
    ExecuteOnceResult {
        stats,
        count: sink.count,
        actual_index,
        fallback_reason,
        candidate_prepare_ns,
        execute_ns,
    }
}

fn index_build_ns(indexes: &BuiltIndexes, requested: IndexKind) -> u128 {
    match requested {
        IndexKind::FullScan => 0,
        IndexKind::Fm => indexes.fm.as_ref().map_or(0, |idx| idx.build_ns),
        // IndexKind::Trigram => indexes.trigram.as_ref().map_or(0, |idx| idx.build_ns),
        IndexKind::Trigram => unimplemented!(),
    }
}

fn ns_per(ns: u128, denom: u64) -> f64 {
    if denom == 0 {
        0.0
    } else {
        ns as f64 / denom as f64
    }
}

fn sample_utf8_row(column: &Utf8Column<'_>, row: RowId, max_bytes: usize) -> String {
    let bytes = column.row_bytes(row);
    let end = bytes.len().min(max_bytes);
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn sample_dna2_row(column: &Dna2Column<'_>, row: RowId, max_bytes: usize) -> String {
    let row = column.row_view(row);
    let mut bytes = Vec::with_capacity((row.logical_len() as usize).min(max_bytes));
    for code in row.iter().take(max_bytes) {
        let ascii = db::DnaBase::from_code(code)
            .expect("stored DNA2 code must be valid")
            .ascii();
        bytes.push(ascii);
    }
    String::from_utf8(bytes).expect("DNA ASCII is valid UTF-8")
}

pub fn write_rows(path: &Path, rows: &[BenchRow]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_row_profiles(path: &Path, rows: &[RowProfileRow]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_summary(path: &Path, rows: &[BenchRow]) -> Result<()> {
    let mut grouped = HashMap::<SummaryKey, Vec<&BenchRow>>::new();
    for row in rows {
        grouped.entry(SummaryKey::from(row)).or_default().push(row);
    }

    let mut summaries = grouped
        .into_iter()
        .map(|(key, group)| summarize_group(key, group))
        .collect::<Vec<_>>();

    summaries.sort_by(|a, b| {
        a.median_query_total_ns
            .partial_cmp(&b.median_query_total_ns)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut writer = csv::Writer::from_path(path)?;
    for row in &summaries {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SummaryKey {
    dataset: String,
    column: String,
    storage: String,
    algorithm: String,
    requested_index: String,
    actual_index: String,
    pattern_name: String,
    pattern: String,
}

impl From<&BenchRow> for SummaryKey {
    fn from(row: &BenchRow) -> Self {
        Self {
            dataset: row.dataset.clone(),
            column: row.column.clone(),
            storage: row.storage.clone(),
            algorithm: row.algorithm.clone(),
            requested_index: row.requested_index.clone(),
            actual_index: row.actual_index.clone(),
            pattern_name: row.pattern_name.clone(),
            pattern: row.pattern.clone(),
        }
    }
}

fn summarize_group(key: SummaryKey, rows: Vec<&BenchRow>) -> SummaryRow {
    debug_assert!(!rows.is_empty());
    let mut total = rows.iter().map(|r| r.query_total_ns).collect::<Vec<_>>();
    let mut exec = rows.iter().map(|r| r.execute_ns).collect::<Vec<_>>();
    let mut prep = rows.iter().map(|r| r.candidate_prepare_ns).collect::<Vec<_>>();
    total.sort_unstable();
    exec.sort_unstable();
    prep.sort_unstable();

    let runs = total.len();
    let sum = total.iter().copied().sum::<u128>();
    let mean = sum as f64 / runs as f64;
    let p90_idx = ((runs as f64 * 0.90).ceil() as usize).saturating_sub(1).min(runs - 1);
    let first = rows[0];
    let median_total = median_u128(&total);

    SummaryRow {
        dataset: key.dataset,
        column: key.column,
        storage: key.storage,
        algorithm: key.algorithm,
        requested_index: key.requested_index,
        actual_index: key.actual_index,
        pattern_name: key.pattern_name,
        pattern: key.pattern,
        runs,
        row_count: first.row_count,
        rows_matched: first.rows_matched,
        sum_query_total_ns: sum,
        min_query_total_ns: total[0],
        median_query_total_ns: median_total,
        mean_query_total_ns: mean,
        geomean_query_total_ns: geomean_u128(&total),
        p90_query_total_ns: total[p90_idx],
        max_query_total_ns: *total.last().expect("non-empty"),
        median_execute_ns: median_u128(&exec),
        median_candidate_prepare_ns: median_u128(&prep),
        median_ns_per_table_row: ns_per(median_total as u128, first.row_count),
        geomean_ns_per_table_row: ns_per(geomean_u128(&total) as u128, first.row_count),
        median_ns_per_candidate_row: ns_per(median_u128(&exec) as u128, first.candidate_rows_seen),
    }
}

fn median_u128(values: &[u128]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        values[n / 2] as f64
    } else {
        (values[n / 2 - 1] as f64 + values[n / 2] as f64) / 2.0
    }
}

fn geomean_u128(values: &[u128]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.iter().any(|&v| v == 0) {
        return 0.0;
    }
    let mean_log = values
        .iter()
        .map(|&v| (v as f64).ln())
        .sum::<f64>()
        / values.len() as f64;
    mean_log.exp()
}
