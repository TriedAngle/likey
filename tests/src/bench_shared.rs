use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    time::{Duration, Instant},
};

use algos::{
    BM, FMIndex, FftConfig, FftStr0, FftStr1, KMP, NaiveMixed, NaiveScalar, NaiveVectorized,
    NaiveVectorizedV2, StdSearch, StringSearch, TrigramIndex, TwoWay,
};
use engine::execute;
use like::{CompileOptions, Pattern, compile_pattern, compile_pattern_with_options, like_match};
use storage::dataset::DataSet;

const FM_SEPARATOR: u8 = 0x1F;
const FM_SENTINEL: u8 = 0x00;

const ALGORITHMS: &[&str] = &[
    "naive-scalar",
    "naive-vector",
    "naive-vector-v2",
    "naive-mixed",
    "kmp",
    "bm",
    "two-way",
    "std",
    "lut-short",
    "fftstr0",
    "fftstr1",
    "fm",
    "trigram",
];

#[derive(Debug)]
struct ResultEntry {
    algo: String,
    pattern_index: usize,
    pattern: String,
    file: String,
    file_type: String,
    duration: Duration,
    found_count: usize,
    skipped: bool,
}

#[derive(Debug)]
struct Mismatch {
    algo: String,
    pattern_index: usize,
    pattern: String,
    file: String,
    expected: usize,
    actual: usize,
}

#[derive(Debug, Clone, Copy)]
struct FmRow<'a> {
    table: &'a str,
    data: &'a str,
    start: usize,
    end: usize,
}

struct FmIndexDatabase<'a> {
    fm: FMIndex,
    rows: Vec<FmRow<'a>>,
    row_starts: Vec<usize>,
    byte_freq: [usize; 256],
    max_range: usize,
}

#[derive(Debug, Clone)]
enum FmLiteralLookup {
    Rows(HashSet<usize>),
    NoMatch,
    TooBroad,
}

#[derive(Debug, Clone)]
enum FmPositionLookup {
    Positions(Vec<usize>),
    NoMatch,
    TooBroad,
}

type FmLiteralCache = HashMap<String, FmLiteralLookup>;

struct TrigramDatabase<'a> {
    index: TrigramIndex<'a>,
    rows: Vec<TrigramRow<'a>>,
}

#[derive(Debug, Clone, Copy)]
struct TrigramRow<'a> {
    table: &'a str,
    data: &'a str,
}

#[derive(Debug, Clone)]
enum TrigramLiteralLookup {
    CandidateIds(Box<[u32]>),
    NoMatch,
}

type TrigramLiteralCache = HashMap<String, TrigramLiteralLookup>;

#[derive(Debug, Clone, Default)]
pub struct BenchOptions {
    skip_algorithms: HashSet<String>,
}

impl BenchOptions {
    pub fn new(skip_algorithms: HashSet<String>) -> Result<Self, String> {
        let mut unknown = Vec::new();
        for name in skip_algorithms.iter() {
            if !ALGORITHMS.iter().any(|candidate| candidate == name) {
                unknown.push(name.clone());
            }
        }

        if !unknown.is_empty() {
            unknown.sort();
            return Err(format!(
                "Unknown algorithm(s): {}. Known values: {}",
                unknown.join(", "),
                ALGORITHMS.join(", ")
            ));
        }

        Ok(Self { skip_algorithms })
    }

    pub fn should_skip(&self, algo_name: &str) -> bool {
        self.skip_algorithms.contains(algo_name)
    }

    pub fn skip_algorithms(&self) -> &HashSet<String> {
        &self.skip_algorithms
    }
}

pub fn available_algorithms() -> &'static [&'static str] {
    ALGORITHMS
}

#[derive(Debug, Clone)]
pub struct PatternSpec {
    pub pattern: String,
    pub description: String,
}

#[derive(Debug, Clone, Copy)]
struct TableStats {
    rows: usize,
    avg_len: f64,
    max_len: usize,
}

impl<'a> FmIndexDatabase<'a> {
    fn row_index_for_pos(&self, pos: usize) -> Option<usize> {
        let idx = match self.row_starts.binary_search(&pos) {
            Ok(i) => i,
            Err(0) => return None,
            Err(i) => i - 1,
        };

        let row = &self.rows[idx];
        if pos < row.end { Some(idx) } else { None }
    }
}

pub fn run_like_benchmarks(
    database: &DataSet<'_>,
    dataset_name: &str,
    patterns: &[PatternSpec],
    options: BenchOptions,
    output_csv: Option<&Path>,
) {
    let skip_lut_short = options.should_skip("lut-short") || !lut_short_available();
    let naive_vector_unavailable = !naive_vector_available();

    println!("--- Starting Like Benchmark ---");
    println!("> Database loaded. Total tables: {}", database.tables.len());

    let fm_database = if options.should_skip("fm") {
        None
    } else {
        println!("> Building FM-index...");
        let (fm_database, fm_build_time) = build_fm_index(database);
        println!("> FM-index built in {} ms", fm_build_time.as_millis());
        Some(fm_database)
    };

    let trigram_database = if options.should_skip("trigram") {
        None
    } else {
        println!("> Building trigram index...");
        let start = Instant::now();
        let trigram = build_trigram_index(database);
        let duration = start.elapsed();
        println!("> Trigram index built in {} ms", duration.as_millis());
        Some(trigram)
    };

    let mut fm_literal_cache: FmLiteralCache = HashMap::new();
    let mut trigram_literal_cache: TrigramLiteralCache = HashMap::new();
    let table_stats = compute_table_stats(database);

    let mut results = Vec::new();
    let total_progress_steps = ALGORITHMS.len().saturating_mul(patterns.len());
    let mut completed_progress_steps = 0usize;
    let mut next_progress_percent = 1usize;

    for algo_name in ALGORITHMS {
        for (pattern_index, pat_spec) in patterns.iter().enumerate() {
            let pat_str = pat_spec.pattern.as_str();
            println!(
                "> Benchmarking Algo: [{}] Pattern: [{}]",
                algo_name, pat_str
            );

            let entries = match *algo_name {
                "naive-scalar" => {
                    if options.should_skip("naive-scalar") {
                        skipped_entries(algo_name, pat_str, pattern_index, database)
                    } else {
                        run_benchmark::<NaiveScalar, _>(
                            algo_name,
                            pat_str,
                            pattern_index,
                            database,
                            |_, pat| unsafe { std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes()) },
                        )
                    }
                }
                "naive-vector" => {
                    if options.should_skip("naive-vector") || naive_vector_unavailable {
                        skipped_entries(algo_name, pat_str, pattern_index, database)
                    } else {
                        run_benchmark::<NaiveVectorized, _>(
                            algo_name,
                            pat_str,
                            pattern_index,
                            database,
                            |_, pat| unsafe { std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes()) },
                        )
                    }
                }
                "kmp" => {
                    if options.should_skip("kmp") {
                        skipped_entries(algo_name, pat_str, pattern_index, database)
                    } else {
                        run_benchmark::<KMP, _>(
                            algo_name,
                            pat_str,
                            pattern_index,
                            database,
                            |_, pat| unsafe { std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes()) },
                        )
                    }
                }
                "naive-vector-v2" => {
                    if options.should_skip("naive-vector-v2") || naive_vector_unavailable {
                        skipped_entries(algo_name, pat_str, pattern_index, database)
                    } else {
                        run_benchmark::<NaiveVectorizedV2, _>(
                            algo_name,
                            pat_str,
                            pattern_index,
                            database,
                            |_, pat| unsafe { std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes()) },
                        )
                    }
                }
                "naive-mixed" => {
                    if options.should_skip("naive-mixed") {
                        skipped_entries(algo_name, pat_str, pattern_index, database)
                    } else {
                        run_benchmark::<NaiveMixed, _>(
                            algo_name,
                            pat_str,
                            pattern_index,
                            database,
                            |_, pat| unsafe { std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes()) },
                        )
                    }
                }
                "bm" => {
                    if options.should_skip("bm") {
                        skipped_entries(algo_name, pat_str, pattern_index, database)
                    } else {
                        run_benchmark::<BM, _>(
                            algo_name,
                            pat_str,
                            pattern_index,
                            database,
                            |_, pat| unsafe { std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes()) },
                        )
                    }
                }
                "two-way" => {
                    if options.should_skip("two-way") {
                        skipped_entries(algo_name, pat_str, pattern_index, database)
                    } else {
                        run_benchmark::<TwoWay, _>(
                            algo_name,
                            pat_str,
                            pattern_index,
                            database,
                            |_, pat| unsafe { std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes()) },
                        )
                    }
                }
                "std" => {
                    if options.should_skip("std") {
                        skipped_entries(algo_name, pat_str, pattern_index, database)
                    } else {
                        run_benchmark::<StdSearch, _>(
                            algo_name,
                            pat_str,
                            pattern_index,
                            database,
                            |_, pat| unsafe { std::mem::transmute::<&str, &str>(pat) },
                        )
                    }
                }
                "lut-short" => {
                    if options.should_skip("lut-short") || skip_lut_short {
                        skipped_entries(algo_name, pat_str, pattern_index, database)
                    } else {
                        run_benchmark::<algos::LutShort, _>(
                            algo_name,
                            pat_str,
                            pattern_index,
                            database,
                            |_, pat| unsafe { std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes()) },
                        )
                    }
                }
                "fftstr0" => {
                    if options.should_skip("fftstr0") || should_skip_fftstr0(pat_str) {
                        skipped_entries(algo_name, pat_str, pattern_index, database)
                    } else {
                        run_benchmark_with_options::<FftStr0, _>(
                            algo_name,
                            pat_str,
                            pattern_index,
                            database,
                            |_, pat| FftConfig::from_str(pat),
                            CompileOptions {
                                treat_underscore_as_literal: true,
                                literal_underscore_is_wildcard: true,
                                ascii_mode: true,
                            },
                        )
                    }
                }
                "fftstr1" => {
                    if options.should_skip("fftstr1") || should_skip_fftstr1(pat_str) {
                        skipped_entries(algo_name, pat_str, pattern_index, database)
                    } else {
                        run_benchmark_with_options::<FftStr1, _>(
                            algo_name,
                            pat_str,
                            pattern_index,
                            database,
                            |_, pat| FftConfig::from_str(pat),
                            CompileOptions {
                                treat_underscore_as_literal: true,
                                literal_underscore_is_wildcard: true,
                                ascii_mode: true,
                            },
                        )
                    }
                }
                "fm" => {
                    if options.should_skip("fm") {
                        skipped_entries(algo_name, pat_str, pattern_index, database)
                    } else {
                        let fm_database = fm_database.as_ref().expect("fm index not built");
                        run_fm_benchmark(
                            algo_name,
                            pat_str,
                            pattern_index,
                            database,
                            fm_database,
                            &mut fm_literal_cache,
                        )
                    }
                }
                "trigram" => {
                    if options.should_skip("trigram") {
                        skipped_entries(algo_name, pat_str, pattern_index, database)
                    } else {
                        let trigram_database =
                            trigram_database.as_ref().expect("trigram index not built");
                        run_trigram_benchmark(
                            algo_name,
                            pat_str,
                            pattern_index,
                            database,
                            trigram_database,
                            &mut trigram_literal_cache,
                        )
                    }
                }
                _ => panic!("Unknown algorithm: {}", algo_name),
            };

            results.extend(entries);

            completed_progress_steps += 1;
            if total_progress_steps > 0 {
                let progress_percent = (completed_progress_steps * 100) / total_progress_steps;
                while next_progress_percent <= progress_percent && next_progress_percent <= 100 {
                    println!(
                        "> Progress: {}% ({}/{})",
                        next_progress_percent, completed_progress_steps, total_progress_steps
                    );
                    next_progress_percent += 1;
                }
            }
        }
    }

    print_summary_table(&results);
    print_algo_ranking(&results);
    print_per_pattern_ranking(&results);
    print_per_file_ranking(&results);
    print_correctness_report(&results);

    if let Some(path) = output_csv {
        write_results_csv(path, dataset_name, patterns, &results, &table_stats)
            .expect("write benchmark csv");
        println!("> Wrote CSV results: {}", path.display());
    }
}

fn lut_short_available() -> bool {
    cfg!(all(target_arch = "x86_64", target_feature = "ssse3"))
        || cfg!(all(target_arch = "aarch64", target_feature = "neon"))
}

fn naive_vector_available() -> bool {
    cfg!(all(target_arch = "x86_64", target_feature = "sse2"))
        || cfg!(all(target_arch = "aarch64", target_feature = "neon"))
}

fn run_benchmark<'a, S, F>(
    algo_name: &str,
    pat_str: &str,
    pattern_index: usize,
    database: &'a DataSet<'a>,
    factory: F,
) -> Vec<ResultEntry>
where
    S: StringSearch,
    F: FnMut(&mut (), &str) -> S::Config + Clone,
{
    let mut results = Vec::new();

    let pattern = compile_pattern::<S, _, _>(pat_str, (), factory);

    for table in database.tables.iter() {
        let table_dataset = DataSet {
            tables: vec![table.clone()].into_boxed_slice(),
        };

        let start = Instant::now();
        let matches = execute(&pattern, &table_dataset);
        let duration = start.elapsed();

        results.push(ResultEntry {
            algo: algo_name.to_string(),
            pattern_index,
            pattern: pat_str.to_string(),
            file: table.name.clone(),
            file_type: infer_file_type(&table.name),
            duration,
            found_count: matches.len(),
            skipped: false,
        });
    }

    results
}

fn run_fm_benchmark<'a>(
    algo_name: &str,
    pat_str: &str,
    pattern_index: usize,
    database: &'a DataSet<'a>,
    fm_database: &FmIndexDatabase<'a>,
    fm_literal_cache: &mut FmLiteralCache,
) -> Vec<ResultEntry> {
    let mut results = Vec::new();
    let pattern = compile_pattern::<StdSearch, _, _>(pat_str, (), |_, pat| pat);

    for table in database.tables.iter() {
        if table.rows.is_empty() {
            results.push(ResultEntry {
                algo: algo_name.to_string(),
                pattern_index,
                pattern: pat_str.to_string(),
                file: table.name.clone(),
                file_type: infer_file_type(&table.name),
                duration: Duration::from_micros(0),
                found_count: 0,
                skipped: false,
            });
            continue;
        }
        let table_name = table.name.as_str();
        let start = Instant::now();
        let found =
            fm_like_search_table(fm_database, table_name, &pattern, pat_str, fm_literal_cache);
        let duration = start.elapsed();

        results.push(ResultEntry {
            algo: algo_name.to_string(),
            pattern_index,
            pattern: pat_str.to_string(),
            file: table.name.clone(),
            file_type: infer_file_type(&table.name),
            duration,
            found_count: found,
            skipped: false,
        });
    }

    results
}

fn run_trigram_benchmark<'a>(
    algo_name: &str,
    pat_str: &str,
    pattern_index: usize,
    database: &'a DataSet<'a>,
    trigram_database: &TrigramDatabase<'a>,
    trigram_literal_cache: &mut TrigramLiteralCache,
) -> Vec<ResultEntry> {
    let mut results = Vec::new();
    let pattern = compile_pattern::<StdSearch, _, _>(pat_str, (), |_, pat| pat);

    for table in database.tables.iter() {
        if table.rows.is_empty() {
            results.push(ResultEntry {
                algo: algo_name.to_string(),
                pattern_index,
                pattern: pat_str.to_string(),
                file: table.name.clone(),
                file_type: infer_file_type(&table.name),
                duration: Duration::from_micros(0),
                found_count: 0,
                skipped: false,
            });
            continue;
        }
        let table_name = table.name.as_str();
        let start = Instant::now();
        let found = trigram_like_search_table(
            trigram_database,
            table_name,
            &pattern,
            pat_str,
            trigram_literal_cache,
        );
        let duration = start.elapsed();

        results.push(ResultEntry {
            algo: algo_name.to_string(),
            pattern_index,
            pattern: pat_str.to_string(),
            file: table.name.clone(),
            file_type: infer_file_type(&table.name),
            duration,
            found_count: found,
            skipped: false,
        });
    }

    results
}

fn run_benchmark_with_options<'a, S, F>(
    algo_name: &str,
    pat_str: &str,
    pattern_index: usize,
    database: &'a DataSet<'a>,
    factory: F,
    options: CompileOptions,
) -> Vec<ResultEntry>
where
    S: StringSearch,
    F: FnMut(&mut (), &str) -> S::Config + Clone,
{
    let mut results = Vec::new();

    let pattern = compile_pattern_with_options::<S, _, _>(pat_str, (), factory, options);

    for table in database.tables.iter() {
        let table_dataset = DataSet {
            tables: vec![table.clone()].into_boxed_slice(),
        };

        let start = Instant::now();
        let matches = execute(&pattern, &table_dataset);
        let duration = start.elapsed();

        results.push(ResultEntry {
            algo: algo_name.to_string(),
            pattern_index,
            pattern: pat_str.to_string(),
            file: table.name.clone(),
            file_type: infer_file_type(&table.name),
            duration,
            found_count: matches.len(),
            skipped: false,
        });
    }

    results
}

fn fm_like_search_table<'a>(
    fm_database: &FmIndexDatabase<'a>,
    table_name: &str,
    pattern: &Pattern<'a, StdSearch>,
    pattern_str: &str,
    fm_literal_cache: &mut FmLiteralCache,
) -> usize {
    match simple_like_kind(pattern_str) {
        SimpleLike::All => return count_all_rows(fm_database, table_name),
        SimpleLike::Exact(lit) => return count_exact_rows(fm_database, table_name, lit),
        SimpleLike::Contains(lit) => {
            return count_rows_with_literal(fm_database, table_name, lit, fm_literal_cache);
        }
        SimpleLike::Prefix(lit) => return count_rows_with_prefix(fm_database, table_name, lit),
        SimpleLike::Suffix(lit) => return count_rows_with_suffix(fm_database, table_name, lit),
        SimpleLike::Complex => {}
    }

    let mut literals = split_literals(pattern_str);
    if literals.is_empty() {
        return count_like_match_all(fm_database, table_name, pattern);
    }

    literals.sort_by_key(|lit| literal_rarity(lit, &fm_database.byte_freq));

    let mut row_sets = Vec::new();
    for lit in literals.iter() {
        match rows_for_literal(fm_database, lit, fm_literal_cache) {
            FmLiteralLookup::Rows(set) => {
                if set.is_empty() {
                    return 0;
                }
                row_sets.push(set);
            }
            FmLiteralLookup::NoMatch => return 0,
            FmLiteralLookup::TooBroad => {}
        }
    }

    if row_sets.is_empty() {
        return count_like_match_all(fm_database, table_name, pattern);
    }

    let candidate_rows = intersect_row_sets(&mut row_sets);
    if candidate_rows.is_empty() {
        return 0;
    }

    let mut matched = 0usize;
    for row_idx in candidate_rows {
        let row = &fm_database.rows[row_idx];
        if row.table != table_name {
            continue;
        }
        if like_match(pattern, row.data) {
            matched += 1;
        }
    }

    matched
}

fn trigram_like_search_table<'a>(
    trigram_database: &TrigramDatabase<'a>,
    table_name: &str,
    pattern: &Pattern<'a, StdSearch>,
    pattern_str: &str,
    trigram_literal_cache: &mut TrigramLiteralCache,
) -> usize {
    let literals = split_literals(pattern_str);
    let literal = literals
        .into_iter()
        .filter(|lit| lit.len() >= 3)
        .max_by_key(|lit| lit.len());

    if let Some(lit) = literal {
        match trigram_candidates_for_literal(trigram_database, lit, trigram_literal_cache) {
            TrigramLiteralLookup::CandidateIds(candidate_ids) => {
                let mut matched = 0usize;
                for &doc_id in candidate_ids.iter() {
                    let row_idx = doc_id as usize;
                    let row = &trigram_database.rows[row_idx];
                    if row.table != table_name {
                        continue;
                    }
                    if like_match(pattern, row.data) {
                        matched += 1;
                    }
                }
                return matched;
            }
            TrigramLiteralLookup::NoMatch => return 0,
        }
    }

    trigram_database
        .rows
        .iter()
        .filter(|row| row.table == table_name && like_match(pattern, row.data))
        .count()
}

fn trigram_candidates_for_literal<'a>(
    trigram_database: &TrigramDatabase<'_>,
    lit: &str,
    trigram_literal_cache: &'a mut TrigramLiteralCache,
) -> &'a TrigramLiteralLookup {
    if !trigram_literal_cache.contains_key(lit) {
        let lookup = match trigram_database.index.search_literal(lit) {
            Some(candidate_ids) => {
                TrigramLiteralLookup::CandidateIds(candidate_ids.into_boxed_slice())
            }
            None => TrigramLiteralLookup::NoMatch,
        };
        trigram_literal_cache.insert(lit.to_string(), lookup);
    }

    trigram_literal_cache
        .get(lit)
        .expect("trigram literal cache must contain lookup")
}

#[derive(Debug, Clone, Copy)]
enum SimpleLike<'a> {
    All,
    Exact(&'a str),
    Contains(&'a str),
    Prefix(&'a str),
    Suffix(&'a str),
    Complex,
}

fn simple_like_kind(pattern: &str) -> SimpleLike<'_> {
    if pattern.contains('_') {
        return SimpleLike::Complex;
    }

    let literals: Vec<&str> = pattern.split('%').filter(|s| !s.is_empty()).collect();
    if literals.is_empty() {
        return SimpleLike::All;
    }
    if literals.len() > 1 {
        return SimpleLike::Complex;
    }

    let lit = literals[0];
    let starts = pattern.starts_with('%');
    let ends = pattern.ends_with('%');
    match (starts, ends) {
        (true, true) => SimpleLike::Contains(lit),
        (true, false) => SimpleLike::Suffix(lit),
        (false, true) => SimpleLike::Prefix(lit),
        (false, false) => SimpleLike::Exact(lit),
    }
}

fn split_literals(pattern: &str) -> Vec<&str> {
    let mut literals = Vec::new();
    let mut start = None;

    for (idx, ch) in pattern.char_indices() {
        if ch == '%' || ch == '_' {
            if let Some(s) = start.take() {
                if s < idx {
                    literals.push(&pattern[s..idx]);
                }
            }
        } else if start.is_none() {
            start = Some(idx);
        }
    }

    if let Some(s) = start {
        if s < pattern.len() {
            literals.push(&pattern[s..]);
        }
    }

    literals
}

fn count_all_rows(fm_database: &FmIndexDatabase<'_>, table_name: &str) -> usize {
    fm_database
        .rows
        .iter()
        .filter(|row| row.table == table_name)
        .count()
}

fn count_exact_rows(fm_database: &FmIndexDatabase<'_>, table_name: &str, lit: &str) -> usize {
    fm_database
        .rows
        .iter()
        .filter(|row| row.table == table_name && row.data == lit)
        .count()
}

fn count_rows_with_literal(
    fm_database: &FmIndexDatabase<'_>,
    table_name: &str,
    lit: &str,
    fm_literal_cache: &mut FmLiteralCache,
) -> usize {
    match rows_for_literal(fm_database, lit, fm_literal_cache) {
        FmLiteralLookup::Rows(rows) => {
            return rows
                .into_iter()
                .filter(|&row_idx| fm_database.rows[row_idx].table == table_name)
                .count();
        }
        FmLiteralLookup::NoMatch => return 0,
        FmLiteralLookup::TooBroad => {}
    }

    let mut s = String::with_capacity(lit.len() + 2);
    s.push('%');
    s.push_str(lit);
    s.push('%');
    let pattern = compile_pattern::<StdSearch, _, _>(&s, (), |_, pat| pat);
    count_like_match_all(fm_database, table_name, &pattern)
}

fn count_rows_with_prefix(fm_database: &FmIndexDatabase<'_>, table_name: &str, lit: &str) -> usize {
    if lit.is_empty() {
        return count_all_rows(fm_database, table_name);
    }

    let positions = match literal_positions(fm_database, lit) {
        FmPositionLookup::Positions(positions) => positions,
        FmPositionLookup::NoMatch => return 0,
        FmPositionLookup::TooBroad => {
            let mut s = String::with_capacity(lit.len() + 1);
            s.push_str(lit);
            s.push('%');
            let pattern = compile_pattern::<StdSearch, _, _>(&s, (), |_, pat| pat);
            return count_like_match_all(fm_database, table_name, &pattern);
        }
    };

    let mut matched = vec![false; fm_database.rows.len()];
    let mut count = 0usize;
    for pos in positions {
        if let Some(row_idx) = fm_database.row_index_for_pos(pos) {
            let row = &fm_database.rows[row_idx];
            if row.table != table_name || matched[row_idx] {
                continue;
            }
            if pos == row.start && row.data.starts_with(lit) {
                matched[row_idx] = true;
                count += 1;
            }
        }
    }

    count
}

fn count_rows_with_suffix(fm_database: &FmIndexDatabase<'_>, table_name: &str, lit: &str) -> usize {
    if lit.is_empty() {
        return count_all_rows(fm_database, table_name);
    }

    let positions = match literal_positions(fm_database, lit) {
        FmPositionLookup::Positions(positions) => positions,
        FmPositionLookup::NoMatch => return 0,
        FmPositionLookup::TooBroad => {
            let mut s = String::with_capacity(lit.len() + 1);
            s.push('%');
            s.push_str(lit);
            let pattern = compile_pattern::<StdSearch, _, _>(&s, (), |_, pat| pat);
            return count_like_match_all(fm_database, table_name, &pattern);
        }
    };

    let mut matched = vec![false; fm_database.rows.len()];
    let mut count = 0usize;
    for pos in positions {
        if let Some(row_idx) = fm_database.row_index_for_pos(pos) {
            let row = &fm_database.rows[row_idx];
            if row.table != table_name || matched[row_idx] {
                continue;
            }
            if pos + lit.len() == row.end && row.data.ends_with(lit) {
                matched[row_idx] = true;
                count += 1;
            }
        }
    }

    count
}

fn count_like_match_all(
    fm_database: &FmIndexDatabase<'_>,
    table_name: &str,
    pattern: &Pattern<'_, StdSearch>,
) -> usize {
    fm_database
        .rows
        .iter()
        .filter(|row| row.table == table_name && like_match(pattern, row.data))
        .count()
}

fn rows_for_literal(
    fm_database: &FmIndexDatabase<'_>,
    lit: &str,
    fm_literal_cache: &mut FmLiteralCache,
) -> FmLiteralLookup {
    if let Some(cached) = fm_literal_cache.get(lit) {
        return cached.clone();
    }

    let range = match fm_database.fm.backward_search(lit.as_bytes()) {
        Some(range) => range,
        None => {
            fm_literal_cache.insert(lit.to_string(), FmLiteralLookup::NoMatch);
            return FmLiteralLookup::NoMatch;
        }
    };

    let range_len = range.1 - range.0;
    if range_len > fm_database.max_range {
        fm_literal_cache.insert(lit.to_string(), FmLiteralLookup::TooBroad);
        return FmLiteralLookup::TooBroad;
    }

    let mut rows = HashSet::new();
    let positions = fm_database.fm.search(lit.as_bytes());
    for pos in positions {
        if let Some(row_idx) = fm_database.row_index_for_pos(pos) {
            let row = &fm_database.rows[row_idx];
            if pos + lit.len() <= row.end {
                rows.insert(row_idx);
            }
        }
    }

    let result = FmLiteralLookup::Rows(rows);
    fm_literal_cache.insert(lit.to_string(), result.clone());
    result
}

fn literal_positions(fm_database: &FmIndexDatabase<'_>, lit: &str) -> FmPositionLookup {
    let range = match fm_database.fm.backward_search(lit.as_bytes()) {
        Some(range) => range,
        None => return FmPositionLookup::NoMatch,
    };

    let range_len = range.1 - range.0;
    if range_len > fm_database.max_range {
        return FmPositionLookup::TooBroad;
    }

    FmPositionLookup::Positions(fm_database.fm.search(lit.as_bytes()))
}

fn literal_rarity(lit: &str, byte_freq: &[usize; 256]) -> usize {
    lit.as_bytes()
        .iter()
        .map(|&b| byte_freq[b as usize])
        .min()
        .unwrap_or(usize::MAX)
}

fn intersect_row_sets(sets: &mut Vec<HashSet<usize>>) -> Vec<usize> {
    if sets.is_empty() {
        return Vec::new();
    }

    sets.sort_by_key(|s| s.len());
    let mut iter = sets.iter();
    let mut acc = iter.next().cloned().unwrap_or_default();
    for set in iter {
        acc.retain(|idx| set.contains(idx));
        if acc.is_empty() {
            break;
        }
    }

    acc.into_iter().collect()
}

fn skipped_entries<'a>(
    algo_name: &str,
    pat_str: &str,
    pattern_index: usize,
    database: &'a DataSet<'a>,
) -> Vec<ResultEntry> {
    database
        .tables
        .iter()
        .map(|table| ResultEntry {
            algo: algo_name.to_string(),
            pattern_index,
            pattern: pat_str.to_string(),
            file: table.name.clone(),
            file_type: infer_file_type(&table.name),
            duration: Duration::from_micros(0),
            found_count: 0,
            skipped: true,
        })
        .collect()
}

fn build_fm_index<'a>(database: &'a DataSet<'a>) -> (FmIndexDatabase<'a>, Duration) {
    let start = Instant::now();
    let mut text = Vec::new();
    let mut rows = Vec::new();
    let mut row_starts = Vec::new();
    let mut byte_freq = [0usize; 256];

    let total_input_bytes: usize = database
        .tables
        .iter()
        .flat_map(|t| t.rows.iter())
        .map(|r| r.data.len())
        .sum();
    let mut processed_bytes = 0usize;
    let mut next_pct = 1usize;

    for table in database.tables.iter() {
        let table_name = table.name.as_str();
        for row in table.rows.iter() {
            let bytes = row.data.as_bytes();
            if bytes.iter().any(|&b| b == FM_SENTINEL || b == FM_SEPARATOR) {
                panic!("row contains reserved FM-index byte");
            }

            let start_offset = text.len();
            text.extend_from_slice(bytes);
            let end_offset = text.len();

            for &b in bytes {
                byte_freq[b as usize] += 1;
                processed_bytes += 1;
                if total_input_bytes > 0 {
                    while next_pct <= 100
                        && processed_bytes.saturating_mul(100)
                            >= total_input_bytes.saturating_mul(next_pct)
                    {
                        println!(
                            "> FM build ingest progress: {}% ({}/{})",
                            next_pct, processed_bytes, total_input_bytes
                        );
                        next_pct += 1;
                    }
                }
            }

            rows.push(FmRow {
                table: table_name,
                data: row.data,
                start: start_offset,
                end: end_offset,
            });
            row_starts.push(start_offset);

            text.push(FM_SEPARATOR);
        }
    }

    let corpus_len = text.len();
    let max_range = std::cmp::max(100_000usize, corpus_len / 100);

    println!(
        "> FM corpus size: {} bytes across {} rows. Starting suffix-array build...",
        corpus_len,
        rows.len()
    );

    let fm = FMIndex::new(text, FM_SENTINEL, Some(FM_SEPARATOR));
    let duration = start.elapsed();

    (
        FmIndexDatabase {
            fm,
            rows,
            row_starts,
            byte_freq,
            max_range,
        },
        duration,
    )
}

fn build_trigram_index<'a>(database: &'a DataSet<'a>) -> TrigramDatabase<'a> {
    let mut index = TrigramIndex::new();
    let mut rows = Vec::new();

    for table in database.tables.iter() {
        let table_name = table.name.as_str();
        for row in table.rows.iter() {
            rows.push(TrigramRow {
                table: table_name,
                data: row.data,
            });
            index.add(row.data);
        }
    }

    TrigramDatabase { index, rows }
}

fn fftstr_max_literal_len(pattern: &str) -> usize {
    pattern.split('%').map(str::len).max().unwrap_or(0)
}

fn log2ceil(value: u64) -> u32 {
    assert!(value > 1);
    let v = value - 1;
    64 - v.leading_zeros()
}

fn fftstr_log2n(pattern: &str) -> u32 {
    let max_literal_len = fftstr_max_literal_len(pattern) as u64;
    let required = max_literal_len.saturating_mul(3);
    if required <= 1 {
        return 0;
    }
    log2ceil(required)
}

fn should_skip_fftstr0(pattern: &str) -> bool {
    let log2n = fftstr_log2n(pattern);
    log2n == 0 || log2n >= 8
}

fn should_skip_fftstr1(pattern: &str) -> bool {
    let log2n = fftstr_log2n(pattern);
    log2n == 0 || log2n > 27
}

fn infer_file_type(file_name: &str) -> String {
    match Path::new(file_name)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("fasta") | Some("fa") | Some("fna") | Some("faa") | Some("fsa") => "FASTA".to_string(),
        _ => "TEXT".to_string(),
    }
}

pub fn load_patterns_from_file(path: &Path) -> Result<Vec<PatternSpec>, String> {
    let file = File::open(path)
        .map_err(|err| format!("Failed to open pattern file {}: {err}", path.display()))?;
    let reader = BufReader::new(file);

    let mut patterns = Vec::new();
    for (line_no, line_res) in reader.lines().enumerate() {
        let line = line_res.map_err(|err| {
            format!(
                "Failed to read line {} from {}: {err}",
                line_no + 1,
                path.display()
            )
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let mut parts = trimmed.splitn(2, '\t');
        let pattern = parts.next().unwrap_or("").trim().to_string();
        if pattern.is_empty() {
            continue;
        }

        let description = parts
            .next()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("Pattern {}", patterns.len() + 1));

        patterns.push(PatternSpec {
            pattern,
            description,
        });
    }

    if patterns.is_empty() {
        return Err(format!(
            "Pattern file {} did not contain any usable patterns",
            path.display()
        ));
    }

    Ok(patterns)
}

fn compute_table_stats(database: &DataSet<'_>) -> HashMap<String, TableStats> {
    let mut map = HashMap::new();
    for table in database.tables.iter() {
        let rows = table.rows.len();
        if rows == 0 {
            map.insert(
                table.name.clone(),
                TableStats {
                    rows: 0,
                    avg_len: 0.0,
                    max_len: 0,
                },
            );
            continue;
        }

        let mut sum = 0usize;
        let mut max_len = 0usize;
        for row in table.rows.iter() {
            let len = row.data.len();
            sum += len;
            if len > max_len {
                max_len = len;
            }
        }

        map.insert(
            table.name.clone(),
            TableStats {
                rows,
                avg_len: (sum as f64) / (rows as f64),
                max_len,
            },
        );
    }
    map
}

fn csv_escape(value: &str) -> String {
    let needs_quotes = value.contains(',') || value.contains('"') || value.contains('\n');
    if !needs_quotes {
        return value.to_string();
    }

    let escaped = value.replace('"', "\"\"");
    format!("\"{}\"", escaped)
}

fn write_results_csv(
    path: &Path,
    dataset_name: &str,
    patterns: &[PatternSpec],
    results: &[ResultEntry],
    table_stats: &HashMap<String, TableStats>,
) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|err| format!("Failed to create CSV {}: {err}", path.display()))?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "dataset,algorithm,pattern_index,pattern,pattern_desc,table,file_type,duration_micros,found_count,skipped,table_rows,table_avg_len,table_max_len"
    )
    .map_err(|err| format!("Failed to write CSV header to {}: {err}", path.display()))?;

    for entry in results {
        let pattern_desc = patterns
            .get(entry.pattern_index)
            .map(|spec| spec.description.as_str())
            .unwrap_or("");
        let stats = table_stats.get(&entry.file).copied().unwrap_or(TableStats {
            rows: 0,
            avg_len: 0.0,
            max_len: 0,
        });

        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{:.4},{}",
            csv_escape(dataset_name),
            csv_escape(&entry.algo),
            entry.pattern_index,
            csv_escape(&entry.pattern),
            csv_escape(pattern_desc),
            csv_escape(&entry.file),
            csv_escape(&entry.file_type),
            entry.duration.as_micros(),
            entry.found_count,
            entry.skipped,
            stats.rows,
            stats.avg_len,
            stats.max_len,
        )
        .map_err(|err| format!("Failed to write CSV row to {}: {err}", path.display()))?;
    }

    writer
        .flush()
        .map_err(|err| format!("Failed to flush CSV {}: {err}", path.display()))?;
    Ok(())
}

fn print_summary_table(results: &[ResultEntry]) {
    println!("\n\n{:=^95}", " RESULTS SUMMARY ");
    println!(
        "{:<15} | {:<20} | {:<20} | {:<6} | {:>10} | {:>15}",
        "Algorithm", "Pattern", "File", "Type", "Hits", "Time (µs)"
    );
    println!("{:-^95}", "");

    for entry in results.iter().filter(|entry| !entry.skipped) {
        let micros = entry.duration.as_micros() as f64;

        let pat_display = if entry.pattern.len() > 20 {
            format!("{}...", &entry.pattern[..17])
        } else {
            entry.pattern.clone()
        };

        let file_display = if entry.file.len() > 20 {
            format!("{}...", &entry.file[..17])
        } else {
            entry.file.clone()
        };

        let hits_display = entry.found_count.to_string();
        let time_display = format!("{:.2}", micros);

        println!(
            "{:<15} | {:<20} | {:<20} | {:<6} | {:>10} | {:>15}",
            entry.algo, pat_display, file_display, entry.file_type, hits_display, time_display
        );
    }
    println!("{:=^95}", " END ");
}

fn print_algo_ranking(results: &[ResultEntry]) {
    println!("\n\n{:=^50}", " SPEED RANKING ");
    println!(
        "{:<5} | {:<15} | {:>20}",
        "Rank", "Algorithm", "Total Time (ms)"
    );
    println!("{:-^50}", "");

    let mut sums: HashMap<String, Duration> = HashMap::new();

    for entry in results.iter().filter(|entry| !entry.skipped) {
        *sums.entry(entry.algo.clone()).or_default() += entry.duration;
    }

    let mut ranked: Vec<(String, Duration)> = sums.into_iter().collect();

    ranked.sort_by_key(|(_, duration)| *duration);

    for (i, (algo, duration)) in ranked.iter().enumerate() {
        let millis = duration.as_millis();
        println!("{:<5} | {:<15} | {:>20}", i + 1, algo, millis);
    }

    println!("{:=^50}", " END ");
}

fn print_per_pattern_ranking(results: &[ResultEntry]) {
    println!("\n\n{:=^60}", " PER PATTERN RANKING ");

    let mut unique_patterns: Vec<&String> = results.iter().map(|r| &r.pattern).collect();
    unique_patterns.sort();
    unique_patterns.dedup();

    for pat in unique_patterns {
        println!("\n>> Pattern: [{}]", pat);
        println!(
            "{:<5} | {:<15} | {:>20}",
            "Rank", "Algorithm", "Total Time (µs)"
        );
        println!("{:-^46}", "");

        let mut sums: HashMap<&String, Duration> = HashMap::new();
        for entry in results.iter().filter(|r| &r.pattern == pat && !r.skipped) {
            *sums.entry(&entry.algo).or_default() += entry.duration;
        }

        let mut ranked: Vec<(&String, Duration)> = sums.into_iter().collect();
        ranked.sort_by_key(|(_, d)| *d);

        for (i, (algo, duration)) in ranked.iter().enumerate() {
            println!("{:<5} | {:<15} | {:>20}", i + 1, algo, duration.as_micros());
        }
    }
    println!("\n{:=^60}", " END PATTERN RANKING ");
}

fn print_per_file_ranking(results: &[ResultEntry]) {
    println!("\n\n{:=^60}", " PER FILE RANKING ");

    let mut unique_files: Vec<&String> = results.iter().map(|r| &r.file).collect();
    unique_files.sort();
    unique_files.dedup();

    for file in unique_files {
        println!("\n>> File: [{}]", file);
        println!(
            "{:<5} | {:<15} | {:>20}",
            "Rank", "Algorithm", "Total Time (µs)"
        );
        println!("{:-^46}", "");

        let mut sums: HashMap<&String, Duration> = HashMap::new();
        for entry in results.iter().filter(|r| &r.file == file && !r.skipped) {
            *sums.entry(&entry.algo).or_default() += entry.duration;
        }

        let mut ranked: Vec<(&String, Duration)> = sums.into_iter().collect();
        ranked.sort_by_key(|(_, d)| *d);

        for (i, (algo, duration)) in ranked.iter().enumerate() {
            println!("{:<5} | {:<15} | {:>20}", i + 1, algo, duration.as_micros());
        }
    }
    println!("\n{:=^60}", " END FILE RANKING ");
}

fn print_correctness_report(results: &[ResultEntry]) {
    let mut baseline = HashMap::<(usize, String), usize>::new();
    let mut mismatches = Vec::<Mismatch>::new();
    let mut total_checks = 0usize;

    for entry in results.iter().filter(|r| r.algo == "std" && !r.skipped) {
        baseline.insert((entry.pattern_index, entry.file.clone()), entry.found_count);
    }

    for entry in results.iter().filter(|r| r.algo != "std" && !r.skipped) {
        total_checks += 1;
        match baseline.get(&(entry.pattern_index, entry.file.clone())) {
            Some(expected) if *expected == entry.found_count => {}
            Some(expected) => mismatches.push(Mismatch {
                algo: entry.algo.clone(),
                pattern_index: entry.pattern_index,
                pattern: entry.pattern.clone(),
                file: entry.file.clone(),
                expected: *expected,
                actual: entry.found_count,
            }),
            None => mismatches.push(Mismatch {
                algo: entry.algo.clone(),
                pattern_index: entry.pattern_index,
                pattern: entry.pattern.clone(),
                file: entry.file.clone(),
                expected: 0,
                actual: entry.found_count,
            }),
        }
    }

    println!("\n\n{:=^70}", " CORRECTNESS REPORT ");
    println!("Checks: {}, Mismatches: {}", total_checks, mismatches.len());

    if mismatches.is_empty() {
        println!("All algorithms match StdSearch baseline.");
        println!("{:=^70}", " END ");
        return;
    }

    println!(
        "{:<12} | {:<5} | {:<18} | {:<20} | {:>8} | {:>8}",
        "Algorithm", "Idx", "Pattern", "File", "Expected", "Actual"
    );
    println!("{:-^70}", "");

    for mismatch in mismatches {
        let pat_display = if mismatch.pattern.len() > 18 {
            format!("{}...", &mismatch.pattern[..15])
        } else {
            mismatch.pattern.clone()
        };

        let file_display = if mismatch.file.len() > 20 {
            format!("{}...", &mismatch.file[..17])
        } else {
            mismatch.file.clone()
        };

        println!(
            "{:<12} | {:<5} | {:<18} | {:<20} | {:>8} | {:>8}",
            mismatch.algo,
            mismatch.pattern_index,
            pat_display,
            file_display,
            mismatch.expected,
            mismatch.actual
        );
    }

    println!("{:=^70}", " END ");
}
