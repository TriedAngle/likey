use std::{
    collections::{HashMap, HashSet},
    path::Path,
    time::{Duration, Instant},
};

use algos::{
    FMIndex, FftConfig, FftStr0, FftStr1, NaiveScalar, NaiveVectorized, StdSearch, StringSearch,
    TrigramIndex, BM, KMP,
};
use engine::execute;
use like::{compile_pattern, compile_pattern_with_options, like_match, CompileOptions, Pattern};
use storage::dataset::DataSet;

const FM_SEPARATOR: u8 = 0x1F;
const FM_SENTINEL: u8 = 0x00;

const ALGORITHMS: &[&str] = &[
    "naive-scalar",
    "naive-vector",
    "kmp",
    "bm",
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

struct TrigramDatabase<'a> {
    index: TrigramIndex<'a>,
    rows: Vec<TrigramRow<'a>>,
}

#[derive(Debug, Clone, Copy)]
struct TrigramRow<'a> {
    table: &'a str,
    data: &'a str,
}

#[derive(Debug, Clone, Copy)]
pub struct BenchOptions {
    pub skip_naive_scalar: bool,
    pub skip_naive_vector: bool,
    pub skip_kmp: bool,
    pub skip_bm: bool,
    pub skip_std: bool,
    pub skip_lut_short: bool,
    pub skip_fftstr0: bool,
    pub skip_fftstr1: bool,
    pub skip_fm: bool,
    pub skip_trigram: bool,
}

impl<'a> FmIndexDatabase<'a> {
    fn row_index_for_pos(&self, pos: usize) -> Option<usize> {
        let idx = match self.row_starts.binary_search(&pos) {
            Ok(i) => i,
            Err(0) => return None,
            Err(i) => i - 1,
        };

        let row = &self.rows[idx];
        if pos < row.end {
            Some(idx)
        } else {
            None
        }
    }
}

pub fn run_like_benchmarks(
    database: &DataSet<'_>,
    patterns: &[(&str, &str)],
    options: BenchOptions,
) {
    let skip_lut_short = options.skip_lut_short || !lut_short_available();
    let skip_naive_vector = options.skip_naive_vector || !naive_vector_available();

    println!("--- Starting Like Benchmark ---");
    println!("> Database loaded. Total tables: {}", database.tables.len());

    let fm_database = if options.skip_fm {
        None
    } else {
        println!("> Building FM-index...");
        let (fm_database, fm_build_time) = build_fm_index(database);
        println!("> FM-index built in {} ms", fm_build_time.as_millis());
        Some(fm_database)
    };

    let trigram_database = if options.skip_trigram {
        None
    } else {
        println!("> Building trigram index...");
        let start = Instant::now();
        let trigram = build_trigram_index(database);
        let duration = start.elapsed();
        println!("> Trigram index built in {} ms", duration.as_millis());
        Some(trigram)
    };

    let mut fm_literal_cache: HashMap<String, HashSet<usize>> = HashMap::new();

    let mut results = Vec::new();

    for algo_name in ALGORITHMS {
        for (pattern_index, (pat_str, _pat_desc)) in patterns.iter().enumerate() {
            println!(
                "> Benchmarking Algo: [{}] Pattern: [{}]",
                algo_name, pat_str
            );

            let entries = match *algo_name {
                "naive-scalar" => {
                    if options.skip_naive_scalar {
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
                    if skip_naive_vector {
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
                    if options.skip_kmp {
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
                "bm" => {
                    if options.skip_bm {
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
                "std" => {
                    if options.skip_std {
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
                    if options.skip_lut_short || skip_lut_short {
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
                    if options.skip_fftstr0 || should_skip_fftstr0(pat_str) {
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
                            },
                        )
                    }
                }
                "fftstr1" => {
                    if options.skip_fftstr1 || should_skip_fftstr1(pat_str) {
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
                            },
                        )
                    }
                }
                "fm" => {
                    if options.skip_fm {
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
                    if options.skip_trigram {
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
                        )
                    }
                }
                _ => panic!("Unknown algorithm: {}", algo_name),
            };

            results.extend(entries);
        }
    }

    print_summary_table(&results);
    print_algo_ranking(&results);
    print_per_pattern_ranking(&results);
    print_per_file_ranking(&results);
    print_correctness_report(&results);
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
    fm_literal_cache: &mut HashMap<String, HashSet<usize>>,
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
        let found = trigram_like_search_table(trigram_database, table_name, &pattern, pat_str);
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
    fm_literal_cache: &mut HashMap<String, HashSet<usize>>,
) -> usize {
    match simple_like_kind(pattern_str) {
        SimpleLike::All => return count_all_rows(fm_database, table_name),
        SimpleLike::Exact(lit) => return count_exact_rows(fm_database, table_name, lit),
        SimpleLike::Contains(lit) => {
            return count_rows_with_literal(fm_database, table_name, lit, fm_literal_cache)
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
        if let Some(set) = rows_for_literal(fm_database, lit, fm_literal_cache) {
            if set.is_empty() {
                return 0;
            }
            row_sets.push(set);
        } else {
            return count_like_match_all(fm_database, table_name, pattern);
        }
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
) -> usize {
    let literals = split_literals(pattern_str);
    let literal = literals
        .into_iter()
        .filter(|lit| lit.len() >= 3)
        .max_by_key(|lit| lit.len());

    if let Some(lit) = literal {
        if let Some(candidate_ids) = trigram_database.index.search_literal(lit) {
            let mut matched = 0usize;
            for doc_id in candidate_ids {
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
    }

    trigram_database
        .rows
        .iter()
        .filter(|row| row.table == table_name && like_match(pattern, row.data))
        .count()
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
    fm_literal_cache: &mut HashMap<String, HashSet<usize>>,
) -> usize {
    if let Some(rows) = rows_for_literal(fm_database, lit, fm_literal_cache) {
        return rows
            .into_iter()
            .filter(|&row_idx| fm_database.rows[row_idx].table == table_name)
            .count();
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
        Some(positions) => positions,
        None => {
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
        Some(positions) => positions,
        None => {
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
    fm_literal_cache: &mut HashMap<String, HashSet<usize>>,
) -> Option<HashSet<usize>> {
    if let Some(cached) = fm_literal_cache.get(lit) {
        return Some(cached.clone());
    }

    let range = fm_database.fm.backward_search(lit.as_bytes())?;
    let range_len = range.1 - range.0;
    if range_len > fm_database.max_range {
        return None;
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

    fm_literal_cache.insert(lit.to_string(), rows.clone());
    Some(rows)
}

fn literal_positions(fm_database: &FmIndexDatabase<'_>, lit: &str) -> Option<Vec<usize>> {
    let range = fm_database.fm.backward_search(lit.as_bytes())?;
    let range_len = range.1 - range.0;
    if range_len > fm_database.max_range {
        return None;
    }

    Some(fm_database.fm.search(lit.as_bytes()))
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
