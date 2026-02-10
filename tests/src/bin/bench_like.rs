use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use algos::{
    FftConfig, FftStr0, FftStr1, NaiveScalar, NaiveVectorized, StdSearch, StringSearch, BM, KMP,
};
use engine::execute;
use like::{compile_pattern, compile_pattern_with_options, CompileOptions};
use storage::{
    dataset::{load_dataset_from_paths, DataSet},
    BumpArena,
};

const ARENA_SIZE: usize = 1024 * 1024 * 1024; // 1 GB

const PLAIN_TEXT_FILES: &[&str] = &["data/ipsum.txt"];

const FASTA_FILES: &[&str] = &[
    "data/fasta/covid19_genome.fasta",
    "data/fasta/ecoli_genome.fasta",
    "data/fasta/human_p53_protein.fasta",
    "data/fasta/human_proteome.fasta",
];

const PATTERNS: &[(&str, &str)] = &[
    ("%TCGC%", "Short DNA"),
    ("%GATTACA%", "Medium DNA"),
    ("%Lorem%", "Common Word"),
    ("h%o", "Wildcard Simple"),
    ("%a_%_b%", "Wildcard Complex"),
    ("%a_%_b%", "Wildcard Complex2"),
    ("%GA_T%____T%%ACA%", "Complex DNA Pattern"),
    ("%TGCGAGATTTGGACGGAC%", "DNA Simple (original complex)"),
    ("%TGCG%A___GGACGGAC%", "Complex DNA Pattern2"),
    (
        "%GTTGCGAGATTTGGACGGACGTTGACGGGGTCTATACCTGCGACCCGCGT
%CAGGTGCCCGATGCGAGGTTGTTGAAGTCGATGTCCTACCAGGAAGCGATGGAGCTTTCCTACTTCGGCG
CTAAAGTTCTTCACCCCCGCACCATTACCCCCATCGCCCAGTTCCAGATCCCTTGCCTGATTAAAAATAC
CGGAAATCCTCAAGCACCAGGTACGCTCATTGGTGCCAGCCGTGATGAAGACGAATTACCGGTCAAGGGC
ATTTCCAATCTGAATAACATGGCAATGTTCAGCGTTTCTGGTCCGGGGATGAAAGGGATGGTCGGCATGG
CGGCGCGCGTCTTTGCAGCGATGTCACGCGCCCGTATTTCCGTGGTGCTGATTACGCAATCATCTTCCGA
ATACAGCATCAGTTTCTGCGTTCCACAAAGCGACTGTGT%",
        "Long DNA Patern",
    ),
];

const ALGORITHMS: &[&str] = &[
    "naive-scalar",
    "naive-vector",
    "kmp",
    "bm",
    "std",
    "fftstr0",
    "fftstr1",
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

fn main() {
    println!("--- Starting Like Benchmark ---");

    println!("> Allocating {} bytes for Arena...", ARENA_SIZE);
    let arena = BumpArena::new(ARENA_SIZE);

    println!("> Loading Database into memory...");
    let database = load_database(&arena);
    println!("> Database loaded. Total files: {}", database.tables.len());
    println!("> Arena used: {} bytes", arena.used());

    let mut results = Vec::new();

    for algo_name in ALGORITHMS {
        for (pattern_index, (pat_str, _pat_desc)) in PATTERNS.iter().enumerate() {
            println!(
                "> Benchmarking Algo: [{}] Pattern: [{}]",
                algo_name, pat_str
            );

            let entries = match *algo_name {
                "naive-scalar" => run_benchmark::<NaiveScalar, _>(
                    algo_name,
                    pat_str,
                    pattern_index,
                    &database,
                    |_, pat| unsafe { std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes()) },
                ),
                "naive-vector" => {
                    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                    {
                        run_benchmark::<NaiveVectorized, _>(
                            algo_name,
                            pat_str,
                            pattern_index,
                            &database,
                            |_, pat| unsafe { std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes()) },
                        )
                    }

                    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
                    {
                        let _: [NaiveVectorized; 0] = [];
                        Vec::new()
                    }
                }
                "kmp" => run_benchmark::<KMP, _>(
                    algo_name,
                    pat_str,
                    pattern_index,
                    &database,
                    |_, pat| unsafe { std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes()) },
                ),
                "bm" => run_benchmark::<BM, _>(
                    algo_name,
                    pat_str,
                    pattern_index,
                    &database,
                    |_, pat| unsafe { std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes()) },
                ),
                "std" => run_benchmark::<StdSearch, _>(
                    algo_name,
                    pat_str,
                    pattern_index,
                    &database,
                    |_, pat| unsafe { std::mem::transmute::<&str, &str>(pat) },
                ),
                "fftstr0" => {
                    if should_skip_fftstr0(pat_str) {
                        skipped_entries(algo_name, pat_str, pattern_index, &database)
                    } else {
                        run_benchmark_with_options::<FftStr0, _>(
                            algo_name,
                            pat_str,
                            pattern_index,
                            &database,
                            |_, pat| FftConfig::from_str(pat),
                            CompileOptions {
                                treat_underscore_as_literal: true,
                                literal_underscore_is_wildcard: true,
                            },
                        )
                    }
                }
                "fftstr1" => {
                    if should_skip_fftstr1(pat_str) {
                        skipped_entries(algo_name, pat_str, pattern_index, &database)
                    } else {
                        run_benchmark_with_options::<FftStr1, _>(
                            algo_name,
                            pat_str,
                            pattern_index,
                            &database,
                            |_, pat| FftConfig::from_str(pat),
                            CompileOptions {
                                treat_underscore_as_literal: true,
                                literal_underscore_is_wildcard: true,
                            },
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

fn load_database<'a>(arena: &'a BumpArena) -> DataSet<'a> {
    let mut paths: Vec<PathBuf> = Vec::new();

    for file_path in PLAIN_TEXT_FILES.iter().chain(FASTA_FILES.iter()) {
        if !Path::new(file_path).exists() {
            eprintln!("  ! Skipping missing file: {}", file_path);
            continue;
        }

        paths.push(PathBuf::from(file_path));
    }

    load_dataset_from_paths(arena, &paths).expect("load dataset")
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

    for entry in results {
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

        let hits_display = if entry.skipped {
            "-".to_string()
        } else {
            entry.found_count.to_string()
        };
        let time_display = if entry.skipped {
            "SKIP".to_string()
        } else {
            format!("{:.2}", micros)
        };

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
