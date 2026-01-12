use std::{
    collections::HashMap,
    fs,
    path::Path,
    time::{Duration, Instant},
};

use algos::{BM, KMP, NaiveScalar, NaiveVectorized, StdSearch, StringSearch};
use like::{compile_pattern, like_match};
use storage::{BumpArena, fasta};

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
    ("%GAT%____TACA%", "Complex Dna"),
];

const ALGORITHMS: &[&str] = &["naive-scalar", "naive-vector", "kmp", "bm", "std"];

#[derive(Debug)]
struct ResultEntry {
    algo: String,
    pattern: String,
    file: String,
    file_type: String,
    duration: Duration,
    found_count: usize,
}

struct DatabaseItem<'a> {
    name: String,
    file_type: String,
    entries: Vec<&'a str>,
}

fn main() {
    println!("--- Starting Like Benchmark ---");

    println!("> Allocating {} bytes for Arena...", ARENA_SIZE);
    let arena = BumpArena::new(ARENA_SIZE);

    println!("> Loading Database into memory...");
    let database = load_database(&arena);
    println!("> Database loaded. Total files: {}", database.len());
    println!("> Arena used: {} bytes", arena.used());

    let mut results = Vec::new();

    for algo_name in ALGORITHMS {
        for (pat_str, _pat_desc) in PATTERNS {
            println!(
                "> Benchmarking Algo: [{}] Pattern: [{}]",
                algo_name, pat_str
            );

            let entries = match *algo_name {
                "naive-scalar" => {
                    run_benchmark::<NaiveScalar, _>(algo_name, pat_str, &database, |_, pat| unsafe {
                        std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes())
                    })
                }
                "naive-vectorized" => {
                    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                    {
                    run_benchmark::<NaiveVectorized, _>(algo_name, pat_str, &database, |_, pat| unsafe {
                        std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes())
                    })
                    }

                    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
                    {
                        let _: [NaiveVectorized; 0] = [];
                        Vec::new()
                    }
                }
                "naive-vector" => {
                    Vec::new()
                }
                "kmp" => run_benchmark::<KMP, _>(algo_name, pat_str, &database, |_, pat| unsafe {
                    std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes())
                }),
                "bm" => run_benchmark::<BM, _>(algo_name, pat_str, &database, |_, pat| unsafe {
                    std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes())
                }),
                "std" => {
                    run_benchmark::<StdSearch, _>(algo_name, pat_str, &database, |_, pat| unsafe {
                        std::mem::transmute::<&str, &str>(pat)
                    })
                }
                _ => panic!("Unknown algorithm: {}", algo_name),
            };

            results.extend(entries);
        }
    }

    print_summary_table(&results);
    print_algo_ranking(&results);
}

fn load_database<'a>(arena: &'a BumpArena) -> Vec<DatabaseItem<'a>> {
    let mut items = Vec::new();

    for file_path in PLAIN_TEXT_FILES {
        if !Path::new(file_path).exists() {
            eprintln!("  ! Skipping missing file: {}", file_path);
            continue;
        }

        let raw_string = fs::read_to_string(file_path).expect("Failed to read text file");

        let arena_str = arena.alloc_str(&raw_string);

        items.push(DatabaseItem {
            name: extract_filename(file_path),
            file_type: "TEXT".to_string(),
            entries: vec![arena_str],
        });
    }

    for file_path in FASTA_FILES {
        if !Path::new(file_path).exists() {
            eprintln!("  ! Skipping missing FASTA: {}", file_path);
            continue;
        }

        let raw_bytes = fs::read(file_path).expect("Failed to read FASTA bytes");

        match fasta::parse_fasta_into_arena(arena, &raw_bytes) {
            Ok(parsed_entries) => {
                let entries: Vec<&str> = parsed_entries.iter().map(|e| e.data).collect();

                items.push(DatabaseItem {
                    name: extract_filename(file_path),
                    file_type: "FASTA".to_string(),
                    entries,
                });
            }
            Err(e) => eprintln!("  ! FASTA Parse Error in {}: {}", file_path, e),
        }
    }

    items
}

fn run_benchmark<'a, S, F>(
    algo_name: &str,
    pat_str: &str,
    database: &[DatabaseItem<'a>],
    factory: F,
) -> Vec<ResultEntry>
where
    S: StringSearch,
    F: FnMut(&mut (), &str) -> S::Config + Clone,
{
    let mut results = Vec::new();

    let pattern = compile_pattern::<S, _, _>(pat_str, (), factory);

    for item in database {
        let start = Instant::now();
        let mut match_count = 0;

        for entry in &item.entries {
            if like_match(&pattern, entry) {
                match_count += 1;
            }
        }

        let duration = start.elapsed();

        results.push(ResultEntry {
            algo: algo_name.to_string(),
            pattern: pat_str.to_string(),
            file: item.name.clone(),
            file_type: item.file_type.clone(),
            duration,
            found_count: match_count,
        });
    }

    results
}

fn extract_filename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
}

fn print_summary_table(results: &[ResultEntry]) {
    println!("\n\n{:=^95}", " RESULTS SUMMARY ");
    println!(
        "{:<15} | {:<15} | {:<20} | {:<6} | {:>10} | {:>15}",
        "Algorithm", "Pattern", "File", "Type", "Hits", "Time (µs)"
    );
    println!("{:-^95}", "");

    for entry in results {
        let micros = entry.duration.as_micros() as f64;

        let pat_display = if entry.pattern.len() > 15 {
            format!("{}...", &entry.pattern[..9])
        } else {
            entry.pattern.clone()
        };

        let file_display = if entry.file.len() > 20 {
            format!("{}...", &entry.file[..17])
        } else {
            entry.file.clone()
        };

        println!(
            "{:<15} | {:<15} | {:<20} | {:<6} | {:>10} | {:>15.2}",
            entry.algo, pat_display, file_display, entry.file_type, entry.found_count, micros
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

    for entry in results {
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
