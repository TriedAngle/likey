use std::fs;
use std::path::{Path, PathBuf};

use storage::{
    dataset::{load_fasta_table, DataSet, Row},
    BumpArena,
};

mod bench_shared;
use bench_shared::{load_patterns_from_file, run_like_benchmarks, BenchOptions, PatternSpec};

const DEFAULT_ARENA_GB: usize = 4;

const PATTERNS: &[(&str, &str)] = &[
    ("ATG%", "Start codon prefix"),
    ("%TAA", "Stop codon suffix"),
    ("%CGCG%", "CpG-rich motif"),
    ("%AATAAA%", "PolyA signal motif"),
    ("%ATG___TAA%", "Codon spacer motif"),
    ("%TTTTT%", "Poly-T run"),
];

fn main() {
    let data_dir = arg_value("--data-dir").unwrap_or_else(|| "data/benchmarks/dna".to_string());
    let arena_gb = arg_value("--arena-gb")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_ARENA_GB);
    let max_bytes = arg_value("--max-bytes").and_then(|v| v.parse::<usize>().ok());
    let max_rows_per_table =
        arg_value("--max-rows-per-table").and_then(|v| v.parse::<usize>().ok());
    let patterns_file = arg_value("--patterns-file");
    let output_csv = arg_value("--output-csv");

    let mut arena_size = arena_gb * 1024 * 1024 * 1024;
    if let Some(max_bytes) = max_bytes {
        arena_size = arena_size.min(max_bytes);
    }
    let arena = BumpArena::new(arena_size);

    println!("--- DNA Like Benchmark ---");
    println!("> Using data dir: {}", data_dir);
    println!("> Allocating {} bytes for Arena...", arena_size);
    if let Some(max_rows_per_table) = max_rows_per_table {
        println!("> Max rows per table cap: {}", max_rows_per_table);
    }

    let dataset = load_fasta_dataset(&arena, Path::new(&data_dir), max_bytes, max_rows_per_table);

    let patterns = match patterns_file {
        Some(path) => {
            let path_ref = Path::new(&path);
            println!("> Loading patterns from: {}", path_ref.display());
            load_patterns_from_file(path_ref).expect("load patterns file")
        }
        None => default_patterns(),
    };

    let options = BenchOptions {
        skip_naive_scalar: has_flag("--skip-naive-scalar"),
        skip_naive_vector: has_flag("--skip-naive-vector"),
        skip_kmp: has_flag("--skip-kmp"),
        skip_bm: has_flag("--skip-bm"),
        skip_two_way: has_flag("--skip-two-way"),
        skip_std: has_flag("--skip-std"),
        skip_lut_short: has_flag("--skip-lut-short"),
        skip_fftstr0: has_flag("--skip-fftstr0"),
        skip_fftstr1: has_flag("--skip-fftstr1"),
        skip_fm: has_flag("--skip-fm"),
        skip_trigram: has_flag("--skip-trigram"),
    };

    run_like_benchmarks(
        &dataset,
        "dna",
        &patterns,
        options,
        output_csv.as_deref().map(Path::new),
    );
}

fn default_patterns() -> Vec<PatternSpec> {
    PATTERNS
        .iter()
        .map(|(pattern, description)| PatternSpec {
            pattern: (*pattern).to_string(),
            description: (*description).to_string(),
        })
        .collect()
}

fn load_fasta_dataset<'a>(
    arena: &'a BumpArena,
    data_dir: &Path,
    max_bytes: Option<usize>,
    max_rows_per_table: Option<usize>,
) -> DataSet<'a> {
    let files = collect_fasta_files(data_dir);
    assert!(
        !files.is_empty(),
        "No FASTA files found in {}",
        data_dir.display()
    );

    let mut tables = Vec::with_capacity(files.len());
    let mut remaining_bytes = max_bytes.unwrap_or(usize::MAX);
    for file in files {
        let mut table = load_fasta_table(arena, &file).expect("load fasta table");
        if let Some(cap) = max_rows_per_table {
            if table.rows.len() > cap {
                let mut rows = std::mem::take(&mut table.rows).into_vec();
                rows.truncate(cap);
                table.rows = rows.into_boxed_slice();
            }
        }

        if remaining_bytes != usize::MAX {
            let mut kept: Vec<Row<'a>> = Vec::new();
            for row in table.rows.iter().cloned() {
                let len = row.data.len();
                if len <= remaining_bytes {
                    kept.push(row);
                    remaining_bytes -= len;
                } else if remaining_bytes > 0 && kept.is_empty() {
                    let partial = &row.data.as_bytes()[..remaining_bytes];
                    let partial = std::str::from_utf8(partial).expect("dna utf8");
                    kept.push(Row {
                        id: row.id,
                        desc: row.desc,
                        data: partial,
                    });
                    remaining_bytes = 0;
                }

                if remaining_bytes == 0 {
                    break;
                }
            }
            table.rows = kept.into_boxed_slice();
        }

        tables.push(table);

        if remaining_bytes == 0 {
            break;
        }
    }

    DataSet {
        tables: tables.into_boxed_slice(),
    }
}

fn collect_fasta_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let entries = fs::read_dir(dir).expect("read data directory");
    for entry in entries {
        let path = entry.expect("read directory entry").path();
        if path.is_file() {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_ascii_lowercase());
            if matches!(ext.as_deref(), Some("fasta" | "fa" | "fna" | "faa" | "fsa")) {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

fn arg_value(flag: &str) -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == flag {
            return args.next();
        }
    }
    None
}

fn has_flag(flag: &str) -> bool {
    std::env::args().any(|arg| arg == flag)
}
