use std::path::{Path, PathBuf};

use storage::{
    dataset::DataSet,
    delimited::{load_delimited_columns, ByteLimit, ColumnSpec, DelimitedOptions},
    BumpArena,
};

mod bench_shared;
use bench_shared::{run_like_benchmarks, BenchOptions};

const DEFAULT_ARENA_GB: usize = 4;

const PATTERNS: &[(&str, &str)] = &[
    ("%Terminator%", "Movie title"),
    ("%Star%", "Movie title"),
    ("%Smith%", "Person name"),
    ("%uncredited%", "Cast note"),
    ("%love%", "Plot info"),
];

fn main() {
    let data_dir = arg_value("--data-dir").unwrap_or_else(|| "data/benchmarks/job".to_string());
    let arena_gb = arg_value("--arena-gb")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_ARENA_GB);
    let max_bytes = arg_value("--max-bytes").and_then(|v| v.parse::<usize>().ok());

    let mut arena_size = arena_gb * 1024 * 1024 * 1024;
    if let Some(max_bytes) = max_bytes {
        arena_size = arena_size.min(max_bytes);
    }
    let arena = BumpArena::new(arena_size);

    println!("--- JOB Like Benchmark ---");
    println!("> Using data dir: {}", data_dir);
    println!("> Allocating {} bytes for Arena...", arena_size);
    if let Some(max_bytes) = max_bytes {
        println!("> Max bytes cap: {}", max_bytes);
    }

    let dataset = load_job_dataset(&arena, Path::new(&data_dir), max_bytes);

    let options = BenchOptions {
        skip_naive_scalar: has_flag("--skip-naive-scalar"),
        skip_naive_vector: has_flag("--skip-naive-vector"),
        skip_kmp: has_flag("--skip-kmp"),
        skip_bm: has_flag("--skip-bm"),
        skip_std: has_flag("--skip-std"),
        skip_lut_short: has_flag("--skip-lut-short"),
        skip_fftstr0: has_flag("--skip-fftstr0"),
        skip_fftstr1: has_flag("--skip-fftstr1"),
        skip_fm: has_flag("--skip-fm"),
        skip_trigram: has_flag("--skip-trigram"),
    };

    run_like_benchmarks(&dataset, PATTERNS, options);
}

fn load_job_dataset<'a>(
    arena: &'a BumpArena,
    data_dir: &Path,
    max_bytes: Option<usize>,
) -> DataSet<'a> {
    let title = find_existing_file(data_dir, "title").expect("title file not found");
    let name = find_existing_file(data_dir, "name").expect("name file not found");
    let movie_info = find_existing_file(data_dir, "movie_info").expect("movie_info file not found");
    let keyword = find_existing_file(data_dir, "keyword").expect("keyword file not found");
    let cast_info = find_existing_file(data_dir, "cast_info").expect("cast_info file not found");

    let options = DelimitedOptions {
        delimiter: b'|',
        has_headers: false,
        trim_fields: true,
    };

    let mut limit = max_bytes.map(ByteLimit::new);

    let mut tables = Vec::new();
    tables.extend(
        load_delimited_columns(
            arena,
            &title,
            &options,
            &[ColumnSpec {
                name: "title".to_string(),
                index: 1,
            }],
            &mut limit,
        )
        .expect("load title columns"),
    );
    tables.extend(
        load_delimited_columns(
            arena,
            &name,
            &options,
            &[ColumnSpec {
                name: "name".to_string(),
                index: 1,
            }],
            &mut limit,
        )
        .expect("load name columns"),
    );
    tables.extend(
        load_delimited_columns(
            arena,
            &movie_info,
            &options,
            &[ColumnSpec {
                name: "info".to_string(),
                index: 3,
            }],
            &mut limit,
        )
        .expect("load movie_info columns"),
    );
    tables.extend(
        load_delimited_columns(
            arena,
            &keyword,
            &options,
            &[ColumnSpec {
                name: "keyword".to_string(),
                index: 1,
            }],
            &mut limit,
        )
        .expect("load keyword columns"),
    );
    tables.extend(
        load_delimited_columns(
            arena,
            &cast_info,
            &options,
            &[ColumnSpec {
                name: "note".to_string(),
                index: 4,
            }],
            &mut limit,
        )
        .expect("load cast_info columns"),
    );

    DataSet {
        tables: tables.into_boxed_slice(),
    }
}

fn find_existing_file(data_dir: &Path, base: &str) -> Option<PathBuf> {
    let candidates = ["csv", "tbl", "dat", "tsv"];
    for ext in candidates {
        let candidate = data_dir.join(format!("{}.{}", base, ext));
        if candidate.exists() {
            return Some(candidate);
        }
        let nested = data_dir.join("imdb").join(format!("{}.{}", base, ext));
        if nested.exists() {
            return Some(nested);
        }
    }

    let raw = data_dir.join(base);
    if raw.exists() {
        return Some(raw);
    }
    let nested_raw = data_dir.join("imdb").join(base);
    if nested_raw.exists() {
        return Some(nested_raw);
    }

    None
}

fn has_flag(flag: &str) -> bool {
    std::env::args().any(|arg| arg == flag)
}

fn arg_value(flag: &str) -> Option<String> {
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if arg == flag {
            return args.next();
        }
    }
    None
}
