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
    ("%Brand #%", "Item brand"),
    ("%Polished%", "Item description polished"),
    ("%blue%", "Item color"),
    ("%Smith%", "Customer last name"),
    ("%Oak%", "Street name"),
    ("%Monday%", "Day name"),
    ("%Premium%", "Call center class"),
];

fn main() {
    let data_dir = arg_value("--data-dir").unwrap_or_else(|| "data/benchmarks/tpcds".to_string());
    let arena_gb = arg_value("--arena-gb")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_ARENA_GB);
    let max_bytes = arg_value("--max-bytes").and_then(|v| v.parse::<usize>().ok());

    let mut arena_size = arena_gb * 1024 * 1024 * 1024;
    if let Some(max_bytes) = max_bytes {
        arena_size = arena_size.min(max_bytes);
    }
    let arena = BumpArena::new(arena_size);

    println!("--- TPC-DS Like Benchmark ---");
    println!("> Using data dir: {}", data_dir);
    println!("> Allocating {} bytes for Arena...", arena_size);
    if let Some(max_bytes) = max_bytes {
        println!("> Max bytes cap: {}", max_bytes);
    }

    let dataset = load_tpcds_dataset(&arena, Path::new(&data_dir), max_bytes);

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

fn load_tpcds_dataset<'a>(
    arena: &'a BumpArena,
    data_dir: &Path,
    max_bytes: Option<usize>,
) -> DataSet<'a> {
    let item = find_existing_file(data_dir, "item").expect("item file not found");
    let customer = find_existing_file(data_dir, "customer").expect("customer file not found");
    let customer_address =
        find_existing_file(data_dir, "customer_address").expect("customer_address file not found");
    let date_dim = find_existing_file(data_dir, "date_dim").expect("date_dim file not found");
    let call_center =
        find_existing_file(data_dir, "call_center").expect("call_center file not found");

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
            &item,
            &options,
            &[
                ColumnSpec {
                    name: "i_item_desc".to_string(),
                    index: 4,
                },
                ColumnSpec {
                    name: "i_color".to_string(),
                    index: 17,
                },
            ],
            &mut limit,
        )
        .expect("load item columns"),
    );
    tables.extend(
        load_delimited_columns(
            arena,
            &customer,
            &options,
            &[ColumnSpec {
                name: "c_last_name".to_string(),
                index: 9,
            }],
            &mut limit,
        )
        .expect("load customer columns"),
    );
    tables.extend(
        load_delimited_columns(
            arena,
            &customer_address,
            &options,
            &[ColumnSpec {
                name: "ca_street_name".to_string(),
                index: 3,
            }],
            &mut limit,
        )
        .expect("load customer_address columns"),
    );
    tables.extend(
        load_delimited_columns(
            arena,
            &date_dim,
            &options,
            &[ColumnSpec {
                name: "d_day_name".to_string(),
                index: 14,
            }],
            &mut limit,
        )
        .expect("load date_dim columns"),
    );
    tables.extend(
        load_delimited_columns(
            arena,
            &call_center,
            &options,
            &[ColumnSpec {
                name: "cc_class".to_string(),
                index: 7,
            }],
            &mut limit,
        )
        .expect("load call_center columns"),
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
    }

    let raw = data_dir.join(base);
    if raw.exists() {
        return Some(raw);
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
