use std::path::{Path, PathBuf};

use storage::{
    dataset::DataSet,
    delimited::{load_delimited_columns, ByteLimit, ColumnSpec, DelimitedOptions},
    BumpArena,
};

mod bench_shared;
use bench_shared::{run_like_benchmarks, BenchOptions};

const DEFAULT_ARENA_GB: usize = 2;

const PATTERNS: &[(&str, &str)] = &[
    ("%STEEL%", "Part type contains STEEL"),
    ("%BOX%", "Container box"),
    ("%almond%", "Part name keyword"),
    ("%special%", "Order comment special"),
    ("%requests%", "Order comment requests"),
    ("%DELIVER IN PERSON%", "Shipping instruction"),
    ("%BUILDING%", "Market segment"),
];

fn main() {
    let data_dir = arg_value("--data-dir").unwrap_or_else(|| "data/benchmarks/tpch".to_string());
    let arena_gb = arg_value("--arena-gb")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_ARENA_GB);
    let max_bytes = arg_value("--max-bytes").and_then(|v| v.parse::<usize>().ok());

    let mut arena_size = arena_gb * 1024 * 1024 * 1024;
    if let Some(max_bytes) = max_bytes {
        arena_size = arena_size.min(max_bytes);
    }
    let arena = BumpArena::new(arena_size);

    println!("--- TPC-H Like Benchmark ---");
    println!("> Using data dir: {}", data_dir);
    println!("> Allocating {} bytes for Arena...", arena_size);
    if let Some(max_bytes) = max_bytes {
        println!("> Max bytes cap: {}", max_bytes);
    }

    let dataset = load_tpch_dataset(&arena, Path::new(&data_dir), max_bytes);

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

fn load_tpch_dataset<'a>(
    arena: &'a BumpArena,
    data_dir: &Path,
    max_bytes: Option<usize>,
) -> DataSet<'a> {
    let part = find_existing_file(data_dir, "part").expect("part file not found");
    let orders = find_existing_file(data_dir, "orders").expect("orders file not found");
    let lineitem = find_existing_file(data_dir, "lineitem").expect("lineitem file not found");
    let customer = find_existing_file(data_dir, "customer").expect("customer file not found");

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
            &part,
            &options,
            &[
                ColumnSpec {
                    name: "p_name".to_string(),
                    index: 1,
                },
                ColumnSpec {
                    name: "p_type".to_string(),
                    index: 4,
                },
                ColumnSpec {
                    name: "p_container".to_string(),
                    index: 6,
                },
            ],
            &mut limit,
        )
        .expect("load part columns"),
    );
    tables.extend(
        load_delimited_columns(
            arena,
            &orders,
            &options,
            &[ColumnSpec {
                name: "o_comment".to_string(),
                index: 8,
            }],
            &mut limit,
        )
        .expect("load orders columns"),
    );
    tables.extend(
        load_delimited_columns(
            arena,
            &lineitem,
            &options,
            &[ColumnSpec {
                name: "l_shipinstruct".to_string(),
                index: 13,
            }],
            &mut limit,
        )
        .expect("load lineitem columns"),
    );
    tables.extend(
        load_delimited_columns(
            arena,
            &customer,
            &options,
            &[ColumnSpec {
                name: "c_mktsegment".to_string(),
                index: 6,
            }],
            &mut limit,
        )
        .expect("load customer columns"),
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
