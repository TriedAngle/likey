use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use clap::{ArgAction, Parser};
use csv::{ReaderBuilder, StringRecord, Trim};
use storage::{
    dataset::{load_fasta_table, DataSet, Row, Table},
    BumpArena,
};
use tests::bench_shared::{
    available_algorithms, load_patterns_from_file, run_like_benchmarks, BenchOptions, PatternSpec,
};

const BYTES_PER_GB: usize = 1024 * 1024 * 1024;
const DEFAULT_ARENA_GB: usize = 4;
const DEFAULT_DATASET_NAME: &str = "custom";

#[derive(Debug, Parser)]
#[command(about = "Unified LIKE benchmark runner")]
struct Cli {
    #[arg(long = "fasta", value_name = "FILE", action = ArgAction::Append)]
    fasta_files: Vec<PathBuf>,

    #[arg(long = "csv", value_name = "FILE", action = ArgAction::Append)]
    csv_files: Vec<PathBuf>,

    #[arg(long = "tsv", value_name = "FILE", action = ArgAction::Append)]
    tsv_files: Vec<PathBuf>,

    #[arg(long = "csv-delimiter", default_value = ",")]
    csv_delimiter: String,

    #[arg(long = "tsv-delimiter", default_value = "\\t")]
    tsv_delimiter: String,

    #[arg(long = "csv-has-headers", action = ArgAction::SetTrue)]
    csv_has_headers: bool,

    #[arg(long = "tsv-has-headers", action = ArgAction::SetTrue)]
    tsv_has_headers: bool,

    #[arg(long = "column", value_name = "FILE:COL", action = ArgAction::Append)]
    columns: Vec<String>,

    #[arg(
        long = "exclude-column",
        value_name = "FILE:COL",
        action = ArgAction::Append
    )]
    exclude_columns: Vec<String>,

    #[arg(
        long = "exclude-fasta-id",
        value_name = "TEXT",
        action = ArgAction::Append
    )]
    exclude_fasta_id: Vec<String>,

    #[arg(
        long = "exclude-fasta-desc",
        value_name = "TEXT",
        action = ArgAction::Append
    )]
    exclude_fasta_desc: Vec<String>,

    #[arg(
        long = "pattern",
        alias = "patterns-file",
        value_name = "FILE",
        action = ArgAction::Append
    )]
    pattern_files: Vec<PathBuf>,

    #[arg(
        long = "skip",
        value_name = "ALGO[,ALGO...]",
        action = ArgAction::Append
    )]
    skip_algorithms: Vec<String>,

    #[arg(long = "arena-gb", default_value_t = DEFAULT_ARENA_GB)]
    arena_gb: usize,

    #[arg(long = "max-bytes")]
    max_bytes: Option<usize>,

    #[arg(long = "max-rows-per-table")]
    max_rows_per_table: Option<usize>,

    #[arg(long = "max-row-bytes")]
    max_row_bytes: Option<usize>,

    #[arg(long = "output-csv", value_name = "FILE")]
    output_csv: Option<PathBuf>,

    #[arg(long = "dataset-name", default_value = DEFAULT_DATASET_NAME)]
    dataset_name: String,
}

#[derive(Debug, Clone)]
struct DelimitedInput {
    path: PathBuf,
    delimiter: u8,
    has_headers: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ColumnSelector {
    Index(usize),
    Name(String),
}

#[derive(Debug, Default)]
struct ColumnRules {
    includes: Vec<(String, ColumnSelector)>,
    excludes: Vec<(String, ColumnSelector)>,
}

#[derive(Debug, Default)]
struct FastaExclusions {
    id_contains: Vec<String>,
    desc_contains: Vec<String>,
}

fn main() {
    let cli = Cli::parse();

    if cli.fasta_files.is_empty() && cli.csv_files.is_empty() && cli.tsv_files.is_empty() {
        fatal("provide at least one --fasta, --csv, or --tsv input file");
    }

    if cli.pattern_files.is_empty() {
        fatal("provide at least one --pattern <FILE>");
    }

    let csv_delimiter = match parse_delimiter(&cli.csv_delimiter, "--csv-delimiter") {
        Ok(delimiter) => delimiter,
        Err(err) => fatal(&err),
    };
    let tsv_delimiter = match parse_delimiter(&cli.tsv_delimiter, "--tsv-delimiter") {
        Ok(delimiter) => delimiter,
        Err(err) => fatal(&err),
    };

    let column_rules = match parse_column_rules(&cli.columns, &cli.exclude_columns) {
        Ok(rules) => rules,
        Err(err) => fatal(&err),
    };
    let fasta_exclusions = FastaExclusions {
        id_contains: cli.exclude_fasta_id,
        desc_contains: cli.exclude_fasta_desc,
    };

    let patterns = match load_patterns(&cli.pattern_files) {
        Ok(patterns) => patterns,
        Err(err) => fatal(&err),
    };

    let options = match BenchOptions::new(parse_skip_algorithms(&cli.skip_algorithms)) {
        Ok(options) => options,
        Err(err) => fatal(&err),
    };

    let mut delimited_inputs = Vec::with_capacity(cli.csv_files.len() + cli.tsv_files.len());
    delimited_inputs.extend(cli.csv_files.into_iter().map(|path| DelimitedInput {
        path,
        delimiter: csv_delimiter,
        has_headers: cli.csv_has_headers,
    }));
    delimited_inputs.extend(cli.tsv_files.into_iter().map(|path| DelimitedInput {
        path,
        delimiter: tsv_delimiter,
        has_headers: cli.tsv_has_headers,
    }));

    let mut remaining_bytes = cli.max_bytes.unwrap_or(usize::MAX);

    let arena_bytes = match cli.arena_gb.checked_mul(BYTES_PER_GB) {
        Some(bytes) => bytes,
        None => fatal("arena size overflow from --arena-gb"),
    };
    let arena_bytes = match cli.max_bytes {
        Some(cap) => arena_bytes.min(cap),
        None => arena_bytes,
    };

    println!("--- Unified Like Benchmark ---");
    println!(
        "> Available algorithms: {}",
        available_algorithms().join(", ")
    );
    println!("> Allocating {} bytes for Arena...", arena_bytes);
    let arena = BumpArena::new(arena_bytes);

    let database = match load_dataset(
        &arena,
        &cli.fasta_files,
        &delimited_inputs,
        &column_rules,
        &fasta_exclusions,
        cli.max_rows_per_table,
        cli.max_row_bytes,
        &mut remaining_bytes,
    ) {
        Ok(dataset) => dataset,
        Err(err) => fatal(&err),
    };

    if database.tables.is_empty() {
        fatal("no tables were loaded from the provided inputs");
    }

    let mut skipped: Vec<String> = options.skip_algorithms().iter().cloned().collect();
    skipped.sort();
    if !skipped.is_empty() {
        println!("> Skipping algorithms: {}", skipped.join(", "));
    }

    println!("> Loaded {} table(s)", database.tables.len());
    println!("> Arena used: {} bytes", arena.used());

    run_like_benchmarks(
        &database,
        &cli.dataset_name,
        &patterns,
        options,
        cli.output_csv.as_deref(),
    );
}

fn load_dataset<'a>(
    arena: &'a BumpArena,
    fasta_files: &[PathBuf],
    delimited_inputs: &[DelimitedInput],
    column_rules: &ColumnRules,
    fasta_exclusions: &FastaExclusions,
    max_rows_per_table: Option<usize>,
    max_row_bytes: Option<usize>,
    remaining_bytes: &mut usize,
) -> Result<DataSet<'a>, String> {
    let mut tables = Vec::new();

    tables.extend(load_fasta_tables(
        arena,
        fasta_files,
        fasta_exclusions,
        max_rows_per_table,
        max_row_bytes,
        remaining_bytes,
    )?);

    if *remaining_bytes != 0 {
        tables.extend(load_delimited_tables(
            arena,
            delimited_inputs,
            column_rules,
            max_rows_per_table,
            max_row_bytes,
            remaining_bytes,
        )?);
    }

    Ok(DataSet {
        tables: tables.into_boxed_slice(),
    })
}

fn load_fasta_tables<'a>(
    arena: &'a BumpArena,
    files: &[PathBuf],
    exclusions: &FastaExclusions,
    max_rows_per_table: Option<usize>,
    max_row_bytes: Option<usize>,
    remaining_bytes: &mut usize,
) -> Result<Vec<Table<'a>>, String> {
    let mut tables = Vec::with_capacity(files.len());

    for path in files {
        ensure_file(path)?;
        let mut table = load_fasta_table(arena, path)
            .map_err(|err| format!("Failed to load FASTA {}: {err}", path.display()))?;

        let mut filtered_rows = Vec::new();
        for row in table.rows.iter().cloned() {
            if should_exclude_fasta_row(&row, exclusions) {
                continue;
            }

            let mut data = row.data;
            if let Some(cap) = max_row_bytes {
                data = clip_utf8(data, cap);
            }

            if *remaining_bytes != usize::MAX {
                let len = data.len();
                if len > *remaining_bytes {
                    if *remaining_bytes == 0 || !filtered_rows.is_empty() {
                        break;
                    }

                    data = clip_utf8(data, *remaining_bytes);
                    *remaining_bytes = 0;
                } else {
                    *remaining_bytes -= len;
                }
            }

            filtered_rows.push(Row {
                id: row.id,
                desc: row.desc,
                data,
            });

            if let Some(cap) = max_rows_per_table {
                if filtered_rows.len() >= cap {
                    break;
                }
            }

            if *remaining_bytes == 0 {
                break;
            }
        }

        table.rows = filtered_rows.into_boxed_slice();
        tables.push(table);

        if *remaining_bytes == 0 {
            break;
        }
    }

    Ok(tables)
}

fn load_delimited_tables<'a>(
    arena: &'a BumpArena,
    inputs: &[DelimitedInput],
    column_rules: &ColumnRules,
    max_rows_per_table: Option<usize>,
    max_row_bytes: Option<usize>,
    remaining_bytes: &mut usize,
) -> Result<Vec<Table<'a>>, String> {
    let mut tables = Vec::new();

    for input in inputs {
        ensure_file(&input.path)?;

        let mut reader = ReaderBuilder::new()
            .delimiter(input.delimiter)
            .has_headers(input.has_headers)
            .flexible(true)
            .trim(Trim::All)
            .from_path(&input.path)
            .map_err(|err| {
                format!(
                    "Failed to open delimited file {}: {err}",
                    input.path.display()
                )
            })?;

        let mut first_record = None;
        let headers = if input.has_headers {
            Some(
                reader
                    .headers()
                    .map_err(|err| {
                        format!(
                            "Failed to read header row from {}: {err}",
                            input.path.display()
                        )
                    })?
                    .clone(),
            )
        } else {
            None
        };

        let column_count = if let Some(headers) = headers.as_ref() {
            headers.len()
        } else {
            let mut record = StringRecord::new();
            let has_row = reader.read_record(&mut record).map_err(|err| {
                format!(
                    "Failed to read first row from {}: {err}",
                    input.path.display()
                )
            })?;

            if has_row {
                first_record = Some(record);
                first_record.as_ref().map(StringRecord::len).unwrap_or(0)
            } else {
                0
            }
        };

        if column_count == 0 {
            continue;
        }

        let include_selectors = selectors_for_file(&column_rules.includes, &input.path);
        let exclude_selectors = selectors_for_file(&column_rules.excludes, &input.path);

        let mut selected_columns = Vec::new();
        for idx in 0..column_count {
            let header_name = headers.as_ref().and_then(|row| row.get(idx));
            let included = include_selectors.is_empty()
                || include_selectors
                    .iter()
                    .any(|selector| selector_matches(selector, idx, header_name));
            if !included {
                continue;
            }

            let excluded = exclude_selectors
                .iter()
                .any(|selector| selector_matches(selector, idx, header_name));
            if excluded {
                continue;
            }

            selected_columns.push((idx, column_label(header_name, idx)));
        }

        if selected_columns.is_empty() {
            continue;
        }

        let mut rows_by_column: Vec<Vec<Row<'a>>> = vec![Vec::new(); selected_columns.len()];
        let mut accepted_rows = 0usize;

        if let Some(record) = first_record.as_ref() {
            let should_continue = append_record(
                arena,
                record,
                &selected_columns,
                &mut rows_by_column,
                &mut accepted_rows,
                max_rows_per_table,
                max_row_bytes,
                remaining_bytes,
            );

            if !should_continue {
                let file_name = filename_from_path(&input.path)?;
                for ((_, label), rows) in
                    selected_columns.into_iter().zip(rows_by_column.into_iter())
                {
                    tables.push(Table {
                        name: format!("{}.{}", file_name, label),
                        rows: rows.into_boxed_slice(),
                    });
                }

                if *remaining_bytes == 0 {
                    break;
                }
                continue;
            }
        }

        for record in reader.records() {
            let record = record
                .map_err(|err| format!("CSV parse error in {}: {err}", input.path.display()))?;

            let should_continue = append_record(
                arena,
                &record,
                &selected_columns,
                &mut rows_by_column,
                &mut accepted_rows,
                max_rows_per_table,
                max_row_bytes,
                remaining_bytes,
            );

            if !should_continue {
                break;
            }
        }

        let file_name = filename_from_path(&input.path)?;
        for ((_, label), rows) in selected_columns.into_iter().zip(rows_by_column.into_iter()) {
            tables.push(Table {
                name: format!("{}.{}", file_name, label),
                rows: rows.into_boxed_slice(),
            });
        }

        if *remaining_bytes == 0 {
            break;
        }
    }

    Ok(tables)
}

fn append_record<'a>(
    arena: &'a BumpArena,
    record: &StringRecord,
    selected_columns: &[(usize, String)],
    rows_by_column: &mut [Vec<Row<'a>>],
    accepted_rows: &mut usize,
    max_rows_per_table: Option<usize>,
    max_row_bytes: Option<usize>,
    remaining_bytes: &mut usize,
) -> bool {
    if let Some(cap) = max_rows_per_table {
        if *accepted_rows >= cap {
            return false;
        }
    }

    let mut values = Vec::with_capacity(selected_columns.len());
    let mut row_bytes = 0usize;
    for (idx, _) in selected_columns {
        let value = record.get(*idx).unwrap_or("");
        let clipped = if let Some(cap) = max_row_bytes {
            clip_utf8(value, cap)
        } else {
            value
        };

        row_bytes += clipped.len();
        values.push(clipped);
    }

    if *remaining_bytes != usize::MAX {
        if row_bytes > *remaining_bytes {
            return false;
        }
        *remaining_bytes -= row_bytes;
    }

    for (rows, value) in rows_by_column.iter_mut().zip(values.into_iter()) {
        rows.push(Row {
            id: "",
            desc: "",
            data: arena.alloc_str(value),
        });
    }

    *accepted_rows += 1;
    *remaining_bytes != 0
}

fn parse_delimiter(raw: &str, flag: &str) -> Result<u8, String> {
    match raw {
        "\\t" => Ok(b'\t'),
        "\\n" => Ok(b'\n'),
        "\\r" => Ok(b'\r'),
        _ => {
            let bytes = raw.as_bytes();
            if bytes.len() == 1 {
                Ok(bytes[0])
            } else {
                Err(format!(
                    "{flag} expects a single byte delimiter or one of \\t, \\n, \\r"
                ))
            }
        }
    }
}

fn parse_skip_algorithms(values: &[String]) -> HashSet<String> {
    let mut skip = HashSet::new();
    for value in values {
        for token in value.split(',') {
            let name = token.trim();
            if !name.is_empty() {
                skip.insert(name.to_string());
            }
        }
    }
    skip
}

fn parse_column_rules(includes: &[String], excludes: &[String]) -> Result<ColumnRules, String> {
    let mut rules = ColumnRules::default();

    for spec in includes {
        rules.includes.push(parse_column_spec(spec)?);
    }
    for spec in excludes {
        rules.excludes.push(parse_column_spec(spec)?);
    }

    Ok(rules)
}

fn parse_column_spec(spec: &str) -> Result<(String, ColumnSelector), String> {
    let (file_match, selector_raw) = spec.rsplit_once(':').ok_or_else(|| {
        format!(
            "Invalid column selector '{spec}'. Expected format FILE:COL (e.g. part.csv:4 or item.csv:i_color)"
        )
    })?;

    let file_match = file_match.trim();
    let selector_raw = selector_raw.trim();

    if file_match.is_empty() || selector_raw.is_empty() {
        return Err(format!(
            "Invalid column selector '{spec}'. FILE and COL must be non-empty"
        ));
    }

    let selector = match selector_raw.parse::<usize>() {
        Ok(idx) => ColumnSelector::Index(idx),
        Err(_) => ColumnSelector::Name(selector_raw.to_string()),
    };

    Ok((file_match.to_string(), selector))
}

fn selectors_for_file(rules: &[(String, ColumnSelector)], path: &Path) -> HashSet<ColumnSelector> {
    let mut set = HashSet::new();
    for (file_match, selector) in rules {
        if path_matches(file_match, path) {
            set.insert(selector.clone());
        }
    }
    set
}

fn path_matches(file_match: &str, path: &Path) -> bool {
    if file_match == "*" {
        return true;
    }

    let path_str = path.to_string_lossy();
    if path_str == file_match || path_str.ends_with(file_match) {
        return true;
    }

    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name == file_match)
        .unwrap_or(false)
}

fn selector_matches(selector: &ColumnSelector, idx: usize, header_name: Option<&str>) -> bool {
    match selector {
        ColumnSelector::Index(target) => *target == idx,
        ColumnSelector::Name(target) => header_name.map(|name| name == target).unwrap_or(false),
    }
}

fn column_label(header_name: Option<&str>, idx: usize) -> String {
    let name = header_name.unwrap_or("").trim();
    if name.is_empty() {
        format!("col{idx}")
    } else {
        name.to_string()
    }
}

fn should_exclude_fasta_row(row: &Row<'_>, exclusions: &FastaExclusions) -> bool {
    exclusions
        .id_contains
        .iter()
        .any(|needle| !needle.is_empty() && row.id.contains(needle))
        || exclusions
            .desc_contains
            .iter()
            .any(|needle| !needle.is_empty() && row.desc.contains(needle))
}

fn clip_utf8(input: &str, max_bytes: usize) -> &str {
    if input.len() <= max_bytes {
        return input;
    }

    let mut end = max_bytes;
    while end > 0 && !input.is_char_boundary(end) {
        end -= 1;
    }
    &input[..end]
}

fn load_patterns(paths: &[PathBuf]) -> Result<Vec<PatternSpec>, String> {
    let mut patterns = Vec::new();
    for path in paths {
        let loaded = load_patterns_from_file(path)?;
        patterns.extend(loaded);
    }

    if patterns.is_empty() {
        return Err("No patterns were loaded from --pattern files".to_string());
    }

    Ok(patterns)
}

fn ensure_file(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Err(format!("Input file does not exist: {}", path.display()));
    }
    if !path.is_file() {
        return Err(format!("Input path is not a file: {}", path.display()));
    }
    Ok(())
}

fn filename_from_path(path: &Path) -> Result<String, String> {
    path.file_name()
        .ok_or_else(|| format!("Missing filename for path {}", path.display()))
        .map(|name| name.to_string_lossy().to_string())
}

fn fatal(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(2);
}
