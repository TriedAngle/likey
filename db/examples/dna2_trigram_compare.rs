use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use db::{
    Column, DbBuilder, Dna2Column, Dna2FixedTrigramIndex, Dna2TableBuilder, RowId, TrigramIndex,
};

#[derive(Debug, Clone)]
struct Config {
    fasta: Option<PathBuf>,
    target_bases: usize,
    rows: usize,
    row_len: usize,
    probes: usize,
    literal_len: usize,
    seed: u64,
}

impl Config {
    fn from_args() -> Self {
        let mut config = Self {
            fasta: None,
            target_bases: 100_000_000,
            rows: 100_000,
            row_len: 150,
            probes: 20_000,
            literal_len: 12,
            seed: 7,
        };

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            let Some(value) = args.next() else {
                usage_and_exit();
            };
            match arg.as_str() {
                "--fasta" => config.fasta = Some(PathBuf::from(value)),
                "--target-bases" => config.target_bases = parse_arg(&arg, &value),
                "--rows" => config.rows = parse_arg(&arg, &value),
                "--row-len" => config.row_len = parse_arg(&arg, &value),
                "--probes" => config.probes = parse_arg(&arg, &value),
                "--literal-len" => config.literal_len = parse_arg(&arg, &value),
                "--seed" => config.seed = parse_arg(&arg, &value),
                "--help" | "-h" => usage_and_exit(),
                _ => usage_and_exit(),
            }
        }

        if config.literal_len < 3 {
            eprintln!("--literal-len must be at least 3");
            std::process::exit(2);
        }
        if config.fasta.is_none() && config.literal_len > config.row_len {
            eprintln!("for synthetic data, --literal-len must be in 3..=--row-len");
            std::process::exit(2);
        }

        config
    }
}

fn main() {
    let config = Config::from_args();
    if let Some(path) = config.fasta.as_ref() {
        println!(
            "fasta={} target_bases={} probes={} literal_len={} seed={}",
            path.display(),
            config.target_bases,
            config.probes,
            config.literal_len,
            config.seed
        );
    } else {
        println!(
            "rows={} row_len={} probes={} literal_len={} seed={}",
            config.rows, config.row_len, config.probes, config.literal_len, config.seed
        );
    }

    let load_start = Instant::now();
    let (db, table_id) = build_db(&config);
    let load_elapsed = load_start.elapsed();
    let table = db.dna2_table(table_id).unwrap();
    let column = table.sequence();
    println!(
        "loaded rows={} bases={} payload_bytes={} load={}",
        column.row_count(),
        column.total_bases(),
        column.packed_payload().len(),
        fmt_duration(load_elapsed),
    );

    let literals = sample_literals(&column, &config);

    let fixed_build_start = Instant::now();
    let fixed = Dna2FixedTrigramIndex::build(&column);
    let fixed_build = fixed_build_start.elapsed();

    let sparse_build_start = Instant::now();
    let sparse = TrigramIndex::build(&column);
    let sparse_build = sparse_build_start.elapsed();

    let fixed_probe = time_probes(&fixed, &literals);
    let sparse_probe = time_probes(&sparse, &literals);

    println!("domain,build,probe,total_candidates");
    println!(
        "fixed64,{},{},{}",
        fmt_duration(fixed_build),
        fmt_duration(fixed_probe.elapsed),
        fixed_probe.total_candidates
    );
    println!(
        "sparse_hash,{},{},{}",
        fmt_duration(sparse_build),
        fmt_duration(sparse_probe.elapsed),
        sparse_probe.total_candidates
    );
    println!(
        "speedup build={:.2}x probe={:.2}x",
        ratio(sparse_build, fixed_build),
        ratio(sparse_probe.elapsed, fixed_probe.elapsed),
    );
}

fn build_db(config: &Config) -> (db::Db, db::TableId) {
    if let Some(path) = config.fasta.as_ref() {
        return build_db_from_fasta(path, config.target_bases);
    }

    let mut rng = Rng::new(config.seed);
    let mut table =
        Dna2TableBuilder::with_capacity("synthetic-dna", config.rows, config.rows * config.row_len);
    let mut row = vec![0u8; config.row_len];

    for _ in 0..config.rows {
        for b in &mut row {
            *b = b"ACGT"[rng.next_usize(4)];
        }
        let row = std::str::from_utf8(&row).unwrap();
        table.push_str(row).unwrap();
    }

    let mut builder = DbBuilder::new();
    let table_id = builder.add_dna2_table(table).unwrap();
    (builder.freeze(), table_id)
}

fn build_db_from_fasta(path: &PathBuf, target_bases: usize) -> (db::Db, db::TableId) {
    let file = File::open(path).unwrap_or_else(|err| panic!("open {}: {err}", path.display()));
    let mut table = Dna2TableBuilder::new("fasta-dna");
    let mut run = Vec::<u8>::new();
    let mut loaded_bases = 0usize;

    for line in BufReader::new(file).lines() {
        let line = line.unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        if line.starts_with('>') {
            flush_run(&mut table, &mut run, target_bases, &mut loaded_bases);
            if loaded_bases >= target_bases {
                break;
            }
            continue;
        }

        for b in line.bytes() {
            if loaded_bases + run.len() >= target_bases {
                break;
            }
            match b.to_ascii_uppercase() {
                b'A' | b'C' | b'G' | b'T' => run.push(b.to_ascii_uppercase()),
                _ => flush_run(&mut table, &mut run, target_bases, &mut loaded_bases),
            }
        }

        if loaded_bases >= target_bases {
            break;
        }
    }
    flush_run(&mut table, &mut run, target_bases, &mut loaded_bases);

    let mut builder = DbBuilder::new();
    let table_id = builder.add_dna2_table(table).unwrap();
    (builder.freeze(), table_id)
}

fn flush_run(
    table: &mut Dna2TableBuilder,
    run: &mut Vec<u8>,
    target_bases: usize,
    loaded_bases: &mut usize,
) {
    if run.len() < 3 {
        run.clear();
        return;
    }
    let remaining = target_bases.saturating_sub(*loaded_bases);
    run.truncate(remaining);
    if run.len() >= 3 {
        table.push_ascii(run).unwrap();
        *loaded_bases += run.len();
    }
    run.clear();
}

fn sample_literals(column: &Dna2Column<'_>, config: &Config) -> Vec<Vec<u8>> {
    let mut rng = Rng::new(config.seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut literals = Vec::with_capacity(config.probes);

    for _ in 0..config.probes {
        let mut row_id = rng.next_usize(column.row_count() as usize) as RowId;
        let mut row = column.row_view(row_id);
        while row.logical_len() < config.literal_len as u32 {
            row_id = rng.next_usize(column.row_count() as usize) as RowId;
            row = column.row_view(row_id);
        }
        let start = rng.next_usize(row.logical_len() as usize - config.literal_len + 1) as u32;
        let mut literal = Vec::with_capacity(config.literal_len);
        for offset in 0..config.literal_len as u32 {
            literal.push(row.base_code_at(start + offset));
        }
        literals.push(literal);
    }

    literals
}

#[derive(Debug, Clone, Copy)]
struct ProbeTiming {
    elapsed: Duration,
    total_candidates: usize,
}

trait SearchLiteral {
    fn search_literal(&self, literal: &[u8]) -> Option<Vec<RowId>>;
}

impl SearchLiteral for Dna2FixedTrigramIndex {
    fn search_literal(&self, literal: &[u8]) -> Option<Vec<RowId>> {
        Dna2FixedTrigramIndex::search_literal(self, literal)
    }
}

impl<'db> SearchLiteral for TrigramIndex<Dna2Column<'db>> {
    fn search_literal(&self, literal: &[u8]) -> Option<Vec<RowId>> {
        TrigramIndex::search_literal(self, literal)
    }
}

fn time_probes<I>(index: &I, literals: &[Vec<u8>]) -> ProbeTiming
where
    I: SearchLiteral,
{
    let start = Instant::now();
    let mut total_candidates = 0usize;
    for literal in literals {
        if let Some(rows) = index.search_literal(literal) {
            total_candidates += rows.len();
        }
    }
    ProbeTiming {
        elapsed: start.elapsed(),
        total_candidates,
    }
}

#[derive(Debug, Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        ((self.next() >> 32) % upper as u64) as usize
    }
}

fn parse_arg<T>(name: &str, value: &str) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match value.parse() {
        Ok(value) => value,
        Err(err) => {
            eprintln!("invalid {name} value {value:?}: {err}");
            std::process::exit(2);
        }
    }
}

fn fmt_duration(duration: Duration) -> String {
    format!("{:.3}ms", duration.as_secs_f64() * 1_000.0)
}

fn ratio(a: Duration, b: Duration) -> f64 {
    let b = b.as_secs_f64();
    if b == 0.0 { 0.0 } else { a.as_secs_f64() / b }
}

fn usage_and_exit() -> ! {
    eprintln!(
        "usage: cargo run -p db --release --example dna2_trigram_compare -- \
         [--fasta PATH] [--target-bases N] [--rows N] [--row-len N] \
         [--probes N] [--literal-len N] [--seed N]"
    );
    std::process::exit(2);
}
