# Likey
Rust string-search algorithms and SQL-like `%` / `_` pattern matching. 
It focuses on fast matching, clear comparisons, and a simple mmap-backed data model.
The intention is to prototype and optimize like queries easily.

## Structure
- `algos`: string search algorithms with a shared `StringSearch` trait.
- `like`: pattern compiler + matcher for SQL-like `LIKE` semantics.
- `storage`: mmap-backed arena and dataset loaders (text + FASTA).
- `engine`: executes compiled patterns against datasets.
- `tests`: benchmarks and integration binaries.

## Architecture (short)
```
files -> storage::BumpArena -> storage::dataset::DataSet
                       |                    |
pattern -> like::compile_pattern ----------> engine::execute
                       |                    |
                    algos::<impl>       matches
```
- Data is loaded once into a big mmap arena and treated as immutable.
- Patterns are compiled once and reused across rows/tables.
- The engine is a thin loop over rows; no query parser or planner.

## Algorithms (core idea)
- Naive: checks each position in the text for a full match.
- NaiveVectorized: SIMD-first-byte scan for NEON.
- LUT-short: SSSE3/NEON LUT filter for very short patterns (<= 8 bytes).
- KMP: avoids re-checking by using a prefix table.
- Boyer–Moore: skips ahead using bad-character/good-suffix heuristics.
- StdSearch: Rust `str::find` as a baseline.
- FM-index: global index over the whole dataset, supports `%/_` via recheck.
- Trigram: inverted index for 3-byte substrings, `%/_` via recheck.

## Testing
- Unit tests: `cargo test -p storage` and `cargo test -p engine`
- Benchmark + correctness report: `cargo run -p tests --bin bench_like --release`

## Benchmark datasets (TPC-H / TPC-DS / JOB)
- Download/generate data: `python3 scripts/download_benchmarks.py`
- Python setup (DuckDB required):
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `python -m pip install --upgrade pip`
  - `python -m pip install duckdb`
- TPC-H: `cargo run -p tests --bin bench_tpch --release -- --data-dir data/benchmarks/tpch`
- TPC-DS: `cargo run -p tests --bin bench_tpcds --release -- --data-dir data/benchmarks/tpcds`
- JOB (IMDB): `cargo run -p tests --bin bench_job --release -- --data-dir data/benchmarks/job`

### Vectorization flags
- x86_64 (SSSE3 + SSE2): `RUSTFLAGS="-C target-feature=+ssse3" cargo run -p tests --bin bench_tpch --release -- --data-dir data/benchmarks/tpch`
- aarch64 (NEON): `RUSTFLAGS="-C target-feature=+neon" cargo run -p tests --bin bench_tpch --release -- --data-dir data/benchmarks/tpch`

### SIMD build notes
- LUT-short requires SSSE3 (x86_64) or NEON (aarch64).
- Example (x86_64):
  ```
  RUSTFLAGS="-C target-feature=+ssse3" cargo run -p tests --bin bench_like --release
  ```

### Benchmark flags
- Skip FM: `--skip-fm`
- Skip trigram: `--skip-trigram`
- Skip fftstr0: `--skip-fftstr0`
- Skip fftstr1: `--skip-fftstr1`


## Further ideas:
- exact naive string search on gpu
- prefix sum based search, (rolling hash, robin carp?) hot spots + exact search "filter + compact"
