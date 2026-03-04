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
- Two-Way: critical factorization + period-based shifts with linear-time guarantee.
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

## Bio benchmark datasets (DNA / Protein FASTA)
- Prepare datasets (~0.5GB-1GB each target):
  ```bash
  python3 scripts/download_bio_benchmarks.py \
    --root data/benchmarks \
    --dna-target-bytes 700000000 \
    --protein-target-bytes 700000000
  ```
- DNA benchmark: `cargo run -p tests --bin bench_dna --release -- --data-dir data/benchmarks/dna`
- Protein benchmark: `cargo run -p tests --bin bench_protein --release -- --data-dir data/benchmarks/protein`
- For very large FASTA inputs, `bench_dna` / `bench_protein` auto-disable FM-index to avoid OOM.

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
- Skip two-way: `--skip-two-way`
- Cap rows per table: `--max-rows-per-table <N>`

### Pattern-file driven benchmarking
- Bench binaries now accept:
  - `--patterns-file <path>`: load patterns from a TSV file (`pattern<TAB>description`)
  - `--output-csv <path>`: write machine-readable benchmark rows
- This enables easy manual vs auto-generated pattern runs.

Example manual pattern file:
```tsv
# pattern	description
%STEEL%	Contains STEEL
%a_%_b%	Mixed wildcard
GATTACA%	Prefix pattern
```

Generate auto patterns from current benchmark data:
```bash
python3 scripts/generate_like_patterns.py \
  --data-root data/benchmarks \
  --datasets tpch,tpcds \
  --max-patterns 200 \
  --output results/patterns_autogen.tsv
```

Run the matrix benchmark using any pattern file:
```bash
python3 scripts/run_like_pattern_benchmarks.py \
  --patterns-file scripts/patterns_manual_example.tsv \
  --datasets tpch,tpcds,job,dna,protein \
  --results-dir results
```

Dataset-specific pattern files are supported:
- DNA defaults to `scripts/patterns_dna.tsv`
- Protein defaults to `scripts/patterns_protein.tsv`
- Override with `--dna-patterns-file` / `--protein-patterns-file`

Swap in auto-generated patterns by pointing `--patterns-file` to `results/patterns_autogen.tsv`.
If `--patterns-file` is omitted, the runner defaults to `scripts/patterns_default_sample.tsv`.
To enable interactive charts, install Plotly in your Python env:
`python -m pip install plotly`

Output artifacts:
- `results/raw_results.csv`
- `results/summary_by_algorithm.csv`
- `results/summary_by_length_complexity.csv`
- `results/summary_wins.csv`
- `results/report.md`
- `results/plots/algorithm_latency.html`
- `results/plots/algorithm_win_rate.html`
- `results/plots/complexity_vs_runtime.html`
- `results/plots/best_algo_heatmap.html`
- `results/plots/dashboard.html`
- `results/plots/pattern_detail_viewer.html`
- `results/plots/table_detail_viewer.html`

Post-analysis (matrix + UMAP):
```bash
python3 scripts/analyze_benchmark_matrix.py --results-dir results/<run-dir>
```
Additional artifacts:
- `results/<run-dir>/analysis/matrix_overall.csv`
- `results/<run-dir>/analysis/recommendation_matrix.csv`
- `results/<run-dir>/analysis/umap_clusters_hybrid.html`
- `results/<run-dir>/analysis/umap_by_dataset/*_umap_2d.html`
- `results/<run-dir>/analysis/umap_by_dataset/*_umap_3d.html`
- `results/<run-dir>/analysis/umap_by_dataset/*_cluster_summary.csv`

Default sample patterns:
- `scripts/patterns_default_sample.tsv` (generated from TPC-H + TPC-DS text tokens)


## Further ideas:
- exact naive string search on gpu
- prefix sum based search, (rolling hash, robin carp?) hot spots + exact search "filter + compact"
