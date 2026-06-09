# Benchmarks

## Run Everything

Run every benchmark in this file with checkpointing:

```bash
python3 scripts/run_all_benchmarks.py \
  --result-root benchmark_results \
  --iterations 3
```

The super runner delegates each case to `scripts/run_bench.py` and writes timestamped outputs under `benchmark_results/<suite>/<mode>/<case>_<timestamp>/`. Progress is tracked in `benchmark_results/checkpoint.csv`; rows already marked `done=true` are skipped on the next invocation. Use `--force` to rerun completed rows, `--dry-run` to print commands without executing them, and `--list` to show the selected benchmark IDs.

Useful filters:

```bash
python3 scripts/run_all_benchmarks.py --only-suite dna --dry-run
python3 scripts/run_all_benchmarks.py --only-mode index-comparison --dry-run
python3 scripts/run_all_benchmarks.py --only-case name_name --dry-run
python3 scripts/run_all_benchmarks.py --only-benchmark job/index-memmem/name_name --dry-run
```

The FFTSTR artificial data generator is run automatically when an FFTSTR benchmark is selected. Use `--skip-fftstr-generate` if the generated CSVs already exist and should not be refreshed. The DNA sparse-N data generator is run automatically when `dna/n-handling/n_sparse` is selected; use `--skip-dna-n-generate` to reuse existing generated files.

## DNA Benchmark

Run the exact-vs-underscore-heavy full-scan algorithm comparison on GENCODE human transcripts with three measured iterations per combination:

```bash
cargo run -p runner --bin runner --release -- \
  --data-csv benchmarks/dna/data_gencode_dna_utf8_dna2.csv \
  --algorithms-csv benchmarks/dna/exact-vs-underscore/algorithms.csv \
  --generic-matcher static \
  --patterns-csv benchmarks/dna/exact-vs-underscore/patterns.csv \
  --indexes-csv benchmarks/dna/exact-vs-underscore/algorithm-indexes.csv \
  --iterations 3 \
  --max-row-bytes 50MB \
  --max-total-bytes 100MB \
  --output-csv results/gencode_algos_raw.csv \
  --summary-csv results/gencode_algos_summary.csv
```

Run the matching exact-vs-underscore index consistency campaign with representative UTF-8 and DNA2 algorithms across all configured indexes:

```bash
cargo run -p runner --bin runner --release -- \
  --data-csv benchmarks/dna/data_gencode_dna_utf8_dna2.csv \
  --algorithms-csv benchmarks/dna/exact-vs-underscore/index-algorithms.csv \
  --generic-matcher static \
  --patterns-csv benchmarks/dna/exact-vs-underscore/patterns.csv \
  --indexes-csv benchmarks/dna/exact-vs-underscore/indexes.csv \
  --iterations 3 \
  --max-row-bytes 50MB \
  --max-total-bytes 100MB \
  --output-csv results/gencode_exact_indexes_raw.csv \
  --summary-csv results/gencode_exact_indexes_summary.csv
```

The checkpointed super-runner splits these into `dna/exact-vs-underscore-algorithms/gencode` and `dna/exact-vs-underscore-indexes/gencode` to avoid the full algorithm × index cross product.

Run the matcher-engine comparison with static, adaptive, and recursive matchers in one invocation. This reuses loaded data and requested indexes across matcher engines:

```bash
cargo run -p runner --bin runner --release -- \
  --data-csv benchmarks/dna/data_gencode_dna_utf8_dna2.csv \
  --algorithms-csv benchmarks/dna/matcher-comparison/algorithms.csv \
  --generic-matcher static,adaptive,recursive \
  --patterns-csv benchmarks/dna/matcher-comparison/patterns.csv \
  --indexes-csv benchmarks/dna/matcher-comparison/indexes.csv \
  --iterations 3 \
  --max-row-bytes 50MB \
  --max-total-bytes 100MB \
  --output-csv results/gencode_matchers_raw.csv \
  --summary-csv results/gencode_matchers_summary.csv
```

Warmups are controlled separately with `--warmups`; warmup runs are not written to the output CSV.

Run the index comparison on prefix/equality-friendly DNA patterns across all configured indexes, including `prefix-btree`:

```bash
cargo run -p runner --bin runner --release -- \
  --data-csv benchmarks/dna/data_gencode_dna_utf8_dna2.csv \
  --algorithms-csv benchmarks/dna/index-comparison/algorithms.csv \
  --generic-matcher static \
  --patterns-csv benchmarks/dna/index-comparison/patterns.csv \
  --indexes-csv benchmarks/dna/index-comparison/indexes.csv \
  --iterations 3 \
  --max-row-bytes 50MB \
  --max-total-bytes 100MB \
  --output-csv results/gencode_indexes_raw.csv \
  --summary-csv results/gencode_indexes_summary.csv
```

Run the deterministic sparse-N DNA benchmark comparing UTF-8 `StdSearch`/`NaiveScalar`/`NaiveAvx2`/`TwoWay` with packed DNA2 `Dna2PackedScalar`/`Dna2PackedAvx2`/`Dna2TwoWay`, both with and without the qgram index:

```bash
python3 scripts/generate_dna_n_benchmark.py

cargo run -p runner --bin runner --release -- \
  --data-csv benchmarks/dna/data_n_sparse_utf8_dna2.csv \
  --algorithms-csv benchmarks/dna/n-handling/algorithms.csv \
  --generic-matcher static \
  --patterns-csv benchmarks/dna/n-handling/patterns.csv \
  --indexes-csv benchmarks/dna/n-handling/indexes.csv \
  --iterations 3 \
  --max-row-bytes 50MB \
  --max-total-bytes 100MB \
  --output-csv results/dna_n_handling_raw.csv \
  --summary-csv results/dna_n_handling_summary.csv
```

The generated FASTA has 2048 entries of length 180, with 512 rows containing `N` so every `N` qgram posting stays below the default broad-posting threshold. The table includes A/C/G/T-only rows, an all-`N` row, and alternating/interleaved `N` rows that stress DNA2 N-range checks. Most LIKE patterns contain `N`; several are at least 15 fixed bytes so `qgram` can use its q=15 postings and avoid the full DNA2 slow path for N-aware verification.

The same case is included in the checkpointed super-runner as `dna/n-handling/n_sparse`:

```bash
python3 scripts/run_all_benchmarks.py \
  --only-benchmark dna/n-handling/n_sparse \
  --result-root benchmark_results \
  --iterations 3
```

## Quotes Benchmark

Run the exact-vs-underscore-heavy algorithm comparison on the quotes dataset with three measured iterations per combination:

```bash
cargo run -p runner --bin runner --release -- \
  --data-csv benchmarks/quotes/data_quotes_utf8.csv \
  --algorithms-csv benchmarks/quotes/exact-vs-underscore/algorithms.csv \
  --generic-matcher static \
  --patterns-csv benchmarks/quotes/exact-vs-underscore/patterns.csv \
  --indexes-csv benchmarks/quotes/exact-vs-underscore/indexes.csv \
  --iterations 3 \
  --max-row-bytes 50MB \
  --max-total-bytes 100MB \
  --output-csv results/quotes_algos_raw.csv \
  --summary-csv results/quotes_algos_summary.csv
```

Run the quote-specific algorithm stress cases for BM, TwoWay, TwoWay2, TwoWay3, PairHorspool, and the Naive V2 prefilter variants:

```bash
cargo run -p runner --bin runner --release -- \
  --data-csv benchmarks/quotes/data_quotes_utf8.csv \
  --algorithms-csv benchmarks/quotes/algorithm-cases/algorithms.csv \
  --generic-matcher static \
  --patterns-csv benchmarks/quotes/algorithm-cases/patterns.csv \
  --indexes-csv benchmarks/quotes/algorithm-cases/indexes.csv \
  --iterations 3 \
  --max-row-bytes 50MB \
  --max-total-bytes 100MB \
  --output-csv results/quotes_algorithm_cases_raw.csv \
  --summary-csv results/quotes_algorithm_cases_summary.csv
```

Run the matcher-engine comparison with `StdSearch`, `NaiveVectorizedV2`, and `NaiveVectorizedV2Wildcard` across static, adaptive, and recursive matchers:

```bash
cargo run -p runner --bin runner --release -- \
  --data-csv benchmarks/quotes/data_quotes_utf8.csv \
  --algorithms-csv benchmarks/quotes/matcher-comparison/algorithms.csv \
  --generic-matcher static,adaptive,recursive \
  --patterns-csv benchmarks/quotes/matcher-comparison/patterns.csv \
  --indexes-csv benchmarks/quotes/matcher-comparison/indexes.csv \
  --iterations 3 \
  --max-row-bytes 50MB \
  --max-total-bytes 100MB \
  --output-csv results/quotes_matchers_raw.csv \
  --summary-csv results/quotes_matchers_summary.csv
```

For matcher-engine comparisons, patterns with multiple `%`-separated fragments are the most useful for wildcard-capable algorithms, because `_` remains inside their literal fragments by default.

Run the index comparison on prefix/equality-friendly quote patterns across all configured indexes, including `prefix-btree`:

```bash
cargo run -p runner --bin runner --release -- \
  --data-csv benchmarks/quotes/data_quotes_utf8.csv \
  --algorithms-csv benchmarks/quotes/index-comparison/algorithms.csv \
  --generic-matcher static \
  --patterns-csv benchmarks/quotes/index-comparison/patterns.csv \
  --indexes-csv benchmarks/quotes/index-comparison/indexes.csv \
  --iterations 3 \
  --max-row-bytes 50MB \
  --max-total-bytes 100MB \
  --output-csv results/quotes_indexes_raw.csv \
  --summary-csv results/quotes_indexes_summary.csv
```

## JOB Benchmark

JOB configs use predicates copied from the Join Order Benchmark SQL files. Patterns are split per column because runner pattern CSVs apply to every data row in one invocation.

Available JOB data/pattern pairs:

| Column | Data CSV | Pattern CSV |
|---|---|---|
| `cast_info.note` | `benchmarks/job/data_cast_info_note.csv` | `benchmarks/job/index-comparison/patterns_cast_info_note.csv` |
| `keyword.keyword` | `benchmarks/job/data_keyword_keyword.csv` | `benchmarks/job/index-comparison/patterns_keyword_keyword.csv` |
| `movie_companies.note` | `benchmarks/job/data_movie_companies_note.csv` | `benchmarks/job/index-comparison/patterns_movie_companies_note.csv` |
| `movie_info.info` | `benchmarks/job/data_movie_info_info.csv` | `benchmarks/job/index-comparison/patterns_movie_info_info.csv` |
| `name.name` | `benchmarks/job/data_name_name.csv` | `benchmarks/job/index-comparison/patterns_name_name.csv` |
| `title.title` | `benchmarks/job/data_title_title.csv` | `benchmarks/job/index-comparison/patterns_title_title.csv` |

The `scripts/run_bench.py` wrapper writes each run to a timestamped directory under `--result-root` with `raw.csv`, `summary.csv`, `python_summary.csv`, copied inputs, `command.txt`, `info.txt`, `hardware.json`, `hardware.txt`, and plots.

Run all JOB benchmark modes for all JOB columns with checkpointing:

```bash
python3 scripts/run_all_benchmarks.py \
  --only-suite job \
  --result-root benchmark_results \
  --iterations 3
```

This writes:

| Mode | Output Directory |
|---|---|
| index consistency cross product | `benchmark_results/job/index-consistency/<column>_<timestamp>/` |
| full-scan algorithm comparison | `benchmark_results/job/algorithm-comparison/<column>_<timestamp>/` |
| matcher-engine comparison | `benchmark_results/job/matcher-comparison/<column>_<timestamp>/` |
| `LibcMemmem`-only index comparison | `benchmark_results/job/index-memmem/<column>_<timestamp>/` |
| UTF8-vs-FSST `LibcMemmem` index comparison | `benchmark_results/job/fsst-index-memmem/<column>_<timestamp>/` |

Progress is tracked in `benchmark_results/checkpoint.csv`; rows already marked `done=true` are skipped on the next invocation. Use `--force` to rerun completed rows. Use `--only-mode` or `--only-case` to run a subset.

The index consistency mode is intentionally a cross product of five verifier algorithms and all indexes to confirm index behavior is consistent across verifier implementations. The matcher-engine comparison runs `StdSearch`, `NaiveVectorizedV2`, and `NaiveVectorizedV2Wildcard` with `static`, `adaptive`, and `recursive` matchers for every JOB column, reusing the same per-column pattern CSVs as the other JOB modes. The `index-memmem` mode includes indexes but fixes the verifier algorithm to `LibcMemmem`, avoiding a full index-by-algorithm cross product. The `fsst-index-memmem` mode uses `utf8;fsst` storage with `LibcMemmem` and all indexes, so it isolates the decoded FSST search cost against the UTF-8 baseline and shows how much candidate indexes reduce that cost.

The same UTF8-vs-FSST fixed-`LibcMemmem` index comparison is available for DNA and quotes:

```bash
python3 scripts/run_all_benchmarks.py --only-mode fsst-index-memmem --dry-run
```

## FFTSTR Benchmark

Generate the artificial `abab...` benchmark table and pattern CSV. The default table has 1000 rows, each with length 512, and patterns include `%aa%`, `%a_b%`, then wildcard cores from length 4 to 512 in steps of 4.

```bash
python3 scripts/generate_fftstr_benchmark.py
```

Run the benchmark comparing scalar/SIMD literal baselines, `TwoWay` variants, glibc-style exact search, `FftStr1`, and `FftstrV2`:

```bash
cargo run -p runner --bin runner --release -- \
  --data-csv benchmarks/fftstr/abab-wildcards/data.csv \
  --algorithms-csv benchmarks/fftstr/abab-wildcards/algorithms.csv \
  --generic-matcher static \
  --patterns-csv benchmarks/fftstr/abab-wildcards/patterns.csv \
  --indexes-csv benchmarks/fftstr/abab-wildcards/indexes.csv \
  --iterations 3 \
  --max-row-bytes 50MB \
  --max-total-bytes 100MB \
  --output-csv results/fftstr_abab_raw.csv \
  --summary-csv results/fftstr_abab_summary.csv
```
