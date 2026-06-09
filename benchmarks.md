# Benchmarks

## DNA Benchmark

Run the exact-vs-underscore-heavy algorithm comparison on GENCODE human transcripts with three measured iterations per combination:

```bash
cargo run -p runner --bin runner --release -- \
  --data-csv benchmarks/dna/data_gencode_dna_utf8_dna2.csv \
  --algorithms-csv benchmarks/dna/exact-vs-underscore/algorithms.csv \
  --generic-matcher static \
  --patterns-csv benchmarks/dna/exact-vs-underscore/patterns.csv \
  --indexes-csv benchmarks/dna/exact-vs-underscore/indexes.csv \
  --iterations 3 \
  --max-row-bytes 50MB \
  --max-total-bytes 100MB \
  --output-csv results/gencode_algos_raw.csv \
  --summary-csv results/gencode_algos_summary.csv
```

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
