# Benchmarks

## DNA Benchmark

Run the exact-vs-underscore-heavy algorithm comparison on GENCODE human transcripts with three measured iterations per combination:

```bash
cargo run -p runner --bin runner --release -- \
  --data-csv benchmarks/dna/data_gencode_dna_utf8_dna2.csv \
  --algorithms-csv benchmarks/dna/algorithms_dna_long_patterns.csv \
  --generic-matcher static \
  --patterns-csv benchmarks/dna/patterns_gencode_exact_vs_underscore.csv \
  --indexes-csv benchmarks/dna/indexes.csv \
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
  --algorithms-csv benchmarks/dna/algorithms_matchers.csv \
  --generic-matcher static,adaptive,recursive \
  --patterns-csv benchmarks/dna/patterns_gencode_matchers.csv \
  --indexes-csv benchmarks/dna/indexes.csv \
  --iterations 3 \
  --max-row-bytes 50MB \
  --max-total-bytes 100MB \
  --output-csv results/gencode_matchers_raw.csv \
  --summary-csv results/gencode_matchers_summary.csv
```

Warmups are controlled separately with `--warmups`; warmup runs are not written to the output CSV.

## Quotes Benchmark

Run the exact-vs-underscore-heavy algorithm comparison on the quotes dataset with three measured iterations per combination:

```bash
cargo run -p runner --bin runner --release -- \
  --data-csv benchmarks/quotes/data_quotes_utf8.csv \
  --algorithms-csv benchmarks/quotes/algorithms_quotes_long_patterns.csv \
  --generic-matcher static \
  --patterns-csv benchmarks/quotes/patterns_quotes_exact_vs_underscore.csv \
  --indexes-csv benchmarks/quotes/indexes.csv \
  --iterations 3 \
  --max-row-bytes 50MB \
  --max-total-bytes 100MB \
  --output-csv results/quotes_algos_raw.csv \
  --summary-csv results/quotes_algos_summary.csv
```

Run the quote-specific algorithm stress cases for BM, TwoWay, TwoWay2, and the Naive V2 prefilter variants:

```bash
cargo run -p runner --bin runner --release -- \
  --data-csv benchmarks/quotes/data_quotes_utf8.csv \
  --algorithms-csv benchmarks/quotes/algorithms_quotes_long_patterns.csv \
  --generic-matcher static \
  --patterns-csv benchmarks/quotes/patterns_quotes_algorithm_cases.csv \
  --indexes-csv benchmarks/quotes/indexes.csv \
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
  --algorithms-csv benchmarks/quotes/algorithms_matchers.csv \
  --generic-matcher static,adaptive,recursive \
  --patterns-csv benchmarks/quotes/patterns_quotes_matchers.csv \
  --indexes-csv benchmarks/quotes/indexes.csv \
  --iterations 3 \
  --max-row-bytes 50MB \
  --max-total-bytes 100MB \
  --output-csv results/quotes_matchers_raw.csv \
  --summary-csv results/quotes_matchers_summary.csv
```

For matcher-engine comparisons, patterns with multiple `%`-separated fragments are the most useful for wildcard-capable algorithms, because `_` remains inside their literal fragments by default.
