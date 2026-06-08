# Bounds-Check Cost Experiments

These results use only the latest 10-iteration rerun.

All runs used full scan, `--generic-matcher static`, `--iterations 10`, `--max-row-bytes 50MB`, and `--max-total-bytes 100MB`.

Aggregate tables use per-pattern `median_execute_ns` from the runner summary. `geomean`, `median`, and `p90` are aggregated across the selected pattern set. Per-pattern tables include `p90_query_total_ns` from the runner summary.

## Datasets

| Dataset | Source config | Storage | Rows loaded | Symbols loaded | Pattern config |
|---|---|---|---:|---:|---|
| DNA | `benchmarks/dna/data_gencode_dna_utf8_dna2.csv` | `utf8` | 54,107 | 100,000,000 | `benchmarks/dna/patterns_gencode_exact_vs_underscore.csv` |
| Quotes | `benchmarks/quotes/data_quotes_utf8.csv` | `utf8` | 75,966 | 10,741,913 | `benchmarks/quotes/patterns_quotes_exact_vs_underscore.csv` |

## Algorithms

| Experiment | Original | Boundless duplicate | What changed |
|---|---|---|---|
| BM | `BM` | `BMBoundless` | BM hot loop uses raw pointers and unchecked table access instead of safe slice indexing. |
| Wildcard naive | `NaiveVectorizedV2Wildcard` | `NaiveVectorizedV2WildcardBoundless` | Row suffix and candidate wildcard-verification slices use unchecked slicing after explicit bounds checks. |

## Source Results

| Experiment | Dataset | Raw CSV | Summary CSV |
|---|---|---|---|
| BM | DNA | `/tmp/opencode/dna_bm_boundless_iter10_raw.csv` | `/tmp/opencode/dna_bm_boundless_iter10_summary.csv` |
| BM | Quotes | `/tmp/opencode/quotes_bm_boundless_iter10_raw.csv` | `/tmp/opencode/quotes_bm_boundless_iter10_summary.csv` |
| Wildcard naive | DNA | `/tmp/opencode/dna_naive_wild_boundless_iter10_raw.csv` | `/tmp/opencode/dna_naive_wild_boundless_iter10_summary.csv` |
| Wildcard naive | Quotes | `/tmp/opencode/quotes_naive_wild_boundless_iter10_raw.csv` | `/tmp/opencode/quotes_naive_wild_boundless_iter10_summary.csv` |

## Aggregate Results

| Experiment | Dataset | Pattern set | Original geomean median execute ms | Boundless geomean median execute ms | Ratio | Original median execute ms | Boundless median execute ms | Original p90 execute ms | Boundless p90 execute ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| BM | DNA | all | 29.024623 | 27.159417 | 0.9357x | 63.530715 | 62.939654 | 134.154077 | 132.488547 |
| BM | DNA | exact | 36.183140 | 33.936415 | 0.9379x | 56.404738 | 55.566551 | 112.649490 | 113.039135 |
| BM | DNA | underscore | 23.282356 | 21.735764 | 0.9336x | 67.697778 | 67.153396 | 163.959410 | 162.889078 |
| BM | Quotes | all | 5.776923 | 6.030421 | 1.0439x | 7.353701 | 7.559871 | 20.701435 | 20.897016 |
| BM | Quotes | exact | 4.935552 | 4.948241 | 1.0026x | 5.052512 | 4.929779 | 9.502897 | 9.508149 |
| BM | Quotes | underscore | 6.761723 | 7.349273 | 1.0869x | 8.640673 | 8.823504 | 25.017039 | 24.426609 |
| Wildcard naive | DNA | all | 10.332733 | 10.310828 | 0.9979x | 32.310542 | 31.326040 | 39.153850 | 36.906790 |
| Wildcard naive | DNA | exact | 13.285478 | 12.817636 | 0.9648x | 31.962180 | 30.896070 | 33.478271 | 31.839678 |
| Wildcard naive | DNA | underscore | 8.036247 | 8.294288 | 1.0321x | 36.900602 | 36.499444 | 42.083200 | 41.044388 |
| Wildcard naive | Quotes | all | 1.333298 | 1.292931 | 0.9697x | 1.592021 | 1.524045 | 1.848316 | 1.762424 |
| Wildcard naive | Quotes | exact | 1.481482 | 1.439955 | 0.9720x | 1.666292 | 1.582502 | 2.050806 | 2.010811 |
| Wildcard naive | Quotes | underscore | 1.199936 | 1.160918 | 0.9675x | 1.525974 | 1.453977 | 1.840340 | 1.762424 |

## BM Per-Pattern Results

### DNA

| Pattern | BM median execute ms | BMBoundless median execute ms | Median ratio | BM p90 query ms | BMBoundless p90 query ms | P90 ratio |
|---|---:|---:|---:|---:|---:|---:|
| `exact_len001` | 3.186545 | 2.062231 | 0.6472x | 4.236536 | 2.513178 | 0.5932x |
| `exact_len002` | 10.611956 | 10.450660 | 0.9848x | 10.720024 | 10.564477 | 0.9855x |
| `exact_len004` | 51.011590 | 49.710336 | 0.9745x | 52.108687 | 50.950261 | 0.9778x |
| `exact_len016` | 91.708669 | 90.669410 | 0.9887x | 93.390434 | 92.847289 | 0.9942x |
| `exact_len032` | 112.649490 | 113.039135 | 1.0035x | 113.100046 | 113.430365 | 1.0029x |
| `exact_len064` | 89.704005 | 89.002524 | 0.9922x | 91.702976 | 89.190436 | 0.9726x |
| `exact_len128` | 61.797886 | 61.422765 | 0.9939x | 62.152025 | 61.877853 | 0.9956x |
| `exact_len256` | 29.739971 | 29.307491 | 0.9855x | 30.242158 | 30.313426 | 1.0024x |
| `underscore_len001` | 0.153616 | 0.149710 | 0.9746x | 0.156151 | 0.149952 | 0.9603x |
| `underscore_len002` | 4.732382 | 2.818849 | 0.5957x | 5.971828 | 3.802301 | 0.6367x |
| `underscore_len004` | 9.372894 | 9.719383 | 1.0370x | 9.897817 | 10.102426 | 1.0207x |
| `underscore_len016` | 163.959410 | 162.889078 | 0.9935x | 171.881814 | 164.194958 | 0.9553x |
| `underscore_len032` | 125.862648 | 125.006105 | 0.9932x | 126.254510 | 126.172352 | 0.9993x |
| `underscore_len064` | 134.154077 | 132.488547 | 0.9876x | 135.769132 | 133.091478 | 0.9803x |
| `underscore_len128` | 65.263543 | 64.456543 | 0.9876x | 66.166299 | 65.489194 | 0.9898x |
| `underscore_len256` | 70.132013 | 69.850250 | 0.9960x | 70.668489 | 70.420608 | 0.9965x |

### Quotes

| Pattern | BM median execute ms | BMBoundless median execute ms | Median ratio | BM p90 query ms | BMBoundless p90 query ms | P90 ratio |
|---|---:|---:|---:|---:|---:|---:|
| `exact_len001` | 3.432377 | 3.384933 | 0.9862x | 3.608136 | 3.448225 | 0.9557x |
| `exact_len002` | 7.718078 | 7.668636 | 0.9936x | 7.761420 | 7.708231 | 0.9931x |
| `exact_len004` | 9.502897 | 9.508149 | 1.0006x | 9.540715 | 9.561949 | 1.0022x |
| `exact_len008` | 6.989324 | 7.451106 | 1.0661x | 7.062747 | 7.500267 | 1.0619x |
| `exact_len016` | 5.052512 | 4.929779 | 0.9757x | 5.062941 | 4.997957 | 0.9872x |
| `exact_len032` | 3.579508 | 3.599137 | 1.0055x | 3.614611 | 3.645993 | 1.0087x |
| `exact_len064` | 2.241942 | 2.226103 | 0.9929x | 2.291266 | 2.272480 | 0.9918x |
| `underscore_len001` | 0.206366 | 0.382134 | 1.8517x | 0.214856 | 0.386226 | 1.7976x |
| `underscore_len002` | 5.694560 | 5.743163 | 1.0085x | 5.762136 | 5.803014 | 1.0071x |
| `underscore_len004` | 20.701435 | 20.897016 | 1.0094x | 20.796805 | 21.731752 | 1.0450x |
| `underscore_len008` | 25.017039 | 24.426609 | 0.9764x | 25.212645 | 24.733083 | 0.9810x |
| `underscore_len016` | 14.768852 | 14.301550 | 0.9684x | 14.951166 | 14.390291 | 0.9625x |
| `underscore_len032` | 8.640673 | 8.823504 | 1.0212x | 9.311653 | 9.202138 | 0.9882x |
| `underscore_len064` | 8.320966 | 8.191664 | 0.9845x | 8.359895 | 8.256517 | 0.9876x |

## Wildcard Naive Per-Pattern Results

### DNA

| Pattern | NaiveVectorizedV2Wildcard median execute ms | NaiveVectorizedV2WildcardBoundless median execute ms | Median ratio | NaiveVectorizedV2Wildcard p90 query ms | NaiveVectorizedV2WildcardBoundless p90 query ms | P90 ratio |
|---|---:|---:|---:|---:|---:|---:|
| `exact_len001` | 0.733684 | 0.750927 | 1.0235x | 1.428330 | 0.967084 | 0.6771x |
| `exact_len002` | 2.823803 | 2.601008 | 0.9211x | 2.965367 | 2.898688 | 0.9775x |
| `exact_len004` | 12.680675 | 12.219352 | 0.9636x | 12.775633 | 12.278198 | 0.9611x |
| `exact_len016` | 31.958409 | 30.810976 | 0.9641x | 32.028852 | 32.330387 | 1.0094x |
| `exact_len032` | 33.078379 | 31.713424 | 0.9587x | 33.565404 | 31.930004 | 0.9513x |
| `exact_len064` | 33.478271 | 31.670914 | 0.9460x | 34.225340 | 33.256178 | 0.9717x |
| `exact_len128` | 32.655134 | 31.839678 | 0.9750x | 33.150628 | 32.274217 | 0.9736x |
| `exact_len256` | 31.965951 | 30.981164 | 0.9692x | 32.016850 | 31.976091 | 0.9987x |
| `underscore_len001` | 0.249441 | 0.224235 | 0.8990x | 0.258026 | 0.230854 | 0.8947x |
| `underscore_len002` | 0.754338 | 1.056286 | 1.4003x | 1.540687 | 1.876354 | 1.2179x |
| `underscore_len004` | 1.060964 | 1.272034 | 1.1989x | 2.597887 | 2.626306 | 1.0109x |
| `underscore_len016` | 39.153850 | 36.906790 | 0.9426x | 39.495981 | 37.157764 | 0.9408x |
| `underscore_len032` | 38.851325 | 36.723546 | 0.9452x | 40.278676 | 36.870953 | 0.9154x |
| `underscore_len064` | 37.618570 | 36.841139 | 0.9793x | 37.816719 | 36.937005 | 0.9767x |
| `underscore_len128` | 42.083200 | 41.044388 | 0.9753x | 42.157812 | 41.774072 | 0.9909x |
| `underscore_len256` | 36.182634 | 36.275343 | 1.0026x | 36.688139 | 37.337328 | 1.0177x |

### Quotes

| Pattern | NaiveVectorizedV2Wildcard median execute ms | NaiveVectorizedV2WildcardBoundless median execute ms | Median ratio | NaiveVectorizedV2Wildcard p90 query ms | NaiveVectorizedV2WildcardBoundless p90 query ms | P90 ratio |
|---|---:|---:|---:|---:|---:|---:|
| `exact_len001` | 0.675799 | 0.680052 | 1.0063x | 0.752000 | 0.733057 | 0.9748x |
| `exact_len002` | 1.327311 | 1.310186 | 0.9871x | 1.593102 | 1.371853 | 0.8611x |
| `exact_len004` | 2.050806 | 2.010811 | 0.9805x | 2.115182 | 2.023187 | 0.9565x |
| `exact_len008` | 1.848316 | 1.756396 | 0.9503x | 1.860884 | 1.773183 | 0.9529x |
| `exact_len016` | 1.831541 | 1.743370 | 0.9519x | 1.864100 | 1.789271 | 0.9599x |
| `exact_len032` | 1.666292 | 1.582502 | 0.9497x | 1.708397 | 1.605749 | 0.9399x |
| `exact_len064` | 1.509440 | 1.478560 | 0.9795x | 1.554558 | 1.513410 | 0.9735x |
| `underscore_len001` | 0.349684 | 0.326437 | 0.9335x | 0.358119 | 0.331759 | 0.9264x |
| `underscore_len002` | 0.858026 | 0.881989 | 1.0279x | 0.886393 | 0.917351 | 1.0349x |
| `underscore_len004` | 1.398923 | 1.399668 | 1.0005x | 1.431815 | 1.438178 | 1.0044x |
| `underscore_len008` | 1.840340 | 1.762424 | 0.9577x | 1.857589 | 1.835491 | 0.9881x |
| `underscore_len016` | 1.832685 | 1.753427 | 0.9568x | 1.864604 | 1.796237 | 0.9633x |
| `underscore_len032` | 1.658070 | 1.569529 | 0.9466x | 1.690630 | 1.591966 | 0.9416x |
| `underscore_len064` | 1.525974 | 1.453977 | 0.9528x | 1.560221 | 1.476405 | 0.9463x |

## Findings

From the 10-iteration rerun, `BMBoundless` is workload-sensitive. It improved DNA across all pattern groups by geomean (`0.9357x` overall), but regressed quotes overall (`1.0439x`) and especially quotes underscore patterns (`1.0869x`).

`NaiveVectorizedV2WildcardBoundless` is a clearer win on quotes, improving all pattern groups by geomean (`0.9697x` overall). On DNA it is roughly neutral overall (`0.9979x`), improves exact patterns (`0.9648x`), and regresses underscore patterns by geomean (`1.0321x`).

Based only on this rerun, removing bounds checks is not a general-purpose optimization. It looks useful in the wildcard vectorized path for quotes and DNA exact patterns, but not consistently for DNA underscore patterns or BM on quotes.
