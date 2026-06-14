# Matcher Multi-Percent Benchmark Report

## Executive Summary

- On DNA, `adaptive` is the strongest aggregate matcher on the recursive-safe comparison set by median-execute geomean: static 5.582ms, adaptive 5.545ms, recursive 7.787ms.
- On quotes, `adaptive` is the strongest aggregate matcher by median-execute geomean: static 1.883ms, adaptive 1.859ms, recursive 2.175ms. Recursive still wins several simple/selective patterns even though it loses aggregate time.
- Median and geomean produce the same aggregate winner for every dataset subset. Pattern-level flips exist, especially in quotes, but they are local rather than changing the overall conclusion.
- Complexity matters: static is favored by DNA patterns with a selective longer internal anchor; adaptive starts from the first segment anchor and then may switch to failed anchors, which can help some broad cases but regress others; recursive is acceptable on simple or anchored cases but fails catastrophically when multi-`%` DNA patterns create many common one-base branches.
- The excluded DNA recursive edge case is not an arbitrary timeout. A one-row repro produced a `sample` stack dominated by `LikePattern::match_from` and `naive_find_neon_v2`, and a token-walk model estimated about 5.07 billion recursive calls for that row.
- Pattern-level winner flips between median and geomean: 5 of 75 pattern summaries.

## Inputs

- DNA static/adaptive: `benchmark_results/dna/matcher-multi-percent/gencode_utf8_20260613_091821`
- DNA recursive-safe: `benchmark_results/dna/matcher-multi-percent-recursive/gencode_utf8_recursive_20260613_092002`
- Quotes: `benchmark_results/quotes/matcher-multi-percent/quotes_20260613_092056`
- All runs use full scans, 10 measured iterations, one warmup, and native CPU Rust flags recorded in each result directory.
- Generic plots are intentionally not generated; the report keeps machine-readable tables for bespoke thesis figures.

## Method and Metrics

Each comparison keeps dataset, storage, algorithm, index choice, pattern name, and pattern text fixed, then varies only the generic matcher. This is important because the literal-search algorithm can dominate absolute time. The report therefore aggregates over comparable groups rather than mixing unrelated rows.

Two metrics are reported. `median_execute` is the median verifier execution time over the 10 measured iterations and excludes compile/load time. `geomean_total` is the geometric mean of total query time from the benchmark summary and includes candidate preparation. In these full-scan `none`-index benchmarks, candidate preparation is effectively zero, so the two metrics mostly test robustness to iteration variance rather than different work.

Pattern complexity is classified from the LIKE pattern text: wildcard-only, single gapped segment, ordered segments, or complex multi-`%`. Specificity is classified from matched-row fraction: all rows, broad, medium, selective, or zero-match.

## Aggregate Results

| dataset | metric | matcher | groups | geomean_ms | sum_ms | wins | clear_wins_5pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DNA shared recursive-safe | median_execute | static | 75 | 5.582 | 3058.035 | 16 | 7 |
| DNA shared recursive-safe | median_execute | adaptive | 75 | 5.545 | 3190.430 | 37 | 12 |
| DNA shared recursive-safe | median_execute | recursive | 75 | 7.787 | 4782.663 | 22 | 11 |
| DNA shared recursive-safe | geomean_total | static | 75 | 5.605 | 3060.672 | 16 | 8 |
| DNA shared recursive-safe | geomean_total | adaptive | 75 | 5.591 | 3191.892 | 36 | 12 |
| DNA shared recursive-safe | geomean_total | recursive | 75 | 7.854 | 4784.595 | 23 | 12 |
| DNA full static/adaptive | median_execute | static | 78 | 6.153 | 3271.463 | 25 | 13 |
| DNA full static/adaptive | median_execute | adaptive | 78 | 6.111 | 3401.206 | 53 | 19 |
| DNA full static/adaptive | geomean_total | static | 78 | 6.177 | 3274.423 | 25 | 14 |
| DNA full static/adaptive | geomean_total | adaptive | 78 | 6.160 | 3402.686 | 53 | 19 |
| Quotes | median_execute | static | 72 | 1.883 | 315.375 | 9 | 0 |
| Quotes | median_execute | adaptive | 72 | 1.859 | 315.147 | 32 | 5 |
| Quotes | median_execute | recursive | 72 | 2.175 | 465.093 | 31 | 7 |
| Quotes | geomean_total | static | 72 | 1.894 | 316.234 | 9 | 0 |
| Quotes | geomean_total | adaptive | 72 | 1.867 | 315.391 | 35 | 6 |
| Quotes | geomean_total | recursive | 72 | 2.180 | 464.650 | 28 | 7 |

## Median vs Geomean Winner Flips

The aggregate dataset-level winner is stable, but a few individual pattern-level winners change between `median_execute` and `geomean_total`. These flips are useful indicators of close races or slightly different run-to-run stability.

| dataset | pattern_name | complexity | specificity | rows_matched | median_execute_winner | geomean_total_winner | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DNA shared recursive-safe | suffix_short_hit | simple_literal | selective | 1 | recursive | adaptive | %GCATTAAAATT |
| DNA full static/adaptive | any_len001 | wildcard_only | all_rows | 50421 | static | adaptive | %_% |
| Quotes | exact_age_issue | simple_literal | selective | 1 | static | adaptive | Age is an issue of mind over matter. If you don't mind, it doesn't matter. |
| Quotes | quote_multi_common4 | complex_multi_percent | medium | 3636 | static | adaptive | %the_%of_%and_%to_% |
| Quotes | quote_people_repeat_stance_dance | complex_multi_percent | selective | 1 | adaptive | static | %people%repeat%stance%dance% |

## Complexity and Specificity Summary

| dataset | complexity | specificity | patterns | median_static_wins | median_adaptive_wins | median_recursive_wins | geomean_static_wins | geomean_adaptive_wins | geomean_recursive_wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DNA full static/adaptive | complex_multi_percent | broad | 5 | 0 | 5 | 0 | 0 | 5 | 0 |
| DNA full static/adaptive | complex_multi_percent | medium | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| DNA full static/adaptive | complex_multi_percent | selective | 2 | 2 | 0 | 0 | 2 | 0 | 0 |
| DNA full static/adaptive | ordered_segments | selective | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| DNA full static/adaptive | simple_literal | selective | 2 | 1 | 1 | 0 | 1 | 1 | 0 |
| DNA full static/adaptive | simple_literal | zero_match | 2 | 1 | 1 | 0 | 1 | 1 | 0 |
| DNA full static/adaptive | single_gapped_segment | all_rows | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| DNA full static/adaptive | single_gapped_segment | broad | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| DNA full static/adaptive | single_gapped_segment | medium | 2 | 0 | 2 | 0 | 0 | 2 | 0 |
| DNA full static/adaptive | single_gapped_segment | selective | 5 | 1 | 4 | 0 | 1 | 4 | 0 |
| DNA full static/adaptive | single_gapped_segment | zero_match | 3 | 1 | 2 | 0 | 1 | 2 | 0 |
| DNA full static/adaptive | wildcard_only | all_rows | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
| DNA shared recursive-safe | complex_multi_percent | broad | 5 | 0 | 4 | 1 | 0 | 4 | 1 |
| DNA shared recursive-safe | complex_multi_percent | selective | 2 | 2 | 0 | 0 | 2 | 0 | 0 |
| DNA shared recursive-safe | ordered_segments | selective | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| DNA shared recursive-safe | simple_literal | selective | 2 | 1 | 0 | 1 | 1 | 1 | 0 |
| DNA shared recursive-safe | simple_literal | zero_match | 2 | 0 | 1 | 1 | 0 | 1 | 1 |
| DNA shared recursive-safe | single_gapped_segment | all_rows | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| DNA shared recursive-safe | single_gapped_segment | broad | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| DNA shared recursive-safe | single_gapped_segment | medium | 2 | 0 | 2 | 0 | 0 | 2 | 0 |
| DNA shared recursive-safe | single_gapped_segment | selective | 5 | 1 | 4 | 0 | 1 | 4 | 0 |
| DNA shared recursive-safe | single_gapped_segment | zero_match | 3 | 1 | 0 | 2 | 1 | 0 | 2 |
| DNA shared recursive-safe | wildcard_only | all_rows | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| Quotes | complex_multi_percent | broad | 2 | 0 | 2 | 0 | 0 | 2 | 0 |
| Quotes | complex_multi_percent | medium | 3 | 1 | 2 | 0 | 0 | 3 | 0 |
| Quotes | complex_multi_percent | selective | 3 | 0 | 1 | 2 | 1 | 0 | 2 |
| Quotes | complex_multi_percent | zero_match | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| Quotes | ordered_segments | selective | 3 | 0 | 1 | 2 | 0 | 1 | 2 |
| Quotes | simple_literal | selective | 3 | 1 | 1 | 1 | 0 | 2 | 1 |
| Quotes | simple_literal | zero_match | 2 | 0 | 2 | 0 | 0 | 2 | 0 |
| Quotes | single_gapped_segment | broad | 2 | 0 | 1 | 1 | 0 | 1 | 1 |
| Quotes | single_gapped_segment | medium | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| Quotes | single_gapped_segment | selective | 3 | 0 | 0 | 3 | 0 | 0 | 3 |
| Quotes | wildcard_only | all_rows | 1 | 0 | 0 | 1 | 0 | 0 | 1 |

## Pattern-Level Winner Summary

### DNA shared recursive-safe

Median winner counts: `adaptive`=13, `recursive`=7, `static`=5
Geomean-total winner counts: `adaptive`=14, `recursive`=6, `static`=5

| pattern_name | complexity | specificity | rows_matched | median_execute_winner | median_execute_second_best_ratio | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| interleaved_with_bigger | single_gapped_segment | selective | 297 | static | 1.84x | %C_G_C_G_CGC_G_C_G% |
| anchored_prefix_gap | single_gapped_segment | zero_match | 0 | recursive | 1.19x | AAGC________ACCG% |
| anchored_prefix_exact_segments | ordered_segments | selective | 55 | recursive | 1.18x | CGCA%GAGA%CGGG% |
| one_fixed_len002 | single_gapped_segment | all_rows | 50421 | adaptive | 1.12x | %G_% |
| interleaved_common | single_gapped_segment | medium | 3085 | adaptive | 1.09x | %C_G_C_G_C_G_C_G% |
| gap_len016 | single_gapped_segment | medium | 538 | adaptive | 1.05x | %AAGC________ACCG% |
| exact_short_nohit | simple_literal | zero_match | 0 | recursive | 1.04x | TTGCACAGGCATTAAAAAA |
| many_common_short_segments | complex_multi_percent | broad | 50420 | recursive | 1.04x | %A_%C_%G_%T_% |
| any_len001 | wildcard_only | all_rows | 50421 | recursive | 1.04x | %_% |
| suffix_short_nohit | simple_literal | zero_match | 0 | adaptive | 1.04x | %TTTTGGGGCCCCAAAA |

### DNA full static/adaptive

Median winner counts: `adaptive`=18, `static`=8
Geomean-total winner counts: `adaptive`=19, `static`=7

| pattern_name | complexity | specificity | rows_matched | median_execute_winner | median_execute_second_best_ratio | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| interleaved_with_bigger | single_gapped_segment | selective | 297 | static | 1.84x | %C_G_C_G_CGC_G_C_G% |
| many_common_short_segments | complex_multi_percent | broad | 50420 | adaptive | 1.27x | %A_%C_%G_%T_% |
| suffix_short_hit | simple_literal | selective | 1 | adaptive | 1.19x | %GCATTAAAATT |
| one_fixed_len002 | single_gapped_segment | all_rows | 50421 | adaptive | 1.17x | %G_% |
| interleaved_common | single_gapped_segment | medium | 3085 | adaptive | 1.10x | %C_G_C_G_C_G_C_G% |
| sparse_len004 | single_gapped_segment | broad | 50407 | adaptive | 1.09x | %C__G% |
| exact_short_hit | simple_literal | selective | 1 | static | 1.09x | TTGCACAGGCATTAAAATT |
| exact_short_nohit | simple_literal | zero_match | 0 | static | 1.09x | TTGCACAGGCATTAAAAAA |
| suffix_short_nohit | simple_literal | zero_match | 0 | adaptive | 1.08x | %TTTTGGGGCCCCAAAA |
| gap_len016 | single_gapped_segment | medium | 538 | adaptive | 1.06x | %AAGC________ACCG% |

### Quotes

Median winner counts: `adaptive`=10, `recursive`=12, `static`=2
Geomean-total winner counts: `adaptive`=11, `recursive`=12, `static`=1

| pattern_name | complexity | specificity | rows_matched | median_execute_winner | median_execute_second_best_ratio | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| anchored_prefix_gap | single_gapped_segment | selective | 1 | recursive | 1.19x | Age is a________________over mat% |
| anchored_prefix_exact | simple_literal | selective | 1 | recursive | 1.08x | Age is an issue% |
| multi_gap_common | complex_multi_percent | broad | 9937 | adaptive | 1.06x | %the_%of_%and_% |
| wild_short | single_gapped_segment | broad | 75409 | adaptive | 1.05x | %t_% |
| quote_multi_common5 | complex_multi_percent | medium | 1747 | adaptive | 1.05x | %the_%of_%and_%to_%in_% |
| quote_learning_gap_multi | complex_multi_percent | selective | 3 | recursive | 1.04x | %A_yone%learning%young%m_nd% |
| wild_any | wildcard_only | all_rows | 75966 | recursive | 1.04x | %_% |
| exact_nohit | simple_literal | zero_match | 0 | adaptive | 1.03x | This exact quote does not exist. |
| ordered_selective | ordered_segments | selective | 1 | recursive | 1.02x | %coffee%obituaries% |
| wild_4 | single_gapped_segment | broad | 58785 | recursive | 1.02x | %t__ % |

## Edge Cases

### static_bad_vs_adaptive

| dataset | algorithm | pattern_name | ratio | complexity | specificity | rows_matched | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DNA | NaiveVectorizedV2 | many_common_short_segments | 1.87x | complex_multi_percent | broad | 50420 | %A_%C_%G_%T_% |
| Quotes | StdSearch | quote_learning_gap_multi | 1.67x | complex_multi_percent | selective | 3 | %A_yone%learning%young%m_nd% |
| DNA | NaiveVectorizedV2 | suffix_short_hit | 1.57x | simple_literal | selective | 1 | %GCATTAAAATT |
| DNA | NaiveVectorizedV2 | one_fixed_len002 | 1.35x | single_gapped_segment | all_rows | 50421 | %G_% |
| DNA | NaiveVectorizedV2 | sparse_len004 | 1.25x | single_gapped_segment | broad | 50407 | %C__G% |
| DNA | StdSearch | gap_len016 | 1.17x | single_gapped_segment | medium | 538 | %AAGC________ACCG% |
| DNA | NaiveVectorizedV2 | interleaved_common | 1.17x | single_gapped_segment | medium | 3085 | %C_G_C_G_C_G_C_G% |
| Quotes | NaiveVectorizedV2 | anchored_prefix_exact | 1.17x | simple_literal | selective | 1 | Age is an issue% |

### adaptive_bad_vs_static

| dataset | algorithm | pattern_name | ratio | complexity | specificity | rows_matched | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DNA | StdSearch | interleaved_with_bigger | 2.80x | single_gapped_segment | selective | 297 | %C_G_C_G_CGC_G_C_G% |
| DNA | NaiveVectorizedV2 | interleaved_with_bigger | 2.27x | single_gapped_segment | selective | 297 | %C_G_C_G_CGC_G_C_G% |
| Quotes | StdSearch | wild_4 | 1.43x | single_gapped_segment | broad | 58785 | %t__ % |
| Quotes | NaiveVectorizedV2 | wild_4 | 1.37x | single_gapped_segment | broad | 58785 | %t__ % |
| DNA | StdSearch | dna_multi_percent_common4 | 1.23x | complex_multi_percent | selective | 494 | %CACA_GATC%AGCA__CACC%GACG_GCGG%GAGA_GCTG% |
| DNA | StdSearch | exact_short_nohit | 1.15x | simple_literal | zero_match | 0 | TTGCACAGGCATTAAAAAA |
| DNA | StdSearch | exact_short_hit | 1.15x | simple_literal | selective | 1 | TTGCACAGGCATTAAAATT |
| Quotes | NaiveVectorizedV2 | exact_age_issue | 1.14x | simple_literal | selective | 1 | Age is an issue of mind over matter. If you don't mind, it doesn't matter. |

### recursive_bad_vs_best_segment

| dataset | algorithm | pattern_name | ratio | complexity | specificity | rows_matched | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DNA | StdSearch | anchored_suffix_gap | 234.69x | single_gapped_segment | zero_match | 0 | %GGGCAC____________________GGTGTC |
| DNA | NaiveVectorizedV2 | anchored_suffix_gap | 201.97x | single_gapped_segment | zero_match | 0 | %GGGCAC____________________GGTGTC |
| DNA | NaiveVectorizedV2Wildcard | ordered_wild_segments | 7.46x | complex_multi_percent | broad | 30963 | %CGC_%GAGA_%CGGG_% |
| DNA | NaiveVectorizedV2 | ordered_wild_segments | 6.98x | complex_multi_percent | broad | 30963 | %CGC_%GAGA_%CGGG_% |
| DNA | NaiveVectorizedV2 | ordered_mixed_segments | 5.57x | complex_multi_percent | broad | 27590 | %AAGC_%GGGA_%ACCG% |
| DNA | NaiveVectorizedV2Wildcard | ordered_mixed_segments | 5.47x | complex_multi_percent | broad | 27590 | %AAGC_%GGGA_%ACCG% |
| DNA | StdSearch | ordered_wild_segments | 4.94x | complex_multi_percent | broad | 30963 | %CGC_%GAGA_%CGGG_% |
| DNA | StdSearch | interleaved_with_bigger | 4.72x | single_gapped_segment | selective | 297 | %C_G_C_G_CGC_G_C_G% |

### recursive_good_vs_best_segment

| dataset | algorithm | pattern_name | ratio | complexity | specificity | rows_matched | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Quotes | StdSearch | anchored_prefix_gap | 1.35x | single_gapped_segment | selective | 1 | Age is a________________over mat% |
| DNA | NaiveVectorizedV2 | anchored_prefix_gap | 1.30x | single_gapped_segment | zero_match | 0 | AAGC________ACCG% |
| Quotes | StdSearch | anchored_prefix_exact | 1.27x | simple_literal | selective | 1 | Age is an issue% |
| DNA | StdSearch | anchored_prefix_gap | 1.23x | single_gapped_segment | zero_match | 0 | AAGC________ACCG% |
| DNA | StdSearch | anchored_prefix_exact_segments | 1.21x | ordered_segments | selective | 55 | CGCA%GAGA%CGGG% |
| DNA | NaiveVectorizedV2 | anchored_prefix_exact_segments | 1.19x | ordered_segments | selective | 55 | CGCA%GAGA%CGGG% |
| DNA | StdSearch | suffix_short_hit | 1.15x | simple_literal | selective | 1 | %GCATTAAAATT |
| Quotes | NaiveVectorizedV2 | anchored_prefix_gap | 1.15x | single_gapped_segment | selective | 1 | Age is a________________over mat% |

## Interpretation

The median-execute and geomean-total metrics mostly agree on the qualitative outcome, but they emphasize different aspects. Median execute isolates verifier time for a typical measured iteration, while geomean total also includes candidate preparation and dampens the effect of very large outliers. Because all new suites use full scans with the `none` index, candidate preparation is close to zero; differences between the two metrics are therefore small and mainly reflect run-to-run dispersion.

For DNA shared recursive-safe under `median_execute`, the lowest aggregate geomean is `adaptive` (5.545ms), while the largest raw win count is `adaptive` (37 groups).
For DNA shared recursive-safe under `geomean_total`, the lowest aggregate geomean is `adaptive` (5.591ms), while the largest raw win count is `adaptive` (36 groups).
For DNA full static/adaptive under `median_execute`, the lowest aggregate geomean is `adaptive` (6.111ms), while the largest raw win count is `adaptive` (53 groups).
For DNA full static/adaptive under `geomean_total`, the lowest aggregate geomean is `adaptive` (6.160ms), while the largest raw win count is `adaptive` (53 groups).
For Quotes under `median_execute`, the lowest aggregate geomean is `adaptive` (1.859ms), while the largest raw win count is `adaptive` (32 groups).
For Quotes under `geomean_total`, the lowest aggregate geomean is `adaptive` (1.867ms), while the largest raw win count is `adaptive` (35 groups).

DNA and quotes behave differently because their alphabets and row lengths differ. DNA has a four-symbol alphabet, so one-character anchors are extremely non-selective; recursive backtracking is therefore fragile when several `%` operators are followed by one-base literals. Quotes have shorter rows and word-like fragments, so recursive remains usable on this benchmark even when it loses on common-word multi-segment cases.

Static is strongest when one literal inside a fixed-width segment is much more selective than other anchors. It can jump directly to that anchor and keep scanning it. Adaptive starts from the first literal anchor in the segment, then switches to the first failed literal anchor during verification. This can help if the failed anchor is a better future probe, but it can also regress by probing broad anchors. Recursive is weakest when common early `%` branches create many equivalent suffix failures.

Largest static-vs-adaptive losses in these runs: `DNA:many_common_short_segments` 1.87x; `Quotes:quote_learning_gap_multi` 1.67x; `DNA:suffix_short_hit` 1.57x.
Largest adaptive-vs-static losses in these runs: `DNA:interleaved_with_bigger` 2.80x; `DNA:interleaved_with_bigger` 2.27x; `Quotes:wild_4` 1.43x.
Largest recursive-vs-segment losses in the safe comparison set: `DNA:anchored_suffix_gap` 234.69x; `DNA:anchored_suffix_gap` 201.97x; `DNA:ordered_wild_segments` 7.46x. The excluded DNA edge case is much worse and is documented separately in `RECURSIVE_MATCHER_EDGECASE.md`.

## Thesis-Oriented Takeaways

1. The generic matcher is not a single universally optimal component. Its best strategy depends on pattern structure and alphabet statistics. This supports evaluating LIKE engines on structured pattern families rather than only on random substrings or pure contains/prefix cases.
2. Segment anchoring is valuable on DNA because the alphabet is small and one-character anchors are weak. Static anchoring often wins by choosing the longest fixed literal inside a segment, avoiding repeated probes of common bases.
3. A previous best-start adaptive experiment was not automatically better than first-start adaptive. Starting from the static anchor and then switching normally can regress, especially when the first failed literal is less selective than the best anchor.
4. Recursive matching is useful as a correctness reference and can be competitive on simple anchored/selective patterns. However, it should not be used as the main verifier for adversarial multi-`%` patterns unless memoization is added.
5. Specificity interacts with complexity. Selective long fragments make static attractive; broad patterns with many common fragments can favor adaptive or expose recursive blow-ups; zero-match suffix cases are especially dangerous for recursive because failure happens only after exploring many prefixes.
6. Median and geomean agree at the aggregate level in these full-scan runs, which strengthens the conclusion. Where they differ at pattern level, the differences are small enough to treat as close races or stability effects rather than evidence for a different default.

## Generated Tables

- `tables/aggregate_by_matcher.csv`
- `tables/pattern_summary.csv`
- `tables/complexity_specificity_summary.csv`
- `tables/metric_winner_flips.csv`
- `tables/edge_cases.csv`
