# Matcher Multi-Percent Benchmark Report

## Executive Summary

- On DNA, `static` is the strongest overall matcher on the recursive-safe comparison set: aggregate geomean of median execution is 12.877ms versus 14.682ms for `adaptive` and 21.202ms for `recursive`.
- On quotes, `adaptive` is the strongest aggregate matcher by geomean of median execution (3.364ms), narrowly ahead of `static` (3.379ms), while `recursive` is slower in aggregate (4.232ms) but still wins several simple/selective patterns.
- Median and geomean produce the same aggregate winner for every dataset subset. Pattern-level flips exist, especially in quotes, but they are local rather than changing the overall conclusion.
- Complexity matters: static is favored by DNA patterns with a selective longer internal anchor; the current adaptive matcher starts from that same best anchor and then may switch to failed anchors, which can help some cases but regress others; recursive is acceptable on simple or anchored cases but fails catastrophically when multi-`%` DNA patterns create many common one-base branches.
- The excluded DNA recursive edge case is not an arbitrary timeout. A one-row repro produced a `sample` stack dominated by `LikePattern::match_from` and `naive_find_neon_v2`, and a token-walk model estimated about 5.07 billion recursive calls for that row.
- Pattern-level winner flips between median and geomean: 5 of 63 pattern summaries.

## Inputs

- DNA static/adaptive: `benchmark_results/dna/matcher-multi-percent/gencode_utf8_20260612_003115`
- DNA recursive-safe: `benchmark_results/dna/matcher-multi-percent-recursive/gencode_utf8_recursive_20260612_003255`
- Quotes: `benchmark_results/quotes/matcher-multi-percent/quotes_20260612_003351`
- All runs use full scans, 10 measured iterations, one warmup, and native CPU Rust flags recorded in each result directory.
- Generic plots are intentionally not generated; the report keeps machine-readable tables for bespoke thesis figures.

## Method and Metrics

Each comparison keeps dataset, storage, algorithm, index choice, pattern name, and pattern text fixed, then varies only the generic matcher. This is important because the literal-search algorithm can dominate absolute time. The report therefore aggregates over comparable groups rather than mixing unrelated rows.

Two metrics are reported. `median_execute` is the median verifier execution time over the 10 measured iterations and excludes compile/load time. `geomean_total` is the geometric mean of total query time from the benchmark summary and includes candidate preparation. In these full-scan `none`-index benchmarks, candidate preparation is effectively zero, so the two metrics mostly test robustness to iteration variance rather than different work.

Pattern complexity is classified from the LIKE pattern text: wildcard-only, single gapped segment, ordered segments, or complex multi-`%`. Specificity is classified from matched-row fraction: all rows, broad, medium, selective, or zero-match.

## Aggregate Results

| dataset | metric | matcher | groups | geomean_ms | sum_ms | wins | clear_wins_5pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DNA shared recursive-safe | median_execute | static | 63 | 12.877 | 2978.889 | 45 | 20 |
| DNA shared recursive-safe | median_execute | adaptive | 63 | 14.682 | 3261.406 | 11 | 6 |
| DNA shared recursive-safe | median_execute | recursive | 63 | 21.202 | 4900.571 | 7 | 4 |
| DNA shared recursive-safe | geomean_total | static | 63 | 12.991 | 2983.812 | 45 | 19 |
| DNA shared recursive-safe | geomean_total | adaptive | 63 | 14.975 | 3279.955 | 12 | 5 |
| DNA shared recursive-safe | geomean_total | recursive | 63 | 21.267 | 4909.232 | 6 | 4 |
| DNA full static/adaptive | median_execute | static | 66 | 13.890 | 3186.330 | 55 | 31 |
| DNA full static/adaptive | median_execute | adaptive | 66 | 15.775 | 3478.019 | 11 | 6 |
| DNA full static/adaptive | geomean_total | static | 66 | 14.008 | 3191.336 | 54 | 30 |
| DNA full static/adaptive | geomean_total | adaptive | 66 | 16.079 | 3497.246 | 12 | 5 |
| Quotes | median_execute | static | 60 | 3.379 | 316.667 | 26 | 2 |
| Quotes | median_execute | adaptive | 60 | 3.364 | 322.885 | 27 | 3 |
| Quotes | median_execute | recursive | 60 | 4.232 | 495.257 | 7 | 3 |
| Quotes | geomean_total | static | 60 | 3.385 | 316.873 | 28 | 3 |
| Quotes | geomean_total | adaptive | 60 | 3.376 | 324.490 | 26 | 3 |
| Quotes | geomean_total | recursive | 60 | 4.241 | 495.726 | 6 | 3 |

## Median vs Geomean Winner Flips

The aggregate dataset-level winner is stable, but a few individual pattern-level winners change between `median_execute` and `geomean_total`. These flips are useful indicators of close races or slightly different run-to-run stability.

| dataset | pattern_name | complexity | specificity | rows_matched | median_execute_winner | geomean_total_winner | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DNA shared recursive-safe | any_len001 | wildcard_only | all_rows | 50421 | recursive | adaptive | %_% |
| DNA shared recursive-safe | gap_len032 | single_gapped_segment | selective | 25 | adaptive | static | %GGGCAC____________________GGTGTC% |
| DNA full static/adaptive | any_len001 | wildcard_only | all_rows | 50421 | static | adaptive | %_% |
| DNA full static/adaptive | gap_len032 | single_gapped_segment | selective | 25 | adaptive | static | %GGGCAC____________________GGTGTC% |
| Quotes | ordered_selective | ordered_segments | selective | 1 | adaptive | static | %coffee%obituaries% |

## Complexity and Specificity Summary

| dataset | complexity | specificity | patterns | median_static_wins | median_adaptive_wins | median_recursive_wins | geomean_static_wins | geomean_adaptive_wins | geomean_recursive_wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DNA full static/adaptive | complex_multi_percent | broad | 5 | 5 | 0 | 0 | 5 | 0 | 0 |
| DNA full static/adaptive | complex_multi_percent | medium | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| DNA full static/adaptive | complex_multi_percent | selective | 2 | 2 | 0 | 0 | 2 | 0 | 0 |
| DNA full static/adaptive | ordered_segments | selective | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| DNA full static/adaptive | single_gapped_segment | all_rows | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| DNA full static/adaptive | single_gapped_segment | broad | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| DNA full static/adaptive | single_gapped_segment | medium | 2 | 0 | 2 | 0 | 0 | 2 | 0 |
| DNA full static/adaptive | single_gapped_segment | selective | 5 | 4 | 1 | 0 | 5 | 0 | 0 |
| DNA full static/adaptive | single_gapped_segment | zero_match | 3 | 3 | 0 | 0 | 3 | 0 | 0 |
| DNA full static/adaptive | wildcard_only | all_rows | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
| DNA shared recursive-safe | complex_multi_percent | broad | 5 | 5 | 0 | 0 | 5 | 0 | 0 |
| DNA shared recursive-safe | complex_multi_percent | selective | 2 | 2 | 0 | 0 | 2 | 0 | 0 |
| DNA shared recursive-safe | ordered_segments | selective | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| DNA shared recursive-safe | single_gapped_segment | all_rows | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| DNA shared recursive-safe | single_gapped_segment | broad | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| DNA shared recursive-safe | single_gapped_segment | medium | 2 | 0 | 2 | 0 | 0 | 2 | 0 |
| DNA shared recursive-safe | single_gapped_segment | selective | 5 | 4 | 1 | 0 | 5 | 0 | 0 |
| DNA shared recursive-safe | single_gapped_segment | zero_match | 3 | 2 | 0 | 1 | 2 | 0 | 1 |
| DNA shared recursive-safe | wildcard_only | all_rows | 1 | 0 | 0 | 1 | 0 | 1 | 0 |
| Quotes | complex_multi_percent | broad | 2 | 1 | 1 | 0 | 1 | 1 | 0 |
| Quotes | complex_multi_percent | medium | 3 | 1 | 2 | 0 | 1 | 2 | 0 |
| Quotes | complex_multi_percent | selective | 3 | 2 | 0 | 1 | 2 | 0 | 1 |
| Quotes | complex_multi_percent | zero_match | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| Quotes | ordered_segments | selective | 3 | 1 | 2 | 0 | 2 | 1 | 0 |
| Quotes | simple_literal | selective | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| Quotes | single_gapped_segment | broad | 2 | 1 | 1 | 0 | 1 | 1 | 0 |
| Quotes | single_gapped_segment | medium | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| Quotes | single_gapped_segment | selective | 3 | 1 | 1 | 1 | 1 | 1 | 1 |
| Quotes | wildcard_only | all_rows | 1 | 0 | 1 | 0 | 0 | 1 | 0 |

## Pattern-Level Winner Summary

### DNA shared recursive-safe

Median winner counts: `adaptive`=3, `recursive`=3, `static`=15
Geomean-total winner counts: `adaptive`=3, `recursive`=2, `static`=16

| pattern_name | complexity | specificity | rows_matched | median_execute_winner | median_execute_second_best_ratio | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| interleaved_with_bigger | single_gapped_segment | selective | 297 | static | 1.87x | %C_G_C_G_CGC_G_C_G% |
| sparse_len004 | single_gapped_segment | broad | 50407 | static | 1.33x | %C__G% |
| many_common_short_segments | complex_multi_percent | broad | 50420 | static | 1.30x | %A_%C_%G_%T_% |
| anchored_suffix_gap | single_gapped_segment | zero_match | 0 | static | 1.20x | %GGGCAC____________________GGTGTC |
| anchored_prefix_gap | single_gapped_segment | zero_match | 0 | recursive | 1.17x | AAGC________ACCG% |
| anchored_prefix_exact_segments | ordered_segments | selective | 55 | recursive | 1.13x | CGCA%GAGA%CGGG% |
| ordered_mixed_segments | complex_multi_percent | broad | 27590 | static | 1.11x | %AAGC_%GGGA_%ACCG% |
| selective_spaced_segments | complex_multi_percent | broad | 35019 | static | 1.09x | %TGCA_%GAGT_%GGGC% |
| dna_multi_percent_common4 | complex_multi_percent | selective | 494 | static | 1.07x | %CACA_GATC%AGCA__CACC%GACG_GCGG%GAGA_GCTG% |
| dna_multi_percent_medium4 | complex_multi_percent | selective | 42 | static | 1.07x | %CCTA_TGAG%GTGG__GCAG%GCTT_CGAG%GAGA_TCCG% |

### DNA full static/adaptive

Median winner counts: `adaptive`=3, `static`=19
Geomean-total winner counts: `adaptive`=3, `static`=19

| pattern_name | complexity | specificity | rows_matched | median_execute_winner | median_execute_second_best_ratio | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| interleaved_with_bigger | single_gapped_segment | selective | 297 | static | 1.87x | %C_G_C_G_CGC_G_C_G% |
| many_common_short_segments | complex_multi_percent | broad | 50420 | static | 1.85x | %A_%C_%G_%T_% |
| anchored_prefix_gap | single_gapped_segment | zero_match | 0 | static | 1.51x | AAGC________ACCG% |
| sparse_len004 | single_gapped_segment | broad | 50407 | static | 1.33x | %C__G% |
| anchored_suffix_gap | single_gapped_segment | zero_match | 0 | static | 1.20x | %GGGCAC____________________GGTGTC |
| anchored_prefix_exact_segments | ordered_segments | selective | 55 | static | 1.12x | CGCA%GAGA%CGGG% |
| ordered_mixed_segments | complex_multi_percent | broad | 27590 | static | 1.11x | %AAGC_%GGGA_%ACCG% |
| dna_multi_percent_common4 | complex_multi_percent | selective | 494 | static | 1.11x | %CACA_GATC%AGCA__CACC%GACG_GCGG%GAGA_GCTG% |
| dna_multi_percent_medium4 | complex_multi_percent | selective | 42 | static | 1.09x | %CCTA_TGAG%GTGG__GCAG%GCTT_CGAG%GAGA_TCCG% |
| selective_spaced_segments | complex_multi_percent | broad | 35019 | static | 1.09x | %TGCA_%GAGT_%GGGC% |

### Quotes

Median winner counts: `adaptive`=10, `recursive`=2, `static`=8
Geomean-total winner counts: `adaptive`=9, `recursive`=2, `static`=9

| pattern_name | complexity | specificity | rows_matched | median_execute_winner | median_execute_second_best_ratio | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| anchored_prefix_gap | single_gapped_segment | selective | 1 | recursive | 1.24x | Age is a________________over mat% |
| quote_learning_gap_multi | complex_multi_percent | selective | 3 | recursive | 1.19x | %A_yone%learning%young%m_nd% |
| wild_short | single_gapped_segment | broad | 75409 | adaptive | 1.10x | %t_% |
| wild_any | wildcard_only | all_rows | 75966 | adaptive | 1.09x | %_% |
| anchored_prefix_exact | simple_literal | selective | 1 | adaptive | 1.05x | Age is an issue% |
| quote_relationship_multi5 | complex_multi_percent | selective | 4 | static | 1.03x | %f_m_l_%relationship%jealousy%anxiety%posturing% |
| quote_multi_common5_inner_underscore | complex_multi_percent | medium | 2017 | static | 1.02x | %t_e%of%a_d%to%in% |
| gap_32 | single_gapped_segment | selective | 1 | static | 1.02x | %Age is a________________over mat% |
| wild_4 | single_gapped_segment | broad | 58785 | static | 1.02x | %t__ % |
| multi_gap_common | complex_multi_percent | broad | 9937 | static | 1.01x | %the_%of_%and_% |

## Edge Cases

### static_bad_vs_adaptive

| dataset | algorithm | pattern_name | ratio | complexity | specificity | rows_matched | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Quotes | NaiveVectorizedV2 | wild_any | 1.35x | wildcard_only | all_rows | 75966 | %_% |
| Quotes | NaiveVectorizedV2 | anchored_prefix_gap | 1.26x | single_gapped_segment | selective | 1 | Age is a________________over mat% |
| Quotes | NaiveVectorizedV2 | wild_short | 1.23x | single_gapped_segment | broad | 75409 | %t_% |
| Quotes | NaiveVectorizedV2 | anchored_prefix_exact | 1.22x | simple_literal | selective | 1 | Age is an issue% |
| DNA | NaiveVectorizedV2 | interleaved_common | 1.13x | single_gapped_segment | medium | 3085 | %C_G_C_G_C_G_C_G% |
| DNA | StdSearch | gap_len016 | 1.12x | single_gapped_segment | medium | 538 | %AAGC________ACCG% |
| DNA | NaiveVectorizedV2 | sparse_len004 | 1.11x | single_gapped_segment | broad | 50407 | %C__G% |
| DNA | NaiveVectorizedV2 | one_fixed_len002 | 1.11x | single_gapped_segment | all_rows | 50421 | %G_% |

### adaptive_bad_vs_static

| dataset | algorithm | pattern_name | ratio | complexity | specificity | rows_matched | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DNA | StdSearch | interleaved_with_bigger | 2.78x | single_gapped_segment | selective | 297 | %C_G_C_G_CGC_G_C_G% |
| DNA | NaiveVectorizedV2 | interleaved_with_bigger | 2.24x | single_gapped_segment | selective | 297 | %C_G_C_G_CGC_G_C_G% |
| DNA | NaiveVectorizedV2 | many_common_short_segments | 2.13x | complex_multi_percent | broad | 50420 | %A_%C_%G_%T_% |
| DNA | StdSearch | many_common_short_segments | 1.84x | complex_multi_percent | broad | 50420 | %A_%C_%G_%T_% |
| DNA | StdSearch | sparse_len004 | 1.73x | single_gapped_segment | broad | 50407 | %C__G% |
| DNA | StdSearch | anchored_prefix_gap | 1.68x | single_gapped_segment | zero_match | 0 | AAGC________ACCG% |
| DNA | NaiveVectorizedV2Wildcard | many_common_short_segments | 1.61x | complex_multi_percent | broad | 50420 | %A_%C_%G_%T_% |
| DNA | NaiveVectorizedV2Wildcard | anchored_prefix_gap | 1.57x | single_gapped_segment | zero_match | 0 | AAGC________ACCG% |

### recursive_bad_vs_best_segment

| dataset | algorithm | pattern_name | ratio | complexity | specificity | rows_matched | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DNA | StdSearch | anchored_suffix_gap | 241.38x | single_gapped_segment | zero_match | 0 | %GGGCAC____________________GGTGTC |
| DNA | NaiveVectorizedV2 | anchored_suffix_gap | 191.21x | single_gapped_segment | zero_match | 0 | %GGGCAC____________________GGTGTC |
| DNA | NaiveVectorizedV2Wildcard | ordered_wild_segments | 7.98x | complex_multi_percent | broad | 30963 | %CGC_%GAGA_%CGGG_% |
| DNA | NaiveVectorizedV2 | ordered_wild_segments | 7.23x | complex_multi_percent | broad | 30963 | %CGC_%GAGA_%CGGG_% |
| DNA | NaiveVectorizedV2 | ordered_mixed_segments | 5.77x | complex_multi_percent | broad | 27590 | %AAGC_%GGGA_%ACCG% |
| DNA | NaiveVectorizedV2Wildcard | ordered_mixed_segments | 5.66x | complex_multi_percent | broad | 27590 | %AAGC_%GGGA_%ACCG% |
| DNA | StdSearch | ordered_wild_segments | 5.10x | complex_multi_percent | broad | 30963 | %CGC_%GAGA_%CGGG_% |
| DNA | StdSearch | interleaved_with_bigger | 4.96x | single_gapped_segment | selective | 297 | %C_G_C_G_CGC_G_C_G% |

### recursive_good_vs_best_segment

| dataset | algorithm | pattern_name | ratio | complexity | specificity | rows_matched | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Quotes | StdSearch | quote_learning_gap_multi | 1.81x | complex_multi_percent | selective | 3 | %A_yone%learning%young%m_nd% |
| DNA | NaiveVectorizedV2 | anchored_prefix_gap | 1.52x | single_gapped_segment | zero_match | 0 | AAGC________ACCG% |
| DNA | NaiveVectorizedV2 | anchored_prefix_exact_segments | 1.43x | ordered_segments | selective | 55 | CGCA%GAGA%CGGG% |
| Quotes | NaiveVectorizedV2 | anchored_prefix_gap | 1.39x | single_gapped_segment | selective | 1 | Age is a________________over mat% |
| Quotes | StdSearch | anchored_prefix_gap | 1.33x | single_gapped_segment | selective | 1 | Age is a________________over mat% |
| DNA | NaiveVectorizedV2Wildcard | anchored_prefix_gap | 1.18x | single_gapped_segment | zero_match | 0 | AAGC________ACCG% |
| DNA | StdSearch | anchored_prefix_exact_segments | 1.05x | ordered_segments | selective | 55 | CGCA%GAGA%CGGG% |
| DNA | NaiveVectorizedV2Wildcard | anchored_suffix_gap | 1.04x | single_gapped_segment | zero_match | 0 | %GGGCAC____________________GGTGTC |

## Interpretation

The median-execute and geomean-total metrics mostly agree on the qualitative outcome, but they emphasize different aspects. Median execute isolates verifier time for a typical measured iteration, while geomean total also includes candidate preparation and dampens the effect of very large outliers. Because all new suites use full scans with the `none` index, candidate preparation is close to zero; differences between the two metrics are therefore small and mainly reflect run-to-run dispersion.

For DNA shared recursive-safe under `median_execute`, the lowest aggregate geomean is `static` (12.877ms), while the largest raw win count is `static` (45 groups).
For DNA shared recursive-safe under `geomean_total`, the lowest aggregate geomean is `static` (12.991ms), while the largest raw win count is `static` (45 groups).
For DNA full static/adaptive under `median_execute`, the lowest aggregate geomean is `static` (13.890ms), while the largest raw win count is `static` (55 groups).
For DNA full static/adaptive under `geomean_total`, the lowest aggregate geomean is `static` (14.008ms), while the largest raw win count is `static` (54 groups).
For Quotes under `median_execute`, the lowest aggregate geomean is `adaptive` (3.364ms), while the largest raw win count is `adaptive` (27 groups).
For Quotes under `geomean_total`, the lowest aggregate geomean is `adaptive` (3.376ms), while the largest raw win count is `static` (28 groups).

DNA and quotes behave differently because their alphabets and row lengths differ. DNA has a four-symbol alphabet, so one-character anchors are extremely non-selective; recursive backtracking is therefore fragile when several `%` operators are followed by one-base literals. Quotes have shorter rows and word-like fragments, so recursive remains usable on this benchmark even when it loses on common-word multi-segment cases.

Static is strongest when one literal inside a fixed-width segment is much more selective than other anchors. It can jump directly to that anchor and keep scanning it. The current adaptive matcher starts from the same best anchor, then switches to the first failed literal anchor during verification. This can help if the failed anchor is a better future probe, but it can also regress by moving away from the static anchor. Recursive is weakest when common early `%` branches create many equivalent suffix failures.

Largest static-vs-adaptive losses in these runs: `Quotes:wild_any` 1.35x; `Quotes:anchored_prefix_gap` 1.26x; `Quotes:wild_short` 1.23x.
Largest adaptive-vs-static losses in these runs: `DNA:interleaved_with_bigger` 2.78x; `DNA:interleaved_with_bigger` 2.24x; `DNA:many_common_short_segments` 2.13x.
Largest recursive-vs-segment losses in the safe comparison set: `DNA:anchored_suffix_gap` 241.38x; `DNA:anchored_suffix_gap` 191.21x; `DNA:ordered_wild_segments` 7.98x. The excluded DNA edge case is much worse and is documented separately in `RECURSIVE_MATCHER_EDGECASE.md`.

## Thesis-Oriented Takeaways

1. The generic matcher is not a single universally optimal component. Its best strategy depends on pattern structure and alphabet statistics. This supports evaluating LIKE engines on structured pattern families rather than only on random substrings or pure contains/prefix cases.
2. Segment anchoring is valuable on DNA because the alphabet is small and one-character anchors are weak. Static anchoring often wins by choosing the longest fixed literal inside a segment, avoiding repeated probes of common bases.
3. Best-start adaptive is not automatically better than first-start adaptive. The before/after comparison shows that starting from the static anchor and then switching normally can regress, especially when the first failed literal is less selective than the best anchor.
4. Recursive matching is useful as a correctness reference and can be competitive on simple anchored/selective patterns. However, it should not be used as the main verifier for adversarial multi-`%` patterns unless memoization is added.
5. Specificity interacts with complexity. Selective long fragments make static attractive; broad patterns with many common fragments can favor adaptive or expose recursive blow-ups; zero-match suffix cases are especially dangerous for recursive because failure happens only after exploring many prefixes.
6. Median and geomean agree at the aggregate level in these full-scan runs, which strengthens the conclusion. Where they differ at pattern level, the differences are small enough to treat as close races or stability effects rather than evidence for a different default.

## Generated Tables

- `tables/aggregate_by_matcher.csv`
- `tables/pattern_summary.csv`
- `tables/complexity_specificity_summary.csv`
- `tables/metric_winner_flips.csv`
- `tables/edge_cases.csv`
