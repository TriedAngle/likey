# Matcher Multi-Percent Benchmark Report

## Executive Summary

- On DNA, `static` is the strongest overall matcher on the recursive-safe comparison set: aggregate geomean of median execution is 12.572ms versus 12.874ms for `adaptive` and 19.603ms for `recursive`.
- On quotes, `adaptive` is the strongest aggregate matcher by geomean of median execution (3.201ms), narrowly ahead of `static` (3.250ms), while `recursive` is slower in aggregate (3.936ms) but still wins several simple/selective patterns.
- Median and geomean produce the same aggregate winner for every dataset subset. Pattern-level flips exist, especially in quotes, but they are local rather than changing the overall conclusion.
- Complexity matters: static is favored by DNA patterns with a selective longer internal anchor; the current adaptive matcher starts from that same best anchor and then may switch to failed anchors, which can help some cases but regress others; recursive is acceptable on simple or anchored cases but fails catastrophically when multi-`%` DNA patterns create many common one-base branches.
- The excluded DNA recursive edge case is not an arbitrary timeout. A one-row repro produced a `sample` stack dominated by `LikePattern::match_from` and `naive_find_neon_v2`, and a token-walk model estimated about 5.07 billion recursive calls for that row.
- Pattern-level winner flips between median and geomean: 4 of 63 pattern summaries.

## Inputs

- DNA static/adaptive: `benchmark_results/dna/matcher-multi-percent/gencode_utf8_20260612_000949`
- DNA recursive-safe: `benchmark_results/dna/matcher-multi-percent-recursive/gencode_utf8_recursive_20260612_001101`
- Quotes: `benchmark_results/quotes/matcher-multi-percent/quotes_20260612_001154`
- All runs use full scans, 10 measured iterations, one warmup, and native CPU Rust flags recorded in each result directory.
- Generic plots are intentionally not generated; the report keeps machine-readable tables for bespoke thesis figures.

## Method and Metrics

Each comparison keeps dataset, storage, algorithm, index choice, pattern name, and pattern text fixed, then varies only the generic matcher. This is important because the literal-search algorithm can dominate absolute time. The report therefore aggregates over comparable groups rather than mixing unrelated rows.

Two metrics are reported. `median_execute` is the median verifier execution time over the 10 measured iterations and excludes compile/load time. `geomean_total` is the geometric mean of total query time from the benchmark summary and includes candidate preparation. In these full-scan `none`-index benchmarks, candidate preparation is effectively zero, so the two metrics mostly test robustness to iteration variance rather than different work.

Pattern complexity is classified from the LIKE pattern text: wildcard-only, single gapped segment, ordered segments, or complex multi-`%`. Specificity is classified from matched-row fraction: all rows, broad, medium, selective, or zero-match.

## Aggregate Results

| dataset | metric | matcher | groups | geomean_ms | sum_ms | wins | clear_wins_5pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DNA shared recursive-safe | median_execute | static | 63 | 12.572 | 2888.437 | 38 | 6 |
| DNA shared recursive-safe | median_execute | adaptive | 63 | 12.874 | 3115.930 | 15 | 6 |
| DNA shared recursive-safe | median_execute | recursive | 63 | 19.603 | 4664.935 | 10 | 9 |
| DNA shared recursive-safe | geomean_total | static | 63 | 12.656 | 2889.940 | 38 | 6 |
| DNA shared recursive-safe | geomean_total | adaptive | 63 | 12.982 | 3119.590 | 15 | 5 |
| DNA shared recursive-safe | geomean_total | recursive | 63 | 19.702 | 4668.045 | 10 | 8 |
| DNA full static/adaptive | median_execute | static | 66 | 13.563 | 3091.430 | 45 | 7 |
| DNA full static/adaptive | median_execute | adaptive | 66 | 13.879 | 3320.308 | 21 | 9 |
| DNA full static/adaptive | geomean_total | static | 66 | 13.651 | 3093.475 | 44 | 7 |
| DNA full static/adaptive | geomean_total | adaptive | 66 | 13.990 | 3324.193 | 22 | 9 |
| Quotes | median_execute | static | 60 | 3.250 | 302.720 | 19 | 0 |
| Quotes | median_execute | adaptive | 60 | 3.201 | 304.335 | 22 | 3 |
| Quotes | median_execute | recursive | 60 | 3.936 | 455.231 | 19 | 6 |
| Quotes | geomean_total | static | 60 | 3.260 | 303.123 | 18 | 0 |
| Quotes | geomean_total | adaptive | 60 | 3.206 | 304.399 | 22 | 4 |
| Quotes | geomean_total | recursive | 60 | 3.935 | 454.969 | 20 | 5 |

## Median vs Geomean Winner Flips

The aggregate dataset-level winner is stable, but a few individual pattern-level winners change between `median_execute` and `geomean_total`. These flips are useful indicators of close races or slightly different run-to-run stability.

| dataset | pattern_name | complexity | specificity | rows_matched | median_execute_winner | geomean_total_winner | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DNA full static/adaptive | many_common_short_segments | complex_multi_percent | broad | 50420 | adaptive | static | %A_%C_%G_%T_% |
| Quotes | gap_32 | single_gapped_segment | selective | 1 | static | recursive | %Age is a________________over mat% |
| Quotes | quote_multi_common5 | complex_multi_percent | medium | 1747 | static | adaptive | %the_%of_%and_%to_%in_% |
| Quotes | selective_first_common_later | ordered_segments | selective | 236 | recursive | adaptive | %relationship with%the% |

## Complexity and Specificity Summary

| dataset | complexity | specificity | patterns | median_static_wins | median_adaptive_wins | median_recursive_wins | geomean_static_wins | geomean_adaptive_wins | geomean_recursive_wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DNA full static/adaptive | complex_multi_percent | broad | 5 | 4 | 1 | 0 | 5 | 0 | 0 |
| DNA full static/adaptive | complex_multi_percent | medium | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| DNA full static/adaptive | complex_multi_percent | selective | 2 | 2 | 0 | 0 | 2 | 0 | 0 |
| DNA full static/adaptive | ordered_segments | selective | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| DNA full static/adaptive | single_gapped_segment | all_rows | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| DNA full static/adaptive | single_gapped_segment | broad | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| DNA full static/adaptive | single_gapped_segment | medium | 2 | 0 | 2 | 0 | 0 | 2 | 0 |
| DNA full static/adaptive | single_gapped_segment | selective | 5 | 5 | 0 | 0 | 5 | 0 | 0 |
| DNA full static/adaptive | single_gapped_segment | zero_match | 3 | 1 | 2 | 0 | 1 | 2 | 0 |
| DNA full static/adaptive | wildcard_only | all_rows | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| DNA shared recursive-safe | complex_multi_percent | broad | 5 | 4 | 0 | 1 | 4 | 0 | 1 |
| DNA shared recursive-safe | complex_multi_percent | selective | 2 | 2 | 0 | 0 | 2 | 0 | 0 |
| DNA shared recursive-safe | ordered_segments | selective | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| DNA shared recursive-safe | single_gapped_segment | all_rows | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| DNA shared recursive-safe | single_gapped_segment | broad | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| DNA shared recursive-safe | single_gapped_segment | medium | 2 | 0 | 2 | 0 | 0 | 2 | 0 |
| DNA shared recursive-safe | single_gapped_segment | selective | 5 | 5 | 0 | 0 | 5 | 0 | 0 |
| DNA shared recursive-safe | single_gapped_segment | zero_match | 3 | 1 | 1 | 1 | 1 | 1 | 1 |
| DNA shared recursive-safe | wildcard_only | all_rows | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| Quotes | complex_multi_percent | broad | 2 | 0 | 2 | 0 | 0 | 2 | 0 |
| Quotes | complex_multi_percent | medium | 3 | 2 | 1 | 0 | 1 | 2 | 0 |
| Quotes | complex_multi_percent | selective | 3 | 1 | 1 | 1 | 1 | 1 | 1 |
| Quotes | complex_multi_percent | zero_match | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| Quotes | ordered_segments | selective | 3 | 0 | 1 | 2 | 0 | 2 | 1 |
| Quotes | simple_literal | selective | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| Quotes | single_gapped_segment | broad | 2 | 0 | 1 | 1 | 0 | 1 | 1 |
| Quotes | single_gapped_segment | medium | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| Quotes | single_gapped_segment | selective | 3 | 1 | 0 | 2 | 0 | 0 | 3 |
| Quotes | wildcard_only | all_rows | 1 | 0 | 1 | 0 | 0 | 1 | 0 |

## Pattern-Level Winner Summary

### DNA shared recursive-safe

Median winner counts: `adaptive`=4, `recursive`=4, `static`=13
Geomean-total winner counts: `adaptive`=4, `recursive`=4, `static`=13

| pattern_name | complexity | specificity | rows_matched | median_execute_winner | median_execute_second_best_ratio | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| interleaved_with_bigger | single_gapped_segment | selective | 297 | static | 1.92x | %C_G_C_G_CGC_G_C_G% |
| anchored_prefix_gap | single_gapped_segment | zero_match | 0 | recursive | 1.21x | AAGC________ACCG% |
| anchored_prefix_exact_segments | ordered_segments | selective | 55 | recursive | 1.13x | CGCA%GAGA%CGGG% |
| one_fixed_len002 | single_gapped_segment | all_rows | 50421 | adaptive | 1.09x | %G_% |
| dna_multi_percent_common4 | complex_multi_percent | selective | 494 | static | 1.06x | %CACA_GATC%AGCA__CACC%GACG_GCGG%GAGA_GCTG% |
| anchored_suffix_gap | single_gapped_segment | zero_match | 0 | adaptive | 1.06x | %GGGCAC____________________GGTGTC |
| interleaved_common | single_gapped_segment | medium | 3085 | adaptive | 1.05x | %C_G_C_G_C_G_C_G% |
| dna_multi_percent_medium4 | complex_multi_percent | selective | 42 | static | 1.05x | %CCTA_TGAG%GTGG__GCAG%GCTT_CGAG%GAGA_TCCG% |
| sparse_len004 | single_gapped_segment | broad | 50407 | recursive | 1.05x | %C__G% |
| gap_len016 | single_gapped_segment | medium | 538 | adaptive | 1.03x | %AAGC________ACCG% |

### DNA full static/adaptive

Median winner counts: `adaptive`=8, `static`=14
Geomean-total winner counts: `adaptive`=7, `static`=15

| pattern_name | complexity | specificity | rows_matched | median_execute_winner | median_execute_second_best_ratio | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| interleaved_with_bigger | single_gapped_segment | selective | 297 | static | 1.92x | %C_G_C_G_CGC_G_C_G% |
| one_fixed_len002 | single_gapped_segment | all_rows | 50421 | adaptive | 1.09x | %G_% |
| dna_multi_percent_common4 | complex_multi_percent | selective | 494 | static | 1.08x | %CACA_GATC%AGCA__CACC%GACG_GCGG%GAGA_GCTG% |
| anchored_prefix_gap | single_gapped_segment | zero_match | 0 | adaptive | 1.07x | AAGC________ACCG% |
| dna_multi_percent_medium4 | complex_multi_percent | selective | 42 | static | 1.06x | %CCTA_TGAG%GTGG__GCAG%GCTT_CGAG%GAGA_TCCG% |
| anchored_suffix_gap | single_gapped_segment | zero_match | 0 | adaptive | 1.06x | %GGGCAC____________________GGTGTC |
| sparse_len004 | single_gapped_segment | broad | 50407 | adaptive | 1.06x | %C__G% |
| interleaved_common | single_gapped_segment | medium | 3085 | adaptive | 1.05x | %C_G_C_G_C_G_C_G% |
| anchored_prefix_exact_segments | ordered_segments | selective | 55 | adaptive | 1.05x | CGCA%GAGA%CGGG% |
| gap_len016 | single_gapped_segment | medium | 538 | adaptive | 1.03x | %AAGC________ACCG% |

### Quotes

Median winner counts: `adaptive`=8, `recursive`=8, `static`=4
Geomean-total winner counts: `adaptive`=10, `recursive`=8, `static`=2

| pattern_name | complexity | specificity | rows_matched | median_execute_winner | median_execute_second_best_ratio | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| anchored_prefix_gap | single_gapped_segment | selective | 1 | recursive | 1.29x | Age is a________________over mat% |
| wild_short | single_gapped_segment | broad | 75409 | adaptive | 1.09x | %t_% |
| anchored_prefix_exact | simple_literal | selective | 1 | adaptive | 1.05x | Age is an issue% |
| wild_any | wildcard_only | all_rows | 75966 | adaptive | 1.05x | %_% |
| quote_learning_gap_multi | complex_multi_percent | selective | 3 | recursive | 1.04x | %A_yone%learning%young%m_nd% |
| multi_gap_common | complex_multi_percent | broad | 9937 | adaptive | 1.04x | %the_%of_%and_% |
| ordered_common | complex_multi_percent | broad | 15174 | adaptive | 1.03x | %the%of%the% |
| quote_relationship_multi5 | complex_multi_percent | selective | 4 | static | 1.02x | %f_m_l_%relationship%jealousy%anxiety%posturing% |
| wild_4 | single_gapped_segment | broad | 58785 | recursive | 1.02x | %t__ % |
| ordered_selective | ordered_segments | selective | 1 | recursive | 1.01x | %coffee%obituaries% |

## Edge Cases

### static_bad_vs_adaptive

| dataset | algorithm | pattern_name | ratio | complexity | specificity | rows_matched | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Quotes | StdSearch | quote_learning_gap_multi | 1.64x | complex_multi_percent | selective | 3 | %A_yone%learning%young%m_nd% |
| DNA | NaiveVectorizedV2 | one_fixed_len002 | 1.34x | single_gapped_segment | all_rows | 50421 | %G_% |
| DNA | NaiveVectorizedV2 | anchored_prefix_gap | 1.30x | single_gapped_segment | zero_match | 0 | AAGC________ACCG% |
| Quotes | NaiveVectorizedV2 | anchored_prefix_gap | 1.29x | single_gapped_segment | selective | 1 | Age is a________________over mat% |
| DNA | NaiveVectorizedV2 | anchored_prefix_exact_segments | 1.27x | ordered_segments | selective | 55 | CGCA%GAGA%CGGG% |
| DNA | NaiveVectorizedV2 | anchored_suffix_gap | 1.24x | single_gapped_segment | zero_match | 0 | %GGGCAC____________________GGTGTC |
| Quotes | NaiveVectorizedV2 | anchored_prefix_exact | 1.23x | simple_literal | selective | 1 | Age is an issue% |
| Quotes | NaiveVectorizedV2 | wild_short | 1.21x | single_gapped_segment | broad | 75409 | %t_% |

### adaptive_bad_vs_static

| dataset | algorithm | pattern_name | ratio | complexity | specificity | rows_matched | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DNA | StdSearch | interleaved_with_bigger | 2.88x | single_gapped_segment | selective | 297 | %C_G_C_G_CGC_G_C_G% |
| DNA | NaiveVectorizedV2 | interleaved_with_bigger | 2.41x | single_gapped_segment | selective | 297 | %C_G_C_G_CGC_G_C_G% |
| Quotes | StdSearch | wild_4 | 1.43x | single_gapped_segment | broad | 58785 | %t__ % |
| Quotes | NaiveVectorizedV2 | wild_4 | 1.40x | single_gapped_segment | broad | 58785 | %t__ % |
| DNA | StdSearch | dna_multi_percent_common4 | 1.23x | complex_multi_percent | selective | 494 | %CACA_GATC%AGCA__CACC%GACG_GCGG%GAGA_GCTG% |
| DNA | StdSearch | dna_multi_percent_medium4 | 1.13x | complex_multi_percent | selective | 42 | %CCTA_TGAG%GTGG__GCAG%GCTT_CGAG%GAGA_TCCG% |
| DNA | StdSearch | anchored_prefix_exact_segments | 1.12x | ordered_segments | selective | 55 | CGCA%GAGA%CGGG% |
| DNA | StdSearch | any_len001 | 1.10x | wildcard_only | all_rows | 50421 | %_% |

### recursive_bad_vs_best_segment

| dataset | algorithm | pattern_name | ratio | complexity | specificity | rows_matched | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DNA | StdSearch | anchored_suffix_gap | 232.35x | single_gapped_segment | zero_match | 0 | %GGGCAC____________________GGTGTC |
| DNA | NaiveVectorizedV2 | anchored_suffix_gap | 205.32x | single_gapped_segment | zero_match | 0 | %GGGCAC____________________GGTGTC |
| DNA | NaiveVectorizedV2Wildcard | ordered_wild_segments | 7.66x | complex_multi_percent | broad | 30963 | %CGC_%GAGA_%CGGG_% |
| DNA | NaiveVectorizedV2 | ordered_wild_segments | 6.99x | complex_multi_percent | broad | 30963 | %CGC_%GAGA_%CGGG_% |
| DNA | NaiveVectorizedV2Wildcard | ordered_mixed_segments | 5.61x | complex_multi_percent | broad | 27590 | %AAGC_%GGGA_%ACCG% |
| DNA | NaiveVectorizedV2 | ordered_mixed_segments | 5.56x | complex_multi_percent | broad | 27590 | %AAGC_%GGGA_%ACCG% |
| DNA | StdSearch | ordered_wild_segments | 4.98x | complex_multi_percent | broad | 30963 | %CGC_%GAGA_%CGGG_% |
| DNA | StdSearch | interleaved_with_bigger | 4.84x | single_gapped_segment | selective | 297 | %C_G_C_G_CGC_G_C_G% |

### recursive_good_vs_best_segment

| dataset | algorithm | pattern_name | ratio | complexity | specificity | rows_matched | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Quotes | NaiveVectorizedV2 | anchored_prefix_gap | 1.45x | single_gapped_segment | selective | 1 | Age is a________________over mat% |
| DNA | NaiveVectorizedV2 | anchored_prefix_gap | 1.35x | single_gapped_segment | zero_match | 0 | AAGC________ACCG% |
| Quotes | StdSearch | anchored_prefix_gap | 1.32x | single_gapped_segment | selective | 1 | Age is a________________over mat% |
| DNA | NaiveVectorizedV2 | anchored_prefix_exact_segments | 1.18x | ordered_segments | selective | 55 | CGCA%GAGA%CGGG% |
| DNA | StdSearch | anchored_prefix_gap | 1.18x | single_gapped_segment | zero_match | 0 | AAGC________ACCG% |
| DNA | NaiveVectorizedV2Wildcard | sparse_len004 | 1.12x | single_gapped_segment | broad | 50407 | %C__G% |
| Quotes | NaiveVectorizedV2Wildcard | anchored_prefix_gap | 1.10x | single_gapped_segment | selective | 1 | Age is a________________over mat% |
| DNA | NaiveVectorizedV2 | sparse_len004 | 1.09x | single_gapped_segment | broad | 50407 | %C__G% |

## Interpretation

The median-execute and geomean-total metrics mostly agree on the qualitative outcome, but they emphasize different aspects. Median execute isolates verifier time for a typical measured iteration, while geomean total also includes candidate preparation and dampens the effect of very large outliers. Because all new suites use full scans with the `none` index, candidate preparation is close to zero; differences between the two metrics are therefore small and mainly reflect run-to-run dispersion.

For DNA shared recursive-safe under `median_execute`, the lowest aggregate geomean is `static` (12.572ms), while the largest raw win count is `static` (38 groups).
For DNA shared recursive-safe under `geomean_total`, the lowest aggregate geomean is `static` (12.656ms), while the largest raw win count is `static` (38 groups).
For DNA full static/adaptive under `median_execute`, the lowest aggregate geomean is `static` (13.563ms), while the largest raw win count is `static` (45 groups).
For DNA full static/adaptive under `geomean_total`, the lowest aggregate geomean is `static` (13.651ms), while the largest raw win count is `static` (44 groups).
For Quotes under `median_execute`, the lowest aggregate geomean is `adaptive` (3.201ms), while the largest raw win count is `adaptive` (22 groups).
For Quotes under `geomean_total`, the lowest aggregate geomean is `adaptive` (3.206ms), while the largest raw win count is `adaptive` (22 groups).

DNA and quotes behave differently because their alphabets and row lengths differ. DNA has a four-symbol alphabet, so one-character anchors are extremely non-selective; recursive backtracking is therefore fragile when several `%` operators are followed by one-base literals. Quotes have shorter rows and word-like fragments, so recursive remains usable on this benchmark even when it loses on common-word multi-segment cases.

Static is strongest when one literal inside a fixed-width segment is much more selective than other anchors. It can jump directly to that anchor and keep scanning it. The current adaptive matcher starts from the same best anchor, then switches to the first failed literal anchor during verification. This can help if the failed anchor is a better future probe, but it can also regress by moving away from the static anchor. Recursive is weakest when common early `%` branches create many equivalent suffix failures.

Largest static-vs-adaptive losses in these runs: `Quotes:quote_learning_gap_multi` 1.64x; `DNA:one_fixed_len002` 1.34x; `DNA:anchored_prefix_gap` 1.30x.
Largest adaptive-vs-static losses in these runs: `DNA:interleaved_with_bigger` 2.88x; `DNA:interleaved_with_bigger` 2.41x; `Quotes:wild_4` 1.43x.
Largest recursive-vs-segment losses in the safe comparison set: `DNA:anchored_suffix_gap` 232.35x; `DNA:anchored_suffix_gap` 205.32x; `DNA:ordered_wild_segments` 7.66x. The excluded DNA edge case is much worse and is documented separately in `RECURSIVE_MATCHER_EDGECASE.md`.

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
