# Adaptive Best-Start Comparison

## Inputs

- Before archive: `benchmark_results/archive/matcher-multi-percent-before-best-start-adaptive_20260612_001154`
- After DNA static/adaptive: `benchmark_results/dna/matcher-multi-percent/gencode_utf8_20260612_003115`
- After DNA recursive-safe: `benchmark_results/dna/matcher-multi-percent-recursive/gencode_utf8_recursive_20260612_003255`
- After quotes: `benchmark_results/quotes/matcher-multi-percent/quotes_20260612_003351`

The code change makes the adaptive segment matcher start from `segment.best_anchor`, matching the static matcher initialization, then keeps the normal adaptive behavior of switching to the first failed literal anchor during verification.

## Adaptive Runtime Change

| comparison_set | metric | groups | geomean_ratio_after_before | geomean_speedup_before_after | median_ratio_after_before | improved_gt_5pct | regressed_gt_5pct | within_5pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DNA full static/adaptive | median_execute | 66 | 1.137 | 0.880x | 1.056 | 1 | 41 | 24 |
| DNA full static/adaptive | geomean_total | 66 | 1.149 | 0.870x | 1.060 | 1 | 37 | 28 |
| Quotes | median_execute | 60 | 1.051 | 0.952x | 1.046 | 0 | 25 | 35 |
| Quotes | geomean_total | 60 | 1.053 | 0.949x | 1.048 | 0 | 25 | 35 |

Ratios below 1.0 mean the new adaptive matcher is faster. Speedups above 1.0 mean the old runtime divided by the new runtime.

## Largest Adaptive Improvements

| comparison_set | algorithm | pattern_name | rows_matched | before_ms | after_ms | ratio_after_before | speedup | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DNA full static/adaptive | NaiveVectorizedV2 | sparse_len004 | 50407 | 3.241 | 2.936 | 0.906 | 1.10x | %C__G% |
| Quotes | NaiveVectorizedV2Wildcard | anchored_prefix_gap | 1 | 0.218 | 0.212 | 0.972 | 1.03x | Age is a________________over mat% |
| Quotes | StdSearch | anchored_prefix_gap | 1 | 0.314 | 0.306 | 0.977 | 1.02x | Age is a________________over mat% |
| DNA full static/adaptive | NaiveVectorizedV2 | interleaved_with_bigger | 297 | 140.420 | 137.833 | 0.982 | 1.02x | %C_G_C_G_CGC_G_C_G% |
| Quotes | NaiveVectorizedV2Wildcard | anchored_prefix_exact | 1 | 0.199 | 0.198 | 0.996 | 1.00x | Age is an issue% |
| Quotes | NaiveVectorizedV2Wildcard | wild_short | 75409 | 1.370 | 1.365 | 0.996 | 1.00x | %t_% |
| DNA full static/adaptive | StdSearch | anchored_suffix_gap | 0 | 0.277 | 0.276 | 0.996 | 1.00x | %GGGCAC____________________GGTGTC |
| DNA full static/adaptive | StdSearch | interleaved_with_bigger | 297 | 225.724 | 225.171 | 0.998 | 1.00x | %C_G_C_G_CGC_G_C_G% |
| Quotes | StdSearch | wild_any | 75966 | 0.108 | 0.108 | 0.999 | 1.00x | %_% |
| DNA full static/adaptive | NaiveVectorizedV2 | any_len001 | 50421 | 0.074 | 0.074 | 1.000 | 1.00x | %_% |
| Quotes | NaiveVectorizedV2 | wild_any | 75966 | 0.110 | 0.110 | 1.000 | 1.00x | %_% |
| Quotes | NaiveVectorizedV2 | quote_learning_gap_multi | 3 | 3.841 | 3.851 | 1.002 | 1.00x | %A_yone%learning%young%m_nd% |

## Largest Adaptive Regressions

| comparison_set | algorithm | pattern_name | rows_matched | before_ms | after_ms | ratio_after_before | speedup | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DNA full static/adaptive | NaiveVectorizedV2 | many_common_short_segments | 50420 | 3.355 | 7.594 | 2.263 | 0.44x | %A_%C_%G_%T_% |
| DNA full static/adaptive | StdSearch | sparse_len004 | 50407 | 3.842 | 7.608 | 1.981 | 0.50x | %C__G% |
| Quotes | StdSearch | quote_learning_gap_multi | 3 | 4.777 | 8.733 | 1.828 | 0.55x | %A_yone%learning%young%m_nd% |
| DNA full static/adaptive | StdSearch | many_common_short_segments | 50420 | 4.551 | 8.036 | 1.766 | 0.57x | %A_%C_%G_%T_% |
| DNA full static/adaptive | StdSearch | anchored_prefix_gap | 0 | 0.256 | 0.417 | 1.628 | 0.61x | AAGC________ACCG% |
| DNA full static/adaptive | NaiveVectorizedV2Wildcard | many_common_short_segments | 50420 | 4.422 | 7.032 | 1.590 | 0.63x | %A_%C_%G_%T_% |
| DNA full static/adaptive | NaiveVectorizedV2Wildcard | anchored_prefix_gap | 0 | 0.307 | 0.484 | 1.575 | 0.64x | AAGC________ACCG% |
| DNA full static/adaptive | NaiveVectorizedV2 | anchored_suffix_gap | 0 | 0.275 | 0.427 | 1.551 | 0.64x | %GGGCAC____________________GGTGTC |
| DNA full static/adaptive | NaiveVectorizedV2Wildcard | sparse_len004 | 50407 | 1.320 | 1.965 | 1.489 | 0.67x | %C__G% |
| DNA full static/adaptive | NaiveVectorizedV2 | anchored_prefix_gap | 0 | 0.256 | 0.379 | 1.479 | 0.68x | AAGC________ACCG% |
| DNA full static/adaptive | StdSearch | one_fixed_len002 | 50421 | 1.258 | 1.815 | 1.443 | 0.69x | %G_% |
| DNA full static/adaptive | NaiveVectorizedV2Wildcard | anchored_suffix_gap | 0 | 0.238 | 0.320 | 1.342 | 0.75x | %GGGCAC____________________GGTGTC |

## Winner Changes

| comparison_set | algorithm | pattern_name | rows_matched | before_median_execute_winner | after_median_execute_winner | before_geomean_total_winner | after_geomean_total_winner | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DNA full static/adaptive | NaiveVectorizedV2 | anchored_prefix_exact_segments | 55 | adaptive | static | adaptive | static | CGCA%GAGA%CGGG% |
| DNA full static/adaptive | NaiveVectorizedV2 | anchored_prefix_gap | 0 | adaptive | static | adaptive | static | AAGC________ACCG% |
| DNA full static/adaptive | NaiveVectorizedV2 | anchored_suffix_gap | 0 | adaptive | static | adaptive | static | %GGGCAC____________________GGTGTC |
| DNA full static/adaptive | NaiveVectorizedV2 | gap_len032 | 25 | static | adaptive | static | adaptive | %GGGCAC____________________GGTGTC% |
| DNA full static/adaptive | NaiveVectorizedV2 | many_common_short_segments | 50420 | adaptive | static | adaptive | static | %A_%C_%G_%T_% |
| DNA full static/adaptive | NaiveVectorizedV2 | ordered_exact_segments | 24211 | adaptive | static | adaptive | static | %CGCA%GAGA%CGGG% |
| DNA full static/adaptive | NaiveVectorizedV2 | ordered_mixed_segments | 27590 | adaptive | static | adaptive | static | %AAGC_%GGGA_%ACCG% |
| DNA full static/adaptive | NaiveVectorizedV2 | selective_spaced_segments | 35019 | adaptive | static | adaptive | static | %TGCA_%GAGT_%GGGC% |
| DNA full static/adaptive | NaiveVectorizedV2Wildcard | anchored_prefix_exact_segments | 55 | adaptive | static | adaptive | static | CGCA%GAGA%CGGG% |
| DNA full static/adaptive | NaiveVectorizedV2Wildcard | any_len001 | 50421 | static | static | static | adaptive | %_% |
| DNA full static/adaptive | StdSearch | anchored_prefix_gap | 0 | static | static | adaptive | static | AAGC________ACCG% |
| DNA full static/adaptive | StdSearch | dna_common_gap_multi_percent | 1586 | adaptive | static | adaptive | static | %A_%C_%G_%T_%CCTA%CAAG__GGGA% |
| DNA full static/adaptive | StdSearch | one_fixed_len002 | 50421 | adaptive | static | adaptive | static | %G_% |
| DNA full static/adaptive | StdSearch | sparse_len004 | 50407 | adaptive | static | adaptive | static | %C__G% |
| Quotes | NaiveVectorizedV2 | common_first_selective_later | 167 | adaptive | static | adaptive | static | %the%relationship with% |
| Quotes | NaiveVectorizedV2 | gap_16 | 1 | static | static | adaptive | static | %ge i________ue o% |
| Quotes | NaiveVectorizedV2 | gap_32 | 1 | static | recursive | recursive | static | %Age is a________________over mat% |
| Quotes | NaiveVectorizedV2 | multi_gap_common | 9937 | adaptive | static | adaptive | static | %the_%of_%and_% |
| Quotes | NaiveVectorizedV2 | multi_gap_selective | 0 | recursive | static | recursive | static | %young_%learning_%mind% |
| Quotes | NaiveVectorizedV2 | ordered_common | 15174 | adaptive | static | adaptive | static | %the%of%the% |
| Quotes | NaiveVectorizedV2 | ordered_selective | 1 | recursive | static | recursive | static | %coffee%obituaries% |
| Quotes | NaiveVectorizedV2 | quote_learning_gap_multi | 3 | static | adaptive | static | adaptive | %A_yone%learning%young%m_nd% |
| Quotes | NaiveVectorizedV2 | quote_multi_common4 | 3636 | adaptive | static | adaptive | static | %the_%of_%and_%to_% |
| Quotes | NaiveVectorizedV2 | quote_multi_common5 | 1747 | adaptive | static | adaptive | static | %the_%of_%and_%to_%in_% |
| Quotes | NaiveVectorizedV2 | selective_first_common_later | 236 | adaptive | static | adaptive | static | %relationship with%the% |
| Quotes | NaiveVectorizedV2 | wild_4 | 58785 | recursive | static | recursive | static | %t__ % |
| Quotes | NaiveVectorizedV2Wildcard | anchored_prefix_exact | 1 | recursive | adaptive | recursive | adaptive | Age is an issue% |
| Quotes | NaiveVectorizedV2Wildcard | gap_16 | 1 | recursive | static | recursive | static | %ge i________ue o% |
| Quotes | NaiveVectorizedV2Wildcard | gap_32 | 1 | recursive | static | recursive | static | %Age is a________________over mat% |
| Quotes | NaiveVectorizedV2Wildcard | gap_8 | 1497 | recursive | adaptive | recursive | adaptive | %le____ng% |
| Quotes | NaiveVectorizedV2Wildcard | multi_gap_common | 9937 | adaptive | static | adaptive | static | %the_%of_%and_% |
| Quotes | NaiveVectorizedV2Wildcard | multi_gap_selective | 0 | recursive | adaptive | recursive | adaptive | %young_%learning_%mind% |
| Quotes | NaiveVectorizedV2Wildcard | ordered_selective | 1 | recursive | adaptive | adaptive | adaptive | %coffee%obituaries% |
| Quotes | NaiveVectorizedV2Wildcard | quote_learning_gap_multi | 3 | recursive | static | recursive | adaptive | %A_yone%learning%young%m_nd% |
| Quotes | NaiveVectorizedV2Wildcard | quote_multi_common4 | 3636 | adaptive | adaptive | adaptive | static | %the_%of_%and_%to_% |
| Quotes | NaiveVectorizedV2Wildcard | quote_multi_common5_inner_underscore | 2017 | adaptive | static | adaptive | adaptive | %t_e%of%a_d%to%in% |
| Quotes | NaiveVectorizedV2Wildcard | quote_relationship_multi5 | 4 | static | adaptive | static | adaptive | %f_m_l_%relationship%jealousy%anxiety%posturing% |
| Quotes | NaiveVectorizedV2Wildcard | selective_first_common_later | 236 | adaptive | static | adaptive | adaptive | %relationship with%the% |
| Quotes | NaiveVectorizedV2Wildcard | wild_4 | 58785 | static | adaptive | recursive | static | %t__ % |
| Quotes | NaiveVectorizedV2Wildcard | wild_any | 75966 | recursive | recursive | recursive | static | %_% |
| Quotes | NaiveVectorizedV2Wildcard | wild_short | 75409 | recursive | adaptive | recursive | adaptive | %t_% |
| Quotes | StdSearch | anchored_prefix_exact | 1 | static | static | static | recursive | Age is an issue% |
| Quotes | StdSearch | common_first_selective_later | 167 | static | adaptive | static | adaptive | %the%relationship with% |
| Quotes | StdSearch | gap_16 | 1 | recursive | adaptive | recursive | adaptive | %ge i________ue o% |
| Quotes | StdSearch | gap_8 | 1497 | static | adaptive | static | adaptive | %le____ng% |
| Quotes | StdSearch | multi_gap_common | 9937 | adaptive | static | adaptive | static | %the_%of_%and_% |
| Quotes | StdSearch | multi_gap_selective | 0 | static | adaptive | static | adaptive | %young_%learning_%mind% |
| Quotes | StdSearch | ordered_common | 15174 | adaptive | adaptive | static | adaptive | %the%of%the% |
| Quotes | StdSearch | ordered_selective | 1 | static | adaptive | static | static | %coffee%obituaries% |
| Quotes | StdSearch | quote_multi_common5 | 1747 | static | adaptive | static | adaptive | %the_%of_%and_%to_%in_% |
| Quotes | StdSearch | quote_multi_common5_inner_underscore | 2017 | static | adaptive | static | static | %t_e%of%a_d%to%in% |
| Quotes | StdSearch | wild_4 | 58785 | recursive | static | recursive | static | %t__ % |
| Quotes | StdSearch | wild_any | 75966 | adaptive | adaptive | static | adaptive | %_% |

## Interpretation

For DNA full static/adaptive, adaptive median execution regressed: geomean after/before ratio 1.137, speedup 0.880x.
For Quotes, adaptive median execution regressed: geomean after/before ratio 1.051, speedup 0.952x.
Winner changes occurred in 53 comparable groups. Because static and recursive code paths did not change, most winner changes indicate adaptive moved enough to cross an existing close boundary rather than a broad behavior change across all patterns.
Largest adaptive improvements: `DNA full static/adaptive:sparse_len004` 1.10x; `Quotes:anchored_prefix_gap` 1.03x; `Quotes:anchored_prefix_gap` 1.02x.
Largest adaptive regressions: `DNA full static/adaptive:many_common_short_segments` 2.26x slower; `DNA full static/adaptive:sparse_len004` 1.98x slower; `Quotes:quote_learning_gap_multi` 1.83x slower.

## Tables

- `tables/adaptive_before_after.csv`
- `tables/adaptive_aggregate_changes.csv`
- `tables/winner_changes.csv`
