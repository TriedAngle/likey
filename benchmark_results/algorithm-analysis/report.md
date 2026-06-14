# Algorithm Analysis Tables

Generated from UTF-8, full-scan, static-matcher rows only. FFTSTR, DNA2, and index rows are excluded.

## Suite Winners

| suite | groups | median_execute winner | ms | geomean_total winner | ms | p90_total winner | ms | agree |
|---|---|---|---|---|---|---|---|---|
| DNA exact/underscore | 16 | NaiveAutoWildcard | 8.671 | NaiveAutoWildcard | 8.919 | NaiveAvx512V2Wildcard | 9.667 | False |
| JOB cast_info.note | 3 | PairHorspool | 64.768 | PairHorspool | 65.470 | PairHorspool | 69.051 | True |
| JOB keyword.keyword | 1 | PairHorspool | 1.597 | PairHorspool | 1.599 | PairHorspool | 1.634 | True |
| JOB movie_companies.note | 16 | PairHorspool | 15.963 | PairHorspool | 16.050 | PairHorspool | 16.455 | True |
| JOB movie_info.info | 8 | NaiveVectorizedV2Wildcard | 48.526 | NaiveVectorizedV2Wildcard | 48.699 | NaiveVectorizedV2Wildcard | 49.784 | True |
| JOB name.name | 13 | PairHorspool | 34.884 | PairHorspool | 34.980 | PairHorspool | 35.697 | True |
| JOB title.title | 18 | PairHorspool | 19.239 | PairHorspool | 19.300 | PairHorspool | 19.884 | True |
| Quotes exact/underscore | 14 | NaiveAutoWildcard | 1.348 | NaiveAutoWildcard | 1.355 | NaiveAutoWildcard | 1.389 | True |

## Common Algorithm Ranking

| algorithm | class | suite-balanced geomean execute ms | relative | suite-balanced p90 total ms | wins | worst slowdown | worst case |
|---|---|---|---|---|---|---|---|
| PairHorspool | non-wildcard | 13.410 | 1.000x | 13.931 | 41/89 | 5.686x | DNA exact/underscore / underscore_len004 |
| NaiveAutoWildcard | wildcard | 14.999 | 1.119x | 15.768 | 4/89 | 2.536x | JOB name.name / contains_b |
| NaiveVectorizedV2Wildcard | wildcard | 15.022 | 1.120x | 15.520 | 11/89 | 2.514x | JOB name.name / contains_b |
| NaiveAvx2V2Wildcard | wildcard | 15.216 | 1.135x | 16.016 | 3/89 | 2.575x | JOB name.name / contains_b |
| NaiveAuto | non-wildcard | 15.743 | 1.174x | 16.216 | 4/89 | 5.337x | DNA exact/underscore / underscore_len004 |
| NaiveVectorizedV2 | non-wildcard | 15.808 | 1.179x | 16.355 | 0/89 | 6.139x | DNA exact/underscore / underscore_len004 |
| NaiveAvx2V2 | non-wildcard | 15.947 | 1.189x | 16.680 | 0/89 | 5.621x | DNA exact/underscore / underscore_len004 |
| TwoWay2 | non-wildcard | 16.550 | 1.234x | 17.315 | 0/89 | 4.485x | Quotes exact/underscore / underscore_len004 |
| TwoWay3 | non-wildcard | 16.889 | 1.259x | 17.660 | 4/89 | 7.051x | DNA exact/underscore / underscore_len004 |
| NaiveWildcard | wildcard | 18.119 | 1.351x | 18.858 | 6/89 | 6.693x | DNA exact/underscore / exact_len256 |
| TwoWay | non-wildcard | 19.587 | 1.461x | 20.303 | 4/89 | 8.521x | Quotes exact/underscore / underscore_len032 |
| Utf8Kmp | non-wildcard | 20.587 | 1.535x | 21.509 | 0/89 | 9.446x | DNA exact/underscore / exact_len256 |
| LibcMemmem | non-wildcard | 20.613 | 1.537x | 21.379 | 4/89 | 3.652x | DNA exact/underscore / underscore_len004 |
| Naive | non-wildcard | 21.406 | 1.596x | 22.281 | 1/89 | 8.574x | DNA exact/underscore / exact_len256 |
| BM | non-wildcard | 22.710 | 1.693x | 23.504 | 0/89 | 15.089x | Quotes exact/underscore / underscore_len004 |
| StdSearch | non-wildcard | 26.195 | 1.953x | 27.000 | 3/89 | 9.268x | DNA exact/underscore / exact_len256 |

## Exact/Underscore Class Summary

| suite | pattern kind | non-wildcard ms | wildcard ms | wildcard/non-wildcard |
|---|---|---|---|---|
| DNA exact/underscore | exact | 21.334 | 16.715 | 0.783x |
| DNA exact/underscore | underscore | 14.243 | 11.335 | 0.796x |
| Quotes exact/underscore | exact | 2.759 | 2.120 | 0.768x |
| Quotes exact/underscore | underscore | 2.897 | 1.607 | 0.555x |

## Selectivity Top Algorithms

| selectivity | groups | rank | algorithm | geomean ms |
|---|---|---|---|---|
| very broad >=50% | 12 | 1 | NaiveAutoWildcard | 0.985 |
| very broad >=50% | 12 | 2 | NaiveVectorizedV2Wildcard | 1.065 |
| very broad >=50% | 12 | 3 | NaiveAvx2V2Wildcard | 1.077 |
| very broad >=50% | 12 | 4 | PairHorspool | 1.455 |
| very broad >=50% | 12 | 5 | LibcMemmem | 1.466 |
| broad 10-50% | 6 | 1 | PairHorspool | 21.674 |
| broad 10-50% | 6 | 2 | TwoWay | 28.636 |
| broad 10-50% | 6 | 3 | TwoWay2 | 28.679 |
| broad 10-50% | 6 | 4 | NaiveVectorizedV2 | 28.881 |
| broad 10-50% | 6 | 5 | NaiveVectorizedV2Wildcard | 29.694 |
| medium 1-10% | 19 | 1 | PairHorspool | 25.485 |
| medium 1-10% | 19 | 2 | NaiveVectorizedV2Wildcard | 28.268 |
| medium 1-10% | 19 | 3 | NaiveVectorizedV2 | 28.831 |
| medium 1-10% | 19 | 4 | NaiveAvx2V2Wildcard | 28.902 |
| medium 1-10% | 19 | 5 | NaiveAutoWildcard | 29.063 |
| selective 0.01-1% | 29 | 1 | PairHorspool | 28.479 |
| selective 0.01-1% | 29 | 2 | NaiveAutoWildcard | 32.673 |
| selective 0.01-1% | 29 | 3 | NaiveAuto | 32.767 |
| selective 0.01-1% | 29 | 4 | NaiveAvx2V2Wildcard | 32.864 |
| selective 0.01-1% | 29 | 5 | NaiveAvx2V2 | 32.868 |
| needle in haystack <0.01% | 23 | 1 | PairHorspool | 9.571 |
| needle in haystack <0.01% | 23 | 2 | NaiveAutoWildcard | 10.071 |
| needle in haystack <0.01% | 23 | 3 | NaiveAvx2V2Wildcard | 10.201 |
| needle in haystack <0.01% | 23 | 4 | NaiveAuto | 10.505 |
| needle in haystack <0.01% | 23 | 5 | NaiveVectorizedV2Wildcard | 10.525 |
