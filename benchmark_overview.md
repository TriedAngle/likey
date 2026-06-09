# Benchmark Overview

| Benchmark run | Intention | Raw row count | Cross-product multiplier |
|---|---|---:|---:|
| `dna/exact-vs-underscore-algorithms/gencode` | Compare full-scan LIKE algorithms on exact and underscore-heavy GENCODE DNA patterns, split from the index cross product. | 54,107 | 592 |
| `dna/exact-vs-underscore-indexes/gencode` | Measure representative UTF-8 and DNA2 algorithms across all indexes for exact and underscore-heavy GENCODE DNA patterns. | 54,107 | 560 |
| `dna/matcher-comparison/gencode` | Compare static, adaptive, and recursive matcher engines on GENCODE DNA patterns while reusing the same data and indexes. | 54,107 | 1,425 |
| `dna/index-comparison/gencode` | Compare full-scan and candidate index performance for prefix/equality-friendly GENCODE DNA patterns. | 54,107 | 210 |
| `dna/fsst-index-memmem/gencode` | Compare UTF-8 vs FSST storage using fixed `LibcMemmem` across candidate indexes on GENCODE DNA. | 54,107 | 60 |
| `dna/n-handling/n_sparse` | Stress deterministic sparse-N DNA data, comparing UTF-8 and DNA2 N-aware algorithms with and without qgram. | 2,048 | 182 |
| `quotes/exact-vs-underscore/quotes` | Compare full-scan LIKE algorithms on exact and underscore-heavy quote patterns. | 75,966 | 2,100 |
| `quotes/algorithm-cases/quotes` | Stress quote-specific algorithm cases for BM, TwoWay variants, PairHorspool, and Naive V2 prefilter variants. | 75,966 | 1,800 |
| `quotes/matcher-comparison/quotes` | Compare static, adaptive, and recursive matcher engines on quote patterns using wildcard-capable algorithms. | 75,966 | 630 |
| `quotes/index-comparison/quotes` | Compare full-scan and candidate index performance for prefix/equality-friendly quote patterns. | 75,966 | 150 |
| `quotes/fsst-index-memmem/quotes` | Compare UTF-8 vs FSST storage using fixed `LibcMemmem` across candidate indexes on quotes. | 75,966 | 60 |
| `fftstr/abab-wildcards/abab` | Compare scalar/SIMD literal baselines, TwoWay variants, glibc-style exact search, `FftStr1`, and `FftstrV2` on artificial `abab...` wildcard patterns. | 1,000 | 1,170 |
| `job/index-consistency/cast_info_note` | Measure index performance on `cast_info.note` LIKE predicates across all configured indexes and several verifier algorithms. | 23,313,199 | 75 |
| `job/algorithm-comparison/cast_info_note` | Compare full-scan algorithm performance for LIKE predicates on `cast_info.note`. | 23,313,199 | 57 |
| `job/matcher-comparison/cast_info_note` | Compare static, adaptive, and recursive matcher engines for LIKE predicates on `cast_info.note`. | 23,313,199 | 135 |
| `job/index-memmem/cast_info_note` | Measure candidate-index performance for LIKE predicates on `cast_info.note` using fixed `LibcMemmem`. | 23,313,199 | 15 |
| `job/fsst-index-memmem/cast_info_note` | Compare UTF-8 vs FSST storage for LIKE predicates on `cast_info.note` using fixed `LibcMemmem` and all candidate indexes. | 23,313,199 | 30 |
| `job/index-consistency/keyword_keyword` | Measure index performance on `keyword.keyword` LIKE predicates across all configured indexes and several verifier algorithms. | 134,170 | 25 |
| `job/algorithm-comparison/keyword_keyword` | Compare full-scan algorithm performance for LIKE predicates on `keyword.keyword`. | 134,170 | 19 |
| `job/matcher-comparison/keyword_keyword` | Compare static, adaptive, and recursive matcher engines for LIKE predicates on `keyword.keyword`. | 134,170 | 45 |
| `job/index-memmem/keyword_keyword` | Measure candidate-index performance for LIKE predicates on `keyword.keyword` using fixed `LibcMemmem`. | 134,170 | 5 |
| `job/fsst-index-memmem/keyword_keyword` | Compare UTF-8 vs FSST storage for LIKE predicates on `keyword.keyword` using fixed `LibcMemmem` and all candidate indexes. | 134,170 | 10 |
| `job/index-consistency/movie_companies_note` | Measure index performance on `movie_companies.note` LIKE predicates across all configured indexes and several verifier algorithms. | 2,609,129 | 400 |
| `job/algorithm-comparison/movie_companies_note` | Compare full-scan algorithm performance for LIKE predicates on `movie_companies.note`. | 2,609,129 | 304 |
| `job/matcher-comparison/movie_companies_note` | Compare static, adaptive, and recursive matcher engines for LIKE predicates on `movie_companies.note`. | 2,609,129 | 720 |
| `job/index-memmem/movie_companies_note` | Measure candidate-index performance for LIKE predicates on `movie_companies.note` using fixed `LibcMemmem`. | 2,609,129 | 80 |
| `job/fsst-index-memmem/movie_companies_note` | Compare UTF-8 vs FSST storage for LIKE predicates on `movie_companies.note` using fixed `LibcMemmem` and all candidate indexes. | 2,609,129 | 160 |
| `job/index-consistency/movie_info_info` | Measure index performance on `movie_info.info` LIKE predicates across all configured indexes and several verifier algorithms. | 6,674,158 | 200 |
| `job/algorithm-comparison/movie_info_info` | Compare full-scan algorithm performance for LIKE predicates on `movie_info.info`. | 6,674,158 | 152 |
| `job/matcher-comparison/movie_info_info` | Compare static, adaptive, and recursive matcher engines for LIKE predicates on `movie_info.info`. | 6,674,158 | 360 |
| `job/index-memmem/movie_info_info` | Measure candidate-index performance for LIKE predicates on `movie_info.info` using fixed `LibcMemmem`. | 6,674,158 | 40 |
| `job/fsst-index-memmem/movie_info_info` | Compare UTF-8 vs FSST storage for LIKE predicates on `movie_info.info` using fixed `LibcMemmem` and all candidate indexes. | 6,674,158 | 80 |
| `job/index-consistency/name_name` | Measure index performance on `name.name` LIKE predicates across all configured indexes and several verifier algorithms. | 4,167,491 | 325 |
| `job/algorithm-comparison/name_name` | Compare full-scan algorithm performance for LIKE predicates on `name.name`. | 4,167,491 | 247 |
| `job/matcher-comparison/name_name` | Compare static, adaptive, and recursive matcher engines for LIKE predicates on `name.name`. | 4,167,491 | 585 |
| `job/index-memmem/name_name` | Measure candidate-index performance for LIKE predicates on `name.name` using fixed `LibcMemmem`. | 4,167,491 | 65 |
| `job/fsst-index-memmem/name_name` | Compare UTF-8 vs FSST storage for LIKE predicates on `name.name` using fixed `LibcMemmem` and all candidate indexes. | 4,167,491 | 130 |
| `job/index-consistency/title_title` | Measure index performance on `title.title` LIKE predicates across all configured indexes and several verifier algorithms. | 2,528,312 | 450 |
| `job/algorithm-comparison/title_title` | Compare full-scan algorithm performance for LIKE predicates on `title.title`. | 2,528,312 | 342 |
| `job/matcher-comparison/title_title` | Compare static, adaptive, and recursive matcher engines for LIKE predicates on `title.title`. | 2,528,312 | 810 |
| `job/index-memmem/title_title` | Measure candidate-index performance for LIKE predicates on `title.title` using fixed `LibcMemmem`. | 2,528,312 | 90 |
| `job/fsst-index-memmem/title_title` | Compare UTF-8 vs FSST storage for LIKE predicates on `title.title` using fixed `LibcMemmem` and all candidate indexes. | 2,528,312 | 180 |

Raw row count is the loaded table row count reported by the benchmark results. The cross-product multiplier is enabled patterns x enabled indexes x compatible algorithms per storage x configured matcher engines. It excludes warmups and measured iteration repeats.

The completed runs in `benchmark_results/` all report `iterations: 5`. If a repeated-execution total is needed, multiply raw row count x cross-product multiplier x 5.

JOB predicates were taken from <https://github.com/gregrahn/join-order-benchmark>; the LIKE predicates were isolated from the SQL WHERE clauses.
