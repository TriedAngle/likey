# Dataset Stats

Generated from prepared `data/tpch`, `data/tpcds`, and `data/job` `key,value` CSV files.

Notes:
- Lengths are UTF-8 byte lengths of the prepared `value` field.
- Each column is capped at 100,000,000 loaded value bytes; the final row is truncated to match runner `--max-total-bytes` behavior.
- Geomean length is computed over positive-length rows; empty rows still contribute to min length and row count.
- Distinct value count uses HyperLogLog; top-10 value coverage uses a bounded Misra-Gries heavy-hitter sketch and is an approximate lower bound.
- Trigram metrics use byte trigrams within each row; trigrams do not cross row boundaries.
- Pattern offset metrics use the first occurrence of the longest literal implied by the listed LIKE pattern.

Metric glossary:
- `Alphabet`: number of distinct byte values seen in the column.
- `Entropy bits/B`: Shannon entropy per byte; lower values mean more repetitive/skewed text.
- `Top byte %`: share of all bytes occupied by the most frequent byte.
- `Distinct values approx %`: estimated distinct row values divided by row count.
- `Top-10 values approx %`: approximate lower-bound share of rows covered by the 10 most frequent values.
- `Distinct trigrams`: number of distinct 3-byte sequences within rows.
- `Top trigram %`: share of all row-local trigrams occupied by the most frequent trigram.
- `Prefix3/Suffix3 distinct`: number of distinct first/last 3-byte prefixes/suffixes.
- `First offset`: byte offset of the first occurrence of the benchmark literal in matching rows.

## Length Stats

| Dataset | Column | Rows | Total MB | Avg | Median | Geomean | Min | Max | P90 | P99 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tpch | customer.c_comment | 1,500 | 0.11 | 73.20 | 73.00 | 68.22 | 29 | 116 | 108.00 | 116.00 |
| tpch | lineitem.l_comment | 60,175 | 1.60 | 26.56 | 27.00 | 24.55 | 10 | 43 | 40.00 | 43.00 |
| tpch | orders.o_comment | 15,000 | 0.73 | 48.49 | 49.00 | 45.04 | 19 | 78 | 73.00 | 78.00 |
| tpch | part.p_name | 2,000 | 0.07 | 32.66 | 32.00 | 32.44 | 23 | 46 | 38.00 | 42.00 |
| tpch | part.p_type | 2,000 | 0.04 | 20.55 | 21.00 | 20.46 | 16 | 25 | 23.00 | 25.00 |
| tpch | part.p_container | 2,000 | 0.02 | 7.61 | 7.00 | 7.51 | 6 | 10 | 9.00 | 10.00 |
| tpch | partsupp.ps_comment | 8,000 | 0.99 | 124.07 | 124.00 | 115.89 | 49 | 198 | 184.00 | 197.00 |
| tpcds | call_center.cc_class | 1 | 0.00 | 5.00 | 5.00 | 5.00 | 5 | 5 | 5.00 | 5.00 |
| tpcds | catalog_page.cp_description | 11,718 | 0.87 | 73.95 | 74.00 | 73.24 | 0 | 99 | 95.00 | 99.00 |
| tpcds | customer.c_last_name | 1,000 | 0.01 | 6.01 | 6.00 | 5.97 | 0 | 13 | 8.00 | 10.00 |
| tpcds | customer_address.ca_street_name | 500 | 0.00 | 8.21 | 8.00 | 7.82 | 0 | 18 | 13.00 | 17.00 |
| tpcds | date_dim.d_day_name | 73,049 | 0.52 | 7.14 | 7.00 | 7.06 | 6 | 9 | 9.00 | 9.00 |
| tpcds | item.i_item_desc | 180 | 0.02 | 104.18 | 106.50 | 81.23 | 2 | 200 | 179.00 | 199.00 |
| tpcds | item.i_color | 180 | 0.00 | 5.33 | 5.00 | 5.12 | 3 | 9 | 8.00 | 9.00 |
| tpcds | promotion.p_channel_details | 3 | 0.00 | 41.67 | 40.00 | 41.21 | 35 | 50 | 48.00 | 49.80 |
| tpcds | store.s_market_desc | 1 | 0.00 | 56.00 | 56.00 | 56.00 | 56 | 56 | 56.00 | 56.00 |
| tpcds | web_site.web_mkt_desc | 1 | 0.00 | 75.00 | 75.00 | 75.00 | 75 | 75 | 75.00 | 75.00 |
| job | cast_info.note | 23,309,167 | 100.00 | 4.29 | 0.00 | 14.49 | 0 | 922 | 20.00 | 30.00 |
| job | keyword.keyword | 134,170 | 2.07 | 15.41 | 14.00 | 13.94 | 2 | 74 | 25.00 | 37.00 |
| job | movie_info.info | 6,417,054 | 100.00 | 15.58 | 12.00 | 11.34 | 1 | 545,657 | 25.00 | 85.00 |
| job | name.name | 4,167,491 | 60.68 | 14.56 | 14.00 | 14.17 | 1 | 106 | 19.00 | 24.00 |
| job | title.title | 2,527,969 | 41.43 | 16.39 | 13.00 | 13.92 | 1 | 4,715 | 29.00 | 56.00 |

## Complexity Stats

| Dataset | Column | Alphabet | Entropy bits/B | Top byte | Top byte % | Letter % | Digit % | Whitespace % | Distinct values approx % | Top-10 values approx % | Distinct trigrams | Top trigram | Top trigram % | Avg unique bytes/row | Median unique bytes/row | Prefix3 distinct | Suffix3 distinct |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| tpch | customer.c_comment | 35 | 4.281 | space | 13.20 | 84.67 | 0.00 | 13.20 | 100.00 | 0.67 | 1,156 | ly  | 2.49 | 20.47 | 21.00 | 429 | 441 |
| tpch | lineitem.l_comment | 35 | 4.281 | space | 13.27 | 84.62 | 0.00 | 13.27 | 96.96 | 0.02 | 1,377 | ly  | 2.49 | 14.20 | 15.00 | 1,086 | 1,095 |
| tpch | orders.o_comment | 35 | 4.280 | space | 13.28 | 84.58 | 0.00 | 13.28 | 99.99 | 0.07 | 1,328 | ly  | 2.46 | 18.16 | 19.00 | 861 | 881 |
| tpch | part.p_name | 26 | 4.289 | space | 12.25 | 87.75 | 0.00 | 12.25 | 99.90 | 0.50 | 819 | er  | 0.58 | 16.12 | 16.00 | 84 | 85 |
| tpch | part.p_type | 21 | 4.111 | E | 11.20 | 90.27 | 0.00 | 9.73 | 7.48 | 10.65 | 100 | ED  | 5.39 | 13.10 | 13.00 | 6 | 5 |
| tpch | part.p_container | 19 | 4.035 | space | 13.14 | 86.86 | 0.00 | 13.14 | 2.00 | 30.55 | 54 |  CA | 4.66 | 7.06 | 7.00 | 5 | 8 |
| tpch | partsupp.ps_comment | 35 | 4.280 | space | 13.28 | 84.62 | 0.00 | 13.28 | 99.48 | 0.14 | 1,361 | ly  | 2.50 | 22.96 | 23.00 | 723 | 740 |
| tpcds | call_center.cc_class | 5 | 2.322 | a | 20.00 | 100.00 | 0.00 | 0.00 | 100.00 | 100.00 | 3 | arg | 33.33 | 5.00 | 5.00 | 1 | 1 |
| tpcds | catalog_page.cp_description | 55 | 4.435 | space | 13.73 | 83.62 | 0.00 | 13.73 | 99.13 | 1.02 | 4,883 | s.  | 1.04 | 22.51 | 23.00 | 795 | 2,381 |
| tpcds | customer.c_last_name | 51 | 4.717 | e | 9.70 | 100.00 | 0.00 | 0.00 | 66.74 | 12.30 | 1,405 | son | 2.32 | 5.42 | 5.00 | 403 | 330 |
| tpcds | customer_address.ca_street_name | 52 | 4.830 | space | 11.79 | 83.58 | 4.63 | 11.79 | 61.37 | 16.40 | 486 | th  | 3.57 | 7.32 | 7.00 | 76 | 100 |
| tpcds | date_dim.d_day_name | 17 | 3.670 | a | 16.00 | 100.00 | 0.00 | 0.00 | 0.01 | 100.00 | 26 | day | 19.44 | 6.71 | 7.00 | 7 | 1 |
| tpcds | item.i_item_desc | 55 | 4.420 | space | 13.91 | 83.23 | 0.00 | 13.91 | 74.75 | 15.56 | 2,550 | s.  | 1.26 | 23.41 | 25.00 | 111 | 125 |
| tpcds | item.i_color | 23 | 4.155 | e | 11.56 | 100.00 | 0.00 | 0.00 | 33.95 | 38.33 | 204 | ros | 1.67 | 4.84 | 5.00 | 59 | 60 |
| tpcds | promotion.p_channel_details | 28 | 4.237 | space | 14.40 | 82.40 | 0.00 | 14.40 | 100.00 | 100.00 | 114 |  me | 1.68 | 18.00 | 18.00 | 3 | 3 |
| tpcds | store.s_market_desc | 20 | 4.075 | space | 12.50 | 83.93 | 0.00 | 12.50 | 100.00 | 100.00 | 53 | gh  | 3.70 | 20.00 | 20.00 | 1 | 1 |
| tpcds | web_site.web_mkt_desc | 21 | 4.028 | space | 12.00 | 85.33 | 0.00 | 12.00 | 100.00 | 100.00 | 72 |  de | 2.74 | 21.00 | 21.00 | 1 | 1 |
| job | cast_info.note | 154 | 4.482 | e | 11.91 | 79.46 | 0.58 | 5.62 | 1.53 | 90.57 | 46,395 | rod | 3.63 | 3.20 | 0.00 | 1,243 | 2,529 |
| job | keyword.keyword | 87 | 4.318 | e | 10.48 | 90.20 | 0.31 | 0.00 | 99.53 | 0.01 | 12,452 | ing | 1.20 | 10.38 | 10.00 | 4,299 | 4,292 |
| job | movie_info.info | 140 | 5.361 | space | 8.84 | 64.66 | 19.11 | 8.96 | 7.38 | 31.10 | 33,973 |  20 | 1.54 | 10.92 | 9.00 | 6,588 | 2,662 |
| job | name.name | 149 | 4.928 | a | 9.13 | 84.17 | 0.00 | 7.46 | 87.28 | 0.00 | 44,465 | n,  | 1.26 | 11.65 | 12.00 | 11,380 | 14,142 |
| job | title.title | 156 | 5.343 | space | 9.99 | 70.47 | 10.04 | 9.99 | 58.72 | 2.46 | 82,141 | he  | 0.97 | 10.91 | 10.00 | 18,203 | 22,626 |

## Pattern Selectivity

| Dataset | Column | Pattern | LIKE | Literal | Rows matched | Matched % | Avg first offset | Median first offset |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| tpch | customer.c_comment | contains_requests | %requests% | requests | 334 | 22.27 | 33.90 | 30.00 |
| tpch | customer.c_comment | contains_goldenrod | %goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpch | customer.c_comment | prefix_goldenrod | goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpch | customer.c_comment | suffix_lace | %lace | lace | 0 | 0.00 | 0.00 | 0.00 |
| tpch | customer.c_comment | contains_powers | %Powers% | Powers | 0 | 0.00 | 0.00 | 0.00 |
| tpch | customer.c_comment | contains_monday | %Monday% | Monday | 0 | 0.00 | 0.00 | 0.00 |
| tpch | customer.c_comment | contains_unknown | %Unknown% | Unknown | 0 | 0.00 | 0.00 | 0.00 |
| tpch | lineitem.l_comment | contains_requests | %requests% | requests | 4,380 | 7.28 | 11.67 | 10.00 |
| tpch | lineitem.l_comment | contains_goldenrod | %goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpch | lineitem.l_comment | prefix_goldenrod | goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpch | lineitem.l_comment | suffix_lace | %lace | lace | 1 | 0.00 | 30.00 | 30.00 |
| tpch | lineitem.l_comment | contains_powers | %Powers% | Powers | 0 | 0.00 | 0.00 | 0.00 |
| tpch | lineitem.l_comment | contains_monday | %Monday% | Monday | 0 | 0.00 | 0.00 | 0.00 |
| tpch | lineitem.l_comment | contains_unknown | %Unknown% | Unknown | 0 | 0.00 | 0.00 | 0.00 |
| tpch | orders.o_comment | contains_requests | %requests% | requests | 2,276 | 15.17 | 23.87 | 21.50 |
| tpch | orders.o_comment | contains_goldenrod | %goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpch | orders.o_comment | prefix_goldenrod | goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpch | orders.o_comment | suffix_lace | %lace | lace | 0 | 0.00 | 0.00 | 0.00 |
| tpch | orders.o_comment | contains_powers | %Powers% | Powers | 0 | 0.00 | 0.00 | 0.00 |
| tpch | orders.o_comment | contains_monday | %Monday% | Monday | 0 | 0.00 | 0.00 | 0.00 |
| tpch | orders.o_comment | contains_unknown | %Unknown% | Unknown | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_name | contains_requests | %requests% | requests | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_name | contains_goldenrod | %goldenrod% | goldenrod | 95 | 4.75 | 14.17 | 13.00 |
| tpch | part.p_name | prefix_goldenrod | goldenrod% | goldenrod | 18 | 0.90 | 0.00 | 0.00 |
| tpch | part.p_name | suffix_lace | %lace | lace | 29 | 1.45 | 27.28 | 27.00 |
| tpch | part.p_name | contains_powers | %Powers% | Powers | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_name | contains_monday | %Monday% | Monday | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_name | contains_unknown | %Unknown% | Unknown | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_type | contains_requests | %requests% | requests | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_type | contains_goldenrod | %goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_type | prefix_goldenrod | goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_type | suffix_lace | %lace | lace | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_type | contains_powers | %Powers% | Powers | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_type | contains_monday | %Monday% | Monday | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_type | contains_unknown | %Unknown% | Unknown | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_container | contains_requests | %requests% | requests | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_container | contains_goldenrod | %goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_container | prefix_goldenrod | goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_container | suffix_lace | %lace | lace | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_container | contains_powers | %Powers% | Powers | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_container | contains_monday | %Monday% | Monday | 0 | 0.00 | 0.00 | 0.00 |
| tpch | part.p_container | contains_unknown | %Unknown% | Unknown | 0 | 0.00 | 0.00 | 0.00 |
| tpch | partsupp.ps_comment | contains_requests | %requests% | requests | 2,946 | 36.83 | 60.77 | 53.00 |
| tpch | partsupp.ps_comment | contains_goldenrod | %goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpch | partsupp.ps_comment | prefix_goldenrod | goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpch | partsupp.ps_comment | suffix_lace | %lace | lace | 0 | 0.00 | 0.00 | 0.00 |
| tpch | partsupp.ps_comment | contains_powers | %Powers% | Powers | 0 | 0.00 | 0.00 | 0.00 |
| tpch | partsupp.ps_comment | contains_monday | %Monday% | Monday | 0 | 0.00 | 0.00 | 0.00 |
| tpch | partsupp.ps_comment | contains_unknown | %Unknown% | Unknown | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | call_center.cc_class | contains_requests | %requests% | requests | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | call_center.cc_class | contains_goldenrod | %goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | call_center.cc_class | prefix_goldenrod | goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | call_center.cc_class | suffix_lace | %lace | lace | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | call_center.cc_class | contains_powers | %Powers% | Powers | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | call_center.cc_class | contains_monday | %Monday% | Monday | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | call_center.cc_class | contains_unknown | %Unknown% | Unknown | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | catalog_page.cp_description | contains_requests | %requests% | requests | 6 | 0.05 | 56.50 | 62.50 |
| tpcds | catalog_page.cp_description | contains_goldenrod | %goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | catalog_page.cp_description | prefix_goldenrod | goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | catalog_page.cp_description | suffix_lace | %lace | lace | 1 | 0.01 | 55.00 | 55.00 |
| tpcds | catalog_page.cp_description | contains_powers | %Powers% | Powers | 10 | 0.09 | 24.50 | 15.50 |
| tpcds | catalog_page.cp_description | contains_monday | %Monday% | Monday | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | catalog_page.cp_description | contains_unknown | %Unknown% | Unknown | 11 | 0.09 | 24.18 | 22.00 |
| tpcds | customer.c_last_name | contains_requests | %requests% | requests | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | customer.c_last_name | contains_goldenrod | %goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | customer.c_last_name | prefix_goldenrod | goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | customer.c_last_name | suffix_lace | %lace | lace | 2 | 0.20 | 3.00 | 3.00 |
| tpcds | customer.c_last_name | contains_powers | %Powers% | Powers | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | customer.c_last_name | contains_monday | %Monday% | Monday | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | customer.c_last_name | contains_unknown | %Unknown% | Unknown | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | customer_address.ca_street_name | contains_requests | %requests% | requests | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | customer_address.ca_street_name | contains_goldenrod | %goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | customer_address.ca_street_name | prefix_goldenrod | goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | customer_address.ca_street_name | suffix_lace | %lace | lace | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | customer_address.ca_street_name | contains_powers | %Powers% | Powers | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | customer_address.ca_street_name | contains_monday | %Monday% | Monday | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | customer_address.ca_street_name | contains_unknown | %Unknown% | Unknown | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | date_dim.d_day_name | contains_requests | %requests% | requests | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | date_dim.d_day_name | contains_goldenrod | %goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | date_dim.d_day_name | prefix_goldenrod | goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | date_dim.d_day_name | suffix_lace | %lace | lace | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | date_dim.d_day_name | contains_powers | %Powers% | Powers | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | date_dim.d_day_name | contains_monday | %Monday% | Monday | 10,436 | 14.29 | 0.00 | 0.00 |
| tpcds | date_dim.d_day_name | contains_unknown | %Unknown% | Unknown | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | item.i_item_desc | contains_requests | %requests% | requests | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | item.i_item_desc | contains_goldenrod | %goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | item.i_item_desc | prefix_goldenrod | goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | item.i_item_desc | suffix_lace | %lace | lace | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | item.i_item_desc | contains_powers | %Powers% | Powers | 1 | 0.56 | 0.00 | 0.00 |
| tpcds | item.i_item_desc | contains_monday | %Monday% | Monday | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | item.i_item_desc | contains_unknown | %Unknown% | Unknown | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | item.i_color | contains_requests | %requests% | requests | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | item.i_color | contains_goldenrod | %goldenrod% | goldenrod | 1 | 0.56 | 0.00 | 0.00 |
| tpcds | item.i_color | prefix_goldenrod | goldenrod% | goldenrod | 1 | 0.56 | 0.00 | 0.00 |
| tpcds | item.i_color | suffix_lace | %lace | lace | 1 | 0.56 | 0.00 | 0.00 |
| tpcds | item.i_color | contains_powers | %Powers% | Powers | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | item.i_color | contains_monday | %Monday% | Monday | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | item.i_color | contains_unknown | %Unknown% | Unknown | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | promotion.p_channel_details | contains_requests | %requests% | requests | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | promotion.p_channel_details | contains_goldenrod | %goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | promotion.p_channel_details | prefix_goldenrod | goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | promotion.p_channel_details | suffix_lace | %lace | lace | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | promotion.p_channel_details | contains_powers | %Powers% | Powers | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | promotion.p_channel_details | contains_monday | %Monday% | Monday | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | promotion.p_channel_details | contains_unknown | %Unknown% | Unknown | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | store.s_market_desc | contains_requests | %requests% | requests | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | store.s_market_desc | contains_goldenrod | %goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | store.s_market_desc | prefix_goldenrod | goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | store.s_market_desc | suffix_lace | %lace | lace | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | store.s_market_desc | contains_powers | %Powers% | Powers | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | store.s_market_desc | contains_monday | %Monday% | Monday | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | store.s_market_desc | contains_unknown | %Unknown% | Unknown | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | web_site.web_mkt_desc | contains_requests | %requests% | requests | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | web_site.web_mkt_desc | contains_goldenrod | %goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | web_site.web_mkt_desc | prefix_goldenrod | goldenrod% | goldenrod | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | web_site.web_mkt_desc | suffix_lace | %lace | lace | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | web_site.web_mkt_desc | contains_powers | %Powers% | Powers | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | web_site.web_mkt_desc | contains_monday | %Monday% | Monday | 0 | 0.00 | 0.00 | 0.00 |
| tpcds | web_site.web_mkt_desc | contains_unknown | %Unknown% | Unknown | 0 | 0.00 | 0.00 | 0.00 |
| job | cast_info.note | contains_love | %love% | love | 323 | 0.00 | 15.60 | 11.00 |
| job | cast_info.note | contains_movie | %movie% | movie | 197 | 0.00 | 12.37 | 6.00 |
| job | cast_info.note | prefix_the | The% | The | 0 | 0.00 | 0.00 | 0.00 |
| job | cast_info.note | suffix_drama | %drama | drama | 0 | 0.00 | 0.00 | 0.00 |
| job | keyword.keyword | contains_love | %love% | love | 554 | 0.41 | 7.61 | 7.00 |
| job | keyword.keyword | contains_movie | %movie% | movie | 324 | 0.24 | 6.60 | 6.00 |
| job | keyword.keyword | prefix_the | The% | The | 0 | 0.00 | 0.00 | 0.00 |
| job | keyword.keyword | suffix_drama | %drama | drama | 56 | 0.04 | 9.64 | 8.00 |
| job | movie_info.info | contains_love | %love% | love | 1,955 | 0.03 | 328.06 | 1.00 |
| job | movie_info.info | contains_movie | %movie% | movie | 4 | 0.00 | 11935.25 | 9.50 |
| job | movie_info.info | prefix_the | The% | The | 3,657 | 0.06 | 0.00 | 0.00 |
| job | movie_info.info | suffix_drama | %drama | drama | 0 | 0.00 | 0.00 | 0.00 |
| job | name.name | contains_love | %love% | love | 1,029 | 0.02 | 2.62 | 1.00 |
| job | name.name | contains_movie | %movie% | movie | 2 | 0.00 | 9.50 | 9.50 |
| job | name.name | prefix_the | The% | The | 2,240 | 0.05 | 0.00 | 0.00 |
| job | name.name | suffix_drama | %drama | drama | 4 | 0.00 | 6.75 | 7.00 |
| job | title.title | contains_love | %love% | love | 917 | 0.04 | 10.74 | 8.00 |
| job | title.title | contains_movie | %movie% | movie | 70 | 0.00 | 19.56 | 18.00 |
| job | title.title | prefix_the | The% | The | 193,190 | 7.64 | 0.00 | 0.00 |
| job | title.title | suffix_drama | %drama | drama | 73 | 0.00 | 13.95 | 11.00 |
