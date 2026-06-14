# Datasets

Benchmark source data is local to the working tree. The `data/` directory is ignored by git; tracked benchmark configuration lives under `benchmarks/` and points at files that must exist under `data/` before the corresponding benchmark can run.

## Layout

```text
data/raw/tpch/        DuckDB TPC-H raw exports
data/raw/tpcds/       DuckDB TPC-DS raw exports
data/raw/job/         downloaded and normalized JOB/IMDB raw tables
data/raw/other/       manually supplied headered CSV datasets, such as quotes and spam
data/tpch/            prepared key,value CSV columns plus data.csv manifest
data/tpcds/           prepared key,value CSV columns plus data.csv manifest
data/job/             prepared key,value CSV columns plus data.csv manifest
data/quotes/          prepared quotes key,value CSV plus data.csv manifest
data/spam/            prepared spam key,value CSV plus data.csv manifest
data/fasta/           downloaded FASTA files and generated FASTA benchmark data
data/fftstr/          generated artificial FFTSTR benchmark table
data/data_all.csv     combined prepared-data manifest
benchmarks/           tracked benchmark manifests, algorithms, patterns, and indexes
benchmark_results/    timestamped benchmark outputs
```

Prepared relational and headered-CSV datasets use this runner-readable shape:

```csv
key,value
```

`key` is a stable row label from the source table when one is available. `value` is the extracted text column. FASTA data is read directly by the runner and does not need this `key,value` conversion.

## Setup

Dataset scripts use Python. From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install duckdb zstandard
```

`duckdb` is required for TPC-H/TPC-DS generation. `zstandard` is required for extracting the JOB `.tzst` archive. `curl` is used by `download_job.py` when available, with Python's URL opener as a fallback.

## Raw Data

Generate TPC-H and TPC-DS raw exports with DuckDB:

```bash
python3 scripts/download_tpc.py
```

Useful options:

```bash
python3 scripts/download_tpc.py --tpch-sf 1 --tpcds-sf 1
python3 scripts/download_tpc.py --skip-tpcds
python3 scripts/download_tpc.py --force
```

Download and normalize JOB/IMDB raw data:

```bash
python3 scripts/download_job.py
```

The default JOB source is `https://db.in.tum.de/~fent/dbgen/job/imdb.tzst`. The script extracts the archive under `data/raw/job/` and normalizes table files to headerless pipe-delimited CSV so the rest of the pipeline can read a single raw dialect.

Headered CSV datasets are not downloaded by a repository script. Place them here when needed:

| Dataset | Expected raw file | Encoding | Delimiter | Source column |
|---|---|---|---|---|
| quotes | `data/raw/other/Quotes.csv` | `utf-8-sig` | `;` | `QUOTE` |
| spam | `data/raw/other/spam.csv` | `latin-1` | `,` | `v2` |

Download FASTA data:

```bash
python3 scripts/download_fasta.py
```

The FASTA downloader writes directly to `data/fasta/` and subsets each source to about 800 MB by default, preserving whole FASTA records. Use `--target-bytes`, `--dna-target-bytes`, `--protein-target-bytes`, `--max-records`, or `--force` to change that behavior.

Downloaded FASTA targets:

| Dataset | File | Type |
|---|---|---|
| Ensembl human cDNA | `data/fasta/ensembl_human_cdna.fna` | `dna-fasta` |
| GENCODE human transcripts | `data/fasta/gencode_human_transcripts.fna` | `dna-fasta` |
| NCBI RefSeq viral genomic | `data/fasta/refseq_viral_genomic.fna` | `dna-fasta` |
| UniProt Swiss-Prot protein | `data/fasta/uniprot_sprot.faa` | `protein-fasta` |
| UniProt TrEMBL protein | `data/fasta/uniprot_trembl.faa` | `protein-fasta` |

## Prepared Data

Prepare raw relational/headered CSV data and write runner manifests:

```bash
python3 scripts/prepare_data.py
```

By default this prepares `tpch,tpcds,job,quotes,spam,dna,protein` with `storage=all`. Runner storage expansion treats `all` as `utf8,fsst,dna2` for `dna-fasta`, and `utf8,fsst` for `protein-fasta` and `job-csv`.

Useful options:

```bash
python3 scripts/prepare_data.py --datasets job,quotes,dna
python3 scripts/prepare_data.py --storage utf8
python3 scripts/prepare_data.py --max-rows 100000
python3 scripts/prepare_data.py --all-string-columns
python3 scripts/prepare_data.py --columns tpch.part.p_name,job.title.title
python3 scripts/prepare_data.py --force
```

Default prepared columns:

| Dataset | Columns |
|---|---|
| TPC-H | `customer.c_comment`, `lineitem.l_comment`, `orders.o_comment`, `part.p_name`, `part.p_type`, `part.p_container`, `partsupp.ps_comment` |
| TPC-DS | `call_center.cc_class`, `catalog_page.cp_description`, `customer.c_last_name`, `customer_address.ca_street_name`, `date_dim.d_day_name`, `item.i_item_desc`, `item.i_color`, `promotion.p_channel_details`, `store.s_market_desc`, `web_site.web_mkt_desc` |
| JOB | `cast_info.note`, `keyword.keyword`, `movie_companies.note`, `movie_info.info`, `name.name`, `title.title` |
| quotes | `quotes.quote` |
| spam | `spam.v2` |

The script writes per-dataset manifests such as `data/job/data.csv` and a combined `data/data_all.csv`. FASTA manifest rows are written when matching files exist in `data/fasta/`; missing FASTA files are skipped.

## Generated Benchmark Data

Some benchmark suites use deterministic generated data in addition to downloaded/prepared datasets.

Generate the sparse-N DNA benchmark data and manifest:

```bash
python3 scripts/generate_dna_n_benchmark.py
```

Defaults: 2048 FASTA rows, length 180, 512 rows containing `N`. Outputs:

```text
data/fasta/dna_n_sparse_180_x2048.fna
benchmarks/dna/data_n_sparse_utf8_dna2.csv
benchmarks/dna/n-handling/patterns.csv
```

Generate the artificial FFTSTR benchmark table and manifest:

```bash
python3 scripts/generate_fftstr_benchmark.py
```

Defaults: 1000 rows, length 512, alternating `abab...` strings. Outputs:

```text
data/fftstr/abab_512_x1000.csv
benchmarks/fftstr/abab-wildcards/data.csv
benchmarks/fftstr/abab-wildcards/patterns.csv
```

`scripts/run_all_benchmarks.py` automatically runs these two generators when their benchmark cases are selected, unless `--skip-dna-n-generate` or `--skip-fftstr-generate` is used.

## Benchmark Manifests

The tracked files under `benchmarks/` are suite-specific runner inputs. They are intentionally separate from `data/data_all.csv` because benchmarks often pin storage combinations, algorithm sets, pattern sets, and index sets.

Important benchmark data dependencies:

| Suite | Benchmark manifests reference |
|---|---|
| DNA GENCODE | `data/fasta/gencode_human_transcripts.fna` |
| DNA sparse-N | `data/fasta/dna_n_sparse_180_x2048.fna` generated by `generate_dna_n_benchmark.py` |
| quotes | `data/quotes/quotes__quote.csv` |
| JOB | prepared files under `data/job/`, including the six default JOB columns |
| FFTSTR | `data/fftstr/abab_512_x1000.csv` generated by `generate_fftstr_benchmark.py` |

Run all configured benchmark cases with checkpointing:

```bash
python3 scripts/run_all_benchmarks.py --result-root benchmark_results --iterations 3
```

See `benchmarks.md` for the benchmark matrix and `benchmark_overview.md` for a compact overview of each run.

## Validation

Validate raw and prepared relational/headered CSV data:

```bash
python3 scripts/validate_csv_data.py
```

Useful options:

```bash
python3 scripts/validate_csv_data.py --datasets job,quotes,spam
python3 scripts/validate_csv_data.py --skip-raw
python3 scripts/validate_csv_data.py --no-compare-prepared
```

Validate that the Rust runner can parse prepared `job-csv` files referenced by a data manifest:

```bash
cargo run -p runner --bin csv_validate -- --data-csv data/data_all.csv
```

`csv_validate` only checks `job-csv` rows in the manifest. FASTA records are validated when the runner loads them.

Generate optional per-column statistics for prepared relational data:

```bash
python3 scripts/dataset_stats.py --datasets tpch,tpcds,job --output DATASET_STATS.md
```

The stats generator streams prepared `key,value` CSV files and can apply a runner-like total byte cap with `--max-total-bytes 100MB`.
