# Datasets

This repository keeps benchmark data local. Raw downloaded/generated data and prepared runner input files are ignored by git.

## Layout

```text
data/raw/<dataset>/     Raw relational exports from DuckDB or downloaded archives
data/<dataset>/         Prepared one-column CSV files for the runner
data/fasta/             FASTA files, directly readable by the runner
```

The runner reads FASTA directly. For relational data it currently reads `job-csv` inputs: one logical string column per CSV file with this header:

```csv
key,value
```

`key` is a stable row label from the source table's ID/key column when available. `value` is the extracted string column.

## Pipeline

Download/generate raw relational exports:

```bash
python3 scripts/download_benchmarks.py
```

This writes to `data/raw/` by default:

- `data/raw/tpch/`
- `data/raw/tpcds/`
- `data/raw/job/`

Download FASTA data:

```bash
python3 scripts/download_bio_benchmarks.py
```

This writes directly to `data/fasta/`:

- `dna_benchmark.fna`
- `protein_benchmark.faa`

Prepare relational data for the runner:

```bash
python3 scripts/prepare_data.py
```

This writes curated column files and manifests:

- `data/tpch/*.csv` and `data/tpch/data.csv`
- `data/tpcds/*.csv` and `data/tpcds/data.csv`
- `data/job/*.csv` and `data/job/data.csv`
- `data/fasta/data.csv`
- `data/data_all.csv`

By default, `prepare_data.py` extracts only the useful benchmark text columns listed below. Use `--all-string-columns` to extract every string column, or `--columns tpch.part.p_name,job.title.title` to choose explicitly.

## TPC-H

Source: DuckDB `tpch` extension via `CALL dbgen(sf=...)`.

Raw format: headerless pipe-delimited CSV files plus `schema.sql` in `data/raw/tpch/`.

Useful prepared columns:

- `customer.c_comment`: synthetic customer free-text comments.
- `lineitem.l_comment`: synthetic line-item free-text comments.
- `orders.o_comment`: synthetic order free-text comments.
- `part.p_name`: product names made from color/material/noun tokens, e.g. `goldenrod lavender spring chocolate lace`.
- `part.p_type`: product type/category phrases, e.g. `PROMO BURNISHED COPPER`.
- `part.p_container`: short packaging/category labels, e.g. `JUMBO PKG`.
- `partsupp.ps_comment`: synthetic supplier-part free-text comments.

Other string columns exist, such as names, addresses, phone numbers, statuses, and small categorical fields. They are skipped by default because they are low-cardinality, identifier-like, or less useful for LIKE search benchmarks.

## TPC-DS

Source: DuckDB `tpcds` extension via `CALL dsdgen(sf=...)`.

Raw format: headerless pipe-delimited CSV files plus `schema.sql` in `data/raw/tpcds/`.

Useful prepared columns:

- `call_center.cc_class`: call-center class/category labels.
- `catalog_page.cp_description`: catalog page descriptive text.
- `customer.c_last_name`: customer surname values.
- `customer_address.ca_street_name`: street names.
- `date_dim.d_day_name`: weekday names.
- `item.i_item_desc`: product description free text.
- `item.i_color`: product color labels.
- `promotion.p_channel_details`: promotional details/free text.
- `store.s_market_desc`: store market description free text.
- `web_site.web_mkt_desc`: website market description free text.

TPC-DS has many more string columns: IDs, flags, names, addresses, categories, URLs, states, ZIPs, and country fields. They can be extracted with `--all-string-columns` if needed.

## JOB / IMDB

Source: downloaded archive from `https://db.in.tum.de/~fent/dbgen/job/imdb.tzst` by default.

Raw format: extracted CSV files plus `schematext.sql` in `data/raw/job/`. Some JOB files are pipe-delimited and some are comma-delimited, so `prepare_data.py` detects the delimiter per file.

Useful prepared columns:

- `title.title`: movie, episode, and series titles.
- `name.name`: person names.
- `keyword.keyword`: keyword tags such as `handcuffed-to-a-bed`.
- `movie_info.info`: movie metadata values, including languages, runtimes, and descriptive facts depending on `info_type_id`.
- `cast_info.note`: cast notes such as `(voice)`.

Other string columns include alternate titles/names, phonetic codes, MD5 hashes, company names, role names, and notes. They are skipped by default because they are either derived identifiers or less representative text search columns.

## DNA FASTA

Source: NCBI GRCh38 genomic FASTA gzip.

Prepared format: direct FASTA at `data/fasta/dna_benchmark.fna`.

The runner can load it as `dna-fasta` with `utf8`, `fsst`, `dna2`, or `all` storage. The current downloader preserves full FASTA records, so a small byte target may still write a large chromosome record.

## Protein FASTA

Source: UniProt Swiss-Prot FASTA gzip, with TrEMBL fallback if Swiss-Prot is smaller than the requested target.

Prepared format: direct FASTA at `data/fasta/protein_benchmark.faa`.

The runner can load it as `protein-fasta` with `utf8`, `fsst`, or `all` storage.
