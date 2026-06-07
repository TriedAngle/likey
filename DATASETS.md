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

Generate TPC-H/TPC-DS raw relational exports:

```bash
python3 scripts/download_tpc.py
```

Download and normalize JOB/IMDB raw data:

```bash
python3 scripts/download_job.py
```

This writes to `data/raw/` by default:

- `data/raw/tpch/`
- `data/raw/tpcds/`
- `data/raw/job/`

Download FASTA data:

```bash
python3 scripts/download_fasta.py
```

This writes directly to `data/fasta/`:

- `ensembl_human_cdna.fna`
- `gencode_human_transcripts.fna`
- `refseq_viral_genomic.fna`
- `uniprot_sprot.faa`
- `uniprot_trembl.faa`

Prepare relational data for the runner:

```bash
python3 scripts/prepare_data.py
```

Validate raw and prepared CSV files:

```bash
python3 scripts/validate_csv_data.py
```

Validate that Rust parses the prepared runner CSVs:

```bash
cargo run -p runner --bin csv_validate -- --data-csv data/data_all.csv
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

Raw format: extracted CSV files plus `schematext.sql` in `data/raw/job/`. The source archive uses a non-standard CSV dialect; `download_job.py` normalizes table files to headerless pipe-delimited CSV so `prepare_data.py` can read one raw dialect consistently.

Useful prepared columns:

- `title.title`: movie, episode, and series titles.
- `name.name`: person names.
- `keyword.keyword`: keyword tags such as `handcuffed-to-a-bed`.
- `movie_info.info`: movie metadata values, including languages, runtimes, and descriptive facts depending on `info_type_id`.
- `cast_info.note`: cast notes such as `(voice)`.

Other string columns include alternate titles/names, phonetic codes, MD5 hashes, company names, role names, and notes. They are skipped by default because they are either derived identifiers or less representative text search columns.

## FASTA

Sources:

- Ensembl human cDNA: `ensembl_human_cdna.fna`.
- GENCODE human transcripts: `gencode_human_transcripts.fna`.
- NCBI RefSeq viral genomic: `refseq_viral_genomic.fna`.
- UniProt Swiss-Prot protein: `uniprot_sprot.faa`.
- UniProt TrEMBL protein: `uniprot_trembl.faa`.

Prepared format: direct FASTA files under `data/fasta/`, plus `data/fasta/data.csv` and `data/data_all.csv` manifest entries.

The `.fna` files load as `dna-fasta` with `utf8`, `fsst`, `dna2`, or `all` storage. The `.faa` files load as `protein-fasta` with `utf8`, `fsst`, or `all` storage. The downloader preserves full FASTA records, so a small byte target may still write one large record.
