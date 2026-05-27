# likey2

Rust workspace for LIKE/search experiments over dense string columns.

## Crates and Tools

```text
db/                    storage, LIKE verification, search algorithms, FM/trigram indexes
runner/                CSV-driven benchmark binary for db
scripts/run_bench.py   release runner wrapper with summaries and plots
```

`db` is the library crate. It provides dense UTF-8 byte columns, FSST-compressed byte columns, packed DNA2 columns, `RowId`, the generic `Column<Symbol = u8>` API, LIKE pattern compilation/verification, candidate providers, result sinks, full scan, an FM-index, and trigram indexes.

`runner` is the benchmark binary. It loads datasets, builds the requested storage and indexes, runs compatible algorithm/pattern/index combinations, and writes timing CSVs. `scripts/run_bench.py` handles release-mode runs, summaries, and plots.

## Basic Commands

Run checks and a few library examples from the workspace root:

```bash
cargo test
cargo run -p db --example full_scan
cargo run -p db --example like_matching
cargo run -p db --example fm_index_utf8_like
cargo run -p db --example like_with_trigram_candidates
cargo run -p db --example fsst_table
```

Run the benchmark binary directly:

```bash
cargo run -p runner --release -- \
  --data-csv runner/examples/data_dna.csv \
  --algorithms-csv runner/examples/algorithms_all.csv \
  --patterns-csv runner/examples/patterns_dna.csv \
  --indexes-csv runner/examples/indexes.csv \
  --output-csv results/raw.csv \
  --summary-csv results/summary.csv
```

Use the Python wrapper when you want a timestamped result directory, copied inputs, summaries, and plots:

```bash
python3 scripts/run_bench.py \
  --name dna_example \
  --data-csv runner/examples/data_dna.csv \
  --algorithms-csv runner/examples/algorithms_all.csv \
  --patterns-csv runner/examples/patterns_dna.csv \
  --indexes-csv runner/examples/indexes.csv \
  --warmups 1 \
  --iterations 5
```

The wrapper writes to `results/<name>_<timestamp>/` with `raw.csv`, `summary.csv`, `python_summary.csv`, `command.txt`, `info.txt`, copied inputs, and plots.

## Benchmark Inputs

Runner input files are CSVs. Paths in `data.csv` are resolved relative to the data CSV file.

`data.csv` columns:

```text
name,path,type,storage,column[,key_column,value_column,enabled]
```

Supported dataset types:

```text
dna-fasta       FASTA sequence data; storage=utf8, dna2, or both
protein-fasta   FASTA sequence data; UTF-8/byte storage only
job-csv         key/value CSV columns; UTF-8/byte storage only
```

For `job-csv`, multiple rows with the same `name` become key-aligned logical columns. Each column is stored as its own dense table in `db`.

Other runner CSVs:

```csv
algorithm,enabled
name,pattern,enabled
index,enabled
```

Example input sets live in `runner/examples/`:

```text
DNA       data_dna.csv      algorithms_all.csv   patterns_dna.csv
Protein   data_protein.csv  algorithms_utf8.csv  patterns_protein.csv
JOB       data_job.csv      algorithms_utf8.csv  patterns_job.csv
Indexes   indexes.csv
```

Use those file names to swap the data, algorithm, and pattern inputs in the direct or wrapper commands above.

If `--indexes-csv` is omitted, the runner benchmarks full scan only. `indexes.csv` supports `none`/`full-scan`, `trigram`, and `fm`.

## Algorithms and Semantics

UTF-8 algorithms run on UTF-8 storage: `std`, `kmp`, `naive`, `naive-scalar`, `naive-vectorized`, `naive-vectorized-v2`, `naive-avx2`, `naive-avx2-v2`, `naive-avx512`, `naive-avx512-v2`, `naive-auto`, `naive-mixed`, wildcard-aware naive variants, `bm`, `two-way`, `two-way2`, `libc-memmem`, `fft0`, and `fft1`.

`dna2` and the DNA2 packed/vectorized variants run on DNA2 storage. DNA2 exposes bases as logical byte symbols: `A=0`, `C=1`, `G=2`, `T=3`.

FSST columns expose decoded bytes as logical symbols, so FM-index and typed trigram index construction work through the same `Column<Symbol = u8>` API. The current FSST LIKE path decodes candidate rows before verification; it is not compressed-domain matching yet.

UTF-8 LIKE matching is byte-based. `_` matches one byte, not one Unicode scalar or grapheme. Index probes only provide candidates; the LIKE verifier is still the correctness check.

## Row Profiling and Limits

Normal benchmark timings do not include per-row timers. Enable the separate profiling pass with the wrapper when you need slow-row analysis:

```bash
python3 scripts/run_bench.py \
  --name dna_profile \
  --data-csv runner/examples/data_dna.csv \
  --algorithms-csv runner/examples/algorithms_all.csv \
  --patterns-csv runner/examples/patterns_dna.csv \
  --indexes-csv runner/examples/indexes.csv \
  --row-profile \
  --row-profile-repeats 5 \
  --row-profile-max-rows 10000
```

With profiling enabled, the wrapper also writes `row_profile.csv`, `row_profile_summary.csv`, `row_profile_top_slow.csv`, `row_profile_top_fast.csv`, and `plots/row_profile_*.png`.

Default load limits:

```text
--max-rows             unbounded
--max-total-bytes      1GiB
--max-row-bytes        50MiB
--row-overflow-policy  truncate
--invalid-dna          skip-record
```

Byte limits accept plain byte counts and units such as `1GiB`, `1GB`, `512MiB`, and `50MB`.
