# Likey
Rust string-search algorithms and SQL-like `%` / `_` pattern matching. 
It focuses on fast matching, clear comparisons, and a simple mmap-backed data model.
The intention is to prototype and optimize like queries easily.

## Structure
- `algos`: string search algorithms with a shared `StringSearch` trait.
- `like`: pattern compiler + matcher for SQL-like `LIKE` semantics.
- `storage`: mmap-backed arena and dataset loaders (text + FASTA).
- `engine`: executes compiled patterns against datasets.
- `tests`: benchmarks and integration binaries.

## Architecture (short)
```
files -> storage::BumpArena -> storage::dataset::DataSet
                       |                    |
pattern -> like::compile_pattern ----------> engine::execute
                       |                    |
                    algos::<impl>       matches
```
- Data is loaded once into a big mmap arena and treated as immutable.
- Patterns are compiled once and reused across rows/tables.
- The engine is a thin loop over rows; no query parser or planner.

## Algorithms (core idea)
- Naive: checks each position in the text for a full match.
- KMP: avoids re-checking by using a prefix table.
- Boyer–Moore: skips ahead using bad-character/good-suffix heuristics.
- StdSearch: Rust `str::find` as a baseline.

## Testing
- Unit tests: `cargo test -p storage` and `cargo test -p engine`
- Benchmark + correctness report: `cargo run -p tests --bin bench_like --release`


## Further ideas:
- LUT table + simd like simdjson (1 table per 8 characters)
- partitioned multithreading hotspot finding
- exact naive string search on gpu
- prefix sum based search, (rolling hash, robin carp?) hot spots + exact search "filter + compact"

- assumption: hotspots and gpu should perform well in sparse.
