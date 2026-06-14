#!/usr/bin/env python3
"""Run existing LIKE benchmark patterns against Umbra.

The script consumes the same data and pattern manifests as the likey2 runner,
loads one enabled dataset into Umbra, executes SQL LIKE predicates, and writes
raw/summary CSVs with an Umbra row shape close to the runner output.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median


TABLE_NAME = "bench_data"
INFO_RE = re.compile(r"execution:\s*\((?P<execution>[^)]*)\).*?compilation:\s*\((?P<compilation>[^)]*)\)")


@dataclass(frozen=True)
class DataSpec:
    name: str
    path: Path
    data_type: str
    column: str
    key_column: str | None
    value_column: str | None


@dataclass(frozen=True)
class PatternSpec:
    name: str
    pattern: str


@dataclass
class LoadStats:
    records_seen: int = 0
    records_loaded: int = 0
    records_skipped: int = 0
    records_truncated: int = 0
    records_invalid_dna: int = 0
    total_input_sequence_bytes: int = 0
    total_loaded_symbols: int = 0
    stopped_by_max_rows: bool = False
    stopped_by_max_total_bytes: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Umbra LIKE benchmarks from likey2 manifests")
    p.add_argument("--data-csv", required=True, type=Path)
    p.add_argument("--patterns-csv", required=True, type=Path)
    p.add_argument("--data-name", help="Select one enabled data row by name when a manifest has multiple rows")
    p.add_argument("--result-root", type=Path, default=Path("benchmark_results/umbra"))
    p.add_argument("--name", default="umbra")
    p.add_argument("--docker-image", default="umbradb/umbra:latest")
    p.add_argument("--platform", help="Optional Docker platform, e.g. linux/arm64 or linux/amd64")
    p.add_argument("--cpus", help="Docker CPU quota, e.g. 1 for a single CPU")
    p.add_argument("--cpuset-cpus", help="Docker CPU set, e.g. 0 to pin to one CPU")
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--max-total-bytes", default="100MB")
    p.add_argument("--max-row-bytes", default="50MiB")
    p.add_argument("--row-overflow-policy", choices=["truncate", "skip", "error"], default="truncate")
    p.add_argument("--uppercase-sequences", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--create-index", action="store_true", help="Create a normal SQL index on value before timing")
    p.add_argument("--limit-patterns", type=int, help="Run only the first N enabled patterns")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be greater than zero")
    if args.warmups < 0:
        raise SystemExit("--warmups cannot be negative")

    max_total_bytes = parse_bytes(args.max_total_bytes)
    max_row_bytes = parse_bytes(args.max_row_bytes)
    if max_total_bytes <= 0:
        raise SystemExit("--max-total-bytes must be greater than zero")
    if max_row_bytes <= 0:
        raise SystemExit("--max-row-bytes must be greater than zero")

    cwd = Path.cwd().resolve()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.result_root / f"{safe_filename(args.name)}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir()

    data_spec = select_data_spec(load_data_specs(args.data_csv), args.data_name)
    patterns = load_patterns(args.patterns_csv)
    if args.limit_patterns is not None:
        patterns = patterns[: args.limit_patterns]
    if not patterns:
        raise SystemExit("pattern CSV produced no enabled patterns")

    shutil.copy2(args.patterns_csv, inputs_dir / f"patterns_{args.patterns_csv.name}")
    write_dataset_paths(inputs_dir, args.data_csv, data_spec)

    loaded_csv = out_dir / "umbra_input.csv"
    load_stats = write_umbra_input_csv(
        data_spec,
        args.data_csv.parent,
        loaded_csv,
        max_rows=args.max_rows,
        max_total_bytes=max_total_bytes,
        max_row_bytes=max_row_bytes,
        row_overflow_policy=args.row_overflow_policy,
        uppercase_sequences=args.uppercase_sequences,
    )
    (out_dir / "load_stats.json").write_text(json.dumps(asdict(load_stats), indent=2) + "\n")

    load_sql = out_dir / "load.sql"
    queries_sql = out_dir / "queries.sql"
    write_load_sql(load_sql, container_path(cwd, loaded_csv), args.create_index)
    query_plan = write_queries_sql(queries_sql, patterns, args.warmups, args.iterations)

    command = ["docker", "run", "--rm"]
    if args.platform:
        command.extend(["--platform", args.platform])
    if args.cpus:
        command.extend(["--cpus", args.cpus])
    if args.cpuset_cpus:
        command.extend(["--cpuset-cpus", args.cpuset_cpus])
    command.extend([
        "-v",
        f"{cwd}:/work",
        "-w",
        "/work",
        args.docker_image,
        "umbra-sql",
        "-createdb",
        "/tmp/umbra.db",
        container_path(cwd, load_sql),
        container_path(cwd, queries_sql),
    ])
    (out_dir / "command.txt").write_text(" ".join(shell_quote(part) for part in command) + "\n")

    print(f"Running Umbra benchmark; output directory: {out_dir}", file=sys.stderr)
    run = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    loaded_csv.unlink(missing_ok=True)
    output = run.stdout or ""
    (out_dir / "umbra_output.txt").write_text(output)
    if run.returncode != 0:
        print(output, file=sys.stderr)
        return run.returncode

    try:
        raw_rows = parse_umbra_output(
            output,
            setup_statement_count=2 + (1 if args.create_index else 0),
            query_plan=query_plan,
            data_spec=data_spec,
            data_csv=args.data_csv,
            load_stats=load_stats,
            create_index=args.create_index,
        )
    except ValueError as exc:
        (out_dir / "parse_error.txt").write_text(str(exc) + "\n")
        print(f"failed to parse Umbra output: {exc}", file=sys.stderr)
        return 2

    raw_csv = out_dir / "raw.csv"
    summary_csv = out_dir / "summary.csv"
    write_csv(raw_csv, RAW_FIELDS, raw_rows)
    write_csv(summary_csv, SUMMARY_FIELDS, summarize(raw_rows))
    print(f"Done. Results in: {out_dir}")
    return 0


def load_data_specs(path: Path) -> list[DataSpec]:
    specs = []
    with path.open(newline="") as f:
        for idx, row in enumerate(csv.DictReader(f), start=2):
            if not parse_boolish(row.get("enabled", "true")):
                continue
            data_type = normalize_name(row.get("type") or "dna-fasta")
            data_path = Path(row.get("path") or "")
            if not str(data_path):
                raise SystemExit(f"data CSV row {idx} has no path")
            file_stem = data_path.stem or "dataset"
            specs.append(
                DataSpec(
                    name=row.get("name") or file_stem,
                    path=data_path,
                    data_type=data_type,
                    column=row.get("column") or ("sequence" if data_type in {"dna-fasta", "protein-fasta"} else file_stem),
                    key_column=row.get("key_column") or None,
                    value_column=row.get("value_column") or None,
                )
            )
    if not specs:
        raise SystemExit(f"data CSV {path} produced no enabled datasets")
    return specs


def select_data_spec(specs: list[DataSpec], data_name: str | None) -> DataSpec:
    if data_name:
        matches = [spec for spec in specs if spec.name == data_name]
        if not matches:
            names = ", ".join(spec.name for spec in specs)
            raise SystemExit(f"--data-name {data_name!r} did not match enabled data rows: {names}")
        return matches[0]
    if len(specs) != 1:
        names = ", ".join(spec.name for spec in specs)
        raise SystemExit(f"data CSV has multiple enabled rows; choose one with --data-name. Available: {names}")
    return specs[0]


def load_patterns(path: Path) -> list[PatternSpec]:
    patterns = []
    with path.open(newline="") as f:
        for idx, row in enumerate(csv.DictReader(f), start=2):
            if not parse_boolish(row.get("enabled", "true")):
                continue
            name = row.get("name") or f"pattern_{idx}"
            if "pattern" not in row:
                raise SystemExit(f"pattern CSV row {idx} has no pattern column")
            patterns.append(PatternSpec(name=name, pattern=row["pattern"]))
    return patterns


def write_dataset_paths(inputs_dir: Path, data_csv: Path, data_spec: DataSpec) -> None:
    data_path = data_csv.parent / data_spec.path
    with (inputs_dir / "datasets.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "column",
                "data_manifest",
                "data_path",
                "data_path_resolved",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": data_spec.name,
                "column": data_spec.column,
                "data_manifest": str(data_csv),
                "data_path": str(data_path),
                "data_path_resolved": str(data_path.resolve()),
            }
        )


def write_umbra_input_csv(
    spec: DataSpec,
    data_base: Path,
    out_path: Path,
    *,
    max_rows: int | None,
    max_total_bytes: int,
    max_row_bytes: int,
    row_overflow_policy: str,
    uppercase_sequences: bool,
) -> LoadStats:
    source = spec.path if spec.path.is_absolute() else data_base / spec.path
    if spec.data_type in {"dna-fasta", "protein-fasta", "fasta", "dna", "fa", "fna", "aa-fasta", "faa", "pep"}:
        return write_fasta_csv(
            source,
            out_path,
            max_rows=max_rows,
            max_total_bytes=max_total_bytes,
            max_row_bytes=max_row_bytes,
            row_overflow_policy=row_overflow_policy,
            uppercase_sequences=uppercase_sequences,
        )
    if spec.data_type in {"job-csv", "csv"}:
        return write_job_csv(
            source,
            out_path,
            key_column=spec.key_column,
            value_column=spec.value_column,
            max_rows=max_rows,
            max_total_bytes=max_total_bytes,
            max_row_bytes=max_row_bytes,
            row_overflow_policy=row_overflow_policy,
        )
    raise SystemExit(f"unsupported data type for Umbra benchmark: {spec.data_type}")


def write_fasta_csv(
    source: Path,
    out_path: Path,
    *,
    max_rows: int | None,
    max_total_bytes: int,
    max_row_bytes: int,
    row_overflow_policy: str,
    uppercase_sequences: bool,
) -> LoadStats:
    stats = LoadStats()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as inp, out_path.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["key", "value"])

        header: str | None = None
        seq = bytearray()
        original_len = 0
        over_row_limit = False

        def flush() -> bool:
            nonlocal header, seq, original_len, over_row_limit
            if header is None:
                return False
            stop = append_record(
                writer,
                header,
                bytes(seq),
                original_len,
                over_row_limit,
                stats,
                max_rows=max_rows,
                max_total_bytes=max_total_bytes,
                row_overflow_policy=row_overflow_policy,
                uppercase_sequences=uppercase_sequences,
            )
            header = None
            seq = bytearray()
            original_len = 0
            over_row_limit = False
            return stop

        for line_no, raw_line in enumerate(inp, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(b">"):
                if flush():
                    break
                header = line[1:].decode("utf-8", errors="replace").strip()
                continue
            if header is None:
                raise SystemExit(f"sequence data appears before first FASTA header at line {line_no}")
            for byte in raw_line:
                if chr(byte).isspace():
                    continue
                original_len += 1
                if original_len > max_row_bytes:
                    over_row_limit = True
                    if row_overflow_policy == "error":
                        raise SystemExit(f"FASTA record exceeds --max-row-bytes ({max_row_bytes})")
                    continue
                seq.append(byte)
        flush()
    return stats


def append_record(
    writer: csv.writer,
    header: str,
    seq: bytes,
    original_len: int,
    over_row_limit: bool,
    stats: LoadStats,
    *,
    max_rows: int | None,
    max_total_bytes: int,
    row_overflow_policy: str,
    uppercase_sequences: bool,
) -> bool:
    if stats.stopped_by_max_rows or stats.stopped_by_max_total_bytes:
        return True
    stats.records_seen += 1
    stats.total_input_sequence_bytes += original_len
    if max_rows is not None and stats.records_loaded >= max_rows:
        stats.stopped_by_max_rows = True
        return True
    if over_row_limit and row_overflow_policy == "skip":
        stats.records_skipped += 1
        return False

    loaded = seq.upper() if uppercase_sequences else seq
    if over_row_limit:
        stats.records_truncated += 1
    loaded = truncate_to_total_limit(loaded, stats, max_total_bytes)
    if not loaded and original_len > 0:
        stats.records_skipped += 1
        return stats.stopped_by_max_total_bytes

    key = header.split(None, 1)[0] if header else ""
    writer.writerow([key, loaded.decode("utf-8", errors="replace")])
    stats.records_loaded += 1
    stats.total_loaded_symbols += len(loaded)
    return stats.stopped_by_max_total_bytes


def write_job_csv(
    source: Path,
    out_path: Path,
    *,
    key_column: str | None,
    value_column: str | None,
    max_rows: int | None,
    max_total_bytes: int,
    max_row_bytes: int,
    row_overflow_policy: str,
) -> LoadStats:
    stats = LoadStats()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="") as inp, out_path.open("w", newline="") as out:
        reader = csv.DictReader(inp)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {source}")
        key_name = select_csv_column(reader.fieldnames, key_column, ["key", "id", "row_id"], 0)
        value_name = select_csv_column(reader.fieldnames, value_column, ["value", "text", "data"], 1)
        writer = csv.writer(out)
        writer.writerow(["key", "value"])
        for row in reader:
            if stats.stopped_by_max_rows or stats.stopped_by_max_total_bytes:
                break
            stats.records_seen += 1
            if max_rows is not None and stats.records_loaded >= max_rows:
                stats.stopped_by_max_rows = True
                break

            key = row.get(key_name, "")
            value = row.get(value_name, "")
            stats.total_input_sequence_bytes += len(value.encode("utf-8"))
            loaded = value.encode("utf-8")
            if len(loaded) > max_row_bytes:
                if row_overflow_policy == "truncate":
                    loaded = loaded[:max_row_bytes]
                    stats.records_truncated += 1
                elif row_overflow_policy == "skip":
                    stats.records_skipped += 1
                    continue
                else:
                    raise SystemExit(f"CSV row exceeds --max-row-bytes ({max_row_bytes}) in {source}")
            loaded = truncate_to_total_limit(loaded, stats, max_total_bytes)
            if not loaded and value:
                stats.records_skipped += 1
                continue
            writer.writerow([key, loaded.decode("utf-8", errors="replace")])
            stats.records_loaded += 1
            stats.total_loaded_symbols += len(loaded)
    return stats


def truncate_to_total_limit(data: bytes, stats: LoadStats, max_total: int) -> bytes:
    if stats.total_loaded_symbols >= max_total:
        stats.stopped_by_max_total_bytes = True
        return b""
    remaining = max_total - stats.total_loaded_symbols
    if len(data) > remaining:
        stats.records_truncated += 1
        stats.stopped_by_max_total_bytes = True
        return data[:remaining]
    return data


def select_csv_column(headers: list[str], requested: str | None, fallbacks: list[str], fallback_idx: int) -> str:
    if requested:
        if requested in headers:
            return requested
        raise SystemExit(f"CSV column {requested!r} not found; headers are {headers}")
    lower = {h.lower(): h for h in headers}
    for name in fallbacks:
        if name.lower() in lower:
            return lower[name.lower()]
    if fallback_idx < len(headers):
        return headers[fallback_idx]
    raise SystemExit(f"CSV has too few columns; headers are {headers}")


def write_load_sql(path: Path, input_csv: str, create_index: bool) -> None:
    lines = [
        f"create table {TABLE_NAME}(key text, value text);",
        f"copy {TABLE_NAME} from {sql_literal(input_csv)} with (format csv, header true);",
    ]
    if create_index:
        lines.append(f"create index {TABLE_NAME}_value_idx on {TABLE_NAME}(value);")
    path.write_text("\n".join(lines) + "\n")


def write_queries_sql(path: Path, patterns: list[PatternSpec], warmups: int, iterations: int) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = [{"kind": "row_count"}]
    lines = [f"select count(*) from {TABLE_NAME};"]
    for pattern in patterns:
        for iteration in range(warmups):
            plan.append({"kind": "warmup", "pattern": pattern, "iteration": iteration})
            lines.append(f"select count(*) from {TABLE_NAME} where value like {sql_literal(pattern.pattern)};")
        for iteration in range(iterations):
            plan.append({"kind": "measure", "pattern": pattern, "iteration": iteration})
            lines.append(f"select count(*) from {TABLE_NAME} where value like {sql_literal(pattern.pattern)};")
    path.write_text("\n".join(lines) + "\n")
    return plan


def parse_umbra_output(
    output: str,
    *,
    setup_statement_count: int,
    query_plan: list[dict[str, object]],
    data_spec: DataSpec,
    data_csv: Path,
    load_stats: LoadStats,
    create_index: bool,
) -> list[dict[str, object]]:
    infos = [parse_info_line(line) for line in output.splitlines() if "INFO:" in line and "execution:" in line]
    counts = [int(line.strip()) for line in output.splitlines() if re.fullmatch(r"\d+", line.strip())]
    if any(info is None for info in infos):
        raise ValueError("could not parse at least one INFO timing line")
    timings = [info for info in infos if info is not None]
    expected_infos = setup_statement_count + len(query_plan)
    if len(timings) != expected_infos:
        raise ValueError(f"expected {expected_infos} INFO lines, found {len(timings)}")
    if len(counts) != len(query_plan):
        raise ValueError(f"expected {len(query_plan)} count rows, found {len(counts)}")

    setup_infos = timings[:setup_statement_count]
    query_infos = timings[setup_statement_count:]
    row_count = counts[0]
    load_ns = sum(info[0] for info in setup_infos)
    index_build_ns = setup_infos[-1][0] if create_index else 0
    raw_rows = []
    requested_index = "btree" if create_index else "none"
    actual_index = "btree" if create_index else "none"
    for item, rows_matched, (execution_ns, compilation_ns) in zip(query_plan, counts, query_infos):
        if item["kind"] != "measure":
            continue
        pattern = item["pattern"]
        assert isinstance(pattern, PatternSpec)
        raw_rows.append(
            {
                "dataset": data_spec.name,
                "column": data_spec.column,
                "data_path": str(data_csv.parent / data_spec.path),
                "data_type": data_spec.data_type,
                "storage": "umbra",
                "algorithm": "UmbraLike",
                "generic_matcher": "sql",
                "requested_index": requested_index,
                "actual_index": actual_index,
                "pattern_name": pattern.name,
                "pattern": pattern.pattern,
                "iteration": item["iteration"],
                "row_count": row_count,
                "total_input_sequence_bytes": load_stats.total_input_sequence_bytes,
                "total_loaded_symbols": load_stats.total_loaded_symbols,
                "records_seen": load_stats.records_seen,
                "records_loaded": load_stats.records_loaded,
                "records_skipped": load_stats.records_skipped,
                "records_truncated": load_stats.records_truncated,
                "records_invalid_dna": load_stats.records_invalid_dna,
                "load_ns": load_ns,
                "index_build_ns": index_build_ns,
                "index_size_bytes": 0,
                "compile_ns": compilation_ns,
                "candidate_prepare_ns": 0,
                "execute_ns": execution_ns,
                "query_total_ns": execution_ns,
                "candidate_rows_seen": row_count,
                "rows_after_len_filter": row_count,
                "rows_matched": rows_matched,
                "ns_per_table_row": execution_ns / row_count if row_count else math.nan,
                "ns_per_candidate_row": execution_ns / row_count if row_count else math.nan,
                "ns_per_loaded_symbol": execution_ns / load_stats.total_loaded_symbols if load_stats.total_loaded_symbols else math.nan,
                "fallback_reason": "",
            }
        )
    return raw_rows


def parse_info_line(line: str) -> tuple[int, int] | None:
    match = INFO_RE.search(line)
    if not match:
        return None
    execution_s = parse_metric_body(match.group("execution"), "median")
    compilation_s = parse_metric_body(match.group("compilation"), "median")
    return round(execution_s * 1_000_000_000), round(compilation_s * 1_000_000_000)


def parse_metric_body(body: str, metric: str) -> float:
    for part in body.split(","):
        fields = part.strip().split()
        if len(fields) >= 2 and fields[1] == metric:
            return float(fields[0])
    raise ValueError(f"INFO metric {metric!r} not found in {body!r}")


RAW_FIELDS = [
    "dataset",
    "column",
    "data_path",
    "data_type",
    "storage",
    "algorithm",
    "generic_matcher",
    "requested_index",
    "actual_index",
    "pattern_name",
    "pattern",
    "iteration",
    "row_count",
    "total_input_sequence_bytes",
    "total_loaded_symbols",
    "records_seen",
    "records_loaded",
    "records_skipped",
    "records_truncated",
    "records_invalid_dna",
    "load_ns",
    "index_build_ns",
    "index_size_bytes",
    "compile_ns",
    "candidate_prepare_ns",
    "execute_ns",
    "query_total_ns",
    "candidate_rows_seen",
    "rows_after_len_filter",
    "rows_matched",
    "ns_per_table_row",
    "ns_per_candidate_row",
    "ns_per_loaded_symbol",
    "fallback_reason",
]

SUMMARY_FIELDS = [
    "dataset",
    "column",
    "storage",
    "algorithm",
    "generic_matcher",
    "requested_index",
    "actual_index",
    "pattern_name",
    "pattern",
    "runs",
    "row_count",
    "rows_matched",
    "index_build_ns",
    "index_size_bytes",
    "sum_query_total_ns",
    "min_query_total_ns",
    "median_query_total_ns",
    "mean_query_total_ns",
    "geomean_query_total_ns",
    "p90_query_total_ns",
    "max_query_total_ns",
    "median_execute_ns",
    "median_compile_ns",
    "median_candidate_prepare_ns",
    "median_ns_per_table_row",
    "geomean_ns_per_table_row",
    "median_ns_per_candidate_row",
]


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[field] for field in SUMMARY_FIELDS[:9])
        groups.setdefault(key, []).append(row)
    out = []
    for key, group in groups.items():
        totals = [as_float(row["query_total_ns"]) for row in group]
        execs = [as_float(row["execute_ns"]) for row in group]
        compiles = [as_float(row["compile_ns"]) for row in group]
        preps = [as_float(row["candidate_prepare_ns"]) for row in group]
        per_rows = [as_float(row["ns_per_table_row"]) for row in group]
        per_candidates = [as_float(row["ns_per_candidate_row"]) for row in group]
        out.append(
            dict(
                zip(SUMMARY_FIELDS[:9], key),
                runs=len(group),
                row_count=group[0]["row_count"],
                rows_matched=group[0]["rows_matched"],
                index_build_ns=group[0]["index_build_ns"],
                index_size_bytes=group[0]["index_size_bytes"],
                sum_query_total_ns=sum(totals),
                min_query_total_ns=min(totals),
                median_query_total_ns=median(totals),
                mean_query_total_ns=sum(totals) / len(totals),
                geomean_query_total_ns=geomean(totals),
                p90_query_total_ns=percentile(totals, 0.90),
                max_query_total_ns=max(totals),
                median_execute_ns=median(execs),
                median_compile_ns=median(compiles),
                median_candidate_prepare_ns=median(preps),
                median_ns_per_table_row=median(per_rows),
                geomean_ns_per_table_row=geomean(per_rows),
                median_ns_per_candidate_row=median(per_candidates),
            )
        )
    out.sort(key=lambda row: (as_float(row["median_query_total_ns"]), str(row["pattern_name"])))
    return out


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_bytes(value: str) -> int:
    value = value.strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([A-Za-z]*)", value)
    if not match:
        raise SystemExit(f"invalid byte value: {value!r}")
    number = float(match.group(1))
    unit = match.group(2).lower()
    multipliers = {
        "": 1,
        "b": 1,
        "byte": 1,
        "bytes": 1,
        "k": 1000,
        "kb": 1000,
        "m": 1000**2,
        "mb": 1000**2,
        "g": 1000**3,
        "gb": 1000**3,
        "t": 1000**4,
        "tb": 1000**4,
        "ki": 1024,
        "kib": 1024,
        "mi": 1024**2,
        "mib": 1024**2,
        "gi": 1024**3,
        "gib": 1024**3,
        "ti": 1024**4,
        "tib": 1024**4,
    }
    if unit not in multipliers:
        raise SystemExit(f"unknown byte unit in {value!r}")
    return math.floor(number * multipliers[unit])


def parse_boolish(value: str) -> bool:
    return normalize_name(value) in {"1", "true", "yes", "y", "on", "enabled"}


def normalize_name(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "umbra"


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def container_path(cwd: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(cwd)
    except ValueError as exc:
        raise SystemExit(f"path {resolved} is not under working directory {cwd}") from exc
    return "/work/" + relative.as_posix()


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "-._/:=+" for ch in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def as_float(value: object) -> float:
    return float(value)


def geomean(values: list[float]) -> float:
    xs = [value for value in values if value > 0 and math.isfinite(value)]
    if not xs:
        return math.nan
    return math.exp(sum(math.log(value) for value in xs) / len(xs))


def percentile(values: list[float], q: float) -> float:
    xs = sorted(value for value in values if math.isfinite(value))
    if not xs:
        return math.nan
    idx = min(len(xs) - 1, max(0, math.ceil(len(xs) * q) - 1))
    return xs[idx]


if __name__ == "__main__":
    raise SystemExit(main())
