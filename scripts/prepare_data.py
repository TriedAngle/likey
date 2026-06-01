#!/usr/bin/env python3
"""Prepare downloaded raw benchmark data for the likey2 runner.

Input layout:
  data/raw/tpch/...   DuckDB TPC-H export
  data/raw/tpcds/...  DuckDB TPC-DS export
  data/raw/job/...    JOB/IMDB export
  data/fasta/...      FASTA files, already runner-readable

Output layout:
  data/tpch/*.csv     one headered key,value file per selected text column
  data/tpcds/*.csv
  data/job/*.csv
  data/<dataset>/data.csv runner manifest for that dataset
  data/data_all.csv combined manifest
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


STRING_TYPE_RE = re.compile(r"\b(character varying|varchar|char|text|string)\b", re.IGNORECASE)

DEFAULT_COLUMNS: dict[str, set[str]] = {
    "tpch": {
        "tpch.customer.c_comment",
        "tpch.lineitem.l_comment",
        "tpch.orders.o_comment",
        "tpch.part.p_name",
        "tpch.part.p_type",
        "tpch.part.p_container",
        "tpch.partsupp.ps_comment",
    },
    "tpcds": {
        "tpcds.call_center.cc_class",
        "tpcds.catalog_page.cp_description",
        "tpcds.customer.c_last_name",
        "tpcds.customer_address.ca_street_name",
        "tpcds.date_dim.d_day_name",
        "tpcds.item.i_item_desc",
        "tpcds.item.i_color",
        "tpcds.promotion.p_channel_details",
        "tpcds.store.s_market_desc",
        "tpcds.web_site.web_mkt_desc",
    },
    "job": {
        "job.cast_info.note",
        "job.keyword.keyword",
        "job.movie_info.info",
        "job.name.name",
        "job.title.title",
    },
}


@dataclass(frozen=True)
class Column:
    name: str
    type_sql: str
    index: int


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: list[Column]


@dataclass(frozen=True)
class ManifestRow:
    name: str
    path: str
    data_type: str
    storage: str
    column: str
    key_column: str = "key"
    value_column: str = "value"
    enabled: str = "true"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare benchmark data for likey2")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--fasta-dir", type=Path, default=Path("data/fasta"))
    parser.add_argument(
        "--datasets",
        default="tpch,tpcds,job,dna,protein",
        help="Comma-separated datasets to prepare",
    )
    parser.add_argument("--storage", default="all", help="Storage value for generated runner rows")
    parser.add_argument("--max-rows", type=int, help="Optional row cap per extracted column")
    parser.add_argument(
        "--columns",
        help="Optional comma-separated allow-list like tpch.part.p_name,job.title.title",
    )
    parser.add_argument(
        "--all-string-columns",
        action="store_true",
        help="Extract every string column instead of the curated useful set",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite prepared CSV files")
    return parser.parse_args()


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def parse_schema_file(path: Path) -> dict[str, TableSchema]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, TableSchema] = {}
    for match in re.finditer(r"CREATE\s+TABLE\s+([\w\"]+)\s*\((.*?)\)\s*;", text, re.I | re.S):
        table = clean_identifier(match.group(1))
        columns = [
            parsed
            for idx, raw_col in enumerate(split_sql_columns(match.group(2)))
            if (parsed := parse_column(raw_col, idx)) is not None
        ]
        out[table] = TableSchema(table, columns)
    return out


def split_sql_columns(body: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for idx, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(body[start:idx].strip())
            start = idx + 1
    tail = body[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def parse_column(raw: str, index: int) -> Column | None:
    raw = " ".join(raw.strip().split())
    if not raw:
        return None
    first = raw.split(" ", 1)[0]
    if first.upper() in {"PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT"}:
        return None
    return Column(clean_identifier(first), raw[len(first) :].strip(), index)


def clean_identifier(value: str) -> str:
    return value.strip().strip('"')


def is_string_column(column: Column) -> bool:
    return bool(STRING_TYPE_RE.search(column.type_sql))


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "value"


def relpath(path: Path, base: Path) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


def find_schema_path(dataset_dir: Path) -> Path | None:
    for name in ("schema.sql", "schematext.sql"):
        path = dataset_dir / name
        if path.exists():
            return path
    return None


def find_table_file(dataset_dir: Path, table: str) -> Path | None:
    for suffix in (".csv", ".tbl", ".dat", ".tsv", ""):
        path = dataset_dir / f"{table}{suffix}"
        if path.exists() and path.is_file():
            return path
        nested = dataset_dir / "imdb" / f"{table}{suffix}"
        if nested.exists() and nested.is_file():
            return nested
    return None


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line:
                return "|" if line.count("|") >= line.count(",") else ","
    return "|"


def parse_column_filter(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or None


def selected_columns(dataset: str, args: argparse.Namespace) -> set[str] | None:
    if args.columns:
        return parse_column_filter(args.columns)
    if args.all_string_columns:
        return None
    return DEFAULT_COLUMNS.get(dataset, set())


def source_key_index(schema: TableSchema) -> int | None:
    for column in schema.columns:
        if column.name == "id" or column.name.endswith("_key") or column.name.endswith("_sk"):
            return column.index
    return 0 if schema.columns else None


def extract_column(
    dataset: str,
    schema: TableSchema,
    source_path: Path,
    column: Column,
    key_idx: int | None,
    out_dir: Path,
    manifest_dir: Path,
    storage: str,
    max_rows: int | None,
    force: bool,
) -> ManifestRow:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sanitize(schema.name)}__{sanitize(column.name)}.csv"
    if out_path.exists() and not force:
        return manifest_row(dataset, schema.name, column.name, out_path, manifest_dir, storage)

    written = 0
    delimiter = detect_delimiter(source_path)
    with source_path.open("r", encoding="utf-8", errors="replace", newline="") as src:
        reader = csv.reader(src, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        with out_path.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.writer(dst)
            writer.writerow(["key", "value"])
            for row_idx, row in enumerate(reader):
                if max_rows is not None and written >= max_rows:
                    break
                if column.index >= len(row):
                    continue
                key = row[key_idx] if key_idx is not None and key_idx < len(row) else str(row_idx)
                writer.writerow([key, row[column.index]])
                written += 1
    print(f"extracted {dataset}.{schema.name}.{column.name}: {written} rows -> {out_path}")
    return manifest_row(dataset, schema.name, column.name, out_path, manifest_dir, storage)


def manifest_row(
    dataset: str,
    table: str,
    column: str,
    out_path: Path,
    manifest_dir: Path,
    storage: str,
) -> ManifestRow:
    return ManifestRow(
        name=sanitize(f"{dataset}_{table}_{column}"),
        path=relpath(out_path, manifest_dir),
        data_type="job-csv",
        storage=storage,
        column=sanitize(f"{table}.{column}"),
    )


def prepare_relational_dataset(dataset: str, args: argparse.Namespace) -> list[ManifestRow]:
    dataset_dir = args.raw_root / dataset
    schema_path = find_schema_path(dataset_dir)
    if schema_path is None:
        print(f"skipping {dataset}: no schema.sql/schematext.sql in {dataset_dir}")
        return []

    wanted = selected_columns(dataset, args)
    out_dir = args.data_root / dataset
    rows: list[ManifestRow] = []
    for table, schema in sorted(parse_schema_file(schema_path).items()):
        source_path = find_table_file(dataset_dir, table)
        if source_path is None:
            continue
        key_idx = source_key_index(schema)
        for column in schema.columns:
            full_name = f"{dataset}.{table}.{column.name}"
            if wanted is not None and full_name not in wanted:
                continue
            if not is_string_column(column):
                continue
            rows.append(
                extract_column(
                    dataset,
                    schema,
                    source_path,
                    column,
                    key_idx,
                    out_dir,
                    out_dir,
                    args.storage,
                    args.max_rows,
                    args.force,
                )
            )
    write_manifest(out_dir / "data.csv", rows)
    return rows


def prepare_fasta_manifests(args: argparse.Namespace) -> list[ManifestRow]:
    specs = [
        ("dna", "dna_benchmark.fna", "dna-fasta", "sequence"),
        ("protein", "protein_benchmark.faa", "protein-fasta", "sequence"),
    ]
    rows: list[ManifestRow] = []
    local_rows: list[ManifestRow] = []
    for dataset, filename, data_type, column in specs:
        path = args.fasta_dir / filename
        if not path.exists():
            continue
        local_rows.append(
            ManifestRow(
                name=dataset,
                path=relpath(path, args.fasta_dir),
                data_type=data_type,
                storage=args.storage,
                column=column,
                key_column="",
                value_column="",
            )
        )
        rows.append(
            ManifestRow(
                name=dataset,
                path=relpath(path, args.data_root),
                data_type=data_type,
                storage=args.storage,
                column=column,
                key_column="",
                value_column="",
            )
        )
    if local_rows:
        write_manifest(args.fasta_dir / "data.csv", local_rows)
    return rows


def write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["name", "path", "type", "storage", "column", "key_column", "value_column", "enabled"]
        )
        for row in rows:
            writer.writerow(
                [
                    row.name,
                    row.path,
                    row.data_type,
                    row.storage,
                    row.column,
                    row.key_column,
                    row.value_column,
                    row.enabled,
                ]
            )
    print(f"wrote manifest: {path} ({len(rows)} rows)")


def main() -> None:
    raise_csv_field_limit()
    args = parse_args()
    datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    args.data_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[ManifestRow] = []
    for dataset in datasets:
        if dataset in {"tpch", "tpcds", "job"}:
            rows = prepare_relational_dataset(dataset, args)
            all_rows.extend(
                ManifestRow(
                    name=row.name,
                    path=relpath((args.data_root / dataset / row.path).resolve(), args.data_root.resolve()),
                    data_type=row.data_type,
                    storage=row.storage,
                    column=row.column,
                    key_column=row.key_column,
                    value_column=row.value_column,
                    enabled=row.enabled,
                )
                for row in rows
            )

    if any(dataset in {"dna", "protein"} for dataset in datasets):
        all_rows.extend(prepare_fasta_manifests(args))

    write_manifest(args.data_root / "data_all.csv", all_rows)


if __name__ == "__main__":
    main()
