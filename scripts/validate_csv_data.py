#!/usr/bin/env python3
"""Validate benchmark CSV files produced by the data pipeline."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from itertools import zip_longest
from pathlib import Path


RAW_DELIMITER = "|"
KV_HEADER = ["key", "value"]
MANIFEST_HEADER = ["name", "path", "type", "storage", "column", "key_column", "value_column", "enabled"]
HEADERED_RAW_DATASETS = {"quotes", "spam"}
MISSING = object()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate benchmark CSV data files")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--raw-root", type=Path, help="Defaults to <data-root>/raw")
    parser.add_argument("--datasets", default="tpch,tpcds,job,quotes,spam")
    parser.add_argument("--skip-raw", action="store_true", help="Do not validate raw tables")
    parser.add_argument("--skip-prepared", action="store_true", help="Do not validate prepared CSVs")
    parser.add_argument(
        "--no-compare-prepared",
        action="store_true",
        help="Skip prepared-vs-raw value comparison",
    )
    parser.add_argument("--max-errors", type=int, default=50)
    return parser.parse_args()


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


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


def clean_identifier(value: str) -> str:
    return value.strip().strip('"')


def parse_schema_file(path: Path) -> dict[str, list[str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, list[str]] = {}
    for match in re.finditer(r"CREATE\s+TABLE\s+([\w\"]+)\s*\((.*?)\)\s*;", text, re.I | re.S):
        table = clean_identifier(match.group(1))
        columns: list[str] = []
        for raw_col in split_sql_columns(match.group(2)):
            raw_col = " ".join(raw_col.strip().split())
            if not raw_col:
                continue
            first = raw_col.split(" ", 1)[0]
            if first.upper() in {"PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT"}:
                continue
            columns.append(clean_identifier(first))
        out[table] = columns
    return out


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


def source_key_index(columns: list[str]) -> int | None:
    for idx, column in enumerate(columns):
        if column == "id" or column.endswith("_key") or column.endswith("_sk"):
            return idx
    return 0 if columns else None


def validate_manifest(path: Path, errors: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != MANIFEST_HEADER:
                errors.append(f"{path}: header {reader.fieldnames!r}, expected {MANIFEST_HEADER!r}")
                return rows
            for row_number, row in enumerate(reader, start=2):
                rows.append(row)
                target = path.parent / row["path"]
                if row["path"] and not target.exists():
                    errors.append(f"{path}: row {row_number} references missing path {target}")
    except (csv.Error, OSError, UnicodeDecodeError) as exc:
        errors.append(f"{path}: manifest parse failed: {exc}")
    return rows


def validate_kv_csv(path: Path, errors: list[str]) -> int:
    rows = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, None)
            if header != KV_HEADER:
                errors.append(f"{path}: header {header!r}, expected {KV_HEADER!r}")
                return rows
            for row_number, row in enumerate(reader, start=2):
                rows += 1
                if len(row) != 2:
                    errors.append(f"{path}: row {row_number} has {len(row)} fields, expected 2")
    except (csv.Error, OSError, UnicodeDecodeError) as exc:
        errors.append(f"{path}: CSV parse failed: {exc}")
    return rows


def validate_raw_table(path: Path, expected_columns: int, errors: list[str]) -> int:
    rows = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter=RAW_DELIMITER, strict=True)
            for row_number, row in enumerate(reader, start=1):
                rows += 1
                if len(row) != expected_columns:
                    errors.append(
                        f"{path}: row {row_number} has {len(row)} fields, "
                        f"expected {expected_columns}"
                    )
                    break
    except csv.Error as exc:
        errors.append(f"{path}: raw CSV parse failed after row {rows}: {exc}")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"{path}: raw CSV read failed: {exc}")
    return rows


def iter_prepared_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader)
        if header != KV_HEADER:
            raise RuntimeError(f"header {header!r}, expected {KV_HEADER!r}")
        for row in reader:
            if len(row) != 2:
                raise RuntimeError(f"row has {len(row)} fields, expected 2")
            yield row[0], row[1]


def iter_raw_values(path: Path, columns: list[str], column: str):
    column_idx = columns.index(column)
    key_idx = source_key_index(columns)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=RAW_DELIMITER, strict=True)
        for row_number, row in enumerate(reader, start=1):
            if len(row) != len(columns):
                raise RuntimeError(f"raw row {row_number} has {len(row)} fields, expected {len(columns)}")
            key = row[key_idx] if key_idx is not None and key_idx < len(row) else str(row_number - 1)
            yield key, row[column_idx]


def compare_prepared_to_raw(
    prepared_path: Path,
    raw_path: Path,
    columns: list[str],
    column: str,
    errors: list[str],
) -> int:
    rows = 0
    try:
        pairs = zip_longest(
            iter_prepared_rows(prepared_path),
            iter_raw_values(raw_path, columns, column),
            fillvalue=MISSING,
        )
        for rows, (prepared, raw) in enumerate(pairs, start=1):
            if prepared is MISSING:
                errors.append(f"{prepared_path}: missing prepared row for raw row {rows}")
                break
            if raw is MISSING:
                errors.append(f"{prepared_path}: extra prepared row {rows}")
                break
            if prepared != raw:
                errors.append(
                    f"{prepared_path}: row {rows} differs from raw {raw_path}; "
                    f"prepared={prepared!r}, raw={raw!r}"
                )
                break
    except (csv.Error, OSError, UnicodeDecodeError, RuntimeError) as exc:
        errors.append(f"{prepared_path}: prepared-vs-raw comparison failed: {exc}")
    return rows


def main() -> None:
    raise_csv_field_limit()
    args = parse_args()
    data_root = args.data_root
    raw_root = args.raw_root or data_root / "raw"
    datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    errors: list[str] = []

    schemas: dict[str, dict[str, list[str]]] = {}
    if not args.skip_raw:
        print("raw tables:")
        for dataset in datasets:
            dataset_dir = raw_root / dataset
            schema_path = find_schema_path(dataset_dir)
            if schema_path is None:
                if dataset in HEADERED_RAW_DATASETS:
                    print(f"  {dataset}: schema-less raw CSV dataset")
                    continue
                errors.append(f"{dataset_dir}: missing schema.sql/schematext.sql")
                continue
            schema = parse_schema_file(schema_path)
            schemas[dataset] = schema
            for table, columns in sorted(schema.items()):
                table_path = find_table_file(dataset_dir, table)
                if table_path is None:
                    errors.append(f"{dataset}.{table}: missing raw table file")
                    continue
                rows = validate_raw_table(table_path, len(columns), errors)
                print(f"  {dataset}.{table}: {rows} rows")

    if not args.skip_prepared:
        print("prepared CSVs:")
        all_manifest = data_root / "data_all.csv"
        if all_manifest.exists():
            rows = validate_manifest(all_manifest, errors)
            print(f"  {all_manifest}: {len(rows)} manifest rows")
        for dataset in datasets:
            manifest_path = data_root / dataset / "data.csv"
            if not manifest_path.exists():
                errors.append(f"{manifest_path}: missing manifest")
                continue
            manifest_rows = validate_manifest(manifest_path, errors)
            print(f"  {manifest_path}: {len(manifest_rows)} manifest rows")
            for row in manifest_rows:
                prepared_path = manifest_path.parent / row["path"]
                if row["type"] != "job-csv" or not prepared_path.exists():
                    continue
                prepared_rows = validate_kv_csv(prepared_path, errors)
                print(f"    {prepared_path}: {prepared_rows} rows")
                if args.no_compare_prepared or args.skip_raw:
                    continue
                if dataset in HEADERED_RAW_DATASETS:
                    continue
                table, column = row["column"].split(".", 1)
                schema = schemas.get(dataset)
                if schema is None:
                    schema_path = find_schema_path(raw_root / dataset)
                    if schema_path is None:
                        errors.append(f"{raw_root / dataset}: missing schema.sql/schematext.sql")
                        continue
                    schema = parse_schema_file(schema_path)
                raw_path = find_table_file(raw_root / dataset, table)
                if raw_path is None:
                    errors.append(f"{dataset}.{table}.{column}: missing raw table file")
                    continue
                compared = compare_prepared_to_raw(
                    prepared_path,
                    raw_path,
                    schema[table],
                    column,
                    errors,
                )
                print(f"      compared {compared} rows with raw")

    if errors:
        print("\nerrors:")
        for error in errors[: args.max_errors]:
            print(f"  {error}")
        if len(errors) > args.max_errors:
            print(f"  ... {len(errors) - args.max_errors} more")
        raise SystemExit(1)

    print("\nCSV validation passed.")


if __name__ == "__main__":
    main()
