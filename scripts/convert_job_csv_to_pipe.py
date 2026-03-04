#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path


DEFAULT_TABLES = ["title", "name", "movie_info", "keyword", "cast_info"]


def ensure_large_csv_field_limit() -> None:
    limit = 1024 * 1024
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2
            if limit <= 1024:
                raise


def looks_pipe_delimited(path: Path) -> bool:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for _ in range(50):
            line = handle.readline()
            if not line:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            return line.count("|") > line.count(",")
    return False


def convert_file(path: Path, backup: bool) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    backup_path = path.with_suffix(path.suffix + ".comma.bak")

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as src:
        reader = csv.reader(src, delimiter=",")
        with open(tmp_path, "w", encoding="utf-8", newline="") as dst:
            writer = csv.writer(
                dst,
                delimiter="|",
                quotechar='"',
                lineterminator="\n",
                quoting=csv.QUOTE_MINIMAL,
            )
            for row in reader:
                writer.writerow(row)

    if backup:
        if backup_path.exists():
            backup_path.unlink()
        os.replace(path, backup_path)
    else:
        path.unlink()

    os.replace(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert JOB comma-delimited CSV files to pipe-delimited format"
    )
    parser.add_argument("--data-dir", default="data/benchmarks/job")
    parser.add_argument(
        "--tables",
        default=",".join(DEFAULT_TABLES),
        help="Comma-separated table base names to convert",
    )
    parser.add_argument(
        "--keep-backup",
        action="store_true",
        help="Keep original comma files as *.comma.bak",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]

    ensure_large_csv_field_limit()

    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    for table in tables:
        path = data_dir / f"{table}.csv"
        if not path.exists():
            raise SystemExit(f"Missing file: {path}")

        if looks_pipe_delimited(path):
            print(f"Skipping already pipe-delimited file: {path}")
            continue

        print(f"Converting {path} ...")
        convert_file(path, backup=args.keep_backup)
        print(f"Done: {path}")

    print("All requested JOB files converted to pipe-delimited format.")


if __name__ == "__main__":
    main()
