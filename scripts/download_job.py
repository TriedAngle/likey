#!/usr/bin/env python3
"""Download JOB/IMDB raw data and normalize it to pipe-delimited CSV."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path


JOB_URL = "https://db.in.tum.de/~fent/dbgen/job/imdb.tzst"
RAW_DELIMITER = "|"
NORMALIZED_MARKER = ".normalized_pipe_csv_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download JOB/IMDB raw CSV data")
    parser.add_argument("--root", type=Path, default=Path("data/raw/job"))
    parser.add_argument("--url", default=JOB_URL, help="JOB archive URL")
    parser.add_argument("--force", action="store_true", help="Overwrite existing downloaded data")
    return parser.parse_args()


def clear_dir(path: Path) -> None:
    if not path.exists():
        return
    for entry in path.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def download_file(url: str, destination: Path) -> None:
    if shutil.which("curl") is not None:
        subprocess.run(
            [
                "curl",
                "-L",
                "--retry",
                "5",
                "--retry-all-errors",
                "--continue-at",
                "-",
                "-o",
                str(destination),
                url,
            ],
            check=True,
        )
        return

    with urllib.request.urlopen(url) as response:
        with destination.open("wb") as out_file:
            shutil.copyfileobj(response, out_file)


def extract_job_archive(archive_path: Path, out_dir: Path) -> None:
    if not archive_path.name.lower().endswith((".tzst", ".zst", ".tar.zst")):
        raise RuntimeError(f"Unsupported JOB archive format: {archive_path.name}")

    try:
        import zstandard  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Extracting .tzst requires python package 'zstandard'. "
            "Install with: python3 -m pip install zstandard"
        ) from exc

    tmp_tar = out_dir / "imdb.tar"
    if tmp_tar.exists():
        tmp_tar.unlink()

    dctx = zstandard.ZstdDecompressor()
    with archive_path.open("rb") as src, tmp_tar.open("wb") as dst:
        dctx.copy_stream(src, dst)

    try:
        with tarfile.open(tmp_tar, "r:") as tar:
            tar.extractall(out_dir)
    finally:
        if tmp_tar.exists():
            tmp_tar.unlink()


def find_schema_path(dataset_dir: Path) -> Path | None:
    for base in (dataset_dir, dataset_dir / "imdb"):
        path = base / "schematext.sql"
        if path.exists():
            return path
    return None


def find_table_file(dataset_dir: Path, table: str) -> Path | None:
    for base in (dataset_dir, dataset_dir / "imdb"):
        path = base / f"{table}.csv"
        if path.exists() and path.is_file():
            return path
    return None


def parse_schema_column_counts(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, int] = {}
    for match in re.finditer(r"CREATE\s+TABLE\s+([\w\"]+)\s*\((.*?)\)\s*;", text, re.I | re.S):
        table = clean_identifier(match.group(1))
        out[table] = sum(
            1 for raw_col in split_sql_columns(match.group(2)) if is_data_column(raw_col)
        )
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


def is_data_column(raw: str) -> bool:
    raw = " ".join(raw.strip().split())
    if not raw:
        return False
    first = raw.split(" ", 1)[0]
    return first.upper() not in {"PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT"}


def clean_identifier(value: str) -> str:
    return value.strip().strip('"')


def normalize_backslash_quotes(lines):
    in_quotes = False
    for line in lines:
        out: list[str] = []
        idx = 0
        while idx < len(line):
            ch = line[idx]
            if in_quotes and ch == "\\" and idx + 1 < len(line):
                next_ch = line[idx + 1]
                if next_ch == '"':
                    out.append('""')
                elif next_ch == "\\":
                    out.append("\\")
                else:
                    out.append(next_ch)
                idx += 2
                continue
            if ch == '"':
                if in_quotes and idx + 1 < len(line) and line[idx + 1] == '"':
                    out.append('""')
                    idx += 2
                    continue
                in_quotes = not in_quotes
            out.append(ch)
            idx += 1
        yield "".join(out)


def iter_csv_rows(path: Path, delimiter: str, backslash_quoted: bool):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        lines = normalize_backslash_quotes(handle) if backslash_quoted else handle
        reader = csv.reader(lines, delimiter=delimiter, strict=True)
        yield from reader


def score_dialect(path: Path, delimiter: str, backslash_quoted: bool, expected_columns: int) -> int:
    exact_rows = 0
    try:
        for sampled_rows, row in enumerate(iter_csv_rows(path, delimiter, backslash_quoted), start=1):
            if len(row) == expected_columns:
                exact_rows += 1
            if sampled_rows >= 100:
                break
    except csv.Error:
        return -1
    return exact_rows


def infer_dialect(path: Path, expected_columns: int) -> tuple[str, bool]:
    candidates = ((RAW_DELIMITER, False), (",", True))
    scored = [
        (score_dialect(path, delimiter, backslash_quoted, expected_columns), delimiter, backslash_quoted)
        for delimiter, backslash_quoted in candidates
    ]
    score, delimiter, backslash_quoted = max(scored)
    if score <= 0:
        raise RuntimeError(f"Could not infer CSV dialect for {path}")
    return delimiter, backslash_quoted


def rewrite_table(path: Path, delimiter: str, backslash_quoted: bool, expected_columns: int) -> None:
    tmp_path = path.with_name(f".{path.name}.normalized.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.writer(dst, delimiter=RAW_DELIMITER, lineterminator="\n")
            for row_number, row in enumerate(iter_csv_rows(path, delimiter, backslash_quoted), start=1):
                if len(row) != expected_columns:
                    raise RuntimeError(
                        f"Cannot normalize {path}: row {row_number} has {len(row)} columns, "
                        f"expected {expected_columns}"
                    )
                writer.writerow(row)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def normalize_job_csvs(dataset_dir: Path) -> None:
    schema_path = find_schema_path(dataset_dir)
    if schema_path is None:
        raise RuntimeError(f"Cannot normalize JOB: no schematext.sql in {dataset_dir}")

    checked = 0
    rewritten = 0
    for table, expected_columns in sorted(parse_schema_column_counts(schema_path).items()):
        source_path = find_table_file(dataset_dir, table)
        if source_path is None:
            continue
        checked += 1
        delimiter, backslash_quoted = infer_dialect(source_path, expected_columns)
        if delimiter != RAW_DELIMITER or backslash_quoted:
            rewrite_table(source_path, delimiter, backslash_quoted, expected_columns)
            rewritten += 1

    print(f"Normalized JOB raw CSV to delimiter {RAW_DELIMITER!r}: {rewritten}/{checked} files rewritten")


def download_job_dataset(out_dir: Path, url: str, force: bool) -> None:
    if force:
        clear_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extract_marker = out_dir / ".extracted"
    normalize_marker = out_dir / NORMALIZED_MARKER
    if extract_marker.exists() and normalize_marker.exists():
        print("JOB dataset already extracted and normalized")
        return

    if not extract_marker.exists():
        archive_name = Path(urllib.parse.urlparse(url).path).name or "imdb.tzst"
        archive_path = out_dir / archive_name
        if not archive_path.exists():
            print(f"Downloading JOB dataset from {url}...")
            download_file(url, archive_path)
            print(f"Saved JOB archive to {archive_path}")

        print("Extracting JOB dataset...")
        extract_job_archive(archive_path, out_dir)
        extract_marker.write_text("ok\n")

    normalize_job_csvs(out_dir)
    normalize_marker.write_text("ok\n")
    print(f"JOB dataset ready: {out_dir}")


def main() -> None:
    args = parse_args()
    download_job_dataset(args.root, args.url, args.force)
    print("Done.")


if __name__ == "__main__":
    main()
