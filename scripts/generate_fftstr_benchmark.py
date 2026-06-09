#!/usr/bin/env python3
"""Generate the artificial FFTSTR benchmark data and patterns.

The generated table has 1000 rows by default. Every row is the alternating
string `ababab...` with length 512. The generated LIKE patterns are designed to
have zero matches while forcing lowered-wildcard engines to inspect many
candidate positions.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--max-pattern-len", type=int, default=512)
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=Path("data/fftstr/abab_512_x1000.csv"),
    )
    parser.add_argument(
        "--patterns-csv",
        type=Path,
        default=Path("benchmarks/fftstr/abab-wildcards/patterns.csv"),
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("benchmarks/fftstr/abab-wildcards/data.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rows <= 0:
        raise SystemExit("--rows must be positive")
    if args.length <= 0:
        raise SystemExit("--length must be positive")
    if args.step <= 0:
        raise SystemExit("--step must be positive")
    if args.max_pattern_len < 4:
        raise SystemExit("--max-pattern-len must be at least 4")

    write_data(args.data_csv, args.rows, args.length)
    write_patterns(args.patterns_csv, args.step, args.max_pattern_len)
    write_manifest(args.manifest_csv, args.data_csv)

    print(f"wrote {args.rows} rows to {args.data_csv}")
    print(f"wrote patterns to {args.patterns_csv}")
    print(f"wrote runner data manifest to {args.manifest_csv}")
    return 0


def write_data(path: Path, rows: int, length: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = ("ab" * ((length + 1) // 2))[:length]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for row in range(rows):
            writer.writerow([row, value])


def write_patterns(path: Path, step: int, max_len: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "pattern", "enabled"])
        writer.writerow(["exact_len002_no_wildcard", "%aa%", "true"])
        writer.writerow(["wild_len003", "%a_b%", "true"])
        for length in range(step, max_len + 1, step):
            if length < 4:
                continue
            end = "a" if length % 2 == 0 else "b"
            core = "a" + ("_" * (length - 2)) + end
            writer.writerow([f"wild_len{length:03d}", f"%{core}%", "true"])


def write_manifest(path: Path, data_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rel_data = Path(os.path.relpath(data_csv, path.parent))
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "path",
                "type",
                "storage",
                "column",
                "key_column",
                "value_column",
                "enabled",
            ]
        )
        writer.writerow(
            [
                "fftstr_abab_512_x1000",
                rel_data.as_posix(),
                "job-csv",
                "utf8",
                "value",
                "key",
                "value",
                "true",
            ]
        )


if __name__ == "__main__":
    raise SystemExit(main())
