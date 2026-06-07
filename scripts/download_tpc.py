#!/usr/bin/env python3
"""Generate TPC-H and TPC-DS raw CSV exports with DuckDB."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


RAW_DELIMITER = "|"
RAW_EXPORT_OPTIONS = "FORMAT CSV, DELIMITER '|', HEADER false"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TPC-H/TPC-DS raw CSV data")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tpch-sf", type=float, default=1, help="TPC-H scale factor")
    parser.add_argument("--tpcds-sf", type=float, default=1, help="TPC-DS scale factor")
    parser.add_argument("--skip-tpch", action="store_true", help="Skip TPC-H")
    parser.add_argument("--skip-tpcds", action="store_true", help="Skip TPC-DS")
    parser.add_argument("--force", action="store_true", help="Overwrite existing generated data")
    return parser.parse_args()


def clear_dir(path: Path) -> None:
    if not path.exists():
        return
    for entry in path.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def verify_no_export_headers(con, out_dir: Path) -> None:
    for (table,) in con.execute("SHOW TABLES").fetchall():
        columns = [row[1] for row in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
        file_path = out_dir / f"{table}.csv"
        if not file_path.exists():
            raise RuntimeError(f"Missing export file: {file_path}")
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            first_line = handle.readline().rstrip("\n")
        if first_line.split(RAW_DELIMITER) == columns:
            raise RuntimeError(f"Header verification failed for {table}: found headers")


def generate_tpc(name: str, out_dir: Path, scale_factor: float, force: bool) -> None:
    if out_dir.exists() and any(out_dir.glob("*.csv")) and not force:
        print(f"{name} dataset already exists in {out_dir}")
        return

    try:
        import duckdb  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("DuckDB python package required for TPC generation") from exc

    clear_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {name} with DuckDB (sf={scale_factor})...")
    con = duckdb.connect(database=":memory:")
    try:
        con.execute(f"INSTALL {name}")
        con.execute(f"LOAD {name}")
        if name == "tpch":
            con.execute(f"CALL dbgen(sf={scale_factor})")
        else:
            con.execute(f"CALL dsdgen(sf={scale_factor})")
        out_dir_sql = str(out_dir).replace("'", "''")
        con.execute(f"EXPORT DATABASE '{out_dir_sql}' ({RAW_EXPORT_OPTIONS})")
        verify_no_export_headers(con, out_dir)
    finally:
        con.close()
    print(f"{name} export complete: {out_dir}")


def main() -> None:
    args = parse_args()
    if not args.skip_tpch:
        generate_tpc("tpch", args.root / "tpch", args.tpch_sf, args.force)
    if not args.skip_tpcds:
        generate_tpc("tpcds", args.root / "tpcds", args.tpcds_sf, args.force)
    print("Done.")


if __name__ == "__main__":
    main()
