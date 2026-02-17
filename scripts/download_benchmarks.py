import argparse
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


JOB_URLS = [
    "https://event.cwi.nl/da/job/imdb.tgz",
    "http://event.cwi.nl/da/job/imdb.tgz",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path) -> None:
    for entry in path.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def download_file(url: str, destination: Path, use_curl: bool) -> None:
    if use_curl and shutil.which("curl") is not None:
        subprocess.run([
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
        ], check=True)
        return

    with urllib.request.urlopen(url) as response:
        with open(destination, "wb") as out_file:
            shutil.copyfileobj(response, out_file)


def download_job_dataset(out_dir: Path, force: bool) -> None:
    ensure_dir(out_dir)
    archive_path = out_dir / "imdb.tgz"

    extract_marker = out_dir / ".extracted"
    if extract_marker.exists() and not force:
        print("JOB dataset already extracted")
        return

    prefer_curl = False
    for attempt in range(2):
        if archive_path.exists() and not force:
            print(f"JOB archive already exists: {archive_path}")
        else:
            last_error = None
            for url in JOB_URLS:
                order = (True, False) if prefer_curl else (False, True)
                for use_curl in order:
                    downloader = "curl" if use_curl else "urllib"
                    try:
                        print(f"Downloading JOB dataset from {url} ({downloader})...")
                        download_file(url, archive_path, use_curl=use_curl)
                        print(f"Saved JOB archive to {archive_path}")
                        last_error = None
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_error = exc
                        if archive_path.exists():
                            archive_path.unlink()
                        print(f"Failed to download from {url} ({downloader}): {exc}")
                if last_error is None:
                    break

            if last_error is not None:
                raise RuntimeError("All JOB dataset downloads failed")

        print("Extracting JOB dataset...")
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(out_dir)
            extract_marker.write_text("ok")
            print(f"JOB dataset extracted to {out_dir}")
            return
        except (tarfile.ReadError, EOFError) as exc:
            print(f"Extract failed (attempt {attempt + 1}): {exc}")
            if archive_path.exists():
                archive_path.unlink()
            clear_dir(out_dir)
            force = True
            prefer_curl = True

    raise RuntimeError("JOB dataset extraction failed after retries")


def duckdb_available() -> bool:
    return shutil.which("duckdb") is not None


def generate_with_duckdb_cli(command: str) -> None:
    subprocess.run(["duckdb", "-c", command], check=True)


def verify_export_headers(
    con,
    out_dir: Path,
    tables: list[str],
    delimiter: str,
    expect_headers: bool,
) -> None:
    for table in tables:
        columns = [row[1] for row in con.execute(
            f"PRAGMA table_info('{table}')"
        ).fetchall()]
        file_path = out_dir / f"{table}.csv"
        if not file_path.exists():
            raise RuntimeError(f"Missing export file: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            first_line = handle.readline().rstrip("\n")
        first_row = first_line.split(delimiter)
        is_header = first_row == columns
        if is_header != expect_headers:
            expected = "with headers" if expect_headers else "without headers"
            found = "with headers" if is_header else "without headers"
            raise RuntimeError(
                f"Header verification failed for {table}: expected {expected}, found {found}"
            )


def generate_tpch_tpcds(
    name: str,
    out_dir: Path,
    scale_factor: int,
    force: bool,
) -> None:
    ensure_dir(out_dir)

    existing_files = list(out_dir.glob("*.csv"))
    if existing_files and not force:
        print(f"{name} dataset already exists in {out_dir}")
        return

    try:
        import duckdb  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("DuckDB python package required for header verification") from exc

    print(f"Generating {name} with duckdb python (sf={scale_factor})...")
    con = duckdb.connect(database=":memory:")
    con.execute(f"INSTALL {name}")
    con.execute(f"LOAD {name}")
    if name == "tpch":
        con.execute(f"CALL dbgen(sf={scale_factor})")
        tables = ["part", "orders", "lineitem", "customer"]
    else:
        con.execute(f"CALL dsdgen(sf={scale_factor})")
        tables = ["item", "customer", "customer_address", "date_dim", "call_center"]
    out_dir_sql = str(out_dir).replace("'", "''")
    con.execute(
        f"EXPORT DATABASE '{out_dir_sql}' (FORMAT CSV, DELIMITER '|', HEADER false)"
    )
    verify_export_headers(con, out_dir, tables, "|", expect_headers=False)
    con.close()
    print(f"{name} export complete: {out_dir}")


def generate_job_duckdb(out_dir: Path, scale_factor: int, force: bool) -> None:
    ensure_dir(out_dir)

    existing_files = list(out_dir.glob("*.csv"))
    if existing_files and not force:
        print(f"job dataset already exists in {out_dir}")
        return

    try:
        import duckdb  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("DuckDB python package required for JOB") from exc

    print(f"Generating job with duckdb python (sf={scale_factor})...")
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL job")
    con.execute("LOAD job")
    con.execute(f"CALL dbgen(sf={scale_factor})")
    out_dir_sql = str(out_dir).replace("'", "''")
    con.execute(
        f"EXPORT DATABASE '{out_dir_sql}' (FORMAT CSV, DELIMITER '|', HEADER false)"
    )
    tables = ["title", "name", "movie_info", "keyword", "cast_info"]
    verify_export_headers(con, out_dir, tables, "|", expect_headers=False)
    con.close()
    print(f"job export complete: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument("--root", default="data/benchmarks", help="Output root directory")
    parser.add_argument("--tpch-sf", type=int, default=1, help="TPC-H scale factor")
    parser.add_argument("--tpcds-sf", type=int, default=1, help="TPC-DS scale factor")
    parser.add_argument("--skip-tpch", action="store_true", help="Skip TPC-H")
    parser.add_argument("--skip-tpcds", action="store_true", help="Skip TPC-DS")
    parser.add_argument("--skip-job", action="store_true", help="Skip JOB")
    parser.add_argument(
        "--job-source",
        default="duckdb",
        choices=["duckdb", "download"],
        help="JOB source: duckdb or download",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing data")
    args = parser.parse_args()

    root = Path(args.root)
    tpch_dir = root / "tpch"
    tpcds_dir = root / "tpcds"
    job_dir = root / "job"

    if not args.skip_tpch:
        generate_tpch_tpcds("tpch", tpch_dir, args.tpch_sf, args.force)

    if not args.skip_tpcds:
        generate_tpch_tpcds("tpcds", tpcds_dir, args.tpcds_sf, args.force)

    if not args.skip_job:
        if args.job_source == "duckdb":
            generate_job_duckdb(job_dir, 1, args.force)
        else:
            download_job_dataset(job_dir, args.force)

    print("Done.")


if __name__ == "__main__":
    main()
