#!/usr/bin/env python3
"""Run every benchmark documented in benchmarks.md with checkpointing.

Each benchmark delegates to `scripts/run_bench.py` and writes to a structured
timestamped directory under `benchmark_results/` by default. The checkpoint CSV
has one row per benchmark and marks `done=true` only after that benchmark exits
successfully.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


JOB_CASES = [
    "cast_info_note",
    "keyword_keyword",
    "movie_companies_note",
    "movie_info_info",
    "name_name",
    "title_title",
]


@dataclass(frozen=True)
class Benchmark:
    benchmark_id: str
    suite: str
    mode: str
    case: str
    result_subdir: str
    data_csv: Path
    algorithms_csv: Path
    patterns_csv: Path
    indexes_csv: Path
    generic_matcher: str = "static"


CHECKPOINT_FIELDS = [
    "benchmark_id",
    "suite",
    "mode",
    "case",
    "data_csv",
    "patterns_csv",
    "algorithms_csv",
    "indexes_csv",
    "generic_matcher",
    "result_dir",
    "done",
    "started_at",
    "finished_at",
    "exit_code",
    "command",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all benchmarks with checkpointing")
    parser.add_argument("--result-root", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--checkpoint-csv", type=Path, help="Defaults to <result-root>/checkpoint.csv")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-total-bytes", default="100MB")
    parser.add_argument("--max-row-bytes", default="50MB")
    parser.add_argument("--batch-rows", type=int, default=4096)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--features", default="")
    parser.add_argument("--generic-matcher", help="Override each benchmark's configured generic matcher")
    parser.add_argument("--only-suite", action="append", choices=["dna", "quotes", "job", "fftstr"])
    parser.add_argument(
        "--only-mode",
        action="append",
        help="Run only matching mode names, e.g. index-comparison or algorithm-comparison",
    )
    parser.add_argument("--only-case", action="append", help="Run only matching case names")
    parser.add_argument("--only-benchmark", action="append", help="Run only exact benchmark_id values")
    parser.add_argument("--skip-fftstr-generate", action="store_true", help="Do not run the FFTSTR data generator")
    parser.add_argument("--force", action="store_true", help="Rerun rows already marked done")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--list", action="store_true", help="List selected benchmarks and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    result_root = args.result_root
    checkpoint_csv = args.checkpoint_csv or result_root / "checkpoint.csv"
    result_root.mkdir(parents=True, exist_ok=True)

    all_defs = all_benchmarks()
    benchmarks = selected_benchmarks(all_defs, args)
    if args.list:
        for bench in benchmarks:
            print(bench.benchmark_id)
        return 0
    if not benchmarks:
        print("No benchmarks selected", file=sys.stderr)
        return 1

    if needs_fftstr_setup(benchmarks) and not args.skip_fftstr_generate:
        setup_command = [sys.executable, "scripts/generate_fftstr_benchmark.py"]
        print(f"[SETUP] FFTSTR data: {shlex.join(setup_command)}")
        if not args.dry_run:
            subprocess.run(setup_command, cwd=repo_root, check=True)

    expected = [checkpoint_row(bench, args) for bench in all_defs]
    checkpoint = merge_checkpoint(read_checkpoint(checkpoint_csv), expected)
    write_checkpoint(checkpoint_csv, checkpoint)
    selected_ids = {bench.benchmark_id for bench in benchmarks}

    for row in checkpoint:
        if row["benchmark_id"] not in selected_ids:
            continue
        if is_done(row) and not args.force:
            print(f"[SKIP] {row['benchmark_id']} already done")
            continue

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = result_root / row["result_subdir"] / f"{row['case']}_{timestamp}"
        command = build_command(args, repo_root, row, result_dir)
        row["result_dir"] = str(result_dir)
        row["started_at"] = now_iso()
        row["finished_at"] = ""
        row["exit_code"] = ""
        row["done"] = "false"
        row["command"] = shlex.join(command)
        write_checkpoint(checkpoint_csv, checkpoint)

        print(f"[START] {row['benchmark_id']}")
        print(f"[COMMAND] {row['command']}")
        if args.dry_run:
            print(f"[DRY-RUN] {row['benchmark_id']}")
            continue

        completed = subprocess.run(command, cwd=repo_root)
        row["finished_at"] = now_iso()
        row["exit_code"] = str(completed.returncode)
        if completed.returncode == 0:
            row["done"] = "true"
            print(f"[DONE] {row['benchmark_id']} -> {result_dir}")
        else:
            row["done"] = "false"
            print(f"[FAILED] {row['benchmark_id']} exit_code={completed.returncode}", file=sys.stderr)
        write_checkpoint(checkpoint_csv, checkpoint)

        if completed.returncode != 0 and not args.continue_on_error:
            return completed.returncode

    print(f"Checkpoint: {checkpoint_csv}")
    return 0


def all_benchmarks() -> list[Benchmark]:
    benchmarks = [
        Benchmark(
            benchmark_id="dna/exact-vs-underscore/gencode",
            suite="dna",
            mode="exact-vs-underscore",
            case="gencode",
            result_subdir="dna/exact-vs-underscore",
            data_csv=Path("benchmarks/dna/data_gencode_dna_utf8_dna2.csv"),
            algorithms_csv=Path("benchmarks/dna/exact-vs-underscore/algorithms.csv"),
            patterns_csv=Path("benchmarks/dna/exact-vs-underscore/patterns.csv"),
            indexes_csv=Path("benchmarks/dna/exact-vs-underscore/indexes.csv"),
        ),
        Benchmark(
            benchmark_id="dna/matcher-comparison/gencode",
            suite="dna",
            mode="matcher-comparison",
            case="gencode",
            result_subdir="dna/matcher-comparison",
            data_csv=Path("benchmarks/dna/data_gencode_dna_utf8_dna2.csv"),
            algorithms_csv=Path("benchmarks/dna/matcher-comparison/algorithms.csv"),
            patterns_csv=Path("benchmarks/dna/matcher-comparison/patterns.csv"),
            indexes_csv=Path("benchmarks/dna/matcher-comparison/indexes.csv"),
            generic_matcher="static,adaptive,recursive",
        ),
        Benchmark(
            benchmark_id="dna/index-comparison/gencode",
            suite="dna",
            mode="index-comparison",
            case="gencode",
            result_subdir="dna/index-comparison",
            data_csv=Path("benchmarks/dna/data_gencode_dna_utf8_dna2.csv"),
            algorithms_csv=Path("benchmarks/dna/index-comparison/algorithms.csv"),
            patterns_csv=Path("benchmarks/dna/index-comparison/patterns.csv"),
            indexes_csv=Path("benchmarks/dna/index-comparison/indexes.csv"),
        ),
        Benchmark(
            benchmark_id="dna/fsst-index-memmem/gencode",
            suite="dna",
            mode="fsst-index-memmem",
            case="gencode",
            result_subdir="dna/fsst-index-memmem",
            data_csv=Path("benchmarks/dna/data_gencode_dna_utf8_fsst.csv"),
            algorithms_csv=Path("benchmarks/dna/fsst-index-memmem/algorithms.csv"),
            patterns_csv=Path("benchmarks/dna/index-comparison/patterns.csv"),
            indexes_csv=Path("benchmarks/dna/fsst-index-memmem/indexes.csv"),
        ),
        Benchmark(
            benchmark_id="quotes/exact-vs-underscore/quotes",
            suite="quotes",
            mode="exact-vs-underscore",
            case="quotes",
            result_subdir="quotes/exact-vs-underscore",
            data_csv=Path("benchmarks/quotes/data_quotes_utf8.csv"),
            algorithms_csv=Path("benchmarks/quotes/exact-vs-underscore/algorithms.csv"),
            patterns_csv=Path("benchmarks/quotes/exact-vs-underscore/patterns.csv"),
            indexes_csv=Path("benchmarks/quotes/exact-vs-underscore/indexes.csv"),
        ),
        Benchmark(
            benchmark_id="quotes/algorithm-cases/quotes",
            suite="quotes",
            mode="algorithm-cases",
            case="quotes",
            result_subdir="quotes/algorithm-cases",
            data_csv=Path("benchmarks/quotes/data_quotes_utf8.csv"),
            algorithms_csv=Path("benchmarks/quotes/algorithm-cases/algorithms.csv"),
            patterns_csv=Path("benchmarks/quotes/algorithm-cases/patterns.csv"),
            indexes_csv=Path("benchmarks/quotes/algorithm-cases/indexes.csv"),
        ),
        Benchmark(
            benchmark_id="quotes/matcher-comparison/quotes",
            suite="quotes",
            mode="matcher-comparison",
            case="quotes",
            result_subdir="quotes/matcher-comparison",
            data_csv=Path("benchmarks/quotes/data_quotes_utf8.csv"),
            algorithms_csv=Path("benchmarks/quotes/matcher-comparison/algorithms.csv"),
            patterns_csv=Path("benchmarks/quotes/matcher-comparison/patterns.csv"),
            indexes_csv=Path("benchmarks/quotes/matcher-comparison/indexes.csv"),
            generic_matcher="static,adaptive,recursive",
        ),
        Benchmark(
            benchmark_id="quotes/index-comparison/quotes",
            suite="quotes",
            mode="index-comparison",
            case="quotes",
            result_subdir="quotes/index-comparison",
            data_csv=Path("benchmarks/quotes/data_quotes_utf8.csv"),
            algorithms_csv=Path("benchmarks/quotes/index-comparison/algorithms.csv"),
            patterns_csv=Path("benchmarks/quotes/index-comparison/patterns.csv"),
            indexes_csv=Path("benchmarks/quotes/index-comparison/indexes.csv"),
        ),
        Benchmark(
            benchmark_id="quotes/fsst-index-memmem/quotes",
            suite="quotes",
            mode="fsst-index-memmem",
            case="quotes",
            result_subdir="quotes/fsst-index-memmem",
            data_csv=Path("benchmarks/quotes/data_quotes_utf8_fsst.csv"),
            algorithms_csv=Path("benchmarks/quotes/fsst-index-memmem/algorithms.csv"),
            patterns_csv=Path("benchmarks/quotes/index-comparison/patterns.csv"),
            indexes_csv=Path("benchmarks/quotes/fsst-index-memmem/indexes.csv"),
        ),
        Benchmark(
            benchmark_id="fftstr/abab-wildcards/abab",
            suite="fftstr",
            mode="abab-wildcards",
            case="abab",
            result_subdir="fftstr/abab-wildcards",
            data_csv=Path("benchmarks/fftstr/abab-wildcards/data.csv"),
            algorithms_csv=Path("benchmarks/fftstr/abab-wildcards/algorithms.csv"),
            patterns_csv=Path("benchmarks/fftstr/abab-wildcards/patterns.csv"),
            indexes_csv=Path("benchmarks/fftstr/abab-wildcards/indexes.csv"),
        ),
    ]
    benchmarks.extend(job_benchmarks())
    return benchmarks


def job_benchmarks() -> list[Benchmark]:
    out = []
    for case in JOB_CASES:
        data_csv = Path(f"benchmarks/job/data_{case}.csv")
        patterns_csv = Path(f"benchmarks/job/index-comparison/patterns_{case}.csv")
        out.extend([
            Benchmark(
                benchmark_id=f"job/index-consistency/{case}",
                suite="job",
                mode="index-consistency",
                case=case,
                result_subdir="job/index-consistency",
                data_csv=data_csv,
                algorithms_csv=Path("benchmarks/job/index-comparison/algorithms.csv"),
                patterns_csv=patterns_csv,
                indexes_csv=Path("benchmarks/job/index-comparison/indexes.csv"),
            ),
            Benchmark(
                benchmark_id=f"job/algorithm-comparison/{case}",
                suite="job",
                mode="algorithm-comparison",
                case=case,
                result_subdir="job/algorithm-comparison",
                data_csv=data_csv,
                algorithms_csv=Path("benchmarks/job/algorithm-comparison/algorithms.csv"),
                patterns_csv=patterns_csv,
                indexes_csv=Path("benchmarks/job/algorithm-comparison/indexes.csv"),
            ),
            Benchmark(
                benchmark_id=f"job/matcher-comparison/{case}",
                suite="job",
                mode="matcher-comparison",
                case=case,
                result_subdir="job/matcher-comparison",
                data_csv=data_csv,
                algorithms_csv=Path("benchmarks/job/matcher-comparison/algorithms.csv"),
                patterns_csv=patterns_csv,
                indexes_csv=Path("benchmarks/job/matcher-comparison/indexes.csv"),
                generic_matcher="static,adaptive,recursive",
            ),
            Benchmark(
                benchmark_id=f"job/index-memmem/{case}",
                suite="job",
                mode="index-memmem",
                case=case,
                result_subdir="job/index-memmem",
                data_csv=data_csv,
                algorithms_csv=Path("benchmarks/job/index-memmem/algorithms.csv"),
                patterns_csv=patterns_csv,
                indexes_csv=Path("benchmarks/job/index-memmem/indexes.csv"),
            ),
            Benchmark(
                benchmark_id=f"job/fsst-index-memmem/{case}",
                suite="job",
                mode="fsst-index-memmem",
                case=case,
                result_subdir="job/fsst-index-memmem",
                data_csv=Path(f"benchmarks/job/data_{case}_utf8_fsst.csv"),
                algorithms_csv=Path("benchmarks/job/fsst-index-memmem/algorithms.csv"),
                patterns_csv=patterns_csv,
                indexes_csv=Path("benchmarks/job/fsst-index-memmem/indexes.csv"),
            ),
        ])
    return out


def selected_benchmarks(benchmarks: list[Benchmark], args: argparse.Namespace) -> list[Benchmark]:
    out = []
    for bench in benchmarks:
        if args.only_suite and bench.suite not in args.only_suite:
            continue
        if args.only_mode and bench.mode not in args.only_mode:
            continue
        if args.only_case and bench.case not in args.only_case:
            continue
        if args.only_benchmark and bench.benchmark_id not in args.only_benchmark:
            continue
        out.append(bench)
    return out


def checkpoint_row(bench: Benchmark, args: argparse.Namespace) -> dict[str, str]:
    return {
        "benchmark_id": bench.benchmark_id,
        "suite": bench.suite,
        "mode": bench.mode,
        "case": bench.case,
        "data_csv": str(bench.data_csv),
        "patterns_csv": str(bench.patterns_csv),
        "algorithms_csv": str(bench.algorithms_csv),
        "indexes_csv": str(bench.indexes_csv),
        "generic_matcher": args.generic_matcher or bench.generic_matcher,
        "result_dir": "",
        "done": "false",
        "started_at": "",
        "finished_at": "",
        "exit_code": "",
        "command": "",
        "result_subdir": bench.result_subdir,
    }


def build_command(args: argparse.Namespace, repo_root: Path, row: dict[str, str], result_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "scripts/run_bench.py"),
        "--name",
        row["case"],
        "--existing-result-dir",
        str(result_dir),
        "--data-csv",
        row["data_csv"],
        "--algorithms-csv",
        row["algorithms_csv"],
        "--generic-matcher",
        row["generic_matcher"],
        "--patterns-csv",
        row["patterns_csv"],
        "--indexes-csv",
        row["indexes_csv"],
        "--iterations",
        str(args.iterations),
        "--warmups",
        str(args.warmups),
        "--batch-rows",
        str(args.batch_rows),
        "--max-total-bytes",
        args.max_total_bytes,
        "--max-row-bytes",
        args.max_row_bytes,
        "--cargo",
        args.cargo,
    ]
    if args.features:
        command.extend(["--features", args.features])
    return command


def needs_fftstr_setup(benchmarks: list[Benchmark]) -> bool:
    return any(bench.suite == "fftstr" for bench in benchmarks)


def read_checkpoint(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [normalize_row(row) for row in csv.DictReader(handle)]


def merge_checkpoint(existing: list[dict[str, str]], expected: list[dict[str, str]]) -> list[dict[str, str]]:
    by_id = {row["benchmark_id"]: normalize_row(row) for row in existing}
    merged = []
    for row in expected:
        if row["benchmark_id"] in by_id:
            current = by_id[row["benchmark_id"]]
            for key in [
                "suite",
                "mode",
                "case",
                "data_csv",
                "patterns_csv",
                "algorithms_csv",
                "indexes_csv",
                "generic_matcher",
                "result_subdir",
            ]:
                current[key] = row[key]
            merged.append(current)
        else:
            merged.append(normalize_row(row))
    return merged


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    fields = CHECKPOINT_FIELDS + ["result_subdir"]
    return {field: row.get(field, "") for field in fields}


def write_checkpoint(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = CHECKPOINT_FIELDS + ["result_subdir"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def is_done(row: dict[str, str]) -> bool:
    return row.get("done", "").strip().lower() in {"1", "true", "yes", "y", "done"}


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
