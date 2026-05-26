#!/usr/bin/env python3
"""Run runner in release mode and create result plots.

The script forwards your data/algorithm/pattern/index CSV files to the Rust
benchmark runner. It writes everything into results/<name>_<timestamp>/:
raw.csv, summary.csv, optional row_profile.csv, Python summaries, copied inputs,
and plots.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run runner and plot results")
    p.add_argument("--data-csv", required=True, type=Path)
    p.add_argument("--algorithms-csv", required=True, type=Path)
    p.add_argument("--patterns-csv", required=True, type=Path)
    p.add_argument("--indexes-csv", type=Path)
    p.add_argument("--result-root", type=Path, default=Path("results"))
    p.add_argument("--name", default="bench")
    p.add_argument("--cargo", default="cargo")
    p.add_argument("--package", default="runner")
    p.add_argument("--features", default="", help="Cargo features to enable, e.g. avx512")
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--batch-rows", type=int, default=4096)
    p.add_argument("--max-rows")
    p.add_argument("--max-total-bytes", default="1GiB")
    p.add_argument("--max-row-bytes", default="50MiB")
    p.add_argument("--row-overflow-policy", choices=["truncate", "skip", "error"], default="truncate")
    p.add_argument("--invalid-dna", choices=["error", "skip-record", "map-to-a"], default="skip-record")
    p.add_argument("--no-uppercase", action="store_true")
    p.add_argument("--row-profile", action="store_true", help="Enable separate per-row verifier profiling pass")
    p.add_argument("--row-profile-repeats", type=int, default=1)
    p.add_argument("--row-profile-max-rows")
    p.add_argument("--row-profile-sample-bytes", type=int, default=120)
    p.add_argument("--row-profile-top", type=int, default=20, help="Rows per group to keep in fast/slow row ranking CSVs")
    p.add_argument("--no-run", action="store_true", help="Do not invoke cargo; only plot existing raw.csv in result dir")
    p.add_argument("--existing-result-dir", type=Path, help="Use an existing directory containing raw.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.existing_result_dir:
        out_dir = args.existing_result_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = safe_filename(args.name)
        out_dir = args.result_root / f"{safe_name}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=False)

    raw_csv = out_dir / "raw.csv"
    summary_csv = out_dir / "summary.csv"
    row_profile_csv = out_dir / "row_profile.csv"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    copy_inputs(args, out_dir)

    command = [
        args.cargo,
        "run",
        "-p",
        args.package,
        "--release",
    ]
    if args.features:
        command.extend(["--features", args.features])
    command.extend([
        "--",
        "--data-csv",
        str(args.data_csv),
        "--algorithms-csv",
        str(args.algorithms_csv),
        "--patterns-csv",
        str(args.patterns_csv),
        "--output-csv",
        str(raw_csv),
        "--summary-csv",
        str(summary_csv),
        "--warmups",
        str(args.warmups),
        "--iterations",
        str(args.iterations),
        "--batch-rows",
        str(args.batch_rows),
        "--max-total-bytes",
        str(args.max_total_bytes),
        "--max-row-bytes",
        str(args.max_row_bytes),
        "--row-overflow-policy",
        args.row_overflow_policy,
        "--invalid-dna",
        args.invalid_dna,
    ])
    if args.indexes_csv:
        command.extend(["--indexes-csv", str(args.indexes_csv)])
    if args.max_rows:
        command.extend(["--max-rows", str(args.max_rows)])
    if args.no_uppercase:
        command.extend(["--uppercase-sequences", "false"])
    if args.row_profile:
        command.extend([
            "--row-profile-csv",
            str(row_profile_csv),
            "--row-profile-repeats",
            str(args.row_profile_repeats),
            "--row-profile-sample-bytes",
            str(args.row_profile_sample_bytes),
        ])
        if args.row_profile_max_rows:
            command.extend(["--row-profile-max-rows", str(args.row_profile_max_rows)])

    (out_dir / "command.txt").write_text(" ".join(shell_quote(x) for x in command) + "\n")

    if not args.no_run:
        print(f"Running benchmark; output directory: {out_dir}", file=sys.stderr)
        subprocess.run(command, check=True)

    if not raw_csv.exists():
        print(f"raw CSV not found: {raw_csv}", file=sys.stderr)
        return 2

    rows = read_rows(raw_csv)
    if not rows:
        print("raw CSV has no rows", file=sys.stderr)
        return 3

    write_python_summary(rows, out_dir / "python_summary.csv")
    try:
        make_plots(rows, plots_dir)
    except ImportError as exc:
        msg = f"Plotting skipped because matplotlib is missing: {exc}\n"
        (out_dir / "plotting_skipped.txt").write_text(msg)
        print(msg, file=sys.stderr)

    if row_profile_csv.exists():
        profile_rows = read_rows(row_profile_csv)
        if profile_rows:
            write_row_profile_analysis(profile_rows, out_dir, args.row_profile_top)
            try:
                make_row_profile_plots(profile_rows, plots_dir, args.row_profile_top)
            except ImportError as exc:
                msg = f"Row profile plotting skipped because matplotlib is missing: {exc}\n"
                (out_dir / "row_profile_plotting_skipped.txt").write_text(msg)
                print(msg, file=sys.stderr)

    write_info(args, out_dir, rows)
    print(f"Done. Results in: {out_dir}")
    return 0


def shell_quote(s: str) -> str:
    if not s:
        return "''"
    if all(c.isalnum() or c in "-._/:=+" for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def copy_inputs(args: argparse.Namespace, out_dir: Path) -> None:
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    for label, path in [
        ("data", args.data_csv),
        ("algorithms", args.algorithms_csv),
        ("patterns", args.patterns_csv),
        ("indexes", args.indexes_csv),
    ]:
        if path:
            shutil.copy2(path, inputs_dir / f"{label}_{path.name}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    return float(value)


def geomean(values: list[float]) -> float:
    xs = [x for x in values if x > 0 and math.isfinite(x)]
    if not xs:
        return math.nan
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def percentile(values: list[float], q: float) -> float:
    xs = sorted(x for x in values if math.isfinite(x))
    if not xs:
        return math.nan
    idx = min(len(xs) - 1, max(0, math.ceil(len(xs) * q) - 1))
    return xs[idx]


def group_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str, str]:
    return (
        row["dataset"],
        row.get("column", ""),
        row["storage"],
        row["requested_index"],
        row["actual_index"],
        row["pattern_name"],
        row["pattern"],
    )


def write_python_summary(rows: list[dict[str, str]], path: Path) -> None:
    groups: dict[tuple[str, str, str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = group_key(row) + (row["algorithm"],)
        groups[key].append(row)

    out_rows = []
    for key, group in groups.items():
        dataset, column, storage, requested_index, actual_index, pattern_name, pattern, algorithm = key
        totals = [as_float(r, "query_total_ns") for r in group]
        execs = [as_float(r, "execute_ns") for r in group]
        preps = [as_float(r, "candidate_prepare_ns") for r in group]
        per_rows = [as_float(r, "ns_per_table_row") for r in group]
        out_rows.append({
            "dataset": dataset,
            "column": column,
            "storage": storage,
            "requested_index": requested_index,
            "actual_index": actual_index,
            "pattern_name": pattern_name,
            "pattern": pattern,
            "algorithm": algorithm,
            "runs": len(group),
            "sum_query_total_ns": sum(totals),
            "median_query_total_ns": median(totals),
            "mean_query_total_ns": sum(totals) / len(totals),
            "geomean_query_total_ns": geomean(totals),
            "p90_query_total_ns": percentile(totals, 0.90),
            "median_execute_ns": median(execs),
            "median_candidate_prepare_ns": median(preps),
            "median_ns_per_table_row": median(per_rows),
            "geomean_ns_per_table_row": geomean(per_rows),
            "rows_matched": group[0].get("rows_matched", ""),
            "candidate_rows_seen": group[0].get("candidate_rows_seen", ""),
        })

    out_rows.sort(key=lambda r: (r["dataset"], r["column"], r["storage"], r["requested_index"], r["pattern_name"], r["median_query_total_ns"]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


def make_plots(rows: list[dict[str, str]], plots_dir: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    grouped: dict[tuple[str, str, str, str, str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[group_key(row)][row["algorithm"]].append(as_float(row, "query_total_ns") / 1_000_000.0)

    global_alg: dict[str, list[float]] = defaultdict(list)

    for key, alg_values in grouped.items():
        dataset, column, storage, requested_index, actual_index, pattern_name, pattern = key
        alg_medians = [(alg, median(vals)) for alg, vals in alg_values.items()]
        alg_medians.sort(key=lambda x: x[1])
        for alg, value in alg_medians:
            global_alg[f"{storage}/{requested_index}/{alg}"].append(value)

        labels = [x[0] for x in alg_medians]
        values = [x[1] for x in alg_medians]
        fig_w = max(7.0, len(labels) * 0.8)
        fig, ax = plt.subplots(figsize=(fig_w, 4.5))
        ax.bar(labels, values)
        ax.set_ylabel("median query_total_ns (ms)")
        ax.set_title(f"{dataset}.{column} | {storage} | {requested_index}->{actual_index} | {pattern_name}: {pattern}")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        filename = safe_filename("_".join([dataset, column, storage, requested_index, actual_index, pattern_name])) + ".png"
        fig.savefig(plots_dir / filename, dpi=160)
        plt.close(fig)

    global_rank = [(label, geomean(vals)) for label, vals in global_alg.items()]
    global_rank.sort(key=lambda x: x[1])
    if global_rank:
        labels = [x[0] for x in global_rank[:40]]
        values = [x[1] for x in global_rank[:40]]
        fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.45), 5.0))
        ax.bar(labels, values)
        ax.set_ylabel("geomean of group median times (ms)")
        ax.set_title("Global algorithm/index ranking")
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        fig.savefig(plots_dir / "global_ranking.png", dpi=160)
        plt.close(fig)


def profile_group_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row["dataset"],
        row.get("column", ""),
        row["storage"],
        row["algorithm"],
        row["pattern_name"],
        row["pattern"],
    )


def write_row_profile_analysis(rows: list[dict[str, str]], out_dir: Path, top_n: int) -> None:
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[profile_group_key(row)].append(row)

    summary = []
    slow = []
    fast = []
    for key, group in groups.items():
        dataset, column, storage, algorithm, pattern_name, pattern = key
        values = [as_float(r, "verify_ns_per_repeat") for r in group]
        sorted_group = sorted(group, key=lambda r: as_float(r, "verify_ns_per_repeat"))
        slow_rows = list(reversed(sorted_group[-top_n:]))
        fast_rows = sorted_group[:top_n]
        for r in slow_rows:
            slow.append(dict(r, rank_kind="slow"))
        for r in fast_rows:
            fast.append(dict(r, rank_kind="fast"))
        summary.append({
            "dataset": dataset,
            "column": column,
            "storage": storage,
            "algorithm": algorithm,
            "pattern_name": pattern_name,
            "pattern": pattern,
            "rows_profiled": len(group),
            "min_verify_ns": min(values),
            "median_verify_ns": median(values),
            "geomean_verify_ns": geomean(values),
            "p90_verify_ns": percentile(values, 0.90),
            "max_verify_ns": max(values),
            "slowest_row_id": slow_rows[0].get("row_id", "") if slow_rows else "",
            "slowest_row_label": slow_rows[0].get("row_label", "") if slow_rows else "",
            "fastest_row_id": fast_rows[0].get("row_id", "") if fast_rows else "",
            "fastest_row_label": fast_rows[0].get("row_label", "") if fast_rows else "",
        })

    write_dict_rows(out_dir / "row_profile_summary.csv", summary)
    write_dict_rows(out_dir / "row_profile_top_slow.csv", slow)
    write_dict_rows(out_dir / "row_profile_top_fast.csv", fast)


def make_row_profile_plots(rows: list[dict[str, str]], plots_dir: Path, top_n: int) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[profile_group_key(row)].append(row)

    for key, group in groups.items():
        dataset, column, storage, algorithm, pattern_name, pattern = key
        slow = sorted(group, key=lambda r: as_float(r, "verify_ns_per_repeat"), reverse=True)[:top_n]
        if not slow:
            continue
        labels = [f"{r.get('row_id')}:{r.get('row_label','')}"[:40] for r in slow]
        values = [as_float(r, "verify_ns_per_repeat") for r in slow]
        fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.6), 5.0))
        ax.bar(labels, values)
        ax.set_ylabel("verify ns per repeat")
        ax.set_title(f"Slowest rows | {dataset}.{column} | {storage}/{algorithm} | {pattern_name}: {pattern}")
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        filename = safe_filename("row_profile_" + "_".join([dataset, column, storage, algorithm, pattern_name])) + ".png"
        fig.savefig(plots_dir / filename, dpi=160)
        plt.close(fig)


def write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:180]


def write_info(args: argparse.Namespace, out_dir: Path, rows: list[dict[str, str]]) -> None:
    datasets = sorted({r["dataset"] for r in rows})
    columns = sorted({r.get("column", "") for r in rows})
    storages = sorted({r["storage"] for r in rows})
    algorithms = sorted({r["algorithm"] for r in rows})
    indexes = sorted({r["requested_index"] for r in rows})
    patterns = sorted({r["pattern_name"] for r in rows})
    text = [
        f"name: {args.name}",
        f"rows: {len(rows)}",
        f"datasets: {', '.join(datasets)}",
        f"columns: {', '.join(columns)}",
        f"storages: {', '.join(storages)}",
        f"algorithms: {', '.join(algorithms)}",
        f"indexes: {', '.join(indexes)}",
        f"patterns: {', '.join(patterns)}",
        f"warmups: {args.warmups}",
        f"iterations: {args.iterations}",
        f"max_total_bytes: {args.max_total_bytes}",
        f"max_row_bytes: {args.max_row_bytes}",
        f"row_profile: {args.row_profile}",
    ]
    (out_dir / "info.txt").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
