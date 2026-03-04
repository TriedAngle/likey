#!/usr/bin/env python3
import argparse
import csv
import math
import subprocess
from collections import defaultdict
from pathlib import Path


BIN_BY_DATASET = {
    "tpch": "bench_tpch",
    "tpcds": "bench_tpcds",
    "job": "bench_job",
    "dna": "bench_dna",
    "protein": "bench_protein",
}


def run_command(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pattern_complexity(pattern: str, w_pct: float, w_us: float, w_len: float) -> float:
    return (w_pct * pattern.count("%")) + (w_us * pattern.count("_")) + (w_len * len(pattern))


def bucket_by_threshold(value: float, low: float, high: float) -> str:
    if value <= low:
        return "low"
    if value <= high:
        return "medium"
    return "high"


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, min(len(sorted_vals) - 1, math.ceil(0.95 * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def tercile_thresholds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    i1 = max(0, min(n - 1, math.floor((n - 1) * (1.0 / 3.0))))
    i2 = max(0, min(n - 1, math.floor((n - 1) * (2.0 / 3.0))))
    low = sorted_vals[i1]
    high = sorted_vals[i2]
    if high < low:
        high = low
    return low, high


def to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plotly_charts(
    results_dir: Path,
    enriched: list[dict[str, object]],
    algo_rows: list[dict[str, object]],
    win_rows: list[dict[str, object]],
    bucket_rows: list[dict[str, object]],
) -> list[Path]:
    try:
        import plotly.graph_objects as go  # type: ignore
        from plotly.subplots import make_subplots  # type: ignore
    except Exception:
        return []

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    algorithms = [str(row["algorithm"]) for row in algo_rows]
    mean_vals = [to_float(row["mean_duration_micros"]) for row in algo_rows]
    p95_vals = [to_float(row["p95_duration_micros"]) for row in algo_rows]

    latency_fig = go.Figure()
    latency_fig.add_bar(name="Mean", x=algorithms, y=mean_vals)
    latency_fig.add_bar(name="P95", x=algorithms, y=p95_vals)
    latency_fig.update_layout(
        barmode="group",
        title="Algorithm Latency (Mean vs P95, lower is better)",
        xaxis_title="Algorithm",
        yaxis_title="Duration (microseconds)",
    )
    latency_path = plots_dir / "algorithm_latency.html"
    latency_fig.write_html(str(latency_path), include_plotlyjs="cdn")
    generated.append(latency_path)

    win_algos = [str(row["algorithm"]) for row in win_rows]
    win_rates = [to_float(row["win_rate"]) for row in win_rows]
    win_fig = go.Figure(go.Bar(x=win_algos, y=win_rates))
    win_fig.update_layout(
        title="Algorithm Win Rate (higher is better)",
        xaxis_title="Algorithm",
        yaxis_title="Win rate",
        yaxis=dict(range=[0, 1]),
    )
    wins_path = plots_dir / "algorithm_win_rate.html"
    win_fig.write_html(str(wins_path), include_plotlyjs="cdn")
    generated.append(wins_path)

    by_algo_points: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for row in enriched:
        algo = str(row["algorithm"])
        xs, ys = by_algo_points[algo]
        xs.append(to_float(row["pattern_complexity_norm"]))
        ys.append(to_float(row["duration_micros"]))

    scatter_fig = go.Figure()
    for algo in sorted(by_algo_points):
        xs, ys = by_algo_points[algo]
        scatter_fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=algo,
                opacity=0.65,
            )
        )
    scatter_fig.update_layout(
        title="Pattern Complexity vs Runtime (lower y is better)",
        xaxis_title="Normalized pattern complexity",
        yaxis_title="Duration (microseconds)",
    )
    scatter_path = plots_dir / "complexity_vs_runtime.html"
    scatter_fig.write_html(str(scatter_path), include_plotlyjs="cdn")
    generated.append(scatter_path)

    bucket_order = ["low", "medium", "high"]
    weighted: dict[tuple[str, str, str], tuple[float, int]] = {}
    for row in bucket_rows:
        key = (
            str(row["pattern_len_bucket"]),
            str(row["complexity_bucket"]),
            str(row["algorithm"]),
        )
        mean_val = to_float(row["mean_duration_micros"])
        count = int(to_float(row["count"]))
        prev_sum, prev_count = weighted.get(key, (0.0, 0))
        weighted[key] = (prev_sum + mean_val * count, prev_count + count)

    z: list[list[float]] = []
    text: list[list[str]] = []
    custom_count: list[list[int]] = []
    for complexity_bucket in bucket_order:
        z_row: list[float] = []
        t_row: list[str] = []
        c_row: list[int] = []
        for pattern_bucket in bucket_order:
            best_algo = "-"
            best_mean = float("inf")
            best_count = 0
            for algo in algorithms:
                key = (pattern_bucket, complexity_bucket, algo)
                if key not in weighted:
                    continue
                sum_val, cnt = weighted[key]
                if cnt == 0:
                    continue
                agg_mean = sum_val / cnt
                if agg_mean < best_mean:
                    best_mean = agg_mean
                    best_algo = algo
                    best_count = cnt
            if best_mean == float("inf"):
                z_row.append(math.nan)
                t_row.append("-")
                c_row.append(0)
            else:
                z_row.append(best_mean)
                t_row.append(f"{best_algo}<br>n={best_count}")
                c_row.append(best_count)
        z.append(z_row)
        text.append(t_row)
        custom_count.append(c_row)

    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=bucket_order,
            y=bucket_order,
            text=text,
            texttemplate="%{text}",
            customdata=custom_count,
            colorbar=dict(title="Mean us (lower better)"),
            hovertemplate="Pattern len: %{x}<br>Complexity: %{y}<br>Best algo: %{text}<br>Mean: %{z:.2f}<br>Samples: %{customdata}<extra></extra>",
        )
    )
    heatmap_fig.update_layout(
        title="Best Algorithm By Pattern Length/Complexity Bucket (lower mean is better)",
        xaxis_title="Pattern length bucket",
        yaxis_title="Complexity bucket",
    )
    heatmap_path = plots_dir / "best_algo_heatmap.html"
    heatmap_fig.write_html(str(heatmap_path), include_plotlyjs="cdn")
    generated.append(heatmap_path)

    patterns = sorted({str(row["pattern"]) for row in enriched})
    tables = sorted({str(row["table"]) for row in enriched})

    pattern_algo_means: dict[tuple[str, str], float] = {}
    pattern_algo_raw: dict[tuple[str, str], list[float]] = defaultdict(list)
    pattern_table_algo: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    table_algo_raw: dict[tuple[str, str], list[float]] = defaultdict(list)
    table_pattern_algo: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for row in enriched:
        pattern = str(row["pattern"])
        table = str(row["table"])
        algo = str(row["algorithm"])
        dur = to_float(row["duration_micros"])
        pattern_algo_raw[(pattern, algo)].append(dur)
        pattern_table_algo[(pattern, table, algo)].append(dur)
        table_algo_raw[(table, algo)].append(dur)
        table_pattern_algo[(table, pattern, algo)].append(dur)

    for (pattern, algo), vals in pattern_algo_raw.items():
        pattern_algo_means[(pattern, algo)] = mean(vals)

    pattern_viewer = go.Figure()
    for idx, pattern in enumerate(patterns):
        z = []
        for algo in algorithms:
            row_vals = []
            for table in tables:
                vals = pattern_table_algo.get((pattern, table, algo), [])
                row_vals.append(mean(vals) if vals else math.nan)
            z.append(row_vals)
        pattern_viewer.add_trace(
            go.Heatmap(
                z=z,
                x=tables,
                y=algorithms,
                colorbar=dict(title="Mean us (lower better)") if idx == 0 else None,
                visible=(idx == 0),
                hovertemplate="Pattern: "
                + pattern
                + "<br>Table: %{x}<br>Algorithm: %{y}<br>Mean: %{z:.2f}<extra></extra>",
            )
        )

    pattern_buttons = []
    for idx, pattern in enumerate(patterns):
        visible = [False] * len(patterns)
        visible[idx] = True
        pattern_buttons.append(
            dict(
                label=pattern,
                method="update",
                args=[
                    {"visible": visible},
                    {
                        "title": f"Pattern Detail Viewer: {pattern} (lower is better)",
                    },
                ],
            )
        )

    pattern_viewer.update_layout(
        title=(
            f"Pattern Detail Viewer: {patterns[0]} (lower is better)"
            if patterns
            else "Pattern Detail Viewer"
        ),
        xaxis_title="Table",
        yaxis_title="Algorithm",
        updatemenus=[
            dict(
                buttons=pattern_buttons,
                direction="down",
                showactive=True,
                x=0.0,
                y=1.15,
                xanchor="left",
                yanchor="top",
            )
        ],
    )
    pattern_viewer_path = plots_dir / "pattern_detail_viewer.html"
    pattern_viewer.write_html(str(pattern_viewer_path), include_plotlyjs="cdn")
    generated.append(pattern_viewer_path)

    table_viewer = go.Figure()
    for idx, table in enumerate(tables):
        z = []
        for algo in algorithms:
            row_vals = []
            for pattern in patterns:
                vals = table_pattern_algo.get((table, pattern, algo), [])
                row_vals.append(mean(vals) if vals else math.nan)
            z.append(row_vals)
        table_viewer.add_trace(
            go.Heatmap(
                z=z,
                x=patterns,
                y=algorithms,
                colorbar=dict(title="Mean us (lower better)") if idx == 0 else None,
                visible=(idx == 0),
                hovertemplate="Table: "
                + table
                + "<br>Pattern: %{x}<br>Algorithm: %{y}<br>Mean: %{z:.2f}<extra></extra>",
            )
        )

    table_buttons = []
    for idx, table in enumerate(tables):
        visible = [False] * len(tables)
        visible[idx] = True
        table_buttons.append(
            dict(
                label=table,
                method="update",
                args=[
                    {"visible": visible},
                    {
                        "title": f"Table Detail Viewer: {table} (lower is better)",
                    },
                ],
            )
        )

    table_viewer.update_layout(
        title=(
            f"Table Detail Viewer: {tables[0]} (lower is better)"
            if tables
            else "Table Detail Viewer"
        ),
        xaxis_title="Pattern",
        yaxis_title="Algorithm",
        updatemenus=[
            dict(
                buttons=table_buttons,
                direction="down",
                showactive=True,
                x=0.0,
                y=1.15,
                xanchor="left",
                yanchor="top",
            )
        ],
    )
    table_viewer_path = plots_dir / "table_detail_viewer.html"
    table_viewer.write_html(str(table_viewer_path), include_plotlyjs="cdn")
    generated.append(table_viewer_path)

    dashboard_fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Algorithm Latency (Mean vs P95)",
            "Algorithm Win Rate",
            "Pattern Complexity vs Runtime",
            "Best Algorithm By Pattern Length/Complexity Bucket",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "scatter"}, {"type": "heatmap"}]],
    )

    dashboard_fig.add_trace(go.Bar(name="Mean", x=algorithms, y=mean_vals), row=1, col=1)
    dashboard_fig.add_trace(go.Bar(name="P95", x=algorithms, y=p95_vals), row=1, col=1)

    dashboard_fig.add_trace(go.Bar(name="Win rate", x=win_algos, y=win_rates), row=1, col=2)

    for algo in sorted(by_algo_points):
        xs, ys = by_algo_points[algo]
        dashboard_fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=f"{algo} (scatter)",
                opacity=0.6,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    dashboard_fig.add_trace(
        go.Heatmap(
            z=z,
            x=bucket_order,
            y=bucket_order,
            text=text,
            texttemplate="%{text}",
            customdata=custom_count,
            colorbar=dict(title="Mean us"),
            hovertemplate="Pattern len: %{x}<br>Complexity: %{y}<br>Best algo: %{text}<br>Mean: %{z:.2f}<br>Samples: %{customdata}<extra></extra>",
            showscale=True,
        ),
        row=2,
        col=2,
    )

    dashboard_fig.update_layout(
        title="LIKE Benchmark Dashboard",
        height=980,
        width=1400,
        barmode="group",
        legend_title_text="Series",
    )
    dashboard_fig.update_xaxes(title_text="Algorithm", row=1, col=1)
    dashboard_fig.update_yaxes(title_text="Duration (microseconds, lower is better)", row=1, col=1)
    dashboard_fig.update_xaxes(title_text="Algorithm", row=1, col=2)
    dashboard_fig.update_yaxes(title_text="Win rate (higher is better)", range=[0, 1], row=1, col=2)
    dashboard_fig.update_xaxes(title_text="Normalized pattern complexity", row=2, col=1)
    dashboard_fig.update_yaxes(title_text="Duration (microseconds, lower is better)", row=2, col=1)
    dashboard_fig.update_xaxes(title_text="Pattern length bucket", row=2, col=2)
    dashboard_fig.update_yaxes(title_text="Complexity bucket", row=2, col=2)

    dashboard_path = plots_dir / "dashboard.html"
    dashboard_fig.write_html(str(dashboard_path), include_plotlyjs="cdn")
    generated.append(dashboard_path)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LIKE benchmarks using a pattern TSV file")
    parser.add_argument(
        "--patterns-file",
        default="scripts/patterns_default_sample.tsv",
        help="Pattern TSV file (pattern TAB description)",
    )
    parser.add_argument(
        "--dna-patterns-file",
        default="scripts/patterns_dna.tsv",
        help="DNA pattern TSV file",
    )
    parser.add_argument(
        "--protein-patterns-file",
        default="scripts/patterns_protein.tsv",
        help="Protein pattern TSV file",
    )
    parser.add_argument("--data-root", default="data/benchmarks")
    parser.add_argument("--datasets", default="tpch,tpcds", help="Comma-separated datasets")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--arena-gb", type=int)
    parser.add_argument("--max-bytes", type=int)
    parser.add_argument(
        "--max-rows-per-table",
        type=int,
        help="Optional row cap per table for all datasets",
    )
    parser.add_argument(
        "--job-max-rows-per-table",
        type=int,
        help="Optional row cap per JOB table (applies only to bench_job)",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-fftstr0", action="store_true")
    parser.add_argument("--skip-fftstr1", action="store_true")
    parser.add_argument("--skip-fm", action="store_true")
    parser.add_argument("--skip-trigram", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Do not generate Plotly charts")
    parser.add_argument("--weight-pct", type=float, default=3.0)
    parser.add_argument("--weight-us", type=float, default=2.0)
    parser.add_argument("--weight-len", type=float, default=1.0)
    parser.add_argument("--cargo-profile", choices=["release", "debug"], default="release")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    results_dir = Path(args.results_dir)
    patterns_file = Path(args.patterns_file)
    dna_patterns_file = Path(args.dna_patterns_file)
    protein_patterns_file = Path(args.protein_patterns_file)

    if not patterns_file.exists():
        raise SystemExit(
            f"Pattern file does not exist: {patterns_file}. "
            "Generate one with scripts/generate_like_patterns.py or pass --patterns-file."
        )
    if not dna_patterns_file.exists():
        raise SystemExit(f"DNA pattern file does not exist: {dna_patterns_file}")
    if not protein_patterns_file.exists():
        raise SystemExit(f"Protein pattern file does not exist: {protein_patterns_file}")

    if not args.skip_build:
        build_cmd = ["cargo", "build", "-p", "tests"]
        if args.cargo_profile == "release":
            build_cmd.append("--release")
        run_command(build_cmd)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    all_rows: list[dict[str, str]] = []
    for dataset in datasets:
        bin_name = BIN_BY_DATASET.get(dataset)
        if not bin_name:
            print(f"Skipping unknown dataset '{dataset}'")
            continue

        ds_dir = data_root / dataset
        if not ds_dir.exists():
            print(f"Skipping missing dataset directory: {ds_dir}")
            continue

        for run_idx in range(args.repeat):
            per_run_csv = results_dir / "raw" / f"{dataset}_run{run_idx + 1}.csv"
            per_run_csv.parent.mkdir(parents=True, exist_ok=True)

            dataset_patterns_file = patterns_file
            if dataset == "dna":
                dataset_patterns_file = dna_patterns_file
            elif dataset == "protein":
                dataset_patterns_file = protein_patterns_file

            cmd = ["cargo", "run", "-p", "tests", "--bin", bin_name]
            if args.cargo_profile == "release":
                cmd.append("--release")
            cmd.extend(
                [
                    "--",
                    "--data-dir",
                    str(ds_dir),
                    "--patterns-file",
                    str(dataset_patterns_file),
                    "--output-csv",
                    str(per_run_csv),
                ]
            )
            if args.arena_gb is not None:
                cmd.extend(["--arena-gb", str(args.arena_gb)])
            if args.max_bytes is not None:
                cmd.extend(["--max-bytes", str(args.max_bytes)])
            per_dataset_row_cap = args.max_rows_per_table
            if dataset == "job" and args.job_max_rows_per_table is not None:
                per_dataset_row_cap = args.job_max_rows_per_table
            if per_dataset_row_cap is not None:
                cmd.extend(["--max-rows-per-table", str(per_dataset_row_cap)])
            if args.skip_fftstr0:
                cmd.append("--skip-fftstr0")
            if args.skip_fftstr1:
                cmd.append("--skip-fftstr1")
            if args.skip_fm:
                cmd.append("--skip-fm")
            if args.skip_trigram:
                cmd.append("--skip-trigram")

            run_command(cmd)

            rows = load_rows(per_run_csv)
            for row in rows:
                row["run"] = str(run_idx + 1)
            all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No benchmark rows collected")

    filtered = [r for r in all_rows if r.get("skipped", "false").lower() != "true"]
    if not filtered:
        raise SystemExit("All benchmark rows are skipped")

    raw_scores = [
        pattern_complexity(r["pattern"], args.weight_pct, args.weight_us, args.weight_len)
        for r in filtered
    ]
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    score_span = max(1e-12, max_score - min_score)

    data_lens = [float(r.get("table_avg_len", "0") or 0.0) for r in filtered]
    len_low, len_high = tercile_thresholds(data_lens)

    pattern_lengths = [float(len(r["pattern"])) for r in filtered]
    pattern_len_low, pattern_len_high = tercile_thresholds(pattern_lengths)

    complexity_low, complexity_high = tercile_thresholds(raw_scores)

    enriched: list[dict[str, object]] = []
    for row in filtered:
        duration = float(row["duration_micros"])
        pattern = row["pattern"]
        raw_complexity = pattern_complexity(pattern, args.weight_pct, args.weight_us, args.weight_len)
        norm_complexity = (raw_complexity - min_score) / score_span
        pattern_len = len(pattern)
        pattern_len_bucket = bucket_by_threshold(float(pattern_len), pattern_len_low, pattern_len_high)
        data_len = float(row.get("table_avg_len", "0") or 0.0)
        data_len_bucket = bucket_by_threshold(data_len, len_low, len_high)
        complexity_bucket = bucket_by_threshold(raw_complexity, complexity_low, complexity_high)

        enriched.append(
            {
                **row,
                "duration_micros": duration,
                "pattern_length": pattern_len,
                "pattern_complexity_raw": raw_complexity,
                "pattern_complexity_norm": norm_complexity,
                "pattern_len_bucket": pattern_len_bucket,
                "data_len_bucket": data_len_bucket,
                "complexity_bucket": complexity_bucket,
            }
        )

    raw_out = results_dir / "raw_results.csv"
    raw_fieldnames = list(enriched[0].keys())
    write_csv(raw_out, enriched, raw_fieldnames)

    by_algo: dict[str, list[float]] = defaultdict(list)
    for row in enriched:
        by_algo[str(row["algorithm"])].append(to_float(row["duration_micros"]))

    algo_rows: list[dict[str, object]] = []
    for algo, durations in sorted(by_algo.items()):
        algo_rows.append(
            {
                "algorithm": algo,
                "count": len(durations),
                "mean_duration_micros": round(mean(durations), 4),
                "p95_duration_micros": round(p95(durations), 4),
                "total_duration_micros": round(sum(durations), 4),
            }
        )

    write_csv(
        results_dir / "summary_by_algorithm.csv",
        algo_rows,
        ["algorithm", "count", "mean_duration_micros", "p95_duration_micros", "total_duration_micros"],
    )

    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in enriched:
        key = (
            str(row["algorithm"]),
            str(row["pattern_len_bucket"]),
            str(row["complexity_bucket"]),
            str(row["data_len_bucket"]),
        )
        grouped[key].append(to_float(row["duration_micros"]))

    bucket_rows: list[dict[str, object]] = []
    for (algo, p_bucket, c_bucket, d_bucket), durations in sorted(grouped.items()):
        bucket_rows.append(
            {
                "algorithm": algo,
                "pattern_len_bucket": p_bucket,
                "complexity_bucket": c_bucket,
                "data_len_bucket": d_bucket,
                "count": len(durations),
                "mean_duration_micros": round(mean(durations), 4),
            }
        )

    write_csv(
        results_dir / "summary_by_length_complexity.csv",
        bucket_rows,
        [
            "algorithm",
            "pattern_len_bucket",
            "complexity_bucket",
            "data_len_bucket",
            "count",
            "mean_duration_micros",
        ],
    )

    scenario_winners: dict[tuple[str, str, str, str], tuple[str, float]] = {}
    scenario_rows: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in enriched:
        scenario_key = (
            str(row["dataset"]),
            str(row["table"]),
            str(row["pattern_index"]),
            str(row["run"]),
        )
        scenario_rows[scenario_key].append(row)

    for key, rows in scenario_rows.items():
        winner = min(rows, key=lambda r: to_float(r["duration_micros"]))
        scenario_winners[key] = (str(winner["algorithm"]), to_float(winner["duration_micros"]))

    wins: dict[str, int] = defaultdict(int)
    for winner_algo, _ in scenario_winners.values():
        wins[winner_algo] += 1

    win_rows = [
        {"algorithm": algo, "wins": count, "win_rate": round(count / len(scenario_winners), 4)}
        for algo, count in sorted(wins.items(), key=lambda item: (-item[1], item[0]))
    ]

    write_csv(results_dir / "summary_wins.csv", win_rows, ["algorithm", "wins", "win_rate"])

    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_paths = write_plotly_charts(results_dir, enriched, algo_rows, win_rows, bucket_rows)
        if plot_paths:
            print(f"Wrote Plotly charts to {results_dir / 'plots'}")
        else:
            print("Skipping plots: plotly is not installed")

    report_path = results_dir / "report.md"
    best_overall = min(algo_rows, key=lambda r: to_float(r["mean_duration_micros"]))
    bucket_coverage: dict[tuple[str, str], int] = defaultdict(int)
    for row in enriched:
        bucket_coverage[(str(row["pattern_len_bucket"]), str(row["complexity_bucket"]))] += 1

    with open(report_path, "w", encoding="utf-8") as report:
        report.write("# LIKE Benchmark Report\n\n")
        report.write(f"- Patterns file: `{patterns_file}`\n")
        report.write(f"- Datasets: `{','.join(datasets)}`\n")
        report.write(f"- Runs per dataset: `{args.repeat}`\n")
        report.write(
            f"- Complexity weights: `%={args.weight_pct}`, `_={args.weight_us}`, `len={args.weight_len}`\n\n"
        )
        report.write("## Metric direction\n\n")
        report.write("- Runtime metrics (`mean_duration_micros`, `p95_duration_micros`): lower is better\n")
        report.write("- Win rate: higher is better\n\n")
        report.write("## Bucket thresholds\n\n")
        report.write(
            f"- Pattern length terciles: low<=`{pattern_len_low:.3f}`, medium<=`{pattern_len_high:.3f}`, high>`{pattern_len_high:.3f}`\n"
        )
        report.write(
            f"- Complexity terciles: low<=`{complexity_low:.3f}`, medium<=`{complexity_high:.3f}`, high>`{complexity_high:.3f}`\n"
        )
        report.write(
            f"- Data-length terciles: low<=`{len_low:.3f}`, medium<=`{len_high:.3f}`, high>`{len_high:.3f}`\n\n"
        )
        report.write("## Overall fastest mean\n\n")
        report.write(
            f"`{best_overall['algorithm']}` with mean `{best_overall['mean_duration_micros']}` microseconds.\n\n"
        )
        report.write("## Artifacts\n\n")
        report.write("- `raw_results.csv`\n")
        report.write("- `summary_by_algorithm.csv`\n")
        report.write("- `summary_by_length_complexity.csv`\n")
        report.write("- `summary_wins.csv`\n")
        report.write("\n## Bucket coverage\n\n")
        for complexity_bucket in ["low", "medium", "high"]:
            for pattern_bucket in ["low", "medium", "high"]:
                count = bucket_coverage[(pattern_bucket, complexity_bucket)]
                report.write(
                    f"- pattern_len={pattern_bucket}, complexity={complexity_bucket}: `{count}` samples\n"
                )
        if plot_paths:
            report.write("\n## Plotly charts\n\n")
            for path in plot_paths:
                report.write(f"- `{path.relative_to(results_dir)}`\n")

    print(f"Wrote results to {results_dir}")


if __name__ == "__main__":
    main()
