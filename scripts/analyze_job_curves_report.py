#!/usr/bin/env python3

import argparse
import csv
import html
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


LEN_RE = re.compile(r"len=(\d+)")

RANGES = [
    ("<=8", 0, 8),
    ("9-16", 9, 16),
    ("17-32", 17, 32),
    ("33-63", 33, 63),
    ("64-95", 64, 95),
    ("96-127", 96, 127),
    ("128-150", 128, 150),
    (">=64", 64, 10_000),
]

CORE_SET = {
    "naive-scalar",
    "naive-vector",
    "naive-vector-v2",
    "naive-mixed",
    "kmp",
    "bm",
    "two-way",
    "std",
    "lut-short",
}

PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#76b7b2",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
]


def parse_literal_length(pattern: str, pattern_desc: str) -> int:
    m = LEN_RE.search(pattern_desc)
    if m:
        return int(m.group(1))
    return sum(1 for ch in pattern if ch not in ("%", "_"))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def p95(values):
    if not values:
        return 0.0
    arr = sorted(values)
    idx = max(0, math.ceil(0.95 * len(arr)) - 1)
    return arr[idx]


def pearson(xs, ys):
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def tier_for(avg_rank, geomean_slowdown):
    if avg_rank <= 2.5 and geomean_slowdown <= 1.08:
        return "S"
    if avg_rank <= 4.5 and geomean_slowdown <= 1.20:
        return "A"
    if avg_rank <= 7.0 and geomean_slowdown <= 1.60:
        return "B"
    if avg_rank <= 10.0 and geomean_slowdown <= 3.0:
        return "C"
    return "D"


def compute_ratings(algorithms, literal_lengths, medians_by_algo_length, algo_all_durs):
    rank_acc = defaultdict(list)
    slowdown_acc = defaultdict(list)
    wins = Counter()

    for l in literal_lengths:
        candidates = []
        for algo in algorithms:
            med = medians_by_algo_length.get((algo, l))
            if med is not None:
                candidates.append((algo, med))
        if not candidates:
            continue

        candidates.sort(key=lambda x: x[1])
        best = candidates[0][1]
        for rank, (algo, med) in enumerate(candidates, start=1):
            rank_acc[algo].append(rank)
            slowdown_acc[algo].append(med / best if best > 0 else 1.0)
            if rank == 1:
                wins[algo] += 1

    rows = []
    for algo in algorithms:
        ranks = rank_acc.get(algo, [len(algorithms)])
        ratios = slowdown_acc.get(algo, [float("inf")])
        geomean = math.exp(mean([math.log(max(r, 1e-12)) for r in ratios]))
        avg_rank = mean(ranks)
        rows.append(
            {
                "algorithm": algo,
                "tier": tier_for(avg_rank, geomean),
                "overall_median_duration_micros": round(statistics.median(algo_all_durs[algo]), 4),
                "overall_mean_duration_micros": round(mean(algo_all_durs[algo]), 4),
                "overall_p95_duration_micros": round(p95(algo_all_durs[algo]), 4),
                "avg_rank_across_lengths": round(avg_rank, 4),
                "wins_by_exact_length": int(wins.get(algo, 0)),
                "geomean_slowdown_vs_best_by_length": round(geomean, 6),
            }
        )

    rows.sort(key=lambda r: (r["tier"], r["avg_rank_across_lengths"], r["overall_median_duration_micros"]))
    return rows


def in_range(length, lo, hi):
    return lo <= length <= hi


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def map_linear(value, src_lo, src_hi, dst_lo, dst_hi):
    if src_hi == src_lo:
        return (dst_lo + dst_hi) / 2.0
    t = (value - src_lo) / (src_hi - src_lo)
    return dst_lo + t * (dst_hi - dst_lo)


def build_svg_curve(
    out_path: Path,
    title: str,
    subtitle: str,
    medians_by_algo_length,
    overall_median_by_algo,
    algorithms,
    lengths,
):
    width = 1800
    height = 980
    margin_left = 95
    margin_right = 420
    margin_top = 90
    margin_bottom = 95

    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    y_values = []
    for algo in algorithms:
        for length in lengths:
            v = medians_by_algo_length.get((algo, length))
            if v is not None and v > 0:
                y_values.append(v)

    if not y_values:
        raise RuntimeError("no y values for chart")

    y_log_min = math.floor(math.log10(min(y_values)))
    y_log_max = math.ceil(math.log10(max(y_values)))
    if y_log_min == y_log_max:
        y_log_max += 1

    x_min = min(lengths)
    x_max = max(lengths)

    def x_px(length):
        return map_linear(length, x_min, x_max, plot_left, plot_right)

    def y_px(value):
        lv = math.log10(max(value, 1e-9))
        return map_linear(lv, y_log_min, y_log_max, plot_bottom, plot_top)

    x_ticks = [
        tick
        for tick in [5, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 144, 150]
        if x_min <= tick <= x_max
    ]
    if x_min not in x_ticks:
        x_ticks = [x_min] + x_ticks
    if x_max not in x_ticks:
        x_ticks.append(x_max)
    x_ticks = sorted(set(x_ticks))

    y_ticks = [10**p for p in range(y_log_min, y_log_max + 1)]

    parts = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />')

    # Title
    parts.append(
        f'<text x="{plot_left}" y="42" font-family="sans-serif" font-size="30" font-weight="700" fill="#111">{html.escape(title)}</text>'
    )
    parts.append(
        f'<text x="{plot_left}" y="68" font-family="sans-serif" font-size="16" fill="#444">{html.escape(subtitle)}</text>'
    )

    # Grid: Y
    for yv in y_ticks:
        y = y_px(yv)
        parts.append(
            f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{plot_left - 10}" y="{y + 5:.2f}" text-anchor="end" font-family="monospace" font-size="12" fill="#666">{int(yv)}</text>'
        )

    # Grid: X
    for xv in x_ticks:
        x = x_px(xv)
        parts.append(
            f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}" stroke="#f0f0f0" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{plot_bottom + 24}" text-anchor="middle" font-family="monospace" font-size="12" fill="#666">{xv}</text>'
        )

    # Axes
    parts.append(
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#333" stroke-width="1.5" />'
    )
    parts.append(
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#333" stroke-width="1.5" />'
    )
    parts.append(
        f'<text x="{(plot_left + plot_right)/2:.2f}" y="{height - 28}" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#111">Pattern Literal Length</text>'
    )
    parts.append(
        f'<text transform="translate(28,{(plot_top + plot_bottom)/2:.2f}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#111">Median Runtime (microseconds, log scale)</text>'
    )

    # Curves
    algo_colors = {algo: PALETTE[i % len(PALETTE)] for i, algo in enumerate(algorithms)}
    for algo in algorithms:
        pts = []
        for length in lengths:
            yv = medians_by_algo_length.get((algo, length))
            if yv is None or yv <= 0:
                continue
            pts.append((x_px(length), y_px(yv)))

        if len(pts) < 2:
            continue

        points_attr = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        color = algo_colors[algo]
        parts.append(
            f'<polyline points="{points_attr}" fill="none" stroke="{color}" stroke-width="2.2" opacity="0.95" />'
        )
        for x, y in pts:
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="{color}" opacity="0.95" />'
            )

    # Legend
    legend_x = plot_right + 26
    legend_y = plot_top + 8
    parts.append(
        f'<text x="{legend_x}" y="{legend_y - 8}" font-family="sans-serif" font-size="15" font-weight="700" fill="#111">Algorithms (overall median us)</text>'
    )

    sorted_for_legend = sorted(algorithms, key=lambda a: overall_median_by_algo.get(a, float("inf")))
    for idx, algo in enumerate(sorted_for_legend):
        y = legend_y + idx * 24
        color = algo_colors[algo]
        med = overall_median_by_algo.get(algo, 0.0)
        parts.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 20}" y2="{y}" stroke="{color}" stroke-width="3" />'
        )
        parts.append(
            f'<text x="{legend_x + 28}" y="{y + 4}" font-family="monospace" font-size="12.5" fill="#222">{html.escape(algo)} ({med:.1f})</text>'
        )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze JOB LIKE benchmark curves and complexity.")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Benchmark results directory containing raw_results.csv",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    raw_csv = results_dir / "raw_results.csv"
    if not raw_csv.exists():
        raise SystemExit(f"missing file: {raw_csv}")

    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pattern_meta = {}

    with raw_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("skipped", "").lower() == "true":
                continue

            pattern = row["pattern"]
            pattern_desc = row.get("pattern_desc", "")
            literal_len = parse_literal_length(pattern, pattern_desc)
            pct_count = pattern.count("%")
            underscore_count = pattern.count("_")
            mode = "nomatch" if "nomatch" in pattern_desc else "exists"

            entry = {
                "algorithm": row["algorithm"],
                "pattern_index": int(row["pattern_index"]),
                "pattern": pattern,
                "pattern_desc": pattern_desc,
                "table": row["table"],
                "duration_micros": float(row["duration_micros"]),
                "found_count": int(row["found_count"]),
                "literal_len": literal_len,
                "percent_count": pct_count,
                "underscore_count": underscore_count,
                "mode": mode,
            }
            rows.append(entry)

            pidx = entry["pattern_index"]
            if pidx not in pattern_meta:
                pattern_meta[pidx] = {
                    "pattern_index": pidx,
                    "pattern": pattern,
                    "pattern_desc": pattern_desc,
                    "literal_len": literal_len,
                    "percent_count": pct_count,
                    "underscore_count": underscore_count,
                    "mode": mode,
                }

    if not rows:
        raise SystemExit("raw_results.csv has no non-skipped rows")

    algorithms = sorted({r["algorithm"] for r in rows})
    literal_lengths = sorted({r["literal_len"] for r in rows})

    algo_all_durs = defaultdict(list)
    algo_len_durs = defaultdict(list)
    algo_len_mode_durs = defaultdict(list)
    algo_pattern_durs = defaultdict(list)
    algo_pattern_hits = defaultdict(list)

    for r in rows:
        algo = r["algorithm"]
        l = r["literal_len"]
        mode = r["mode"]
        dur = r["duration_micros"]

        algo_all_durs[algo].append(dur)
        algo_len_durs[(algo, l)].append(dur)
        algo_len_mode_durs[(algo, l, mode)].append(dur)
        algo_pattern_durs[(algo, r["pattern_index"])].append(dur)
        algo_pattern_hits[(algo, r["pattern_index"])].append(r["found_count"])

    # Per-algorithm/per-length curve stats
    curve_rows = []
    medians_by_algo_length = {}
    for algo in algorithms:
        for l in literal_lengths:
            durs = algo_len_durs.get((algo, l), [])
            if not durs:
                continue
            med = statistics.median(durs)
            medians_by_algo_length[(algo, l)] = med
            curve_rows.append(
                {
                    "algorithm": algo,
                    "literal_len": l,
                    "samples": len(durs),
                    "median_duration_micros": round(med, 4),
                    "mean_duration_micros": round(mean(durs), 4),
                    "p95_duration_micros": round(p95(durs), 4),
                }
            )

    write_csv(
        analysis_dir / "curve_median_by_length.csv",
        [
            "algorithm",
            "literal_len",
            "samples",
            "median_duration_micros",
            "mean_duration_micros",
            "p95_duration_micros",
        ],
        curve_rows,
    )

    # Per-mode curve stats
    curve_mode_rows = []
    for algo in algorithms:
        for l in literal_lengths:
            for mode in ("exists", "nomatch"):
                durs = algo_len_mode_durs.get((algo, l, mode), [])
                if not durs:
                    continue
                curve_mode_rows.append(
                    {
                        "algorithm": algo,
                        "literal_len": l,
                        "mode": mode,
                        "samples": len(durs),
                        "median_duration_micros": round(statistics.median(durs), 4),
                        "mean_duration_micros": round(mean(durs), 4),
                        "p95_duration_micros": round(p95(durs), 4),
                    }
                )

    write_csv(
        analysis_dir / "curve_median_by_length_mode.csv",
        [
            "algorithm",
            "literal_len",
            "mode",
            "samples",
            "median_duration_micros",
            "mean_duration_micros",
            "p95_duration_micros",
        ],
        curve_mode_rows,
    )

    # Ratings
    overall_median_by_algo = {algo: statistics.median(algo_all_durs[algo]) for algo in algorithms}
    overall_mean_by_algo = {algo: mean(algo_all_durs[algo]) for algo in algorithms}
    overall_p95_by_algo = {algo: p95(algo_all_durs[algo]) for algo in algorithms}
    rating_rows = compute_ratings(algorithms, literal_lengths, medians_by_algo_length, algo_all_durs)
    core_algorithms = [algo for algo in algorithms if algo in CORE_SET]
    core_rating_rows = compute_ratings(core_algorithms, literal_lengths, medians_by_algo_length, algo_all_durs)

    write_csv(
        analysis_dir / "algorithm_ratings.csv",
        [
            "algorithm",
            "tier",
            "overall_median_duration_micros",
            "overall_mean_duration_micros",
            "overall_p95_duration_micros",
            "avg_rank_across_lengths",
            "wins_by_exact_length",
            "geomean_slowdown_vs_best_by_length",
        ],
        rating_rows,
    )

    write_csv(
        analysis_dir / "algorithm_ratings_core.csv",
        [
            "algorithm",
            "tier",
            "overall_median_duration_micros",
            "overall_mean_duration_micros",
            "overall_p95_duration_micros",
            "avg_rank_across_lengths",
            "wins_by_exact_length",
            "geomean_slowdown_vs_best_by_length",
        ],
        core_rating_rows,
    )

    # Range winners
    range_rows = []
    for algo in algorithms:
        for name, lo, hi in RANGES:
            durs = [r["duration_micros"] for r in rows if r["algorithm"] == algo and in_range(r["literal_len"], lo, hi)]
            if not durs:
                continue
            range_rows.append(
                {
                    "algorithm": algo,
                    "len_range": name,
                    "samples": len(durs),
                    "median_duration_micros": round(statistics.median(durs), 4),
                    "mean_duration_micros": round(mean(durs), 4),
                    "p95_duration_micros": round(p95(durs), 4),
                }
            )

    write_csv(
        analysis_dir / "range_stats_by_algorithm.csv",
        [
            "algorithm",
            "len_range",
            "samples",
            "median_duration_micros",
            "mean_duration_micros",
            "p95_duration_micros",
        ],
        range_rows,
    )

    def compute_winners(subset):
        winners = []
        for name, _, _ in RANGES:
            cand = [r for r in subset if r["len_range"] == name]
            if not cand:
                continue
            cand.sort(key=lambda r: r["median_duration_micros"])
            best = cand[0]
            second = cand[1] if len(cand) > 1 else cand[0]
            gap = ((second["median_duration_micros"] / best["median_duration_micros"]) - 1.0) * 100.0 if best["median_duration_micros"] else 0.0
            winners.append(
                {
                    "len_range": name,
                    "winner": best["algorithm"],
                    "winner_median_duration_micros": round(best["median_duration_micros"], 4),
                    "second": second["algorithm"],
                    "second_median_duration_micros": round(second["median_duration_micros"], 4),
                    "gap_to_second_pct": round(gap, 4),
                }
            )
        return winners

    all_winners = compute_winners(range_rows)
    core_winners = compute_winners([r for r in range_rows if r["algorithm"] in CORE_SET])

    write_csv(
        analysis_dir / "winners_by_length_range_all.csv",
        [
            "len_range",
            "winner",
            "winner_median_duration_micros",
            "second",
            "second_median_duration_micros",
            "gap_to_second_pct",
        ],
        all_winners,
    )
    write_csv(
        analysis_dir / "winners_by_length_range_core.csv",
        [
            "len_range",
            "winner",
            "winner_median_duration_micros",
            "second",
            "second_median_duration_micros",
            "gap_to_second_pct",
        ],
        core_winners,
    )

    # Complexity profile
    complexity_rows = sorted(pattern_meta.values(), key=lambda r: r["pattern_index"])
    write_csv(
        analysis_dir / "pattern_complexity_profile.csv",
        [
            "pattern_index",
            "pattern",
            "pattern_desc",
            "literal_len",
            "percent_count",
            "underscore_count",
            "mode",
        ],
        complexity_rows,
    )

    # Pattern-level outliers: compare each pattern to algo+length baseline
    pattern_algo_median = {}
    for key, durs in algo_pattern_durs.items():
        pattern_algo_median[key] = statistics.median(durs)

    baseline_by_algo_len = defaultdict(list)
    for (algo, pidx), med in pattern_algo_median.items():
        l = pattern_meta[pidx]["literal_len"]
        baseline_by_algo_len[(algo, l)].append(med)

    baseline_by_algo_len = {
        key: statistics.median(vals)
        for key, vals in baseline_by_algo_len.items()
        if vals
    }

    outlier_rows = []
    for (algo, pidx), med in pattern_algo_median.items():
        meta = pattern_meta[pidx]
        l = meta["literal_len"]
        base = baseline_by_algo_len.get((algo, l), 0.0)
        if base <= 0:
            continue
        ratio = med / base
        if ratio >= 1.8 or ratio <= 0.55:
            outlier_rows.append(
                {
                    "algorithm": algo,
                    "pattern_index": pidx,
                    "pattern_desc": meta["pattern_desc"],
                    "pattern": meta["pattern"],
                    "mode": meta["mode"],
                    "literal_len": l,
                    "percent_count": meta["percent_count"],
                    "underscore_count": meta["underscore_count"],
                    "median_duration_micros": round(med, 4),
                    "algo_len_baseline_median_micros": round(base, 4),
                    "ratio_to_algo_len_baseline": round(ratio, 6),
                }
            )

    outlier_rows.sort(
        key=lambda r: abs(math.log(max(r["ratio_to_algo_len_baseline"], 1e-12))),
        reverse=True,
    )

    write_csv(
        analysis_dir / "pattern_outliers_by_algo_length.csv",
        [
            "algorithm",
            "pattern_index",
            "pattern_desc",
            "pattern",
            "mode",
            "literal_len",
            "percent_count",
            "underscore_count",
            "median_duration_micros",
            "algo_len_baseline_median_micros",
            "ratio_to_algo_len_baseline",
        ],
        outlier_rows,
    )

    # Build SVG curves
    algo_order = sorted(algorithms, key=lambda a: overall_median_by_algo[a])
    build_svg_curve(
        analysis_dir / "median_time_vs_length_all_algorithms.svg",
        title="JOB Benchmark: Median Runtime vs Pattern Length",
        subtitle="All algorithms. X=literal length, Y=median runtime (log scale).",
        medians_by_algo_length=medians_by_algo_length,
        overall_median_by_algo=overall_median_by_algo,
        algorithms=algo_order,
        lengths=literal_lengths,
    )

    core_order = [a for a in algo_order if a in CORE_SET]
    build_svg_curve(
        analysis_dir / "median_time_vs_length_core_algorithms.svg",
        title="JOB Benchmark: Median Runtime vs Pattern Length (Core Scanners)",
        subtitle="Core set excludes FM/trigram/fftstr*. X=literal length, Y=median runtime (log scale).",
        medians_by_algo_length=medians_by_algo_length,
        overall_median_by_algo=overall_median_by_algo,
        algorithms=core_order,
        lengths=literal_lengths,
    )

    # Complexity insights
    percent_counts = sorted({meta["percent_count"] for meta in pattern_meta.values()})
    underscore_counts = sorted({meta["underscore_count"] for meta in pattern_meta.values()})
    mode_counts = Counter(meta["mode"] for meta in pattern_meta.values())

    # Correlation of log(median runtime) vs length per algorithm
    corr_rows = []
    for algo in algorithms:
        xs = []
        ys = []
        for l in literal_lengths:
            med = medians_by_algo_length.get((algo, l))
            if med is None or med <= 0:
                continue
            xs.append(float(l))
            ys.append(math.log10(med))
        corr_rows.append(
            {
                "algorithm": algo,
                "pearson_corr_log_runtime_vs_length": round(pearson(xs, ys), 6),
                "points": len(xs),
            }
        )
    corr_rows.sort(key=lambda r: r["pearson_corr_log_runtime_vs_length"])
    write_csv(
        analysis_dir / "length_runtime_correlation.csv",
        ["algorithm", "pearson_corr_log_runtime_vs_length", "points"],
        corr_rows,
    )

    # Report
    ratings_sorted = sorted(rating_rows, key=lambda r: (r["tier"], r["avg_rank_across_lengths"]))
    core_ratings_sorted = sorted(core_rating_rows, key=lambda r: (r["tier"], r["avg_rank_across_lengths"]))
    outliers_slow = [r for r in outlier_rows if r["ratio_to_algo_len_baseline"] > 1.0][:15]
    outliers_fast = [r for r in outlier_rows if r["ratio_to_algo_len_baseline"] < 1.0][:15]

    lines = []
    lines.append("# JOB Full Report (FFTSTR Enabled)")
    lines.append("")
    lines.append(f"- Source: `{raw_csv}`")
    lines.append(f"- Non-skipped rows: `{len(rows)}`")
    lines.append(f"- Algorithms: `{len(algorithms)}`")
    lines.append(f"- Unique pattern lengths: `{len(literal_lengths)}` (min `{min(literal_lengths)}`, max `{max(literal_lengths)}`)")
    lines.append("")
    lines.append("## Curves")
    lines.append("- All algorithms curve plot: `analysis/median_time_vs_length_all_algorithms.svg`")
    lines.append("- Core scan-only curve plot: `analysis/median_time_vs_length_core_algorithms.svg`")
    lines.append("- Curve data: `analysis/curve_median_by_length.csv`")
    lines.append("")
    lines.append("## Algorithm Ratings")
    lines.append("")
    lines.append("Tier meaning: `S` elite, `A` strong, `B` solid, `C` situational, `D` poor for this workload.")
    lines.append("")
    lines.append("| Algorithm | Tier | Overall median us | Avg rank by length | Wins by exact length | Geomean slowdown vs best |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in ratings_sorted:
        lines.append(
            f"| `{r['algorithm']}` | `{r['tier']}` | {r['overall_median_duration_micros']:.2f} | {r['avg_rank_across_lengths']:.2f} | {r['wins_by_exact_length']} | {r['geomean_slowdown_vs_best_by_length']:.3f}x |"
        )

    lines.append("")
    lines.append("### Core Scan Ratings")
    lines.append("(excludes `fm`, `trigram`, `fftstr0`, `fftstr1`)\n")
    lines.append("| Algorithm | Tier | Overall median us | Avg rank by length | Wins by exact length | Geomean slowdown vs core best |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in core_ratings_sorted:
        lines.append(
            f"| `{r['algorithm']}` | `{r['tier']}` | {r['overall_median_duration_micros']:.2f} | {r['avg_rank_across_lengths']:.2f} | {r['wins_by_exact_length']} | {r['geomean_slowdown_vs_best_by_length']:.3f}x |"
        )

    lines.append("")
    lines.append("## Winners By Length Range")
    lines.append("### All Algorithms")
    for r in all_winners:
        lines.append(
            f"- `{r['len_range']}`: `{r['winner']}` ({r['winner_median_duration_micros']:.2f} us), next `{r['second']}` ({r['second_median_duration_micros']:.2f} us), gap `{r['gap_to_second_pct']:.2f}%`"
        )
    lines.append("")
    lines.append("### Core Scan Algorithms Only")
    for r in core_winners:
        lines.append(
            f"- `{r['len_range']}`: `{r['winner']}` ({r['winner_median_duration_micros']:.2f} us), next `{r['second']}` ({r['second_median_duration_micros']:.2f} us), gap `{r['gap_to_second_pct']:.2f}%`"
        )

    lines.append("")
    lines.append("## Complexity Findings")
    lines.append(f"- Pattern wildcard `%` counts observed: `{percent_counts}`")
    lines.append(f"- Pattern wildcard `_` counts observed: `{underscore_counts}`")
    lines.append(f"- Pattern mode counts: `exists={mode_counts.get('exists', 0)}`, `nomatch={mode_counts.get('nomatch', 0)}`")
    if len(percent_counts) == 1 and len(underscore_counts) == 1:
        lines.append("- Wildcard-shape complexity is nearly constant in this pattern set; performance differences are driven mostly by literal length and match selectivity.")
    else:
        lines.append("- Wildcard shape varies, so both literal length and wildcard composition contribute to runtime.")
    lines.append("")
    lines.append("### Length vs Runtime Correlation (log runtime)")
    for r in sorted(corr_rows, key=lambda x: x["pearson_corr_log_runtime_vs_length"]):
        lines.append(
            f"- `{r['algorithm']}`: corr `{r['pearson_corr_log_runtime_vs_length']:+.3f}` over `{r['points']}` length points"
        )

    lines.append("")
    lines.append("## Pattern Outliers")
    lines.append("Outlier definition: per-algorithm/per-length pattern median differs from that algo+length baseline by >=1.8x (slow) or <=0.55x (fast).")
    lines.append("")
    lines.append("### Slow Outliers")
    if outliers_slow:
        for r in outliers_slow[:12]:
            pat = r["pattern"]
            snippet = pat if len(pat) <= 66 else pat[:63] + "..."
            lines.append(
                f"- `{r['algorithm']}` len `{r['literal_len']}` ratio `{r['ratio_to_algo_len_baseline']:.2f}x` mode `{r['mode']}` pattern `{snippet}`"
            )
    else:
        lines.append("- No slow outliers found with current threshold.")

    lines.append("")
    lines.append("### Fast Outliers")
    if outliers_fast:
        for r in outliers_fast[:12]:
            pat = r["pattern"]
            snippet = pat if len(pat) <= 66 else pat[:63] + "..."
            lines.append(
                f"- `{r['algorithm']}` len `{r['literal_len']}` ratio `{r['ratio_to_algo_len_baseline']:.2f}x` mode `{r['mode']}` pattern `{snippet}`"
            )
    else:
        lines.append("- No fast outliers found with current threshold.")

    lines.append("")
    lines.append("## Artifacts")
    for fn in [
        "curve_median_by_length.csv",
        "curve_median_by_length_mode.csv",
        "algorithm_ratings.csv",
        "algorithm_ratings_core.csv",
        "range_stats_by_algorithm.csv",
        "winners_by_length_range_all.csv",
        "winners_by_length_range_core.csv",
        "pattern_complexity_profile.csv",
        "pattern_outliers_by_algo_length.csv",
        "length_runtime_correlation.csv",
        "median_time_vs_length_all_algorithms.svg",
        "median_time_vs_length_core_algorithms.svg",
    ]:
        lines.append(f"- `analysis/{fn}`")

    (analysis_dir / "full_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote analysis into {analysis_dir}")


if __name__ == "__main__":
    main()
