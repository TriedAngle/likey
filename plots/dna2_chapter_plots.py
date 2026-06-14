"""
dna2_chapter_plots.py
=====================
Figures for the DNA2 packed-storage analysis, computed from benchmark summary
CSVs.

Usage
-----
    python3 plots/dna2_chapter_plots.py

Figures
-------
    fig_dna2_no_n_fullscan_lengths.png   no-N full-scan query time by length
    fig_dna2_n_stress_summary.png        synthetic sparse-N stress summary
    fig_dna2_qgram_ladder.png            qgram speedup ladder + query times
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 14,
    "axes.linewidth": 1.2,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "plots" / "outputs" / "08"
DEFAULT_FULL_SCAN = (
    PROJECT_ROOT
    / "benchmark_results_arm64_nfastpath/dna/exact-vs-underscore-algorithms"
    / "gencode_20260614_041422/summary.csv"
)
DEFAULT_QGRAM = (
    PROJECT_ROOT
    / "benchmark_results_arm64_nfastpath/dna/exact-vs-underscore-indexes"
    / "gencode_20260614_042622/summary.csv"
)
DEFAULT_N_STRESS = (
    PROJECT_ROOT
    / "benchmark_results_arm64_nfastpath/dna/n-handling"
    / "n_sparse_with_twoway2_horspool_20260614_044749/summary.csv"
)

GRID = "#cfcfcf"
COLORS = {
    "Dna2PackedNeon": "#2b5d8a",
    "Dna2": "#2b5d8a",
    "PairHorspool": "#3f7d5a",
    "NaiveVectorizedV2": "#e8923a",
    "TwoWay": "#7d4e9e",
    "Dna2TwoWay": "#c0392b",
}
LABELS = {
    "Dna2PackedNeon": "DNA2 packed NEON",
    "Dna2": "DNA2 packed",
    "PairHorspool": "UTF-8 PairHorspool",
    "NaiveVectorizedV2": "UTF-8 VectorizedV2",
    "TwoWay": "UTF-8 TwoWay",
    "Dna2TwoWay": "DNA2 TwoWay",
}


def geomean(values) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    x = x[x > 0]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def pattern_len(name: str) -> int:
    match = re.search(r"len(\d+)", str(name))
    if not match:
        raise ValueError(f"cannot parse pattern length from {name!r}")
    return int(match.group(1))


def pattern_group(name: str) -> str:
    if str(name).startswith("exact_"):
        return "Exact"
    if str(name).startswith("underscore_"):
        return "Underscore"
    return "Other"


def read_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["query_ns"] = pd.to_numeric(df["geomean_query_total_ns"], errors="coerce")
    return df


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def set_log_ticks(ax, ticks, label_fmt="g") -> None:
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"%{label_fmt}"))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def plot_no_n_fullscan(full_scan: pd.DataFrame, out_dir: Path) -> None:
    algos = ["Dna2PackedNeon", "PairHorspool", "NaiveVectorizedV2", "TwoWay"]
    df = full_scan[
        (full_scan["actual_index"] == "full-scan")
        & (full_scan["pattern_name"].str.startswith(("exact_", "underscore_")))
        & (full_scan["algorithm"].isin(algos))
    ].copy()
    df["length"] = df["pattern_name"].map(pattern_len)
    df["group"] = df["pattern_name"].map(pattern_group)
    df["query_ms"] = df["query_ns"] / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, group in zip(axes, ["Exact", "Underscore"]):
        sub = df[df["group"] == group]
        lengths = sorted(sub["length"].unique())
        ax.axvspan(0.8, 3.5, color="#f3dfd9", alpha=0.55, zorder=0)
        for algo in algos:
            cur = sub[sub["algorithm"] == algo].sort_values("length")
            ax.plot(
                cur["length"],
                cur["query_ms"],
                marker="o",
                linewidth=2.3,
                markersize=5.5,
                color=COLORS[algo],
                label=LABELS[algo],
            )
        ax.set_title(group)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        set_log_ticks(ax, [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100], ".2g")
        ax.set_xticks(lengths)
        ax.set_xticklabels([str(x) for x in lengths], rotation=0)
        ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8, alpha=0.8)
        ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.4, alpha=0.35)
        ax.set_xlabel("pattern length (bp)")
    axes[0].set_ylabel("query time (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("No-N GENCODE full scan: packed DNA2 vs UTF-8 baselines", y=1.03)
    fig.text(0.5, 0.065, "Shaded region: sub-4 bp, current packed-DNA2 bad case; y-axis is log scaled.",
             ha="center", va="center", fontsize=10, color="#555")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, out_dir, "fig_dna2_no_n_fullscan_lengths")


def plot_n_stress(n_stress: pd.DataFrame, out_dir: Path) -> None:
    algos = ["PairHorspool", "Dna2PackedNeon", "Dna2TwoWay", "TwoWay", "NaiveVectorizedV2"]
    df = n_stress[
        (n_stress["actual_index"] == "full-scan") & (n_stress["algorithm"].isin(algos))
    ].copy()
    df["group"] = np.where(df["pattern"].astype(str).str.contains("N"), "N-containing patterns", "No-N patterns")
    agg = (
        df.groupby(["group", "storage", "algorithm"], as_index=False)
        .agg(query_ns=("query_ns", geomean), pattern_count=("pattern_name", "nunique"))
    )
    agg["query_ms"] = agg["query_ns"] / 1e6

    groups = ["No-N patterns", "N-containing patterns"]
    x = np.arange(len(groups))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    for idx, algo in enumerate(algos):
        vals = []
        for group in groups:
            cur = agg[(agg["group"] == group) & (agg["algorithm"] == algo)]
            vals.append(float(cur["query_ms"].iloc[0]) if len(cur) else np.nan)
        offsets = x + (idx - (len(algos) - 1) / 2) * width
        ax.bar(offsets, vals, width=width, label=LABELS[algo], color=COLORS[algo])

    ax.set_yscale("log")
    set_log_ticks(ax, [0.02, 0.03, 0.05, 0.1, 0.2, 0.5], ".2g")
    ax.set_ylabel("query time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(["3 no-N patterns", "10 N-heavy patterns"])
    ax.set_title("Synthetic sparse-N stress test (2,048 x 180bp reads)\n512 rows contain N; y-axis is log scaled")
    ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.4, alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(fig, out_dir, "fig_dna2_n_stress_summary")


def qgram_ladder_data(full_scan: pd.DataFrame, qgram: pd.DataFrame) -> pd.DataFrame:
    full = full_scan[full_scan["actual_index"] == "full-scan"].copy()
    idx = qgram[qgram["actual_index"] == "qgram"].copy()
    patterns = sorted(
        set(idx["pattern_name"]),
        key=lambda name: (0 if str(name).startswith("exact_") else 1, pattern_len(name)),
    )
    rows = []
    for name in patterns:
        u_full = full[(full["pattern_name"] == name) & (full["storage"] == "utf8") & (full["algorithm"] == "PairHorspool")]
        u_q = idx[(idx["pattern_name"] == name) & (idx["storage"] == "utf8") & (idx["algorithm"] == "PairHorspool")]
        d_full = full[(full["pattern_name"] == name) & (full["storage"] == "dna2") & (full["algorithm"] == "Dna2")]
        d_q = idx[(idx["pattern_name"] == name) & (idx["storage"] == "dna2") & (idx["algorithm"] == "Dna2")]
        if not (len(u_full) and len(u_q) and len(d_full) and len(d_q)):
            continue
        uf = float(u_full["query_ns"].iloc[0])
        uq = float(u_q["query_ns"].iloc[0])
        df = float(d_full["query_ns"].iloc[0])
        dq = float(d_q["query_ns"].iloc[0])
        rows.append({
            "pattern": name,
            "rows_matched": int(u_q["rows_matched"].iloc[0]),
            "utf8_pair_horspool_full_ms": uf / 1e6,
            "utf8_pair_horspool_qgram_us": uq / 1e3,
            "utf8_speedup": uf / uq,
            "dna2_full_ms": df / 1e6,
            "dna2_qgram_us": dq / 1e3,
            "dna2_speedup": df / dq,
            "dna2_speedup_over_utf8": (df / dq) / (uf / uq),
        })
    return pd.DataFrame(rows)


def plot_qgram_ladder(full_scan: pd.DataFrame, qgram: pd.DataFrame, out_dir: Path) -> None:
    ladder = qgram_ladder_data(full_scan, qgram)

    labels = [p.replace("exact_len", "E").replace("underscore_len", "U") for p in ladder["pattern"]]
    x = np.arange(len(ladder))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))

    ax = axes[0]
    ax.bar(x - width / 2, ladder["utf8_speedup"], width=width, color=COLORS["PairHorspool"], label="UTF-8 PairHorspool")
    ax.bar(x + width / 2, ladder["dna2_speedup"], width=width, color=COLORS["Dna2"], label="DNA2 packed")
    ax.set_yscale("log")
    ax.set_title("Qgram speedup over full scan")
    set_log_ticks(ax, [300, 1000, 3000, 10000, 30000, 100000, 300000], ".0f")
    ax.set_ylabel("speedup vs full scan (x)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.4, alpha=0.35)

    ax = axes[1]
    ax.bar(x - width / 2, ladder["utf8_pair_horspool_qgram_us"], width=width, color=COLORS["PairHorspool"], label="UTF-8 PairHorspool")
    ax.bar(x + width / 2, ladder["dna2_qgram_us"], width=width, color=COLORS["Dna2"], label="DNA2 packed")
    ax.set_yscale("log")
    ax.set_title("Qgram query time")
    set_log_ticks(ax, [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50], ".2g")
    ax.set_ylabel("query time (us)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.4, alpha=0.35)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Qgram ladder: indexing accelerates both UTF-8 and DNA2 (log-scaled y axes)", y=1.03)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, out_dir, "fig_dna2_qgram_ladder")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-scan", type=Path, default=DEFAULT_FULL_SCAN)
    parser.add_argument("--qgram", type=Path, default=DEFAULT_QGRAM)
    parser.add_argument("--n-stress", type=Path, default=DEFAULT_N_STRESS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    full_scan = read_summary(args.full_scan)
    qgram = read_summary(args.qgram)
    n_stress = read_summary(args.n_stress)

    plot_no_n_fullscan(full_scan, out_dir)
    plot_n_stress(n_stress, out_dir)
    plot_qgram_ladder(full_scan, qgram, out_dir)
    print(f"wrote DNA2 plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
