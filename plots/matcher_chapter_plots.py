"""
matcher_chapter_plots.py
========================
Figures for the generic LIKE matcher chapter, computed from raw per-iteration
benchmark CSVs.

Usage
-----
    python3 plots/matcher_chapter_plots.py
    python3 plots/matcher_chapter_plots.py --only aggregate --only ablation

Figures (name -> output file -> LaTeX label)
    aggregate         fig_matcher_dna_aggregate_ratios.png     fig:matcher-dna-aggregate-ratios
    quotes_aggregate  fig_matcher_quotes_aggregate_ratios.png  fig:matcher-quotes-aggregate-ratios
    ratios            fig_matcher_dna_pattern_ratios.png       fig:matcher-dna-pattern-ratios
    mechanism         fig_matcher_mechanism_examples.png       fig:matcher-mechanism-examples
    ablation          fig_matcher_fastpath_ablation.png         fig:matcher-fastpath-ablation

How values are derived
    - raw per-iteration rows are aggregated to one median execute time per
      (suite, algorithm, generic matcher, pattern);
    - ratios are adaptive/static for matcher comparisons;
    - fast-path ablation ratios are disabled/enabled for identical rows;
    - LIKE strategy is reconstructed from the pattern and whether the algorithm
      supports underscores, matching the analysis tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ----------------------------------------------------------------------
# Style (matches the thesis reference plots / other chapter scripts)
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 14, "axes.linewidth": 1.2,
})
GRID = "#cfcfcf"
STATIC = "#2b5d8a"
ADAPTIVE = "#e8923a"
RECURSIVE = "#7d4e9e"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "plots" / "outputs" / "05"

DEFAULT_ENABLED = {
    "DNA full": PROJECT_ROOT / "benchmark_results/dna/matcher-multi-percent/gencode_utf8_20260613_091821/raw.csv",
    "DNA recursive-safe": PROJECT_ROOT / "benchmark_results/dna/matcher-multi-percent-recursive/gencode_utf8_recursive_20260613_092002/raw.csv",
    "Quotes": PROJECT_ROOT / "benchmark_results/quotes/matcher-multi-percent/quotes_20260613_092056/raw.csv",
}
DEFAULT_DISABLED = {
    "DNA full": PROJECT_ROOT / "benchmark_results/dna/matcher-multi-percent/gencode_utf8_20260613_092230/raw.csv",
    "DNA recursive-safe": PROJECT_ROOT / "benchmark_results/dna/matcher-multi-percent-recursive/gencode_utf8_recursive_20260613_092411/raw.csv",
    "Quotes": PROJECT_ROOT / "benchmark_results/quotes/matcher-multi-percent/quotes_20260613_092512/raw.csv",
}

COL = dict(
    suite="suite", build="build", dataset="dataset", storage="storage",
    algorithm="algorithm", matcher="generic_matcher", requested_index="requested_index",
    actual_index="actual_index", pattern_name="pattern_name", pattern="pattern",
    iteration="iteration", execute_ns="execute_ns", rows_matched="rows_matched",
    row_count="row_count",
)

FIG_FILES = {
    "aggregate": "fig_matcher_dna_aggregate_ratios.png",
    "quotes_aggregate": "fig_matcher_quotes_aggregate_ratios.png",
    "ratios": "fig_matcher_dna_pattern_ratios.png",
    "mechanism": "fig_matcher_mechanism_examples.png",
    "ablation": "fig_matcher_fastpath_ablation.png",
}

STRATEGY_COLORS = {
    "All": "#8d99ae",
    "Exact": "#6b8e23",
    "Prefix": "#3f7d5a",
    "Suffix": "#c0392b",
    "Contains": "#7d4e9e",
    "PercentOnly": "#e8923a",
    "General": "#2b5d8a",
}
ALGO_MARKERS = {"StdSearch": "o", "NaiveVectorizedV2": "s", "NaiveVectorizedV2Wildcard": "^"}
FAST_STRATEGIES = ["All", "Exact", "Prefix", "Suffix", "Contains"]
GENERIC_STRATEGIES = ["PercentOnly", "General"]


# ----------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------
def is_wildcard_algo(algorithm: str) -> bool:
    return str(algorithm).endswith("Wildcard")


def like_strategy(pattern: str, algorithm: str) -> str:
    pass_underscore = is_wildcard_algo(algorithm)
    tokens: list[tuple[str, str]] = []
    literal = []
    has_any = False
    has_skip = False

    def flush_literal() -> None:
        if literal:
            tokens.append(("lit", "".join(literal)))
            literal.clear()

    for ch in str(pattern):
        if ch == "%":
            flush_literal()
            tokens.append(("any", ch))
            has_any = True
        elif ch == "_" and not pass_underscore:
            flush_literal()
            tokens.append(("skip", ch))
            has_skip = True
        else:
            literal.append(ch)
    flush_literal()

    if not tokens:
        return "Exact"
    literal_count = sum(kind == "lit" for kind, _ in tokens)
    if literal_count == 0:
        return "All"
    if has_skip:
        return "General"
    if literal_count > 1:
        return "PercentOnly" if has_any else "General"

    starts_any = tokens[0][0] == "any"
    ends_any = tokens[-1][0] == "any"
    if starts_any and ends_any:
        return "Contains"
    if starts_any:
        return "Suffix"
    if ends_any:
        return "Prefix"
    return "Exact"


def gmean(values) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    x = x[x > 0]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def short_pattern(pattern: str, max_len: int = 22) -> str:
    p = str(pattern)
    if len(p) <= max_len:
        return p
    keep = max_len // 2 - 1
    return p[:keep] + "..." + p[-keep:]


def read_raw(paths: dict[str, Path], build: str) -> pd.DataFrame:
    frames = []
    for suite, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df[COL["suite"]] = suite
        df[COL["build"]] = build
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out[(out[COL["storage"]] == "utf8") & (out[COL["requested_index"]] == "full-scan")]
    out["strategy"] = [like_strategy(p, a) for p, a in zip(out[COL["pattern"]], out[COL["algorithm"]])]
    out["execute_ms"] = pd.to_numeric(out[COL["execute_ns"]], errors="coerce") / 1e6
    out[COL["rows_matched"]] = pd.to_numeric(out[COL["rows_matched"]], errors="coerce")
    out[COL["row_count"]] = pd.to_numeric(out[COL["row_count"]], errors="coerce")
    return out


def aggregate_raw(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        COL["suite"], COL["build"], COL["dataset"], COL["algorithm"], COL["matcher"],
        COL["requested_index"], COL["actual_index"], COL["pattern_name"], COL["pattern"],
        "strategy",
    ]
    agg = df.groupby(keys, dropna=False).agg(
        median_execute_ms=("execute_ms", "median"),
        min_execute_ms=("execute_ms", "min"),
        max_execute_ms=("execute_ms", "max"),
        rows_matched=(COL["rows_matched"], "median"),
        row_count=(COL["row_count"], "median"),
    ).reset_index()
    return agg


def matcher_pairs(agg: pd.DataFrame, suite="DNA full") -> pd.DataFrame:
    a = agg[(agg[COL["suite"]] == suite) & (agg[COL["build"]] == "enabled")]
    key = [COL["algorithm"], COL["pattern_name"], COL["pattern"], "strategy"]
    rows = []
    for values, g in a.groupby(key, dropna=False):
        by_matcher = {r[COL["matcher"]]: r for _, r in g.iterrows()}
        if "static" not in by_matcher or "adaptive" not in by_matcher:
            continue
        static = float(by_matcher["static"]["median_execute_ms"])
        adaptive = float(by_matcher["adaptive"]["median_execute_ms"])
        if static <= 0 or adaptive <= 0:
            continue
        rows.append({
            "algorithm": values[0], "pattern_name": values[1], "pattern": values[2],
            "strategy": values[3], "static_ms": static, "adaptive_ms": adaptive,
            "ratio": adaptive / static,
            "rows_matched": float(by_matcher["static"]["rows_matched"]),
            "row_count": float(by_matcher["static"]["row_count"]),
        })
    return pd.DataFrame(rows)


def _save(fig, path: Path, tight=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight" if tight else None)
    print(f"saved {path}")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure: DNA aggregate static/adaptive ratios
# ----------------------------------------------------------------------
def plot_aggregate_ratios(agg: pd.DataFrame, path: Path, suite: str):
    pairs = matcher_pairs(agg, suite)
    scopes = [
        ("non-wildcard generic", (~pairs["algorithm"].map(is_wildcard_algo)) & pairs["strategy"].isin(GENERIC_STRATEGIES)),
        ("all generic", pairs["strategy"].isin(GENERIC_STRATEGIES)),
        ("all matcher rows", pd.Series(True, index=pairs.index)),
    ]
    labels, ratios, counts = [], [], []
    for label, mask in scopes:
        p = pairs[mask]
        labels.append(label)
        ratio = gmean(p["ratio"])
        ratios.append(ratio)
        counts.append(len(p))

    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    x = np.arange(len(labels))
    ax.axhspan(0.97, 1.03, color="#dbeafe", alpha=0.75, zorder=0)
    ax.axhline(0.97, color="#6b8fb5", lw=0.9, ls="--", zorder=1)
    ax.axhline(1.03, color="#6b8fb5", lw=0.9, ls="--", zorder=1)
    ax.axhline(1.0, color="#111", lw=1.2, zorder=1)
    colors = [STATIC if r > 1.0 else ADAPTIVE for r in ratios]
    bars = ax.bar(x, ratios, color=colors, edgecolor="#222", linewidth=0.8, width=0.58, zorder=3)
    for bar, ratio, n in zip(bars, ratios, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, ratio + 0.006,
                f"{ratio:.3f}x\nn={n}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace(" ", "\n") for label in labels], fontsize=10)
    ax.set_ylabel("adaptive / static runtime")
    ax.set_ylim(0.94, 1.06)
    ax.set_yticks([0.94, 0.97, 1.00, 1.03, 1.06])
    ax.text(2.55, 1.034, "±3% practical tie band", ha="right", va="bottom", fontsize=9.5, color="#315c83")
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, path)


# ----------------------------------------------------------------------
# Figure: DNA pattern-level ratios
# ----------------------------------------------------------------------
def plot_pattern_ratios(agg: pd.DataFrame, path: Path):
    pairs = matcher_pairs(agg, "DNA full").copy()
    order = ["General", "PercentOnly", "Exact", "Prefix", "Suffix", "Contains", "All"]
    pairs = pairs[pairs["strategy"].isin(order)]
    strategy_pos = {s: i for i, s in enumerate(order)}

    # Stable, deterministic jitter keeps repeated renders identical.
    jitter_map = {"StdSearch": -0.18, "NaiveVectorizedV2": 0.0, "NaiveVectorizedV2Wildcard": 0.18}
    pairs["x"] = [strategy_pos[s] + jitter_map.get(a, 0.0) for s, a in zip(pairs["strategy"], pairs["algorithm"])]
    pairs["winner"] = np.where(pairs["ratio"].between(0.95, 1.05), "tie", np.where(pairs["ratio"] > 1.0, "static", "adaptive"))

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.axhspan(0.95, 1.05, color="#edf2f7", zorder=0)
    ax.axhline(1.0, color="#111", lw=1.0, zorder=1)
    ax.axvline(1.5, color=GRID, lw=1.0, ls="--", zorder=1)
    winner_colors = {"static": STATIC, "adaptive": ADAPTIVE, "tie": "#9aa7b1"}
    for algo, marker in ALGO_MARKERS.items():
        sub = pairs[pairs["algorithm"] == algo]
        colors = [winner_colors[w] for w in sub["winner"]]
        ax.scatter(sub["x"], sub["ratio"], marker=marker, s=44, color=colors,
                   edgecolor="#222", linewidth=0.45, alpha=0.9, zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(0.45, 3.3)
    ax.set_yticks([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    ax.set_yticklabels(["0.5", "0.75", "1", "1.5", "2", "3"])
    ax.set_xlim(-0.65, len(order) - 0.35)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("runtime ratio: adaptive / static", fontweight="semibold", labelpad=11)
    ax.set_xlabel("compiled LIKE strategy")
    ax.text(0.5, 0.48, "generic matcher rows", ha="center", va="bottom", fontsize=9, color="#555")
    ax.text(4.1, 0.48, "fast-path strategy rows", ha="center", va="bottom", fontsize=9, color="#555")
    ax.text(0.03, 1.035, ">1: static faster     <1: adaptive faster",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color="#555")
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    winner_handles = [
        Line2D([0], [0], marker="o", ls="", color=STATIC, markeredgecolor="#222", label="static >5%"),
        Line2D([0], [0], marker="o", ls="", color=ADAPTIVE, markeredgecolor="#222", label="adaptive >5%"),
        Line2D([0], [0], marker="o", ls="", color="#9aa7b1", markeredgecolor="#222", label="within ±5%"),
    ]
    algo_handles = [Line2D([0], [0], marker=m, ls="", color="#777", markeredgecolor="#222", label=a)
                    for a, m in ALGO_MARKERS.items()]
    fig.legend(handles=winner_handles + algo_handles, frameon=False, fontsize=8.5,
               ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               columnspacing=1.5, handletextpad=0.5)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.76, bottom=0.20)
    _save(fig, path, tight=False)


# ----------------------------------------------------------------------
# Figure: mechanism examples
# ----------------------------------------------------------------------
MECHANISM_CASES = [
    ("interleaved_with_bigger", "NaiveVectorizedV2", "selective anchor\nstatic wins"),
    ("many_common_short_segments", "NaiveVectorizedV2", "common bases\nadaptive wins"),
    ("one_fixed_len002", "NaiveVectorizedV2", "one-base gap\nadaptive wins"),
    ("dna_common_gap_multi_percent", "NaiveVectorizedV2", "multi-%\nnear tie"),
]


def plot_mechanism_examples(agg: pd.DataFrame, path: Path):
    pairs = matcher_pairs(agg, "DNA full")
    rows = []
    for pattern_name, algorithm, label in MECHANISM_CASES:
        hit = pairs[(pairs["pattern_name"] == pattern_name) & (pairs["algorithm"] == algorithm)]
        if not len(hit):
            continue
        row = hit.iloc[0]
        rows.append((label, row["static_ms"], row["adaptive_ms"], row["ratio"], short_pattern(row["pattern"], 18)))

    fig, ax = plt.subplots(figsize=(6.9, 4.2))
    x = np.arange(len(rows))
    width = 0.34
    static_vals = [r[1] for r in rows]
    adaptive_vals = [r[2] for r in rows]
    ax.bar(x - width / 2, static_vals, width, color=STATIC, edgecolor="#222", linewidth=0.7, label="static")
    ax.bar(x + width / 2, adaptive_vals, width, color=ADAPTIVE, edgecolor="#222", linewidth=0.7, label="adaptive")
    for i, row in enumerate(rows):
        top = max(row[1], row[2])
        ax.text(i, top * 1.12, f"{row[3]:.2f}x", ha="center", va="bottom", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylabel("median execute (ms, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7, which="both")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, path)


# ----------------------------------------------------------------------
# Figure: fast-path ablation
# ----------------------------------------------------------------------
def ablation_ratios(agg: pd.DataFrame) -> pd.DataFrame:
    keys = [COL["suite"], COL["algorithm"], COL["matcher"], COL["pattern_name"], COL["pattern"], "strategy"]
    rows = []
    for values, g in agg.groupby(keys, dropna=False):
        by_build = {r[COL["build"]]: r for _, r in g.iterrows()}
        if "enabled" not in by_build or "disabled" not in by_build:
            continue
        enabled = float(by_build["enabled"]["median_execute_ms"])
        disabled = float(by_build["disabled"]["median_execute_ms"])
        if enabled <= 0 or disabled <= 0:
            continue
        rows.append({
            "suite": values[0], "algorithm": values[1], "matcher": values[2],
            "pattern_name": values[3], "pattern": values[4], "strategy": values[5],
            "ratio": disabled / enabled, "enabled_ms": enabled, "disabled_ms": disabled,
        })
    return pd.DataFrame(rows)


def plot_fastpath_ablation(agg: pd.DataFrame, path: Path):
    ratios = ablation_ratios(agg)
    suites = ["DNA full", "DNA recursive-safe", "Quotes"]
    columns = FAST_STRATEGIES + ["Generic\ncontrols"]
    values = np.full((len(suites), len(columns)), np.nan)
    for i, suite in enumerate(suites):
        for j, col in enumerate(columns):
            if col == "Generic\ncontrols":
                sub = ratios[(ratios["suite"] == suite) & (ratios["strategy"].isin(GENERIC_STRATEGIES))]
            else:
                sub = ratios[(ratios["suite"] == suite) & (ratios["strategy"] == col)]
            if len(sub):
                values[i, j] = gmean(sub["ratio"])

    fig, ax = plt.subplots(figsize=(6.9, 4.3))
    x = np.arange(len(columns))
    width = 0.25
    colors = {"DNA full": STATIC, "DNA recursive-safe": RECURSIVE, "Quotes": ADAPTIVE}
    offsets = [-width, 0, width]
    for i, suite in enumerate(suites):
        ax.bar(x + offsets[i], values[i], width, color=colors[suite], edgecolor="#222",
               linewidth=0.65, label=suite, zorder=3)
        for j, val in enumerate(values[i]):
            if np.isfinite(val) and (val >= 3 or j == len(columns) - 1):
                ax.text(x[j] + offsets[i], val * 1.08, f"{val:.1f}x", ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="#111", lw=1.0)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 800)
    ax.set_ylabel("disabled / enabled (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(columns, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7, which="both")
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, path)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate matcher chapter plots")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--only", action="append", choices=sorted(FIG_FILES), help="Generate only selected figure(s)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    enabled = read_raw(DEFAULT_ENABLED, "enabled")
    disabled = read_raw(DEFAULT_DISABLED, "disabled")
    agg = aggregate_raw(pd.concat([enabled, disabled], ignore_index=True))
    wanted = set(args.only or FIG_FILES.keys())

    if "aggregate" in wanted:
        plot_aggregate_ratios(agg, args.out_dir / FIG_FILES["aggregate"], "DNA full")
    if "quotes_aggregate" in wanted:
        plot_aggregate_ratios(agg, args.out_dir / FIG_FILES["quotes_aggregate"], "Quotes")
    if "ratios" in wanted:
        plot_pattern_ratios(agg, args.out_dir / FIG_FILES["ratios"])
    if "mechanism" in wanted:
        plot_mechanism_examples(agg, args.out_dir / FIG_FILES["mechanism"])
    if "ablation" in wanted:
        plot_fastpath_ablation(agg, args.out_dir / FIG_FILES["ablation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
