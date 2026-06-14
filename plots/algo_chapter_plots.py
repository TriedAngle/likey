"""
algo_chapter_plots.py
=====================
All figures for the scan-based literal-search chapter (05_algorithms.tex),
computed from scratch from the RAW per-iteration benchmark CSV(s).

Usage
-----
    python3 plots/algo_chapter_plots.py
    python3 plots/algo_chapter_plots.py --csv dna_raw.csv quotes_raw.csv job_raw.csv
    python3 plots/algo_chapter_plots.py --csv ... --only matrix
    python3 plots/algo_chapter_plots.py --csv ... --only tailrisk --split

Figures (name -> output file -> LaTeX label)
    length    fig_algo_length_*.png      fig:algo-length
    tailrisk  fig_algo_tailrisk_*.png    fig:algo-tailrisk
    winner    fig_algo_winner_map.png    fig:algo-winner-map
    matrix    fig_algo_matrix.png        fig:algo-matrix
    sel       fig_algo_selectivity.png   fig:algo-selectivity
    base      fig_algo_baselines.png     fig:algo-baselines
    ladder    fig_algo_ladder.png        fig:algo-ladder

How values are derived (nothing is hard-coded)
    - per-iteration rows are aggregated to one MEDIAN execute time per
      (dataset, column, algorithm, pattern);
    - selectivity = rows_matched / row_count;
    - length / kind come from the pattern core (text between the %), so
      '%CGCA%' -> length 4 exact, '%C__G%' -> length 4 underscore;
    - class = wildcard if the algorithm name ends in 'Wildcard';
    - tail-risk ratios are recomputed against the best algorithm per pattern;
    - the baseline figure uses real per-iteration spread for error bars.

Rows with storage != utf8, generic_matcher != static, a non-full-scan requested
or actual index, or a Dna2/FftStr/NaiveAvx512 algorithm are dropped, matching the
chapter's isolation. NaiveAvx512 is excluded because the current benchmark build
does not enable the avx512 Cargo feature, so those rows are fallback paths.

=== PLACEHOLDERS (set for the thesis; safe auto-fallbacks let it run as-is) ===
    MATRIX_PATTERNS  curated pattern_name list for the heatmap (None -> all).
    BASELINE_PATTERN representative pattern for the baseline bars (None -> largest table).
    LADDER_CASES     representative patterns for the optimization ladder.
    LADDER_STEPS     ordered algorithm names = the optimization steps.
    LABELS           raw dataset name -> short panel title.
"""

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------
# Style + palette (matches the thesis reference plots)
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 14, "axes.linewidth": 1.2,
})
NW, WC, GRID = "#2b5d8a", "#e8923a", "#cfcfcf"

FAMILIES = ["scalar naive", "vectorized naive", "wildcard naive", "prefiltered", "classical", "library"]
FAM_COLORS = ["#8d99ae", "#2b5d8a", "#e8923a", "#c0392b", "#3f7d5a", "#7d4e9e"]
FAM_CMAP = ListedColormap(FAM_COLORS)
FAM_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], FAM_CMAP.N)

# ----------------------------------------------------------------------
# Raw per-iteration column names (verified against the gencode utf8 raw.csv)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "plots" / "outputs" / "06"
DEFAULT_RAW_CSVS = [
    PROJECT_ROOT / "benchmark_results/dna/exact-vs-underscore-algorithms/gencode_20260609_111318/raw.csv",
    PROJECT_ROOT / "benchmark_results/quotes/exact-vs-underscore/quotes_20260609_062306/raw.csv",
    PROJECT_ROOT / "benchmark_results/job/algorithm-comparison/cast_info_note_20260609_063817/raw.csv",
    PROJECT_ROOT / "benchmark_results/job/algorithm-comparison/keyword_keyword_20260609_070739/raw.csv",
    PROJECT_ROOT / "benchmark_results/job/algorithm-comparison/movie_companies_note_20260609_070944/raw.csv",
    PROJECT_ROOT / "benchmark_results/job/algorithm-comparison/movie_info_info_20260609_073015/raw.csv",
    PROJECT_ROOT / "benchmark_results/job/algorithm-comparison/name_name_20260609_075623/raw.csv",
    PROJECT_ROOT / "benchmark_results/job/algorithm-comparison/title_title_20260609_081241/raw.csv",
]

COL = dict(
    dataset="dataset", column="column", storage="storage", algorithm="algorithm",
    matcher="generic_matcher", requested_index="requested_index", actual_index="actual_index",
    pattern_name="pattern_name", pattern="pattern", iteration="iteration",
    execute_ns="execute_ns", compile_ns="compile_ns",
    rows_matched="rows_matched", row_count="row_count",
)
DROP_PREFIXES = ("Dna2", "FftStr", "Fftstr", "NaiveAvx512")
SINGLE_SEGMENT = re.compile(r"^%[^%]*%$")

# ----------------------------------------------------------------------
# Placeholders
# ----------------------------------------------------------------------
LABELS = {
    "GENCODE human transcripts": "DNA",
    "quotes": "Quotes",
    "job_cast_info_note": "JOB cast_info.note",
    "job_keyword_keyword": "JOB keyword.keyword",
    "job_movie_companies_note": "JOB movie_companies.note",
    "job_movie_info_info": "JOB movie_info.info",
    "job_name_name": "JOB name.name",
    "job_title_title": "JOB title.title",
}
MATRIX_PATTERNS = [
    "exact_len004", "underscore_len004", "exact_len016", "underscore_len016",
    "exact_len064", "underscore_len064", "contains_producer", "contains_sequel",
    "contains_co_production", "contains_b", "prefix_usa_200", "contains_champion",
    "contains_kung_fu_panda",
]
BASELINE_PATTERN = "contains_producer"
BASELINES = ["LibcMemmem", "StdSearch", "BM", "Utf8Kmp", "TwoWay"]
DEFAULT = "PairHorspool"
LADDER_GROUPS = {
    "dna": [
        ("GENCODE human transcripts", "sequence", "exact_len004", "short exact"),
        ("GENCODE human transcripts", "sequence", "underscore_len004", "short wildcard"),
        ("GENCODE human transcripts", "sequence", "exact_len064", "long exact"),
        ("GENCODE human transcripts", "sequence", "underscore_len064", "long wildcard"),
    ],
    "quotes": [
        ("quotes", "quotes.quote", "exact_len004", "short exact"),
        ("quotes", "quotes.quote", "underscore_len004", "short wildcard"),
        ("quotes", "quotes.quote", "exact_len064", "long exact"),
        ("quotes", "quotes.quote", "underscore_len064", "long wildcard"),
    ],
    "job": [
        ("job_name_name", "name.name", "contains_b", "short broad"),
        ("job_cast_info_note", "cast_info.note", "contains_producer", "contains"),
        ("job_name_name", "name.name", "contains_downey_robert", "multi-%"),
        ("job_movie_info_info", "movie_info.info", "prefix_usa_200", "anchored"),
    ],
}
LADDER_STEPS = [
    "Naive", "NaiveVectorizedV2", "NaiveAvx2V2",
    "NaiveVectorizedV2Wildcard", "NaiveAutoWildcard", "PairHorspool",
]
LADDER_LABELS = {
    "Naive": "Naive",
    "NaiveVectorizedV2": "VectorizedV2",
    "NaiveAvx2V2": "Avx2V2",
    "NaiveVectorizedV2Wildcard": "VectorizedV2Wildcard",
    "NaiveAutoWildcard": "Avx2Wildcard",
    "PairHorspool": "PairHorspool",
}
LENGTH_SERIES = [
    ("Avx2V2", ("NaiveAvx2V2", "NaiveAuto"), "#2b5d8a"),
    ("Avx2V2WC", ("NaiveAvx2V2Wildcard", "NaiveAutoWildcard"), "#e8923a"),
    ("Horspool", ("PairHorspool",), "#3f7d5a"),
    ("TwoWay2", ("TwoWay2",), "#7d4e9e"),
]
LENGTH_RAW_ALGORITHMS = [algo for _, algos, _ in LENGTH_SERIES for algo in algos]
LENGTH_COLORS = {name: color for name, _, color in LENGTH_SERIES}

FIG_FILES = {
    "length": "fig_algo_length.png",
    "tailrisk": "fig_algo_tailrisk.png",
    "winner": "fig_algo_winner_map.png",
    "matrix": "fig_algo_matrix.png",
    "sel": "fig_algo_selectivity.png",
    "base": "fig_algo_baselines.png",
    "ladder": "fig_algo_ladder.png",
}


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def label(ds):
    return LABELS.get(ds, ds)


def plot_group(ds):
    ds = str(ds)
    if ds == "GENCODE human transcripts":
        return "DNA"
    if ds == "quotes":
        return "Quotes"
    if ds.startswith("job_"):
        return "JOB"
    return label(ds)


def gmean(s):
    x = pd.to_numeric(s, errors="coerce").dropna(); x = x[x > 0]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def is_wildcard(algo):
    return str(algo).lower().endswith("wildcard")


def family_of(algo):
    a = str(algo).lower()
    if a.endswith("wildcard"):
        return 2
    if a in ("naive", "naivescalar"):
        return 0
    if a in ("twoway2", "twoway3"):
        return 3
    if a in ("bm", "utf8kmp", "twoway", "pairhorspool"):
        return 4
    if a in ("stdsearch", "libcmemmem"):
        return 5
    return 1


def safe_name(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def compact_pattern(pattern, max_inner=18):
    pattern = str(pattern)
    if len(pattern) <= max_inner + 2:
        return pattern
    prefix = pattern[: max_inner // 2]
    suffix = pattern[-max_inner // 2 :]
    return f"{prefix}...{suffix}"


def pattern_core_len(pattern):
    pattern = str(pattern)
    if pattern.startswith("%") and pattern.endswith("%"):
        return len(pattern) - 2
    return len(pattern)


def load_raw(csvs):
    """Filtered per-iteration rows with execute_ms and selectivity columns."""
    csv_paths = [Path(c).expanduser() for c in csvs]
    missing = [str(c) for c in csv_paths if not c.exists()]
    if missing:
        raise FileNotFoundError("missing benchmark CSV(s): " + ", ".join(missing))
    df = pd.concat([pd.read_csv(c) for c in csv_paths], ignore_index=True)
    df = df[df[COL["storage"]].astype(str).str.lower().eq("utf8")]
    df = df[df[COL["matcher"]].astype(str).str.lower().eq("static")]
    df = df[df[COL["requested_index"]].astype(str).str.lower().isin(["full-scan", "none"])]
    df = df[df[COL["actual_index"]].astype(str).str.lower().eq("full-scan")]
    df = df[~df[COL["algorithm"]].astype(str).str.startswith(DROP_PREFIXES)]
    df["execute_ms"] = pd.to_numeric(df[COL["execute_ns"]], errors="coerce") / 1e6
    m = pd.to_numeric(df[COL["rows_matched"]], errors="coerce")
    r = pd.to_numeric(df[COL["row_count"]], errors="coerce").replace(0, np.nan)
    df["selectivity"] = m / r
    return df


def cells(df):
    """One median execute (ms) + selectivity per (dataset, column, algorithm, pattern)."""
    keys = [COL["dataset"], COL["column"], COL["algorithm"], COL["pattern_name"], COL["pattern"]]
    return df.groupby(keys, as_index=False).agg(execute_ms=("execute_ms", "median"),
                                                selectivity=("selectivity", "median"))


# ======================================================================
# Figure: length sweep
# ======================================================================
def length_sweep(df, path="fig_algo_length.png"):
    c = cells(df)
    c = c[c[COL["pattern"]].astype(str).str.match(SINGLE_SEGMENT)].copy()
    c = c[c[COL["algorithm"]].isin(LENGTH_RAW_ALGORITHMS)].copy()
    series_by_algo = {
        algo: series
        for series, algos, _ in LENGTH_SERIES
        for algo in algos
    }
    inner = c[COL["pattern"]].astype(str).str.slice(1, -1)
    c["length"] = inner.str.len()
    c["kind"] = np.where(inner.str.contains("_"), "underscore", "exact")
    c["panel"] = c[COL["dataset"]].map(label)
    c["series"] = c[COL["algorithm"]].map(series_by_algo)
    g = (c.groupby(["panel", "length", "kind", "series"], as_index=False)["execute_ms"]
           .mean()
           .rename(columns={"execute_ms": "ms"}))

    panels = sorted(g["panel"].unique(), key=lambda p: (not p.startswith(("DNA", "Quotes")), p))
    kinds = [("exact", "-"), ("underscore", "--")]
    base = Path(path)
    for panel in panels:
        fig, ax = plt.subplots(figsize=(6.5, 4.7))
        sub = g[g["panel"].eq(panel)]
        for series, _, _ in LENGTH_SERIES:
            for kind, ls in kinds:
                line = sub[(sub["series"] == series) & (sub["kind"] == kind)].sort_values("length")
                if not line.empty:
                    ax.plot(line["length"], line["ms"], color=LENGTH_COLORS[series], ls=ls, marker="o",
                            ms=4.3, lw=1.8, label="_nolegend_")
        xs = sorted(sub["length"].unique())
        ax.set_xscale("log", base=2); ax.set_xticks(xs)
        if xs:
            ax.set_xlim(min(xs) / 1.08, max(xs) * 1.08)
        rotation = 45 if len(xs) > 8 else 0
        ax.set_xticklabels([int(x) for x in xs], rotation=rotation, ha="right" if rotation else "center")
        ax.minorticks_off()
        ax.set_xlabel("pattern length"); ax.set_title(panel, fontsize=15)
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_ylabel("median execute [ms]")
        handles = [
            Line2D([0], [0], color=LENGTH_COLORS[series], marker="o", lw=1.8,
                   label=series)
            for series, _, _ in LENGTH_SERIES
        ] + [
            Line2D([0], [0], color="#333", lw=1.8, ls="-", label="exact"),
            Line2D([0], [0], color="#333", lw=1.8, ls="--", label="underscore"),
        ]
        ax.legend(handles=handles, frameon=True, framealpha=0.88, facecolor="white",
                  edgecolor="none", fontsize=8.3, ncol=1, loc="upper left",
                  bbox_to_anchor=(0.015, 0.985), borderaxespad=0.0,
                  handlelength=2.0, labelspacing=0.20)
        _save(fig, base.with_name(f"{base.stem}_{safe_name(panel)}{base.suffix}"))


# ======================================================================
# Figure: tail risk (merged + optional per-dataset split)
# ======================================================================
def _tail_stats(frame):
    out = {}
    for algo, gg in frame.groupby(COL["algorithm"]):
        out[algo] = (gg["cls"].iloc[0], gmean(gg["ratio"]), float(gg["ratio"].max()))
    return out


def _tail_panel(ax, stats, order, title, xmax, show_labels):
    n = len(order)
    for i, algo in enumerate(order):
        if algo not in stats:
            continue
        cls, gm, mx = stats[algo]
        y = n - 1 - i
        c = NW if cls == "nw" else WC
        ax.plot([gm, mx], [y, y], color=c, lw=2.2, zorder=2, solid_capstyle="round")
        ax.scatter([mx], [y], color=c, s=66, zorder=3)
        ax.scatter([gm], [y], color="white", edgecolor=c, s=52, lw=1.8, zorder=4)
        ax.text(mx + 0.02 * xmax, y, f"{mx:.1f}", va="center", ha="left", fontsize=10, color="#333")
    ax.axvline(1.0, color="#888", lw=1.0, ls=":", zorder=1)
    ax.set_yticks(range(n)); ax.set_yticklabels(list(reversed(order)), fontsize=11)
    ax.tick_params(left=False, labelleft=show_labels)
    ax.set_xlim(0.8, xmax); ax.set_title(title, fontsize=14)
    ax.grid(True, axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def _tail_legend(ax):
    handles = [
        Line2D([0], [0], marker="o", ls="", mfc="white", mec="#444", ms=8, label="geomean slowdown"),
        Line2D([0], [0], marker="o", ls="", mfc="#444", mec="#444", ms=8, label="worst-case slowdown"),
        Line2D([0], [0], marker="s", ls="", mfc=NW, mec=NW, ms=9, label="non-wildcard"),
        Line2D([0], [0], marker="s", ls="", mfc=WC, mec=WC, ms=9, label="wildcard"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=10.5, loc="lower right")


def tailrisk(df, split=False, path="fig_algo_tailrisk.png"):
    path = Path(path)
    c = cells(df)
    c["workload"] = c[COL["dataset"]].map(plot_group)
    c["best"] = c.groupby([COL["dataset"], COL["column"], COL["pattern_name"]])["execute_ms"].transform("min")
    c["ratio"] = c["execute_ms"] / c["best"]
    c["cls"] = np.where(c[COL["algorithm"]].map(is_wildcard), "wc", "nw")

    all_stats = _tail_stats(c)
    order = sorted(all_stats, key=lambda a: all_stats[a][1])
    panels = [
        ("DNA", c[c["workload"].eq("DNA")]),
        ("Quotes", c[c["workload"].eq("Quotes")]),
        ("JOB", c[c["workload"].eq("JOB")]),
        ("All", c),
    ]
    for title, sub in panels:
        if sub.empty:
            continue
        stats = _tail_stats(sub)
        xmax = max(v[2] for v in stats.values()) * 1.12
        fig, ax = plt.subplots(figsize=(9, 6.2))
        _tail_panel(ax, stats, order, title, xmax, show_labels=True)
        ax.set_xlabel("slowdown vs. best algorithm  ($\\times$)")
        _tail_legend(ax)
        _save(fig, path.with_name(f"{path.stem}_{safe_name(title)}{path.suffix}"))


# ======================================================================
# Figure: winner map
# ======================================================================
def winner_map(df, path="fig_algo_winner_map.png"):
    c = cells(df)
    c = c[c[COL["pattern"]].astype(str).str.match(SINGLE_SEGMENT)].copy()
    c["length"] = c[COL["pattern"]].astype(str).str.slice(1, -1).str.len()
    c["fam"] = c[COL["algorithm"]].map(family_of)
    edges = [0, 1e-4, 1e-2, 1e-1, 0.5, 1.01]
    sel_lab = ["<1e-4", "1e-4..1e-2", "1e-2..0.1", "0.1..0.5", ">=0.5"]
    c["sb"] = pd.cut(c["selectivity"], edges, labels=False)

    datasets = sorted(c[COL["dataset"]].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.6), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = c[c[COL["dataset"]].eq(ds)]
        lengths = sorted(sub["length"].dropna().unique())
        grid = np.full((len(sel_lab), len(lengths)), np.nan)
        for ri in range(len(sel_lab)):
            for ci, L in enumerate(lengths):
                cell = sub[(sub["sb"] == ri) & (sub["length"] == L)]
                if not cell.empty:
                    grid[ri, ci] = cell.loc[cell["execute_ms"].idxmin(), "fam"]
        ax.pcolormesh(np.arange(len(lengths) + 1), np.arange(len(sel_lab) + 1),
                      np.ma.masked_invalid(grid), cmap=FAM_CMAP, norm=FAM_NORM,
                      edgecolors="white", linewidth=2)
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xticks(np.arange(len(lengths)) + 0.5); ax.set_xticklabels([int(x) for x in lengths])
        ax.set_yticks(np.arange(len(sel_lab)) + 0.5); ax.set_yticklabels(sel_lab, fontsize=10)
        ax.set_xlabel("pattern length"); ax.set_title(label(ds), fontsize=15)
    axes[0][0].set_ylabel("selectivity")
    handles = [Line2D([0], [0], marker="s", ls="", ms=12, mfc=col, mec="none", label=l)
               for col, l in zip(FAM_COLORS, FAMILIES)]
    fig.legend(handles=handles, frameon=False, ncol=len(FAMILIES),
               loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig); print("saved", path)


# ======================================================================
# Figure: evidence matrix
# ======================================================================
def evidence_matrix(df, path="fig_algo_matrix.png"):
    c = cells(df)
    c["plot_pattern"] = c[COL["dataset"]].map(label) + "\n" + c[COL["pattern_name"]].astype(str)
    if MATRIX_PATTERNS:
        c = c[
            c[COL["pattern_name"]].isin(MATRIX_PATTERNS)
            | c["plot_pattern"].isin(MATRIX_PATTERNS)
        ]
    piv = c.pivot_table(index=COL["algorithm"], columns="plot_pattern",
                        values="execute_ms", aggfunc="median")
    if not MATRIX_PATTERNS:
        print("  note: MATRIX_PATTERNS is None -> showing all patterns; curate for the thesis.")
    if piv.empty:
        print("  note: no matrix patterns found. Skipping matrix.")
        return
    piv = piv.loc[sorted(piv.index, key=lambda a: (family_of(a) == 2, family_of(a), a))]
    rel = np.log2(piv.div(piv.min(axis=0), axis=1)).clip(upper=2.5)
    fig, ax = plt.subplots(figsize=(0.55 * piv.shape[1] + 3, 0.42 * piv.shape[0] + 1.6))
    im = ax.imshow(rel.values, cmap="magma_r", aspect="auto", vmin=0, vmax=2.5)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=9)
    for ci in range(piv.shape[1]):
        col = piv.values[:, ci]
        if np.isfinite(col).any():
            ax.scatter(ci, int(np.nanargmin(col)), marker="o", s=18,
                       facecolor="white", edgecolor="black", lw=0.8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02).set_label("$\\log_2$( time / column best )")
    _save(fig, path)


# ======================================================================
# Figure: time vs selectivity
# ======================================================================
def selectivity_plot(df, path="fig_algo_selectivity.png"):
    c = cells(df)
    c["workload"] = c[COL["dataset"]].map(plot_group)
    c["best"] = c.groupby([COL["dataset"], COL["column"], COL["pattern_name"]])["execute_ms"].transform("min")
    c["family"] = c[COL["algorithm"]].map(lambda a: FAMILIES[family_of(a)])
    edges = [0, 1e-4, 1e-2, 1e-1, 0.5, 1.01]
    bins = ["<1e-4", "1e-4..1e-2", "1e-2..0.1", "0.1..0.5", ">=0.5"]
    c["sel_bin"] = pd.cut(c["selectivity"], edges, labels=bins, include_lowest=True)

    pattern_keys = ["workload", COL["dataset"], COL["column"], COL["pattern_name"], "sel_bin"]
    fam = (c.groupby(pattern_keys + ["family"], observed=True)
             .agg(family_best_ms=("execute_ms", "min"), best_ms=("best", "first"))
             .reset_index())
    fam["slowdown"] = fam["family_best_ms"] / fam["best_ms"]

    pattern_bins = c.drop_duplicates([COL["dataset"], COL["column"], COL["pattern_name"]])

    panels = [
        ("DNA", fam[fam["workload"].eq("DNA")], pattern_bins[pattern_bins["workload"].eq("DNA")]),
        ("Quotes", fam[fam["workload"].eq("Quotes")], pattern_bins[pattern_bins["workload"].eq("Quotes")]),
        ("JOB", fam[fam["workload"].eq("JOB")], pattern_bins[pattern_bins["workload"].eq("JOB")]),
        ("All", fam, pattern_bins),
    ]
    fig = plt.figure(figsize=(13.6, 7.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.035], wspace=0.08, hspace=0.50)
    axes = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
    ])
    cax = fig.add_subplot(gs[:, 2])
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#f3f4f6")
    last_im = None
    for ax, (title, sub, pat_sub) in zip(axes.ravel(), panels):
        grid = np.full((len(FAMILIES), len(bins)), np.nan)
        counts = []
        for bi, bin_label in enumerate(bins):
            counts.append(int(pat_sub[pat_sub["sel_bin"].eq(bin_label)].shape[0]))
            for fi, family in enumerate(FAMILIES):
                vals = sub[sub["family"].eq(family) & sub["sel_bin"].eq(bin_label)]["slowdown"].dropna()
                if len(vals):
                    grid[fi, bi] = gmean(vals)
        last_im = ax.imshow(np.ma.masked_invalid(np.log2(grid)), cmap=cmap, vmin=0, vmax=3.2,
                            aspect="auto")
        for fi in range(len(FAMILIES)):
            for bi in range(len(bins)):
                val = grid[fi, bi]
                text = "-" if np.isnan(val) else f"{val:.2f}x"
                color = "white" if not np.isnan(val) and np.log2(val) > 1.8 else "#111"
                ax.text(bi, fi, text, ha="center", va="center", fontsize=8.3, color=color)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(bins, counts)],
                           rotation=20, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(FAMILIES)))
        ax.set_yticklabels(FAMILIES, fontsize=9)
        if ax in (axes[1][0], axes[1][1]):
            ax.set_xlabel("row selectivity bin")
        if ax not in (axes[0][0], axes[1][0]):
            ax.tick_params(axis="y", labelleft=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("family-best geomean slowdown", fontsize=11, labelpad=8)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(["1x", "2x", "4x", "8x"])
        cbar.ax.tick_params(labelsize=10)
    _save(fig, path, tight=False)


# ======================================================================
# Figure: baselines with real per-iteration error bars
# ======================================================================
def baselines_plot(df, path="fig_algo_baselines.png"):
    pat = BASELINE_PATTERN
    if pat is None or pat not in set(df[COL["pattern_name"]].unique()):
        pat = df.loc[pd.to_numeric(df[COL["row_count"]], errors="coerce").idxmax(), COL["pattern_name"]]
        print(f"  note: BASELINE_PATTERN missing -> auto-picked '{pat}'.")
    sub = df[df[COL["pattern_name"]].eq(pat)]
    # restrict to one (dataset, column) -- the largest table -- to avoid pooling
    key = sub.loc[pd.to_numeric(sub[COL["row_count"]], errors="coerce").idxmax(),
                  [COL["dataset"], COL["column"]]]
    sub = sub[(sub[COL["dataset"]] == key[COL["dataset"]]) & (sub[COL["column"]] == key[COL["column"]])]

    algos = [a for a in [DEFAULT] + BASELINES if a in sub[COL["algorithm"]].unique()]
    g = sub.groupby(COL["algorithm"])["execute_ms"]
    med, sd = g.median(), g.std().fillna(0)
    base = med[DEFAULT]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, a in enumerate(algos):
        c = NW if a == DEFAULT else "#9aa7b1"
        ax.bar(i, med[a], color=c, yerr=sd[a], capsize=3, ecolor="#555")
        if a != DEFAULT:
            ax.text(i, med[a] + sd[a], f"+{100*(med[a]/base-1):.0f}%", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(range(len(algos))); ax.set_xticklabels(algos, rotation=25, ha="right")
    pattern = sub[COL["pattern"]].iloc[0]
    ax.set_ylabel("execute [ms]"); ax.set_title(f"pattern: {pattern}", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    if COL["compile_ns"] in df.columns:
        cm = pd.to_numeric(sub[sub[COL["algorithm"]].eq(DEFAULT)][COL["compile_ns"]], errors="coerce").median()
        ax.text(0.02, 0.95, f"{DEFAULT}\ncompile_ns (once)\n{cm/1e6:.4f} ms",
                transform=ax.transAxes, fontsize=9, va="top", color="#444")
    _save(fig, path)


# ======================================================================
# Figure: optimization ladder
# ======================================================================
def ladder_plot(df, path="fig_algo_ladder.png"):
    base = Path(path)
    for group_name, cases in LADDER_GROUPS.items():
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), squeeze=False)
        for ax, (dataset, column, pattern_name, title_prefix) in zip(axes.ravel(), cases):
            sub = df[
                df[COL["dataset"]].eq(dataset)
                & df[COL["column"]].eq(column)
                & df[COL["pattern_name"]].eq(pattern_name)
            ]
            if sub.empty:
                ax.set_axis_off()
                ax.set_title(f"missing: {title_prefix}")
                continue
            _draw_ladder_panel(ax, sub, title_prefix)
        _save(fig, base.with_name(f"{base.stem}_{group_name}{base.suffix}"))

    c = cells(df)
    aggregate_cases = [
        ("DNA", c[c[COL["dataset"]].map(plot_group).eq("DNA")], "DNA aggregate"),
        ("Quotes", c[c[COL["dataset"]].map(plot_group).eq("Quotes")], "Quotes aggregate"),
        ("JOB", c[c[COL["dataset"]].map(plot_group).eq("JOB")], "JOB aggregate"),
        ("All", c, "All workloads aggregate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), squeeze=False)
    for ax, (_, sub, title) in zip(axes.ravel(), aggregate_cases):
        if sub.empty:
            ax.set_axis_off()
            ax.set_title(f"missing: {title}")
            continue
        _draw_ladder_panel(ax, sub, title, aggregate=True)
    _save(fig, base.with_name(f"{base.stem}_aggregate{base.suffix}"))


def _draw_ladder_panel(ax, sub, title, aggregate=False):
    steps = [s for s in LADDER_STEPS if s in sub[COL["algorithm"]].unique()]
    if aggregate:
        med = sub[sub[COL["algorithm"]].isin(steps)].groupby(COL["algorithm"])["execute_ms"].apply(gmean)
    else:
        med = sub.groupby(COL["algorithm"])["execute_ms"].median()
        pattern = sub[COL["pattern"]].iloc[0]
        compact = compact_pattern(pattern)
        title = f"{title}: {compact} (len {pattern_core_len(pattern)})"
    colors = [WC if is_wildcard(s) else ("#3f7d5a" if s == "PairHorspool" else NW) for s in steps]
    ax.bar(range(len(steps)), [med[s] for s in steps], color=colors)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([LADDER_LABELS.get(s, s) for s in steps], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("geomean execute [ms]" if aggregate else "execute [ms]")
    ax.set_title(title, fontsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)


def _save(fig, path, tight=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig); print("saved", path)


FIGS = {"length": length_sweep, "winner": winner_map, "matrix": evidence_matrix,
        "sel": selectivity_plot, "base": baselines_plot, "ladder": ladder_plot}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", default=DEFAULT_RAW_CSVS,
                    help="one or more raw per-iteration CSVs; defaults to the chapter 6 benchmark outputs")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="directory for generated plots")
    ap.add_argument("--only", choices=list(FIGS) + ["tailrisk"], help="generate just one figure")
    ap.add_argument("--split", action="store_true", help="tail-risk: also emit per-dataset panels")
    a = ap.parse_args()
    df = load_raw(a.csv)
    names = [a.only] if a.only else (["length", "tailrisk", "sel", "base", "ladder"])
    for name in names:
        path = a.out_dir / FIG_FILES[name]
        if name == "tailrisk":
            tailrisk(df, split=a.split, path=path)
        else:
            FIGS[name](df, path=path)
