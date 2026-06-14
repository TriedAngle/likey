"""
index_chapter_plots.py
======================
All figures for the index-assisted candidate-generation chapter
(07_indexes.tex), computed from scratch from the RAW per-iteration
benchmark CSV(s). Companion to algo_chapter_plots.py and follows the same
conventions (style, helpers, placeholders, argparse).

Usage
-----
    python3 plots/index_chapter_plots.py
    python3 plots/index_chapter_plots.py --main-csv dna_idx.csv quotes_idx.csv 'job/index-memmem/*/raw.csv'
    python3 plots/index_chapter_plots.py --only applicability
    python3 plots/index_chapter_plots.py --only sensitivity --sens-csv 'job/index-consistency/*/raw.csv'
    python3 plots/index_chapter_plots.py --all-heatmap-patterns

Figures (name -> output file -> LaTeX label)
    applicability  fig_index_applicability_heatmap.png  fig:index-applicability-heatmap
    candidate      fig_index_candidate_fraction.png     fig:index-candidate-fraction
    sensitivity    fig_index_algo_sensitivity.png       fig:index-algo-sensitivity

How values are derived (nothing is hard-coded)
    - per-iteration rows are aggregated to one MEDIAN query_total time per
      (dataset, column, algorithm, requested_index, pattern);
    - speedup = median_query_total(full-scan) / median_query_total(index)
      for the same (dataset, column, algorithm, pattern); fallbacks land near 1x;
    - candidate fraction = candidate_rows_seen / row_count;
    - selectivity = rows_matched / row_count;
    - "applied" iff fallback_reason is empty; otherwise the request fell back to
      full scan and the cell is categorised by its fallback reason.

Two CSV groups, matching the analysis isolation
    --main-csv  the LibcMemmem index runs used for applicability / speedup /
                candidate-fraction (DNA + quotes index-comparison, JOB index-memmem).
    --sens-csv  the runs that contain LibcMemmem, PairHorspool and
                NaiveAutoWildcard for the verifier-sensitivity figure
                (DNA + quotes index-comparison, JOB index-consistency).
    Both are filtered to storage=utf8, generic_matcher=static.

=== PLACEHOLDERS (safe auto-fallbacks let it run as-is) ===
    HEATMAP_CASES     representative cases used by default for the heatmaps.
                      Use --all-heatmap-patterns to show every resolved pattern.
    LABELS            raw dataset name -> short panel title.
"""

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ----------------------------------------------------------------------
# Style (matches the thesis reference plots / algo_chapter_plots.py)
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 14, "axes.linewidth": 1.2,
})
GRID = "#cfcfcf"

# ----------------------------------------------------------------------
# Index identity, ordering, colours and markers (reused across figures)
# ----------------------------------------------------------------------
INDEX_ORDER = ["prefix-btree", "qgram", "trigram", "fm"]
INDEX_LABEL = {"prefix-btree": "B-tree", "qgram": "q-gram", "trigram": "trigram", "fm": "FM"}
INDEX_COLORS = {
    "prefix-btree": "#2b5d8a",   # blue
    "qgram": "#3f7d5a",          # green
    "trigram": "#e8923a",        # orange
    "fm": "#7d4e9e",             # purple
}
INDEX_MARKERS = {
    "prefix-btree": "o", "qgram": "s", "trigram": "^", "fm": "D", "full-scan": "x",
}
FALLBACK_ORDER = [
    "no-indexable-prefix",
    "literal-shorter-than-qgram",
    "literal-shorter-than-trigram",
    "too-broad",
    "no-indexable-literal",
]
FALLBACK_COLORS = {
    "no-indexable-prefix": "#9aa7b1",
    "literal-shorter-than-qgram": "#cbb682",
    "literal-shorter-than-trigram": "#c98f5a",
    "too-broad": "#c0392b",
    "no-indexable-literal": "#6b7280",
    "other": "#d9d9d9",
}
# categorical order for the applicability map: applied-as-<index> then fallback reasons
CAT_ORDER = INDEX_ORDER + FALLBACK_ORDER + ["other"]
CAT_COLORS = [INDEX_COLORS[i] for i in INDEX_ORDER] + \
             [FALLBACK_COLORS[f] for f in FALLBACK_ORDER] + [FALLBACK_COLORS["other"]]
CAT_CODE = {name: i for i, name in enumerate(CAT_ORDER)}
CAT_TEXT = {
    "prefix-btree": "B",
    "qgram": "Q",
    "trigram": "T",
    "fm": "FM",
    "no-indexable-prefix": "no prefix",
    "literal-shorter-than-qgram": "<15",
    "literal-shorter-than-trigram": "<3",
    "too-broad": "broad",
    "no-indexable-literal": "no lit",
    "other": "-",
}
CAT_TEXT_COLOR = {
    "prefix-btree": "white",
    "qgram": "white",
    "trigram": "#111",
    "fm": "white",
    "too-broad": "white",
}

ALG_ORDER = ["LibcMemmem", "PairHorspool", "NaiveAutoWildcard"]
ALG_LABEL = {"LibcMemmem": "LibcMemmem", "PairHorspool": "PairHorspool",
             "NaiveAutoWildcard": "NaiveAutoWildcard"}
ALG_COLORS = {"LibcMemmem": "#2b5d8a", "PairHorspool": "#3f7d5a", "NaiveAutoWildcard": "#e8923a"}

SUITE_ORDER = ["DNA", "Quotes", "JOB"]

# ----------------------------------------------------------------------
# Default raw CSVs (the index-benchmark outputs described in index_analysis.md)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "plots" / "outputs" / "07"
_DNA_IDX = PROJECT_ROOT / "benchmark_results/dna/index-comparison/gencode_20260609_120418/raw.csv"
_DNA_EXACT_IDX = PROJECT_ROOT / "benchmark_results/dna/exact-vs-underscore-indexes/gencode_20260609_113634/raw.csv"
_QUOTES_IDX = PROJECT_ROOT / "benchmark_results/quotes/index-comparison/quotes_20260609_063013/raw.csv"
_QUOTES_EXACT = PROJECT_ROOT / "benchmark_results/quotes/exact-vs-underscore/quotes_20260609_062306/raw.csv"
_QUOTES_CASES = PROJECT_ROOT / "benchmark_results/quotes/algorithm-cases/quotes_20260609_062629/raw.csv"
DEFAULT_MAIN_CSVS = [
    str(_DNA_IDX),
    str(_DNA_EXACT_IDX),
    str(_QUOTES_IDX),
    str(_QUOTES_EXACT),
    str(_QUOTES_CASES),
    str(PROJECT_ROOT / "benchmark_results/job/index-memmem/*/raw.csv"),
]
DEFAULT_SENS_CSVS = [
    str(_DNA_IDX),
    str(_QUOTES_IDX),
    str(PROJECT_ROOT / "benchmark_results/job/index-consistency/*/raw.csv"),
]

COL = dict(
    dataset="dataset", column="column", storage="storage", algorithm="algorithm",
    matcher="generic_matcher", requested_index="requested_index", actual_index="actual_index",
    pattern_name="pattern_name", pattern="pattern", iteration="iteration",
    row_count="row_count", index_build_ns="index_build_ns",
    query_total_ns="query_total_ns", candidate_rows_seen="candidate_rows_seen",
    rows_matched="rows_matched", fallback_reason="fallback_reason",
)

LABELS = {"GENCODE human transcripts": "DNA", "quotes": "Quotes"}

# These rows are intentionally chosen to make the heatmaps explain index
# semantics instead of becoming a long benchmark dump.
HEATMAP_CASES = [
    ("DNA", "equality_nohit_len016", "equality no-hit"),
    ("DNA", "prefix_exact_len032", "prefix selective"),
    ("DNA", "prefix_gap_len016", "prefix + _ gap"),
    ("DNA", "contains_exact_len032", "contains long"),
    ("DNA", "exact_len004", "contains 4 bp"),
    ("DNA", "exact_len032", "contains 32 bp"),
    ("DNA", "underscore_len032", "contains + _ gap"),
    ("Quotes", "equality_age_issue_full", "equality quote"),
    ("Quotes", "prefix_learning", "prefix selective"),
    ("Quotes", "prefix_gap_age", "prefix + _ gap"),
    ("Quotes", "contains_relationship", "contains phrase"),
    ("Quotes", "exact_len004", "contains common"),
    ("Quotes", "underscore_len004", "contains + _ short"),
    ("Quotes", "exact_len032", "contains long"),
    ("Quotes", "bm_good_rare_nohit", "rare no-hit"),
    ("JOB", "prefix_birdemic", "prefix rare"),
    ("JOB", "prefix_b", "prefix broad"),
    ("JOB", "contains_b", "contains 1 char"),
    ("JOB", "contains_ang", "contains 3 chars"),
    ("JOB", "contains_co_production", "contains long"),
    ("JOB", "contains_usa", "contains broad"),
    ("JOB", "prefix_usa_200", "prefix + %"),
    ("JOB", "contains_downey_robert", "two fragments"),
    ("JOB", "contains_200_wild", "internal %"),
]

FIG_FILES = {
    "applicability": "fig_index_applicability_heatmap.png",
    "candidate": "fig_index_candidate_fraction.png",
    "sensitivity": "fig_index_algo_sensitivity.png",
}


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def label(ds):
    return LABELS.get(ds, ds)


def plot_group(ds):
    s = str(ds).lower()
    if "gencode" in s or "transcript" in s or s.startswith("dna"):
        return "DNA"
    if s.startswith("quote"):
        return "Quotes"
    if s.startswith("job") or s.startswith("imdb"):
        return "JOB"
    return str(ds)


def gmean(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    x = x[x > 0]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def safe_name(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def compact_pattern(pattern, max_inner=22):
    pattern = str(pattern)
    if len(pattern) <= max_inner:
        return pattern
    half = (max_inner - 3) // 2
    return f"{pattern[:half]}...{pattern[-half:]}"


def _norm_index(value):
    v = str(value).strip().lower().replace("_", "-")
    return v


def _expand(csvs):
    """Expand globs, keep existing files; error only if nothing resolves."""
    out, missing = [], []
    for entry in csvs:
        s = str(Path(entry).expanduser())
        if "*" in s or "?" in s:
            hits = sorted(glob.glob(s))
            out.extend(hits)
            if not hits:
                missing.append(s)
        elif Path(s).exists():
            out.append(s)
        else:
            missing.append(s)
    out = list(dict.fromkeys(out))  # dedupe, keep order
    if missing:
        print("  note: no files matched: " + ", ".join(missing))
    if not out:
        raise FileNotFoundError("no benchmark CSVs resolved from: " + ", ".join(map(str, csvs)))
    return out


def load_raw(csvs, algorithms=None):
    """Filtered per-iteration rows with derived ms / fraction columns."""
    paths = _expand(csvs)
    df = pd.concat([pd.read_csv(c) for c in paths], ignore_index=True)
    df = df[df[COL["storage"]].astype(str).str.lower().eq("utf8")]
    df = df[df[COL["matcher"]].astype(str).str.lower().eq("static")]
    if algorithms is not None:
        df = df[df[COL["algorithm"]].isin(algorithms)]
    df = df.copy()
    df[COL["requested_index"]] = df[COL["requested_index"]].map(_norm_index)
    df[COL["actual_index"]] = df[COL["actual_index"]].map(_norm_index)
    df["query_ms"] = pd.to_numeric(df[COL["query_total_ns"]], errors="coerce") / 1e6
    df["build_ms"] = pd.to_numeric(df[COL["index_build_ns"]], errors="coerce") / 1e6
    rc = pd.to_numeric(df[COL["row_count"]], errors="coerce").replace(0, np.nan)
    df["candidate_frac"] = pd.to_numeric(df[COL["candidate_rows_seen"]], errors="coerce") / rc
    df["selectivity"] = pd.to_numeric(df[COL["rows_matched"]], errors="coerce") / rc
    fb = df[COL["fallback_reason"]].astype(str).str.strip()
    df["fallback"] = fb.where(~fb.isin(["", "nan", "None"]), other=np.nan)
    df["applied"] = df["fallback"].isna()
    # drop accidental duplicate iterations from overlapping CSVs
    key = [COL["dataset"], COL["column"], COL["algorithm"], COL["requested_index"],
           COL["pattern_name"], COL["pattern"], COL["iteration"]]
    df = df.drop_duplicates(subset=key)
    return df


def speedups(df):
    """One median query (ms) + speedup vs full scan per index/pattern/algorithm cell."""
    keys = [COL["dataset"], COL["column"], COL["algorithm"], COL["requested_index"],
            COL["pattern_name"], COL["pattern"]]
    agg = df.groupby(keys, as_index=False).agg(
        query_ms=("query_ms", "median"),
        candidate_frac=("candidate_frac", "median"),
        selectivity=("selectivity", "median"),
        build_ms=("build_ms", "median"),
        row_count=(COL["row_count"], "first"),
        actual_index=(COL["actual_index"], "first"),
        applied=("applied", "first"),
        fallback=("fallback", "first"),
    )
    base = (agg[agg[COL["requested_index"]].eq("full-scan")]
            [[COL["dataset"], COL["column"], COL["algorithm"], COL["pattern_name"], "query_ms"]]
            .rename(columns={"query_ms": "fullscan_ms"}))
    agg = agg.merge(base, on=[COL["dataset"], COL["column"], COL["algorithm"], COL["pattern_name"]],
                    how="left")
    agg["speedup"] = agg["fullscan_ms"] / agg["query_ms"]
    agg["workload"] = agg[COL["dataset"]].map(plot_group)
    agg["pkey"] = (agg[COL["dataset"]].astype(str) + "||" + agg[COL["column"]].astype(str)
                   + "||" + agg[COL["pattern_name"]].astype(str) + "||" + agg[COL["pattern"]].astype(str))
    agg["plabel"] = agg[COL["pattern"]].map(compact_pattern)
    return agg


def _suite_panels(agg):
    return [s for s in SUITE_ORDER if (agg["workload"].eq(s)).any()]


def _heatmap_cases(agg, use_cases=True):
    if not use_cases:
        out = agg.copy()
        out["case_rank"] = np.arange(len(out))
        return out

    chunks = []
    for rank, (suite, pattern_name, case_label) in enumerate(HEATMAP_CASES):
        sub = agg[(agg["workload"].eq(suite)) & (agg[COL["pattern_name"]].eq(pattern_name))]
        if sub.empty:
            continue
        sub = sub.copy()
        sub["case_rank"] = rank
        sub["plabel"] = case_label
        chunks.append(sub)
    if not chunks:
        return agg.copy()
    return pd.concat(chunks, ignore_index=True)


def _ordered_pkeys(sub):
    sort_cols = ["case_rank", COL["column"], COL["pattern_name"], "pkey"] \
        if "case_rank" in sub.columns else [COL["column"], COL["pattern_name"], "pkey"]
    rows = (sub[["pkey", "plabel", COL["column"], COL["pattern_name"], "selectivity"]
                + (["case_rank"] if "case_rank" in sub.columns else [])]
            .drop_duplicates("pkey")
            .sort_values(sort_cols))
    return list(rows["pkey"]), dict(zip(rows["pkey"], rows["plabel"]))


def _cell_lookup(sub):
    return {(r.pkey, getattr(r, COL["requested_index"])): r for r in sub.itertuples()}


def _cat_name(rec):
    if rec.applied:
        return getattr(rec, COL["requested_index"])
    reason = str(rec.fallback)
    return reason if reason in CAT_CODE else "other"


def _format_speedup(value):
    if not np.isfinite(value) or value <= 0:
        return ""
    if value < 1.15:
        return "1x"
    if value < 10:
        return f"{value:.1f}x"
    return f"{value:.0f}x"


# ======================================================================
# Figure: applicability / fallback map (categorical, "best-choice grid" style)
# ======================================================================
def _applicability_code(rec):
    return CAT_CODE.get(_cat_name(rec), CAT_CODE["other"])


def _best_index_by_row(look, pkeys):
    best = {}
    for pk in pkeys:
        candidates = []
        for idx in INDEX_ORDER:
            rec = look.get((pk, idx))
            if rec is None or not rec.applied:
                continue
            if np.isfinite(rec.speedup) and rec.speedup > 1.0:
                candidates.append((rec.speedup, idx))
        if candidates:
            best[pk] = max(candidates)[1]
    return best


def applicability_heatmap(agg, path, use_cases=True):
    a = agg[agg[COL["requested_index"]].isin(INDEX_ORDER)].copy()
    a = _heatmap_cases(a, use_cases=use_cases)
    suites = _suite_panels(a)
    per_suite = {s: _ordered_pkeys(a[a["workload"].eq(s)]) for s in suites}
    counts = [len(per_suite[s][0]) for s in suites]
    total = sum(counts)

    cmap = ListedColormap(CAT_COLORS)
    cmap.set_bad("white")
    norm = BoundaryNorm(np.arange(-0.5, len(CAT_ORDER) + 0.5), cmap.N)

    fig = plt.figure(figsize=(6.2, max(3.2, 0.26 * total + 1.3 * len(suites))))
    gs = fig.add_gridspec(len(suites), 1, height_ratios=counts, hspace=0.10 + 0.02 * len(suites))
    present_codes = set()
    for si, suite in enumerate(suites):
        ax = fig.add_subplot(gs[si, 0])
        pkeys, labels = per_suite[suite]
        look = _cell_lookup(a[a["workload"].eq(suite)])
        best = _best_index_by_row(look, pkeys)
        grid = np.full((len(pkeys), len(INDEX_ORDER)), np.nan)
        for i, pk in enumerate(pkeys):
            for j, idx in enumerate(INDEX_ORDER):
                rec = look.get((pk, idx))
                if rec is not None:
                    code = _applicability_code(rec)
                    grid[i, j] = code
                    present_codes.add(code)
        ax.pcolormesh(np.arange(len(INDEX_ORDER) + 1), np.arange(len(pkeys) + 1),
                      np.ma.masked_invalid(grid), cmap=cmap, norm=norm,
                      edgecolors="white", linewidth=1.6)
        for i, pk in enumerate(pkeys):
            for j, idx in enumerate(INDEX_ORDER):
                rec = look.get((pk, idx))
                if rec is None:
                    continue
                cat = _cat_name(rec)
                if rec.applied:
                    is_winner = best.get(pk) == idx
                    if not is_winner:
                        ax.add_patch(Rectangle((j, i), 1, 1, facecolor="white",
                                               edgecolor="none", alpha=0.48, zorder=2))
                    else:
                        ax.add_patch(Rectangle((j, i), 1, 1, fill=False,
                                               edgecolor="#111", linewidth=2.0, zorder=3))
                    speed = _format_speedup(rec.speedup)
                    text = f"{CAT_TEXT.get(cat, '-')} {speed}".strip()
                    size = 6.9 if is_winner else 6.7
                    color = CAT_TEXT_COLOR.get(cat, "#111") if is_winner else "#111"
                    weight = "bold" if is_winner else "semibold"
                else:
                    text = CAT_TEXT.get(cat, "-")
                    size = 7.2
                    color = CAT_TEXT_COLOR.get(cat, "#111")
                    weight = "bold"
                ax.text(j + 0.5, i + 0.5, text, ha="center", va="center",
                        fontsize=size, color=color, fontweight=weight, zorder=4)
        ax.invert_yaxis()
        ax.set_yticks(np.arange(len(pkeys)) + 0.5)
        ax.set_yticklabels([labels[pk] for pk in pkeys], fontsize=8)
        ax.set_xticks(np.arange(len(INDEX_ORDER)) + 0.5)
        if si == len(suites) - 1:
            ax.set_xticklabels([INDEX_LABEL[i] for i in INDEX_ORDER], fontsize=11)
        else:
            ax.set_xticklabels([])
        ax.tick_params(length=0)
        ax.set_ylabel(suite, fontsize=12)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)

    handles = []
    for idx in INDEX_ORDER:
        if CAT_CODE[idx] in present_codes:
            handles.append(Line2D([0], [0], marker="s", ls="", ms=10, mec="none",
                                  mfc=INDEX_COLORS[idx], label=f"applied: {INDEX_LABEL[idx]}"))
    for fb in FALLBACK_ORDER + ["other"]:
        if CAT_CODE[fb] in present_codes:
            handles.append(Line2D([0], [0], marker="s", ls="", ms=10, mec="none",
                                  mfc=FALLBACK_COLORS[fb], label=fb))
    fig.legend(handles=handles, frameon=False, ncol=3, fontsize=8.5,
               loc="lower center", bbox_to_anchor=(0.5, -0.015))
    _save(fig, path, tight=False)


# ======================================================================
# Figure: speedup vs candidate fraction (log-log scatter, marker per index)
# ======================================================================
def candidate_fraction(agg, path):
    a = agg[agg[COL["requested_index"]].isin(INDEX_ORDER)].copy()
    a = a[np.isfinite(a["speedup"]) & a["speedup"].gt(0)]
    floor = 1e-7
    a["cf"] = a["candidate_frac"].clip(lower=floor).fillna(1.0)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for idx in INDEX_ORDER:
        appl = a[a[COL["requested_index"]].eq(idx) & a["applied"]]
        if not appl.empty:
            ax.scatter(appl["cf"], appl["speedup"], marker=INDEX_MARKERS[idx], s=42,
                       facecolor=INDEX_COLORS[idx], edgecolor="white", linewidth=0.6,
                       alpha=0.9, zorder=3, label=INDEX_LABEL[idx])
    fell = a[~a["applied"]]
    if not fell.empty:
        ax.scatter(fell["cf"], fell["speedup"], marker="x", s=34, color="#c0392b",
                   linewidth=1.2, zorder=2, label="fell back (incl. too-broad)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.0, color="#888", lw=1.0, ls=":", zorder=1)
    ax.set_xlabel("candidate fraction  (candidate rows / row count)")
    ax.set_ylabel("speedup over full scan  ($\\times$)")
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=True, framealpha=0.9, facecolor="white", edgecolor="none",
              fontsize=10, loc="upper right")
    ax.set_title("Speedup tracks candidate reduction", fontsize=14)
    _save(fig, path)


# ======================================================================
# Figure: verifier-algorithm sensitivity (grouped bars per suite, log y)
# ======================================================================
def algo_sensitivity(agg, path):
    a = agg[agg[COL["requested_index"]].isin(INDEX_ORDER)].copy()
    g = (a.groupby(["workload", COL["algorithm"], COL["requested_index"]])["speedup"]
           .apply(gmean).reset_index(name="geomean"))
    suites = [s for s in SUITE_ORDER if (g["workload"].eq(s)).any()]
    algos = [al for al in ALG_ORDER if (g[COL["algorithm"]].eq(al)).any()]

    fig, axes = plt.subplots(1, len(suites), figsize=(4.6 * len(suites), 4.8),
                             squeeze=False, sharey=True)
    x = np.arange(len(INDEX_ORDER))
    width = 0.8 / max(len(algos), 1)
    for ax, suite in zip(axes[0], suites):
        sub = g[g["workload"].eq(suite)]
        for k, al in enumerate(algos):
            vals = []
            for idx in INDEX_ORDER:
                cell = sub[(sub[COL["algorithm"]].eq(al)) & (sub[COL["requested_index"]].eq(idx))]
                vals.append(float(cell["geomean"].iloc[0]) if not cell.empty else np.nan)
            offset = (k - (len(algos) - 1) / 2) * width
            ax.bar(x + offset, vals, width=width * 0.95, color=ALG_COLORS[al],
                   label=ALG_LABEL[al] if ax is axes[0][0] else "_nolegend_")
        ax.set_yscale("log")
        ax.axhline(1.0, color="#888", lw=1.0, ls=":", zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([INDEX_LABEL[i] for i in INDEX_ORDER], fontsize=11)
        ax.set_title(suite, fontsize=14)
        ax.grid(True, axis="y", which="major", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel("geomean speedup vs full scan  ($\\times$)")
    handles = [Line2D([0], [0], marker="s", ls="", ms=10, mec="none", mfc=ALG_COLORS[al],
                      label=ALG_LABEL[al]) for al in algos]
    fig.legend(handles=handles, frameon=False, ncol=len(algos), fontsize=10,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Index speedup is stable across verifier algorithms", fontsize=14)
    _save(fig, path, rect=(0, 0.06, 1, 0.96))


def _save(fig, path, tight=True, rect=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout(rect=rect) if rect is not None else fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# Figures that consume the LibcMemmem main runs vs the multi-algorithm sens runs.
MAIN_FIGS = {"applicability": applicability_heatmap,
             "candidate": candidate_fraction}
SENS_FIGS = {"sensitivity": algo_sensitivity}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-csv", nargs="+", default=DEFAULT_MAIN_CSVS,
                    help="LibcMemmem index runs (DNA/quotes index-comparison, JOB index-memmem)")
    ap.add_argument("--sens-csv", nargs="+", default=DEFAULT_SENS_CSVS,
                    help="multi-algorithm runs for the sensitivity figure "
                         "(DNA/quotes index-comparison, JOB index-consistency)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="directory for generated plots (chapter 7)")
    ap.add_argument("--only", choices=list(FIG_FILES), help="generate just one figure")
    ap.add_argument("--all-heatmap-patterns", action="store_true",
                    help="show every resolved pattern in heatmaps instead of representative cases")
    args = ap.parse_args()

    names = [args.only] if args.only else list(FIG_FILES)
    main_speed = sens_speed = None
    for name in names:
        out = args.out_dir / FIG_FILES[name]
        if name in MAIN_FIGS:
            if main_speed is None:
                main_speed = speedups(load_raw(args.main_csv, algorithms=["LibcMemmem"]))
            if name in {"applicability", "speedup"}:
                MAIN_FIGS[name](main_speed, out, use_cases=not args.all_heatmap_patterns)
            else:
                MAIN_FIGS[name](main_speed, out)
        else:
            if sens_speed is None:
                sens_speed = speedups(load_raw(args.sens_csv, algorithms=ALG_ORDER))
            SENS_FIGS[name](sens_speed, out)
