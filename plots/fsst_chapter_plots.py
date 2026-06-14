"""
FSST chapter plots.

Usage:
    python3 plots/fsst_chapter_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 14,
    "axes.linewidth": 1.2,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "plots" / "outputs" / "09"
GRID = "#cfcfcf"

STORAGE_ROWS = [
    ("GENCODE human transcripts", "sequence", "utf8", 50421, 93728002, 403376, 201684, 93122942, 0, 93122942),
    ("GENCODE human transcripts", "sequence", "fsst", 50421, 31679548, 403376, 201684, 31073867, 621, 93122942),
    ("quotes", "quotes.quote", "utf8", 75966, 11653513, 607736, 303864, 10741913, 0, 10741913),
    ("quotes", "quotes.quote", "fsst", 75966, 6364923, 607736, 303864, 5451109, 2214, 10741913),
    ("job_cast_info_note", "cast_info.note", "utf8", 23313199, 379758396, 186505600, 93252796, 100000000, 0, 100000000),
    ("job_cast_info_note", "cast_info.note", "fsst", 23313199, 313698020, 186505600, 93252796, 33937635, 1989, 100000000),
    ("job_keyword_keyword", "keyword.keyword", "utf8", 134170, 3677005, 1073368, 536680, 2066957, 0, 2066957),
    ("job_keyword_keyword", "keyword.keyword", "fsst", 134170, 2698852, 1073368, 536680, 1086554, 2250, 2066957),
    ("job_movie_companies_note", "movie_companies.note", "utf8", 2609129, 62759873, 20873040, 10436516, 31450317, 0, 31450317),
    ("job_movie_companies_note", "movie_companies.note", "fsst", 2609129, 41644840, 20873040, 10436516, 10333430, 1854, 31450317),
    ("job_movie_info_info", "movie_info.info", "utf8", 6674158, 180089904, 53393272, 26696632, 100000000, 0, 100000000),
    ("job_movie_info_info", "movie_info.info", "fsst", 6674158, 124659429, 53393272, 26696632, 44567590, 1935, 100000000),
    ("job_name_name", "name.name", "utf8", 4167491, 110691746, 33339936, 16669964, 60681846, 0, 60681846),
    ("job_name_name", "name.name", "fsst", 4167491, 85144728, 33339936, 16669964, 35132551, 2277, 60681846),
    ("job_title_title", "title.title", "utf8", 2528312, 71749129, 20226504, 10113248, 41409377, 0, 41409377),
    ("job_title_title", "title.title", "fsst", 2528312, 56189170, 20226504, 10113248, 25847123, 2295, 41409377),
]

RUNS = [
    (
        "DNA",
        "DNA\nsequence",
        PROJECT_ROOT
        / "benchmark_results/dna/fsst-index-memmem/gencode_20260609_061408/summary.csv",
    ),
    (
        "Quotes",
        "Quotes\nquote",
        PROJECT_ROOT
        / "benchmark_results/quotes/fsst-index-memmem/quotes_20260609_063027/summary.csv",
    ),
    (
        "cast_info.note",
        "JOB\ncast_info.note",
        PROJECT_ROOT
        / "benchmark_results/job/fsst-index-memmem/cast_info_note_20260609_065557/summary.csv",
    ),
    (
        "keyword.keyword",
        "JOB\nkeyword.keyword",
        PROJECT_ROOT
        / "benchmark_results/job/fsst-index-memmem/keyword_keyword_20260609_070744/summary.csv",
    ),
    (
        "movie_companies.note",
        "JOB\nmovie_companies.note",
        PROJECT_ROOT
        / "benchmark_results/job/fsst-index-memmem/movie_companies_note_20260609_072120/summary.csv",
    ),
    (
        "movie_info.info",
        "JOB\nmovie_info.info",
        PROJECT_ROOT
        / "benchmark_results/job/fsst-index-memmem/movie_info_info_20260609_074249/summary.csv",
    ),
    (
        "name.name",
        "JOB\nname.name",
        PROJECT_ROOT
        / "benchmark_results/job/fsst-index-memmem/name_name_20260609_080657/summary.csv",
    ),
    (
        "title.title",
        "JOB\ntitle.title",
        PROJECT_ROOT
        / "benchmark_results/job/fsst-index-memmem/title_title_20260609_082151/summary.csv",
    ),
]

STORAGE_LABELS = {
    ("GENCODE human transcripts", "sequence"): "DNA\nsequence",
    ("quotes", "quotes.quote"): "Quotes\nquote",
    ("job_cast_info_note", "cast_info.note"): "JOB\ncast_info.note",
    ("job_keyword_keyword", "keyword.keyword"): "JOB\nkeyword.keyword",
    ("job_movie_companies_note", "movie_companies.note"): "JOB\nmovie_companies.note",
    ("job_movie_info_info", "movie_info.info"): "JOB\nmovie_info.info",
    ("job_name_name", "name.name"): "JOB\nname.name",
    ("job_title_title", "title.title"): "JOB\ntitle.title",
}
ORDER_LABELS = [label for _, label, _ in RUNS]


def geomean(values) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    x = x[x > 0]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def read_timing_summary() -> pd.DataFrame:
    rows = []
    pattern_rows = []
    for run_name, label, path in RUNS:
        df = pd.read_csv(path)
        for pattern in sorted(df["pattern_name"].unique()):
            values = {"run": run_name, "label": label, "pattern_name": pattern}
            ok = True
            for storage in ["utf8", "fsst"]:
                cur = df[(df["storage"] == storage) & (df["pattern_name"] == pattern)]
                full = cur[
                    (cur["requested_index"] == "full-scan")
                    & (cur["actual_index"] == "full-scan")
                ]
                indexed = cur[cur["actual_index"] != "full-scan"]
                if full.empty or indexed.empty:
                    ok = False
                    break
                full_time = float(full["median_query_total_ns"].min())
                best = indexed.loc[indexed["median_query_total_ns"].idxmin()]
                values[f"{storage}_full_ns"] = full_time
                values[f"{storage}_best_index_ns"] = float(best["median_query_total_ns"])
                values[f"{storage}_best_index"] = best["actual_index"]
            if not ok:
                continue
            values["full_overhead"] = values["fsst_full_ns"] / values["utf8_full_ns"]
            values["indexed_overhead"] = (
                values["fsst_best_index_ns"] / values["utf8_best_index_ns"]
            )
            values["utf8_index_speedup"] = (
                values["utf8_full_ns"] / values["utf8_best_index_ns"]
            )
            values["fsst_index_speedup"] = (
                values["fsst_full_ns"] / values["fsst_best_index_ns"]
            )
            values["fsst_index_sensitivity"] = (
                values["fsst_index_speedup"] / values["utf8_index_speedup"]
            )
            pattern_rows.append(values)

        cur_patterns = pd.DataFrame([row for row in pattern_rows if row["run"] == run_name])
        rows.append({
            "run": run_name,
            "label": label,
            "patterns": len(cur_patterns),
            "utf8_full_ns": geomean(cur_patterns["utf8_full_ns"]),
            "fsst_full_ns": geomean(cur_patterns["fsst_full_ns"]),
            "full_overhead": geomean(cur_patterns["full_overhead"]),
            "indexed_overhead": geomean(cur_patterns["indexed_overhead"]),
            "utf8_index_speedup": geomean(cur_patterns["utf8_index_speedup"]),
            "fsst_index_speedup": geomean(cur_patterns["fsst_index_speedup"]),
            "fsst_index_sensitivity": geomean(cur_patterns["fsst_index_sensitivity"]),
        })

    pattern_df = pd.DataFrame(pattern_rows)
    summary = pd.DataFrame(rows)
    return summary


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overhead(summary: pd.DataFrame) -> None:
    x = np.arange(len(summary))
    width = 0.34
    fig, ax = plt.subplots(figsize=(13.6, 6.2))
    ax.bar(
        x - width / 2,
        summary["full_overhead"],
        width,
        label="Full scan",
        color="#c0392b",
    )
    ax.bar(
        x + width / 2,
        summary["indexed_overhead"],
        width,
        label="Best index",
        color="#2b5d8a",
    )
    ax.axhline(1.0, color="#333", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("FSST time / UTF-8 time (log scale)")
    ax.set_title("FSST overhead over plain UTF-8: full scan vs indexed read path")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["label"], rotation=35, ha="right")
    ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.34,
        0.96,
        "Geomean: full scan 6.73x slower; best indexed 2.71x slower",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color="#444",
    )
    fig.tight_layout()
    save_figure(fig, "fig_fsst_overhead_full_vs_indexed")


def plot_fullscan_query_time(summary: pd.DataFrame) -> None:
    x = np.arange(len(summary))
    width = 0.34
    fig, ax = plt.subplots(figsize=(13.6, 6.2))
    ax.bar(
        x - width / 2,
        summary["utf8_full_ns"] / 1e6,
        width,
        label="UTF-8 full scan",
        color="#2b5d8a",
    )
    ax.bar(
        x + width / 2,
        summary["fsst_full_ns"] / 1e6,
        width,
        label="FSST full scan",
        color="#c0392b",
    )
    ax.set_yscale("log")
    ax.set_ylabel("full-scan query time (ms, log scale)")
    ax.set_title("Plain full-scan query time: UTF-8 vs decoded FSST")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["label"], rotation=35, ha="right")
    ax.set_ylim(
        min(summary["utf8_full_ns"].min(), summary["fsst_full_ns"].min()) / 1e6 * 0.55,
        max(summary["utf8_full_ns"].max(), summary["fsst_full_ns"].max()) / 1e6 * 2.2,
    )
    ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.55, 0.99), ncol=2)
    ax.text(
        0.01,
        0.96,
        "No indexes: every FSST candidate row is decoded before verification",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color="#444",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.5},
    )
    fig.tight_layout()
    save_figure(fig, "fig_fsst_fullscan_query_time_by_column")


def plot_index_speedup(summary: pd.DataFrame) -> None:
    x = np.arange(len(summary))
    width = 0.34
    fig, ax = plt.subplots(figsize=(13.6, 6.2))
    ax.bar(
        x - width / 2,
        summary["utf8_index_speedup"],
        width,
        label="UTF-8",
        color="#2b5d8a",
    )
    ax.bar(
        x + width / 2,
        summary["fsst_index_speedup"],
        width,
        label="FSST",
        color="#e8923a",
    )
    ax.axhline(1.0, color="#333", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("Full scan / best-index time (log scale)")
    ax.set_title("Index speedup by column: pruning helps FSST more because it avoids decode")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["label"], rotation=35, ha="right")
    ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.36, 0.99), ncol=2)
    ax.text(
        0.01,
        0.96,
        "Geomean speedup: UTF-8 128.8x; FSST 319.6x",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color="#444",
    )
    fig.tight_layout()
    save_figure(fig, "fig_fsst_index_speedup_by_column")


def read_storage() -> pd.DataFrame:
    df = pd.DataFrame(
        STORAGE_ROWS,
        columns=[
            "dataset",
            "column",
            "storage",
            "row_count",
            "storage_total_bytes",
            "offsets_bytes",
            "logical_lens_bytes",
            "payload_bytes",
            "codec_bytes",
            "logical_payload_bytes",
        ],
    )
    df["label"] = [STORAGE_LABELS[(row.dataset, row.column)] for row in df.itertuples()]
    order = {label: idx for idx, label in enumerate(ORDER_LABELS)}
    df["order"] = df["label"].map(order)
    return df.sort_values(["order", "storage"])


def plot_storage_breakdown(storage: pd.DataFrame) -> None:
    components = [
        ("payload_bytes", "payload", "#2b5d8a"),
        ("offsets_bytes", "row offsets", "#8d99ae"),
        ("logical_lens_bytes", "row lengths", "#c9d1d9"),
        ("codec_bytes", "FSST codec", "#7d4e9e"),
    ]
    labels = ORDER_LABELS
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(14.2, 6.8))

    for storage_name, offset in [("utf8", -width / 2), ("fsst", width / 2)]:
        sub = storage[storage["storage"] == storage_name].set_index("label").loc[labels]
        bottom = np.zeros(len(labels))
        for field, name, color in components:
            values = sub[field].to_numpy(dtype=float) / 1e6
            bars = ax.bar(
                x + offset,
                values,
                width,
                bottom=bottom,
                label=name if storage_name == "utf8" else None,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
            bottom += values
        if storage_name == "utf8":
            max_total = bottom.copy()
        else:
            max_total = np.maximum(max_total, bottom)

    ax.set_ylabel("retained column storage (MB)")
    ax.set_title("UTF-8 vs FSST storage breakdown: payload shrinks, row metadata remains")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.tick_params(axis="x", pad=28)
    for xi in x:
        ax.text(
            xi - width / 2,
            -0.055,
            "UTF8",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="#333",
        )
        ax.text(
            xi + width / 2,
            -0.055,
            "FSST",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="#333",
        )
    ax.set_ylim(0, float(np.nanmax(max_total)) * 1.18)
    ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.text(
        0.01,
        0.96,
        "Offsets + logical lengths are unchanged; they dominate many-short-row JOB columns",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color="#444",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, "fig_fsst_storage_breakdown")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = read_timing_summary()
    storage = read_storage()
    plot_overhead(summary)
    plot_fullscan_query_time(summary)
    plot_index_speedup(summary)
    plot_storage_breakdown(storage)
    print(f"wrote FSST plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
