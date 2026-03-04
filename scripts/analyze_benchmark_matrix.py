#!/usr/bin/env python3
# pyright: reportGeneralTypeIssues=false
import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import umap
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pattern_kind(pattern: str) -> str:
    if "%" not in pattern and "_" not in pattern:
        return "exact"
    if pattern.startswith("%") and pattern.endswith("%") and pattern.count("%") == 2 and "_" not in pattern:
        return "contains"
    return "wildcard"


def compute_matrix(rows: list[dict[str, str]]) -> dict[str, Any]:
    filtered: list[dict[str, Any]] = []
    for r in rows:
        if r.get("skipped", "false").lower() == "true":
            continue
        enriched: dict[str, Any] = dict(r)
        enriched["duration_micros"] = float(r["duration_micros"])
        enriched["table_avg_len"] = float(r.get("table_avg_len") or 0.0)
        enriched["pattern_kind"] = pattern_kind(r["pattern"])
        filtered.append(enriched)

    algorithms = sorted({str(r["algorithm"]) for r in filtered})
    best_mean = min(
        sum(rr["duration_micros"] for rr in filtered if rr["algorithm"] == a)
        / max(1, sum(1 for rr in filtered if rr["algorithm"] == a)
        )
        for a in algorithms
    )

    overall_rows = []
    for a in algorithms:
        vals = [r["duration_micros"] for r in filtered if r["algorithm"] == a]
        vals_sorted = sorted(vals)
        p95 = vals_sorted[max(0, math.ceil(0.95 * len(vals_sorted)) - 1)] if vals_sorted else 0.0
        mean = sum(vals) / len(vals) if vals else 0.0
        overall_rows.append(
            {
                "algorithm": a,
                "count": len(vals),
                "mean_duration_micros": round(mean, 4),
                "p95_duration_micros": round(p95, 4),
                "relative_to_best_mean_pct": round((mean / best_mean - 1.0) * 100.0, 4) if best_mean else 0.0,
            }
        )
    overall_rows.sort(key=lambda r: r["mean_duration_micros"])

    dataset_rows = []
    by_dataset_algo: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in filtered:
        by_dataset_algo[(r["dataset"], r["algorithm"])].append(r["duration_micros"])
    for dataset in sorted({str(r["dataset"]) for r in filtered}):
        ranked = []
        for algo in algorithms:
            vals = by_dataset_algo[(dataset, algo)]
            if not vals:
                continue
            ranked.append((sum(vals) / len(vals), algo, len(vals)))
        ranked.sort()
        for rank, (mean, algo, count) in enumerate(ranked, start=1):
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "algorithm": algo,
                    "rank": rank,
                    "mean_duration_micros": round(mean, 4),
                    "count": count,
                }
            )

    kind_rows = []
    by_kind_algo: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in filtered:
        by_kind_algo[(r["pattern_kind"], r["algorithm"])].append(r["duration_micros"])
    for kind in ["exact", "contains", "wildcard"]:
        ranked = []
        for algo in algorithms:
            vals = by_kind_algo[(kind, algo)]
            if not vals:
                continue
            ranked.append((sum(vals) / len(vals), algo, len(vals)))
        ranked.sort()
        for rank, (mean, algo, count) in enumerate(ranked, start=1):
            kind_rows.append(
                {
                    "pattern_kind": kind,
                    "algorithm": algo,
                    "rank": rank,
                    "mean_duration_micros": round(mean, 4),
                    "count": count,
                }
            )

    vals = sorted(float(r["table_avg_len"]) for r in filtered)
    low = vals[len(vals) // 3]
    high = vals[(2 * len(vals)) // 3]

    def bucket(v: float) -> str:
        if v <= low:
            return "short"
        if v <= high:
            return "mid"
        return "long"

    recomm_rows = []
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in filtered:
        key = (r["dataset"], bucket(r["table_avg_len"]), r["pattern_kind"])
        grouped[key][r["algorithm"]].append(r["duration_micros"])

    for key, amap in sorted(grouped.items()):
        ranked = sorted((sum(v) / len(v), a, len(v)) for a, v in amap.items())
        if not ranked:
            continue
        best_mean, best_algo, sample = ranked[0]
        second_algo = ranked[1][1] if len(ranked) > 1 else ""
        second_mean = ranked[1][0] if len(ranked) > 1 else best_mean
        recomm_rows.append(
            {
                "dataset": key[0],
                "data_bucket": key[1],
                "pattern_kind": key[2],
                "best_algorithm": best_algo,
                "best_mean_micros": round(best_mean, 4),
                "second_algorithm": second_algo,
                "second_mean_micros": round(second_mean, 4),
                "best_vs_second_pct": round((second_mean / best_mean - 1.0) * 100.0, 4) if best_mean else 0.0,
                "sample_count": sample,
            }
        )

    wins_rows = []
    scenario_map: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in filtered:
        scenario_key = (r["dataset"], r["table"], r["pattern_index"], r["run"])
        scenario_map[scenario_key].append(r)
    by_dataset_wins: dict[str, Counter[str]] = defaultdict(Counter)
    for key, vals_in in scenario_map.items():
        winner = min(vals_in, key=lambda x: x["duration_micros"])["algorithm"]
        by_dataset_wins[key[0]][winner] += 1
    for ds, counter in sorted(by_dataset_wins.items()):
        total = sum(counter.values())
        for algo, wins in counter.most_common():
            wins_rows.append(
                {
                    "dataset": ds,
                    "algorithm": algo,
                    "wins": wins,
                    "win_rate": round(wins / total, 6) if total else 0.0,
                }
            )

    return {
        "overall": overall_rows,
        "dataset": dataset_rows,
        "pattern_kind": kind_rows,
        "recommendation": recomm_rows,
        "wins_by_dataset": wins_rows,
        "filtered": filtered,
    }


def _fit_umap(X: np.ndarray, dims: int):
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        n_components=dims,
    )
    return reducer.fit_transform(X)


def build_umap(filtered: list[dict[str, str]], out_dir: Path) -> None:
    algorithms = sorted({r["algorithm"] for r in filtered})
    scenarios: dict[tuple[str, str, str, str, str], dict[str, float]] = defaultdict(dict)
    meta: dict[tuple[str, str, str, str, str], dict[str, str]] = {}

    for r in filtered:
        key = (r["dataset"], r["table"], r["pattern_index"], r["pattern"], r["run"])
        scenarios[key][r["algorithm"]] = float(r["duration_micros"])
        meta[key] = {
            "dataset": r["dataset"],
            "table": r["table"],
            "pattern_index": r["pattern_index"],
            "pattern": r["pattern"],
            "run": r["run"],
            "pattern_kind": r["pattern_kind"],
            "table_avg_len": str(r.get("table_avg_len", "0")),
            "table_max_len": str(r.get("table_max_len", "0")),
            "table_rows": str(r.get("table_rows", "0")),
        }

    keys = sorted(scenarios.keys())
    X = np.zeros((len(keys), len(algorithms)), dtype=np.float64)
    X_meta = np.zeros((len(keys), 9), dtype=np.float64)
    winners = []
    for i, k in enumerate(keys):
        profile = scenarios[k]
        for j, algo in enumerate(algorithms):
            X[i, j] = math.log1p(profile.get(algo, 0.0))

        pattern = meta[k]["pattern"]
        pct_count = pattern.count("%")
        us_count = pattern.count("_")
        pattern_len = len(pattern)
        complexity = 3.0 * pct_count + 2.0 * us_count + pattern_len
        kind = meta[k]["pattern_kind"]
        X_meta[i, 0] = float(pattern_len)
        X_meta[i, 1] = float(pct_count)
        X_meta[i, 2] = float(us_count)
        X_meta[i, 3] = float(complexity)
        X_meta[i, 4] = float(meta[k]["table_avg_len"] or 0.0)
        X_meta[i, 5] = float(meta[k]["table_max_len"] or 0.0)
        X_meta[i, 6] = float(meta[k]["table_rows"] or 0.0)
        X_meta[i, 7] = 1.0 if kind == "exact" else 0.0
        X_meta[i, 8] = 1.0 if kind == "contains" else 0.0

        winners.append(min(profile.items(), key=lambda x: x[1])[0])

    Xs_profile = StandardScaler().fit_transform(X)
    Xs_meta = StandardScaler().fit_transform(X_meta)
    Xs_hybrid = np.hstack([Xs_profile, Xs_meta])
    emb = _fit_umap(Xs_profile, dims=2)
    labels = KMeans(n_clusters=6, random_state=42, n_init="auto").fit_predict(emb)
    emb_h = _fit_umap(Xs_hybrid, dims=2)
    labels_h = KMeans(n_clusters=6, random_state=42, n_init="auto").fit_predict(emb_h)

    rows = []
    for i, k in enumerate(keys):
        m = meta[k]
        rows.append(
            {
                **m,
                "winner": winners[i],
                "cluster_profile": int(labels[i]),
                "cluster_hybrid": int(labels_h[i]),
                "umap_profile_x": float(emb[i, 0]),
                "umap_profile_y": float(emb[i, 1]),
                "umap_hybrid_x": float(emb_h[i, 0]),
                "umap_hybrid_y": float(emb_h[i, 1]),
            }
        )

    write_csv(
        out_dir / "umap_scenarios.csv",
        rows,
        [
            "dataset",
            "table",
            "pattern_index",
            "pattern",
            "run",
            "pattern_kind",
            "table_avg_len",
            "table_max_len",
            "table_rows",
            "winner",
            "cluster_profile",
            "cluster_hybrid",
            "umap_profile_x",
            "umap_profile_y",
            "umap_hybrid_x",
            "umap_hybrid_y",
        ],
    )

    fig1 = go.Figure()
    winners_set = sorted({str(r["winner"]) for r in rows})
    for winner in winners_set:
        subset = [r for r in rows if str(r["winner"]) == winner]
        fig1.add_trace(
            go.Scatter(
                x=[r["umap_profile_x"] for r in subset],
                y=[r["umap_profile_y"] for r in subset],
                mode="markers",
                name=winner,
                text=[f"{r['dataset']} | {r['table']} | {r['pattern']}" for r in subset],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig1.update_layout(title="UMAP of Scenario Runtime Profiles (colored by winning algorithm)")
    fig1.write_html(str(out_dir / "umap_winner.html"), include_plotlyjs="cdn")

    fig2 = go.Figure()
    clusters_set = sorted({int(r["cluster_profile"]) for r in rows})
    for cluster in clusters_set:
        subset = [r for r in rows if int(r["cluster_profile"]) == cluster]
        fig2.add_trace(
            go.Scatter(
                x=[r["umap_profile_x"] for r in subset],
                y=[r["umap_profile_y"] for r in subset],
                mode="markers",
                name=f"cluster {cluster}",
                text=[f"{r['dataset']} | {r['table']} | {r['winner']}" for r in subset],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig2.update_layout(title="UMAP Scenario Clusters (profile-only)")
    fig2.write_html(str(out_dir / "umap_clusters.html"), include_plotlyjs="cdn")

    fig3 = go.Figure()
    clusters_h = sorted({int(r["cluster_hybrid"]) for r in rows})
    for cluster in clusters_h:
        subset = [r for r in rows if int(r["cluster_hybrid"]) == cluster]
        fig3.add_trace(
            go.Scatter(
                x=[r["umap_hybrid_x"] for r in subset],
                y=[r["umap_hybrid_y"] for r in subset],
                mode="markers",
                name=f"cluster {cluster}",
                text=[f"{r['dataset']} | {r['table']} | {r['winner']}" for r in subset],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig3.update_layout(title="UMAP Scenario Clusters (hybrid: runtimes + complexity/text features)")
    fig3.write_html(str(out_dir / "umap_clusters_hybrid.html"), include_plotlyjs="cdn")

    per_ds_dir = out_dir / "umap_by_dataset"
    per_ds_dir.mkdir(parents=True, exist_ok=True)
    marker_symbols = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "star",
    ]
    marker_symbols_3d = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "circle-open",
        "square-open",
        "diamond-open",
    ]
    cluster_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    datasets = sorted({str(r["dataset"]) for r in rows})
    for dataset in datasets:
        ds_rows = [r.copy() for r in rows if str(r["dataset"]) == dataset]
        if len(ds_rows) < 20:
            continue

        idx = np.array([i for i, r in enumerate(rows) if str(r["dataset"]) == dataset], dtype=np.int64)
        X_ds = Xs_hybrid[idx, :]
        emb2 = _fit_umap(X_ds, dims=2)
        emb3 = _fit_umap(X_ds, dims=3)
        k = min(6, max(2, len(ds_rows) // 40))
        clusters = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(emb2)

        for i, r in enumerate(ds_rows):
            r["umap2_x"] = float(emb2[i, 0])
            r["umap2_y"] = float(emb2[i, 1])
            r["umap3_x"] = float(emb3[i, 0])
            r["umap3_y"] = float(emb3[i, 1])
            r["umap3_z"] = float(emb3[i, 2])
            r["dataset_cluster"] = int(clusters[i])

        write_csv(
            per_ds_dir / f"{dataset}_umap_scenarios.csv",
            ds_rows,
            [
                "dataset",
                "table",
                "pattern_index",
                "pattern",
                "run",
                "pattern_kind",
                "table_avg_len",
                "table_max_len",
                "table_rows",
                "winner",
                "cluster_profile",
                "cluster_hybrid",
                "umap_profile_x",
                "umap_profile_y",
                "umap_hybrid_x",
                "umap_hybrid_y",
                "dataset_cluster",
                "umap2_x",
                "umap2_y",
                "umap3_x",
                "umap3_y",
                "umap3_z",
            ],
        )

        cluster_summary_rows: list[dict[str, object]] = []
        for cluster in sorted({int(r["dataset_cluster"]) for r in ds_rows}):
            subset = [r for r in ds_rows if int(r["dataset_cluster"]) == cluster]
            total = len(subset)
            winners = Counter(str(r["winner"]) for r in subset)
            kinds = Counter(str(r["pattern_kind"]) for r in subset)
            tables = Counter(str(r["table"]) for r in subset)

            top_winner, top_winner_count = winners.most_common(1)[0]
            top_kind, top_kind_count = kinds.most_common(1)[0]
            top_tables = ", ".join(f"{name}:{cnt}" for name, cnt in tables.most_common(3))

            cluster_summary_rows.append(
                {
                    "dataset": dataset,
                    "cluster": cluster,
                    "scenario_count": total,
                    "top_winner": top_winner,
                    "top_winner_rate": round(top_winner_count / total, 4),
                    "top_pattern_kind": top_kind,
                    "top_pattern_kind_rate": round(top_kind_count / total, 4),
                    "top_tables": top_tables,
                }
            )

        write_csv(
            per_ds_dir / f"{dataset}_cluster_summary.csv",
            cluster_summary_rows,
            [
                "dataset",
                "cluster",
                "scenario_count",
                "top_winner",
                "top_winner_rate",
                "top_pattern_kind",
                "top_pattern_kind_rate",
                "top_tables",
            ],
        )

        winner_to_symbol = {
            winner: marker_symbols[i % len(marker_symbols)]
            for i, winner in enumerate(sorted({str(r["winner"]) for r in ds_rows}))
        }
        winner_to_symbol_3d = {
            winner: marker_symbols_3d[i % len(marker_symbols_3d)]
            for i, winner in enumerate(sorted({str(r["winner"]) for r in ds_rows}))
        }

        fig_ds2 = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.72, 0.28],
            specs=[[{"type": "xy"}], [{"type": "table"}]],
            vertical_spacing=0.05,
            subplot_titles=(
                f"UMAP 2D Clusters ({dataset}, hybrid features)",
                "Cluster Summary",
            ),
        )

        for cluster in sorted({int(r["dataset_cluster"]) for r in ds_rows}):
            color = cluster_colors[cluster % len(cluster_colors)]
            for winner in sorted({str(r["winner"]) for r in ds_rows}):
                subset = [
                    r
                    for r in ds_rows
                    if int(r["dataset_cluster"]) == cluster and str(r["winner"]) == winner
                ]
                if not subset:
                    continue
                fig_ds2.add_trace(
                    go.Scatter(
                        x=[r["umap2_x"] for r in subset],
                        y=[r["umap2_y"] for r in subset],
                        mode="markers",
                        name=f"c{cluster} | {winner}",
                        marker={
                            "color": color,
                            "symbol": winner_to_symbol[winner],
                            "size": 7,
                            "line": {"width": 0.4, "color": "#222"},
                        },
                        text=[f"{r['table']} | {r['pattern']} | winner={r['winner']}" for r in subset],
                        hovertemplate="%{text}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        fig_ds2.add_trace(
            go.Table(
                header={
                    "values": [
                        "cluster",
                        "n",
                        "top winner",
                        "winner rate",
                        "top kind",
                        "kind rate",
                        "top tables",
                    ]
                },
                cells={
                    "values": [
                        [r["cluster"] for r in cluster_summary_rows],
                        [r["scenario_count"] for r in cluster_summary_rows],
                        [r["top_winner"] for r in cluster_summary_rows],
                        [r["top_winner_rate"] for r in cluster_summary_rows],
                        [r["top_pattern_kind"] for r in cluster_summary_rows],
                        [r["top_pattern_kind_rate"] for r in cluster_summary_rows],
                        [r["top_tables"] for r in cluster_summary_rows],
                    ]
                },
            ),
            row=2,
            col=1,
        )

        fig_ds2.update_layout(height=980)
        fig_ds2.write_html(str(per_ds_dir / f"{dataset}_umap_2d.html"), include_plotlyjs="cdn")

        fig_ds3 = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.72, 0.28],
            specs=[[{"type": "scene"}], [{"type": "table"}]],
            vertical_spacing=0.05,
            subplot_titles=(
                f"UMAP 3D Clusters ({dataset}, hybrid features)",
                "Cluster Summary",
            ),
        )

        for cluster in sorted({int(r["dataset_cluster"]) for r in ds_rows}):
            color = cluster_colors[cluster % len(cluster_colors)]
            for winner in sorted({str(r["winner"]) for r in ds_rows}):
                subset = [
                    r
                    for r in ds_rows
                    if int(r["dataset_cluster"]) == cluster and str(r["winner"]) == winner
                ]
                if not subset:
                    continue
                fig_ds3.add_trace(
                    go.Scatter3d(
                        x=[r["umap3_x"] for r in subset],
                        y=[r["umap3_y"] for r in subset],
                        z=[r["umap3_z"] for r in subset],
                        mode="markers",
                        name=f"c{cluster} | {winner}",
                        marker={
                            "color": color,
                            "symbol": winner_to_symbol_3d[winner],
                            "size": 4,
                        },
                        text=[f"{r['table']} | {r['pattern']} | winner={r['winner']}" for r in subset],
                        hovertemplate="%{text}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        fig_ds3.add_trace(
            go.Table(
                header={
                    "values": [
                        "cluster",
                        "n",
                        "top winner",
                        "winner rate",
                        "top kind",
                        "kind rate",
                        "top tables",
                    ]
                },
                cells={
                    "values": [
                        [r["cluster"] for r in cluster_summary_rows],
                        [r["scenario_count"] for r in cluster_summary_rows],
                        [r["top_winner"] for r in cluster_summary_rows],
                        [r["top_winner_rate"] for r in cluster_summary_rows],
                        [r["top_pattern_kind"] for r in cluster_summary_rows],
                        [r["top_pattern_kind_rate"] for r in cluster_summary_rows],
                        [r["top_tables"] for r in cluster_summary_rows],
                    ]
                },
            ),
            row=2,
            col=1,
        )

        fig_ds3.update_layout(height=980)
        fig_ds3.write_html(str(per_ds_dir / f"{dataset}_umap_3d.html"), include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark comparison matrix and UMAP clusters")
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    raw_path = results_dir / "raw_results.csv"
    if not raw_path.exists():
        raise SystemExit(f"Missing raw results: {raw_path}")

    rows = load_csv(raw_path)
    matrix = compute_matrix(rows)

    out_dir = results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        out_dir / "matrix_overall.csv",
        matrix["overall"],
        ["algorithm", "count", "mean_duration_micros", "p95_duration_micros", "relative_to_best_mean_pct"],
    )
    write_csv(
        out_dir / "matrix_by_dataset.csv",
        matrix["dataset"],
        ["dataset", "algorithm", "rank", "mean_duration_micros", "count"],
    )
    write_csv(
        out_dir / "matrix_by_pattern_kind.csv",
        matrix["pattern_kind"],
        ["pattern_kind", "algorithm", "rank", "mean_duration_micros", "count"],
    )
    write_csv(
        out_dir / "recommendation_matrix.csv",
        matrix["recommendation"],
        [
            "dataset",
            "data_bucket",
            "pattern_kind",
            "best_algorithm",
            "best_mean_micros",
            "second_algorithm",
            "second_mean_micros",
            "best_vs_second_pct",
            "sample_count",
        ],
    )
    write_csv(
        out_dir / "wins_by_dataset.csv",
        matrix["wins_by_dataset"],
        ["dataset", "algorithm", "wins", "win_rate"],
    )

    build_umap(matrix["filtered"], out_dir)

    print(f"Wrote analysis artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
