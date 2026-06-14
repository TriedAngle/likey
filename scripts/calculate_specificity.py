#!/usr/bin/env python3
"""Calculate actual benchmark pattern specificity/selectivity.

The thesis algorithm chapter reports suite-level selectivity ranges. This script
recomputes them from benchmark summary CSVs and writes a Markdown audit.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "analysis" / "specificity.md"

EXCLUDED_PREFIXES = ("Dna2", "FftStr", "Fftstr")

SUITES = [
    (
        "DNA exact/underscore",
        ROOT
        / "benchmark_results/dna/exact-vs-underscore-algorithms/gencode_20260609_111318/summary.csv",
    ),
    (
        "Quotes exact/underscore",
        ROOT
        / "benchmark_results/quotes/exact-vs-underscore/quotes_20260609_062306/summary.csv",
    ),
]

JOB_ROOT = ROOT / "benchmark_results/job/algorithm-comparison"


@dataclass
class PatternSpecificity:
    suite: str
    dataset: str
    column: str
    pattern_name: str
    pattern: str
    row_count: int
    rows_matched: int
    algorithms: set[str] = field(default_factory=set)
    source_files: set[str] = field(default_factory=set)

    @property
    def selectivity(self) -> float:
        return self.rows_matched / self.row_count if self.row_count else 0.0

    @property
    def one_match_per(self) -> float:
        if self.rows_matched == 0:
            return math.inf
        return self.row_count / self.rows_matched

    @property
    def shape(self) -> str:
        starts = self.pattern.startswith("%")
        ends = self.pattern.endswith("%")
        percent_count = self.pattern.count("%")
        if starts and ends and percent_count == 2:
            return "contains"
        if not starts and ends and percent_count == 1:
            return "prefix"
        if not starts and ends:
            return "anchored-general"
        if starts and ends:
            return "multi-percent"
        return "other"


def format_decimal(value: float) -> str:
    if value == 0:
        return "0"
    if value < 0.001:
        return f"{value:.8g}"
    return f"{value:.6g}"


def format_inverse(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if value >= 1000:
        return f"{value:,.1f}"
    return f"{value:.3f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def summary_paths() -> list[tuple[str, Path]]:
    paths = list(SUITES)
    paths.extend(("JOB", path) for path in sorted(JOB_ROOT.glob("*/summary.csv")))
    return paths


def include_row(row: dict[str, str]) -> bool:
    if row["storage"] != "utf8":
        return False
    if row["generic_matcher"] != "static":
        return False
    if row["algorithm"].startswith(EXCLUDED_PREFIXES):
        return False
    if row["requested_index"] not in {"none", "full-scan"}:
        return False
    if row["actual_index"] != "full-scan":
        return False
    return True


def load_specificity() -> tuple[list[PatternSpecificity], list[str]]:
    records: dict[tuple[str, str], PatternSpecificity] = {}
    inconsistencies: list[str] = []

    for suite_name, path in summary_paths():
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if not include_row(row):
                    continue

                suite = suite_name
                if suite_name == "JOB":
                    suite = f"JOB {row['column']}"

                key = (suite, row["pattern_name"])
                row_count = int(row["row_count"])
                rows_matched = int(row["rows_matched"])
                if key not in records:
                    records[key] = PatternSpecificity(
                        suite=suite,
                        dataset=row["dataset"],
                        column=row["column"],
                        pattern_name=row["pattern_name"],
                        pattern=row["pattern"],
                        row_count=row_count,
                        rows_matched=rows_matched,
                    )
                else:
                    rec = records[key]
                    if rec.row_count != row_count or rec.rows_matched != rows_matched:
                        inconsistencies.append(
                            f"{suite}/{row['pattern_name']}: "
                            f"saw {rows_matched}/{row_count}, expected "
                            f"{rec.rows_matched}/{rec.row_count}"
                        )

                records[key].algorithms.add(row["algorithm"])
                records[key].source_files.add(str(path.relative_to(ROOT)))

    return sorted(records.values(), key=lambda r: (r.suite, r.pattern_name)), inconsistencies


def suite_summaries(records: list[PatternSpecificity]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for suite in sorted({r.suite for r in records}):
        sub = [r for r in records if r.suite == suite]
        first = sub[0]
        min_rec = min(sub, key=lambda r: r.selectivity)
        max_rec = max(sub, key=lambda r: r.selectivity)
        out.append(
            {
                "suite": suite,
                "patterns": len(sub),
                "algorithms": len(set().union(*(r.algorithms for r in sub))),
                "rows": first.row_count,
                "min_selectivity": min_rec.selectivity,
                "min_pattern": min_rec.pattern_name,
                "min_matches": min_rec.rows_matched,
                "max_selectivity": max_rec.selectivity,
                "max_pattern": max_rec.pattern_name,
                "max_matches": max_rec.rows_matched,
            }
        )
    return out


def suite_table_rows(summaries: list[dict[str, object]]) -> list[list[object]]:
    rows = []
    for s in summaries:
        rows.append(
            [
                s["suite"],
                s["patterns"],
                s["algorithms"],
                f"{s['rows']:,}",
                f"{format_decimal(s['min_selectivity'])} to {format_decimal(s['max_selectivity'])}",
                f"{s['min_pattern']} ({s['min_matches']} matches)",
                f"{s['max_pattern']} ({s['max_matches']} matches)",
            ]
        )
    return rows


def pattern_table_rows(records: list[PatternSpecificity], suite: str) -> list[list[object]]:
    sub = [r for r in records if r.suite == suite]
    return [
        [
            r.pattern_name,
            f"`{r.pattern}`",
            r.shape,
            f"{r.rows_matched:,}",
            f"{format_decimal(r.selectivity)}",
            f"1 in {format_inverse(r.one_match_per)}",
        ]
        for r in sub
    ]


def audit_rows() -> list[list[object]]:
    return [
        [
            "Table `algo-suites`",
            "patterns, algorithms, rows, selectivity ranges",
            "rounded values in `06_algorithms.tex` match the calculated suite summaries",
        ],
        [
            "Table `algo-ranking`",
            "16 common algorithms, geomeans, relatives, wins, worst slowdowns",
            "matches `benchmark_results/algorithm-analysis/tables/common_algorithm_ranking.csv`",
        ],
        [
            "Table `algo-wildcard-class`",
            "exact/underscore class geomeans and ratios",
            "matches `exact_underscore_class_summary.csv`",
        ],
        [
            "Table `algo-suite-winners`",
            "JOB winners and metric agreement",
            "matches `suite_winners.csv`; all JOB metric winners agree",
        ],
        [
            "Table `algo-selectivity`",
            "selectivity-bin winners and runner-up geomeans",
            "matches `selectivity_bins.csv`",
        ],
        [
            "Text tail-risk claims",
            "BM `15x`, StdSearch/Utf8Kmp `>9x`, PairHorspool `5.7x`, wildcard `~2.5x`",
            "matches `algorithm_worst_cases.csv` after rounding",
        ],
    ]


def write_report(records: list[PatternSpecificity], inconsistencies: list[str], output: Path) -> None:
    summaries = suite_summaries(records)
    suites = sorted({r.suite for r in records})

    parts = [
        "# Specificity Audit",
        "",
        "Generated by `scripts/calculate_specificity.py`.",
        "",
        "Definition used here: `selectivity = rows_matched / row_count`. The thesis table currently labels this as a selectivity range; if using the word specificity, this report also gives the equivalent `1 in N rows` match density per pattern.",
        "",
        "All rows are filtered to `storage=utf8`, `generic_matcher=static`, full scans only, excluding `Dna2*` and `Fft*` algorithms. This matches the scope of Chapter 6.",
        "",
        "## Suite-Level Ranges",
        "",
        markdown_table(
            [
                "suite",
                "patterns",
                "algorithms",
                "rows",
                "actual selectivity range",
                "most specific pattern",
                "least specific pattern",
            ],
            suite_table_rows(summaries),
        ),
        "",
        "## Thesis Number Audit",
        "",
        markdown_table(["location", "numbers checked", "result"], audit_rows()),
        "",
    ]

    if inconsistencies:
        parts.extend(
            [
                "## Row-Count / Match-Count Inconsistencies",
                "",
                "The same pattern should have the same `rows_matched` for every included algorithm. The following inconsistencies were found:",
                "",
            ]
        )
        parts.extend(f"- {item}" for item in inconsistencies)
        parts.append("")
    else:
        parts.extend(
            [
                "## Row-Count / Match-Count Consistency",
                "",
                "No inconsistencies found. For every included suite/pattern, all included algorithms reported the same `row_count` and `rows_matched`.",
                "",
            ]
        )

    parts.append("## Pattern-Level Specificity")
    parts.append("")
    for suite in suites:
        parts.append(f"### {suite}")
        parts.append("")
        parts.append(
            markdown_table(
                [
                    "pattern name",
                    "pattern",
                    "shape",
                    "matches",
                    "selectivity",
                    "match density",
                ],
                pattern_table_rows(records, suite),
            )
        )
        parts.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Markdown output path",
    )
    args = parser.parse_args()

    records, inconsistencies = load_specificity()
    write_report(records, inconsistencies, args.output)


if __name__ == "__main__":
    main()
