#!/usr/bin/env python3
"""Generate per-column dataset statistics for prepared relational benchmark data.

The script reads prepared `key,value` CSV files listed by `data/<dataset>/data.csv`.
Rows are streamed in blocks so large JOB columns do not need to fit in memory.
Most metrics are exact; distinct-value and top-value concentration metrics use
bounded-memory sketches and are reported as approximate.
"""

from __future__ import annotations

import argparse
import csv
import math
import string
import sys
import zlib
from array import array
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "tpch": [
        ("contains_requests", "%requests%"),
        ("contains_goldenrod", "%goldenrod%"),
        ("prefix_goldenrod", "goldenrod%"),
        ("suffix_lace", "%lace"),
        ("contains_powers", "%Powers%"),
        ("contains_monday", "%Monday%"),
        ("contains_unknown", "%Unknown%"),
    ],
    "tpcds": [
        ("contains_requests", "%requests%"),
        ("contains_goldenrod", "%goldenrod%"),
        ("prefix_goldenrod", "goldenrod%"),
        ("suffix_lace", "%lace"),
        ("contains_powers", "%Powers%"),
        ("contains_monday", "%Monday%"),
        ("contains_unknown", "%Unknown%"),
    ],
    "job": [
        ("contains_love", "%love%"),
        ("contains_movie", "%movie%"),
        ("prefix_the", "The%"),
        ("suffix_drama", "%drama"),
    ],
}


PRINTABLE_BYTES = set(bytes(string.printable, "ascii"))


@dataclass(frozen=True)
class ManifestColumn:
    dataset: str
    column: str
    path: Path
    value_column: str


@dataclass(frozen=True)
class PatternSpec:
    name: str
    pattern: str
    literal: bytes
    kind: str


@dataclass
class PatternStats:
    spec: PatternSpec
    matched_rows: int = 0
    offset_sum: int = 0
    offset_hist: Counter[int] = field(default_factory=Counter)


class HyperLogLog:
    """Small 64-bit HyperLogLog using two CRC32 passes for stable hashing."""

    def __init__(self, precision: int = 14) -> None:
        self.precision = precision
        self.registers = bytearray(1 << precision)

    def add(self, value: bytes) -> None:
        h1 = zlib.crc32(value) & 0xFFFFFFFF
        h2 = zlib.crc32(value, 0x9E3779B9) & 0xFFFFFFFF
        hashed = (h1 << 32) | h2
        width = 64 - self.precision
        idx = hashed >> width
        rest = hashed & ((1 << width) - 1)
        rank = self._rank(rest, width)
        if rank > self.registers[idx]:
            self.registers[idx] = rank

    def estimate(self) -> float:
        m = len(self.registers)
        alpha = 0.7213 / (1.0 + 1.079 / m)
        inv_sum = sum(2.0 ** -reg for reg in self.registers)
        estimate = alpha * m * m / inv_sum
        zeros = self.registers.count(0)
        if estimate <= 2.5 * m and zeros:
            return m * math.log(m / zeros)
        return estimate

    @staticmethod
    def _rank(value: int, width: int) -> int:
        if value == 0:
            return width + 1
        return width - value.bit_length() + 1


class SpaceSaving:
    """Bounded-memory Misra-Gries sketch for approximate top values."""

    def __init__(self, capacity: int = 2048) -> None:
        self.capacity = capacity
        self.counts: dict[bytes, int] = {}

    def add(self, value: bytes) -> None:
        if value in self.counts:
            self.counts[value] += 1
            return
        if len(self.counts) < self.capacity:
            self.counts[value] = 1
            return
        remove: list[bytes] = []
        for key in self.counts:
            self.counts[key] -= 1
            if self.counts[key] == 0:
                remove.append(key)
        for key in remove:
            del self.counts[key]

    def top(self, n: int) -> list[tuple[bytes, int]]:
        return sorted(self.counts.items(), key=lambda item: item[1], reverse=True)[:n]


@dataclass
class ColumnStats:
    dataset: str
    column: str
    path: Path
    rows: int = 0
    total_bytes: int = 0
    min_len: int | None = None
    max_len: int = 0
    positive_len_rows: int = 0
    log_len_sum: float = 0.0
    len_hist: Counter[int] = field(default_factory=Counter)
    unique_byte_hist: Counter[int] = field(default_factory=Counter)
    byte_counts: list[int] = field(default_factory=lambda: [0] * 256)
    class_counts: Counter[str] = field(default_factory=Counter)
    prefix_sets: list[set[int]] = field(default_factory=lambda: [set(), set(), set()])
    suffix_sets: list[set[int]] = field(default_factory=lambda: [set(), set(), set()])
    trigram_counts: array = field(default_factory=lambda: array("I", [0]) * (1 << 24))
    trigram_total: int = 0
    distinct_trigrams: int = 0
    top_trigram: int | None = None
    top_trigram_count: int = 0
    value_hll: HyperLogLog = field(default_factory=HyperLogLog)
    value_heavy_hitters: SpaceSaving = field(default_factory=SpaceSaving)
    pattern_stats: list[PatternStats] = field(default_factory=list)

    def add_value(self, value: bytes) -> None:
        length = len(value)
        self.rows += 1
        self.total_bytes += length
        self.min_len = length if self.min_len is None else min(self.min_len, length)
        self.max_len = max(self.max_len, length)
        if length > 0:
            self.positive_len_rows += 1
            self.log_len_sum += math.log(length)
        self.len_hist[length] += 1

        unique_bytes = set(value)
        self.unique_byte_hist[len(unique_bytes)] += 1
        for byte in value:
            self.byte_counts[byte] += 1
            self.class_counts[byte_class(byte)] += 1

        for size in (1, 2, 3):
            if length >= size:
                self.prefix_sets[size - 1].add(pack_bytes(value[:size]))
                self.suffix_sets[size - 1].add(pack_bytes(value[-size:]))

        if length >= 3:
            b0, b1 = value[0], value[1]
            for b2 in value[2:]:
                trigram = (b0 << 16) | (b1 << 8) | b2
                self.trigram_counts[trigram] += 1
                self.trigram_total += 1
                b0, b1 = b1, b2

        self.value_hll.add(value)
        self.value_heavy_hitters.add(value)

        for pattern in self.pattern_stats:
            offset = match_pattern(value, pattern.spec)
            if offset is not None:
                pattern.matched_rows += 1
                pattern.offset_sum += offset
                pattern.offset_hist[offset] += 1

    def finalize_heavy_metrics(self) -> None:
        top_count = 0
        top_trigram: int | None = None
        distinct = 0
        for trigram, count in enumerate(self.trigram_counts):
            if count:
                distinct += 1
                if count > top_count:
                    top_count = count
                    top_trigram = trigram
        self.distinct_trigrams = distinct
        self.top_trigram = top_trigram
        self.top_trigram_count = top_count
        self.trigram_counts = array("I")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DATASET_STATS.md")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("DATASET_STATS.md"))
    parser.add_argument("--datasets", default="tpch,tpcds,job")
    parser.add_argument("--block-rows", type=int, default=100_000)
    parser.add_argument(
        "--max-total-bytes",
        type=parse_bytes,
        default=None,
        help="Optional loaded value-byte cap per column, e.g. 100mb. The final row is truncated like the runner.",
    )
    parser.add_argument("--value-sketch-size", type=int, default=2048)
    parser.add_argument("--hll-precision", type=int, default=14)
    return parser.parse_args()


def parse_bytes(raw: str) -> int:
    text = raw.strip().lower().replace("_", "")
    units = [
        ("gib", 1024**3),
        ("gb", 1000**3),
        ("mib", 1024**2),
        ("mb", 1000**2),
        ("kib", 1024),
        ("kb", 1000),
        ("b", 1),
    ]
    for suffix, scale in units:
        if text.endswith(suffix):
            number = text[: -len(suffix)]
            break
    else:
        number = text
        scale = 1
    try:
        value = float(number)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid byte value: {raw!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("byte value must be greater than zero")
    return int(value * scale)


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def load_manifest(data_root: Path, dataset: str) -> list[ManifestColumn]:
    manifest_path = data_root / dataset / "data.csv"
    columns: list[ManifestColumn] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("enabled", "true").lower() != "true":
                continue
            if row.get("type") != "job-csv":
                continue
            columns.append(
                ManifestColumn(
                    dataset=dataset,
                    column=row["column"],
                    path=data_root / dataset / row["path"],
                    value_column=row.get("value_column") or "value",
                )
            )
    return columns


def compile_patterns(dataset: str) -> list[PatternStats]:
    return [PatternStats(compile_pattern(name, pattern)) for name, pattern in DEFAULT_PATTERNS[dataset]]


def compile_pattern(name: str, pattern: str) -> PatternSpec:
    if "_" in pattern or pattern.count("%") > 2:
        raise ValueError(f"unsupported default pattern shape: {pattern}")
    raw = pattern.encode("utf-8")
    starts = raw.startswith(b"%")
    ends = raw.endswith(b"%")
    literal = raw.strip(b"%")
    if starts and ends:
        kind = "contains"
    elif starts:
        kind = "suffix"
    elif ends:
        kind = "prefix"
    else:
        kind = "exact"
    return PatternSpec(name=name, pattern=pattern, literal=literal, kind=kind)


def match_pattern(value: bytes, pattern: PatternSpec) -> int | None:
    literal = pattern.literal
    if pattern.kind == "contains":
        offset = value.find(literal)
        return offset if offset >= 0 else None
    if pattern.kind == "prefix":
        return 0 if value.startswith(literal) else None
    if pattern.kind == "suffix":
        return len(value) - len(literal) if value.endswith(literal) else None
    return 0 if value == literal else None


def iter_value_blocks(path: Path, value_column: str, block_rows: int) -> tuple[int, list[bytes]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        block: list[bytes] = []
        for row in reader:
            block.append((row.get(value_column) or "").encode("utf-8"))
            if len(block) >= block_rows:
                yield len(block), block
                block = []
        if block:
            yield len(block), block


def collect_column_stats(column: ManifestColumn, args: argparse.Namespace) -> ColumnStats:
    stats = ColumnStats(dataset=column.dataset, column=column.column, path=column.path)
    stats.value_hll = HyperLogLog(args.hll_precision)
    stats.value_heavy_hitters = SpaceSaving(args.value_sketch_size)
    stats.pattern_stats = compile_patterns(column.dataset)
    loaded_bytes = 0

    for _, block in iter_value_blocks(column.path, column.value_column, args.block_rows):
        for value in block:
            if args.max_total_bytes is not None:
                if loaded_bytes >= args.max_total_bytes:
                    return stats
                remaining = args.max_total_bytes - loaded_bytes
                if len(value) > remaining:
                    value = value[:remaining]
                loaded_bytes += len(value)
            stats.add_value(value)
    return stats


def byte_class(byte: int) -> str:
    if byte in (9, 10, 11, 12, 13, 32):
        return "whitespace"
    if 65 <= byte <= 90 or 97 <= byte <= 122:
        return "letter"
    if 48 <= byte <= 57:
        return "digit"
    return "other"


def pack_bytes(value: bytes) -> int:
    out = 0
    for byte in value:
        out = (out << 8) | byte
    return out


def quantile_from_hist(hist: Counter[int], total: int, q: float) -> float:
    if total == 0:
        return 0.0
    target = (total - 1) * q
    left_rank = int(math.floor(target))
    right_rank = int(math.ceil(target))
    left = value_at_rank(hist, left_rank)
    right = value_at_rank(hist, right_rank)
    if left_rank == right_rank:
        return float(left)
    return left + (right - left) * (target - left_rank)


def value_at_rank(hist: Counter[int], rank: int) -> int:
    seen = 0
    for value in sorted(hist):
        seen += hist[value]
        if seen > rank:
            return value
    return 0


def entropy_bits(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def byte_pct(stats: ColumnStats, cls: str) -> float:
    return pct(stats.class_counts[cls], stats.total_bytes)


def pct(part: float, whole: float) -> float:
    return 0.0 if whole == 0 else part * 100.0 / whole


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def fmt_int(value: int | float) -> str:
    return f"{int(round(value)):,}"


def fmt_bytes_mb(value: int) -> str:
    return fmt_float(value / 1_000_000, 2)


def escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "\\n")


def render_byte(byte: int) -> str:
    if byte == 32:
        return "space"
    if byte == 9:
        return "tab"
    if byte in PRINTABLE_BYTES and byte not in (10, 11, 12, 13):
        return chr(byte)
    return f"0x{byte:02x}"


def render_trigram(value: int) -> str:
    raw = bytes([(value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF])
    return "".join(render_byte(b) if render_byte(b) not in {"space", "tab"} else " " for b in raw)


def geomean_length(stats: ColumnStats) -> float:
    if stats.positive_len_rows == 0:
        return 0.0
    return math.exp(stats.log_len_sum / stats.positive_len_rows)


def write_markdown(path: Path, stats: list[ColumnStats], max_total_bytes: int | None) -> None:
    lines: list[str] = []
    lines.append("# Dataset Stats")
    lines.append("")
    lines.append("Generated from prepared `data/tpch`, `data/tpcds`, and `data/job` `key,value` CSV files.")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Lengths are UTF-8 byte lengths of the prepared `value` field.")
    if max_total_bytes is not None:
        lines.append(
            f"- Each column is capped at {max_total_bytes:,} loaded value bytes; the final row is truncated to match runner `--max-total-bytes` behavior."
        )
    lines.append("- Geomean length is computed over positive-length rows; empty rows still contribute to min length and row count.")
    lines.append("- Distinct value count uses HyperLogLog; top-10 value coverage uses a bounded Misra-Gries heavy-hitter sketch and is an approximate lower bound.")
    lines.append("- Trigram metrics use byte trigrams within each row; trigrams do not cross row boundaries.")
    lines.append("- Pattern offset metrics use the first occurrence of the longest literal implied by the listed LIKE pattern.")
    lines.append("")

    render_length_table(lines, stats)
    render_complexity_table(lines, stats)
    render_pattern_table(lines, stats)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_length_table(lines: list[str], stats: list[ColumnStats]) -> None:
    lines.append("## Length Stats")
    lines.append("")
    lines.append("| Dataset | Column | Rows | Total MB | Avg | Median | Geomean | Min | Max | P90 | P99 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for stat in stats:
        avg = stat.total_bytes / stat.rows if stat.rows else 0.0
        median = quantile_from_hist(stat.len_hist, stat.rows, 0.5)
        p90 = quantile_from_hist(stat.len_hist, stat.rows, 0.9)
        p99 = quantile_from_hist(stat.len_hist, stat.rows, 0.99)
        lines.append(
            "| "
            + " | ".join(
                [
                    stat.dataset,
                    escape_md(stat.column),
                    fmt_int(stat.rows),
                    fmt_bytes_mb(stat.total_bytes),
                    fmt_float(avg),
                    fmt_float(median),
                    fmt_float(geomean_length(stat)),
                    fmt_int(stat.min_len or 0),
                    fmt_int(stat.max_len),
                    fmt_float(p90),
                    fmt_float(p99),
                ]
            )
            + " |"
        )
    lines.append("")


def render_complexity_table(lines: list[str], stats: list[ColumnStats]) -> None:
    lines.append("## Complexity Stats")
    lines.append("")
    lines.append(
        "| Dataset | Column | Alphabet | Entropy bits/B | Top byte | Top byte % | Letter % | Digit % | Whitespace % | Distinct values approx % | Top-10 values approx % | Distinct trigrams | Top trigram | Top trigram % | Avg unique bytes/row | Median unique bytes/row | Prefix3 distinct | Suffix3 distinct |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |"
    )
    for stat in stats:
        top_byte, top_byte_count = max(enumerate(stat.byte_counts), key=lambda item: item[1])
        if stat.top_trigram is not None:
            top_trigram_text = render_trigram(stat.top_trigram)
        else:
            top_trigram_text = ""
        distinct_value_estimate = min(stat.value_hll.estimate(), float(stat.rows))
        top10_estimate = min(sum(count for _, count in stat.value_heavy_hitters.top(10)), stat.rows)
        avg_unique = sum(k * v for k, v in stat.unique_byte_hist.items()) / stat.rows if stat.rows else 0.0
        median_unique = quantile_from_hist(stat.unique_byte_hist, stat.rows, 0.5)
        lines.append(
            "| "
            + " | ".join(
                [
                    stat.dataset,
                    escape_md(stat.column),
                    fmt_int(sum(1 for count in stat.byte_counts if count)),
                    fmt_float(entropy_bits(stat.byte_counts), 3),
                    escape_md(render_byte(top_byte)),
                    fmt_float(pct(top_byte_count, stat.total_bytes)),
                    fmt_float(byte_pct(stat, "letter")),
                    fmt_float(byte_pct(stat, "digit")),
                    fmt_float(byte_pct(stat, "whitespace")),
                    fmt_float(pct(distinct_value_estimate, stat.rows)),
                    fmt_float(pct(top10_estimate, stat.rows)),
                    fmt_int(stat.distinct_trigrams),
                    escape_md(top_trigram_text),
                    fmt_float(pct(stat.top_trigram_count, stat.trigram_total)),
                    fmt_float(avg_unique),
                    fmt_float(median_unique),
                    fmt_int(len(stat.prefix_sets[2])),
                    fmt_int(len(stat.suffix_sets[2])),
                ]
            )
            + " |"
        )
    lines.append("")


def render_pattern_table(lines: list[str], stats: list[ColumnStats]) -> None:
    lines.append("## Pattern Selectivity")
    lines.append("")
    lines.append("| Dataset | Column | Pattern | LIKE | Literal | Rows matched | Matched % | Avg first offset | Median first offset |")
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for stat in stats:
        for pattern in stat.pattern_stats:
            avg_offset = pattern.offset_sum / pattern.matched_rows if pattern.matched_rows else 0.0
            median_offset = quantile_from_hist(pattern.offset_hist, pattern.matched_rows, 0.5)
            lines.append(
                "| "
                + " | ".join(
                    [
                        stat.dataset,
                        escape_md(stat.column),
                        escape_md(pattern.spec.name),
                        escape_md(pattern.spec.pattern),
                        escape_md(pattern.spec.literal.decode("utf-8", errors="replace")),
                        fmt_int(pattern.matched_rows),
                        fmt_float(pct(pattern.matched_rows, stat.rows)),
                        fmt_float(avg_offset),
                        fmt_float(median_offset),
                    ]
                )
                + " |"
            )
    lines.append("")


def main() -> None:
    args = parse_args()
    if args.block_rows <= 0:
        raise SystemExit("--block-rows must be greater than zero")
    raise_csv_field_limit()

    datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    all_columns: list[ManifestColumn] = []
    for dataset in datasets:
        if dataset not in DEFAULT_PATTERNS:
            raise SystemExit(f"unsupported dataset {dataset!r}; expected one of {sorted(DEFAULT_PATTERNS)}")
        all_columns.extend(load_manifest(args.data_root, dataset))

    all_stats: list[ColumnStats] = []
    for idx, column in enumerate(all_columns, 1):
        print(f"[{idx}/{len(all_columns)}] {column.dataset}.{column.column} {column.path}", file=sys.stderr)
        stats = collect_column_stats(column, args)
        stats.finalize_heavy_metrics()
        all_stats.append(stats)

    write_markdown(args.output, all_stats, args.max_total_bytes)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
