#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Callable, Iterator


JOB_SPECS: list[tuple[str, int]] = [
    ("title", 1),
    ("name", 1),
    ("movie_info", 3),
    ("keyword", 1),
    ("cast_info", 4),
]

DEFAULT_LENGTHS: list[int] = [
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
    136,
    144,
    150,
    160,
    170,
    180,
    190,
    200,
    220,
    240,
    260,
    280,
    300,
    320,
    340,
    360,
    380,
    400,
    420,
    440,
    460,
    480,
    500,
]

DEFAULT_NOMATCH_LENGTHS: list[int] = [120, 260, 400, 500]

FORBIDDEN_LITERAL_CHARS = {"%", "_", "\t", "\r", "\n"}


def parse_lengths(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("length list is empty")
    out = sorted(set(out))
    if min(out) <= 0:
        raise ValueError("all lengths must be > 0")
    return out


def find_existing_file(data_dir: Path, base: str) -> Path:
    for ext in ("csv", "tbl", "dat", "tsv"):
        p = data_dir / f"{base}.{ext}"
        if p.exists():
            return p
        nested = data_dir / "imdb" / f"{base}.{ext}"
        if nested.exists():
            return nested

    raw = data_dir / base
    if raw.exists():
        return raw
    nested_raw = data_dir / "imdb" / base
    if nested_raw.exists():
        return nested_raw

    raise FileNotFoundError(f"Missing JOB source file for {base} in {data_dir}")


def stream_job_rows(
    data_dir: Path,
    max_bytes: int | None,
    max_rows_per_table: int | None,
) -> Iterator[str]:
    csv.field_size_limit(10**9)
    remaining = max_bytes

    for base, column_index in JOB_SPECS:
        path = find_existing_file(data_dir, base)
        accepted_rows = 0

        with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter="|")
            for record in reader:
                if max_rows_per_table is not None and accepted_rows >= max_rows_per_table:
                    break

                value = record[column_index] if column_index < len(record) else ""
                value = value.strip()

                byte_len = len(value.encode("utf-8"))
                if remaining is not None and byte_len > remaining:
                    break
                if remaining is not None:
                    remaining -= byte_len

                yield value
                accepted_rows += 1

        if remaining == 0:
            break


def fasta_records(path: Path) -> Iterator[str]:
    chunks: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                if chunks:
                    yield "".join(chunks)
                    chunks.clear()
            else:
                chunks.append(line.strip())
    if chunks:
        yield "".join(chunks)


def stream_fasta_rows(
    data_dir: Path,
    max_bytes: int | None,
    max_rows_per_table: int | None,
    max_row_bytes: int | None,
) -> Iterator[str]:
    files = sorted(
        p
        for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".fasta", ".fa", ".fna", ".faa", ".fsa"}
    )
    if not files:
        raise FileNotFoundError(f"No FASTA files found in {data_dir}")

    remaining = max_bytes
    for path in files:
        kept_in_file = 0
        for row in fasta_records(path):
            if max_rows_per_table is not None and kept_in_file >= max_rows_per_table:
                break

            if max_row_bytes is not None and len(row) > max_row_bytes:
                row = row[:max_row_bytes]

            if remaining is None:
                yield row
                kept_in_file += 1
                continue

            row_len = len(row)
            if row_len <= remaining:
                yield row
                remaining -= row_len
                kept_in_file += 1
            elif remaining > 0 and kept_in_file == 0:
                yield row[:remaining]
                remaining = 0
                kept_in_file += 1

            if remaining == 0:
                break

        if remaining == 0:
            break


def is_ascii_text(value: str) -> bool:
    return all(ord(ch) < 128 for ch in value)


def is_valid_literal_segment(segment: str) -> bool:
    if not segment:
        return False
    if segment[0].isspace() or segment[-1].isspace():
        return False
    if any(ch in FORBIDDEN_LITERAL_CHARS for ch in segment):
        return False
    if not is_ascii_text(segment):
        return False
    return True


def try_extract_segment(row: str, length: int, rng: random.Random) -> str | None:
    if len(row) < length:
        return None

    max_start = len(row) - length
    starts: list[int] = [0, max_start // 2, max_start]
    if max_start > 0:
        for _ in range(256):
            starts.append(rng.randint(0, max_start))

    seen: set[int] = set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        segment = row[start : start + length]
        if is_valid_literal_segment(segment):
            return segment

    if max_start <= 50_000:
        for start in range(max_start + 1):
            if start in seen:
                continue
            segment = row[start : start + length]
            if is_valid_literal_segment(segment):
                return segment

    return None


def collect_candidate_rows(
    row_stream_factory: Callable[[], Iterator[str]],
    min_len: int,
    target_count: int = 256,
) -> list[str]:
    candidates: list[str] = []
    inspected = 0
    for row in row_stream_factory():
        inspected += 1
        if len(row) < min_len:
            continue
        probe_rng = random.Random(10_000 + inspected)
        if try_extract_segment(row, min_len, probe_rng) is None:
            continue
        candidates.append(row)
        if len(candidates) >= target_count:
            break

    if not candidates:
        raise RuntimeError(f"Could not find any row with length >= {min_len}")
    return candidates


def build_exists_literals(
    candidates: list[str],
    lengths: list[int],
    rng: random.Random,
) -> dict[int, str]:
    exists: dict[int, str] = {}
    indices = list(range(len(candidates)))

    for length in lengths:
        rng.shuffle(indices)
        chosen: str | None = None
        for idx in indices:
            chosen = try_extract_segment(candidates[idx], length, rng)
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError(f"Could not construct an exists literal of length {length}")
        exists[length] = chosen

    return exists


def mutate_literal(base: str, dataset: str, attempt: int) -> str:
    chars = list(base)
    length = len(chars)

    if dataset in {"dna", "protein"}:
        replacements = "0123456789"
        pos = length // 2
        rep = replacements[attempt % len(replacements)]
        if chars[pos] == rep:
            rep = replacements[(attempt + 1) % len(replacements)]
        chars[pos] = rep
        return "".join(chars)

    token = f"QZXJ{attempt % 10}KQ"
    start = max(0, (length // 2) - (len(token) // 2))
    for i, ch in enumerate(token):
        j = start + i
        if j >= length:
            break
        chars[j] = ch
    return "".join(chars)


def find_literals_present(
    row_stream_factory: Callable[[], Iterator[str]],
    literals: dict[int, str],
) -> set[int]:
    pending = set(literals.keys())
    if not pending:
        return set()

    for row in row_stream_factory():
        if not pending:
            break
        for key in tuple(pending):
            if literals[key] in row:
                pending.remove(key)

    return set(literals.keys()) - pending


def build_nomatch_literals(
    dataset: str,
    row_stream_factory: Callable[[], Iterator[str]],
    exists_literals: dict[int, str],
    nomatch_lengths: list[int],
) -> dict[int, str]:
    lengths = [length for length in nomatch_lengths if length in exists_literals]
    nomatch: dict[int, str] = {}

    for length in lengths:
        nomatch[length] = mutate_literal(exists_literals[length], dataset, 0)

    for attempt in range(1, 25):
        present = find_literals_present(row_stream_factory, nomatch)
        if not present:
            return nomatch

        for length in present:
            nomatch[length] = mutate_literal(exists_literals[length], dataset, attempt)

    raise RuntimeError(f"Unable to construct guaranteed no-match literals for {dataset}")


def write_pattern_file(
    path: Path,
    dataset: str,
    lengths: list[int],
    exists_literals: dict[int, str],
    nomatch_literals: dict[int, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    nomatch_ratio = len(nomatch_literals) / (len(exists_literals) + len(nomatch_literals))
    if nomatch_ratio >= 0.10:
        raise RuntimeError(
            f"No-match ratio is {nomatch_ratio:.4f} for {dataset}; must stay below 0.10"
        )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# pattern\tdescription\n")
        for length in lengths:
            literal = exists_literals[length]
            handle.write(f"%{literal}%\t{dataset}-len-exists|len={length}\n")
        for length in sorted(nomatch_literals):
            literal = nomatch_literals[length]
            handle.write(f"%{literal}%\t{dataset}-len-nomatch|len={length}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dataset-specific length-sweep LIKE pattern files (up to 500)"
    )
    parser.add_argument("--data-root", default="data/benchmarks")
    parser.add_argument("--max-bytes", type=int, default=200_000_000)
    parser.add_argument("--job-max-rows-per-table", type=int)
    parser.add_argument("--dna-max-row-bytes", type=int, default=66_262_271)
    parser.add_argument("--protein-max-row-bytes", type=int, default=500)
    parser.add_argument(
        "--lengths",
        default=",".join(str(v) for v in DEFAULT_LENGTHS),
        help="Comma-separated literal lengths",
    )
    parser.add_argument(
        "--nomatch-lengths",
        default=",".join(str(v) for v in DEFAULT_NOMATCH_LENGTHS),
        help="Comma-separated no-match literal lengths",
    )
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--job-output", default="scripts/patterns_job_len500.tsv")
    parser.add_argument("--dna-output", default="scripts/patterns_dna_len500.tsv")
    parser.add_argument("--protein-output", default="scripts/patterns_protein_len500.tsv")
    args = parser.parse_args()

    lengths = parse_lengths(args.lengths)
    nomatch_lengths = parse_lengths(args.nomatch_lengths)
    if max(nomatch_lengths) > max(lengths):
        raise ValueError("nomatch lengths must be contained in lengths")

    data_root = Path(args.data_root)
    max_len = max(lengths)

    dataset_builders: list[tuple[str, Callable[[], Iterator[str]], Path]] = [
        (
            "job",
            lambda: stream_job_rows(
                data_root / "job",
                args.max_bytes,
                args.job_max_rows_per_table,
            ),
            Path(args.job_output),
        ),
        (
            "dna",
            lambda: stream_fasta_rows(
                data_root / "dna",
                args.max_bytes,
                None,
                args.dna_max_row_bytes,
            ),
            Path(args.dna_output),
        ),
        (
            "protein",
            lambda: stream_fasta_rows(
                data_root / "protein",
                args.max_bytes,
                None,
                args.protein_max_row_bytes,
            ),
            Path(args.protein_output),
        ),
    ]

    for idx, (dataset, row_stream_factory, out_path) in enumerate(dataset_builders):
        rng = random.Random(args.seed + idx)
        candidates = collect_candidate_rows(row_stream_factory, min_len=max_len)
        exists_literals = build_exists_literals(candidates, lengths, rng)
        nomatch_literals = build_nomatch_literals(
            dataset,
            row_stream_factory,
            exists_literals,
            nomatch_lengths,
        )

        write_pattern_file(out_path, dataset, lengths, exists_literals, nomatch_literals)
        total = len(exists_literals) + len(nomatch_literals)
        nomatch_pct = (len(nomatch_literals) / total) * 100.0
        print(
            f"[{dataset}] wrote {total} patterns -> {out_path} "
            f"(exists={len(exists_literals)}, nomatch={len(nomatch_literals)}, nomatch={nomatch_pct:.2f}%)"
        )


if __name__ == "__main__":
    main()
