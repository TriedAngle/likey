#!/usr/bin/env python3
"""Generate a deterministic sparse DNA `N` handling benchmark.

The generated FASTA has fixed-length rows with a mix of A/C/G/T-only rows,
sparse-N rows, an all-N row, and interleaved-N worst cases. The pattern CSV is
substring-only LIKE patterns, mostly containing `N`; several have fixed source
fragments of at least 15 bytes so the qgram index can be effective.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


BASES = "ACGT"
MOTIFS = [
    "ACGTACGTACGTACG",
    "TGCATGCATGCATGC",
    "ACGTNNNNNNNNNNNTGCA",
    "TGCANNNNNNNNNNNACGT",
    "ANANANANANANANANANA",
    "NACGNACGNACGNACG",
    "NNNNNNNNNNNNNNNACGT",
    "CGTANNNNACGTNNNNTACG",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--length", type=int, default=180)
    parser.add_argument(
        "--n-rows",
        type=int,
        default=512,
        help="Rows containing N, including the all-N and interleaved special rows",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--fasta",
        type=Path,
        default=Path("data/fasta/dna_n_sparse_180_x2048.fna"),
    )
    parser.add_argument(
        "--patterns-csv",
        type=Path,
        default=Path("benchmarks/dna/n-handling/patterns.csv"),
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("benchmarks/dna/data_n_sparse_utf8_dna2.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rows < 4:
        raise SystemExit("--rows must be at least 4")
    if args.length < 60:
        raise SystemExit("--length must be at least 60")
    if args.n_rows < 3:
        raise SystemExit("--n-rows must be at least 3")
    if args.n_rows >= args.rows:
        raise SystemExit("--n-rows must be smaller than --rows")

    rows = build_rows(args.rows, args.length, args.n_rows, args.seed)
    write_fasta(args.fasta, rows)
    write_patterns(args.patterns_csv)
    write_manifest(args.manifest_csv, args.fasta)

    print(f"wrote {len(rows)} FASTA rows to {args.fasta}")
    print(f"wrote patterns to {args.patterns_csv}")
    print(f"wrote runner data manifest to {args.manifest_csv}")
    return 0


def build_rows(rows: int, length: int, n_rows: int, seed: int) -> list[tuple[str, str]]:
    out = [
        ("no_n_repeat", repeat_to_length("ACGT", length)),
        ("all_n", "N" * length),
        ("interleaved_n", interleaved_row(length, n_first=False)),
        ("interleaved_n_inverse", interleaved_row(length, n_first=True)),
    ]

    for row_idx in range(4, n_rows + 1):
        seq = acgt_row(row_idx, length, seed)
        motif = MOTIFS[2 + (row_idx % (len(MOTIFS) - 2))]
        insert_at = 7 + ((row_idx * 37 + seed) % (length - len(motif) - 14))
        seq = seq[:insert_at] + motif + seq[insert_at + len(motif) :]
        seq = sprinkle_sparse_n(seq, row_idx, seed)
        out.append((f"sparse_n_{row_idx:05d}", seq))

    for row_idx in range(n_rows + 1, rows):
        seq = acgt_row(row_idx, length, seed)
        motif = MOTIFS[row_idx % 2]
        insert_at = 7 + ((row_idx * 37 + seed) % (length - len(motif) - 14))
        seq = seq[:insert_at] + motif + seq[insert_at + len(motif) :]
        out.append((f"no_n_{row_idx:05d}", seq))

    return out


def repeat_to_length(seed: str, length: int) -> str:
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def interleaved_row(length: int, n_first: bool) -> str:
    chars = []
    for idx in range(length):
        if (idx % 2 == 0) == n_first:
            chars.append("N")
        else:
            chars.append(BASES[idx % len(BASES)])
    return "".join(chars)


def acgt_row(row_idx: int, length: int, seed: int) -> str:
    state = (seed ^ (row_idx * 0x9E3779B1)) & 0xFFFFFFFF
    chars = []
    for pos in range(length):
        state = lcg(state)
        chars.append(BASES[(row_idx + pos + state) % len(BASES)])
    return "".join(chars)


def sprinkle_sparse_n(seq: str, row_idx: int, seed: int) -> str:
    chars = list(seq)
    state = (seed ^ (row_idx * 0x85EBCA6B)) & 0xFFFFFFFF
    for _ in range(2):
        state = lcg(state)
        start = state % len(chars)
        state = lcg(state)
        run_len = 1 + (state % 3)
        for offset in range(run_len):
            chars[(start + offset) % len(chars)] = "N"
    return "".join(chars)


def lcg(value: int) -> int:
    return (value * 1664525 + 1013904223) & 0xFFFFFFFF


def write_fasta(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        for name, seq in rows:
            handle.write(f">{name}\n")
            for start in range(0, len(seq), 60):
                handle.write(seq[start : start + 60] + "\n")


def write_patterns(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    patterns = [
        ("no_n_exact_acgt", "%ACGTACGTACGTACG%"),
        ("no_n_exact_tgca", "%TGCATGCATGCATGC%"),
        ("no_n_with_underscore", "%ACGTACGTACGTACG_%"),
        ("n_single_range_exact", "%ACGTNNNNNNNNNNNTGCA%"),
        ("n_single_range_with_underscore", "%ACGTNNNNNNNNNNNTGCA_%"),
        ("n_suffix_exact", "%TGCANNNNNNNNNNNACGT%"),
        ("n_interleaved_exact", "%ANANANANANANANANANA%"),
        ("n_interleaved_with_underscore", "%ANANANANANANANANANA_%"),
        ("n_many_short_ranges", "%NACGNACGNACGNACG%"),
        ("n_prefix_exact", "%NNNNNNNNNNNNNNNACGT%"),
        ("n_multi_region_exact", "%CGTANNNNACGTNNNNTACG%"),
        ("all_n_exact", "%NNNNNNNNNNNNNNN%"),
        ("all_n_with_underscore", "%NNNNNNNNNNNNNNN_NNNNNNNNNNNNNNN%"),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "pattern", "enabled"])
        for name, pattern in patterns:
            writer.writerow([name, pattern, "true"])


def write_manifest(path: Path, fasta: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rel_fasta = Path(os.path.relpath(fasta, path.parent))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "path", "type", "storage", "column", "enabled"])
        writer.writerow(
            [
                "DNA N-sparse 180bp reads",
                rel_fasta.as_posix(),
                "dna-fasta",
                "utf8;dna2",
                "sequence",
                "true",
            ]
        )


if __name__ == "__main__":
    raise SystemExit(main())
