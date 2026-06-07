#!/usr/bin/env python3
"""Download/subset FASTA benchmark datasets."""

from __future__ import annotations

import argparse
import gzip
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FastaDownload:
    name: str
    filename: str
    url: str
    kind: str


FASTA_DOWNLOADS = [
    FastaDownload(
        "Ensembl human cDNA",
        "ensembl_human_cdna.fna",
        "https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz",
        "dna",
    ),
    FastaDownload(
        "GENCODE human transcripts",
        "gencode_human_transcripts.fna",
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/latest_release/gencode.v49.transcripts.fa.gz",
        "dna",
    ),
    FastaDownload(
        "NCBI RefSeq viral genomic",
        "refseq_viral_genomic.fna",
        "https://ftp.ncbi.nlm.nih.gov/refseq/release/viral/viral.1.1.genomic.fna.gz",
        "dna",
    ),
    FastaDownload(
        "UniProt Swiss-Prot protein",
        "uniprot_sprot.faa",
        "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz",
        "protein",
    ),
    FastaDownload(
        "UniProt TrEMBL protein",
        "uniprot_trembl.faa",
        "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz",
        "protein",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and subset FASTA benchmark datasets")
    parser.add_argument("--root", type=Path, default=Path("data/fasta"), help="FASTA output directory")
    parser.add_argument("--target-bytes", type=int, default=800_000_000)
    parser.add_argument("--dna-target-bytes", type=int, help="Override --target-bytes for .fna datasets")
    parser.add_argument("--protein-target-bytes", type=int, help="Override --target-bytes for .faa datasets")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--dna-max-records", type=int, help="Override --max-records for .fna datasets")
    parser.add_argument("--protein-max-records", type=int, help="Override --max-records for .faa datasets")
    parser.add_argument("--force", action="store_true", help="Overwrite existing FASTA files")
    return parser.parse_args()


def stream_fasta_subset(
    url: str,
    out_file: Path,
    target_bytes: int,
    max_records: int | None,
) -> tuple[int, int]:
    written = 0
    records = 0
    current_lines: list[bytes] = []
    current_size = 0

    def flush_record(force_first: bool = False) -> bool:
        nonlocal written, records, current_lines, current_size
        if not current_lines:
            return True
        if written > 0 and written + current_size > target_bytes and not force_first:
            return False
        with out_file.open("ab") as out:
            out.writelines(current_lines)
        written += current_size
        records += 1
        current_lines = []
        current_size = 0
        return True

    if out_file.exists():
        out_file.unlink()

    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            for raw_line in gz:
                if raw_line.startswith(b">"):
                    if current_lines:
                        if not flush_record(force_first=(records == 0)):
                            break
                        if max_records is not None and records >= max_records:
                            break
                    current_lines = [raw_line]
                    current_size = len(raw_line)
                elif current_lines:
                    current_lines.append(raw_line)
                    current_size += len(raw_line)

            if current_lines and (max_records is None or records < max_records):
                flush_record(force_first=(records == 0))

    return records, written


def target_bytes_for(spec: FastaDownload, args: argparse.Namespace) -> int:
    if spec.kind == "dna" and args.dna_target_bytes is not None:
        return args.dna_target_bytes
    if spec.kind == "protein" and args.protein_target_bytes is not None:
        return args.protein_target_bytes
    return args.target_bytes


def max_records_for(spec: FastaDownload, args: argparse.Namespace) -> int | None:
    if spec.kind == "dna" and args.dna_max_records is not None:
        return args.dna_max_records
    if spec.kind == "protein" and args.protein_max_records is not None:
        return args.protein_max_records
    return args.max_records


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    for spec in FASTA_DOWNLOADS:
        out_file = args.root / spec.filename
        if out_file.exists() and not args.force:
            print(f"{spec.name} already exists: {out_file}")
            continue

        print(f"Downloading/subsetting {spec.name} from {spec.url}")
        records, bytes_written = stream_fasta_subset(
            spec.url,
            out_file,
            target_bytes=target_bytes_for(spec, args),
            max_records=max_records_for(spec, args),
        )
        print(f"Wrote {spec.name}: {out_file} ({bytes_written} bytes, {records} records)")

    print("Done.")


if __name__ == "__main__":
    main()
