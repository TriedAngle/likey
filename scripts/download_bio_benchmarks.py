#!/usr/bin/env python3
import argparse
import gzip
import urllib.request
from pathlib import Path


DNA_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz"
PROTEIN_URL = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
PROTEIN_FALLBACK_URL = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stream_fasta_subset(
    url: str,
    out_file: Path,
    target_bytes: int,
    max_records: int | None,
    append: bool = False,
) -> tuple[int, int]:
    written = out_file.stat().st_size if append and out_file.exists() else 0
    records = 0

    current_lines: list[bytes] = []
    current_size = 0

    def flush_record(force_first: bool = False) -> bool:
        nonlocal written, records, current_lines, current_size
        if not current_lines:
            return True
        if written > 0 and written + current_size > target_bytes and not force_first:
            return False
        with open(out_file, "ab") as out:
            out.writelines(current_lines)
        written += current_size
        records += 1
        current_lines = []
        current_size = 0
        return True

    if not append and out_file.exists():
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
                else:
                    if current_lines:
                        current_lines.append(raw_line)
                        current_size += len(raw_line)

            if current_lines and (max_records is None or records < max_records):
                flush_record(force_first=(records == 0))

    return records, written


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare DNA/protein benchmark FASTA datasets")
    parser.add_argument("--root", default="data/benchmarks", help="Benchmark data root")
    parser.add_argument("--dna-target-bytes", type=int, default=800_000_000)
    parser.add_argument("--protein-target-bytes", type=int, default=800_000_000)
    parser.add_argument("--dna-max-records", type=int, default=None)
    parser.add_argument("--protein-max-records", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing prepared files")
    args = parser.parse_args()

    root = Path(args.root)
    dna_dir = root / "dna"
    protein_dir = root / "protein"
    ensure_dir(dna_dir)
    ensure_dir(protein_dir)

    dna_out = dna_dir / "dna_benchmark.fna"
    protein_out = protein_dir / "protein_benchmark.faa"

    if dna_out.exists() and not args.force:
        print(f"DNA dataset already exists: {dna_out}")
    else:
        print(f"Downloading/subsetting DNA FASTA from {DNA_URL}")
        recs, bytes_written = stream_fasta_subset(
            DNA_URL,
            dna_out,
            target_bytes=args.dna_target_bytes,
            max_records=args.dna_max_records,
        )
        print(f"Wrote DNA FASTA: {dna_out} ({bytes_written} bytes, {recs} records)")

    if protein_out.exists() and not args.force:
        print(f"Protein dataset already exists: {protein_out}")
    else:
        print(f"Downloading/subsetting protein FASTA from {PROTEIN_URL}")
        recs, bytes_written = stream_fasta_subset(
            PROTEIN_URL,
            protein_out,
            target_bytes=args.protein_target_bytes,
            max_records=args.protein_max_records,
        )
        if bytes_written < int(args.protein_target_bytes * 0.8):
            print("Primary protein set smaller than target; appending from TrEMBL...")
            recs, bytes_written = stream_fasta_subset(
                PROTEIN_FALLBACK_URL,
                protein_out,
                target_bytes=args.protein_target_bytes,
                max_records=args.protein_max_records,
                append=True,
            )
        print(f"Wrote protein FASTA: {protein_out} ({bytes_written} bytes, {recs} records)")

    print("Done.")


if __name__ == "__main__":
    main()
