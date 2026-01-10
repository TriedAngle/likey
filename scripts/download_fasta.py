import os
import urllib.request
import gzip
import shutil

# Configuration
DATA_DIR = "./data/fasta"
os.makedirs(DATA_DIR, exist_ok=True)

TARGETS = {
    # 1. Protein: Human p53 tumor suppressor (UniProt)
    # Format: Protein amino acids
    "human_p53_protein.fasta": {
        "url": "https://rest.uniprot.org/uniprotkb/P04637.fasta",
        "is_gzipped": False
    },

    # 2. DNA: SARS-CoV-2 Complete Genome (NCBI RefSeq)
    # Format: Nucleotides (A, C, G, T) ~30kb
    "covid19_genome.fasta": {
        "url": "https://www.ncbi.nlm.nih.gov/sviewer/viewer.fcgi?id=NC_045512.2&db=nuccore&report=fasta&retmode=text",
        "is_gzipped": False
    },

    # 3. DNA: E. Coli K-12 Substr. MG1655 (NCBI RefSeq)
    # Format: Nucleotides, larger file (~4.6 MB)
    "ecoli_genome.fasta": {
        "url": "https://www.ncbi.nlm.nih.gov/sviewer/viewer.fcgi?id=NC_000913.3&db=nuccore&report=fasta&retmode=text",
        "is_gzipped": False
    },

    # 4. MANY ENTRIES: The entire Human Proteome (Swiss-Prot reviewed)
    # ~20,400 entries. URL returns GZIP stream.
    "human_proteome.fasta": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28proteome:UP000005640%29%20AND%20%28reviewed:true%29&compressed=true",
        "is_gzipped": True
    }
}

def download_file(filename, config):
    filepath = os.path.join(DATA_DIR, filename)
    url = config["url"]
    
    if os.path.exists(filepath):
        print(f"Skipping {filename} (already exists)")
        return

    print(f"Downloading {filename}...")
    
    try:
        with urllib.request.urlopen(url) as response:
            if config["is_gzipped"]:
                print("  -> Decompressing GZIP stream on the fly...")
                with gzip.GzipFile(fileobj=response) as uncompressed:
                    with open(filepath, 'wb') as out_file:
                        shutil.copyfileobj(uncompressed, out_file)
            else:
                print("  -> Downloading plain text...")
                with open(filepath, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                    
        print(f"  [OK] Saved to {filepath}")
            
    except Exception as e:
        print(f"  [ERROR] Failed to download {filename}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)

def main():
    print(f"--- Setting up FASTA data in {DATA_DIR} ---")
    
    for filename, config in TARGETS.items():
        download_file(filename, config)

    print("\nDone.")

if __name__ == "__main__":
    main()
