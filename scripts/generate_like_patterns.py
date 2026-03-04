#!/usr/bin/env python3
import argparse
import random
import re
from pathlib import Path


TABLE_FILES = {
    "tpch": ["part", "orders", "lineitem", "customer"],
    "tpcds": ["item", "customer", "customer_address", "date_dim", "call_center"],
    "job": ["title", "name", "movie_info", "keyword", "cast_info"],
}


def find_existing_file(data_dir: Path, base: str) -> Path | None:
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
    return None


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in re.split(r"[^A-Za-z0-9]+", text):
        tok = tok.strip()
        if len(tok) < 3:
            continue
        if tok.isdigit():
            continue
        tokens.append(tok)
    return tokens


def sample_tokens_from_file(path: Path, max_lines: int, max_tokens: int, rng: random.Random) -> list[str]:
    tokens: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for idx, line in enumerate(handle):
            if idx >= max_lines:
                break
            fields = line.rstrip("\n").split("|")
            if not fields:
                continue
            value = rng.choice(fields)
            tokens.extend(tokenize(value))
            if len(tokens) >= max_tokens:
                break
    return tokens


def complexity_score(pattern: str, weight_pct: float = 3.0, weight_us: float = 2.0, weight_len: float = 1.0) -> float:
    return (weight_pct * pattern.count("%")) + (weight_us * pattern.count("_")) + (weight_len * len(pattern))


def split_terciles(values: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    if not values:
        return [], [], []

    ranked = sorted(values, key=lambda pair: complexity_score(pair[0]))
    n = len(ranked)
    cut1 = max(1, n // 3)
    cut2 = max(cut1 + 1, (2 * n) // 3)
    low = ranked[:cut1]
    mid = ranked[cut1:cut2]
    high = ranked[cut2:]
    return low, mid, high


def stratified_pick(values: list[tuple[str, str]], max_patterns: int, rng: random.Random) -> list[tuple[str, str]]:
    if len(values) <= max_patterns:
        return values

    low, mid, high = split_terciles(values)
    bins = [low, mid, high]
    for b in bins:
        rng.shuffle(b)

    selected: list[tuple[str, str]] = []
    used = set()
    idx = 0
    while len(selected) < max_patterns and any(idx < len(b) for b in bins):
        for b in bins:
            if idx >= len(b):
                continue
            pat, desc = b[idx]
            if pat not in used:
                used.add(pat)
                selected.append((pat, desc))
                if len(selected) >= max_patterns:
                    break
        idx += 1

    return selected


def generate_patterns(tokens: list[str], max_patterns: int, rng: random.Random) -> list[tuple[str, str]]:
    uniq_tokens = []
    seen = set()
    for tok in tokens:
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq_tokens.append(tok)

    if not uniq_tokens:
        return []

    rng.shuffle(uniq_tokens)
    base_tokens = uniq_tokens[: max(20, max_patterns)]

    out: list[tuple[str, str]] = []
    used = set()

    def push(pattern: str, desc: str) -> None:
        if pattern in used:
            return
        used.add(pattern)
        out.append((pattern, desc))

    for tok in base_tokens:
        push(tok, f"exact:{tok}")
        push(f"%{tok}%", f"contains:{tok}")
        push(f"{tok}%", f"prefix:{tok}")
        push(f"%{tok}", f"suffix:{tok}")

        if len(tok) >= 5:
            mid = len(tok) // 2
            push(f"%{tok[:mid]}_{tok[mid + 1:]}%", f"underscore:{tok}")
            push(f"{tok[:mid]}%{tok[mid:]}", f"split-mid:{tok}")

        if len(tok) >= 8:
            q1 = len(tok) // 4
            q3 = (3 * len(tok)) // 4
            push(f"%{tok[:q1]}%{tok[q3:]}%", f"two-literal-short:{tok}")
            push(f"{tok[:q1]}_{tok[q1 + 1:q3]}_{tok[q3 + 1:]}", f"double-underscore:{tok}")

        if len(tok) >= 4:
            frag = tok[:2]
            tail = tok[-2:]
            push(f"%{frag}_{tail}%", f"short-gap:{tok}")

    pair_count = min(len(base_tokens) // 2, max_patterns // 6)
    for i in range(pair_count):
        a = base_tokens[i]
        b = base_tokens[-(i + 1)]
        push(f"%{a}%{b}%", f"two-literals:{a}+{b}")
        push(f"{a[:3]}%{b[-3:]}", f"bridge:{a}+{b}")
        push(f"%{a[:2]}_{b[-2:]}%", f"short-bridge:{a}+{b}")

    if base_tokens:
        for i in range(min(20, len(base_tokens))):
            tok = base_tokens[i]
            first = tok[0]
            push(f"%{first}%", f"single-char-contains:{tok}")
            push(f"_{first}_%", f"underscore-heavy:{tok}")

    selected = stratified_pick(out, max_patterns, rng)
    return selected


def load_manual_patterns(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            pattern = parts[0].strip()
            if not pattern:
                continue
            desc = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "manual"
            out.append((pattern, desc))
    return out


def write_patterns(path: Path, patterns: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# pattern\tdescription\n")
        for pattern, desc in patterns:
            handle.write(f"{pattern}\t{desc}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LIKE pattern TSV files")
    parser.add_argument("--data-root", default="data/benchmarks", help="Root benchmark directory")
    parser.add_argument("--datasets", default="tpch,tpcds", help="Comma-separated datasets")
    parser.add_argument("--max-lines-per-file", type=int, default=20000)
    parser.add_argument("--max-tokens-per-file", type=int, default=10000)
    parser.add_argument("--max-patterns", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--manual-patterns", help="Optional manual pattern TSV to merge in")
    parser.add_argument("--output", default="results/patterns_autogen.tsv", help="Output TSV file")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_root = Path(args.data_root)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    all_tokens: list[str] = []
    for dataset in datasets:
        files = TABLE_FILES.get(dataset)
        if not files:
            print(f"Skipping unknown dataset '{dataset}'")
            continue
        ds_dir = data_root / dataset
        if not ds_dir.exists():
            print(f"Skipping missing dataset dir: {ds_dir}")
            continue
        for base in files:
            p = find_existing_file(ds_dir, base)
            if p is None:
                continue
            tokens = sample_tokens_from_file(
                p,
                max_lines=args.max_lines_per_file,
                max_tokens=args.max_tokens_per_file,
                rng=rng,
            )
            all_tokens.extend(tokens)
            print(f"Scanned {p} -> {len(tokens)} tokens")

    generated = generate_patterns(all_tokens, args.max_patterns, rng)
    merged = list(generated)

    if args.manual_patterns:
        manual = load_manual_patterns(Path(args.manual_patterns))
        merged.extend(manual)
        print(f"Merged manual patterns: {len(manual)}")

    deduped: list[tuple[str, str]] = []
    seen_patterns = set()
    for pattern, desc in merged:
        if pattern in seen_patterns:
            continue
        seen_patterns.add(pattern)
        deduped.append((pattern, desc))

    out_path = Path(args.output)
    write_patterns(out_path, deduped)
    print(f"Wrote {len(deduped)} patterns to {out_path}")


if __name__ == "__main__":
    main()
