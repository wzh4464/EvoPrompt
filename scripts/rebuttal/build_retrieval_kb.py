#!/usr/bin/env python3
"""Build a lightweight retrieval KB for P1 single-pass + RAG experiments."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evoprompt.data.cwe_hierarchy import CWE_TO_MIDDLE, MIDDLE_TO_MAJOR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "rebuttal" / "retrieval_kb.json"
DEFAULT_TRAIN_CANDIDATES = [
    Path("/Volumes/Mac_Ext/link_cache/codes/primevul/primevul_train.jsonl"),
    PROJECT_ROOT / "data" / "primevul" / "primevul" / "primevul_train_fixed.jsonl",
    PROJECT_ROOT / "data" / "primevul" / "primevul" / "primevul_train.jsonl",
]

ROUTER_VULN_CATEGORIES = [
    "Memory",
    "Input Validation",
    "Integer",
    "Null Pointer",
    "Concurrency",
    "Authentication",
    "Cryptography",
    "Resource",
    "Logic",
]

MAJOR_TO_ROUTER = {
    "Memory": "Memory",
    "Injection": "Input Validation",
    "Input": "Input Validation",
    "Crypto": "Cryptography",
    "Logic": "Logic",
    "Benign": "Benign",
}

CWE_TO_ROUTER = {
    "Memory": {
        119, 120, 121, 122, 123, 124, 125, 126, 127, 129, 130, 131, 415, 416, 457, 665, 787, 788,
    },
    "Input Validation": {
        20, 22, 23, 36, 73, 77, 78, 79, 89, 90, 94, 95, 113, 116, 117, 134, 643, 918,
    },
    "Integer": {
        189, 190, 191, 369, 680, 681, 682, 704, 843,
    },
    "Null Pointer": {
        476, 617, 690, 824,
    },
    "Concurrency": {
        362, 367, 412, 413, 667, 833,
    },
    "Authentication": {
        269, 284, 285, 287, 306, 307, 346, 352, 862, 863,
    },
    "Cryptography": {
        254, 295, 310, 311, 312, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 347,
        354, 780, 916,
    },
    "Resource": {
        399, 400, 401, 404, 703, 770, 772, 835, 908, 909,
    },
}


def resolve_train_path(user_path: str | None) -> Path:
    """Resolve training data path from CLI or known defaults."""
    if user_path:
        p = Path(user_path).expanduser()
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            return p
        raise FileNotFoundError(f"Train data not found: {p}")

    for candidate in DEFAULT_TRAIN_CANDIDATES:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(p) for p in DEFAULT_TRAIN_CANDIDATES)
    raise FileNotFoundError(f"Could not locate train data. Tried:\n{searched}")


def extract_primary_cwe(sample: dict) -> str:
    """Extract primary CWE string from a sample."""
    cwe_field = sample.get("cwe", [])
    if isinstance(cwe_field, list):
        return str(cwe_field[0]) if cwe_field else "UNKNOWN"
    return str(cwe_field) if cwe_field else "UNKNOWN"


def extract_cwe_id(cwe: str) -> int | None:
    """Extract numeric CWE ID from string like CWE-119."""
    m = re.search(r"(\d+)", str(cwe))
    return int(m.group(1)) if m else None


def map_vuln_category(cwe_id: int | None) -> str:
    """Map CWE ID to router-aligned vulnerable category."""
    if cwe_id is None:
        return "Logic"

    for category, cwe_ids in CWE_TO_ROUTER.items():
        if cwe_id in cwe_ids:
            return category

    if cwe_id in CWE_TO_MIDDLE:
        middle = CWE_TO_MIDDLE[cwe_id]
        major = MIDDLE_TO_MAJOR.get(middle, "Logic")
        return MAJOR_TO_ROUTER.get(major, "Logic")

    return "Logic"


def short_description(sample: dict) -> str:
    """Construct a compact text description for prompt context."""
    for key in ("cve_desc", "commit_message"):
        text = (sample.get(key) or "").strip()
        if text:
            return " ".join(text.split())[:220]
    return "No description available."


def reservoir_insert(bucket: list[dict], item: dict, seen: int, limit: int) -> None:
    """Reservoir sampling update for a fixed-size bucket."""
    if len(bucket) < limit:
        bucket.append(item)
        return

    replacement = random.randint(0, seen - 1)
    if replacement < limit:
        bucket[replacement] = item


def build_kb(
    train_path: Path,
    output_path: Path,
    per_category: int,
    benign_count: int,
    max_code_chars: int,
    seed: int,
) -> dict:
    random.seed(seed)

    buckets = {cat: [] for cat in ROUTER_VULN_CATEGORIES}
    buckets["Benign"] = []
    seen_counts = defaultdict(int)

    total_lines = 0
    parsed_samples = 0

    with open(train_path) as f:
        for line in f:
            total_lines += 1
            if not line.strip():
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue

            code = (sample.get("func") or "").strip()
            if not code:
                continue

            parsed_samples += 1
            target = int(sample.get("target", 0))
            primary_cwe = extract_primary_cwe(sample)
            cwe_id = extract_cwe_id(primary_cwe)

            if target == 0:
                category = "Benign"
                limit = benign_count
            else:
                category = map_vuln_category(cwe_id)
                limit = per_category

            seen_counts[category] += 1
            entry = {
                "idx": sample.get("idx"),
                "target": target,
                "cwe": primary_cwe,
                "major_category": category,
                "description": short_description(sample),
                "code": code[:max_code_chars],
            }
            reservoir_insert(buckets[category], entry, seen_counts[category], limit)

    entries = []
    order = ROUTER_VULN_CATEGORIES + ["Benign"]
    counter = 0
    for category in order:
        for item in buckets.get(category, []):
            counter += 1
            item["id"] = f"kb_{counter:03d}"
            entries.append(item)

    kb = {
        "meta": {
            "source_train_path": str(train_path),
            "seed": seed,
            "per_vuln_category_target": per_category,
            "benign_target": benign_count,
            "max_code_chars": max_code_chars,
            "total_lines_seen": total_lines,
            "valid_samples_seen": parsed_samples,
            "seen_per_category": {k: int(v) for k, v in sorted(seen_counts.items())},
            "sampled_per_category": {k: len(v) for k, v in buckets.items()},
            "total_entries": len(entries),
        },
        "entries": entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(kb, f, indent=2)

    return kb


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval KB for P1 single-pass + RAG baseline.")
    parser.add_argument("--train-data", default=None, help="Training JSONL path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output KB JSON path.")
    parser.add_argument("--per-category", type=int, default=20, help="Samples per vulnerable category.")
    parser.add_argument("--benign-count", type=int, default=20, help="Benign samples to include.")
    parser.add_argument("--max-code-chars", type=int, default=1500, help="Max characters kept per code sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    train_path = resolve_train_path(args.train_data)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    print(f"Building retrieval KB from: {train_path}")
    kb = build_kb(
        train_path=train_path,
        output_path=output_path,
        per_category=args.per_category,
        benign_count=args.benign_count,
        max_code_chars=args.max_code_chars,
        seed=args.seed,
    )

    print("\nKB build complete:")
    print(f"  Output: {output_path}")
    print(f"  Total entries: {kb['meta']['total_entries']}")
    for cat, count in kb["meta"]["sampled_per_category"].items():
        print(f"  - {cat:16s}: {count}")


if __name__ == "__main__":
    main()
