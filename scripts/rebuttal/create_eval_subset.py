"""Create a stratified evaluation subset from PrimeVul test set for evolution.

Strategy:
- From vulnerable (549 total): take ALL samples from rare CWEs (< 10 samples),
  and sample proportionally from common CWEs. Target ~180 vulnerable covering
  all 69 unique CWEs.
- From benign (1358 total): take 120 random samples.
- Total: ~300 samples.

Usage:
    uv run python scripts/rebuttal/create_eval_subset.py
"""

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evoprompt.data.cwe_hierarchy import CWE_TO_MIDDLE, MIDDLE_TO_MAJOR

random.seed(42)

DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "primevul"
    / "primevul"
    / "primevul_test_fixed.jsonl"
)
OUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "rebuttal"
    / "cwe130"
    / "eval_subset_300.jsonl"
)

TARGET_VULN = 180
TARGET_BENIGN = 120
RARE_THRESHOLD = 10  # CWEs with fewer than this many samples are "rare"


def extract_primary_cwe(sample: dict) -> str:
    """Extract the primary CWE string from a sample."""
    cwes = sample.get("cwe", [])
    return cwes[0] if cwes else "UNKNOWN"


def extract_cwe_id(cwe_str: str) -> int | None:
    """Extract numeric CWE ID from string like 'CWE-119'."""
    m = re.search(r"CWE-(\d+)", str(cwe_str))
    return int(m.group(1)) if m else None


def get_middle_category(cwe_str: str) -> str:
    """Map a CWE string to its middle category."""
    cwe_id = extract_cwe_id(cwe_str)
    if cwe_id and cwe_id in CWE_TO_MIDDLE:
        return CWE_TO_MIDDLE[cwe_id]
    return "Other"


def get_major_category(middle: str) -> str:
    """Map a middle category to its major category."""
    return MIDDLE_TO_MAJOR.get(middle, "Logic")


def main() -> None:
    # ---- Load data ----
    samples = []
    with open(DATA_PATH) as f:
        for line in f:
            samples.append(json.loads(line))

    vuln = [s for s in samples if s["target"] == 1]
    benign = [s for s in samples if s["target"] == 0]
    print(f"Loaded {len(samples)} samples (vuln={len(vuln)}, benign={len(benign)})")

    # ---- Group vulnerable samples by primary CWE ----
    cwe_groups: dict[str, list[dict]] = defaultdict(list)
    for s in vuln:
        cwe_groups[extract_primary_cwe(s)].append(s)

    n_unique_cwes = len(cwe_groups)
    print(f"Unique CWEs in vulnerable set: {n_unique_cwes}")

    # ---- Separate rare and common CWEs ----
    rare_cwes = {cwe: grp for cwe, grp in cwe_groups.items() if len(grp) < RARE_THRESHOLD}
    common_cwes = {cwe: grp for cwe, grp in cwe_groups.items() if len(grp) >= RARE_THRESHOLD}

    rare_total = sum(len(g) for g in rare_cwes.values())
    common_total = sum(len(g) for g in common_cwes.values())
    print(
        f"Rare CWEs (< {RARE_THRESHOLD} samples): {len(rare_cwes)} CWEs, "
        f"{rare_total} samples"
    )
    print(
        f"Common CWEs (>= {RARE_THRESHOLD} samples): {len(common_cwes)} CWEs, "
        f"{common_total} samples"
    )

    # ---- Sample vulnerable: all rare + proportional from common ----
    sampled_vuln: list[dict] = []

    # Take ALL rare CWE samples (ensures CWE diversity)
    for cwe in sorted(rare_cwes.keys()):
        sampled_vuln.extend(rare_cwes[cwe])

    # Proportionally sample from common CWEs
    target_from_common = max(0, TARGET_VULN - len(sampled_vuln))
    if target_from_common > 0 and common_total > 0:
        for cwe in sorted(common_cwes.keys()):
            grp = common_cwes[cwe]
            # Proportional allocation, at least 2 per common CWE
            n = max(2, round(len(grp) / common_total * target_from_common))
            n = min(n, len(grp))
            sampled_vuln.extend(random.sample(grp, n))

    # Trim to target if we overshot
    if len(sampled_vuln) > TARGET_VULN:
        # Keep all rare samples, trim from common
        rare_samples = [s for s in sampled_vuln if extract_primary_cwe(s) in rare_cwes]
        common_samples = [s for s in sampled_vuln if extract_primary_cwe(s) in common_cwes]
        n_common_keep = TARGET_VULN - len(rare_samples)
        if n_common_keep > 0:
            # Trim common samples while preserving at least 1 per CWE
            common_by_cwe: dict[str, list[dict]] = defaultdict(list)
            for s in common_samples:
                common_by_cwe[extract_primary_cwe(s)].append(s)

            # Guarantee 1 per CWE first
            kept_common: list[dict] = []
            remainder: list[dict] = []
            for cwe, grp in common_by_cwe.items():
                kept_common.append(grp[0])
                remainder.extend(grp[1:])

            # Fill remaining budget from remainder
            budget = n_common_keep - len(kept_common)
            if budget > 0:
                kept_common.extend(random.sample(remainder, min(budget, len(remainder))))

            sampled_vuln = rare_samples + kept_common
        else:
            sampled_vuln = random.sample(rare_samples, TARGET_VULN)

    print(f"\nSampled vulnerable: {len(sampled_vuln)}")

    # ---- Sample benign ----
    sampled_benign = random.sample(benign, TARGET_BENIGN)
    print(f"Sampled benign: {len(sampled_benign)}")

    # ---- Combine and shuffle ----
    sampled = sampled_vuln + sampled_benign
    random.shuffle(sampled)
    total = len(sampled)
    print(f"Total subset: {total}")

    # ---- Statistics ----
    # CWE coverage
    sampled_cwe_set: set[str] = set()
    cwe_counter: Counter = Counter()
    middle_counter: Counter = Counter()
    major_counter: Counter = Counter()

    for s in sampled_vuln:
        cwe_str = extract_primary_cwe(s)
        sampled_cwe_set.add(cwe_str)
        cwe_counter[cwe_str] += 1
        middle = get_middle_category(cwe_str)
        middle_counter[middle] += 1
        major_counter[get_major_category(middle)] += 1

    print(f"\n{'='*60}")
    print(f"EVALUATION SUBSET STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples:      {total}")
    print(f"  Vulnerable:       {len(sampled_vuln)}")
    print(f"  Benign:           {len(sampled_benign)}")
    print(f"  Vuln ratio:       {len(sampled_vuln)/total:.1%}")
    print(f"CWE coverage:       {len(sampled_cwe_set)}/{n_unique_cwes} unique CWEs")

    print(f"\n--- Major Category Distribution ---")
    for cat, count in major_counter.most_common():
        print(f"  {cat:20s}: {count:4d}")

    print(f"\n--- Middle Category Distribution ---")
    for cat, count in middle_counter.most_common():
        print(f"  {cat:25s}: {count:4d}")

    print(f"\n--- Per-CWE Distribution (vulnerable only) ---")
    for cwe, count in cwe_counter.most_common():
        cwe_id = extract_cwe_id(cwe)
        middle = get_middle_category(cwe)
        marker = "" if count < RARE_THRESHOLD else " (common)"
        print(f"  {cwe:10s} [{middle:25s}]: {count:3d}{marker}")

    # ---- Save ----
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for s in sampled:
            f.write(json.dumps(s) + "\n")

    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
