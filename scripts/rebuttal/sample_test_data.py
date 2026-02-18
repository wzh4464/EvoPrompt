"""Stratified sampling of 150 test samples from PrimeVul test set."""
import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "primevul" / "primevul" / "primevul_test_fixed.jsonl"
OUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "rebuttal" / "sampled_150.jsonl"

# Load all test samples
samples = []
with open(DATA_PATH) as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Total test samples: {len(samples)}")

# Separate vulnerable and benign
vuln = [s for s in samples if s["target"] == 1]
benign = [s for s in samples if s["target"] == 0]
print(f"Vulnerable: {len(vuln)}, Benign: {len(benign)}")

# Stratified: keep roughly same ratio (549:1358 ≈ 0.288:0.712)
# 150 * 0.288 ≈ 43 vulnerable, 107 benign
n_vuln = 43
n_benign = 107

# For vulnerable samples: stratify by CWE to cover diversity
cwe_groups = defaultdict(list)
for s in vuln:
    cwes = s.get("cwe", [])
    key = cwes[0] if cwes else "UNKNOWN"
    cwe_groups[key].append(s)

print(f"CWE types in vulnerable set: {len(cwe_groups)}")

# Sample proportionally from each CWE group
sampled_vuln = []
cwe_counts = {k: len(v) for k, v in cwe_groups.items()}
total_vuln = sum(cwe_counts.values())

for cwe, group in sorted(cwe_groups.items(), key=lambda x: -len(x[1])):
    # Proportional allocation, at least 1
    n = max(1, round(len(group) / total_vuln * n_vuln))
    n = min(n, len(group))
    sampled_vuln.extend(random.sample(group, n))

# Trim or pad to exactly n_vuln
if len(sampled_vuln) > n_vuln:
    sampled_vuln = random.sample(sampled_vuln, n_vuln)
elif len(sampled_vuln) < n_vuln:
    remaining = [s for s in vuln if s not in sampled_vuln]
    sampled_vuln.extend(random.sample(remaining, n_vuln - len(sampled_vuln)))

# Sample benign
sampled_benign = random.sample(benign, n_benign)

sampled = sampled_vuln + sampled_benign
random.shuffle(sampled)

print(f"\nSampled: {len(sampled)} (vuln={len(sampled_vuln)}, benign={len(sampled_benign)})")

# Count CWE coverage
sampled_cwes = set()
for s in sampled_vuln:
    for c in s.get("cwe", []):
        sampled_cwes.add(c)
print(f"CWE types covered: {len(sampled_cwes)}")

# Save
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    for s in sampled:
        f.write(json.dumps(s) + "\n")

print(f"\nSaved to {OUT_PATH}")
