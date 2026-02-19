"""Stratified sampling of 200 samples with good major-category coverage."""
import json, random, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from llm_utils_v2 import sample_ground_truth

random.seed(42)
DATA = Path(__file__).resolve().parents[3] / "data/primevul/primevul/primevul_test_fixed.jsonl"
OUT = Path(__file__).resolve().parents[3] / "outputs/rebuttal/sampled_v2_200.jsonl"

samples = [json.loads(l) for l in open(DATA)]

# Label each sample
for s in samples:
    s["_gt_category"] = sample_ground_truth(s)

# Group by category
groups = defaultdict(list)
for s in samples:
    groups[s["_gt_category"]].append(s)

print("Full test set distribution:")
for cat in sorted(groups, key=lambda c: -len(groups[c])):
    print(f"  {cat}: {len(groups[cat])}")

# Stratified sample: ~200 total, proportional but min 2 per category
N = 200
total = len(samples)
sampled = []
for cat, items in sorted(groups.items(), key=lambda x: -len(x[1])):
    n = max(2, round(len(items) / total * N))
    n = min(n, len(items))
    sampled.extend(random.sample(items, n))

# Trim to N
if len(sampled) > N:
    sampled = random.sample(sampled, N)

random.shuffle(sampled)

# Report
cats = defaultdict(int)
for s in sampled:
    cats[s["_gt_category"]] += 1
print(f"\nSampled {len(sampled)} samples:")
for cat in sorted(cats, key=lambda c: -cats[c]):
    print(f"  {cat}: {cats[cat]}")

# Remove temp field and save
for s in sampled:
    del s["_gt_category"]
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    for s in sampled:
        f.write(json.dumps(s) + "\n")
print(f"\nSaved to {OUT}")
