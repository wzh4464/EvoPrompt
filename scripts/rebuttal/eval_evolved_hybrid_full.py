#!/usr/bin/env python3
"""Run the best evolved hybrid LLM+kNN on full test set.

Uses the high-recall "assume vulnerable" prompt evolved from hybrid_evolution.
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evoprompt.llm.client import create_llm_client, load_env_vars


# Best evolved prompt (from hybrid evolution gen 2, 14.25% Macro-F1, 99.4% recall)
EVOLVED_PROMPT = """Carefully analyze this C/C++ code. Most real-world code contains subtle vulnerabilities. Look for ANY security weakness, even minor ones.

```c
{code}
```

Common vulnerability patterns to check:
- Buffer overflow, out-of-bounds access
- NULL pointer dereference, use-after-free
- Integer overflow, divide by zero
- Missing error/return value checks
- Race conditions, improper locking
- Path traversal, input validation
- Information leakage

Even if the code looks mostly safe, check edge cases. Report ANY potential vulnerability.

Output JSON: {{"vulnerable": true/false, "confidence": 0.0-1.0, "reason": "brief"}}"""

EVOLVED_KNN_K = 3


def tokenize(code: str) -> set:
    return set(re.findall(r'[a-zA-Z_]\w*', code))


class VulnKNN:
    def __init__(self, train_path: str):
        self.data = []
        self.tokens = []
        print("Loading kNN training data...", flush=True)
        with open(train_path) as f:
            for line in f:
                item = json.loads(line)
                if int(item.get("target", 0)) == 0:
                    continue
                cwe_codes = item.get("cwe", [])
                if isinstance(cwe_codes, str):
                    cwe_codes = [cwe_codes]
                if not cwe_codes:
                    continue
                cwe = cwe_codes[0]
                if not cwe.startswith("CWE-"):
                    m = re.search(r"(\d+)", str(cwe))
                    cwe = f"CWE-{m.group(1)}" if m else cwe
                code = item.get("func", "")
                self.data.append({"code": code, "cwe": cwe})
                self.tokens.append(tokenize(code))
        print(f"  kNN: {len(self.data)} vuln samples, {len(Counter(d['cwe'] for d in self.data))} CWEs", flush=True)

    def predict(self, code: str, k: int = 3) -> str:
        tokens = tokenize(code)
        sims = []
        for i, tt in enumerate(self.tokens):
            inter = len(tokens & tt)
            union = len(tokens | tt)
            sims.append((i, inter / union if union > 0 else 0))
        sims.sort(key=lambda x: x[1], reverse=True)
        votes = defaultdict(float)
        for idx, sim in sims[:k]:
            votes[self.data[idx]["cwe"]] += sim
        return max(votes, key=votes.get) if votes else "Unknown"


def parse_binary(response: str) -> bool:
    try:
        m = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if m:
            return bool(json.loads(m.group(0)).get("vulnerable", False))
    except:
        pass
    return "true" in response.lower()[:200]


def get_gt_cwe(item):
    if int(item.get("target", 0)) == 0:
        return "Benign"
    cwe_codes = item.get("cwe", [])
    if isinstance(cwe_codes, str):
        cwe_codes = [cwe_codes]
    if not cwe_codes:
        return "Unknown"
    cwe = cwe_codes[0]
    if not cwe.startswith("CWE-"):
        m = re.search(r"(\d+)", str(cwe))
        cwe = f"CWE-{m.group(1)}" if m else cwe
    return cwe


def eval_sample(client, item, knn):
    code = item.get("func", "")
    gt_cwe = get_gt_cwe(item)
    prompt = EVOLVED_PROMPT.format(code=code[:8000])
    try:
        resp = client.generate(prompt, max_tokens=150, temperature=0.0)
        is_vuln = parse_binary(resp)
    except:
        is_vuln = False
    if not is_vuln:
        return {"gt_cwe": gt_cwe, "pred_cwe": "Benign"}
    pred_cwe = knn.predict(code, k=EVOLVED_KNN_K)
    return {"gt_cwe": gt_cwe, "pred_cwe": pred_cwe}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/rebuttal/cwe130/evolved_hybrid")
    args = parser.parse_args()

    load_env_vars()

    with open(args.data) as f:
        all_data = [json.loads(l) for l in f]

    shard_size = len(all_data) // args.num_shards
    start = args.shard_id * shard_size
    end = start + shard_size if args.shard_id < args.num_shards - 1 else len(all_data)
    data = all_data[start:end]

    print(f"{'='*70}")
    print(f"Evolved Hybrid LLM+kNN - Shard {args.shard_id}/{args.num_shards}")
    print(f"{'='*70}")
    print(f"  Data: {len(data)} samples (shard {args.shard_id})")
    print(f"  Model: {args.model}, kNN k={EVOLVED_KNN_K}")
    print(flush=True)

    knn = VulnKNN(args.train)

    results = []
    errors = 0
    start_time = time.time()

    def eval_one(item):
        client = create_llm_client(model_name=args.model)
        return eval_sample(client, item, knn)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(eval_one, item): i for i, item in enumerate(data)}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except:
                i = futures[future]
                results.append({"gt_cwe": get_gt_cwe(data[i]), "pred_cwe": "Error"})
                errors += 1
            if len(results) % 100 == 0:
                elapsed = time.time() - start_time
                correct = sum(1 for r in results if r["gt_cwe"] == r["pred_cwe"])
                print(f"  [{len(results)}/{len(data)}] {len(results)/elapsed:.1f}/s, acc={correct}/{len(results)}, err={errors}", flush=True)

    elapsed = time.time() - start_time
    print(f"\nShard {args.shard_id}: {len(results)} samples in {elapsed:.0f}s", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"shard_{args.shard_id}.jsonl")
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Output: {output_file}", flush=True)


if __name__ == "__main__":
    main()
