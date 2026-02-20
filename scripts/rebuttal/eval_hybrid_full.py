#!/usr/bin/env python3
"""Full-scale hybrid LLM+kNN evaluation on complete test set.

Combines LLM binary classification with kNN CWE prediction.
Supports sharding for parallel execution.
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


BINARY_PROMPT = """Analyze this C/C++ code for security vulnerabilities.

```c
{code}
```

Is this code vulnerable? Output JSON:
{{"vulnerable": true/false, "confidence": 0.0-1.0, "reason": "brief"}}"""


def tokenize(code: str) -> set:
    return set(re.findall(r'[a-zA-Z_]\w*', code))


class VulnKNN:
    """kNN on vulnerable-only training data."""

    def __init__(self, train_path: str):
        self.data = []
        self.tokens = []
        print(f"Loading kNN training data...", flush=True)
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

        print(f"  Loaded {len(self.data)} vulnerable training samples, {len(Counter(d['cwe'] for d in self.data))} CWEs", flush=True)

    def predict(self, code: str, k: int = 3) -> str:
        query_tokens = tokenize(code)
        sims = []
        for i, train_tokens in enumerate(self.tokens):
            inter = len(query_tokens & train_tokens)
            union = len(query_tokens | train_tokens)
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


def get_gt_cwe(item: dict) -> str:
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


def eval_sample(client, item, knn, knn_k):
    code = item.get("func", "")
    gt_cwe = get_gt_cwe(item)

    prompt = BINARY_PROMPT.format(code=code[:8000])
    try:
        resp = client.generate(prompt, max_tokens=150, temperature=0.0)
        is_vuln = parse_binary(resp)
    except:
        is_vuln = False

    if not is_vuln:
        return {"gt_cwe": gt_cwe, "pred_cwe": "Benign"}

    pred_cwe = knn.predict(code, k=knn_k)
    return {"gt_cwe": gt_cwe, "pred_cwe": pred_cwe}


def compute_metrics(results):
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)

    for r in results:
        gt, pred = r["gt_cwe"], r["pred_cwe"]
        if gt == pred:
            class_tp[gt] += 1
        else:
            class_fn[gt] += 1
            class_fp[pred] += 1

    gt_classes = set(r["gt_cwe"] for r in results)
    gt_counts = Counter(r["gt_cwe"] for r in results)

    class_f1 = {}
    for cls in gt_classes:
        tp, fp, fn = class_tp[cls], class_fp[cls], class_fn[cls]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        class_f1[cls] = f1

    f1s = list(class_f1.values())
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0
    total = sum(gt_counts.values())
    weighted_f1 = sum(class_f1.get(c, 0) * gt_counts[c] / total for c in gt_classes)
    coverage = sum(1 for v in class_f1.values() if v > 0)

    bin_tp = sum(1 for r in results if r["gt_cwe"] != "Benign" and r["pred_cwe"] != "Benign")
    bin_fp = sum(1 for r in results if r["gt_cwe"] == "Benign" and r["pred_cwe"] != "Benign")
    bin_fn = sum(1 for r in results if r["gt_cwe"] != "Benign" and r["pred_cwe"] == "Benign")
    bin_prec = bin_tp / (bin_tp + bin_fp) if (bin_tp + bin_fp) > 0 else 0
    bin_rec = bin_tp / (bin_tp + bin_fn) if (bin_tp + bin_fn) > 0 else 0
    bin_f1 = 2 * bin_prec * bin_rec / (bin_prec + bin_rec) if (bin_prec + bin_rec) > 0 else 0

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "coverage": coverage,
        "total_classes": len(gt_classes),
        "binary_f1": bin_f1,
        "binary_prec": bin_prec,
        "binary_rec": bin_rec,
        "accuracy": sum(1 for r in results if r["gt_cwe"] == r["pred_cwe"]) / len(results),
        "class_f1": {k: round(v, 4) for k, v in sorted(class_f1.items(), key=lambda x: x[1], reverse=True)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/rebuttal/cwe130/hybrid")
    args = parser.parse_args()

    load_env_vars()

    with open(args.data) as f:
        all_data = [json.loads(l) for l in f]

    # Shard the data
    shard_size = len(all_data) // args.num_shards
    start = args.shard_id * shard_size
    end = start + shard_size if args.shard_id < args.num_shards - 1 else len(all_data)
    data = all_data[start:end]

    print(f"{'='*70}")
    print(f"Hybrid LLM+kNN - Shard {args.shard_id}/{args.num_shards}")
    print(f"{'='*70}")
    print(f"  Data: {len(data)} samples (shard {args.shard_id}, total {len(all_data)})")
    print(f"  Model: {args.model}, kNN k={args.knn_k}")
    print(flush=True)

    knn = VulnKNN(args.train)

    results = []
    errors = 0
    start_time = time.time()

    def eval_one(item):
        client = create_llm_client(model_name=args.model)
        return eval_sample(client, item, knn, args.knn_k)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(eval_one, item): i for i, item in enumerate(data)}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:
                i = futures[future]
                gt = get_gt_cwe(data[i])
                results.append({"gt_cwe": gt, "pred_cwe": "Error"})
                errors += 1

            if len(results) % 100 == 0:
                elapsed = time.time() - start_time
                correct = sum(1 for r in results if r["gt_cwe"] == r["pred_cwe"])
                print(f"  [{len(results)}/{len(data)}] {len(results)/elapsed:.1f}/s, acc={correct}/{len(results)}, err={errors}", flush=True)

    elapsed = time.time() - start_time
    metrics = compute_metrics(results)

    print(f"\nShard {args.shard_id} completed: {len(results)} samples in {elapsed:.0f}s")
    print(f"  Macro-F1: {metrics['macro_f1']*100:.2f}%")
    print(f"  Weighted-F1: {metrics['weighted_f1']*100:.2f}%")
    print(f"  Binary F1: {metrics['binary_f1']*100:.2f}%")
    print(f"  Coverage: {metrics['coverage']}/{metrics['total_classes']}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"shard_{args.shard_id}.jsonl")
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Also save metrics
    metrics_file = os.path.join(args.output_dir, f"metrics_shard_{args.shard_id}.json")
    with open(metrics_file, "w") as f:
        json.dump({"shard": args.shard_id, "n_samples": len(results), "metrics": metrics}, f, indent=2)

    print(f"  Output: {output_file}", flush=True)


if __name__ == "__main__":
    main()
