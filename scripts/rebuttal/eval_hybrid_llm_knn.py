#!/usr/bin/env python3
"""Hybrid LLM+kNN CWE classification.

Stage 1: LLM binary classification (high recall for vulnerabilities)
Stage 2: kNN retrieval for specific CWE (only on predicted-vulnerable)

Combines LLM's strength in binary classification with kNN's ability
to match specific vulnerability patterns from training data.
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
    """Tokenize code for Jaccard similarity."""
    return set(re.findall(r'[a-zA-Z_]\w*', code))


class VulnKNN:
    """kNN classifier for CWE, trained on VULNERABLE samples only."""

    def __init__(self, train_path: str, max_samples: int = 0):
        self.data = []
        self.tokens = []
        self._load(train_path, max_samples)

    def _load(self, path: str, max_samples: int):
        """Load only vulnerable training samples."""
        total = 0
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                total += 1
                target = int(item.get("target", 0))
                if target == 0:
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

                if max_samples > 0 and len(self.data) >= max_samples:
                    break

        cwe_dist = Counter(d["cwe"] for d in self.data)
        print(f"  Loaded {len(self.data)} vulnerable samples (from {total} total)")
        print(f"  {len(cwe_dist)} unique CWE classes")
        print(f"  Top CWEs: {cwe_dist.most_common(10)}")

    def predict(self, code: str, k: int = 3) -> str:
        """Predict CWE using weighted kNN."""
        query_tokens = tokenize(code)

        sims = []
        for i, train_tokens in enumerate(self.tokens):
            intersection = len(query_tokens & train_tokens)
            union = len(query_tokens | train_tokens)
            sim = intersection / union if union > 0 else 0
            sims.append((i, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        top_k = sims[:k]

        votes = defaultdict(float)
        for idx, sim in top_k:
            votes[self.data[idx]["cwe"]] += sim

        return max(votes, key=votes.get) if votes else "Unknown"


def classify_sample(client, item, knn, knn_k):
    """Hybrid classification: LLM binary + kNN CWE."""
    code = item.get("func", "")
    target = int(item.get("target", 0))

    # Ground truth
    if target == 0:
        gt_cwe = "Benign"
    else:
        cwe_codes = item.get("cwe", [])
        if isinstance(cwe_codes, str):
            cwe_codes = [cwe_codes]
        gt_cwe = cwe_codes[0] if cwe_codes else "Unknown"
        if not gt_cwe.startswith("CWE-"):
            m = re.search(r"(\d+)", str(gt_cwe))
            gt_cwe = f"CWE-{m.group(1)}" if m else gt_cwe

    # Stage 1: LLM binary classification
    prompt = BINARY_PROMPT.format(code=code[:8000])
    try:
        resp = client.generate(prompt, max_tokens=150, temperature=0.0)
        is_vuln = parse_binary(resp)
    except Exception:
        is_vuln = False

    if not is_vuln:
        return {"gt_cwe": gt_cwe, "pred_cwe": "Benign"}

    # Stage 2: kNN for CWE classification
    pred_cwe = knn.predict(code, k=knn_k)
    return {"gt_cwe": gt_cwe, "pred_cwe": pred_cwe}


def parse_binary(response: str) -> bool:
    try:
        m = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            return bool(data.get("vulnerable", False))
    except:
        pass
    return "true" in response.lower()[:200]


def compute_metrics(results):
    """Compute macro-F1."""
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
        tp = class_tp[cls]
        fp = class_fp[cls]
        fn = class_fn[cls]
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
        "binary_precision": bin_prec,
        "binary_recall": bin_rec,
        "class_f1": {k: round(v, 4) for k, v in sorted(class_f1.items(), key=lambda x: x[1], reverse=True)},
        "accuracy": sum(1 for r in results if r["gt_cwe"] == r["pred_cwe"]) / len(results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--train", required=True, help="Training data for kNN")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--output", default="outputs/rebuttal/cwe130/hybrid_llm_knn_300.json")
    args = parser.parse_args()

    load_env_vars()

    with open(args.data) as f:
        test_data = [json.loads(l) for l in f]

    print(f"Test data: {len(test_data)} samples")
    print(f"Loading kNN model (vulnerable samples only)...")
    knn = VulnKNN(args.train, max_samples=args.max_train)

    print(f"\n{'='*70}")
    print(f"Hybrid LLM+kNN CWE Classification (kNN k={args.knn_k})")
    print(f"{'='*70}")
    print(f"  Model: {args.model}")
    print(f"  Workers: {args.workers}")
    print(flush=True)

    results = []
    errors = 0
    start = time.time()

    def eval_one(item):
        client = create_llm_client(model_name=args.model)
        return classify_sample(client, item, knn, args.knn_k)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(eval_one, item): i for i, item in enumerate(test_data)}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({"gt_cwe": "Unknown", "pred_cwe": "Error"})
                errors += 1

            if len(results) % 50 == 0:
                elapsed = time.time() - start
                correct = sum(1 for r in results if r["gt_cwe"] == r["pred_cwe"])
                print(f"  [{len(results)}/{len(test_data)}] {len(results)/elapsed:.1f}/s, acc={correct}/{len(results)}", flush=True)

    elapsed = time.time() - start
    metrics = compute_metrics(results)

    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"{'='*70}")
    print(f"  Macro-F1: {metrics['macro_f1']*100:.2f}%")
    print(f"  Weighted-F1: {metrics['weighted_f1']*100:.2f}%")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Binary F1: {metrics['binary_f1']*100:.2f}% (P={metrics['binary_precision']*100:.1f}%, R={metrics['binary_recall']*100:.1f}%)")
    print(f"  CWE Coverage: {metrics['coverage']}/{metrics['total_classes']}")
    print(f"\n  Top-20 class F1:")
    for cls, f1 in list(metrics['class_f1'].items())[:20]:
        print(f"    {cls:15s}: {f1:.4f}")

    # Try different knn_k values
    print(f"\n  Sensitivity to kNN k:")
    for test_k in [1, 3, 5, 7, 11]:
        temp_results = []
        for r in results:
            if r["pred_cwe"] == "Benign":
                temp_results.append(r)
            else:
                # Re-predict CWE with different k
                item_idx = results.index(r)
                code = test_data[item_idx].get("func", "") if item_idx < len(test_data) else ""
                pred = knn.predict(code, k=test_k)
                temp_results.append({"gt_cwe": r["gt_cwe"], "pred_cwe": pred})
        m = compute_metrics(temp_results)
        print(f"    k={test_k:2d}: Macro-F1={m['macro_f1']*100:.2f}%, cov={m['coverage']}/{m['total_classes']}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"method": "hybrid_llm_knn", "metrics": metrics}, f, indent=2)
    print(f"\n  Saved to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
