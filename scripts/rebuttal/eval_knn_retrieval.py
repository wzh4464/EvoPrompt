#!/usr/bin/env python3
"""kNN retrieval-based CWE classification.

Uses code similarity (Jaccard on tokens) to find the k most similar
training samples and predicts the majority CWE label.

No LLM needed - pure retrieval baseline.
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def tokenize(code: str) -> set:
    """Simple tokenization for Jaccard similarity."""
    # Split on non-alphanumeric characters
    tokens = re.findall(r'[a-zA-Z_]\w*', code)
    return set(tokens)


def jaccard_similarity(s1: set, s2: set) -> float:
    """Jaccard similarity between two token sets."""
    if not s1 or not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


def ngram_similarity(code1: str, code2: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    def get_ngrams(text, n):
        return set(text[i:i+n] for i in range(len(text) - n + 1))

    ng1 = get_ngrams(code1, n)
    ng2 = get_ngrams(code2, n)
    if not ng1 or not ng2:
        return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)


class KNNClassifier:
    """kNN-based CWE classifier using training data."""

    def __init__(self, train_path: str, max_train: int = 0, sim_method: str = "token"):
        self.train_data = []
        self.train_tokens = []
        self.sim_method = sim_method
        self._load_train(train_path, max_train)

    def _load_train(self, path: str, max_train: int):
        """Load training data."""
        print(f"Loading training data from {path}...")
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                target = int(item.get("target", 0))
                if target == 0:
                    cwe = "Benign"
                else:
                    cwe_codes = item.get("cwe", [])
                    if isinstance(cwe_codes, str):
                        cwe_codes = [cwe_codes]
                    cwe = cwe_codes[0] if cwe_codes else "Unknown"
                    if not cwe.startswith("CWE-"):
                        m = re.search(r"(\d+)", str(cwe))
                        cwe = f"CWE-{m.group(1)}" if m else cwe

                code = item.get("func", "")
                self.train_data.append({"code": code, "cwe": cwe})
                self.train_tokens.append(tokenize(code))

                if max_train > 0 and len(self.train_data) >= max_train:
                    break

        cwe_counts = Counter(d["cwe"] for d in self.train_data)
        print(f"  Loaded {len(self.train_data)} training samples")
        print(f"  {cwe_counts['Benign']} benign, {len(self.train_data) - cwe_counts['Benign']} vulnerable")
        print(f"  {len(cwe_counts)} unique CWE classes")

    def predict(self, code: str, k: int = 5) -> str:
        """Predict CWE for a code sample using kNN."""
        query_tokens = tokenize(code)

        # Compute similarity to all training samples
        if self.sim_method == "token":
            similarities = [
                (i, jaccard_similarity(query_tokens, train_tokens))
                for i, train_tokens in enumerate(self.train_tokens)
            ]
        else:  # ngram
            similarities = [
                (i, ngram_similarity(code, d["code"]))
                for i, d in enumerate(self.train_data)
            ]

        # Get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        # Majority vote (weighted by similarity)
        votes = defaultdict(float)
        for idx, sim in top_k:
            cwe = self.train_data[idx]["cwe"]
            votes[cwe] += sim

        if not votes:
            return "Unknown"

        return max(votes, key=votes.get)

    def predict_with_details(self, code: str, k: int = 5) -> dict:
        """Predict with full details."""
        query_tokens = tokenize(code)

        similarities = [
            (i, jaccard_similarity(query_tokens, train_tokens))
            for i, train_tokens in enumerate(self.train_tokens)
        ]

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        votes = defaultdict(float)
        for idx, sim in top_k:
            cwe = self.train_data[idx]["cwe"]
            votes[cwe] += sim

        pred = max(votes, key=votes.get) if votes else "Unknown"
        top_k_cwes = [self.train_data[idx]["cwe"] for idx, _ in top_k]

        return {
            "prediction": pred,
            "top_k_cwes": top_k_cwes,
            "top_k_sims": [round(sim, 4) for _, sim in top_k],
            "vote_weights": {k: round(v, 4) for k, v in sorted(votes.items(), key=lambda x: x[1], reverse=True)[:5]},
        }


def compute_metrics(results):
    """Compute macro-F1 and per-class metrics."""
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
    parser.add_argument("--data", required=True, help="Test data JSONL")
    parser.add_argument("--train", required=True, help="Training data JSONL for kNN")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors")
    parser.add_argument("--max-train", type=int, default=0, help="Max training samples")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="outputs/rebuttal/cwe130/knn_results.json")
    args = parser.parse_args()

    # Load test data
    with open(args.data) as f:
        test_data = [json.loads(l) for l in f]
    print(f"Test data: {len(test_data)} samples")

    # Build kNN classifier
    knn = KNNClassifier(args.train, max_train=args.max_train)

    print(f"\n{'='*70}")
    print(f"kNN CWE Classification (k={args.k})")
    print(f"{'='*70}")

    results = []
    start = time.time()

    for i, item in enumerate(test_data):
        code = item.get("func", "")
        target = int(item.get("target", 0))

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

        details = knn.predict_with_details(code, k=args.k)
        pred_cwe = details["prediction"]

        results.append({
            "gt_cwe": gt_cwe,
            "pred_cwe": pred_cwe,
            "top_k_cwes": details["top_k_cwes"],
            "top_k_sims": details["top_k_sims"],
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            correct = sum(1 for r in results if r["gt_cwe"] == r["pred_cwe"])
            print(f"  [{i+1}/{len(test_data)}] {(i+1)/elapsed:.1f} samples/sec, accuracy={correct}/{len(results)}", flush=True)

    elapsed = time.time() - start
    print(f"\nCompleted: {len(results)} samples in {elapsed:.0f}s ({len(results)/elapsed:.1f} samples/sec)")

    metrics = compute_metrics(results)

    print(f"\n{'='*70}")
    print(f"Results (k={args.k}):")
    print(f"{'='*70}")
    print(f"  Macro-F1 (CWE-level): {metrics['macro_f1']*100:.2f}%")
    print(f"  Weighted-F1: {metrics['weighted_f1']*100:.2f}%")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Binary F1: {metrics['binary_f1']*100:.2f}% (P={metrics['binary_precision']*100:.1f}%, R={metrics['binary_recall']*100:.1f}%)")
    print(f"  CWE Coverage: {metrics['coverage']}/{metrics['total_classes']}")

    print(f"\n  Top-20 class F1:")
    for cls, f1 in list(metrics['class_f1'].items())[:20]:
        print(f"    {cls:15s}: {f1:.4f}")

    # Try different k values
    print(f"\n  Sensitivity to k:")
    for test_k in [1, 3, 5, 7, 11, 15, 21]:
        temp_results = []
        for item, r in zip(test_data, results):
            code = item.get("func", "")
            details = knn.predict_with_details(code, k=test_k)
            temp_results.append({"gt_cwe": r["gt_cwe"], "pred_cwe": details["prediction"]})
        temp_metrics = compute_metrics(temp_results)
        print(f"    k={test_k:2d}: Macro-F1={temp_metrics['macro_f1']*100:.2f}%, Coverage={temp_metrics['coverage']}/{temp_metrics['total_classes']}, Binary-F1={temp_metrics['binary_f1']*100:.2f}%")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "method": "knn_retrieval",
        "k": args.k,
        "n_train": len(knn.train_data),
        "n_test": len(results),
        "metrics": metrics,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
