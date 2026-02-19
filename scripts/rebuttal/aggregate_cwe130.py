#!/usr/bin/env python3
"""Aggregate CWE-130 evaluation shard results and compute Macro-F1.

Reads JSONL shard files from run_cwe130_eval.py, merges predictions,
and computes 130-class Macro-F1 using MultiClassMetrics.

Usage:
    uv run python scripts/rebuttal/aggregate_cwe130.py \
        --shards outputs/rebuttal/cwe130/shard_*.jsonl \
        --output outputs/rebuttal/cwe130/results.json
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evoprompt.evaluators.multiclass_metrics import (
    MultiClassMetrics,
    compare_averaging_methods,
)


def load_shard_results(shard_files: list) -> list:
    """Load and merge results from all shard JSONL files."""
    all_results = []
    for path in shard_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        print(f"  Loaded {path}: {sum(1 for _ in open(path) if _.strip())} records")
    return all_results


def compute_cwe130_metrics(results: list) -> dict:
    """Compute 130-class Macro-F1 and per-class metrics."""

    # Filter out errors
    valid = [r for r in results if r.get("error") is None]
    errored = [r for r in results if r.get("error") is not None]

    print(f"\n  Valid predictions: {len(valid)}")
    print(f"  Errors: {len(errored)}")

    if not valid:
        print("  ERROR: No valid predictions!")
        return {}

    # Build MultiClassMetrics
    metrics = MultiClassMetrics()
    for r in valid:
        gt = r["gt_cwe"]
        pred = r["pred_cwe"]
        metrics.add_prediction(pred, gt)

    # Compute all metrics
    macro_f1 = metrics.compute_macro_f1()
    weighted_f1 = metrics.compute_weighted_f1()
    micro_f1 = metrics.compute_micro_f1()
    accuracy = metrics.accuracy

    # Get per-class report
    report = metrics.get_classification_report()

    # Count classes
    gt_classes = set(r["gt_cwe"] for r in valid)
    pred_classes = set(r["pred_cwe"] for r in valid)
    all_classes = gt_classes | pred_classes

    # GT distribution
    gt_counter = Counter(r["gt_cwe"] for r in valid)

    # Binary metrics (vuln vs benign)
    tp = sum(1 for r in valid if r["gt_cwe"] != "Benign" and r["pred_cwe"] != "Benign")
    fp = sum(1 for r in valid if r["gt_cwe"] == "Benign" and r["pred_cwe"] != "Benign")
    fn = sum(1 for r in valid if r["gt_cwe"] != "Benign" and r["pred_cwe"] == "Benign")
    tn = sum(1 for r in valid if r["gt_cwe"] == "Benign" and r["pred_cwe"] == "Benign")

    p_v = tp / (tp + fp) if (tp + fp) > 0 else 0
    r_v = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_v = 2 * p_v * r_v / (p_v + r_v) if (p_v + r_v) > 0 else 0
    p_b = tn / (tn + fn) if (tn + fn) > 0 else 0
    r_b = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_b = 2 * p_b * r_b / (p_b + r_b) if (p_b + r_b) > 0 else 0
    binary_macro_f1 = (f1_v + f1_b) / 2

    return {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(results),
        "valid_samples": len(valid),
        "errors": len(errored),
        "num_gt_classes": len(gt_classes),
        "num_pred_classes": len(pred_classes),
        "num_all_classes": len(all_classes),
        "cwe130_macro_f1": round(macro_f1, 4),
        "cwe130_weighted_f1": round(weighted_f1, 4),
        "cwe130_micro_f1": round(micro_f1, 4),
        "cwe130_accuracy": round(accuracy, 4),
        "macro_precision": round(report["macro_avg"]["precision"], 4),
        "macro_recall": round(report["macro_avg"]["recall"], 4),
        "binary_metrics": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision_vuln": round(p_v, 4),
            "recall_vuln": round(r_v, 4),
            "f1_vuln": round(f1_v, 4),
            "precision_benign": round(p_b, 4),
            "recall_benign": round(r_b, 4),
            "f1_benign": round(f1_b, 4),
            "binary_macro_f1": round(binary_macro_f1, 4),
        },
        "gt_distribution": dict(gt_counter.most_common()),
        "per_class_metrics": report["per_class_metrics"],
        "classification_report": report,
    }


def print_summary(result: dict):
    """Print formatted summary."""
    print("\n" + "=" * 70)
    print("MulVul 130-class CWE Evaluation Results")
    print("=" * 70)

    print(f"\n  Total samples:        {result['total_samples']}")
    print(f"  Valid predictions:    {result['valid_samples']}")
    print(f"  Errors:               {result['errors']}")
    print(f"  Ground truth classes: {result['num_gt_classes']}")
    print(f"  Predicted classes:    {result['num_pred_classes']}")

    print(f"\n  --- CWE-level Multi-class Metrics ---")
    print(f"  Macro-F1:    {result['cwe130_macro_f1']:.4f} ({result['cwe130_macro_f1']*100:.2f}%)")
    print(f"  Weighted-F1: {result['cwe130_weighted_f1']:.4f}")
    print(f"  Micro-F1:    {result['cwe130_micro_f1']:.4f}")
    print(f"  Accuracy:    {result['cwe130_accuracy']:.4f}")
    print(f"  Macro-P:     {result['macro_precision']:.4f}")
    print(f"  Macro-R:     {result['macro_recall']:.4f}")

    bm = result["binary_metrics"]
    print(f"\n  --- Binary (Vuln vs Benign) ---")
    print(f"  Binary Macro-F1: {bm['binary_macro_f1']:.4f} ({bm['binary_macro_f1']*100:.2f}%)")
    print(f"  Vuln:   P={bm['precision_vuln']:.4f}  R={bm['recall_vuln']:.4f}  F1={bm['f1_vuln']:.4f}")
    print(f"  Benign: P={bm['precision_benign']:.4f}  R={bm['recall_benign']:.4f}  F1={bm['f1_benign']:.4f}")
    print(f"  TP={bm['tp']}  FP={bm['fp']}  FN={bm['fn']}  TN={bm['tn']}")

    # Top 10 CWEs by support
    print(f"\n  --- Top 15 CWEs by Support ---")
    print(f"  {'CWE':<15} {'Support':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*55}")

    per_class = result["per_class_metrics"]
    sorted_classes = sorted(
        per_class.items(),
        key=lambda x: x[1].get("support", 0),
        reverse=True,
    )
    for cls_name, cls_metrics in sorted_classes[:15]:
        print(
            f"  {cls_name:<15} "
            f"{cls_metrics['support']:>8} "
            f"{cls_metrics['precision']:>10.4f} "
            f"{cls_metrics['recall']:>10.4f} "
            f"{cls_metrics['f1_score']:>10.4f}"
        )

    print("\n" + "=" * 70)
    target = 34.79
    actual = result['cwe130_macro_f1'] * 100
    diff = actual - target
    print(f"  Target (MulVul paper):  {target:.2f}%")
    print(f"  Our result:             {actual:.2f}%")
    print(f"  Difference:             {diff:+.2f}%")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Aggregate CWE-130 shard results")
    parser.add_argument(
        "--shards", nargs="+", required=True,
        help="Shard JSONL files (supports glob patterns)"
    )
    parser.add_argument(
        "--output", default="outputs/rebuttal/cwe130/results.json",
        help="Output JSON file"
    )

    args = parser.parse_args()

    # Expand glob patterns
    shard_files = []
    for pattern in args.shards:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            shard_files.extend(expanded)
        elif os.path.exists(pattern):
            shard_files.append(pattern)
        else:
            print(f"WARNING: No files match pattern: {pattern}")

    if not shard_files:
        print("ERROR: No shard files found!")
        return 1

    print(f"Aggregating {len(shard_files)} shard files:")
    results = load_shard_results(shard_files)

    if not results:
        print("ERROR: No results loaded!")
        return 1

    print(f"\nTotal records: {len(results)}")

    # Compute metrics
    summary = compute_cwe130_metrics(results)

    # Print
    print_summary(summary)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
