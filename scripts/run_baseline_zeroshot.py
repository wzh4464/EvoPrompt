#!/usr/bin/env python3
"""GPT-4o Zero-shot Baseline for Vulnerability Detection.

Only provides CWE categories, no examples or evidence.

Usage:
    uv run python scripts/run_baseline_zeroshot.py
    uv run python scripts/run_baseline_zeroshot.py --samples 500
"""

import os
import sys
import json
import random
import time
import argparse
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "src")

from tqdm import tqdm
from evoprompt.llm.client import create_llm_client, load_env_vars

# All CWE categories from Primevul
CWE_CATEGORIES = [
    "CWE-119", "CWE-125", "CWE-20", "CWE-787", "CWE-200", "CWE-476", "CWE-416",
    "CWE-190", "CWE-399", "CWE-264", "CWE-189", "CWE-703", "CWE-401", "CWE-362",
    "CWE-772", "CWE-369", "CWE-415", "CWE-310", "CWE-284", "CWE-835", "CWE-120",
    "CWE-617", "CWE-22", "CWE-400", "CWE-78", "CWE-59", "CWE-79", "CWE-269",
    "CWE-770", "CWE-94", "CWE-667", "CWE-89", "CWE-327", "CWE-77", "Benign"
]

ZERO_SHOT_PROMPT = """You are a security expert. Analyze the following code and determine if it contains a vulnerability.

## Possible Categories:
{categories}

## Code:
```
{code}
```

## Output (JSON only):
{{"prediction": "CWE-XXX or Benign", "confidence": 0.0-1.0, "reason": "brief explanation"}}"""


def load_jsonl(path: str):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return samples


def parse_response(response: str) -> tuple:
    """Parse LLM response to extract prediction."""
    import re
    try:
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            pred = data.get("prediction", "Unknown")
            conf = float(data.get("confidence", 0.5))
            return pred, conf
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: keyword matching
    response_upper = response.upper()
    for cwe in CWE_CATEGORIES:
        if cwe.upper() in response_upper:
            return cwe, 0.5

    if "BENIGN" in response_upper or "NO VULNERABILITY" in response_upper:
        return "Benign", 0.5

    return "Unknown", 0.0


def evaluate_sample(item: dict, llm_client, categories_str: str) -> dict:
    """Evaluate a single sample."""
    code = item.get("func", "")[:4000]
    target = int(item.get("target", 0))

    # Ground truth
    if target == 0:
        gt_cwe = "Benign"
    else:
        cwe_codes = item.get("cwe", [])
        if isinstance(cwe_codes, str):
            cwe_codes = [cwe_codes] if cwe_codes else []
        gt_cwe = cwe_codes[0] if cwe_codes else "Unknown"

    # Query LLM
    prompt = ZERO_SHOT_PROMPT.format(categories=categories_str, code=code)

    try:
        response = llm_client.generate(prompt, temperature=0.1)
        pred_cwe, confidence = parse_response(response)
    except Exception as e:
        pred_cwe, confidence = "Error", 0.0

    # Binary classification
    gt_binary = "Vulnerable" if target == 1 else "Benign"
    pred_binary = "Benign" if pred_cwe == "Benign" else "Vulnerable"

    return {
        "gt_cwe": gt_cwe,
        "pred_cwe": pred_cwe,
        "gt_binary": gt_binary,
        "pred_binary": pred_binary,
        "confidence": confidence,
        "correct_binary": gt_binary == pred_binary,
        "correct_cwe": gt_cwe == pred_cwe,
    }


def main():
    parser = argparse.ArgumentParser(description="GPT-4o Zero-shot Baseline")
    parser.add_argument("--data", default="./data/primevul/primevul/primevul_valid.jsonl")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers")
    parser.add_argument("--output", default="./outputs/baseline_zeroshot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced", action="store_true", default=True)

    args = parser.parse_args()
    load_env_vars()

    print("=" * 70)
    print("GPT-4o Zero-shot Baseline")
    print("=" * 70)

    # Load data
    print(f"\n Loading: {args.data}")
    all_samples = load_jsonl(args.data)
    print(f"   Total: {len(all_samples)}")

    # Balance and sample
    random.seed(args.seed)
    if args.balanced:
        benign = [s for s in all_samples if int(s.get("target", 0)) == 0]
        vuls = [s for s in all_samples if int(s.get("target", 0)) == 1]
        n = min(len(benign), len(vuls), args.samples // 2)
        samples = random.sample(benign, n) + random.sample(vuls, n)
        random.shuffle(samples)
        print(f"   Balanced: {len(samples)} (benign:vul = 1:1)")
    else:
        samples = random.sample(all_samples, min(args.samples, len(all_samples)))
        print(f"   Sampled: {len(samples)}")

    # Create client
    llm_client = create_llm_client()
    categories_str = ", ".join(CWE_CATEGORIES)

    # Evaluate
    print(f"\n Evaluating (workers={args.workers})...")
    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate_sample, s, llm_client, categories_str): s for s in samples}

        with tqdm(total=len(samples), desc="Zero-shot", unit="sample") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    tqdm.write(f"Error: {e}")
                pbar.update(1)

    elapsed = time.time() - start_time

    # Calculate metrics
    binary_correct = sum(1 for r in results if r["correct_binary"])
    cwe_correct = sum(1 for r in results if r["correct_cwe"])

    # Binary confusion matrix
    tp = sum(1 for r in results if r["gt_binary"] == "Vulnerable" and r["pred_binary"] == "Vulnerable")
    tn = sum(1 for r in results if r["gt_binary"] == "Benign" and r["pred_binary"] == "Benign")
    fp = sum(1 for r in results if r["gt_binary"] == "Benign" and r["pred_binary"] == "Vulnerable")
    fn = sum(1 for r in results if r["gt_binary"] == "Vulnerable" and r["pred_binary"] == "Benign")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n Binary Classification:")
    print(f"   Accuracy:  {binary_correct / len(results):.2%} ({binary_correct}/{len(results)})")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall:    {recall:.2%}")
    print(f"   F1 Score:  {f1:.2%}")

    print(f"\n CWE Classification:")
    print(f"   Accuracy:  {cwe_correct / len(results):.2%} ({cwe_correct}/{len(results)})")

    print(f"\n Confusion Matrix (Binary):")
    print(f"                 Pred Vul    Pred Benign")
    print(f"   Actual Vul    {tp:>8}    {fn:>11}")
    print(f"   Actual Benign {fp:>8}    {tn:>11}")

    print(f"\n Performance:")
    print(f"   Time: {elapsed:.1f}s")
    print(f"   Throughput: {len(results) / elapsed:.1f} samples/sec")

    # Save results
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output, f"zeroshot_{timestamp}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "samples": len(results),
                "balanced": args.balanced,
                "seed": args.seed,
            },
            "metrics": {
                "binary_accuracy": binary_correct / len(results),
                "binary_precision": precision,
                "binary_recall": recall,
                "binary_f1": f1,
                "cwe_accuracy": cwe_correct / len(results),
            },
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "elapsed": elapsed,
            "sample_results": results[:100],
        }, f, indent=2, ensure_ascii=False)

    print(f"\n Results saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
