#!/usr/bin/env python3
"""Direct CWE classification - single prompt approach.

Instead of Router → Detector → Aggregator, uses a single comprehensive
prompt that classifies code directly into one of 130 CWE classes or Benign.

This is a fundamentally different approach from MulVul's multi-agent pipeline.
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
from evoprompt.data.cwe_hierarchy import CWE_TO_MIDDLE, MIDDLE_TO_MAJOR

# All CWE types found in PrimeVul dataset
ALL_CWES = {
    # Memory - Buffer Errors
    119: "Improper Restriction of Operations within the Bounds of a Memory Buffer",
    120: "Buffer Copy without Checking Size of Input (Classic Buffer Overflow)",
    121: "Stack-based Buffer Overflow",
    122: "Heap-based Buffer Overflow",
    125: "Out-of-bounds Read",
    131: "Incorrect Calculation of Buffer Size",
    787: "Out-of-bounds Write",
    805: "Buffer Access with Incorrect Length Value",
    # Memory - Integer Errors
    189: "Numeric Errors",
    190: "Integer Overflow or Wraparound",
    191: "Integer Underflow",
    369: "Divide By Zero",
    # Memory - Memory Management
    401: "Missing Release of Memory after Effective Lifetime",
    415: "Double Free",
    416: "Use After Free",
    772: "Missing Release of Resource after Effective Lifetime",
    # Memory - Pointer Dereference
    476: "NULL Pointer Dereference",
    617: "Reachable Assertion",
    # Injection
    74: "Improper Neutralization of Special Elements in Output (Injection)",
    77: "Improper Neutralization of Special Elements used in a Command (Command Injection)",
    78: "Improper Neutralization of Special Elements used in an OS Command (OS Command Injection)",
    79: "Improper Neutralization of Input During Web Page Generation (Cross-site Scripting)",
    89: "Improper Neutralization of Special Elements used in an SQL Command (SQL Injection)",
    94: "Improper Control of Generation of Code (Code Injection)",
    # Logic - Concurrency
    362: "Concurrent Execution using Shared Resource with Improper Synchronization (Race Condition)",
    667: "Improper Locking",
    # Logic - Resource Management
    399: "Resource Management Errors",
    400: "Uncontrolled Resource Consumption",
    770: "Allocation of Resources Without Limits or Throttling",
    835: "Loop with Unreachable Exit Condition (Infinite Loop)",
    # Logic - Access Control
    264: "Permissions, Privileges, and Access Controls",
    269: "Improper Privilege Management",
    284: "Improper Access Control",
    862: "Missing Authorization",
    # Logic - Information Exposure
    200: "Exposure of Sensitive Information to an Unauthorized Actor",
    209: "Generation of Error Message Containing Sensitive Information",
    # Input - Path Traversal
    22: "Improper Limitation of a Pathname to a Restricted Directory (Path Traversal)",
    59: "Improper Link Resolution Before File Access (Link Following)",
    # Input - Validation
    20: "Improper Input Validation",
    703: "Improper Check or Handling of Exceptional Conditions",
    704: "Incorrect Type Conversion or Cast",
    754: "Improper Check for Unusual or Exceptional Conditions",
    # Crypto
    254: "7PK - Security Features",
    310: "Cryptographic Issues",
    311: "Missing Encryption of Sensitive Data",
    312: "Cleartext Storage of Sensitive Information",
    326: "Inadequate Encryption Strength",
    327: "Use of a Broken or Risky Cryptographic Algorithm",
    330: "Use of Insufficiently Random Values",
    # Other/Rare
    17: "DEPRECATED: Code",
    134: "Use of Externally-Controlled Format String",
    252: "Unchecked Return Value",
    276: "Incorrect Default Permissions",
    285: "Improper Authorization",
    287: "Improper Authentication",
    295: "Improper Certificate Validation",
    320: "DEPRECATED: Key Management Errors",
    347: "Improper Verification of Cryptographic Signature",
    388: "7PK - Errors",
    404: "Improper Resource Shutdown or Release",
    426: "Untrusted Search Path",
    665: "Improper Initialization",
    674: "Uncontrolled Recursion",
    682: "Incorrect Calculation",
    732: "Incorrect Permission Assignment for Critical Resource",
    834: "Excessive Iteration",
    843: "Access of Resource Using Incompatible Type (Type Confusion)",
    908: "Use of Uninitialized Resource",
    909: "Missing Initialization of Resource",
    918: "Server-Side Request Forgery (SSRF)",
}

# Build formatted CWE list grouped by category
def build_cwe_reference():
    """Build a compact CWE reference list."""
    lines = []
    # Group by major category
    categories = defaultdict(list)
    for cwe_id, desc in sorted(ALL_CWES.items()):
        major = MIDDLE_TO_MAJOR.get(CWE_TO_MIDDLE.get(cwe_id, "Other"), "Other")
        categories[major].append((cwe_id, desc))

    for cat in ["Memory", "Injection", "Logic", "Input", "Crypto", "Other"]:
        if cat not in categories:
            continue
        lines.append(f"\n### {cat}:")
        for cwe_id, desc in categories[cat]:
            # Shorten description
            short_desc = desc.split("(")[0].strip() if "(" in desc else desc
            if len(short_desc) > 60:
                short_desc = short_desc[:57] + "..."
            lines.append(f"  CWE-{cwe_id}: {short_desc}")
    return "\n".join(lines)


CWE_REFERENCE = build_cwe_reference()

DIRECT_CLASSIFICATION_PROMPT = """You are a security expert analyzing C/C++ source code for vulnerabilities.

## Task
Analyze the code below and classify it as either:
- **Benign**: No vulnerability
- **CWE-XXX**: A specific CWE vulnerability type

## CWE Reference (all possible types):
{cwe_reference}

## Code:
```c
{code}
```

## Instructions:
1. Carefully read the code looking for vulnerability patterns
2. If no vulnerability exists, output "Benign"
3. If vulnerable, identify the MOST SPECIFIC CWE that matches
4. Key patterns to look for:
   - Buffer operations without bounds checking → CWE-119/120/122/125/787
   - NULL pointer dereference → CWE-476
   - Use after free → CWE-416, Double free → CWE-415
   - Integer overflow → CWE-190, Divide by zero → CWE-369
   - Missing error/exception handling → CWE-703/754
   - Reachable assertion → CWE-617
   - Race condition → CWE-362
   - Memory leak → CWE-401
   - Path traversal → CWE-22
   - Input validation → CWE-20
   - Information exposure → CWE-200

## Output (JSON only):
{{
  "classification": "CWE-XXX" or "Benign",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of vulnerability pattern found"
}}"""


def classify_sample(client, item, cwe_reference):
    """Classify a single sample."""
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

    prompt = DIRECT_CLASSIFICATION_PROMPT.format(
        code=code[:8000],  # Limit code length
        cwe_reference=cwe_reference,
    )

    try:
        response = client.generate(prompt, max_tokens=300, temperature=0.0)
        pred_cwe = parse_classification(response)
    except Exception as e:
        pred_cwe = "Error"

    return {"gt_cwe": gt_cwe, "pred_cwe": pred_cwe}


def parse_classification(response: str) -> str:
    """Parse CWE classification from response."""
    # Try JSON parsing
    try:
        # Find JSON in response
        m = re.search(r'\{[^{}]*"classification"\s*:\s*"([^"]+)"[^{}]*\}', response, re.DOTALL)
        if m:
            cls = m.group(1).strip()
            if cls.lower() == "benign":
                return "Benign"
            m2 = re.search(r"CWE-(\d+)", cls)
            if m2:
                return f"CWE-{m2.group(1)}"
    except:
        pass

    # Fallback: look for CWE-XXX pattern
    m = re.search(r"CWE-(\d+)", response)
    if m:
        return f"CWE-{m.group(1)}"

    # Check for benign indicators
    if any(w in response.lower() for w in ["benign", "no vulnerability", "not vulnerable", "safe"]):
        return "Benign"

    return "Unknown"


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

    # GT classes only
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

    # Macro-F1 over GT classes
    f1s = list(class_f1.values())
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0

    # Weighted-F1
    total = sum(gt_counts.values())
    weighted_f1 = sum(class_f1.get(c, 0) * gt_counts[c] / total for c in gt_classes)

    # CWE coverage
    coverage = sum(1 for v in class_f1.values() if v > 0)

    # Binary F1
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
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default="outputs/rebuttal/cwe130/direct_cwe_results.json")
    args = parser.parse_args()

    load_env_vars()

    # Load data
    with open(args.data) as f:
        data = [json.loads(l) for l in f]
    if args.max_samples > 0:
        data = data[:args.max_samples]

    print(f"{'='*70}")
    print(f"Direct CWE Classification Evaluation")
    print(f"{'='*70}")
    print(f"  Data: {args.data} ({len(data)} samples)")
    print(f"  Model: {args.model}")
    print(f"  Workers: {args.workers}")
    print(flush=True)

    results = []
    errors = 0
    start = time.time()

    def eval_one(item):
        client = create_llm_client(model_name=args.model)
        return classify_sample(client, item, CWE_REFERENCE)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(eval_one, item): i for i, item in enumerate(data)}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                idx = futures[future]
                results.append({"gt_cwe": "Unknown", "pred_cwe": "Error"})
                errors += 1

            if len(results) % 50 == 0:
                elapsed = time.time() - start
                rate = len(results) / elapsed
                print(f"  [{len(results)}/{len(data)}] {rate:.1f} samples/sec, errors={errors}", flush=True)

    elapsed = time.time() - start
    print(f"\nCompleted: {len(results)} samples in {elapsed:.0f}s ({len(results)/elapsed:.1f} samples/sec)")

    # Compute metrics
    metrics = compute_metrics(results)

    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"{'='*70}")
    print(f"  Macro-F1 (CWE-level): {metrics['macro_f1']*100:.2f}%")
    print(f"  Weighted-F1: {metrics['weighted_f1']*100:.2f}%")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Binary F1: {metrics['binary_f1']*100:.2f}% (P={metrics['binary_precision']*100:.1f}%, R={metrics['binary_recall']*100:.1f}%)")
    print(f"  CWE Coverage: {metrics['coverage']}/{metrics['total_classes']}")

    # Show top-20 class F1s
    print(f"\n  Top-20 class F1:")
    for cls, f1 in list(metrics['class_f1'].items())[:20]:
        print(f"    {cls:15s}: {f1:.4f}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "method": "direct_cwe_classification",
        "model": args.model,
        "n_samples": len(results),
        "metrics": metrics,
        "predictions": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
