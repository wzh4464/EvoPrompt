#!/usr/bin/env python3
"""All-category RAG CWE classification.

Stage 1: Binary classification (1 call)
Stage 2: Run ALL 5 category detectors in parallel (5 calls, only for vulnerable)
Stage 3: Aggregate - pick highest confidence CWE prediction

Uses KB examples as few-shot demonstrations.
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

# ============================================================================
# Stage 1: Binary classification
# ============================================================================

BINARY_PROMPT = """Analyze this C/C++ code and determine if it contains a security vulnerability.

## Code:
```c
{code}
```

## Output (JSON only):
{{
  "vulnerable": true or false,
  "confidence": 0.0-1.0,
  "reason": "Brief explanation"
}}"""

# ============================================================================
# Stage 2: Per-category CWE classification with examples
# ============================================================================

CATEGORY_CONFIGS = {
    "Memory": {
        "cwes": {
            119: "Buffer overflow - operations exceed buffer bounds",
            120: "Classic buffer overflow - copy without bounds check",
            121: "Stack-based buffer overflow",
            122: "Heap-based buffer overflow",
            125: "Out-of-bounds read",
            131: "Incorrect buffer size calculation",
            189: "Numeric errors",
            190: "Integer overflow or wraparound",
            191: "Integer underflow",
            369: "Divide by zero",
            401: "Memory leak",
            415: "Double free",
            416: "Use after free",
            476: "NULL pointer dereference",
            617: "Reachable assertion",
            772: "Resource leak",
            787: "Out-of-bounds write",
            805: "Buffer access with incorrect length",
        },
        "distinctions": """Key patterns:
- memcpy/strcpy/strcat without size check → CWE-120 (classic overflow)
- Array index beyond bounds (write) → CWE-787 (OOB write)
- Array index beyond bounds (read) → CWE-125 (OOB read)
- Generic buffer bounds issue → CWE-119
- ptr = malloc(); ... free(ptr); ... *ptr → CWE-416 (use-after-free)
- free(ptr); free(ptr) → CWE-415 (double free)
- malloc() without matching free() on error path → CWE-401 (memory leak)
- if (!ptr) deref → CWE-476 (NULL deref)
- assert()/BUG_ON() with attacker-controlled condition → CWE-617
- x * y without overflow check → CWE-190 (integer overflow)
- x / y without y!=0 check → CWE-369 (divide by zero)""",
    },
    "Injection": {
        "cwes": {
            74: "Injection (general)",
            77: "Command injection",
            78: "OS command injection",
            79: "Cross-site scripting (XSS)",
            89: "SQL injection",
            94: "Code injection",
        },
        "distinctions": """Key patterns:
- User input in system()/exec()/popen() → CWE-78
- User input in SQL query string → CWE-89
- User input in eval()/code execution → CWE-94
- User input in HTML output → CWE-79
- Other injection → CWE-74""",
    },
    "Logic": {
        "cwes": {
            200: "Information exposure",
            209: "Info leak via error message",
            264: "Permissions/privileges",
            269: "Improper privilege management",
            284: "Improper access control",
            362: "Race condition (TOCTOU)",
            399: "Resource management errors",
            400: "Uncontrolled resource consumption",
            667: "Improper locking",
            770: "Resource allocation without limits",
            835: "Infinite loop",
            862: "Missing authorization",
        },
        "distinctions": """Key patterns:
- check(file) then use(file) without lock → CWE-362 (race condition)
- Missing lock/unlock around shared resource → CWE-667
- Sensitive data in log/error/response → CWE-200/209
- Missing permission/auth check before operation → CWE-264/284
- setuid/setgid misuse → CWE-269
- Unbounded allocation/loop → CWE-400/770/835
- Resource not properly cleaned up → CWE-399""",
    },
    "Input": {
        "cwes": {
            20: "Improper input validation",
            22: "Path traversal",
            59: "Symlink following",
            703: "Improper exception handling",
            704: "Incorrect type conversion",
            754: "Improper check for unusual conditions",
        },
        "distinctions": """Key patterns:
- "../" in file path not checked → CWE-22
- Following symlinks to escape directory → CWE-59
- Return value not checked (e.g., malloc, read, open) → CWE-703/252
- Missing validation of size/format/range → CWE-20
- Cast to wrong type (e.g., signed/unsigned mismatch) → CWE-704
- Unusual conditions (NULL, empty, negative) not handled → CWE-754""",
    },
    "Crypto": {
        "cwes": {
            254: "Security features",
            310: "Cryptographic issues",
            311: "Missing encryption",
            312: "Cleartext storage",
            326: "Inadequate encryption strength",
            327: "Broken crypto algorithm",
            330: "Insufficiently random values",
        },
        "distinctions": """Key patterns:
- Using DES/MD5/SHA1 for security → CWE-327
- Password/key in plaintext → CWE-312
- Network data without TLS → CWE-311
- rand()/time() for crypto → CWE-330
- Short key length → CWE-326""",
    },
}

# Additional CWE → category mapping for rare CWEs
EXTRA_CWE_TO_CAT = {
    17: "Logic", 134: "Memory", 252: "Input", 276: "Logic",
    285: "Logic", 287: "Logic", 295: "Crypto", 320: "Crypto",
    347: "Crypto", 388: "Input", 404: "Memory", 426: "Input",
    665: "Memory", 674: "Logic", 682: "Memory", 732: "Logic",
    824: "Memory", 834: "Logic", 843: "Memory", 908: "Memory",
    909: "Memory", 918: "Injection",
}


CWE_DETECTOR_PROMPT = """Analyze this code for **{category}** vulnerabilities. If you find a vulnerability in this category, output the specific CWE. If no {category} vulnerability exists, output "Benign".

## {category} CWE Types:
{cwe_list}

## {distinctions}

{examples}

## Code:
```c
{code}
```

## Output (JSON only):
{{
  "prediction": "CWE-XXX" or "Benign",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}}"""


class KBExamples:
    """Load and serve KB examples."""

    def __init__(self, kb_path: str):
        self.examples_by_cwe = defaultdict(list)
        with open(kb_path) as f:
            kb = json.load(f)
        for cwe_key, examples in kb.get("by_cwe", {}).items():
            m = re.search(r"(\d+)", str(cwe_key))
            if m:
                cwe_id = int(m.group(1))
                for ex in examples[:2]:
                    code = ex.get("func", ex.get("code", ""))
                    if code:
                        lines = code.split("\n")
                        if len(lines) > 25:
                            code = "\n".join(lines[:25]) + "\n// ... truncated"
                        self.examples_by_cwe[cwe_id].append(code)

    def get_examples_for_category(self, category: str, max_examples: int = 3) -> str:
        """Get formatted KB examples for a category."""
        config = CATEGORY_CONFIGS.get(category, {})
        cwes = config.get("cwes", {})
        parts = []
        count = 0
        for cwe_id in cwes:
            if cwe_id in self.examples_by_cwe and count < max_examples:
                ex = self.examples_by_cwe[cwe_id][0]
                parts.append(f"**Example CWE-{cwe_id} ({cwes[cwe_id]}):**\n```c\n{ex}\n```")
                count += 1
        if parts:
            return "## Reference Examples:\n" + "\n\n".join(parts)
        return ""


def classify_sample(client, item, kb):
    """Classify with binary + all-category approach."""
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

    # Stage 1: Binary
    prompt1 = BINARY_PROMPT.format(code=code[:8000])
    try:
        resp1 = client.generate(prompt1, max_tokens=150, temperature=0.0)
        is_vuln = parse_binary(resp1)
    except Exception:
        return {"gt_cwe": gt_cwe, "pred_cwe": "Error"}

    if not is_vuln:
        return {"gt_cwe": gt_cwe, "pred_cwe": "Benign"}

    # Stage 2: Run ALL 5 category detectors
    predictions = []
    for cat, config in CATEGORY_CONFIGS.items():
        cwe_list = "\n".join(f"  - CWE-{cid}: {desc}" for cid, desc in config["cwes"].items())
        examples = kb.get_examples_for_category(cat, max_examples=2)

        prompt2 = CWE_DETECTOR_PROMPT.format(
            category=cat,
            cwe_list=cwe_list,
            distinctions=config["distinctions"],
            examples=examples,
            code=code[:5000],
        )

        try:
            resp2 = client.generate(prompt2, max_tokens=200, temperature=0.0)
            pred, conf = parse_cwe_response(resp2)
            if pred != "Benign":
                predictions.append((pred, conf, cat))
        except Exception:
            pass

    # Stage 3: Aggregate - pick highest confidence
    if not predictions:
        return {"gt_cwe": gt_cwe, "pred_cwe": "Benign"}

    # Sort by confidence, pick highest
    predictions.sort(key=lambda x: x[1], reverse=True)
    best_pred = predictions[0][0]

    return {"gt_cwe": gt_cwe, "pred_cwe": best_pred, "all_preds": [(p, c, cat) for p, c, cat in predictions]}


def parse_binary(response: str) -> bool:
    """Parse binary classification."""
    try:
        m = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            return bool(data.get("vulnerable", False))
    except:
        pass
    return "true" in response.lower()[:100]


def parse_cwe_response(response: str) -> tuple:
    """Parse CWE prediction and confidence."""
    try:
        m = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            pred = data.get("prediction", "")
            conf = float(data.get("confidence", 0.5))
            if pred.lower() in ("benign", "safe", "none"):
                return "Benign", conf
            m2 = re.search(r"CWE-(\d+)", pred)
            if m2:
                return f"CWE-{m2.group(1)}", conf
    except:
        pass

    # Fallback
    m = re.search(r"CWE-(\d+)", response)
    if m:
        return f"CWE-{m.group(1)}", 0.5

    if any(w in response.lower() for w in ["benign", "no vulnerability"]):
        return "Benign", 0.5

    return "Unknown", 0.0


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

    # Binary
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
    parser.add_argument("--kb", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default="outputs/rebuttal/cwe130/allcat_rag_300.json")
    args = parser.parse_args()

    load_env_vars()

    with open(args.data) as f:
        data = [json.loads(l) for l in f]
    if args.max_samples > 0:
        data = data[:args.max_samples]

    kb = KBExamples(args.kb)
    print(f"Loaded KB: {sum(len(v) for v in kb.examples_by_cwe.values())} examples across {len(kb.examples_by_cwe)} CWEs")

    print(f"\n{'='*70}")
    print(f"All-Category RAG CWE Classification")
    print(f"{'='*70}")
    print(f"  Data: {args.data} ({len(data)} samples)")
    print(f"  Model: {args.model}")
    print(f"  Workers: {args.workers}")
    print(f"  Calls per vulnerable sample: 6 (1 binary + 5 category)")
    print(flush=True)

    results = []
    errors = 0
    start = time.time()

    def eval_one(item):
        client = create_llm_client(model_name=args.model)
        return classify_sample(client, item, kb)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(eval_one, item): i for i, item in enumerate(data)}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({"gt_cwe": "Unknown", "pred_cwe": "Error"})
                errors += 1

            if len(results) % 50 == 0:
                elapsed = time.time() - start
                rate = len(results) / elapsed
                correct = sum(1 for r in results if r["gt_cwe"] == r["pred_cwe"])
                print(f"  [{len(results)}/{len(data)}] {rate:.1f} samples/sec, accuracy={correct}/{len(results)}, errors={errors}", flush=True)

    elapsed = time.time() - start
    print(f"\nCompleted: {len(results)} samples in {elapsed:.0f}s ({len(results)/elapsed:.1f} samples/sec)")

    metrics = compute_metrics(results)

    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"{'='*70}")
    print(f"  Macro-F1 (CWE-level): {metrics['macro_f1']*100:.2f}%")
    print(f"  Weighted-F1: {metrics['weighted_f1']*100:.2f}%")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Binary F1: {metrics['binary_f1']*100:.2f}% (P={metrics['binary_precision']*100:.1f}%, R={metrics['binary_recall']*100:.1f}%)")
    print(f"  CWE Coverage: {metrics['coverage']}/{metrics['total_classes']}")

    print(f"\n  Top-20 class F1:")
    for cls, f1 in list(metrics['class_f1'].items())[:20]:
        print(f"    {cls:15s}: {f1:.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "method": "all_category_rag",
        "model": args.model,
        "n_samples": len(results),
        "n_errors": errors,
        "elapsed": elapsed,
        "metrics": metrics,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
