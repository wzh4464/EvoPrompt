#!/usr/bin/env python3
"""Two-stage RAG CWE classification.

Stage 1: Binary + Major category classification (1 LLM call)
Stage 2: CWE-specific classification with KB examples (1 LLM call, only for vulnerable)

Uses knowledge base examples as few-shot demonstrations to help the model
distinguish between similar CWE types.
"""

import os
import sys
import json
import time
import re
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evoprompt.llm.client import create_llm_client, load_env_vars
from evoprompt.data.cwe_hierarchy import (
    CWE_TO_MIDDLE, MIDDLE_TO_MAJOR, CWE_DESCRIPTIONS,
    get_cwes_for_major, cwe_to_major,
)

# ============================================================================
# CWE descriptions for each major category
# ============================================================================

CATEGORY_CWES = {
    "Memory": {
        119: "Buffer overflow - operations exceed memory buffer bounds",
        120: "Classic buffer overflow - copying without bounds checking",
        121: "Stack-based buffer overflow",
        122: "Heap-based buffer overflow",
        125: "Out-of-bounds read",
        131: "Incorrect buffer size calculation",
        189: "Numeric errors - general arithmetic issues",
        190: "Integer overflow or wraparound",
        191: "Integer underflow",
        369: "Divide by zero",
        401: "Memory leak - missing release after use",
        415: "Double free",
        416: "Use after free",
        476: "NULL pointer dereference",
        617: "Reachable assertion",
        772: "Resource leak - missing release of resource",
        787: "Out-of-bounds write",
        805: "Buffer access with incorrect length",
    },
    "Injection": {
        74: "Injection - general injection",
        77: "Command injection",
        78: "OS command injection",
        79: "Cross-site scripting (XSS)",
        89: "SQL injection",
        94: "Code injection",
    },
    "Logic": {
        200: "Information exposure",
        209: "Info leak via error message",
        264: "Permissions/privileges issues",
        269: "Improper privilege management",
        284: "Improper access control",
        362: "Race condition",
        399: "Resource management errors",
        400: "Uncontrolled resource consumption",
        667: "Improper locking",
        770: "Resource allocation without limits",
        835: "Infinite loop",
        862: "Missing authorization",
    },
    "Input": {
        20: "Improper input validation",
        22: "Path traversal",
        59: "Improper link resolution (symlink following)",
        703: "Improper exception handling",
        704: "Incorrect type conversion",
        754: "Improper check for unusual conditions",
    },
    "Crypto": {
        254: "Security features issues",
        310: "Cryptographic issues",
        311: "Missing encryption",
        312: "Cleartext storage",
        326: "Inadequate encryption strength",
        327: "Broken/risky crypto algorithm",
        330: "Insufficiently random values",
    },
}

# CWEs that don't fit neatly into categories - map to closest
EXTRA_CWE_MAPPING = {
    17: "Logic",
    134: "Memory",
    252: "Input",
    276: "Logic",
    285: "Logic",
    287: "Logic",
    295: "Crypto",
    320: "Crypto",
    347: "Crypto",
    388: "Input",
    404: "Memory",
    426: "Input",
    665: "Memory",
    674: "Logic",
    682: "Memory",
    732: "Logic",
    824: "Memory",
    834: "Logic",
    843: "Memory",
    908: "Memory",
    909: "Memory",
    918: "Injection",
}


# ============================================================================
# Stage 1: Binary + Category classification
# ============================================================================

STAGE1_PROMPT = """You are a security expert analyzing C/C++ source code.

## Task
1. Determine if the code contains a vulnerability
2. If vulnerable, identify the vulnerability CATEGORY

## Categories:
- **Memory**: Buffer overflow, integer overflow, null pointer, use-after-free, double free, memory leak, divide-by-zero, assertion failure
- **Injection**: SQL/command/code injection, XSS
- **Logic**: Race condition, resource management, access control, information exposure, improper locking
- **Input**: Path traversal, input validation, exception handling, type conversion
- **Crypto**: Encryption issues, weak crypto, cleartext storage

## Code:
```c
{code}
```

## Output (JSON only):
{{
  "vulnerable": true/false,
  "category": "Memory" or "Injection" or "Logic" or "Input" or "Crypto" or "Benign",
  "confidence": 0.0-1.0,
  "reason": "Brief explanation"
}}"""

# ============================================================================
# Stage 2: CWE classification with examples
# ============================================================================

STAGE2_PROMPT_TEMPLATE = """You are a security expert. The code below has been identified as having a **{category}** vulnerability.
Your task is to identify the SPECIFIC CWE type.

## Possible CWE Types:
{cwe_list}

## Reference Examples from Knowledge Base:
{examples}

## Code to Classify:
```c
{code}
```

## Key Distinctions:
{distinctions}

## Output (JSON only):
{{
  "cwe": "CWE-XXX",
  "confidence": 0.0-1.0,
  "reasoning": "Why this specific CWE and not alternatives"
}}"""

CATEGORY_DISTINCTIONS = {
    "Memory": """- Buffer overflow (CWE-119/120/121/122/787): Look for array/buffer operations without bounds checking
  - CWE-119: Generic buffer overflow (use when specific subtype unclear)
  - CWE-120: strcpy/strcat without length check
  - CWE-121: Local/stack buffer overflow
  - CWE-122: malloc'd buffer overflow
  - CWE-125: Reading beyond buffer (information leak pattern)
  - CWE-787: Writing beyond buffer (data corruption pattern)
- NULL pointer (CWE-476): Dereferencing pointer that could be NULL (missing NULL check)
- Assertion (CWE-617): assert() or BUG_ON() that can be triggered by attacker input
- Use-after-free (CWE-416): Accessing freed memory (look for free() followed by use)
- Double free (CWE-415): Calling free() twice on same pointer
- Memory leak (CWE-401/772): Allocating without freeing (missing free on error path)
- Integer overflow (CWE-190): Arithmetic exceeding INT_MAX/UINT_MAX
- Divide by zero (CWE-369): Division without checking divisor != 0
- Numeric errors (CWE-189): General arithmetic issues""",

    "Injection": """- OS command injection (CWE-78): User input in system()/exec() calls
- SQL injection (CWE-89): User input in SQL queries
- Code injection (CWE-94): User input evaluated as code
- XSS (CWE-79): User input reflected in HTML output
- General injection (CWE-74): Other injection not fitting above""",

    "Logic": """- Race condition (CWE-362): TOCTOU, shared resource without synchronization
- Improper locking (CWE-667): Lock/unlock ordering issues, missing locks
- Information exposure (CWE-200): Leaking sensitive data to unauthorized users
- Access control (CWE-264/284): Missing permission checks
- Privilege management (CWE-269): Improper privilege elevation/dropping
- Resource management (CWE-399/400/770): Resource exhaustion, DoS
- Infinite loop (CWE-835): Loop that never terminates""",

    "Input": """- Path traversal (CWE-22): "../" in file paths not sanitized
- Symlink following (CWE-59): Following symbolic links to unintended files
- Input validation (CWE-20): Missing/insufficient validation of user input
- Exception handling (CWE-703): Unchecked return values, missing error handling
- Type conversion (CWE-704): Incorrect casts leading to data corruption
- Unusual conditions (CWE-754): Not handling edge cases properly""",

    "Crypto": """- Broken crypto (CWE-327): Using DES, MD5, SHA1 for security
- Missing encryption (CWE-311): Sensitive data transmitted in cleartext
- Cleartext storage (CWE-312): Passwords/keys stored in plaintext
- Weak randomness (CWE-330): Using rand()/time() for crypto
- General crypto issues (CWE-310): Other cryptographic weaknesses""",
}


class KnowledgeBaseExamples:
    """Retrieves few-shot examples from the knowledge base."""

    def __init__(self, kb_path: str, max_examples_per_cwe: int = 2):
        self.max_per_cwe = max_examples_per_cwe
        self.examples_by_cwe = defaultdict(list)
        self._load(kb_path)

    def _load(self, kb_path: str):
        """Load KB organized by CWE."""
        with open(kb_path) as f:
            kb = json.load(f)

        # Extract examples from by_cwe section
        by_cwe = kb.get("by_cwe", {})
        for cwe_key, examples in by_cwe.items():
            # cwe_key might be like "CWE-119" or "119"
            m = re.search(r"(\d+)", str(cwe_key))
            if m:
                cwe_id = int(m.group(1))
                for ex in examples[:self.max_per_cwe]:
                    code = ex.get("func", ex.get("code", ""))
                    if code:
                        # Truncate long examples
                        if len(code) > 2000:
                            code = code[:2000] + "\n// ... truncated ..."
                        self.examples_by_cwe[cwe_id].append(code)

    def get_examples_for_category(self, category: str, max_total: int = 6) -> str:
        """Get formatted examples for a vulnerability category."""
        cwes = CATEGORY_CWES.get(category, {})
        examples = []
        per_cwe = max(1, max_total // max(len(cwes), 1))

        for cwe_id in cwes:
            exs = self.examples_by_cwe.get(cwe_id, [])
            if exs:
                ex = exs[0]  # Take first example
                # Shorten to key snippet
                lines = ex.split("\n")
                if len(lines) > 30:
                    # Take first 30 lines
                    ex = "\n".join(lines[:30]) + "\n// ... truncated ..."
                examples.append(f"### Example: CWE-{cwe_id} ({cwes[cwe_id]})\n```c\n{ex}\n```")
                if len(examples) >= max_total:
                    break

        if not examples:
            return "(No examples available)"
        return "\n\n".join(examples)


def classify_two_stage(client, item, kb_examples):
    """Two-stage classification: binary+category, then CWE."""
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

    # Stage 1: Binary + Category
    prompt1 = STAGE1_PROMPT.format(code=code[:8000])
    try:
        resp1 = client.generate(prompt1, max_tokens=200, temperature=0.0)
        stage1 = parse_stage1(resp1)
    except Exception as e:
        return {"gt_cwe": gt_cwe, "pred_cwe": "Error", "stage1": "error", "stage2": "skipped"}

    if not stage1["vulnerable"] or stage1["category"] == "Benign":
        return {"gt_cwe": gt_cwe, "pred_cwe": "Benign", "stage1": stage1, "stage2": "skipped"}

    # Stage 2: CWE classification with examples
    category = stage1["category"]
    if category not in CATEGORY_CWES:
        # Try to map to a known category
        category = "Memory"  # default fallback

    cwe_list = "\n".join(
        f"  - CWE-{cid}: {desc}" for cid, desc in CATEGORY_CWES.get(category, {}).items()
    )
    examples = kb_examples.get_examples_for_category(category, max_total=4)
    distinctions = CATEGORY_DISTINCTIONS.get(category, "")

    prompt2 = STAGE2_PROMPT_TEMPLATE.format(
        category=category,
        cwe_list=cwe_list,
        examples=examples,
        code=code[:6000],
        distinctions=distinctions,
    )

    try:
        resp2 = client.generate(prompt2, max_tokens=200, temperature=0.0)
        pred_cwe = parse_stage2(resp2)
    except Exception as e:
        pred_cwe = f"Vulnerable-{category}"

    return {"gt_cwe": gt_cwe, "pred_cwe": pred_cwe, "stage1": stage1, "stage2": pred_cwe}


def parse_stage1(response: str) -> dict:
    """Parse stage 1 response."""
    try:
        m = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            return {
                "vulnerable": bool(data.get("vulnerable", False)),
                "category": data.get("category", "Unknown"),
                "confidence": float(data.get("confidence", 0.5)),
            }
    except:
        pass

    # Fallback
    if "true" in response.lower() and any(c in response for c in ["Memory", "Injection", "Logic", "Input", "Crypto"]):
        for c in ["Memory", "Injection", "Logic", "Input", "Crypto"]:
            if c in response:
                return {"vulnerable": True, "category": c, "confidence": 0.5}
    return {"vulnerable": False, "category": "Benign", "confidence": 0.5}


def parse_stage2(response: str) -> str:
    """Parse stage 2 response."""
    try:
        m = re.search(r'\{[^{}]*"cwe"\s*:\s*"([^"]+)"[^{}]*\}', response, re.DOTALL)
        if m:
            cwe = m.group(1).strip()
            m2 = re.search(r"CWE-(\d+)", cwe)
            if m2:
                return f"CWE-{m2.group(1)}"
    except:
        pass

    m = re.search(r"CWE-(\d+)", response)
    if m:
        return f"CWE-{m.group(1)}"

    if any(w in response.lower() for w in ["benign", "no vulnerability"]):
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
    parser.add_argument("--kb", required=True, help="Knowledge base JSON path")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default="outputs/rebuttal/cwe130/twostage_rag_results.json")
    args = parser.parse_args()

    load_env_vars()

    # Load data
    with open(args.data) as f:
        data = [json.loads(l) for l in f]
    if args.max_samples > 0:
        data = data[:args.max_samples]

    # Load KB examples
    kb_examples = KnowledgeBaseExamples(args.kb, max_examples_per_cwe=2)
    print(f"Loaded KB: {sum(len(v) for v in kb_examples.examples_by_cwe.values())} examples across {len(kb_examples.examples_by_cwe)} CWEs")

    print(f"\n{'='*70}")
    print(f"Two-Stage RAG CWE Classification")
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
        return classify_two_stage(client, item, kb_examples)

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

    # Stage analysis
    vuln_correct = sum(1 for r in results if r.get("stage1", {}).get("vulnerable", False) == (r["gt_cwe"] != "Benign"))
    print(f"\n  Stage 1 binary accuracy: {vuln_correct}/{len(results)} ({vuln_correct/len(results)*100:.1f}%)")

    # Category analysis
    cat_correct = 0
    cat_total = 0
    for r in results:
        if r["gt_cwe"] != "Benign" and isinstance(r.get("stage1"), dict):
            cat_total += 1
            pred_cat = r["stage1"].get("category", "")
            gt_cwe_id = re.search(r"(\d+)", r["gt_cwe"])
            if gt_cwe_id:
                gt_major = MIDDLE_TO_MAJOR.get(CWE_TO_MIDDLE.get(int(gt_cwe_id.group(1)), "Other"), "Other")
                if pred_cat == gt_major:
                    cat_correct += 1
    if cat_total > 0:
        print(f"  Stage 1 category accuracy: {cat_correct}/{cat_total} ({cat_correct/cat_total*100:.1f}%)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "method": "two_stage_rag",
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
