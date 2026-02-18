"""Exp 2 - MulVul baseline: Coarse-to-Fine (Router → Detector) with token tracking.

Simulates MulVul's architecture:
  1. Router: Predict top-k coarse categories
  2. Detector: For each routed category, identify vulnerability type
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "rebuttal"))
from llm_utils import (
    TokenStats, call_llm, compute_metrics, create_client,
    load_samples, parse_vulnerability_label,
)

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "rebuttal" / "exp2_agent_comparison"

ROUTER_SYSTEM = """You are a security code analyst. Your task is to categorize the potential vulnerability type of the given code.

Analyze the code and predict the top-2 most likely vulnerability categories from:
- Memory (buffer overflow, use-after-free, double-free, etc.)
- Input Validation (improper input validation, injection, etc.)
- Integer (integer overflow/underflow, type confusion, etc.)
- Null Pointer (null pointer dereference, assertion failure, etc.)
- Concurrency (race condition, deadlock, etc.)
- Authentication (improper authentication, missing authorization, etc.)
- Cryptography (weak crypto, improper certificate validation, etc.)
- Resource (resource leak, missing release, etc.)
- Logic (incorrect calculation, off-by-one, etc.)
- Benign (no vulnerability)

Output ONLY a JSON list of the top-2 categories, e.g.: ["Memory", "Benign"]"""

DETECTOR_SYSTEM = """You are an expert vulnerability detector specializing in {category} vulnerabilities.

Analyze the following code carefully. Determine if it contains a {category}-related vulnerability.

If vulnerable, respond with: Vulnerable - [CWE-ID] [brief description]
If not vulnerable, respond with: Benign - no {category} vulnerability found.

Be precise and concise."""


def run():
    client = create_client()
    stats = TokenStats()
    samples = load_samples()

    print(f"Running MulVul baseline on {len(samples)} samples...")
    predictions = []
    details = []

    for i, sample in enumerate(samples):
        code = sample["func"]
        gt = sample["target"]

        # Step 1: Router - predict top-2 categories
        router_prompt = f"Analyze this code and predict the top-2 vulnerability categories:\n\n```c\n{code[:3000]}\n```"
        router_resp = call_llm(client, router_prompt, stats, system_prompt=ROUTER_SYSTEM, max_tokens=128)

        # Parse router output
        categories = []
        try:
            parsed = json.loads(router_resp)
            if isinstance(parsed, list):
                categories = parsed[:2]
        except json.JSONDecodeError:
            # Fallback: extract from text
            for cat in ["Memory", "Input Validation", "Integer", "Null Pointer",
                        "Concurrency", "Authentication", "Cryptography", "Resource", "Logic", "Benign"]:
                if cat.lower() in router_resp.lower():
                    categories.append(cat)
            categories = categories[:2]

        if not categories:
            categories = ["Memory", "Benign"]

        # Step 2: Detector - for each routed category
        detector_responses = []
        for cat in categories:
            det_sys = DETECTOR_SYSTEM.format(category=cat)
            det_prompt = f"Analyze this code for {cat} vulnerabilities:\n\n```c\n{code[:3000]}\n```"
            det_resp = call_llm(client, det_prompt, stats, system_prompt=det_sys, max_tokens=256)
            detector_responses.append({"category": cat, "response": det_resp})

        # Step 3: Aggregate - if any detector says vulnerable, predict vulnerable
        final_pred = "benign"
        for dr in detector_responses:
            label = parse_vulnerability_label(dr["response"])
            if label == "vulnerable":
                final_pred = "vulnerable"
                break

        predictions.append(final_pred)
        details.append({
            "idx": sample.get("idx"),
            "ground_truth": gt,
            "prediction": final_pred,
            "router_categories": categories,
            "detector_responses": detector_responses,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] calls={stats.total_calls}, tokens={stats.total_tokens}")

    # Compute metrics
    ground_truths = [s["target"] for s in samples]
    metrics = compute_metrics(predictions, ground_truths)
    cost_summary = stats.summary(len(samples))

    results = {
        "method": "MulVul (Router-Detector)",
        "model": "from_env",
        "n_samples": len(samples),
        "metrics": metrics,
        "cost": cost_summary,
    }

    print("\n=== MulVul Results ===")
    print(json.dumps(results, indent=2))

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "mulvul_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "mulvul_details.jsonl", "w") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\nSaved to {OUT_DIR}")
    return results


if __name__ == "__main__":
    run()
