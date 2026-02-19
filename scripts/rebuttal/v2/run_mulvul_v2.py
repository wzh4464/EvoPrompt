"""MulVul v2: Coarse-to-Fine CWE category classification (Router → Detector)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from llm_utils_v2 import (
    TokenStats, call_llm, compute_multiclass_metrics, create_client,
    load_samples, parse_category, sample_ground_truth,
    CATEGORIES_FOR_PROMPT, OUT_DIR,
)

ROUTER_SYS = """You are a security code analyst performing vulnerability triage.
Analyze the code and predict the top-2 most likely vulnerability categories.

""" + CATEGORIES_FOR_PROMPT + """

Output ONLY a JSON list of the top-2 categories, e.g.: ["Buffer Errors", "Benign"]
If you believe the code is safe, output: ["Benign"]"""

DETECTOR_SYS = """You are an expert vulnerability detector specializing in {category} vulnerabilities.

Analyze the code. Determine if it contains a {category}-related vulnerability.
If vulnerable, classify it into the MOST SPECIFIC category from:

""" + CATEGORIES_FOR_PROMPT + """

Output format: <category name>
Example: "Buffer Errors" or "Benign"
Output ONLY the category name, nothing else."""


def run():
    client = create_client()
    stats = TokenStats()
    samples = load_samples()
    print(f"MulVul v2 on {len(samples)} samples (CWE category classification)...")

    predictions, details = [], []
    for i, s in enumerate(samples):
        code = s["func"][:3000]
        gt = sample_ground_truth(s)

        # Router: predict top-2 categories
        rp = f"Predict top-2 vulnerability categories:\n\n```c\n{code}\n```"
        rr = call_llm(client, rp, stats, system_prompt=ROUTER_SYS, max_tokens=128)
        cats = []
        try:
            parsed = json.loads(rr)
            if isinstance(parsed, list):
                cats = [str(c) for c in parsed[:2]]
        except json.JSONDecodeError:
            cats = [parse_category(rr)]
        if not cats:
            cats = ["Benign"]

        # Detector: classify per routed category
        det_results = []
        for cat in cats:
            if cat == "Benign":
                det_results.append("Benign")
                continue
            dp = f"Classify vulnerability type:\n\n```c\n{code}\n```"
            dr = call_llm(client, dp, stats,
                          system_prompt=DETECTOR_SYS.format(category=cat), max_tokens=64)
            det_results.append(parse_category(dr))

        # Aggregate: pick first non-Benign, or Benign
        final = "Benign"
        for r in det_results:
            if r != "Benign":
                final = r
                break

        predictions.append(final)
        details.append({"idx": s.get("idx"), "gt": gt, "pred": final,
                        "router": cats, "detectors": det_results})
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}] calls={stats.total_calls} tokens={stats.total_tokens}")

    gts = [sample_ground_truth(s) for s in samples]
    metrics = compute_multiclass_metrics(predictions, gts)
    results = {"method": "MulVul (Router-Detector)", "metrics": metrics,
               "cost": stats.summary(len(samples))}
    print("\n=== MulVul v2 ===")
    print(json.dumps({k: v for k, v in results.items() if k != "metrics"}, indent=2))
    print(f"Macro-F1: {metrics['macro_f1']:.4f}  Accuracy: {metrics['accuracy']:.4f}")
    print(f"Binary Vuln Recall: {metrics['binary_vuln_recall']:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "mulvul_v2_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "mulvul_v2_details.jsonl", "w") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    run()
