#!/usr/bin/env python3
"""Exp 1A - Single-pass baseline without RAG."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "rebuttal"))
from llm_utils import (  # noqa: E402
    TokenStats,
    call_llm,
    compute_metrics,
    create_client,
    load_samples,
    parse_vulnerability_label,
)


OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "rebuttal" / "exp1_singlepass"

SYSTEM_PROMPT = """You are a security code analyst. Analyze the given code and determine
if it contains a security vulnerability.

If vulnerable, respond: Vulnerable - [CWE-ID] [brief description]
If not vulnerable, respond: Benign - no vulnerability found."""


def build_prompt(code: str) -> str:
    clipped = code[:3500]
    return (
        "Analyze the following target code.\n\n"
        f"```c\n{clipped}\n```\n\n"
        "Is this code vulnerable?"
    )


def run() -> dict:
    client = create_client()
    stats = TokenStats()
    samples = load_samples()

    print(f"Running single-pass no-RAG baseline on {len(samples)} samples...")
    predictions = []
    details = []

    for i, sample in enumerate(samples):
        code = sample.get("func", "")
        gt = int(sample.get("target", 0))

        prompt = build_prompt(code)
        response = call_llm(
            client=client,
            prompt=prompt,
            stats=stats,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=256,
        )
        pred = parse_vulnerability_label(response)
        predictions.append(pred)

        details.append(
            {
                "idx": sample.get("idx"),
                "ground_truth": gt,
                "prediction": pred,
                "response": response,
            }
        )

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] calls={stats.total_calls}, tokens={stats.total_tokens}")

    metrics = compute_metrics(predictions, [int(s.get("target", 0)) for s in samples])
    cost_summary = stats.summary(len(samples))
    results = {
        "method": "Single-pass (no RAG)",
        "model": "from_env",
        "n_samples": len(samples),
        "metrics": metrics,
        "cost": cost_summary,
        "retrieval_enabled": False,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "singlepass_norag_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "singlepass_norag_details.jsonl", "w") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n=== Single-pass (no RAG) Results ===")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {OUT_DIR}")
    return results


if __name__ == "__main__":
    run()
