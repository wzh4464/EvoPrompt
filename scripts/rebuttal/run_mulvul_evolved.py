#!/usr/bin/env python3
"""Exp 2 variant - MulVul using evolved best prompts."""

from __future__ import annotations

import json
import re
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "outputs" / "rebuttal" / "cwe130_evolution" / "best_prompts.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "rebuttal" / "exp2_agent_comparison"

DETECTOR_CATEGORIES = ["Memory", "Injection", "Logic", "Input", "Crypto"]


def load_evolved_prompts(path: Path) -> tuple[str, dict[str, str], str]:
    """Load evolved router and detector prompts."""
    if not path.exists():
        raise FileNotFoundError(f"Evolved prompt file not found: {path}")

    payload = json.loads(path.read_text())
    prompt_root = payload.get("prompts", payload)
    router_prompt = prompt_root.get("router_prompt", "")
    detector_prompts = prompt_root.get("detector_prompts", {})
    prompt_id = prompt_root.get("id", "unknown")

    if not router_prompt or not detector_prompts:
        raise ValueError(f"Invalid evolved prompt format in {path}")

    return router_prompt, detector_prompts, prompt_id


def fill_template(template: str, code: str, evidence: str = "") -> str:
    """Fill prompt placeholders while keeping literal braces untouched."""
    text = template.replace("{evidence}", evidence).replace("{code}", code)
    return text


def normalize_category(cat: str) -> str:
    """Normalize router category names to detector categories."""
    c = (cat or "").strip().lower()
    if not c:
        return "Benign"

    mapping = {
        "memory": "Memory",
        "injection": "Injection",
        "logic": "Logic",
        "input": "Input",
        "input validation": "Input",
        "crypto": "Crypto",
        "cryptography": "Crypto",
        "benign": "Benign",
        "safe": "Benign",
        "integer": "Memory",
        "null pointer": "Memory",
        "concurrency": "Logic",
        "authentication": "Logic",
        "resource": "Logic",
    }
    if c in mapping:
        return mapping[c]

    for key, value in mapping.items():
        if key in c:
            return value
    return "Benign"


def parse_router_categories(response: str) -> list[str]:
    """Parse top routed categories from router response."""
    categories = []

    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            preds = parsed.get("predictions", [])
            if isinstance(preds, list):
                for p in preds:
                    if isinstance(p, dict):
                        cat = normalize_category(str(p.get("category", "")))
                        if cat and cat not in categories:
                            categories.append(cat)
        elif isinstance(parsed, list):
            for item in parsed:
                cat = normalize_category(str(item))
                if cat and cat not in categories:
                    categories.append(cat)
    except json.JSONDecodeError:
        pass

    if not categories:
        for raw in ["Memory", "Injection", "Logic", "Input", "Crypto", "Benign"]:
            if raw.lower() in response.lower():
                cat = normalize_category(raw)
                if cat not in categories:
                    categories.append(cat)

    if not categories:
        categories = ["Memory", "Benign"]
    return categories[:2]


def run() -> dict:
    client = create_client()
    stats = TokenStats()
    samples = load_samples()
    router_template, detector_templates, prompt_id = load_evolved_prompts(PROMPT_PATH)

    print(f"Running MulVul with evolved prompts on {len(samples)} samples...")
    print(f"Loaded prompt set: {prompt_id} from {PROMPT_PATH}")

    predictions = []
    details = []

    for i, sample in enumerate(samples):
        code = sample.get("func", "")[:3000]
        gt = int(sample.get("target", 0))

        router_prompt = fill_template(router_template, code=code, evidence="")
        router_resp = call_llm(
            client=client,
            prompt=router_prompt,
            stats=stats,
            system_prompt=None,
            max_tokens=180,
        )
        categories = parse_router_categories(router_resp)

        detector_responses = []
        for cat in categories:
            if cat == "Benign":
                detector_responses.append({"category": cat, "response": "Benign"})
                continue

            det_template = detector_templates.get(cat)
            if not det_template:
                # Last fallback by fuzzy detector key match
                for k in detector_templates:
                    if k.lower() == cat.lower():
                        det_template = detector_templates[k]
                        break

            if not det_template:
                detector_responses.append({"category": cat, "response": "Benign"})
                continue

            det_prompt = fill_template(det_template, code=code, evidence="")
            det_resp = call_llm(
                client=client,
                prompt=det_prompt,
                stats=stats,
                system_prompt=None,
                max_tokens=220,
            )
            detector_responses.append({"category": cat, "response": det_resp})

        final_pred = "benign"
        for dr in detector_responses:
            label = parse_vulnerability_label(dr["response"])
            if label == "vulnerable":
                final_pred = "vulnerable"
                break

        predictions.append(final_pred)
        details.append(
            {
                "idx": sample.get("idx"),
                "ground_truth": gt,
                "prediction": final_pred,
                "router_response": router_resp,
                "router_categories": categories,
                "detector_responses": detector_responses,
            }
        )

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] calls={stats.total_calls}, tokens={stats.total_tokens}")

    ground_truths = [int(s.get("target", 0)) for s in samples]
    metrics = compute_metrics(predictions, ground_truths)
    cost_summary = stats.summary(len(samples))

    results = {
        "method": "MulVul (evolved prompts)",
        "model": "from_env",
        "n_samples": len(samples),
        "prompt_source": str(PROMPT_PATH),
        "prompt_id": prompt_id,
        "detector_categories": DETECTOR_CATEGORIES,
        "metrics": metrics,
        "cost": cost_summary,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "mulvul_evolved_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "mulvul_evolved_details.jsonl", "w") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("\n=== MulVul (evolved prompts) Results ===")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {OUT_DIR}")
    return results


if __name__ == "__main__":
    run()
