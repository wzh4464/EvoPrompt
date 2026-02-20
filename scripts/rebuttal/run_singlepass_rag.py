#!/usr/bin/env python3
"""Exp 1B - Single-pass baseline with lightweight Jaccard retrieval."""

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
KB_PATH = PROJECT_ROOT / "outputs" / "rebuttal" / "retrieval_kb.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "rebuttal" / "exp1_singlepass"

SYSTEM_PROMPT = """You are a security code analyst. You are given reference examples of
known vulnerabilities and clean code, followed by a target code to analyze.

Use the reference examples to calibrate your judgment. Determine if the
target code contains a security vulnerability.

If vulnerable, respond: Vulnerable - [CWE-ID] [brief description]
If not vulnerable, respond: Benign - no vulnerability found."""


def tokenize(code: str) -> set[str]:
    """Simple tokenization for Jaccard retrieval."""
    return set(re.findall(r"[A-Za-z_]\w*", code.lower()))


def jaccard_similarity(tokens1: set[str], tokens2: set[str]) -> float:
    """Jaccard similarity in [0, 1]."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union else 0.0


def load_kb(path: Path) -> list[dict]:
    """Load retrieval KB and precompute token sets."""
    if not path.exists():
        raise FileNotFoundError(
            f"KB not found at {path}. Run scripts/rebuttal/build_retrieval_kb.py first."
        )

    payload = json.loads(path.read_text())
    entries = payload.get("entries", payload if isinstance(payload, list) else [])
    processed = []
    for row in entries:
        code = row.get("code", "")
        processed.append(
            {
                "id": row.get("id"),
                "major_category": row.get("major_category", "Logic"),
                "cwe": row.get("cwe", "UNKNOWN"),
                "description": row.get("description", "No description available."),
                "code": code,
                "tokens": tokenize(code),
            }
        )
    return processed


def retrieve_examples(query_code: str, kb_entries: list[dict]) -> tuple[list[tuple[dict, float]], list[tuple[dict, float]]]:
    """Retrieve top-3 vulnerable and top-1 benign examples by Jaccard."""
    query_tokens = tokenize(query_code)
    vuln_scored = []
    benign_scored = []

    for entry in kb_entries:
        score = jaccard_similarity(query_tokens, entry["tokens"])
        if entry["major_category"] == "Benign":
            benign_scored.append((entry, score))
        else:
            vuln_scored.append((entry, score))

    vuln_scored.sort(key=lambda x: x[1], reverse=True)
    benign_scored.sort(key=lambda x: x[1], reverse=True)

    return vuln_scored[:3], benign_scored[:1]


def format_example_title(example: dict, idx: int, is_benign: bool) -> str:
    """Render reference example title."""
    if is_benign:
        return f"[Example {idx} - Clean Code]"

    cwe = example.get("cwe", "CWE-Unknown")
    description = example.get("description", "No description available.")
    desc_brief = description.split(".")[0][:90].strip()
    if not desc_brief:
        desc_brief = "no description"
    return f"[Example {idx} - Vulnerable ({cwe}: {desc_brief})]"


def build_prompt(query_code: str, vuln_refs: list[tuple[dict, float]], benign_refs: list[tuple[dict, float]]) -> str:
    """Build user prompt with retrieved references and target code."""
    lines = ["=== Reference Examples ===", ""]
    example_idx = 1

    for example, _score in vuln_refs:
        title = format_example_title(example, example_idx, is_benign=False)
        lines.append(title)
        lines.append("```c")
        lines.append(example.get("code", "")[:1500])
        lines.append("```")
        lines.append("")
        example_idx += 1

    for example, _score in benign_refs:
        title = format_example_title(example, example_idx, is_benign=True)
        lines.append(title)
        lines.append("```c")
        lines.append(example.get("code", "")[:1500])
        lines.append("```")
        lines.append("")
        example_idx += 1

    lines.extend(
        [
            "=== Target Code to Analyze ===",
            "```c",
            query_code[:3500],
            "```",
            "",
            "Analyze the target code. Is it vulnerable?",
        ]
    )
    return "\n".join(lines)


def run() -> dict:
    client = create_client()
    stats = TokenStats()
    samples = load_samples()
    kb_entries = load_kb(KB_PATH)

    print(f"Running single-pass + RAG baseline on {len(samples)} samples...")
    print(f"Loaded retrieval KB with {len(kb_entries)} entries from {KB_PATH}")

    predictions = []
    details = []
    similarity_values = []

    for i, sample in enumerate(samples):
        code = sample.get("func", "")
        gt = int(sample.get("target", 0))

        vuln_refs, benign_refs = retrieve_examples(code, kb_entries)
        prompt = build_prompt(code, vuln_refs, benign_refs)
        response = call_llm(
            client=client,
            prompt=prompt,
            stats=stats,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=256,
        )
        pred = parse_vulnerability_label(response)
        predictions.append(pred)

        retrieved = []
        for example, score in vuln_refs:
            similarity_values.append(score)
            retrieved.append(
                {
                    "id": example.get("id"),
                    "kind": "vulnerable",
                    "major_category": example.get("major_category"),
                    "cwe": example.get("cwe"),
                    "similarity": round(score, 4),
                }
            )
        for example, score in benign_refs:
            similarity_values.append(score)
            retrieved.append(
                {
                    "id": example.get("id"),
                    "kind": "benign",
                    "major_category": example.get("major_category"),
                    "cwe": example.get("cwe"),
                    "similarity": round(score, 4),
                }
            )

        details.append(
            {
                "idx": sample.get("idx"),
                "ground_truth": gt,
                "prediction": pred,
                "retrieved_examples": retrieved,
                "response": response,
            }
        )

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] calls={stats.total_calls}, tokens={stats.total_tokens}")

    metrics = compute_metrics(predictions, [int(s.get("target", 0)) for s in samples])
    cost_summary = stats.summary(len(samples))
    avg_sim = sum(similarity_values) / len(similarity_values) if similarity_values else 0.0

    results = {
        "method": "Single-pass + RAG",
        "model": "from_env",
        "n_samples": len(samples),
        "metrics": metrics,
        "cost": cost_summary,
        "retrieval_enabled": True,
        "retrieval": {
            "kb_path": str(KB_PATH),
            "kb_entries": len(kb_entries),
            "top_k_vulnerable": 3,
            "top_k_benign": 1,
            "avg_retrieved_similarity": round(avg_sim, 4),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "singlepass_rag_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "singlepass_rag_details.jsonl", "w") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n=== Single-pass + RAG Results ===")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {OUT_DIR}")
    return results


if __name__ == "__main__":
    run()
