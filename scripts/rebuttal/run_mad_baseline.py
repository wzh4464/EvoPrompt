"""Exp 2 - Multi-Agent Debate (MAD) baseline: Attacker vs Defender → Judge.

Implements Multi-Agent Debate (Liang et al., 2023) for vulnerability detection:
  Round 1:
    Agent A (Security Auditor):  Aggressively find vulnerabilities
    Agent B (Developer):         Defend the code, argue it's safe
  Round 2:
    Agent A: Counter-argue against B's defense
    Agent B: Final rebuttal
  Verdict:
    Agent C (Judge): Read full debate, make final ruling
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

AUDITOR_SYSTEM = """You are an aggressive security auditor. Your job is to find vulnerabilities in code.
Err on the side of caution - if there's any possibility of a vulnerability, flag it.
Be specific about what CWE type it could be and where in the code the issue exists."""

DEVELOPER_SYSTEM = """You are an experienced software developer who wrote this code.
Your job is to defend the code against security accusations.
Point out why the flagged issues are not real vulnerabilities, or explain the safety mechanisms in place.
Be specific and reference the actual code logic."""

JUDGE_SYSTEM = """You are a Chief Security Officer making the final ruling on a security review.
You have read a debate between a Security Auditor (who found potential vulnerabilities) and
a Developer (who defended the code).

Based on the evidence from both sides, make your FINAL ruling:
- If the code IS vulnerable, respond: Vulnerable - [CWE-ID] [brief explanation]
- If the code is NOT vulnerable, respond: Benign - no confirmed vulnerability

Consider the strength of arguments from both sides. Be fair and evidence-based."""


def run():
    client = create_client()
    stats = TokenStats()
    samples = load_samples()

    print(f"Running Multi-Agent Debate baseline on {len(samples)} samples...")
    predictions = []
    details = []

    for i, sample in enumerate(samples):
        code = sample["func"]
        gt = sample["target"]
        code_snippet = code[:3000]

        # === Round 1 ===
        # Agent A (Auditor): Find vulnerabilities
        auditor_r1_prompt = (
            f"Perform a thorough security audit of this code. "
            f"Identify all potential vulnerabilities:\n\n```c\n{code_snippet}\n```"
        )
        auditor_r1 = call_llm(client, auditor_r1_prompt, stats,
                              system_prompt=AUDITOR_SYSTEM, max_tokens=384)

        # Agent B (Developer): Defend against auditor's findings
        developer_r1_prompt = (
            f"A security auditor has flagged the following concerns about your code:\n\n"
            f"**Auditor's findings:**\n{auditor_r1}\n\n"
            f"**Your code:**\n```c\n{code_snippet}\n```\n\n"
            f"Respond to each concern. Explain why the code is safe or acknowledge genuine issues."
        )
        developer_r1 = call_llm(client, developer_r1_prompt, stats,
                                system_prompt=DEVELOPER_SYSTEM, max_tokens=384)

        # === Round 2 ===
        # Agent A: Counter-argue
        auditor_r2_prompt = (
            f"The developer responded to your audit:\n\n"
            f"**Developer's defense:**\n{developer_r1}\n\n"
            f"**Original code:**\n```c\n{code_snippet}\n```\n\n"
            f"Counter-argue if you still believe there are vulnerabilities. "
            f"Provide additional evidence or concede points that were well-defended."
        )
        auditor_r2 = call_llm(client, auditor_r2_prompt, stats,
                              system_prompt=AUDITOR_SYSTEM, max_tokens=384)

        # Agent B: Final rebuttal (optional but adds cost for comparison)
        developer_r2_prompt = (
            f"The auditor has responded again:\n\n"
            f"**Auditor's counter-argument:**\n{auditor_r2}\n\n"
            f"Provide your final defense."
        )
        developer_r2 = call_llm(client, developer_r2_prompt, stats,
                                system_prompt=DEVELOPER_SYSTEM, max_tokens=256)

        # === Verdict ===
        # Agent C (Judge): Make final ruling
        judge_prompt = (
            f"Review this complete security debate and make your final ruling.\n\n"
            f"**Code under review:**\n```c\n{code_snippet}\n```\n\n"
            f"**== Round 1 ==**\n"
            f"**Auditor:** {auditor_r1}\n\n"
            f"**Developer:** {developer_r1}\n\n"
            f"**== Round 2 ==**\n"
            f"**Auditor:** {auditor_r2}\n\n"
            f"**Developer:** {developer_r2}\n\n"
            f"**Your final ruling:**"
        )
        judge_resp = call_llm(client, judge_prompt, stats,
                              system_prompt=JUDGE_SYSTEM, max_tokens=256)

        pred = parse_vulnerability_label(judge_resp)
        predictions.append(pred)
        details.append({
            "idx": sample.get("idx"),
            "ground_truth": gt,
            "prediction": pred,
            "n_api_calls": 5,
            "auditor_r1": auditor_r1,
            "developer_r1": developer_r1,
            "auditor_r2": auditor_r2,
            "developer_r2": developer_r2,
            "judge": judge_resp,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] calls={stats.total_calls}, tokens={stats.total_tokens}")

    # Compute metrics
    ground_truths = [s["target"] for s in samples]
    metrics = compute_metrics(predictions, ground_truths)
    cost_summary = stats.summary(len(samples))

    results = {
        "method": "Multi-Agent Debate (MAD)",
        "model": "from_env",
        "n_samples": len(samples),
        "metrics": metrics,
        "cost": cost_summary,
    }

    print("\n=== Multi-Agent Debate Results ===")
    print(json.dumps(results, indent=2))

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "mad_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "mad_details.jsonl", "w") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\nSaved to {OUT_DIR}")
    return results


if __name__ == "__main__":
    run()
