"""Exp 2 - Reflexion baseline: Iterative self-correction (Actor → Critic → Refinement).

Implements Reflexion (Shinn et al., NeurIPS 2023) pattern for vulnerability detection:
  Turn 0 (Actor):      Initial vulnerability prediction
  Turn 1 (Critic):     Self-critique of the prediction
  Turn 2 (Refinement): Revised prediction based on critique
  [Optional Turn 3]:   Second round if still uncertain
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "rebuttal"))
from llm_utils import (
    TokenStats, call_llm_multi_turn, compute_metrics, create_client,
    load_samples, parse_vulnerability_label,
)

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "rebuttal" / "exp2_agent_comparison"

ACTOR_SYSTEM = """You are a security code analyst. Analyze the given code for potential vulnerabilities.

If the code contains a vulnerability, respond with:
  Vulnerable - [CWE-ID] [brief explanation]

If the code is safe, respond with:
  Benign - no vulnerability detected.

Be thorough but concise."""

CRITIC_SYSTEM = """You are a senior security reviewer. Your task is to critically evaluate a vulnerability assessment.

Review the original code AND the initial assessment. Consider:
1. Are there any missed vulnerabilities?
2. Are there any false positives (flagging safe code as vulnerable)?
3. Is the CWE classification correct?

Provide your critique, pointing out specific issues or confirming the assessment is correct."""

MAX_ROUNDS = 3  # Actor + Critic + Refinement (possibly one more round)


def run():
    client = create_client()
    stats = TokenStats()
    samples = load_samples()

    print(f"Running Reflexion baseline on {len(samples)} samples...")
    predictions = []
    details = []

    for i, sample in enumerate(samples):
        code = sample["func"]
        gt = sample["target"]
        code_snippet = code[:3000]

        messages = [{"role": "system", "content": ACTOR_SYSTEM}]

        # Turn 0: Actor - initial prediction
        actor_prompt = f"Analyze this code for vulnerabilities:\n\n```c\n{code_snippet}\n```"
        messages.append({"role": "user", "content": actor_prompt})
        actor_resp = call_llm_multi_turn(client, messages, stats, max_tokens=384)
        messages.append({"role": "assistant", "content": actor_resp})

        # Turn 1: Critic - evaluate the prediction
        critic_prompt = (
            f"Now critically review your assessment above.\n\n"
            f"Original code:\n```c\n{code_snippet}\n```\n\n"
            f"Your assessment: {actor_resp}\n\n"
            f"Are there any errors in this assessment? Did you miss anything? "
            f"Are there false positives? Provide your critique."
        )
        messages.append({"role": "user", "content": critic_prompt})
        critic_resp = call_llm_multi_turn(client, messages, stats, max_tokens=384)
        messages.append({"role": "assistant", "content": critic_resp})

        # Turn 2: Refinement - revise based on critique
        refine_prompt = (
            f"Based on your critique, provide your FINAL revised assessment.\n\n"
            f"If vulnerable, respond with: Vulnerable - [CWE-ID] [brief explanation]\n"
            f"If safe, respond with: Benign - no vulnerability detected.\n\n"
            f"Give ONLY the final verdict."
        )
        messages.append({"role": "user", "content": refine_prompt})
        refine_resp = call_llm_multi_turn(client, messages, stats, max_tokens=256)
        messages.append({"role": "assistant", "content": refine_resp})

        # Check if another round is needed (if response is uncertain)
        final_resp = refine_resp
        n_rounds = 3

        if "uncertain" in refine_resp.lower() or "possibly" in refine_resp.lower():
            # Turn 3: One more critique-refinement
            extra_prompt = (
                f"Your response seems uncertain. Make a definitive decision.\n"
                f"Respond with ONLY: 'Vulnerable - [reason]' or 'Benign - [reason]'"
            )
            messages.append({"role": "user", "content": extra_prompt})
            final_resp = call_llm_multi_turn(client, messages, stats, max_tokens=128)
            n_rounds = 4

        pred = parse_vulnerability_label(final_resp)
        predictions.append(pred)
        details.append({
            "idx": sample.get("idx"),
            "ground_truth": gt,
            "prediction": pred,
            "n_rounds": n_rounds,
            "actor_response": actor_resp,
            "critic_response": critic_resp,
            "refinement_response": refine_resp,
            "final_response": final_resp,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] calls={stats.total_calls}, tokens={stats.total_tokens}")

    # Compute metrics
    ground_truths = [s["target"] for s in samples]
    metrics = compute_metrics(predictions, ground_truths)
    cost_summary = stats.summary(len(samples))

    results = {
        "method": "Reflexion (Self-Correction)",
        "model": "from_env",
        "n_samples": len(samples),
        "metrics": metrics,
        "cost": cost_summary,
    }

    print("\n=== Reflexion Results ===")
    print(json.dumps(results, indent=2))

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "reflexion_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "reflexion_details.jsonl", "w") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\nSaved to {OUT_DIR}")
    return results


if __name__ == "__main__":
    run()
