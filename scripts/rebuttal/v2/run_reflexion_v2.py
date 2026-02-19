"""Reflexion v2: Iterative self-correction for CWE category classification."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from llm_utils_v2 import (
    TokenStats, call_llm_multi_turn, compute_multiclass_metrics, create_client,
    load_samples, parse_category, sample_ground_truth,
    CATEGORIES_FOR_PROMPT, OUT_DIR,
)

ACTOR_SYS = """You are a security code analyst. Classify the vulnerability type of the given code.

""" + CATEGORIES_FOR_PROMPT + """

Output format: <category name>
If the code is safe, output: Benign"""

CRITIC_SYS = """You are a senior security reviewer. Critically evaluate a vulnerability classification.

Review the code AND the initial classification. Consider:
1. Is the category correct?
2. Are there missed vulnerabilities in a different category?
3. Should this actually be Benign?

Provide specific critique."""


def run():
    client = create_client()
    stats = TokenStats()
    samples = load_samples()
    print(f"Reflexion v2 on {len(samples)} samples (CWE category classification)...")

    predictions, details = [], []
    for i, s in enumerate(samples):
        code = s["func"][:3000]
        gt = sample_ground_truth(s)

        msgs = [{"role": "system", "content": ACTOR_SYS}]

        # Turn 0: Actor
        msgs.append({"role": "user", "content":
            f"Classify vulnerability type:\n\n```c\n{code}\n```\n\nOutput ONLY the category name."})
        actor = call_llm_multi_turn(client, msgs, stats, max_tokens=128)
        msgs.append({"role": "assistant", "content": actor})

        # Turn 1: Critic
        msgs.append({"role": "user", "content":
            f"Critique your classification '{actor}'. Is it correct? "
            f"Consider the actual code logic carefully. Point out errors."})
        critic = call_llm_multi_turn(client, msgs, stats, max_tokens=384)
        msgs.append({"role": "assistant", "content": critic})

        # Turn 2: Refinement
        msgs.append({"role": "user", "content":
            f"Based on your critique, give your FINAL classification. "
            f"Output ONLY the category name from the list."})
        refined = call_llm_multi_turn(client, msgs, stats, max_tokens=64)

        pred = parse_category(refined)
        predictions.append(pred)
        details.append({"idx": s.get("idx"), "gt": gt, "pred": pred,
                        "actor": actor, "critic": critic, "refined": refined})
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}] calls={stats.total_calls} tokens={stats.total_tokens}")

    gts = [sample_ground_truth(s) for s in samples]
    metrics = compute_multiclass_metrics(predictions, gts)
    results = {"method": "Reflexion (Self-Correction)", "metrics": metrics,
               "cost": stats.summary(len(samples))}
    print("\n=== Reflexion v2 ===")
    print(json.dumps({k: v for k, v in results.items() if k != "metrics"}, indent=2))
    print(f"Macro-F1: {metrics['macro_f1']:.4f}  Accuracy: {metrics['accuracy']:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "reflexion_v2_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "reflexion_v2_details.jsonl", "w") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    run()
