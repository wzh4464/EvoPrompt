"""Multi-Agent Debate v2: CWE category classification via adversarial debate."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from llm_utils_v2 import (
    TokenStats, call_llm, compute_multiclass_metrics, create_client,
    load_samples, parse_category, sample_ground_truth,
    CATEGORIES_FOR_PROMPT, OUT_DIR,
)

AUDITOR_SYS = """You are an aggressive security auditor. Find vulnerabilities and classify them.

""" + CATEGORIES_FOR_PROMPT + """

Be specific about the vulnerability category. If you find a vulnerability, state the category.
If the code appears safe, say "Benign"."""

DEVELOPER_SYS = """You are the code developer. Defend your code against security accusations.
Argue why the flagged issues are not real vulnerabilities, or correct the classification
if the auditor picked the wrong category."""

JUDGE_SYS = """You are a Chief Security Officer making the final ruling.
Based on the security debate, classify this code into exactly ONE category.

""" + CATEGORIES_FOR_PROMPT + """

Output ONLY the category name. Nothing else."""


def run():
    client = create_client()
    stats = TokenStats()
    samples = load_samples()
    print(f"MAD v2 on {len(samples)} samples (CWE category classification)...")

    predictions, details = [], []
    for i, s in enumerate(samples):
        code = s["func"][:3000]
        gt = sample_ground_truth(s)

        # R1: Auditor
        a1 = call_llm(client,
            f"Audit this code and classify its vulnerability type:\n\n```c\n{code}\n```",
            stats, system_prompt=AUDITOR_SYS, max_tokens=384)

        # R1: Developer
        d1 = call_llm(client,
            f"Auditor's finding:\n{a1}\n\nYour code:\n```c\n{code}\n```\n\n"
            f"Defend the code or correct the classification.",
            stats, system_prompt=DEVELOPER_SYS, max_tokens=384)

        # R2: Auditor counter
        a2 = call_llm(client,
            f"Developer's defense:\n{d1}\n\nCode:\n```c\n{code}\n```\n\n"
            f"Counter-argue or concede.",
            stats, system_prompt=AUDITOR_SYS, max_tokens=256)

        # R2: Developer rebuttal
        d2 = call_llm(client,
            f"Auditor's counter:\n{a2}\n\nFinal defense.",
            stats, system_prompt=DEVELOPER_SYS, max_tokens=256)

        # Judge
        judge = call_llm(client,
            f"Code:\n```c\n{code}\n```\n\n"
            f"Auditor R1: {a1}\nDeveloper R1: {d1}\n"
            f"Auditor R2: {a2}\nDeveloper R2: {d2}\n\n"
            f"Final classification (output ONLY the category name):",
            stats, system_prompt=JUDGE_SYS, max_tokens=64)

        pred = parse_category(judge)
        predictions.append(pred)
        details.append({"idx": s.get("idx"), "gt": gt, "pred": pred,
                        "a1": a1, "d1": d1, "a2": a2, "d2": d2, "judge": judge})
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}] calls={stats.total_calls} tokens={stats.total_tokens}")

    gts = [sample_ground_truth(s) for s in samples]
    metrics = compute_multiclass_metrics(predictions, gts)
    results = {"method": "Multi-Agent Debate (MAD)", "metrics": metrics,
               "cost": stats.summary(len(samples))}
    print("\n=== MAD v2 ===")
    print(json.dumps({k: v for k, v in results.items() if k != "metrics"}, indent=2))
    print(f"Macro-F1: {metrics['macro_f1']:.4f}  Accuracy: {metrics['accuracy']:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "mad_v2_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "mad_v2_details.jsonl", "w") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    run()
