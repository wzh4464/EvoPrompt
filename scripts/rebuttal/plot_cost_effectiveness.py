"""Generate cost-effectiveness scatter plot for Exp 2 rebuttal."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "rebuttal" / "exp2_agent_comparison"
FIG_DIR = Path(__file__).resolve().parents[2] / "outputs" / "rebuttal" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load results
methods = {}
for name in ["mulvul", "reflexion", "mad"]:
    p = OUT_DIR / f"{name}_results.json"
    if p.exists():
        methods[name] = json.loads(p.read_text())

# Extract data
labels = {
    "mulvul": "MulVul (Ours)",
    "reflexion": "Reflexion",
    "mad": "Multi-Agent Debate",
}
colors = {
    "mulvul": "#2563EB",
    "reflexion": "#DC2626",
    "mad": "#D97706",
}
markers = {
    "mulvul": "★",
    "reflexion": "o",
    "mad": "s",
}

fig, ax = plt.subplots(1, 1, figsize=(7, 5))

mulvul_tokens = methods["mulvul"]["cost"]["avg_tokens_per_sample"]

offsets = {
    "mulvul": (0, 18),
    "reflexion": (0, -28),
    "mad": (0, 18),
}

for name, data in methods.items():
    x = data["cost"]["avg_tokens_per_sample"]
    y = data["metrics"]["macro_f1"] * 100
    ratio = x / mulvul_tokens
    marker_map = {"mulvul": "*", "reflexion": "o", "mad": "s"}
    size_map = {"mulvul": 300, "reflexion": 140, "mad": 140}

    ax.scatter(x, y, s=size_map[name], c=colors[name], marker=marker_map[name],
               zorder=5, edgecolors="black", linewidth=0.5, label=labels[name])

    ox, oy = offsets[name]
    cost_label = "1.0x" if ratio == 1.0 else f"{ratio:.1f}x"
    ax.annotate(
        f"{labels[name]}\nF1={y:.1f}%  Cost={cost_label}",
        (x, y),
        textcoords="offset points",
        xytext=(ox, oy),
        fontsize=9,
        ha="center",
        fontweight="bold" if name == "mulvul" else "normal",
    )

ax.set_xlabel("Avg Tokens per Sample", fontsize=12)
ax.set_ylabel("Macro-F1 (%)", fontsize=12)
ax.set_title("Cost-Effectiveness: MulVul vs. Agentic Baselines", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(d["cost"]["avg_tokens_per_sample"] for d in methods.values()) * 1.2)
ax.set_ylim(35, 75)

# Add Pareto annotation
ax.annotate(
    "Pareto Optimal\n(Higher F1, Lower Cost)",
    xy=(0.15, 0.85), xycoords="axes fraction",
    fontsize=9, fontstyle="italic", color="green",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3),
)

plt.tight_layout()
fig.savefig(FIG_DIR / "cost_effectiveness_scatter.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "cost_effectiveness_scatter.png", dpi=200, bbox_inches="tight")
print(f"Saved to {FIG_DIR / 'cost_effectiveness_scatter.pdf'}")

# Also print comparison table
print("\n" + "=" * 80)
print("COMPARISON TABLE FOR REBUTTAL")
print("=" * 80)
print(f"{'Method':<30} {'Macro-F1':>10} {'Vuln-Recall':>12} {'Tokens/Sample':>14} {'Cost Ratio':>11}")
print("-" * 80)
for name in ["mulvul", "reflexion", "mad"]:
    d = methods[name]
    f1 = d["metrics"]["macro_f1"] * 100
    recall = d["metrics"]["recall_vuln"] * 100
    tokens = d["cost"]["avg_tokens_per_sample"]
    ratio = tokens / mulvul_tokens
    print(f"{labels[name]:<30} {f1:>9.1f}% {recall:>11.1f}% {tokens:>14,.0f} {ratio:>10.1f}x")
print("=" * 80)
