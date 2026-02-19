"""Generate comparison plots for v2 fine-grained CWE classification experiments."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "rebuttal" / "exp2_v2"
FIG_DIR = Path(__file__).resolve().parents[3] / "outputs" / "rebuttal" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

methods = {}
for name in ["mulvul", "reflexion", "mad"]:
    p = OUT_DIR / f"{name}_v2_results.json"
    if p.exists():
        methods[name] = json.loads(p.read_text())

labels = {"mulvul": "MulVul (Ours)", "reflexion": "Reflexion", "mad": "Multi-Agent Debate"}
colors = {"mulvul": "#2563EB", "reflexion": "#DC2626", "mad": "#D97706"}
markers = {"mulvul": "*", "reflexion": "o", "mad": "s"}
sizes = {"mulvul": 300, "reflexion": 140, "mad": 140}

mulvul_tok = methods["mulvul"]["cost"]["avg_tokens_per_sample"]

# ── Figure 1: Cost vs Macro-F1 ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
for name, data in methods.items():
    x = data["cost"]["avg_tokens_per_sample"]
    y = data["metrics"]["macro_f1"] * 100
    ratio = x / mulvul_tok
    ax.scatter(x, y, s=sizes[name], c=colors[name], marker=markers[name],
               edgecolors="black", linewidth=0.5, zorder=5)
    oy = 12 if name == "mulvul" else (-18 if name == "reflexion" else 12)
    ax.annotate(f"{labels[name]}\nF1={y:.1f}%  Cost={ratio:.1f}x",
                (x, y), textcoords="offset points", xytext=(0, oy),
                fontsize=9, ha="center",
                fontweight="bold" if name == "mulvul" else "normal")

ax.set_xlabel("Avg Tokens per Sample", fontsize=11)
ax.set_ylabel("Macro-F1 (%, 14 CWE categories)", fontsize=11)
ax.set_title("(a) Cost vs. Category Classification (Macro-F1)", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(d["cost"]["avg_tokens_per_sample"] for d in methods.values()) * 1.25)

# ── Figure 2: Cost vs Vuln Recall ──
ax = axes[1]
for name, data in methods.items():
    x = data["cost"]["avg_tokens_per_sample"]
    y = data["metrics"]["binary_vuln_recall"] * 100
    ratio = x / mulvul_tok
    ax.scatter(x, y, s=sizes[name], c=colors[name], marker=markers[name],
               edgecolors="black", linewidth=0.5, zorder=5)
    oy = -20 if name == "mulvul" else (12 if name == "reflexion" else -20)
    ax.annotate(f"{labels[name]}\nRecall={y:.1f}%  Cost={ratio:.1f}x",
                (x, y), textcoords="offset points", xytext=(0, oy),
                fontsize=9, ha="center",
                fontweight="bold" if name == "mulvul" else "normal")

ax.set_xlabel("Avg Tokens per Sample", fontsize=11)
ax.set_ylabel("Vulnerability Recall (%)", fontsize=11)
ax.set_title("(b) Cost vs. Vulnerability Detection Recall", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(d["cost"]["avg_tokens_per_sample"] for d in methods.values()) * 1.25)

plt.tight_layout()
fig.savefig(FIG_DIR / "v2_cost_effectiveness.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "v2_cost_effectiveness.png", dpi=200, bbox_inches="tight")
print(f"Saved figures to {FIG_DIR}")

# ── Print comparison table ──
print("\n" + "=" * 95)
print("V2 COMPARISON TABLE — 14-class CWE Category Classification (200 samples)")
print("=" * 95)
print(f"{'Method':<25} {'Macro-F1':>10} {'Accuracy':>10} {'Vuln-Recall':>12} {'Tok/Sample':>12} {'Cost':>8}")
print("-" * 95)
for name in ["mulvul", "reflexion", "mad"]:
    d = methods[name]
    f1 = d["metrics"]["macro_f1"] * 100
    acc = d["metrics"]["accuracy"] * 100
    rec = d["metrics"]["binary_vuln_recall"] * 100
    tok = d["cost"]["avg_tokens_per_sample"]
    ratio = tok / mulvul_tok
    print(f"{labels[name]:<25} {f1:>9.1f}% {acc:>9.1f}% {rec:>11.1f}% {tok:>12,.0f} {ratio:>7.1f}x")
print("=" * 95)
