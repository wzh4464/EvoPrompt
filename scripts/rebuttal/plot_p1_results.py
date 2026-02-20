#!/usr/bin/env python3
"""Plot P1 rebuttal results for single-pass no-RAG vs +RAG (+ optional MulVul + RAG)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "outputs" / "rebuttal" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RESULT_SPECS = [
    (
        "singlepass_norag",
        PROJECT_ROOT / "outputs" / "rebuttal" / "exp1_singlepass" / "singlepass_norag_results.json",
        "Single-pass (no RAG)",
        "#2563EB",
        "o",
    ),
    (
        "singlepass_rag",
        PROJECT_ROOT / "outputs" / "rebuttal" / "exp1_singlepass" / "singlepass_rag_results.json",
        "Single-pass + RAG",
        "#059669",
        "o",
    ),
    (
        "mulvul_norag_evolved",
        PROJECT_ROOT / "outputs" / "rebuttal" / "exp2_agent_comparison" / "mulvul_evolved_results.json",
        "MulVul (evolved, no RAG)",
        "#D97706",
        "*",
    ),
    (
        "mulvul_norag",
        PROJECT_ROOT / "outputs" / "rebuttal" / "exp2_agent_comparison" / "mulvul_results.json",
        "MulVul (no RAG)",
        "#D97706",
        "*",
    ),
    (
        "mulvul_rag",
        PROJECT_ROOT / "outputs" / "rebuttal" / "exp1_singlepass" / "mulvul_rag_results.json",
        "MulVul + RAG",
        "#7C3AED",
        "*",
    ),
]


def load_methods() -> dict:
    methods = {}
    for key, path, label, color, marker in RESULT_SPECS:
        if path.exists():
            methods[key] = {
                "path": path,
                "label": label,
                "color": color,
                "marker": marker,
                "data": json.loads(path.read_text()),
            }
    # Prefer evolved MulVul if present; keep one no-RAG MulVul point in default table/plot.
    if "mulvul_norag_evolved" in methods and "mulvul_norag" in methods:
        methods.pop("mulvul_norag")
    return methods


def plot_scatter(methods: dict) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    if "singlepass_norag" in methods:
        baseline_tokens = methods["singlepass_norag"]["data"]["cost"]["avg_tokens_per_sample"]
    else:
        any_key = next(iter(methods))
        baseline_tokens = methods[any_key]["data"]["cost"]["avg_tokens_per_sample"]

    for key, spec in methods.items():
        data = spec["data"]
        x = data["cost"]["avg_tokens_per_sample"]
        y = data["metrics"]["macro_f1"] * 100
        ratio = x / baseline_tokens if baseline_tokens else 0.0

        size = 230 if spec["marker"] == "*" else 120
        ax.scatter(
            x,
            y,
            s=size,
            c=spec["color"],
            marker=spec["marker"],
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.annotate(
            f"{spec['label']}\nF1={y:.1f}%  Cost={ratio:.2f}x",
            (x, y),
            textcoords="offset points",
            xytext=(0, 14 if "rag" in key else -28),
            ha="center",
            fontsize=9,
            fontweight="bold" if key == "singlepass_rag" else "normal",
        )

    ax.set_xlabel("Avg Tokens per Sample", fontsize=12)
    ax.set_ylabel("Macro-F1 (%)", fontsize=12)
    ax.set_title("P1 Rebuttal: RAG Contribution vs Architecture", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(m["data"]["cost"]["avg_tokens_per_sample"] for m in methods.values()) * 1.2)
    ax.set_ylim(30, 80)

    plt.tight_layout()
    out_png = FIG_DIR / "p1_rag_scatter.png"
    out_pdf = FIG_DIR / "p1_rag_scatter.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    return out_png


def plot_bar(methods: dict) -> Path:
    order = [k for k, *_ in RESULT_SPECS if k in methods]
    labels = [methods[k]["label"] for k in order]
    f1_values = [methods[k]["data"]["metrics"]["macro_f1"] * 100 for k in order]
    colors = [methods[k]["color"] for k in order]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    bars = ax.bar(labels, f1_values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Macro-F1 (%)", fontsize=12)
    ax.set_title("P1 Rebuttal: Macro-F1 Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(f1_values) * 1.35)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=12)

    for bar, value in zip(bars, f1_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_png = FIG_DIR / "p1_rag_bars.png"
    out_pdf = FIG_DIR / "p1_rag_bars.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    return out_png


def print_summary_table(methods: dict) -> None:
    print("\n" + "=" * 90)
    print("P1 COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Method':<28} {'Macro-F1':>10} {'Accuracy':>10} {'Tokens/Sample':>15} {'Calls/Sample':>13}")
    print("-" * 90)
    for key, _, _, _, _ in RESULT_SPECS:
        if key not in methods:
            continue
        d = methods[key]["data"]
        label = methods[key]["label"]
        macro_f1 = d["metrics"]["macro_f1"] * 100
        acc = d["metrics"]["accuracy"] * 100
        tok = d["cost"]["avg_tokens_per_sample"]
        calls = d["cost"]["avg_calls_per_sample"]
        print(f"{label:<28} {macro_f1:>9.1f}% {acc:>9.1f}% {tok:>15,.1f} {calls:>13.2f}")
    print("=" * 90)

    if "singlepass_norag" in methods and "singlepass_rag" in methods:
        f1_no = methods["singlepass_norag"]["data"]["metrics"]["macro_f1"]
        f1_rag = methods["singlepass_rag"]["data"]["metrics"]["macro_f1"]
        delta = (f1_rag - f1_no) * 100
        print(f"RAG contribution (Single-pass): {delta:+.2f} Macro-F1 points")

    mulvul_key = "mulvul_norag_evolved" if "mulvul_norag_evolved" in methods else "mulvul_norag"
    if "singlepass_rag" in methods and mulvul_key in methods:
        f1_sp_rag = methods["singlepass_rag"]["data"]["metrics"]["macro_f1"]
        f1_mulvul = methods[mulvul_key]["data"]["metrics"]["macro_f1"]
        delta = (f1_mulvul - f1_sp_rag) * 100
        print(f"Architecture contribution (MulVul no-RAG - Single-pass + RAG): {delta:+.2f} points")


def main() -> None:
    methods = load_methods()
    if len(methods) < 2:
        raise RuntimeError(
            "Need at least two result files to plot. Expected files under "
            "outputs/rebuttal/exp1_singlepass and outputs/rebuttal/exp2_agent_comparison."
        )

    scatter_path = plot_scatter(methods)
    bar_path = plot_bar(methods)
    print_summary_table(methods)
    print(f"\nSaved scatter: {scatter_path}")
    print(f"Saved bar chart: {bar_path}")


if __name__ == "__main__":
    main()
