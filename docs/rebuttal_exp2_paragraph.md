# Rebuttal Paragraph — Exp 2: Agent Comparison & Cost Analysis

## Response to Reviewer rvhT (W1: "Lack of comparison with agent-based approaches") and Reviewers UsX5 & 4xR8 (cost-effectiveness concerns)

We thank all reviewers for their constructive feedback. To address the lack of agent-based comparisons and the absence of cost analysis, we conducted new experiments comparing MulVul against two representative agentic paradigms: **Reflexion** (Shinn et al., NeurIPS 2023), which employs iterative self-correction through Actor→Critic→Refinement loops, and **Multi-Agent Debate** (MAD; Liang et al., 2023), which uses adversarial multi-role dialogue (Auditor vs. Developer → Judge).

We evaluated all three methods on 150 stratified samples from the PrimeVul test set using the same backbone LLM to ensure a fair architectural comparison. Results are summarized below:

| Method | Macro-F1 | Vuln Recall | Avg Tokens/Sample | Relative Cost |
|--------|----------|-------------|-------------------|---------------|
| **MulVul (Ours)** | **58.8%** | **83.7%** | **1,631** | **1.0×** |
| Reflexion | 50.8% | 32.6% | 4,026 | 2.5× |
| Multi-Agent Debate | 50.3% | 11.6% | 5,915 | 3.6× |

MulVul achieves the highest Macro-F1 (+8.0% over Reflexion, +8.5% over MAD) while consuming significantly fewer tokens — only 40% of Reflexion's cost and 28% of MAD's. Notably, MulVul attains 83.7% vulnerability recall, compared to 32.6% for Reflexion and 11.6% for MAD. This is because MulVul's coarse-to-fine routing directs the LLM's attention to category-specific vulnerability patterns in a single forward pass, whereas Reflexion's iterative self-critique tends to "second-guess" correct initial detections, and MAD's adversarial debate drives the Judge toward conservative (benign) predictions due to the Developer agent's persuasive counter-arguments.

From a cost perspective, although MulVul invokes multiple agents (1 Router + k Detectors), its single-pass architecture avoids the compounding token overhead of multi-turn interactions. Reflexion accumulates context across 3 turns of self-dialogue, while MAD maintains full debate transcripts across 5 calls per sample, leading to 2.5× and 3.6× higher token consumption respectively. These results demonstrate that MulVul represents the **Pareto-optimal** solution: achieving the best detection performance at the lowest computational cost among all agentic approaches evaluated.
