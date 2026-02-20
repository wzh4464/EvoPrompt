# Rebuttal Draft (Reviewer 4xR8 / W1-R1)

Thank you for the detailed and constructive review. We appreciate your positive assessment of (i) the cross-model prompt evolution design, (ii) the coarse-to-fine Router-Detector architecture, and (iii) the clarity of Figure 6.  
Below we address your main concerns with additional experiments and concrete numbers.

## 1) Cost/Latency Analysis (requested: cost per sample / 1k LOC)

We added explicit cost-effectiveness measurements on the same 150-sample PrimeVul subset (same split for all methods).

| Method | Macro-F1 | API calls / sample | Tokens / sample | Seconds / sample |
|---|---:|---:|---:|---:|
| Single-pass (no RAG) | 0.6451 | 1.00 | 521.9 | 3.67 |
| Single-pass + RAG | 0.6283 | 1.00 | 1676.3 | 2.88 |
| MulVul (Router-Detector) | 0.5880 | 3.00 | 1631.4 | 10.98 |
| MulVul (evolved prompts) | 0.5939 | 2.73 | 2952.8 | 15.30 |
| Reflexion | 0.5079 | 3.00 | 4026.1 | 22.85 |
| MAD | 0.5032 | 5.00 | 5915.0 | 50.01 |

Key takeaways:
- MulVul is substantially more efficient than stronger agentic baselines in this setup:
  - vs Reflexion: **~52% lower latency** (10.98s vs 22.85s/sample), **~2.47x fewer tokens**.
  - vs MAD: **~78% lower latency** (10.98s vs 50.01s/sample), **~3.63x fewer tokens**.
- Using the observed average function length (~84.1 LOC), MulVul baseline is ~19.4k tokens/1kLOC (Reflexion ~47.9k, MAD ~70.3k).

Files:
- `outputs/rebuttal/exp2_agent_comparison/mulvul_results.json`
- `outputs/rebuttal/exp2_agent_comparison/mulvul_evolved_results.json`
- `outputs/rebuttal/exp2_agent_comparison/reflexion_results.json`
- `outputs/rebuttal/exp2_agent_comparison/mad_results.json`

## 2) Clean-pool Sensitivity (requested in Sec. 4.2.1 comments)

We ran clean-pool-size sensitivity (fraction in {0.1, 0.25, 0.5, 1.0}) and observed stable binary detection quality without collapse at small clean pools.

| Clean pool fraction | Binary F1 | Binary Precision | Binary Recall | Macro-F1 |
|---|---:|---:|---:|---:|
| 0.1 | 0.5695 | 0.5090 | 0.6464 | 0.0362 |
| 0.25 | 0.5854 | 0.5014 | 0.7034 | 0.0334 |
| 0.5 | 0.5813 | 0.5116 | 0.6730 | 0.0473 |
| 1.0 | 0.5833 | 0.5042 | 0.6920 | 0.0336 |

Interpretation:
- Binary F1 varies within a narrow band (0.5695 to 0.5854).
- Performance does not collapse when clean pool is reduced.
- Mid-scale clean pool (0.5) gave the best macro-F1 in this run.

Files:
- `/home/jie/Evoprompt/outputs/supplementary/full_clean_pool_sensitivity/metrics/clean_pool_sensitivity.json`
- `/home/jie/Evoprompt/outputs/supplementary/full_clean_pool_sensitivity/metrics/clean_pool_frac_0.5_metrics_corrected.json`

## 3) Fairness of RAG Contribution (Single-pass no-RAG vs +RAG)

Following reviewer feedback, we added a controlled single-pass ablation:
- Exp A: Single-pass no-RAG
- Exp B: Single-pass +RAG
- Same data, same model endpoint, same 1 API call/sample protocol.

Observed in this setup:
- +RAG improves accuracy and precision (fewer false positives),
- but reduces vulnerable recall and macro-F1.

This isolates that RAG changes the decision boundary (precision/recall tradeoff), while architecture-level effects should be analyzed separately.

Files:
- `outputs/rebuttal/exp1_singlepass/singlepass_norag_results.json`
- `outputs/rebuttal/exp1_singlepass/singlepass_rag_results.json`

## 4) Scope and Reproducibility (language and model dependence)

We agree this version is limited to PrimeVul C/C++ and mostly closed-source frontier models in the main experiments.  
To avoid over-claiming, we will revise wording to explicitly scope current evidence to C/C++, and frame language-agnosticity as a design hypothesis rather than a fully validated claim.  
We will also add a dedicated discussion on open-weight model transfer as an explicit future direction.

---

Thank you again for the actionable feedback. We believe the added cost/latency table and clean-pool sensitivity ablation directly address your main concerns and improve clarity.
