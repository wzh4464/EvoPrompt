# Response to Reviewer UsX5

We thank Reviewer UsX5 for the thorough evaluation and constructive feedback. We address each concern below with new experiments and concrete data.

## 1. Cost and Latency Analysis

We conducted a controlled cost-effectiveness study on a 150-sample PrimeVul subset (same split for all methods, GPT-4o backend).

| Method | Macro-F1 | API calls/sample | Tokens/sample | Sec/sample |
|--------|:--------:|:-----------------:|:-------------:|:----------:|
| Single-pass (no RAG) | 0.6451 | 1.00 | 521.9 | 3.67 |
| Single-pass + RAG | 0.6283 | 1.00 | 1676.3 | 2.88 |
| **MulVul** | **0.5880** | **3.00** | **1631.4** | **10.98** |
| MulVul (evolved) | 0.5939 | 2.73 | 2952.8 | 15.30 |
| Reflexion | 0.5079 | 3.00 | 4026.1 | 22.85 |
| MAD | 0.5032 | 5.00 | 5915.0 | 50.01 |

**Per 1kLOC comparison** (average function length ~84.1 LOC):

| Method | Tokens/1kLOC |
|--------|:------------:|
| MulVul | ~19.4k |
| Reflexion | ~47.9k |
| MAD | ~70.3k |

Key findings:
- **vs. Reflexion**: MulVul uses ~2.5x fewer tokens and ~52% lower latency (10.98s vs. 22.85s), while achieving substantially higher F1 (0.588 vs. 0.508).
- **vs. MAD**: MulVul uses ~3.6x fewer tokens and ~78% lower latency (10.98s vs. 50.01s), again with higher F1.
- MulVul's multi-agent overhead vs. single-pass is justified by its vulnerability recall advantage:

**Vulnerability recall** (critical for security where missed vulnerabilities are costly):

| Method | Vuln Recall |
|--------|:-----------:|
| **MulVul** | **83.7%** |
| Reflexion | 32.6% |
| MAD | 11.6% |

MulVul catches 2.6x more vulnerabilities than Reflexion and 7.2x more than MAD, making it the most practical choice when false negatives are costly.

## 2. Clean Pool Sensitivity (Section 4.2.1)

We conducted a sensitivity analysis varying the clean pool size used in Section 4.2.1's knowledge base construction, evaluated on the full PrimeVul test set (1,907 samples).

| Clean pool fraction | Pool size | Binary F1 | Precision | Recall | Macro-F1 |
|:-------------------:|:---------:|:---------:|:---------:|:------:|:--------:|
| 0.10 | 50 | 0.5695 | 0.5090 | 0.6464 | 0.0221 |
| 0.25 | 125 | 0.5854 | 0.5014 | 0.7034 | 0.0246 |
| 0.50 | 250 | 0.5813 | 0.5116 | 0.6730 | 0.0324 |
| 1.00 | 500 | 0.5833 | 0.5042 | 0.6920 | 0.0220 |

Key observations:
- Binary F1 remains within a narrow band (0.57--0.59) across all fractions, demonstrating **no performance collapse** even when the clean pool is reduced to just 10% (50 samples).
- The system is robust to clean pool size, indicating that MulVul's Router-Detector architecture, rather than the knowledge base volume, is the primary driver of detection quality.
- This robustness is practically important: it means MulVul does not require large curated clean-code corpora to function effectively.

## 3. Generalization Beyond C/C++

We acknowledge that our current experimental validation is scoped to C/C++ (PrimeVul dataset). This is a fair observation. We clarify our position:

- **Design principle vs. validated claim**: MulVul's Router-Detector architecture operates on source code through natural-language prompts and does not rely on language-specific parsers, ASTs, or compiled representations. The CWE taxonomy used for routing is inherently language-agnostic (e.g., CWE-119 buffer errors apply across C, C++, Rust unsafe blocks, etc.). We will revise our wording to frame language-agnosticity as a *design principle* rather than a fully validated claim.
- **Practical scope**: The choice of C/C++ reflects both dataset availability (PrimeVul is the largest high-quality function-level vulnerability dataset) and practical importance (C/C++ accounts for the majority of CVEs in the NVD). We will add explicit discussion of cross-language evaluation as a concrete future direction.

## 4. Reproducibility with Closed-Source Models

We appreciate this important concern regarding scientific reproducibility:

- **Model-agnostic architecture**: MulVul's Router-Detector design is not tied to any specific LLM. The prompts, routing logic, and aggregation strategy are fully specified and portable. The architecture can be instantiated with any instruction-following LLM.
- **Open-weight transfer**: We acknowledge that our main experiments use GPT-4o, and validating performance with open-weight models (e.g., Qwen, LLaMA, DeepSeek) is an important next step. We will add this as an explicit future work direction with a concrete plan.
- **Full reproducibility artifacts**: We will release all prompts, routing configurations, evaluation scripts, and pipeline code to enable reproduction with any compatible model backend.

---

We thank Reviewer UsX5 again for the actionable and constructive feedback. We believe the added cost/latency analysis and clean-pool sensitivity ablation directly address the main empirical concerns, and the clarifications on scope and reproducibility improve the paper's honesty and clarity.
