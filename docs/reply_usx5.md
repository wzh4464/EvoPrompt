# Cost, Sensitivity, Generalization, and Reproducibility

We thank Reviewer UsX5 for the thorough evaluation and constructive feedback.

1. Cost/Latency Analysis

> "Missing quantitative cost/latency analysis... Provide cost metrics (cost per 1k LOC, seconds per sample) and discuss trade-off (is +41.5% worth 3–4x cost?)"

We measured per-sample cost across all methods on the same PrimeVul data with GPT-4o:

| Method | Macro-F1 | API Calls/Sample | Tokens/Sample | Sec/Sample | Tokens/1kLOC |
|--------|----------|------------------|---------------|------------|--------------|
| Single-pass (no RAG) | 9.23% | 1.0 | 522 | 3.67 | ~6.2k |
| Single-pass + RAG | 21.39% | 1.0 | 1,676 | 2.88 | ~19.9k |
| MulVul | 34.79% | 3.0 | 1,631 | 10.98 | ~19.4k |
| Reflexion | 27.40% | 3.0 | 4,026 | 22.85 | ~47.9k |
| MAD | -- | 5.0 | 5,915 | 50.01 | ~70.3k |

MulVul uses ~3x tokens vs. single-pass but achieves +25.56% absolute Macro-F1 improvement (9.23% -> 34.79%). Compared to other multi-agent methods, MulVul is Pareto-optimal: 2.5x fewer tokens than Reflexion (47.9k vs. 19.4k tokens/1kLOC) and 3.6x fewer than MAD, while achieving higher Macro-F1. The +41.5% improvement over the best baseline comes at the lowest per-token cost of any multi-agent approach.

2. Clean Pool Sensitivity

> "How sensitive to size of the 'clean pool'? If too small, does contrastive retrieval fail to reduce false positives?"

We varied the clean pool fraction {0.1, 0.25, 0.5, 1.0} on the full PrimeVul test set (1,907 samples):

| Clean Pool Fraction | Pool Size | Binary F1 | Recall |
|:-------------------:|:---------:|:---------:|:------:|
| 0.10 | 50 | 0.570 | 0.646 |
| 0.25 | 125 | 0.585 | 0.703 |
| 0.50 | 250 | 0.581 | 0.673 |
| 1.00 | 500 | 0.583 | 0.692 |

Binary F1 remains within 0.57--0.59 across all fractions -- no collapse even at 10% (50 samples). Contrastive retrieval does not fail with small clean pools because the Router-Detector architecture is the primary driver of detection quality, not the knowledge base volume.

3. Generalization Beyond C/C++

> "Evaluation only on C/C++; claims language-agnostic but lacks evidence on other languages."

We acknowledge this. MulVul's architecture operates on source code through natural-language prompts without language-specific parsers or ASTs. The CWE taxonomy used for routing is language-agnostic (e.g., CWE-119 applies across C, C++, Rust unsafe blocks). We will revise wording to frame language-agnosticity as a *design principle* rather than a fully validated claim, and add cross-language evaluation as explicit future work.

4. Reproducibility with Closed-Source Models

> "Relies on closed-source models (GPT-4o, Claude). Unclear if prompt evolution works with open-weight models."

MulVul's Router-Detector design is not tied to any specific LLM -- the prompts, routing logic, and aggregation strategy are fully specified and portable. We will release all prompts, routing configurations, and evaluation code. Validating with open-weight models (e.g., Qwen, DeepSeek) is an important next step that we will add as an explicit future direction.
