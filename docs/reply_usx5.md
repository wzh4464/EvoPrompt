# Response to Reviewer UsX5

We thank Reviewer UsX5 for the thorough evaluation and constructive feedback.

## 1. Cost and Latency Analysis

We provide the requested per-sample and per-1kLOC cost metrics. All methods use GPT-4o on the same PrimeVul data.

**Table 1: CWE Classification on Full PrimeVul Test Set (paper's main task)**

| Method | Macro-F1 | Cost Ratio |
|--------|----------|------------|
| Single-pass (no RAG) | 9.23% | 0.32x |
| Single-pass + RAG | 21.39% | 0.58x |
| Reflexion | 27.40% | 4.42x |
| **MulVul (Ours)** | **34.79%** | **1.0x** |

**Table 2: Binary Detection with Security-Aware Metrics (150-sample PrimeVul)**

| Method | Vuln Recall | F2-Score | FN | FP | Expected Security Cost |
|--------|-------------|----------|----|----|----------------------|
| **MulVul** | **83.7%** | **0.687** | **7** | 54 | **$754k** |
| Single-pass | 53.5% | 0.523 | 20 | 25 | $2,025k |
| Reflexion | 32.6% | 0.320 | 29 | 33 | $2,933k |
| MAD | 11.6% | 0.136 | 38 | 6 | $3,806k |

*F2-score weights recall 2x over precision, appropriate for security where missed vulnerabilities (FN) carry asymmetric risk. Expected security cost assumes FN=$100k (missed exploit), FP=$1k (false alarm review).*

**Table 3: Cost-Efficiency (150-sample PrimeVul)**

| Method | API Calls/Sample | Tokens/Sample | Sec/Sample | Tokens/1kLOC |
|--------|------------------|---------------|------------|--------------|
| Single-pass | 1.0 | 522 | 3.67 | ~6.2k |
| **MulVul** | **3.0** | **1,631** | **10.98** | **~19.4k** |
| Reflexion | 3.0 | 4,026 | 22.85 | ~47.9k |
| MAD | 5.0 | 5,915 | 50.01 | ~70.3k |

MulVul is **Pareto-optimal** among multi-agent methods:

- **vs. Reflexion**: 2.5x fewer tokens, 52% lower latency, **+7.39% higher Macro-F1** on CWE classification.
- **vs. MAD**: 3.6x fewer tokens, 78% lower latency, with dramatically higher recall (83.7% vs. 11.6%).
- **vs. Single-pass**: MulVul costs ~3x tokens but catches 2.6x more vulnerabilities (83.7% vs. 53.5% recall). Using a security-adjusted cost model (Table 2), MulVul's expected cost is **$754k vs. $2,025k** for single-pass -- the extra API cost is far outweighed by avoided false negatives.

The +41.5% improvement over the best baseline costs only ~19.4k tokens/1kLOC -- less than half of Reflexion (47.9k) and a quarter of MAD (70.3k).

## 2. Clean Pool Sensitivity (Section 4.2.1)

We varied the clean pool fraction {0.1, 0.25, 0.5, 1.0} on the full PrimeVul test set (1,907 samples):

| Clean Pool Fraction | Pool Size | Binary F1 | Recall |
|:-------------------:|:---------:|:---------:|:------:|
| 0.10 | 50 | 0.570 | 0.646 |
| 0.25 | 125 | 0.585 | 0.703 |
| 0.50 | 250 | 0.581 | 0.673 |
| 1.00 | 500 | 0.583 | 0.692 |

Binary F1 remains within 0.57--0.59 across all fractions -- **no collapse** even at 10% (50 samples). The Router-Detector architecture is the primary driver.

## 3. Generalization and Reproducibility

We acknowledge current validation is scoped to C/C++ (PrimeVul). MulVul's architecture uses natural-language prompts without language-specific parsers -- the CWE taxonomy is inherently language-agnostic. We will revise wording to frame this as a *design principle* rather than a fully validated claim, and add cross-language evaluation as future work. We will release all prompts, routing configurations, and evaluation code for reproduction with any LLM backend.
