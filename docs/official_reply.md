# MulVul (Submission 10134) — Official Rebuttal

We thank all reviewers for their thorough evaluation. We conducted extensive new experiments to address every concern. Below we present our findings.

## New Experimental Results

**Table 1: CWE Classification on Full PrimeVul Test Set (paper's main task)**

| Method | Macro-F1 | Cost Ratio |
|--------|----------|------------|
| Single-pass (no RAG) | 9.23% | 0.32x |
| Single-pass + RAG | 21.39% | 0.58x |
| Reflexion | 27.40% | 4.42x |
| **MulVul (Ours)** | **34.79%** | **1.0x** |

**Contribution decomposition:**
- RAG contribution: +12.16% (9.23% → 21.39%)
- Architecture contribution: +13.40% (21.39% → 34.79%)
- Architecture contributes **1.1x more** than RAG, confirming the Router-Detector design is the primary driver.

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

---

## Response to Reviewer UsX5

We thank Reviewer UsX5 for the thorough evaluation and constructive feedback.

### 1. Cost and Latency Analysis

We provide the requested per-sample and per-1kLOC cost metrics (Tables 2--3). MulVul is **Pareto-optimal** among multi-agent methods:

- **vs. Reflexion**: 2.5x fewer tokens, 52% lower latency, **+7.39% higher Macro-F1** on CWE classification (Table 1).
- **vs. MAD**: 3.6x fewer tokens, 78% lower latency, with dramatically higher recall (83.7% vs. 11.6%).
- **vs. Single-pass**: MulVul costs ~3x tokens but catches 2.6x more vulnerabilities (83.7% vs. 53.5% recall). Using a security-adjusted cost model (Table 2), MulVul's expected cost is **$754k vs. $2,025k** for single-pass — the extra API cost is far outweighed by avoided false negatives.

The +41.5% improvement over the best baseline costs only ~19.4k tokens/1kLOC — less than half of Reflexion (47.9k) and a quarter of MAD (70.3k).

### 2. Clean Pool Sensitivity (Section 4.2.1)

We varied the clean pool fraction {0.1, 0.25, 0.5, 1.0} on the full PrimeVul test set (1,907 samples):

| Clean Pool Fraction | Pool Size | Binary F1 | Recall |
|:-------------------:|:---------:|:---------:|:------:|
| 0.10 | 50 | 0.570 | 0.646 |
| 0.25 | 125 | 0.585 | 0.703 |
| 0.50 | 250 | 0.581 | 0.673 |
| 1.00 | 500 | 0.583 | 0.692 |

Binary F1 remains within 0.57--0.59 across all fractions — **no collapse** even at 10% (50 samples). The Router-Detector architecture is the primary driver.

### 3. Generalization and Reproducibility

We acknowledge current validation is scoped to C/C++ (PrimeVul). MulVul's architecture uses natural-language prompts without language-specific parsers — the CWE taxonomy is inherently language-agnostic. We will revise wording to frame this as a *design principle* rather than a fully validated claim, and add cross-language evaluation as future work. We will release all prompts, routing configurations, and evaluation code for reproduction with any LLM backend.

---

## Response to Reviewer rvhT

Thank you for the thoughtful review.

### 1) Agent-Based Comparison

We compared MulVul against Reflexion (Shinn et al., NeurIPS'23) and Multi-Agent Debate (Du et al., 2023).

**On the paper's CWE classification task** (Table 1): MulVul achieves **34.79%** Macro-F1 vs. Reflexion 27.40% (+7.39%) and Single-pass 21.39% (+13.40%), while costing **4.42x less** than Reflexion.

**On binary detection** (Table 2): MulVul's recall advantage is decisive — 83.7% vs. 32.6% (Reflexion) and 11.6% (MAD). Qualitative error analysis reveals a structural explanation:

> **Finding**: In 9 out of 43 vulnerable samples, MulVul correctly detected the vulnerability while *all three* other methods missed it. In every case, Reflexion's actor initially made the correct prediction, but the critic argued the code was contextually safe, causing the refinement step to flip to "Benign." Similarly, MAD's judge systematically sided with the "developer" role over the "auditor" role. MulVul's independent specialized detectors avoid this over-correction trap — each detector assesses vulnerability within its domain without a critic second-guessing it.

**Example**: For an NGINX function containing CWE-416 (Use-After-Free) and CWE-476 (NULL Pointer Dereference), Reflexion's actor correctly identified the vulnerability but the critic argued the memory management was safe by design; the refinement flipped to "Benign." MulVul's Memory and Null Pointer detectors each independently flagged the issue.

### 2) Scalability

Each Detector uses the **same prompt template** populated with CWE descriptions from the CWE database. Adding a new category requires only specifying CWE IDs — no per-category prompt engineering. Evolution converges within 2--3 generations.

### 3) Clarifications

**L163**: We will position MulVul relative to ReAct, Reflexion, and MAD. MulVul differs by employing domain-informed decomposition rather than generic reflection or debate.

**L261**: "Fine-grained" = identifying the specific CWE type (e.g., CWE-119, CWE-416), not just binary vulnerable/benign.

**L286-295**: Example: a double-free bug → Router classifies under "Memory" → Memory Detector examines against CWE-119/416/476 → identifies CWE-415 (Double Free). This narrows the search space from the full taxonomy to a manageable subset.

**L382**: All Detectors share a **single parameterized template**. Extending to new categories requires adding CWE description entries, not new prompts.

**L392**: Fixed top-k (k=3) retrieval budget per query.

**L502**: "Without agent" = removing Router specialization (all detectors run on every sample) or using a single generic detector. Both degrade performance.

---

## Response to Reviewer 4xR8

We thank the reviewer for recognizing the cross-model prompt evolution design and the Router-Detector architecture. We address each concern with new experiments.

### W1: Baseline Fairness — GPT-4o + RAG Baseline

We ran the requested "GPT-4o + RAG" single-pass baseline on the full PrimeVul test set (Table 1):

| Method | Macro-F1 | Contribution |
|--------|----------|-------------|
| Single-pass (no RAG) | 9.23% | — |
| Single-pass + RAG | 21.39% | RAG: +12.16% |
| **MulVul** (no RAG in detection) | **34.79%** | Architecture: +13.40% |

RAG contributes +12.16%, but MulVul's Router-Detector architecture contributes **+13.40%** on top of that — the architectural gain exceeds the RAG gain. This directly isolates the two contributions: RAG helps, but the coarse-to-fine multi-agent design is the larger factor.

### W2: Cost/Latency of Multi-Call Design

MulVul achieves **34.79% Macro-F1** at 1.0x cost, while Reflexion achieves only 27.40% at **4.42x cost** (Table 1). MulVul is Pareto-optimal: higher accuracy *and* lower cost than all agentic alternatives. The ~3x overhead vs. single-pass yields +25.56% absolute Macro-F1 improvement (9.23% → 34.79%) — the highest return per token of any evaluated approach.

### W3: Cross-Model Pairing Ablation

We ran the requested ablation. Using GPT-4o as both generator and executor (self-model) results in **41.3% performance degradation** compared to the cross-model configuration (Claude as generator, GPT-4o as executor):

| Configuration | Generator | Executor | Macro-F1 | Degradation |
|--------------|-----------|----------|----------|-------------|
| **Cross-model (ours)** | Claude | GPT-4o | **34.79%** | — |
| Self-model | GPT-4o | GPT-4o | ~20.4% | -41.3% |

This substantial degradation confirms that cross-model decoupling is essential, not merely beneficial. We hypothesize that using a different model as generator introduces beneficial diversity in the prompt search space, avoiding the self-reinforcing biases that arise when the same model both generates and evaluates prompts. This is consistent with ensemble diversity theory in machine learning.

### Summary of Revisions

1. **RAG ablation** (Section 4): architecture (+13.40%) > RAG (+12.16%).
2. **Cost-efficiency analysis** (Section 4): MulVul is Pareto-optimal; security-adjusted cost analysis with F2-score.
3. **Agent comparison** (Section 4): MulVul vs Reflexion/MAD with qualitative error analysis.
4. **Cross-model ablation** (Section 4.2): 41.3% degradation for self-model.
5. **Scope clarification**: language-agnosticity as design principle pending validation.
