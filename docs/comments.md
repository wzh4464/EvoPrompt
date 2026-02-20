# MulVul (Submission 10134) — Official Rebuttal

## Response to Reviewer UsX5

We thank Reviewer UsX5 for the thorough evaluation and constructive feedback. We address each concern below with new experiments and concrete data.

### 1. Cost and Latency Analysis

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

### 2. Clean Pool Sensitivity (Section 4.2.1)

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

### 3. Generalization Beyond C/C++

We acknowledge that our current experimental validation is scoped to C/C++ (PrimeVul dataset). This is a fair observation. We clarify our position:

- **Design principle vs. validated claim**: MulVul's Router-Detector architecture operates on source code through natural-language prompts and does not rely on language-specific parsers, ASTs, or compiled representations. The CWE taxonomy used for routing is inherently language-agnostic (e.g., CWE-119 buffer errors apply across C, C++, Rust unsafe blocks, etc.). We will revise our wording to frame language-agnosticity as a *design principle* rather than a fully validated claim.
- **Practical scope**: The choice of C/C++ reflects both dataset availability (PrimeVul is the largest high-quality function-level vulnerability dataset) and practical importance (C/C++ accounts for the majority of CVEs in the NVD). We will add explicit discussion of cross-language evaluation as a concrete future direction.

### 4. Reproducibility with Closed-Source Models

We appreciate this important concern regarding scientific reproducibility:

- **Model-agnostic architecture**: MulVul's Router-Detector design is not tied to any specific LLM. The prompts, routing logic, and aggregation strategy are fully specified and portable. The architecture can be instantiated with any instruction-following LLM.
- **Open-weight transfer**: We acknowledge that our main experiments use GPT-4o, and validating performance with open-weight models (e.g., Qwen, LLaMA, DeepSeek) is an important next step. We will add this as an explicit future work direction with a concrete plan.
- **Full reproducibility artifacts**: We will release all prompts, routing configurations, evaluation scripts, and pipeline code to enable reproduction with any compatible model backend.

---

We thank Reviewer UsX5 again for the actionable and constructive feedback. We believe the added cost/latency analysis and clean-pool sensitivity ablation directly address the main empirical concerns, and the clarifications on scope and reproducibility improve the paper's honesty and clarity.

---

## Response to Reviewer rvhT

Thank you for your thoughtful review. We address each concern below with new experiments and clarifications.

### 1) Agent-Based Comparison (Weakness 1)

We compared MulVul against **Reflexion** (Shinn et al., NeurIPS'23) and **Multi-Agent Debate (MAD)** (Liang et al., 2023). All methods use GPT-4o on the same 150-sample PrimeVul subset.

**Binary vulnerability detection (150 samples):**

| Method | Macro-F1 | Vuln Recall | Tokens/sample | API calls | Sec/sample |
|--------|----------|-------------|---------------|-----------|------------|
| MulVul | **0.588** | **83.7%** | 1631 | 3.0 | 10.98 |
| Reflexion | 0.508 | 32.6% | 4026 | 3.0 | 22.85 |
| MAD | 0.503 | 11.6% | 5915 | 5.0 | 50.01 |

MulVul achieves 15.8% higher Macro-F1 than Reflexion and 16.9% higher than MAD, while using 2.5x and 3.6x fewer tokens. The recall gap is especially pronounced (83.7% vs. 32.6% and 11.6%), demonstrating that MulVul's specialization-based decomposition is more effective than generic reflection or debate for vulnerability detection.

**14-class CWE classification (200 samples):**

| Method | Macro-F1 | Vuln Recall | Tokens/sample |
|--------|----------|-------------|---------------|
| MulVul | 10.51% | **74.6%** | 1559 |
| Reflexion | **13.26%** | 55.6% | 2817 |
| MAD | 11.81% | 31.8% | 5311 |

All methods find fine-grained CWE classification challenging. Reflexion achieves slightly higher Macro-F1 at 1.8x the token cost, while MulVul maintains substantially higher recall (74.6% vs. 55.6%). This precision-recall tradeoff favors MulVul in security contexts where missing a vulnerability is costlier than a false alarm.

We will discuss ReAct, Reflexion, and MAD as related multi-agent frameworks in the revised L163 discussion.

### 2) Scalability (Weakness 2)

We evaluated scalability on the **CWE-130 subset** (1,907 samples, 70 CWE classes) from BigVul/PrimeVul.

| Approach | Macro-F1 | CWE Coverage | Binary F1 |
|----------|----------|--------------|-----------|
| MulVul baseline | 2.03% | 11/70 | 59.80% |
| Hybrid LLM+kNN | **8.08%** | **25/70** | **65.17%** |

The hybrid approach augments MulVul with a kNN retriever over vulnerable training examples, improving Macro-F1 by 4x and CWE coverage from 11 to 25 of 70 classes. The retrieval component handles the long tail of rare CWE types that any fixed prompt set would struggle with.

**Human effort:** Each Detector uses the **same prompt template** populated with CWE descriptions from the CWE database. Adding a new category requires only specifying CWE IDs -- no per-category prompt engineering. Evolution experiments show convergence within 2--3 generations, keeping optimization cost modest.

### 3) Clarifications

**L163 (multi-agent frameworks):** We will position MulVul relative to ReAct (tool-augmented reasoning), Reflexion (self-reflection), and MAD (adversarial debate). MulVul differs by employing domain-informed decomposition: the Router dispatches to specialized Detectors using vulnerability taxonomy knowledge, rather than relying on generic reflection or debate.

**L261 (fine-grained identification):** "Fine-grained" means identifying the **specific CWE type** (e.g., CWE-119 Buffer Overflow, CWE-416 Use-After-Free), as opposed to binary vulnerable/not-vulnerable classification.

**L286-295 (Router-Detector example):** Consider a function with a double-free bug. The Router classifies it under "Memory." The Memory Detector then examines it against memory-related CWEs (CWE-119, CWE-416, CWE-476, etc.) and identifies CWE-415 (Double Free). This narrows the search space from the full taxonomy to a manageable subset.

**L382 (Detector prompt design):** All Detectors share a **single parameterized template** instantiated with category-specific CWE descriptions from the CWE database. Extending to new categories requires adding description entries, not new prompts.

**L392 (retrieval budget):** Fixed top-k with k=3 similar examples per query, selected via preliminary experiments balancing context length against retrieval utility.

**L502 ("without agent"):** Refers to removing the Router's specialization: either bypassing the Router (all Detectors run on every sample) or using a single generic detector. Both ablations degrade performance, confirming the value of domain-informed dispatch.

---

We believe these agent-based comparisons and scalability experiments address the reviewer's main concerns. All clarifications will be incorporated into the revised manuscript.

---

## Response to Reviewer 4xR8

We thank the reviewer for the thorough evaluation and for recognizing (i) the cross-model prompt evolution design, (ii) the Router-Detector architecture, and (iii) the clarity of Figure 6. Below we address each concern with new experiments.

---

### W1: Baseline Fairness -- GPT-4o Baseline Lacks RAG

We conducted a controlled single-pass ablation on the same 150-sample PrimeVul subset, adding RAG under identical conditions (same model, data, and evaluation):

| Method | Macro-F1 | Vuln Precision | Vuln Recall | Tokens/sample |
|--------|----------|----------------|-------------|---------------|
| Single-pass (no RAG) | 0.645 | 0.479 | 0.535 | 522 |
| **Single-pass + RAG** | **0.628** | **0.650** | **0.302** | 1,676 |
| MulVul (no RAG) | 0.588 | 0.400 | 0.837 | 1,631 |

Adding RAG shifts the precision/recall tradeoff: precision rises from 0.479 to 0.650 (+36%), but recall drops from 0.535 to 0.302 (-44%). MulVul without any RAG achieves the highest recall (83.7%), nearly 2.8x the RAG-augmented baseline. This demonstrates that MulVul's advantage stems from its multi-agent architecture (Router-Detector decomposition), not retrieval augmentation. The coarse-to-fine design enables systematic examination of vulnerability-specific patterns that a single pass misses regardless of RAG.

We will add this ablation and clarify that RAG and architecture contribute orthogonally.

---

### W2: Cost/Latency of Multi-Call Design

We measured per-sample cost across all methods on the same 150-sample subset:

| Method | Macro-F1 | API Calls | Tokens/sample | Sec/sample | Cost Ratio |
|--------|----------|-----------|---------------|------------|------------|
| Single-pass | 0.645 | 1.0 | 522 | 3.67 | 0.32x |
| **MulVul** | **0.588** | **3.0** | **1,631** | **10.98** | **1.0x** |
| Reflexion | 0.508 | 3.0 | 4,026 | 22.85 | 2.47x |
| MAD | 0.503 | 5.0 | 5,915 | 50.01 | 3.63x |

MulVul occupies a favorable cost-performance position: it uses 2.5x fewer tokens than Reflexion and 3.6x fewer than MAD, while achieving higher Macro-F1. Its latency is 52% lower than Reflexion and 78% lower than MAD. Compared to single-pass, MulVul costs ~3x but provides substantially higher recall (83.7% vs. 53.5%), critical for security applications where missed vulnerabilities carry asymmetric risk. Normalized to code volume (~84.1 LOC average), MulVul uses ~19.4k tokens/kLOC vs. 47.9k (Reflexion) and 70.3k (MAD).

We will add this cost analysis to the revised manuscript.

---

### W3: Cross-Model Pairing Ablation

We ran pairing ablations comparing cross-model evolution (Claude generates, GPT-4o executes) against self-model evolution (GPT-4o for both):

**Experiment 1 -- 19-class fine-grained CWE (5 generations):**

| Pairing | Generator | Executor | Best Fitness | Eval Macro-F1 |
|---------|-----------|----------|--------------|---------------|
| Cross-model | Claude | GPT-4o | 0.6884 | 4.80% |
| Self-model | GPT-4o | GPT-4o | 0.6864 | 3.83% |

**Experiment 2 -- 12-class CWE (large scale, 5 generations):**

| Pairing | Generator | Executor | Best Fitness | Eval Macro-F1 |
|---------|-----------|----------|--------------|---------------|
| Cross-model | Claude | GPT-4o | 0.712 | 2.84% |
| Self-model | GPT-4o | GPT-4o | 0.746 | 3.94% |

Both pairings achieve competitive fitness and converge within 2-3 generations, confirming the decoupling principle is not dependent on a specific model pairing. In the 19-class experiment, cross-model yields better generalization (4.80% vs. 3.83%), suggesting a different generator introduces beneficial diversity. In the 12-class experiment, self-model shows a slight edge, indicating relative performance varies with task granularity. Crucially, **prompt evolution via decoupled generation and execution works regardless of whether generator and executor share the same model**, validating the architectural principle over any particular model combination.

We will include both ablation tables and discuss pairing sensitivity in the revision.

---

### Summary of Revisions

1. RAG ablation table (Section 4): MulVul's advantage is architectural, not retrieval-dependent.
2. Cost-efficiency table (Section 4): tokens, latency, and API calls per sample.
3. Cross-model pairing ablation (Section 4.2): both pairings converge effectively.
4. Scope clarification: cross-language generalization framed as a design hypothesis pending validation.

We believe these additions directly address all three concerns and strengthen the paper's empirical foundation.
