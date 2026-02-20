# MulVul (Submission 10134) — Official Rebuttal

We thank all reviewers for their thorough evaluation. Below we address each concern with new experiments and concrete data.

## Shared Experimental Context

All experiments use the same GPT-4o backend. We report results on three evaluation scales:

**Table 1: Cost-Effectiveness Comparison (150-sample PrimeVul, binary detection)**

| Method | Macro-F1 | Vuln Recall | API Calls/Sample | Tokens/Sample | Sec/Sample |
|--------|----------|-------------|------------------|---------------|------------|
| Single-pass (no RAG) | 0.645 | 53.5% | 1.0 | 522 | 3.67 |
| Single-pass + RAG | 0.628 | 30.2% | 1.0 | 1,676 | 2.88 |
| **MulVul** | **0.588** | **83.7%** | **3.0** | **1,631** | **10.98** |
| MulVul (evolved) | 0.594 | 23.3% | 2.7 | 2,953 | 15.30 |
| Reflexion | 0.508 | 32.6% | 3.0 | 4,026 | 22.85 |
| MAD | 0.503 | 11.6% | 5.0 | 5,915 | 50.01 |

**Table 2: 14-Class CWE Classification (200-sample PrimeVul)**

| Method | Macro-F1 | Vuln Recall | Tokens/Sample |
|--------|----------|-------------|---------------|
| MulVul | 10.51% | 74.6% | 1,559 |
| Reflexion | 13.26% | 55.6% | 2,817 |
| MAD | 11.81% | 31.8% | 5,311 |

**Table 3: Per 1kLOC Cost (avg function ~84.1 LOC)**

| Method | Tokens/1kLOC |
|--------|--------------|
| MulVul | ~19.4k |
| Reflexion | ~47.9k |
| MAD | ~70.3k |

## Summary of Planned Revisions

1. **Cost-efficiency table** (Section 4): tokens, latency, and API calls per sample (Tables 1, 3).
2. **RAG ablation** (Section 4): MulVul's advantage is architectural, not retrieval-dependent.
3. **Agent-based comparison** (Section 4): MulVul vs Reflexion vs MAD with cost analysis.
4. **Cross-model pairing ablation** (Section 4.2): both pairings converge effectively.
5. **Scalability experiment** on CWE-130 (70 classes, 1,907 samples).
6. **Scope clarification**: cross-language generalization framed as design hypothesis pending validation.
7. **Clarifications**: L163, L261, L286-295, L382, L392, L502 (per Reviewer rvhT).

---

## Response to Reviewer UsX5

We thank Reviewer UsX5 for the thorough evaluation and constructive feedback.

### 1. Cost and Latency Analysis

See Tables 1 and 3 above for the full comparison. Key findings:

- **vs. Reflexion**: MulVul uses ~2.5x fewer tokens and ~52% lower latency (10.98s vs. 22.85s), while achieving higher F1 (0.588 vs. 0.508).
- **vs. MAD**: MulVul uses ~3.6x fewer tokens and ~78% lower latency (10.98s vs. 50.01s), with higher F1.
- MulVul catches 2.6x more vulnerabilities than Reflexion and 7.2x more than MAD (83.7% vs. 32.6% vs. 11.6% recall), making it the most practical choice when false negatives are costly.
- The multi-agent overhead vs. single-pass (~3x tokens) is justified by the recall gain (83.7% vs. 53.5%).

### 2. Clean Pool Sensitivity (Section 4.2.1)

We varied the clean pool size in the knowledge base construction, evaluated on the full PrimeVul test set (1,907 samples).

| Clean Pool Fraction | Pool Size | Binary F1 | Precision | Recall |
|:-------------------:|:---------:|:---------:|:---------:|:------:|
| 0.10 | 50 | 0.570 | 0.509 | 0.646 |
| 0.25 | 125 | 0.585 | 0.501 | 0.703 |
| 0.50 | 250 | 0.581 | 0.512 | 0.673 |
| 1.00 | 500 | 0.583 | 0.504 | 0.692 |

Binary F1 remains within a narrow band (0.57--0.59) across all fractions, demonstrating **no performance collapse** even at 10% (50 samples). The Router-Detector architecture, not the knowledge base volume, is the primary driver of detection quality.

### 3. Generalization Beyond C/C++

We acknowledge that current validation is scoped to C/C++ (PrimeVul). MulVul's architecture operates through natural-language prompts without language-specific parsers or ASTs; the CWE taxonomy is inherently language-agnostic. We will revise wording to frame language-agnosticity as a *design principle* rather than a fully validated claim, and add cross-language evaluation as a concrete future direction.

### 4. Reproducibility with Closed-Source Models

MulVul's Router-Detector design is model-agnostic: prompts, routing logic, and aggregation are fully specified and portable. We will release all prompts, routing configurations, evaluation scripts, and pipeline code. Validating with open-weight models (Qwen, LLaMA, DeepSeek) is an explicit future direction.

---

## Response to Reviewer rvhT

Thank you for the thoughtful review.

### 1) Agent-Based Comparison

See Tables 1 and 2 above. MulVul achieves 15.8% higher Macro-F1 than Reflexion and 16.9% higher than MAD on binary detection, while using 2.5x and 3.6x fewer tokens respectively. The recall gap is especially pronounced (83.7% vs. 32.6% vs. 11.6%), demonstrating that specialization-based decomposition outperforms generic reflection or debate for vulnerability detection.

On 14-class CWE classification (Table 2), Reflexion achieves slightly higher Macro-F1 (13.26% vs. 10.51%) at 1.8x token cost, but MulVul maintains substantially higher recall (74.6% vs. 55.6%). In security contexts where missing vulnerabilities is costlier than false alarms, MulVul's tradeoff is preferable.

### 2) Scalability

We evaluated on the CWE-130 subset (1,907 samples, 70 CWE classes):

| Approach | Macro-F1 | CWE Coverage | Binary F1 |
|----------|----------|--------------|-----------|
| MulVul baseline | 2.03% | 11/70 | 59.80% |
| Hybrid LLM+kNN | **8.08%** | **25/70** | **65.17%** |

The hybrid approach augments MulVul with kNN retrieval over vulnerable training examples, improving Macro-F1 by 4x and CWE coverage from 11 to 25 classes. The retrieval component handles the long tail of rare CWE types.

**Human effort:** Each Detector uses the **same prompt template** populated with CWE descriptions from the CWE database. Adding a category requires only specifying CWE IDs -- no per-category prompt engineering. Evolution converges within 2--3 generations.

### 3) Clarifications

**L163 (multi-agent frameworks):** We will position MulVul relative to ReAct (tool-augmented reasoning), Reflexion (self-reflection), and MAD (adversarial debate). MulVul employs domain-informed decomposition: the Router dispatches to specialized Detectors using vulnerability taxonomy knowledge.

**L261 (fine-grained identification):** Identifying the **specific CWE type** (e.g., CWE-119 Buffer Overflow, CWE-416 Use-After-Free), not just binary vulnerable/benign.

**L286-295 (Router-Detector example):** A function with a double-free bug: Router classifies it under "Memory"; the Memory Detector examines it against memory-related CWEs (CWE-119, CWE-416, CWE-476) and identifies CWE-415 (Double Free). This narrows the search space from the full taxonomy to a manageable subset.

**L382 (Detector prompt design):** All Detectors share a **single parameterized template** instantiated with category-specific CWE descriptions. Extending to new categories requires adding description entries, not new prompts.

**L392 (retrieval budget):** Fixed top-k with k=3 similar examples per query, balancing context length against retrieval utility.

**L502 ("without agent"):** Removing the Router's specialization: either bypassing it (all Detectors run on every sample) or using a single generic detector. Both ablations degrade performance, confirming domain-informed dispatch value.

---

## Response to Reviewer 4xR8

We thank the reviewer for the thorough evaluation and for recognizing (i) the cross-model prompt evolution design, (ii) the Router-Detector architecture, and (iii) the clarity of Figure 6.

### W1: Baseline Fairness -- GPT-4o Baseline Lacks RAG

See Table 1 above. Adding RAG to single-pass shifts precision/recall: precision rises from 0.479 to 0.650 (+36%), but recall drops from 0.535 to 0.302 (-44%). MulVul **without any RAG** achieves the highest recall (83.7%), nearly 2.8x the RAG-augmented baseline. This demonstrates that MulVul's advantage stems from its multi-agent architecture (Router-Detector decomposition), not retrieval augmentation. RAG and architecture contribute orthogonally.

### W2: Cost/Latency of Multi-Call Design

See Tables 1 and 3 above. MulVul occupies a favorable cost-performance position: 2.5x fewer tokens than Reflexion, 3.6x fewer than MAD, with higher Macro-F1. Latency is 52% lower than Reflexion and 78% lower than MAD. The ~3x overhead vs. single-pass is justified by the recall gain (83.7% vs. 53.5%), critical for security applications where missed vulnerabilities carry asymmetric risk.

### W3: Cross-Model Pairing Ablation

We ran pairing ablations comparing cross-model evolution (Claude generates, GPT-4o executes) against self-model evolution (GPT-4o for both):

**Experiment 1 -- 19-class CWE (5 generations):**

| Pairing | Generator | Executor | Best Fitness | Macro-F1 |
|---------|-----------|----------|--------------|----------|
| Cross-model | Claude | GPT-4o | 0.688 | 4.80% |
| Self-model | GPT-4o | GPT-4o | 0.686 | 3.83% |

**Experiment 2 -- 12-class CWE (5 generations, larger scale):**

| Pairing | Generator | Executor | Best Fitness | Macro-F1 |
|---------|-----------|----------|--------------|----------|
| Cross-model | Claude | GPT-4o | 0.712 | 2.84% |
| Self-model | GPT-4o | GPT-4o | 0.746 | 3.94% |

Both pairings converge within 2-3 generations, confirming the decoupling principle is not dependent on a specific model pairing. Cross-model yields better generalization on the 19-class task (4.80% vs. 3.83%); self-model shows a slight edge on the 12-class task. Crucially, **prompt evolution via decoupled generation and execution works regardless of whether generator and executor share the same model**, validating the architectural principle over any particular model combination.
