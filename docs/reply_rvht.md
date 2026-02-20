# Response to Reviewer rvhT

Thank you for your thoughtful review. We address each concern below with new experiments and clarifications.

## 1) Agent-Based Comparison (Weakness 1)

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

## 2) Scalability (Weakness 2)

We evaluated scalability on the **CWE-130 subset** (1,907 samples, 70 CWE classes) from BigVul/PrimeVul.

| Approach | Macro-F1 | CWE Coverage | Binary F1 |
|----------|----------|--------------|-----------|
| MulVul baseline | 2.03% | 11/70 | 59.80% |
| Hybrid LLM+kNN | **8.08%** | **25/70** | **65.17%** |

The hybrid approach augments MulVul with a kNN retriever over vulnerable training examples, improving Macro-F1 by 4x and CWE coverage from 11 to 25 of 70 classes. The retrieval component handles the long tail of rare CWE types that any fixed prompt set would struggle with.

**Human effort:** Each Detector uses the **same prompt template** populated with CWE descriptions from the CWE database. Adding a new category requires only specifying CWE IDs -- no per-category prompt engineering. Evolution experiments show convergence within 2--3 generations, keeping optimization cost modest.

## 3) Clarifications

**L163 (multi-agent frameworks):** We will position MulVul relative to ReAct (tool-augmented reasoning), Reflexion (self-reflection), and MAD (adversarial debate). MulVul differs by employing domain-informed decomposition: the Router dispatches to specialized Detectors using vulnerability taxonomy knowledge, rather than relying on generic reflection or debate.

**L261 (fine-grained identification):** "Fine-grained" means identifying the **specific CWE type** (e.g., CWE-119 Buffer Overflow, CWE-416 Use-After-Free), as opposed to binary vulnerable/not-vulnerable classification.

**L286-295 (Router-Detector example):** Consider a function with a double-free bug. The Router classifies it under "Memory." The Memory Detector then examines it against memory-related CWEs (CWE-119, CWE-416, CWE-476, etc.) and identifies CWE-415 (Double Free). This narrows the search space from the full taxonomy to a manageable subset.

**L382 (Detector prompt design):** All Detectors share a **single parameterized template** instantiated with category-specific CWE descriptions from the CWE database. Extending to new categories requires adding description entries, not new prompts.

**L392 (retrieval budget):** Fixed top-k with k=3 similar examples per query, selected via preliminary experiments balancing context length against retrieval utility.

**L502 ("without agent"):** Refers to removing the Router's specialization: either bypassing the Router (all Detectors run on every sample) or using a single generic detector. Both ablations degrade performance, confirming the value of domain-informed dispatch.

---

We believe these agent-based comparisons and scalability experiments address the reviewer's main concerns. All clarifications will be incorporated into the revised manuscript.
