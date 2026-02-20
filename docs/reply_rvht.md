# Response to Reviewer rvhT

Thank you for your thoughtful review. We address each concern below with new experiments and clarifications.

## 1) Agent-Based Comparison (Weakness 1)

We compared MulVul against **Reflexion** (Shinn et al., NeurIPS'23) and **Multi-Agent Debate (MAD)** (Liang et al., 2023). All methods use GPT-4o.

**CWE classification on full PrimeVul test set (paper's main task):**

| Method | Macro-F1 | Cost Ratio |
|--------|----------|------------|
| Single-pass (no RAG) | 9.23% | 0.32x |
| Single-pass + RAG | 21.39% | 0.58x |
| Reflexion | 27.40% | 4.42x |
| **MulVul (Ours)** | **34.79%** | **1.0x** |

MulVul achieves **34.79%** Macro-F1 vs. Reflexion 27.40% (+7.39%) and Single-pass 21.39% (+13.40%), while costing **4.42x less** than Reflexion.

**Binary detection (150-sample PrimeVul):**

| Method | Vuln Recall | F2-Score | Expected Security Cost |
|--------|-------------|----------|----------------------|
| **MulVul** | **83.7%** | **0.687** | **$754k** |
| Single-pass | 53.5% | 0.523 | $2,025k |
| Reflexion | 32.6% | 0.320 | $2,933k |
| MAD | 11.6% | 0.136 | $3,806k |

MulVul's recall advantage is decisive -- 83.7% vs. 32.6% (Reflexion) and 11.6% (MAD). Qualitative error analysis reveals a structural explanation:

> **Finding**: In 9 out of 43 vulnerable samples, MulVul correctly detected the vulnerability while *all three* other methods missed it. In every case, Reflexion's actor initially made the correct prediction, but the critic argued the code was contextually safe, causing the refinement step to flip to "Benign." Similarly, MAD's judge systematically sided with the "developer" role over the "auditor" role. MulVul's independent specialized detectors avoid this over-correction trap -- each detector assesses vulnerability within its domain without a critic second-guessing it.

**Example**: For an NGINX function containing CWE-416 (Use-After-Free) and CWE-476 (NULL Pointer Dereference), Reflexion's actor correctly identified the vulnerability but the critic argued the memory management was safe by design; the refinement flipped to "Benign." MulVul's Memory and Null Pointer detectors each independently flagged the issue.

We will discuss ReAct, Reflexion, and MAD as related multi-agent frameworks in the revised L163 discussion.

## 2) Scalability (Weakness 2)

Each Detector uses the **same prompt template** populated with CWE descriptions from the CWE database. Adding a new category requires only specifying CWE IDs -- no per-category prompt engineering. Evolution converges within 2--3 generations.

We evaluated scalability on the **CWE-130 subset** (1,907 samples, 70 CWE classes):

| Approach | Macro-F1 | CWE Coverage | Binary F1 |
|----------|----------|--------------|-----------|
| MulVul baseline | 2.03% | 11/70 | 59.80% |
| Hybrid LLM+kNN | **8.08%** | **25/70** | **65.17%** |

The hybrid approach augments MulVul with a kNN retriever over vulnerable training examples, improving Macro-F1 by 4x and CWE coverage from 11 to 25 of 70 classes.

## 3) Clarifications

**L163 (multi-agent frameworks):** We will position MulVul relative to ReAct (tool-augmented reasoning), Reflexion (self-reflection), and MAD (adversarial debate). MulVul differs by employing domain-informed decomposition rather than generic reflection or debate.

**L261 (fine-grained identification):** "Fine-grained" = identifying the specific CWE type (e.g., CWE-119, CWE-416), not just binary vulnerable/benign.

**L286-295 (example):** A double-free bug -> Router classifies under "Memory" -> Memory Detector examines against CWE-119/416/476 -> identifies CWE-415 (Double Free). This narrows the search space from the full taxonomy to a manageable subset.

**L382 (Detector prompt):** All Detectors share a **single parameterized template**. Extending to new categories requires adding CWE description entries, not new prompts.

**L392 (retrieval budget):** Fixed top-k (k=3) retrieval budget per query.

**L502 ("without agent"):** Removing Router specialization (all detectors run on every sample) or using a single generic detector. Both degrade performance.
