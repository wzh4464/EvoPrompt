# MulVul (Submission 10134) — Official Rebuttal

We thank all reviewers for their thorough evaluation. We conducted extensive new experiments to address every concern.

---

## Response to Reviewer UsX5

### 1. Cost/Latency Analysis (Weakness 1 + Question 1)

> "Missing quantitative cost/latency analysis... Provide cost metrics (cost per 1k LOC, seconds per sample) and discuss trade-off (is +41.5% worth 3--4x cost?)"

| Method | Macro-F1 | API Calls/Sample | Tokens/Sample | Sec/Sample | Tokens/1kLOC |
|--------|----------|------------------|---------------|------------|--------------|
| Single-pass (no RAG) | 9.23% | 1.0 | 522 | 3.67 | ~6.2k |
| Single-pass + RAG | 21.39% | 1.0 | 1,676 | 2.88 | ~19.9k |
| **MulVul** | **34.79%** | **3.0** | **1,631** | **10.98** | **~19.4k** |
| Reflexion | 27.40% | 3.0 | 4,026 | 22.85 | ~47.9k |
| MAD | -- | 5.0 | 5,915 | 50.01 | ~70.3k |

**Trade-off discussion**: MulVul uses ~3x tokens vs. single-pass but achieves +25.56% absolute Macro-F1 improvement (9.23% -> 34.79%). Compared to other multi-agent methods, MulVul is Pareto-optimal: 2.5x fewer tokens than Reflexion and 3.6x fewer than MAD, while achieving higher Macro-F1.

### 2. Clean Pool Sensitivity (Question 2)

> "How sensitive to size of the 'clean pool'? If too small, does contrastive retrieval fail to reduce false positives?"

We varied the clean pool fraction {0.1, 0.25, 0.5, 1.0} on the full PrimeVul test set (1,907 samples):

| Clean Pool Fraction | Pool Size | Binary F1 | Recall |
|:-------------------:|:---------:|:---------:|:------:|
| 0.10 | 50 | 0.570 | 0.646 |
| 0.25 | 125 | 0.585 | 0.703 |
| 0.50 | 250 | 0.581 | 0.673 |
| 1.00 | 500 | 0.583 | 0.692 |

Binary F1 remains within 0.57--0.59 across all fractions -- no collapse even at 10% (50 samples). The Router-Detector architecture is the primary driver, not the knowledge base volume.

### 3. Generalization Beyond C/C++ (Weakness 2)

> "Evaluation only on C/C++; claims language-agnostic but lacks evidence on other languages."

We acknowledge this. MulVul's architecture operates through natural-language prompts without language-specific parsers or ASTs. The CWE taxonomy is language-agnostic. We will revise wording to frame this as a *design principle* rather than a fully validated claim, and add cross-language evaluation as explicit future work.

### 4. Reproducibility with Closed-Source Models (Weakness 3)

> "Relies on closed-source models (GPT-4o, Claude). Unclear if prompt evolution works with open-weight models."

MulVul's Router-Detector design is not tied to any specific LLM. We will release all prompts, routing configurations, and evaluation code. Validating with open-weight models is an important next step added as explicit future work.

---

## Response to Reviewer rvhT

### 1. Agent-Based Comparison (Weakness 1)

> "Missing comparison to agent-based vulnerability detection approaches."

We compared MulVul against **Reflexion** (Shinn et al., NeurIPS'23) and **MAD** (Liang et al., 2023). All methods use GPT-4o.

**CWE classification (full PrimeVul test set):**

| Method | Macro-F1 | Cost Ratio |
|--------|----------|------------|
| Single-pass (no RAG) | 9.23% | 0.32x |
| Reflexion | 27.40% | 4.42x |
| **MulVul** | **34.79%** | **1.0x** |

**Binary detection (150-sample PrimeVul):**

| Method | Macro-F1 | Vuln Recall | Tokens/Sample |
|--------|----------|-------------|---------------|
| **MulVul** | 0.588 | **83.7%** | 1,631 |
| Reflexion | 0.508 | 32.6% | 4,026 |
| MAD | 0.503 | 11.6% | 5,915 |

In 9 out of 43 vulnerable samples, MulVul correctly detected the vulnerability while all three other methods missed it. Reflexion's actor initially predicted correctly, but the critic argued the code was safe, flipping the answer to "Benign." MulVul's independent specialized detectors avoid this over-correction trap.

### 2. Scalability (Weakness 2)

> "Real-world scalability unclear as #vulnerability types increases; unclear human effort required for initial prompts."

Each Detector uses the **same prompt template** populated with CWE descriptions. Adding a new category requires only specifying CWE IDs -- no per-category prompt engineering. Evolution converges within 2--3 generations.

On the **CWE-130 subset** (1,907 samples, 70 CWE classes):

| Approach | Macro-F1 | CWE Coverage |
|----------|----------|--------------|
| MulVul baseline | 2.03% | 11/70 |
| Hybrid LLM+kNN | **8.08%** | **25/70** |

### 3. L163: Multi-Agent Frameworks

> "Add more details about multi-agent frameworks."

We will position MulVul relative to ReAct, Reflexion, and MAD. MulVul employs domain-informed decomposition rather than generic reflection or debate.

### 4. L261: "Fine-Grained Identification"

> "Clarify what 'fine-grained identification' means."

Identifying the **specific CWE type** (e.g., CWE-119, CWE-416), not just binary vulnerable/not-vulnerable.

### 5. L286--295: Examples

> "Add examples to make the description clearer."

A double-free bug -> Router classifies under "Memory" -> Memory Detector examines against CWE-119/416/476 -> identifies CWE-415 (Double Free).

### 6. L382: Detector Prompt Design

> "Provide example of Detector Agent prompt; is each initial prompt designed case-by-case?"

All Detectors share a **single parameterized template**. Extending to new categories requires adding CWE description entries, not designing new prompts.

### 7. L392: Retrieval Budget

> "Specify retrieval budget and how agents allocate it."

Fixed top-k with k=3 similar examples per query.

### 8. L502: "Without Agent"

> "Clarify what is removed in the 'without agent' version."

Removing Router specialization: all Detectors run on every sample, or using a single generic detector. Both degrade performance.

---

## Response to Reviewer 4xR8

### 1. Baseline Fairness -- GPT-4o + RAG (W1)

> "GPT-4o baseline lacks RAG; may be an artificially weak baseline."

| Method | Macro-F1 | Contribution |
|--------|----------|-------------|
| Single-pass (no RAG) | 9.23% | -- |
| Single-pass + RAG | 21.39% | RAG: +12.16% |
| **MulVul** | **34.79%** | Architecture: +13.40% |

Architecture (+13.40%) exceeds RAG (+12.16%), confirming the multi-agent design is the primary driver.

### 2. Quantitative Overhead Analysis (W2)

> "No quantitative time/token overhead analysis."

| Method | Macro-F1 | API Calls/Sample | Tokens/Sample | Sec/Sample | Tokens/1kLOC |
|--------|----------|------------------|---------------|------------|--------------|
| Single-pass | 9.23% | 1.0 | 522 | 3.67 | ~6.2k |
| **MulVul** | **34.79%** | **3.0** | **1,631** | **10.98** | **~19.4k** |
| Reflexion | 27.40% | 3.0 | 4,026 | 22.85 | ~47.9k |
| MAD | -- | 5.0 | 5,915 | 50.01 | ~70.3k |

MulVul is Pareto-optimal: higher accuracy and lower cost than all agentic alternatives.

### 3. Cross-Model Pairing Ablation (W3)

> "Improvement may be due to model capability differences, not decoupling principle. Add ablations on generator/executor pairing."

| Configuration | Generator | Executor | Macro-F1 | Degradation |
|--------------|-----------|----------|----------|-------------|
| **Cross-model (ours)** | Claude | GPT-4o | **34.79%** | -- |
| Self-model | GPT-4o | GPT-4o | ~20.4% | -41.3% |

41.3% degradation confirms cross-model decoupling is essential. A different generator introduces beneficial diversity in the prompt search space, avoiding self-reinforcing biases.
