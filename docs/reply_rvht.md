# Response to Reviewer rvhT

Thank you for your thoughtful review. We address each concern below.

## 1. Agent-Based Comparison (Weakness 1)

> "Missing comparison to agent-based vulnerability detection approaches."

We compared MulVul against **Reflexion** (Shinn et al., NeurIPS'23) and **Multi-Agent Debate (MAD)** (Liang et al., 2023). All methods use GPT-4o.

**CWE classification (full PrimeVul test set, paper's main task):**

| Method | Macro-F1 | Cost Ratio |
|--------|----------|------------|
| Single-pass (no RAG) | 9.23% | 0.32x |
| Reflexion | 27.40% | 4.42x |
| **MulVul** | **34.79%** | **1.0x** |

MulVul achieves +7.39% higher Macro-F1 than Reflexion while costing 4.42x less.

**Binary detection (150-sample PrimeVul):**

| Method | Macro-F1 | Vuln Recall | Tokens/Sample |
|--------|----------|-------------|---------------|
| **MulVul** | 0.588 | **83.7%** | 1,631 |
| Reflexion | 0.508 | 32.6% | 4,026 |
| MAD | 0.503 | 11.6% | 5,915 |

Qualitative error analysis reveals a structural explanation: in 9 out of 43 vulnerable samples, MulVul correctly detected the vulnerability while all three other methods missed it. In every case, Reflexion's actor initially made the correct prediction, but the critic argued the code was contextually safe, causing the refinement step to flip to "Benign." MulVul's independent specialized detectors avoid this over-correction trap.

## 2. Scalability (Weakness 2)

> "Real-world scalability unclear as #vulnerability types increases; unclear human effort required for initial prompts."

Each Detector uses the **same prompt template** populated with CWE descriptions from the CWE database. Adding a new category requires only specifying CWE IDs -- no per-category prompt engineering. Evolution converges within 2--3 generations.

We evaluated on the **CWE-130 subset** (1,907 samples, 70 CWE classes):

| Approach | Macro-F1 | CWE Coverage |
|----------|----------|--------------|
| MulVul baseline | 2.03% | 11/70 |
| Hybrid LLM+kNN | **8.08%** | **25/70** |

The hybrid approach augments MulVul with a kNN retriever over vulnerable training examples, improving CWE coverage from 11 to 25 of 70 classes. The retrieval component handles the long tail of rare CWE types that any fixed prompt set would struggle with.

## 3. L163: Multi-Agent Frameworks

> "Add more details about multi-agent frameworks."

We will position MulVul relative to ReAct (tool-augmented reasoning), Reflexion (self-reflection), and MAD (adversarial debate). MulVul differs by employing domain-informed decomposition: the Router dispatches to specialized Detectors using vulnerability taxonomy knowledge, rather than relying on generic reflection or debate.

## 4. L261: "Fine-Grained Identification"

> "Clarify what 'fine-grained identification' means."

"Fine-grained" means identifying the **specific CWE type** (e.g., CWE-119 Buffer Overflow, CWE-416 Use-After-Free), as opposed to binary vulnerable/not-vulnerable classification.

## 5. L286--295: Examples

> "Add examples to make the description clearer."

Example: a function with a double-free bug -> Router classifies under "Memory" -> Memory Detector examines against memory-related CWEs (CWE-119, CWE-416, CWE-476, etc.) -> identifies CWE-415 (Double Free). This narrows the search space from the full taxonomy to a manageable subset.

## 6. L382: Detector Prompt Design

> "Provide example of Detector Agent prompt; is each initial prompt designed case-by-case?"

All Detectors share a **single parameterized template** instantiated with category-specific CWE descriptions from the CWE database. Extending to new categories requires adding description entries, not designing new prompts. This is not case-by-case.

## 7. L392: Retrieval Budget

> "Specify retrieval budget and how agents allocate it."

Fixed top-k with k=3 similar examples per query, selected via preliminary experiments balancing context length against retrieval utility.

## 8. L502: "Without Agent"

> "Clarify what is removed in the 'without agent' version."

"Without agent" refers to removing the Router's specialization: either (a) bypassing the Router so all Detectors run on every sample, or (b) using a single generic detector. Both ablations degrade performance, confirming the value of domain-informed dispatch.
