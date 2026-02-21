# Response to Reviewer rvhT

Thank you for the precise and practical comments. You pointed out several places where our original text was too terse, especially on scalability and implementation details. We addressed those gaps with additional experiments and clearer line-level explanations.

1.

> "Missing comparison to agent-based vulnerability detection approaches."

We added direct comparisons with Reflexion (Shinn et al., NeurIPS'23) and MAD (Liang et al., 2023), all under GPT-4o.

CWE classification (full PrimeVul test set):

| Method | Macro-F1 | Cost Ratio |
|--------|----------|------------|
| Single-pass (no RAG) | 3.86% | 0.32x |
| Reflexion | 27.40% | 4.42x |
| MulVul | 34.79% | 1.0x |

Binary detection (150-sample PrimeVul):

| Method | Macro-F1 | Vuln Recall | Tokens/Sample |
|--------|----------|-------------|---------------|
| MulVul | 0.588 | 83.7% | 1,631 |
| Reflexion | 0.508 | 32.6% | 4,026 |
| MAD | 0.503 | 11.6% | 5,915 |

We also checked error cases: in 9/43 vulnerable samples, MulVul was correct while all three alternatives missed.

2.

> "Real-world scalability unclear as #vulnerability types increases; unclear human effort required for initial prompts."

All Detectors use one shared template, parameterized by CWE descriptions. In practice, adding a category means adding CWE IDs/descriptions; it does not require writing a new prompt from scratch each time.

CWE-130 subset (1,907 samples, 70 CWE classes):

| Approach | Macro-F1 | CWE Coverage |
|----------|----------|--------------|
| MulVul baseline | 2.03% | 11/70 |
| Hybrid LLM+kNN | 8.08% | 25/70 |

We also added a cost/latency view so scalability is not discussed only in terms of F1:

| Method | Macro-F1 | API Calls/Sample | Tokens/Sample | Sec/Sample | Tokens/1kLOC |
|--------|----------|------------------|---------------|------------|--------------|
| Single-pass (no RAG) | 3.86% | 1.0 | 522 | 3.67 | ~6.2k |
| Single-pass + RAG | 21.39% | 1.0 | 1,676 | 2.88 | ~19.9k |
| MulVul | 34.79% | 3.0 | 1,631 | 10.98 | ~19.4k |
| Reflexion | 27.40% | 3.0 | 4,026 | 22.85 | ~47.9k |
| MAD | 12.33% | 5.0 | 5,915 | 50.01 | ~70.3k |

3.

> "L163: add more details about multi-agent frameworks."

In the revised text, we now position MulVul against ReAct, Reflexion, and MAD, and explain more clearly why Router->specialized Detector decomposition differs from generic reflection/debate loops.

4.

> "L261: clarify what 'fine-grained identification' means (e.g., identify a specific CWE type)."

We now state this explicitly: fine-grained means predicting concrete CWE types (e.g., CWE-119, CWE-416), not only vulnerable vs. benign.

5.

> "L286--295: add examples to make the description clearer."

We added a step-by-step example: a double-free sample is routed to the memory branch, then evaluated by memory-related detectors, and finally mapped to CWE-415.

6.

> "L382: provide example of Detector Agent prompt; is each initial prompt designed case-by-case (scalability concern)?"

We now include one Detector prompt example and make clear that Detectors share the same parameterized template; extension is done by filling CWE descriptions, not by case-by-case prompt writing.

7.

> "L392: specify retrieval budget and how agents allocate it."

We now specify the retrieval setting directly (fixed top-k, k=3 per query) and describe where those examples are consumed in the pipeline.

8.

> "L502: clarify what is removed in the 'without agent' version (tools? memory? interactions?)."

We now define this ablation precisely: remove Router specialization by either running all Detectors for each sample or replacing them with a single generic detector.
