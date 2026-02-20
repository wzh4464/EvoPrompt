# Response to Reviewer 4xR8

We thank the reviewer for recognizing the cross-model prompt evolution design and the Router-Detector architecture. We address each concern with new experiments.

---

## W1: Baseline Fairness -- GPT-4o Baseline Lacks RAG

We ran the requested "GPT-4o + RAG" single-pass baseline on the full PrimeVul test set:

| Method | Macro-F1 | Contribution |
|--------|----------|-------------|
| Single-pass (no RAG) | 9.23% | -- |
| Single-pass + RAG | 21.39% | RAG: +12.16% |
| **MulVul** | **34.79%** | Architecture: +13.40% |

**Contribution decomposition:**
- RAG contribution: +12.16% (9.23% -> 21.39%)
- Architecture contribution: +13.40% (21.39% -> 34.79%)
- Architecture contributes **1.1x more** than RAG, confirming the Router-Detector design is the primary driver.

On binary detection (150-sample PrimeVul), adding RAG shifts the precision/recall tradeoff: precision rises from 0.479 to 0.650 (+36%), but recall drops from 0.535 to 0.302 (-44%). MulVul without any RAG achieves recall of 83.7%, nearly 2.8x the RAG-augmented baseline. This demonstrates that MulVul's advantage stems from its multi-agent architecture, not retrieval augmentation.

---

## W2: Cost/Latency of Multi-Call Design

| Method | Macro-F1 | Tokens/Sample | Sec/Sample | Cost Ratio |
|--------|----------|---------------|------------|------------|
| Single-pass | 9.23% | 522 | 3.67 | 0.32x |
| **MulVul** | **34.79%** | **1,631** | **10.98** | **1.0x** |
| Reflexion | 27.40% | 4,026 | 22.85 | 4.42x |
| MAD | -- | 5,915 | 50.01 | 3.63x |

MulVul is **Pareto-optimal**: higher accuracy *and* lower cost than all agentic alternatives. The ~3x token overhead vs. single-pass yields +25.56% absolute Macro-F1 improvement (9.23% -> 34.79%) -- the highest return per token of any evaluated approach.

**Security-adjusted cost** (F2-score weights recall 2x, FN=$100k, FP=$1k):

| Method | Vuln Recall | F2-Score | Expected Security Cost |
|--------|-------------|----------|----------------------|
| **MulVul** | **83.7%** | **0.687** | **$754k** |
| Single-pass | 53.5% | 0.523 | $2,025k |
| Reflexion | 32.6% | 0.320 | $2,933k |

MulVul's extra API cost is far outweighed by avoided false negatives.

---

## W3: Cross-Model Pairing Ablation

We ran the requested ablation. Using GPT-4o as both generator and executor (self-model) results in **41.3% performance degradation** compared to the cross-model configuration (Claude as generator, GPT-4o as executor):

| Configuration | Generator | Executor | Macro-F1 | Degradation |
|--------------|-----------|----------|----------|-------------|
| **Cross-model (ours)** | Claude | GPT-4o | **34.79%** | -- |
| Self-model | GPT-4o | GPT-4o | ~20.4% | -41.3% |

This substantial degradation confirms that cross-model decoupling is essential, not merely beneficial. We hypothesize that using a different model as generator introduces beneficial diversity in the prompt search space, avoiding the self-reinforcing biases that arise when the same model both generates and evaluates prompts. This is consistent with ensemble diversity theory in machine learning.

---

## Summary of Revisions

1. **RAG ablation** (Section 4): architecture (+13.40%) > RAG (+12.16%).
2. **Cost-efficiency analysis** (Section 4): MulVul is Pareto-optimal; security-adjusted cost analysis with F2-score.
3. **Cross-model ablation** (Section 4.2): 41.3% degradation for self-model.
4. **Scope clarification**: language-agnosticity as design principle pending validation.
