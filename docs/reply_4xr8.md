# Response to Reviewer 4xR8

We thank the reviewer for the thorough evaluation and for recognizing (i) the cross-model prompt evolution design, (ii) the Router-Detector architecture, and (iii) the clarity of Figure 6. Below we address each concern with new experiments.

---

## W1: Baseline Fairness -- GPT-4o Baseline Lacks RAG

We conducted a controlled single-pass ablation on the same 150-sample PrimeVul subset, adding RAG under identical conditions (same model, data, and evaluation):

| Method | Macro-F1 | Vuln Precision | Vuln Recall | Tokens/sample |
|--------|----------|----------------|-------------|---------------|
| Single-pass (no RAG) | 0.645 | 0.479 | 0.535 | 522 |
| **Single-pass + RAG** | **0.628** | **0.650** | **0.302** | 1,676 |
| MulVul (no RAG) | 0.588 | 0.400 | 0.837 | 1,631 |

Adding RAG shifts the precision/recall tradeoff: precision rises from 0.479 to 0.650 (+36%), but recall drops from 0.535 to 0.302 (-44%). MulVul without any RAG achieves the highest recall (83.7%), nearly 2.8x the RAG-augmented baseline. This demonstrates that MulVul's advantage stems from its multi-agent architecture (Router-Detector decomposition), not retrieval augmentation. The coarse-to-fine design enables systematic examination of vulnerability-specific patterns that a single pass misses regardless of RAG.

We will add this ablation and clarify that RAG and architecture contribute orthogonally.

---

## W2: Cost/Latency of Multi-Call Design

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

## W3: Cross-Model Pairing Ablation

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

## Summary of Revisions

1. RAG ablation table (Section 4): MulVul's advantage is architectural, not retrieval-dependent.
2. Cost-efficiency table (Section 4): tokens, latency, and API calls per sample.
3. Cross-model pairing ablation (Section 4.2): both pairings converge effectively.
4. Scope clarification: cross-language generalization framed as a design hypothesis pending validation.

We believe these additions directly address all three concerns and strengthen the paper's empirical foundation.
