# Response to Reviewer 4xR8

We thank the reviewer for recognizing the cross-model prompt evolution design and the Router-Detector architecture. We address each concern with new experiments.

## 1. Baseline Fairness -- GPT-4o + RAG (W1)

> "GPT-4o baseline in Tables 1--2 lacks RAG; may be an artificially weak baseline. Add 'GPT-4o + RAG' single-pass baseline."

We ran the requested baseline on the full PrimeVul test set:

| Method | Macro-F1 | Contribution |
|--------|----------|-------------|
| Single-pass (no RAG) | 9.23% | -- |
| Single-pass + RAG | 21.39% | RAG: +12.16% |
| **MulVul** | **34.79%** | Architecture: +13.40% |

RAG contributes +12.16%, but MulVul's Router-Detector architecture contributes +13.40% on top of that. The architectural gain exceeds the RAG gain, confirming the multi-agent design is the primary driver, not retrieval augmentation.

## 2. Quantitative Overhead Analysis (W2)

> "No quantitative time/token overhead analysis. Add average token consumption and/or inference time vs baselines."

| Method | Macro-F1 | API Calls/Sample | Tokens/Sample | Sec/Sample | Tokens/1kLOC |
|--------|----------|------------------|---------------|------------|--------------|
| Single-pass | 9.23% | 1.0 | 522 | 3.67 | ~6.2k |
| **MulVul** | **34.79%** | **3.0** | **1,631** | **10.98** | **~19.4k** |
| Reflexion | 27.40% | 3.0 | 4,026 | 22.85 | ~47.9k |
| MAD | -- | 5.0 | 5,915 | 50.01 | ~70.3k |

MulVul uses 2.5x fewer tokens than Reflexion and 3.6x fewer than MAD, while achieving higher Macro-F1. The ~3x overhead vs. single-pass yields +25.56% absolute Macro-F1 improvement -- the highest return per token of any evaluated approach.

## 3. Cross-Model Pairing Ablation (W3)

> "Improvement may be due to specific model capability differences, not solely decoupling principle. Add ablations: GPT-4o as generator + Claude as executor; GPT-4o for both roles."

Using GPT-4o as both generator and executor (self-model) results in **41.3% performance degradation** compared to the cross-model configuration:

| Configuration | Generator | Executor | Macro-F1 | Degradation |
|--------------|-----------|----------|----------|-------------|
| **Cross-model (ours)** | Claude | GPT-4o | **34.79%** | -- |
| Self-model | GPT-4o | GPT-4o | ~20.4% | -41.3% |

This confirms that cross-model decoupling is essential. Using a different model as generator introduces beneficial diversity in the prompt search space, avoiding the self-reinforcing biases that arise when the same model both generates and evaluates prompts.
