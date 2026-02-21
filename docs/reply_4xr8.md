# Response to Reviewer 4xR8

Thank you for the rigorous review. You were right to ask for stronger controlled evidence on fairness, overhead, and model-pair effects. In this rebuttal, we fully addressed fairness/overhead, completed the GPT-4o/GPT-4o pairing control, and scheduled GPT-4o/Claude for camera-ready due to rebuttal-time budget limits.

1.

> "GPT-4o baseline in Tables 1--2 lacks RAG; may be an artificially weak baseline. Add 'GPT-4o + RAG' single-pass baseline."

We added exactly this baseline on the full PrimeVul test set:

| Method | Macro-F1 | Contribution |
|--------|----------|-------------|
| Single-pass (no RAG) | 3.86% | -- |
| Single-pass + RAG | 21.39% | RAG: +17.53% |
| MulVul | 34.79% | Architecture: +13.40% |

So RAG helps a lot, but the Router-Detector architecture still adds a large gain on top.

2.

> "No quantitative time/token overhead analysis. Add average token consumption and/or inference time vs baselines."

| Method | Macro-F1 | API Calls/Sample | Tokens/Sample | Sec/Sample | Tokens/1kLOC |
|--------|----------|------------------|---------------|------------|--------------|
| Single-pass | 3.86% | 1.0 | 522 | 3.67 | ~6.2k |
| MulVul | 34.79% | 3.0 | 1,631 | 10.98 | ~19.4k |
| Reflexion | 27.40% | 3.0 | 4,026 | 22.85 | ~47.9k |
| MAD | 12.33% | 5.0 | 5,915 | 50.01 | ~70.3k |

This puts the trade-off in plain terms: MulVul is costlier than single-pass, but much cheaper than other agentic baselines and still best in Macro-F1 in our setting.

3.

> "Improvement may be due to specific model capability differences, not solely decoupling principle. Add ablations: GPT-4o as generator + Claude as executor; GPT-4o for both roles."

We added the self-model control (GPT-4o as both generator and executor):

| Configuration | Generator | Executor | Macro-F1 | Degradation |
|--------------|-----------|----------|----------|-------------|
| Cross-model (ours) | Claude | GPT-4o | 34.79% | -- |
| Self-model | GPT-4o | GPT-4o | 20.4% | -41.3% |

The drop is substantial, which is why we believe decoupling is doing real work here, not just one model tuning itself. This directly addresses the core confound that improvements might come only from single-model self-optimization.
For completeness: we did not finish the reverse pairing (GPT-4o generator + Claude executor) in the rebuttal window due to cost/time limits. In our current pipeline, Claude is used in the meta step (roughly one update per batch), but reverse pairing would move Claude into sample-level execution. With batch size 16 and ~3 executor calls/sample, this is about 48 Claude execution calls per batch (vs ~1 Claude meta call per batch now), so cost rises sharply. We will include this reverse-pairing result in the camera-ready version.
