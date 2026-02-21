# Response to Reviewer UsX5

Thank you for the careful and demanding review. You were right that our original draft did not provide enough evidence on practicality and reproducibility. We took your comments seriously, ran the missing analyses, and integrated the results into the revised draft.

1.

> "Missing quantitative cost/latency analysis... Provide cost metrics (cost per 1k LOC, seconds per sample) and discuss trade-off (is +41.5% worth 3–4x cost?)"

You are absolutely right that this was missing. We now report cost/latency on the same PrimeVul setting (GPT-4o backend):

| Method | Macro-F1 | API Calls/Sample | Tokens/Sample | Sec/Sample | Tokens/1kLOC |
|--------|----------|------------------|---------------|------------|--------------|
| Single-pass (no RAG) | 3.86% | 1.0 | 522 | 3.67 | ~6.2k |
| Single-pass + RAG | 21.39% | 1.0 | 1,676 | 2.88 | ~19.9k |
| MulVul | 34.79% | 3.0 | 1,631 | 10.98 | ~19.4k |
| Reflexion | 27.40% | 3.0 | 4,026 | 22.85 | ~47.9k |
| MAD | 12.33% | 5.0 | 5,915 | 50.01 | ~70.3k |

So MulVul is about 3x the token budget of single-pass, but gains +30.93 absolute Macro-F1 (3.86 -> 34.79). Compared with other agentic baselines, it is much cheaper while still more accurate in this setup.

2.

> "How sensitive to size of the 'clean pool'? If too small, does contrastive retrieval fail to reduce false positives?"

We tested clean-pool fractions {0.1, 0.25, 0.5, 1.0} on the full PrimeVul test set (1,907 samples):

| Clean Pool Fraction | Pool Size | Binary F1 | Recall |
|:-------------------:|:---------:|:---------:|:------:|
| 0.10 | 50 | 0.570 | 0.646 |
| 0.25 | 125 | 0.585 | 0.703 |
| 0.50 | 250 | 0.581 | 0.673 |
| 1.00 | 500 | 0.583 | 0.692 |

The key takeaway is stability: F1 stays in a narrow 0.57--0.59 range, including the 10% case.

3.

> "Evaluation only on C/C++; claims language-agnostic but lacks evidence on other languages."

We agree that current evidence is limited to PrimeVul C/C++. In the revision, we explicitly scope our conclusions to C/C++, and list cross-language validation as next-step work.

4.

> "Relies on closed-source models (GPT-4o, Claude). Unclear if prompt evolution works with open-weight models."

We agree this is an important reproducibility concern. We will release prompts, routing configurations, and evaluation code, and we now explicitly include open-weight validation as future work.

5.

> "Section 4.2.1: note about excluding the instance itself."

Thank you for catching this. We do exclude the query instance itself during retrieval in training/evaluation. In the camera-ready version, we will make this explicit in Section 4.2.1 to avoid ambiguity.

6.

> "Figure 6: prompt comparison is strong/illustrative."

Thank you for this positive feedback. In the camera-ready version, we will keep Figure 6 and improve the caption/text linkage so readers can more easily connect the visual prompt edits to the observed performance changes.
