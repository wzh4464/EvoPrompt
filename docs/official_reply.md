# MulVul (Submission 10134) -- Official Rebuttal

Thank you to all reviewers and AC for the careful feedback. You pointed out real weaknesses in our original draft, especially around evidence quality and clarity. We addressed all major requests with new experiments and clearer write-up; one reverse-pairing ablation is partially addressed in rebuttal and scheduled for camera-ready.

## Note to AC

We appreciate the guidance. In response to reviewer feedback, we added fairness controls (including single-pass + RAG), full cost/latency accounting, clean-pool sensitivity, agentic baselines (Reflexion/MAD), and one model-pair ablation control for prompt evolution (GPT-4o/GPT-4o). We also narrowed over-strong claims and clarified implementation details that were previously underexplained; the reverse model pairing is queued for camera-ready due to rebuttal-time budget limits.

---

## Reviewer UsX5

Thank you for pressing us on practicality and reproducibility.

1.

> "Missing quantitative cost/latency analysis..."

We now report full token/time numbers on PrimeVul: single-pass (522 tok, 3.67s, 3.86 F1), single-pass+RAG (1,676 tok, 2.88s, 21.39), MulVul (1,631 tok, 10.98s, 34.79), Reflexion (4,026 tok, 22.85s, 27.40), MAD (5,915 tok, 50.01s, 12.33).

2.

> "How sensitive to size of the clean pool?"

Across clean-pool fractions {0.1, 0.25, 0.5, 1.0}, Binary F1 remains stable (0.570/0.585/0.581/0.583), with no collapse at small pool size.

3.

> "Evaluation only on C/C++..."

We agree that current evidence is limited to PrimeVul C/C++. In the revision, we explicitly scope our conclusions to C/C++, and list cross-language validation as future work.

4.

> "Relies on closed-source models..."

We agree this matters. We will release prompts, routing configs, and evaluation code, and we explicitly list open-weight validation as next-step work.

5.

> "Section 4.2.1 / Figure 6 minor comments"

We appreciate these minor comments as well. In the camera-ready version, we will explicitly state the self-exclusion rule in Section 4.2.1 and polish the Figure 6 caption-text linkage for readability.

---

## Reviewer rvhT

Thank you for the precise comments on missing system details.

1.

> "Missing comparison to agent-based approaches"

We added Reflexion and MAD comparisons under the same GPT-4o setting. On CWE classification: MulVul 34.79 vs Reflexion 27.40 vs single-pass 3.86. On binary detection (150 samples): MulVul F1 0.588 / recall 83.7%, Reflexion 0.508 / 32.6%, MAD 0.503 / 11.6%.

2.

> "Scalability unclear as vulnerability types increase"

Detectors share one parameterized template, so adding categories only needs CWE entries. On CWE-130 (70 classes), coverage improves from 11/70 to 25/70 with hybrid LLM+kNN. We also added cost/latency reporting to make scaling trade-offs explicit.

3.

> "L163/L261/L286-295/L382/L392/L502 clarifications"

All requested text edits are now included: relation to ReAct/Reflexion/MAD, explicit fine-grained CWE definition, Router->Detector walkthrough, Detector prompt template example, retrieval budget (top-k=3), and exact "without agent" ablation definition.

---

## Reviewer 4xR8

Thank you for pushing us on controlled validation.
For this reviewer, we fully addressed the fairness and overhead requests. For pairing ablations, we completed GPT-4o/GPT-4o and queued GPT-4o/Claude for camera-ready.

1.

> "Add GPT-4o + RAG baseline"

Added as requested: 3.86 (single-pass), 21.39 (single-pass+RAG), 34.79 (MulVul). This separates retrieval gain from architecture gain.

2.

> "Add quantitative overhead analysis"

Added token/time accounting across methods. MulVul is costlier than single-pass, but much cheaper than other agentic baselines in our setting.

3.

> "Model-pairing ablation for decoupling"

Added self-model control (GPT-4o generator + GPT-4o executor): Macro-F1 drops from 34.79 to 20.4 (41.3% decrease), suggesting the gain is not from one model self-optimizing alone.
Due to API cost and time limits during rebuttal, we did not complete the reverse pairing (GPT-4o generator + Claude executor). In our current setup, Claude mainly appears in meta updates (about once per batch), while reverse pairing would move Claude into sample-level execution. With batch size 16 and ~3 executor calls/sample, this is about 48 Claude execution calls per batch (vs ~1 Claude meta call per batch now), so cost rises sharply. We will include this reverse-pairing result in the camera-ready version.
Within the rebuttal window, we believe the completed self-model control addresses the core attribution concern, and the reverse pairing will further strengthen this point.
