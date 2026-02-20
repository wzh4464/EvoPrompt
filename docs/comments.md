# MulVul (Submission 10134) — Consolidated Reviewer Comments

## Reviewer UsX5

### Summary (What they think the paper is)
- MulVul: retrieval-augmented multi-agent vulnerability detection with coarse-to-fine Router→Detector design.
- Uses structured retrieval (SCALE) to ground responses and mitigate hallucinations.
- Key contribution: Cross-Model Prompt Evolution (Claude as generator; GPT-4o as executor/evaluator) to reduce self-correction bias.

### Strengths
- Cross-Model Prompt Evolution is compelling; decoupling generator/executor addresses self-correction bias (vs single-model prompt optimization like OPRO).
- Coarse-to-fine Router/Detector fits CWE hierarchy; avoids running all detectors.
- Strong empirical gains, especially few-shot; retrieval helps robustness.

### Weaknesses / Concerns
- Practical scalability: still many expensive LLM calls per sample (Router + multiple Detectors + retrieval). Missing quantitative cost/latency analysis vs baselines/static tools.
- Generalization: evaluation only on C/C++; claims language-agnostic but lacks evidence on other languages (e.g., Java/Python).
- Reproducibility/accessibility: relies on closed-source models (GPT-4o, Claude). Unclear if prompt evolution works with open-weight models.

### Requested Additions / Questions
- Provide cost metrics (e.g., “cost per 1k LOC”, “seconds per sample”) and discuss trade-off (is +41.5% worth 3–4× cost?).
- Sensitivity: Section 4.2.1 knowledge base partitioning — how sensitive to size of the “clean pool” (clean examples)? If too small, does contrastive retrieval fail to reduce false positives?

### Minor / Typos
- Section 4.2.1: note about excluding the instance itself.
- Figure 6: prompt comparison is strong/illustrative.

---

## Reviewer rvhT

### Summary (What they think the paper is)
- MulVul: Router routes code to candidate categories; Detector identifies CWE types.
- Cross-Model Prompt Evolution optimizes multiple prompts automatically.

### Strengths
- Simple and effective overall framework.
- Router/Detector agent design is interesting.
- Cross-Model Prompt Evolution improves scalability by automating prompt optimization.

### Weaknesses / Concerns
- Missing comparison to agent-based vulnerability detection approaches.
- Real-world scalability unclear as #vulnerability types increases; unclear human effort required for initial prompts.
- Several unclear details that need clarification.

### Requested Clarifications / Edits (line-referenced)
- L163: add more details about multi-agent frameworks.
- L261: clarify what “fine-grained identification” means (e.g., identify a specific CWE type).
- L286–295: add examples to make the description clearer.
- L382: provide example of Detector Agent prompt; is each initial prompt designed case-by-case (scalability concern)?
- L392: specify retrieval budget and how agents allocate it.
- L502: clarify what is removed in the “without agent” version (tools? memory? interactions?).

---

## Reviewer 4xR8

### Summary (What they think the paper is)
- MulVul: Router predicts vulnerability categories; specialized Detectors identify specific CWE types.
- Prompt evolution: Claude generates prompts; GPT-4o executes/evaluates.

### Strengths
- Router+Detector coverage addresses heterogeneous vulnerability patterns better than a single model.
- Cross-Model Prompt Evolution is innovative; decoupling reduces self-correction bias and scales prompt engineering.
- Strong empirical results: Macro-F1 34.79% over 130 CWE types; +41.5% over best baseline.
- Claims robustness on long-tail and few-shot types.

### Weaknesses / Concerns
- Baseline fairness: GPT-4o baseline in Tables 1–2 lacks RAG; may be an artificially weak baseline.
- Cost/latency: multi-turn multi-call design increases inference cost; no quantitative time/token overhead analysis.
- Attribution ambiguity: improvement may be due to specific model capability differences (Claude Opus vs GPT-4o), not solely decoupling principle; lacks alternative pairings.

### Requested Additions / Experiments
- Add “GPT-4o + RAG” single-pass baseline using the same knowledge base/retrieval to isolate architectural gains vs RAG gains.
- Add quantitative overhead analysis: average token consumption and/or inference time vs baselines.
- Add ablations on generator/executor pairing:
  - GPT-4o as generator + Claude as executor
  - GPT-4o for both roles
  - (Goal: verify decoupling benefit independent of specific model pairing)

---

