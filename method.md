# 4 Method

## 4.1 Framework Overview

Figure 2 illustrates **MulVul**, a retrieval-augmented multi-agent framework for multi-class vulnerability detection. MulVul operates in **two phases**: **offline preparation** and **online multi-agent detection**.

**Offline preparation**
- Construct a vulnerability knowledge base **K** by converting labeled samples into **SCALE** representations, which preserve structural semantics critical for vulnerability reasoning.
- Perform **cross-model prompt evolution** to optimize prompts for all agents:
  - An **Evolution Agent** generates candidate prompts using one LLM (e.g., Claude).
  - Target agents execute the prompts using a different LLM (e.g., GPT-4o) and return performance feedback.
- Through this process:
  - The **Router Agent** obtains a prompt optimized for **category-level recall**.
  - Each **Detector Agent** receives a prompt tailored for **precise type-specific identification**.

**Online multi-agent detection**
- MulVul adopts a **coarse-to-fine Router–Detector architecture** aligned with the **CWE taxonomy**.
- Given a code snippet `x`:
  1. A **Router Agent** retrieves **cross-type contrastive evidence** from `K` and predicts the **top-k** CWE categories.
  2. Only the corresponding **category-specific Detector Agents** are invoked, each retrieving **type-specific exemplars** to identify the exact vulnerability type.
- This design reduces inference cost from **O(|Y|)** to **O(k)** while maintaining recall via **top-k routing**.
- Each Detector operates **independently** with evidence-backed outputs, preventing **cascading errors** across agents.

---

## 4.2 Offline Preparation

The offline phase constructs the retrieval infrastructure and optimizes prompts for all agents.

### 4.2.1 Knowledge Base Construction

We construct a vulnerability knowledge base `K` to provide grounding evidence for both Router and Detector Agents. Given a labeled dataset:

\[
D=\{(x_i, y_i)\}_{i=1}^{N}
\]

where `x_i` is a code snippet and `y_i ∈ Y` is its vulnerability label, we convert each sample into its SCALE representation `T(x_i)` following SCALE. SCALE normalizes control-flow structures and enriches security-relevant statements with semantic annotations, reducing sensitivity to superficial code variations while preserving patterns critical for vulnerability reasoning.

We index all transformed samples to form the knowledge base:

\[
K=\{(T(x_i), y_i)\}_{i=1}^{N} \tag{3}
\]

For efficient retrieval, we encode each `T(x_i)` using a pre-trained code encoder and build a vector index supporting approximate nearest neighbor search. We further partition `K` into category-specific subsets \(\{K_m\}_{m=1}^{M}\), where `K_m` contains samples belonging to category `m`.

During detection:
- The **Router Agent** retrieves **cross-type contrastive examples** spanning multiple categories.
- Each **Detector Agent** retrieves only from its corresponding subset `K_m`.

---

### 4.2.2 Cross-Model Evolutionary Optimization

Since LLM-based detection is highly sensitive to prompt design, we propose **cross-model evolutionary prompt optimization** to obtain robust prompts. As illustrated in Figure 3, the key idea is to **decouple prompt generation from execution across different LLMs**:
- A **proposal model** \( \mathcal{M}_{\text{evo}} \) generates and refines candidate prompts.
- An **execution model** \( \mathcal{M}_{\text{exec}} \) evaluates them on the detection task.

This separation prevents prompts from exploiting idiosyncrasies of a single model and encourages strategies that capture generalizable vulnerability patterns. Algorithm 1 presents the optimization procedure, which proceeds in two stages.

#### Algorithm 1: Cross-Model Prompt Evolution

```text
Require: Proposal model M_evo, Execution model M_exec,
         Knowledge base K, Training set D_tr, Validation set D_val,
         Number of categories M, Iterations T, Population size n
Ensure:  Optimized router prompt p_R*, Detector prompts {p_m*}_{m=1..M}

// Stage I: Router Prompt Optimization
1: Initialize population P_R ← {p1, …, pn}
2: p_R* ← argmax_{p ∈ P_R} Recall@k(p, M_exec, K, D_val)
3: for t = 1 to T do
4:     for each p ∈ P_R do
5:         F(p) ← Recall@k(p, M_exec, K, D_tr)
6:     end for
7:     P_R ← EVOLVE(P_R, {F(p)}, M_evo)
8:     p ← argmax_{p ∈ P_R} Recall@k(p, M_exec, K, D_val)
9:     if score(p) > score(p_R*) then
10:        p_R* ← p
11:    end if
12: end for

// Stage II: Detector Prompt Optimization
13: for m = 1 to M in parallel do
14:     Initialize population P_m ← {p1, …, pn}
15:     p_m* ← argmax_{p ∈ P_m} F1(p, M_exec, K_m, D_val^(m))
16:     for t = 1 to T do
17:         for each p ∈ P_m do
18:             F(p) ← F1(p, M_exec, K_m, D_tr^(m))
19:         end for
20:         P_m ← EVOLVE(P_m, {F(p)}, M_evo)
21:         Update p_m* if validation F1 improves
22:     end for
23: end for
24: return p_R*, {p_m*}_{m=1..M}
```

**Stage I: Router Prompt Optimization**
- Maintain a population of n candidate prompts (P_R) for the Router Agent.
- Each prompt is executed by \( \mathcal{M}_{\text{exec}} \) on training samples with retrieved evidence from K.
- Use Recall@k as the fitness function to ensure the correct category is included in top-k predictions.
- Use EVOLVE (Algorithm 2): retain elites and generate new candidates via LLM-driven mutation (e.g., rephrasing instructions, adding constraints, adjusting output format).
- The best validation prompt is retained as \( p_R^* \).

**Stage II: Detector Prompt Optimization**
- Fix \( p_R^* \), then optimize each Detector prompt independently (and in parallel).
- For category m, build subset \( D^{(m)} \) containing samples whose ground-truth category is m.
- Fitness is F1 score (precision/recall balance) for fine-grained type identification.
- Each Detector retrieves evidence from its category-specific subset (K_m).
- Parallelization across M categories ensures efficiency.

Unlike random perturbation, \( \mathcal{M}_{\text{evo}} \) generates semantically meaningful prompt variants: given a parent prompt and its fitness score, it analyzes weaknesses and produces improved variants, enabling more effective exploration of the discrete prompt space.

#### Algorithm 2: EVOLVE (LLM-Driven Prompt Evolution)

```text
Require: Population P, Fitness scores {F(p)}_{p∈P},
         Proposal model M_evo, Elite ratio α
Ensure:  Updated population P'

1: P' ← top-⌊α|P|⌋ prompts ranked by F   // Elite selection
2: while |P'| < |P| do
3:     Sample parent p from P with probability ∝ F(p)
4:     p' ← M_evo(mutate, p, F(p))      // LLM mutation
5:     P' ← P' ∪ {p'}
6: end while
7: return P'
```


---

## 4.3 Online Multi-Agent Detection

MulVul employs a coarse-to-fine hierarchical design aligned with the CWE taxonomy (CWE View 1400). The CWE taxonomy organizes vulnerabilities into a two-level hierarchy:
- A high-level category (e.g., Memory Buffer Errors, Injection)
- Fine-grained vulnerability types within that category (e.g., CWE-119 Buffer Overflow, CWE-125 Out-of-bounds Read under Memory Buffer Errors)

Based on this structure:
1. A Router Agent predicts relevant categories.
2. Category-specific Detector Agents identify the exact vulnerability types within each predicted category.

Given optimized prompts \( p_R^* \) and \( \{p_m^*\}_{m=1..M} \), MulVul performs retrieval-augmented multi-agent detection at inference time. Algorithm 3 summarizes the procedure.

#### Algorithm 3: MulVul Online Detection

```text
Require: Code snippet x,
         Knowledge base K, Category subsets {K_m}_{m=1..M},
         Router prompt p_R*, Detector prompts {p_m*}_{m=1..M},
         Retrieval size r, Top-k routing
Ensure:  Prediction ŷ, Evidence ê

// Stage I: Coarse-grained Routing
1: T(x) ← SCALE(x)                      // Structured representation
2: E_R ← RETRIEVE(T(x), K, r)           // Cross-type evidence
3: C_top-k ← ROUTER(p_R*, x, E_R)       // Top-k categories

// Stage II: Fine-grained Detection
4: for m ∈ C_top-k in parallel do
5:     E_m ← RETRIEVE(T(x), K_m, r)     // Type-specific evidence
6:     (y_m, c_m, e_m) ← DETECTOR_m(p_m*, x, E_m)
7: end for

// Stage III: Decision Aggregation
8: if all y_m = non-vul then
9:     ŷ ← non-vul; ê ← ∅
10: else
11:    m* ← argmax_m c_m · I[y_m ≠ non-vul]
12:    ŷ ← y_{m*}; ê ← e_{m*}
13: end if
14: return ŷ, ê
```

### 4.3.1 Router Agent

Given an input code snippet x, the Router Agent predicts the most likely coarse-grained CWE categories.

1. Convert x into its SCALE representation T(x) and retrieve r cross-type contrastive examples from K:

\[
E_R = \text{RETRIEVE}(T(x), K, r) \tag{4}
\]

The retrieved examples are selected to span multiple vulnerability categories, providing contrastive patterns that highlight distinguishing features between categories.

2. The Router Agent takes \( p_R^* \), x, and evidence \( E_R \) as input, outputting a ranked list of top-k category predictions:

\[
C_{\text{top-k}} = \text{ROUTER}(p_R^*, x, E_R) \tag{5}
\]

We set k to balance recall and efficiency: a larger k reduces the risk of missing the correct category but increases downstream computation. In practice, k = 3 achieves over 95% category-level recall while invoking only a small subset of Detectors.

### 4.3.2 Detector Agents

For each predicted category \( m \in C_{\text{top-k}} \), the corresponding Detector Agent performs fine-grained vulnerability type identification.

1. Each Detector retrieves type-specific evidence from its category subset \( K_m \):

\[
E_m = \text{RETRIEVE}(T(x), K_m, r) \tag{6}
\]

The retrieved evidence includes:
- Positive exemplars (confirmed vulnerabilities of types within category m)
- Negative exemplars (non-vulnerable code or different vulnerability types)

2. Each Detector Agent independently produces a structured prediction:

\[
(y_m, c_m, e_m) = \text{DETECTOR}_m(p_m^*, x, E_m) \tag{7}
\]

where:
- \( y_m \in Y_m \cup \{\text{non-vul}\} \) is the predicted vulnerability type (or non-vulnerable),
- \( c_m \in [0,1] \) is the confidence score,
- \( e_m \) is the supporting evidence extracted from the code.

After all invoked Detector Agents return predictions, MulVul aggregates results to produce the final output.

Key design principle: Detector Agents operate independently without inter-agent communication. Each Detector receives only the original code and retrieved evidence, not outputs from other Detectors. This isolation prevents cascading errors: if one Detector hallucinates, the error remains contained and does not influence other Detectors’ reasoning. Combined with evidence-backed structured outputs, this design improves reliability compared to multi-agent systems with shared reasoning chains.
