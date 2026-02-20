# P1 实验计划：Single-pass + RAG 基线

## Context

Reviewer 4xR8 (W1/R1) 指出原论文 Tables 1-2 中 GPT-4o 基线没有 RAG，而 MulVul 有 RAG，这不公平。需要补一个 "Single-pass + RAG" 基线来隔离 RAG 贡献 vs 架构贡献。

当前 P0 实验（MulVul/Reflexion/MAD）都不用 RAG，是纯架构对比。P1 要增加一组带 RAG 的对比。

## 实验设计

### 2x2 对比矩阵

| | 无 RAG | 有 RAG |
|---|---|---|
| **Single-pass** (单轮) | Exp A: `singlepass_norag` | Exp B: `singlepass_rag` |
| **MulVul** (Router→Detector) | 已有 P0 结果 (58.8% F1) | Exp C: `mulvul_rag` (可选) |

**必做**: Exp A + Exp B（隔离 RAG 贡献）
**可选**: Exp C（完整 2x2，隔离架构贡献）

### 预期结论

```
Single-pass (no RAG) < Single-pass + RAG < MulVul (no RAG) ≤ MulVul + RAG
```

- RAG 贡献 = B - A
- 架构贡献 = P0_MulVul - B（更大）
- 证明：RAG 有用，但架构是主要贡献

## 实现步骤

### Step 1: 构建知识库 (KB)

**文件**: `scripts/rebuttal/build_retrieval_kb.py`

从 PrimeVul 训练集 (`/Volumes/Mac_Ext/link_cache/codes/primevul/primevul_train.jsonl`) 构建轻量知识库：

- 按 10 个 Major 类别（与 P0 Router 的类别对齐）分组训练集中的漏洞样本
- 每类采样 **20 个**代表性例子（总计 ~200 个，够检索用）
- 加入 **20 个 benign** 对比样本
- 每个样本保留：code (截断到 1500 chars), cwe, major_category, description
- 输出: `outputs/rebuttal/retrieval_kb.json`

检索方法：**Jaccard similarity**（已有 `src/evoprompt/rag/retriever.py` 的实现，或直接在脚本中实现简易版）

### Step 2: Single-pass 无 RAG (`run_singlepass_norag.py`)

**文件**: `scripts/rebuttal/run_singlepass_norag.py`

模式：完全复用 `llm_utils.py`，跟 P0 脚本同一模式

```
每个样本:
  1. 构造 prompt = system_prompt + code
  2. 1 次 LLM call (max_tokens=256)
  3. parse_vulnerability_label(response)
```

System prompt:
```
You are a security code analyst. Analyze the given code and determine
if it contains a security vulnerability.

If vulnerable, respond: Vulnerable - [CWE-ID] [brief description]
If not vulnerable, respond: Benign - no vulnerability found.
```

- 数据: `sampled_150.jsonl`（与 P0 完全相同的 150 样本）
- API calls/sample: **1**
- 输出: `exp1_singlepass/singlepass_norag_results.json` + `_details.jsonl`

### Step 3: Single-pass + RAG (`run_singlepass_rag.py`)

**文件**: `scripts/rebuttal/run_singlepass_rag.py`

```
每个样本:
  1. 用 Jaccard 相似度从 KB 检索 top-3 最相似的漏洞代码 + top-1 benign 代码
  2. 构造 prompt = system_prompt + retrieved_examples + code
  3. 1 次 LLM call (max_tokens=256)
  4. parse_vulnerability_label(response)
```

System prompt:
```
You are a security code analyst. You are given reference examples of
known vulnerabilities and clean code, followed by a target code to analyze.

Use the reference examples to calibrate your judgment. Determine if the
target code contains a security vulnerability.

If vulnerable, respond: Vulnerable - [CWE-ID] [brief description]
If not vulnerable, respond: Benign - no vulnerability found.
```

User prompt 结构:
```
=== Reference Examples ===

[Example 1 - Vulnerable (CWE-119: Buffer Overflow)]
```c
<retrieved vulnerable code 1>
```

[Example 2 - Vulnerable (CWE-416: Use After Free)]
```c
<retrieved vulnerable code 2>
```

[Example 3 - Vulnerable (CWE-190: Integer Overflow)]
```c
<retrieved vulnerable code 3>
```

[Example 4 - Clean Code]
```c
<retrieved benign code>
```

=== Target Code to Analyze ===
```c
<target code>
```

Analyze the target code. Is it vulnerable?
```

- 数据: `sampled_150.jsonl`
- API calls/sample: **1**（检索不算 API call）
- 输出: `exp1_singlepass/singlepass_rag_results.json` + `_details.jsonl`

### Step 4: (可选) MulVul + RAG

**文件**: `scripts/rebuttal/run_mulvul_rag.py`

在现有 `run_mulvul_baseline.py` 基础上修改：
- Router prompt 里加入 retrieved examples（每个类别 1 个示例）
- Detector prompt 里加入 retrieved examples（检索 top-2 同类别漏洞）
- 其余逻辑不变

### Step 5: 绘图与分析

**文件**: `scripts/rebuttal/plot_p1_results.py`

更新 cost-effectiveness 散点图：
- 在现有图上添加 Single-pass (no RAG) 和 Single-pass + RAG 两个点
- 或生成新的对比柱状图

## 关键文件

| 文件 | 用途 |
|------|------|
| `scripts/rebuttal/llm_utils.py` | 复用：TokenStats, call_llm, compute_metrics, load_samples, parse_vulnerability_label |
| `scripts/rebuttal/run_mulvul_baseline.py` | 参考模板（同样的代码结构） |
| `/Volumes/Mac_Ext/link_cache/codes/primevul/primevul_train.jsonl` | 训练数据，用于构建知识库 |
| `outputs/rebuttal/sampled_150.jsonl` | 测试数据（与 P0 完全相同） |
| `outputs/rebuttal/exp1_singlepass/` | 输出目录 |

## 执行顺序

```
1. build_retrieval_kb.py          → outputs/rebuttal/retrieval_kb.json (~2min, 本地)
2. run_singlepass_norag.py        → 150 API calls (~10min)
3. run_singlepass_rag.py          → 150 API calls (~10min)
4. (可选) run_mulvul_rag.py       → 450 API calls (~30min)
5. plot_p1_results.py             → 更新图表
```

总计 API 成本：~300-750 calls，约 30-60 分钟

## 验证方式

1. 检查 `singlepass_norag_results.json` 和 `singlepass_rag_results.json` 的 metrics 格式与 P0 一致
2. 确认 RAG 版本的 tokens/sample 高于 no-RAG 版本（prompt 更长）
3. 确认 RAG 版本的 F1 高于 no-RAG 版本（检索有帮助）
4. 确认 MulVul(P0) 的 F1 高于 Single-pass+RAG（架构贡献 > RAG 贡献）
5. 运行 `plot_p1_results.py` 生成对比图
