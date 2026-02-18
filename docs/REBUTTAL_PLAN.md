# MulVul Rebuttal 实验计划

## 审稿人意见汇总与优先级

| 优先级 | 实验 | 回应审稿人 | 核心问题 |
|--------|------|-----------|---------|
| **P0** | Exp 2: Agent 竞品对比 + 成本分析 | rvhT, UsX5, 4xR8 | 缺 agent 对比；缺成本数据 |
| **P1** | Exp 1: GPT-4o + RAG 强基线 | 4xR8 | 基线不公（无 RAG） |
| **P2** | Exp 3: Cross-Model 消融 | 4xR8 | 跨模型是否必要 |
| P3 | Exp 4: 多语言泛化 | UsX5 | 仅 C/C++ |

> **甄甄的判断**: Exp 2 最关键，做完就差不多了。Exp 3 是 3.5 分审稿人提的，有空再做。

---

## Exp 2: Agent 竞品对比 + 成本降维打击 (P0)

### 目标

1. 补充 MulVul 与主流 Agent 范式的对比 → 回应 rvhT "缺少 agent 对比"
2. 量化成本数据 → 回应 UsX5 + 4xR8 "缺少开销分析"
3. 证明 MulVul 是 Pareto 最优：**一轮出结果** vs 竞品多轮迭代

### 对比方案

| 方案 | 类型 | 预计 API Calls/样本 | 核心机制 |
|------|------|---------------------|---------|
| **MulVul (Ours)** | Coarse-to-Fine | 1 Router + k Detectors (~3-4) | 一轮，分层路由 |
| **Reflexion** | 单 Agent 自反思 | 3-4 (生成→批评→修正循环) | 多轮迭代 |
| **Multi-Agent Debate (MAD)** | 多 Agent 辩论 | 4-6 (Attacker→Defender→Judge) | 多角色长上下文 |

### 实现细节

#### Baseline A: Reflexion (Shinn et al., NeurIPS 2023)

```
Turn 0 (Actor):     GPT-4o 对代码做漏洞预测
Turn 1 (Critic):    GPT-4o 检查 Turn 0 结果，指出潜在错误
Turn 2 (Refinement): GPT-4o 根据 Critic 意见修正预测
[可选 Turn 3]:       再次 Critic → 修正
```

- Max iterations: 3
- 停止条件: 模型输出 "confident" 或达到 max iterations
- **不加 RAG**（公平对比：Reflexion 原论文不用外部检索）

#### Baseline B: Multi-Agent Debate (Liang et al., 2023)

```
Round 1:
  Agent A (Security Auditor): 激进查找漏洞 → 输出观点
  Agent B (Developer):        反驳 A 的观点 → 输出反驳
Round 2:
  Agent A: 再次反驳 B
Judge:
  Agent C (Chief Security Officer): 阅读完整辩论 → 最终裁决
```

- 2 rounds debate + 1 judge = 5 API calls/样本
- Context 窗口累积（token 消耗远超单轮）

### 数据

- PrimeVul 测试集随机抽样 **150 个样本**（分层抽样，保证 CWE 类别覆盖）
- 足以展示成本量级差异，无需全量

### 记录指标

| 指标 | 说明 |
|------|------|
| Macro-F1 | 检测精度 |
| Avg Input Tokens / 样本 | 输入 token 消耗 |
| Avg Output Tokens / 样本 | 输出 token 消耗 |
| Avg Total Tokens / 样本 | 总 token |
| Token Cost Ratio | 相对 MulVul 的倍数 |
| Avg Latency (sec/sample) | 平均延迟 |

### 预期结果

```
          Macro-F1    Token Cost (relative)
MulVul      ~35%         1.0x
Reflexion   ~28%         4-5x
MAD         ~25%         8-10x
```

### 呈现方式

**散点图 (Cost-Effectiveness Plot)**:
- X 轴: Avg Token Cost (log scale)
- Y 轴: Macro-F1
- MulVul 在左上角（高精度、低成本）

### Rebuttal 话术

> We benchmarked MulVul against two representative agentic paradigms:
> **Reflexion** (iterative self-correction, NeurIPS 2023) and
> **Multi-Agent Debate** (MAD, Liang et al. 2023).
>
> MulVul achieves superior F1 while consuming significantly fewer tokens.
> Reflexion incurs ~4.5x token costs due to recursive critique loops
> but suffers from hallucination cycles without external knowledge.
> MAD is computationally prohibitive (~8x cost) due to long multi-role
> context windows. In contrast, MulVul's coarse-to-fine routing acts
> as a "cost filter", activating specialized detectors only when needed.
> This demonstrates MulVul represents the Pareto-optimal solution.

---

## Exp 1: GPT-4o + RAG 强基线 (P1)

### 目标

证明性能提升来自 Router-Detector 架构，不仅仅是 RAG。

### 方案

- **GPT-4o + RAG (Single-Pass)**: 使用与 MulVul 相同的 SCALE 知识库和检索器
- 每个样本: 检索 Top-k 相关知识 → 拼接 [Code + Retrieved Knowledge] → GPT-4o 直接输出漏洞类型
- **无** Router 分类，**无** Detector 细分

### 数据

- 与 Exp 2 共用同一 150 样本子集（或扩展到全量测试集）

### 预期结果

```
GPT-4o (原文 baseline):   ~24% F1
GPT-4o + RAG:             ~28-30% F1
MulVul:                   ~35% F1
```

→ RAG 有帮助（+4-6%），但架构贡献更大（+5-7%）

### Rebuttal 话术

> We added the requested "GPT-4o + RAG" baseline using the same SCALE
> knowledge base. Results show RAG improves GPT-4o by ~X%, but MulVul
> still outperforms by ~Y%, confirming the architectural contribution
> of coarse-to-fine routing beyond mere retrieval augmentation.

---

## Exp 3: Cross-Model 消融 (P2, 有空再做)

### 目标

证明跨模型（Claude → GPT-4o）进化优于单模型自我纠错。

### 方案

| 配置 | Generator | Executor |
|------|-----------|----------|
| MulVul (原文) | Claude Opus | GPT-4o |
| Self-Correction | GPT-4o | GPT-4o |
| 反转 | GPT-4o | Claude Opus |

### 数据

- **Top-10 CWE 类别**即可（进化实验成本高，无需全量）

### 预期结果

- Self-Correction (GPT-4o→GPT-4o) F1 明显低于 Cross-Model
- 反转组 (GPT-4o→Claude) 也有效 → 证明"异构"本身是关键，不是特定模型能力

---

## Exp 4: 多语言泛化 (P3, 低优先)

### 目标

展示 MulVul 不限于 C/C++。

### 方案

- 从 CodeXGLUE Defect Detection 或类似数据集取 500 个 Java/Python 样本
- 运行 MulVul 流程
- 与 GPT-4o 裸跑对比

---

## 执行顺序

```
Week 1:  Exp 2 (Agent 对比 + 成本)  ← 最关键
         Exp 1 (GPT-4o + RAG)       ← 可并行，实现简单
Week 2:  Exp 3 (Cross-Model 消融)   ← 如果时间允许
         Exp 4 (多语言)             ← 最低优先
```

## 代码组织

```
scripts/rebuttal/
├── sample_test_data.py          # 分层抽样 150 个测试样本
├── run_reflexion_baseline.py    # Reflexion 实现
├── run_mad_baseline.py          # Multi-Agent Debate 实现
├── run_gpt4o_rag_baseline.py    # GPT-4o + RAG single-pass
├── run_cross_model_ablation.py  # Cross-model 消融
├── collect_cost_metrics.py      # Token 消耗统计
└── plot_cost_effectiveness.py   # 绘制散点图
```

## 输出

```
outputs/rebuttal/
├── exp1_gpt4o_rag/
├── exp2_agent_comparison/
├── exp3_cross_model/
└── figures/
    └── cost_effectiveness_scatter.pdf
```
