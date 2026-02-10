# EvoPrompt 快速开始指南

## 系统架构

```
输入代码
    ↓
[可选] 知识库构建 (RAG)
    ↓
[可选] Scale增强
    ↓
三层检测
    ├─→ Layer 1: 大类 (Memory/Injection/Logic/Input/Crypto/Benign)
    ├─→ Layer 2: 中类 (Buffer Overflow/SQL Injection/etc.)
    └─→ Layer 3: CWE (CWE-120/CWE-89/etc.)
    ↓
[可选] Multi-Agent训练优化
    ↓
输出: CWE + 检测路径
```

## 前置准备

### 1. 环境配置

创建 `.env` 文件:

```bash
# ModelScope API (推荐)
API_BASE_URL=https://api-inference.modelscope.cn/v1/
API_KEY=your_api_key_here
MODEL_NAME=Qwen/Qwen3-Coder-480B-A35B-Instruct

# Multi-agent训练需要 (可选)
META_MODEL_NAME=claude-4.5
```

### 2. 数据准备

确保数据集存在:

```bash
data/primevul_1percent_sample/
├── train.txt  # 526 样本
├── dev.txt    # 50 样本
└── test.txt   # 50 样本
```

## 使用流程

### 🧪 步骤0: 快速测试 (推荐第一步)

验证系统是否正常工作:

```bash
uv run python scripts/test_quick.py
```

**预期输出**:
```
🧪 EvoPrompt 快速测试
======================================================================
✅ 环境配置:
   Model: Qwen/Qwen3-Coder-480B-A35B-Instruct

======================================================================
测试1: 基础三层检测
======================================================================
...
✅ 检测完成!
   Layer 1: Memory
   Layer 2: Buffer Overflow
   Layer 3: CWE-120

======================================================================
测试2: RAG增强检测
======================================================================
...
🔎 RAG检索信息:
   Layer 1: 检索到 2 个示例

======================================================================
测试3: Scale增强
======================================================================
...

📊 测试总结
======================================================================
   基础检测: ✅ 通过
   RAG检测: ✅ 通过
   Scale增强: ✅ 通过

🎉 所有测试通过!
```

### 📊 步骤1: 仅评估 (不训练)

#### 1.1 基础评估

最简单的使用方式:

```bash
uv run python scripts/train_three_layer.py \
    --eval-samples 50
```

**说明**:
- 使用默认prompt
- 不使用RAG
- 不使用Scale
- 仅评估50个样本

#### 1.2 RAG增强评估

使用RAG提升准确性:

```bash
uv run python scripts/train_three_layer.py \
    --use-rag \
    --eval-samples 50
```

**说明**:
- 自动构建默认知识库
- 每层检索2个相似示例
- 预期准确性 +10-15%

#### 1.3 完整评估 (RAG + Scale)

最佳配置:

```bash
uv run python scripts/train_three_layer.py \
    --use-rag \
    --use-scale \
    --eval-samples 50
```

### 🚀 步骤2: 训练优化

#### 2.1 快速训练 (小规模测试)

```bash
uv run python scripts/train_three_layer.py \
    --train \
    --population-size 3 \
    --max-generations 5 \
    --eval-samples 30
```

**预计时间**: ~30分钟

**说明**:
- 种群大小: 3
- 代数: 5
- 评估样本: 30

#### 2.2 完整训练 (论文实验)

```bash
uv run python scripts/train_three_layer.py \
    --train \
    --use-rag \
    --kb-from-dataset \
    --kb-samples-per-category 5 \
    --population-size 5 \
    --max-generations 20 \
    --eval-samples 100
```

**预计时间**: 数小时

**说明**:
- 使用RAG
- 从数据集构建知识库 (每类5个样本)
- 种群大小: 5
- 代数: 20
- 评估样本: 100

#### 2.3 最佳配置 (RAG + Scale + 训练)

```bash
uv run python scripts/train_three_layer.py \
    --train \
    --use-rag \
    --use-scale \
    --kb-from-dataset \
    --kb-samples-per-category 5 \
    --population-size 5 \
    --max-generations 20 \
    --batch-size 20 \
    --meta-improve-interval 3 \
    --eval-samples 100
```

## 参数说明

### 数据集参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train-file` | `data/.../train.txt` | 训练数据 |
| `--eval-file` | `data/.../dev.txt` | 评估数据 |
| `--eval-samples` | `50` | 评估样本数 |

### RAG参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use-rag` | `False` | 启用RAG |
| `--kb-path` | `outputs/knowledge_base.json` | 知识库路径 |
| `--kb-from-dataset` | `False` | 从数据集构建KB |
| `--kb-samples-per-category` | `3` | 每类采样数 |
| `--rag-top-k` | `2` | 检索top-k |
| `--rag-retriever-type` | `lexical` | 检索器类型 |

### Scale参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use-scale` | `False` | 启用Scale增强 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train` | `False` | 运行训练 |
| `--population-size` | `5` | 种群大小 |
| `--max-generations` | `10` | 最大代数 |
| `--elite-size` | `1` | 精英保留数 |
| `--mutation-rate` | `0.3` | 变异率 |
| `--batch-size` | `10` | 批处理大小 |
| `--meta-improve-interval` | `3` | Meta优化间隔 |
| `--meta-improve-count` | `2` | 每次Meta优化数 |

### 输出参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | 自动生成 | 输出目录 |

## 输出文件

每次运行会生成如下文件:

```
outputs/three_layer_eval_rag_20250122_143000/
├── config.json      # 运行配置
├── metrics.json     # 评估指标
├── prompts.json     # Prompt集合 (JSON)
└── prompts.txt      # Prompt集合 (可读)
```

### metrics.json 内容

```json
{
  "total_samples": 50,
  "layer1_accuracy": 0.85,
  "layer2_accuracy": 0.75,
  "layer3_accuracy": 0.65,
  "full_path_accuracy": 0.55,
  "results": [...]
}
```

## 常见使用场景

### 场景1: 快速验证系统

```bash
# 快速测试
uv run python scripts/test_quick.py
```

### 场景2: 评估不同配置

```bash
# 基线 (无RAG, 无Scale)
uv run python scripts/train_three_layer.py --eval-samples 50

# + RAG
uv run python scripts/train_three_layer.py --use-rag --eval-samples 50

# + Scale
uv run python scripts/train_three_layer.py --use-scale --eval-samples 50

# + RAG + Scale
uv run python scripts/train_three_layer.py --use-rag --use-scale --eval-samples 50
```

### 场景3: 训练优化

```bash
# 快速训练测试
uv run python scripts/train_three_layer.py \
    --train \
    --population-size 3 \
    --max-generations 5 \
    --eval-samples 30

# 完整训练
uv run python scripts/train_three_layer.py \
    --train \
    --use-rag \
    --kb-from-dataset \
    --population-size 5 \
    --max-generations 20 \
    --eval-samples 100
```

### 场景4: 论文实验

```bash
# 对比实验: 基线 vs RAG vs 训练 vs RAG+训练

# 1. 基线
uv run python scripts/train_three_layer.py \
    --eval-samples 100 \
    --output-dir outputs/exp1_baseline

# 2. + RAG
uv run python scripts/train_three_layer.py \
    --use-rag \
    --eval-samples 100 \
    --output-dir outputs/exp2_rag

# 3. + 训练
uv run python scripts/train_three_layer.py \
    --train \
    --max-generations 20 \
    --eval-samples 100 \
    --output-dir outputs/exp3_train

# 4. RAG + 训练 (最佳)
uv run python scripts/train_three_layer.py \
    --train \
    --use-rag \
    --kb-from-dataset \
    --max-generations 20 \
    --eval-samples 100 \
    --output-dir outputs/exp4_rag_train
```

## 预期性能

| 配置 | Layer 1 | Layer 2 | Layer 3 | Full Path | 时间 |
|------|---------|---------|---------|-----------|------|
| 基线 | 75% | 60% | 50% | 30% | ~5分钟 |
| + RAG | 80% | 70% | 60% | 40% | ~5分钟 |
| + 训练 | 85% | 75% | 65% | 45% | ~2小时 |
| RAG+训练 | 90% | 80% | 70% | 55% | ~2小时 |

## 故障排查

### 问题1: API调用失败

**错误信息**: `API_KEY not found`

**解决**:
```bash
# 检查 .env 文件
cat .env

# 确保包含:
API_KEY=your_key_here
```

### 问题2: 找不到数据集

**错误信息**: `Dataset not found`

**解决**:
```bash
# 检查数据集
ls data/primevul_1percent_sample/

# 应该看到:
# train.txt  dev.txt  test.txt
```

### 问题3: 知识库构建失败

**错误信息**: `Knowledge base creation failed`

**解决**:
```bash
# 使用默认知识库 (不使用 --kb-from-dataset)
uv run python scripts/train_three_layer.py --use-rag

# 或手动构建
uv run python scripts/build_knowledge_base.py \
    --source default \
    --output outputs/kb.json
```

### 问题4: 训练卡住

**症状**: 长时间无输出

**解决**:
1. 检查API是否正常
2. 减小batch-size: `--batch-size 5`
3. 减小种群: `--population-size 3`
4. 查看详细日志

## 下一步

1. ✅ 运行快速测试验证系统
2. 📊 评估不同配置找到最佳设置
3. 🚀 运行完整训练优化prompt
4. 📈 分析结果并发表论文

## 相关文档

- `INTEGRATION_GUIDE.md` - 完整集成指南
- `THREE_LAYER_README.md` - 三层检测详解
- `RAG_README.md` - RAG增强详解
- `MULTIAGENT_README.md` - Multi-agent训练详解
