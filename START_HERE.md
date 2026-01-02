# 🚀 从这里开始 - EvoPrompt

## 欢迎!

你现在看到的是 **EvoPrompt v2.0** - 基于Multi-Agent协同进化的三层层级漏洞检测系统。

## 系统特点

✅ **三层层级检测**: Major → Middle → CWE, 24+可训练prompt
✅ **RAG增强**: 自动构建知识库，检索相似示例，+10-15%准确率
✅ **Multi-Agent训练**: GPT-4 + Claude 4.5协同优化，+15-25%准确率
✅ **完全自动化**: RAG和Scale可通过参数自动启用/关闭
✅ **端到端训练**: 从数据加载到结果保存全自动

## 快速开始 (5分钟)

### 步骤1: 配置环境

创建 `.env` 文件:

```bash
cat > .env << EOF
API_BASE_URL=https://api-inference.modelscope.cn/v1/
API_KEY=your_api_key_here
MODEL_NAME=Qwen/Qwen3-Coder-480B-A35B-Instruct
EOF
```

### 步骤2: 快速测试

```bash
uv run python scripts/test_quick.py
```

**预期输出**:
```
🧪 EvoPrompt 快速测试
======================================================================
✅ 环境配置:
   Model: Qwen/Qwen3-Coder-480B-A35B-Instruct

测试1: 基础三层检测 ✅ 通过
测试2: RAG增强检测 ✅ 通过
测试3: Scale增强 ✅ 通过

🎉 所有测试通过!
```

### 步骤3: 评估性能

```bash
# 基础评估
uv run python scripts/train_three_layer.py --eval-samples 50

# RAG增强评估
uv run python scripts/train_three_layer.py --use-rag --eval-samples 50
```

**完成!** 🎉

---

## 核心脚本

### 🧪 测试脚本

```bash
# 快速测试所有功能 (2-3分钟) - 推荐第一步
uv run python scripts/test_quick.py
```

### 🎯 主脚本

```bash
# 评估 (5-10分钟)
uv run python scripts/train_three_layer.py --use-rag --eval-samples 50

# 训练 (2-4小时)
uv run python scripts/train_three_layer.py \
    --train --use-rag --kb-from-dataset \
    --population-size 5 --max-generations 20
```

---

## 参数说明

### RAG控制

```bash
# 启用RAG (自动构建默认知识库)
--use-rag

# 从数据集构建知识库
--use-rag --kb-from-dataset --kb-samples-per-category 5

# 使用已有知识库
--use-rag --kb-path outputs/my_kb.json
```

### Scale控制

```bash
# 启用Scale增强
--use-scale
```

### 训练控制

```bash
# 启用训练
--train --population-size 5 --max-generations 20
```

### 组合使用

```bash
# 仅评估 (基线)
uv run python scripts/train_three_layer.py --eval-samples 50

# 评估 + RAG
uv run python scripts/train_three_layer.py --use-rag --eval-samples 50

# 评估 + RAG + Scale
uv run python scripts/train_three_layer.py --use-rag --use-scale --eval-samples 50

# 完整训练 (所有功能)
uv run python scripts/train_three_layer.py \
    --train --use-rag --use-scale --kb-from-dataset \
    --population-size 5 --max-generations 20 --eval-samples 100
```

---

## 文档导航

### 📚 必读文档 (按顺序)

1. **[QUICKSTART.md](QUICKSTART.md)** (10分钟)
   - 环境配置
   - 基本使用
   - 参数说明

2. **[SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md)** (15分钟)
   - 所有脚本说明
   - 推荐工作流程
   - 快速命令参考

3. **[COMPLETE_FLOW.md](COMPLETE_FLOW.md)** (20分钟)
   - 完整系统流程
   - RAG自动构建
   - Scale自动启用
   - 参数控制详解

### 📖 深入文档

4. **[THREE_LAYER_README.md](THREE_LAYER_README.md)** (30分钟)
   - 三层检测详解
   - 训练策略
   - 性能基准

5. **[RAG_README.md](RAG_README.md)** (25分钟)
   - RAG原理
   - 知识库构建
   - 检索策略

6. **[MULTIAGENT_README.md](MULTIAGENT_README.md)** (30分钟)
   - Multi-agent架构
   - 协同进化
   - 统计反馈

### 🔍 参考文档

- **[SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)** - 系统总览
- **[WORKFLOW.md](WORKFLOW.md)** - 工作流程图
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - 集成指南
- **[README_INDEX.md](README_INDEX.md)** - 文档索引

---

## 常见使用场景

### 场景1: 我想快速验证系统 (5分钟)

```bash
uv run python scripts/test_quick.py
```

### 场景2: 我想评估性能 (10分钟)

```bash
uv run python scripts/train_three_layer.py --use-rag --eval-samples 50
```

### 场景3: 我想对比不同配置 (1小时)

```bash
# 基线
uv run python scripts/train_three_layer.py \
    --eval-samples 50 --output-dir outputs/baseline

# + RAG
uv run python scripts/train_three_layer.py \
    --use-rag --eval-samples 50 --output-dir outputs/with_rag

# + Scale
uv run python scripts/train_three_layer.py \
    --use-scale --eval-samples 50 --output-dir outputs/with_scale

# RAG + Scale
uv run python scripts/train_three_layer.py \
    --use-rag --use-scale --eval-samples 50 --output-dir outputs/rag_scale
```

### 场景4: 我想训练优化prompt (2-4小时)

```bash
uv run python scripts/train_three_layer.py \
    --train \
    --use-rag \
    --kb-from-dataset \
    --population-size 5 \
    --max-generations 20 \
    --eval-samples 100
```

---

## 预期性能

### 主要指标：Macro-F1 ⭐

**为什么使用Macro-F1？**
- 漏洞检测数据严重不平衡（安全代码 >> 漏洞代码）
- Macro-F1确保所有类别（包括少数类）都被重视
- 避免被多数类主导的误导性高分

详见：[METRICS_GUIDE.md](METRICS_GUIDE.md)

### 性能基准（Macro-F1）

| 配置 | Layer 1 | Layer 2 | Layer 3 | Full Path | 时间 |
|------|---------|---------|---------|-----------|------|
| 基线 | 0.65 | 0.55 | 0.45 | 0.30 | 5分钟 |
| + RAG | 0.72 | 0.63 | 0.52 | 0.40 | 5分钟 |
| + 训练 | 0.80 | 0.70 | 0.60 | 0.45 | 2小时 |
| RAG+训练 | 0.88 | 0.78 | 0.68 | 0.55 | 3小时 |

**注**: 系统会同时报告Macro/Weighted/Micro F1，但推荐关注Macro-F1

---

## 输出文件

每次运行会生成:

```
outputs/three_layer_eval_rag_20250122_143000/
├── config.json      # 运行配置
├── metrics.json     # 评估指标
├── prompts.json     # Prompt集合
└── prompts.txt      # 可读Prompt

# 如使用RAG
outputs/knowledge_base.json  # 知识库
```

---

## 故障排查

### 问题1: API调用失败

**检查**:
```bash
cat .env | grep API_KEY
```

**确保包含**:
```
API_KEY=your_key_here
```

### 问题2: 测试失败

**运行**:
```bash
uv run python scripts/test_quick.py
```

**查看错误信息并参考**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### 问题3: 找不到数据集

**检查**:
```bash
ls data/primevul_1percent_sample/
# 应该看到: train.txt  dev.txt  test.txt
```

---

## 下一步

1. ✅ 运行 `test_quick.py` 验证环境
2. 📊 运行评估了解性能
3. 📚 阅读 `QUICKSTART.md` 和 `COMPLETE_FLOW.md`
4. 🚀 根据需求选择配置
5. 📈 运行实验并分析结果

---

## 推荐学习路径

### 新手路径 (1小时)

```
1. 阅读本文档 (5分钟)
    ↓
2. 运行 test_quick.py (5分钟)
    ↓
3. 阅读 QUICKSTART.md (10分钟)
    ↓
4. 阅读 COMPLETE_FLOW.md (20分钟)
    ↓
5. 运行评估实验 (20分钟)
```

### 研究路径 (1天)

```
1. 新手路径 (1小时)
    ↓
2. 阅读核心技术文档 (2小时)
   - THREE_LAYER_README.md
   - RAG_README.md
   - MULTIAGENT_README.md
    ↓
3. 设计实验方案 (1小时)
    ↓
4. 运行对比实验 (4小时)
    ↓
5. 分析结果 (2小时)
```

---

## 快速命令参考

```bash
# 测试
uv run python scripts/test_quick.py

# 基础评估
uv run python scripts/train_three_layer.py --eval-samples 50

# RAG评估
uv run python scripts/train_three_layer.py --use-rag --eval-samples 50

# 快速训练
uv run python scripts/train_three_layer.py \
    --train --population-size 3 --max-generations 5 --eval-samples 30

# 完整训练
uv run python scripts/train_three_layer.py \
    --train --use-rag --kb-from-dataset \
    --population-size 5 --max-generations 20 --eval-samples 100
```

---

## 获取帮助

1. 查看 [README_INDEX.md](README_INDEX.md) 找到相关文档
2. 查看 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 解决问题
3. 运行 `test_quick.py` 验证环境
4. 查看示例脚本的输出

---

**准备好了吗？**

开始使用 EvoPrompt:

```bash
# 第一步: 快速测试
uv run python scripts/test_quick.py

# 第二步: 评估性能
uv run python scripts/train_three_layer.py --use-rag --eval-samples 50

# 第三步: 查看文档
cat QUICKSTART.md
cat COMPLETE_FLOW.md
```

**祝你使用愉快!** 🎉
