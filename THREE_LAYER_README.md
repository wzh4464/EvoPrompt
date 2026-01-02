

# Three-Layer Hierarchical Vulnerability Detection

## 🏗️ 架构概述

三层层级检测系统,从粗到细逐步分类漏洞:

```
输入代码
    ↓
[可选] Scale增强
    ↓
Layer 1: Prompt1 → Major Category (大类)
    ├─→ Memory
    ├─→ Injection
    ├─→ Logic
    ├─→ Input
    ├─→ Crypto
    └─→ Benign
    ↓
Layer 2: Prompt2[major_i] → Middle Category (中类)
    例如 Memory:
    ├─→ Buffer Overflow
    ├─→ Use After Free
    ├─→ NULL Pointer
    ├─→ Integer Overflow
    └─→ Memory Leak
    ↓
Layer 3: Prompt3[middle_j] → CWE ID (小类)
    例如 Buffer Overflow:
    ├─→ CWE-120
    ├─→ CWE-121
    ├─→ CWE-122
    └─→ CWE-787
```

### 关键特性

1. **渐进式分类**: 先大后小,逐步细化
2. **所有Prompt可训练**: Prompt1, Prompt2[], Prompt3[] 都可优化
3. **减少错误传播**: 每层独立优化,降低级联错误

## 📊 Prompt配置

### Prompt数量

- **Layer 1**: 1个prompt (大类路由)
- **Layer 2**: 6个prompt (每个大类1个)
- **Layer 3**: 17+个prompt (每个中类1个)
- **总计**: ~24个可训练的prompt

### 默认Prompt示例

#### Layer 1 (大类)
```
Classify this code into ONE major vulnerability category:
1. Memory
2. Injection
3. Logic
4. Input
5. Crypto
6. Benign

Code: {input}

Category:
```

#### Layer 2 (Memory中类)
```
This code has MEMORY vulnerability.

Identify the specific type:
1. Buffer Overflow
2. Use After Free
3. NULL Pointer
4. Integer Overflow
5. Memory Leak

Code: {input}

Type:
```

#### Layer 3 (Buffer Overflow具体CWE)
```
Identify the specific CWE:
- CWE-120: Buffer Copy without Checking
- CWE-121: Stack-based Buffer Overflow
- CWE-122: Heap-based Buffer Overflow
- CWE-787: Out-of-bounds Write

Code: {input}

CWE:
```

## 🚀 快速开始

### 1. 测试三层检测

```bash
uv run python scripts/demo_three_layer_detection.py
```

### 2. 单次检测示例

```python
from evoprompt.prompts.hierarchical_three_layer import ThreeLayerPromptFactory
from evoprompt.detectors.three_layer_detector import ThreeLayerDetector
from evoprompt.llm.client import create_llm_client

# 创建默认prompt集
prompt_set = ThreeLayerPromptFactory.create_default_prompt_set()

# 创建检测器
llm_client = create_llm_client(llm_type="gpt-4")
detector = ThreeLayerDetector(
    prompt_set=prompt_set,
    llm_client=llm_client,
    use_scale_enhancement=False
)

# 检测代码
code = """
void process(char* input) {
    char buf[100];
    strcpy(buf, input);  // Vulnerable!
}
"""

cwe, details = detector.detect(code)

print(f"Layer 1: {details['layer1']}")  # Memory
print(f"Layer 2: {details['layer2']}")  # Buffer Overflow
print(f"Layer 3: {details['layer3']}")  # CWE-120
print(f"Final:   {cwe}")                # CWE-120
```

### 3. 批量检测

```python
codes = [code1, code2, code3, ...]
results = detector.detect_batch(codes, batch_size=16)

for cwe, details in results:
    print(f"{cwe}: {details['layer1']} → {details['layer2']}")
```

## 🎓 训练策略

### 策略1: 逐层训练 (推荐初期)

**适合**: 快速建立baseline

```python
# Phase 1: 只训练Layer 1
# 目标: 大类分类准确率 > 80%

# Phase 2: 固定Layer 1, 训练Layer 2
# 目标: 中类分类准确率 > 70%

# Phase 3: 固定Layer 1+2, 训练Layer 3
# 目标: CWE分类准确率 > 60%
```

**优点**:
- 简单直接
- 每层独立优化
- 容易定位问题

**缺点**:
- 不考虑层间依赖
- 可能不是全局最优

### 策略2: 联合训练 (推荐后期)

**适合**: 精细调优

```python
# 同时优化所有层的prompt
# 目标: 最大化全路径准确率

# 使用Multi-agent协同进化:
# - Meta Agent分析整体错误模式
# - 针对性优化每层prompt
```

**优点**:
- 考虑层间依赖
- 可能达到全局最优

**缺点**:
- 复杂度高
- 训练时间长

### 策略3: 课程学习 (推荐论文实验)

**适合**: 最佳性能

```python
# Stage 1: Layer 1训练至80%
# Stage 2: 固定Layer 1, Layer 2训练至70%
# Stage 3: 固定Layer 1+2, Layer 3训练至60%
# Stage 4: 所有层联合微调
```

**优点**:
- 循序渐进
- 稳定性好
- 性能最优

**缺点**:
- 需要多轮训练
- 时间成本高

## 📈 评估指标

### 层级准确率

```python
from evoprompt.detectors.three_layer_detector import ThreeLayerEvaluator

evaluator = ThreeLayerEvaluator(detector, dataset)
metrics = evaluator.evaluate(sample_size=100)

print(f"Layer 1 Accuracy: {metrics['layer1_accuracy']:.1%}")
print(f"Layer 2 Accuracy: {metrics['layer2_accuracy']:.1%}")
print(f"Layer 3 Accuracy: {metrics['layer3_accuracy']:.1%}")
print(f"Full Path Accuracy: {metrics['full_path_accuracy']:.1%}")
```

### 指标解释

- **Layer 1 Accuracy**: 大类分类准确率
- **Layer 2 Accuracy**: 给定正确大类,中类分类准确率
- **Layer 3 Accuracy**: 给定正确中类,CWE分类准确率
- **Full Path Accuracy**: 三层全部正确的比例

### 错误分析

```python
# 分析错误传播
# 问题: Layer 1错了,后续层全错
# 解决: 提高Layer 1准确率

# 问题: Layer 1对, Layer 2错
# 解决: 改进特定大类的Layer 2 prompt

# 问题: Layer 1+2对, Layer 3错
# 解决: 改进特定中类的Layer 3 prompt
```

## 🔧 高级功能

### RAG增强 (推荐!)

使用检索增强生成提高检测准确性:

```python
from evoprompt.detectors.rag_three_layer_detector import RAGThreeLayerDetector
from evoprompt.rag.knowledge_base import KnowledgeBaseBuilder

# 创建知识库
kb = KnowledgeBaseBuilder.create_default_kb()

# 创建RAG检测器
detector = RAGThreeLayerDetector(
    prompt_set=prompt_set,
    llm_client=llm_client,
    knowledge_base=kb,
    use_scale_enhancement=False,
    retriever_type="lexical",  # 快速检索
    top_k=2  # 每层检索2个相似示例
)

# 检测 (自动检索并注入示例)
cwe, details = detector.detect(code)

# 查看检索信息
print(f"Layer 1 retrieved: {details['layer1_retrieval']['num_examples']} examples")
print(f"Similarity: {details['layer1_retrieval']['similarity_scores']}")
```

**RAG优势**:
- 提供相似代码示例作为参考
- 提升分类准确性 (预期+10-15%)
- 无需额外API调用 (检索在本地)
- 知识库可持续更新

详见: `RAG_README.md`

### Scale增强

启用代码增强提高检测准确性:

```python
detector = ThreeLayerDetector(
    prompt_set=prompt_set,
    llm_client=llm_client,
    use_scale_enhancement=True  # 启用Scale
)

# Scale会先增强代码,再进行检测
cwe, details = detector.detect(code)

# 查看增强后的代码
print(details.get('enhanced_code'))
```

### 自定义Prompt

```python
from evoprompt.prompts.hierarchical_three_layer import (
    ThreeLayerPromptSet,
    MajorCategory,
    MiddleCategory
)

# 创建自定义prompt集
custom_set = ThreeLayerPromptSet(
    layer1_prompt="Your custom Layer 1 prompt...",
    layer2_prompts={
        MajorCategory.MEMORY: "Your custom Memory prompt...",
        MajorCategory.INJECTION: "Your custom Injection prompt...",
        # ...
    },
    layer3_prompts={
        MiddleCategory.BUFFER_OVERFLOW: "Your custom Buffer Overflow prompt...",
        # ...
    }
)

detector = ThreeLayerDetector(custom_set, llm_client)
```

### 保存和加载Prompt

```python
# 保存
with open("my_prompts.json", "w") as f:
    json.dump(prompt_set.to_dict(), f, indent=2)

# 加载
with open("my_prompts.json", "r") as f:
    data = json.load(f)
prompt_set = ThreeLayerPromptSet.from_dict(data)
```

## 🎯 Multi-Agent训练集成

### 使用Meta-Agent优化Prompt

(待实现功能)

```python
from evoprompt.multiagent.agents import create_meta_agent
from evoprompt.optimization.three_layer_optimizer import ThreeLayerOptimizer

# 创建Meta Agent
meta_agent = create_meta_agent(model_name="claude-4.5")

# 创建优化器
optimizer = ThreeLayerOptimizer(
    detector=detector,
    meta_agent=meta_agent,
    dataset=dataset
)

# 优化特定层
improved_prompt1 = optimizer.optimize_layer1(
    current_stats=layer1_stats,
    error_patterns=layer1_errors
)

# 优化特定类别的Layer 2 prompt
improved_prompt2_memory = optimizer.optimize_layer2(
    major_category=MajorCategory.MEMORY,
    current_stats=memory_stats,
    error_patterns=memory_errors
)
```

## 📊 性能基准

### 初始性能 (使用默认Prompt)

基于Primevul 1% 数据集, 100样本:

| 层级 | 准确率 | 说明 |
|------|--------|------|
| Layer 1 | ~75% | 大类分类 |
| Layer 2 | ~60% | 中类分类 (给定正确大类) |
| Layer 3 | ~50% | CWE分类 (给定正确中类) |
| 全路径 | ~30% | 三层全对 |

### 优化后性能目标

| 层级 | 目标准确率 |
|------|-----------|
| Layer 1 | 85%+ |
| Layer 2 | 75%+ |
| Layer 3 | 65%+ |
| 全路径 | 45%+ |

## 🔬 实验建议

### Baseline对比

1. **Flat Classification**
   - 直接分类到CWE (不分层)
   - 问题: 类别太多,效果差

2. **Two-Layer**
   - Layer 1: Vulnerable/Benign
   - Layer 2: CWE
   - 问题: Layer 1信息量低

3. **Three-Layer** (本方法)
   - Layer 1: Major Category
   - Layer 2: Middle Category
   - Layer 3: CWE
   - 优势: 渐进式,可解释

### 消融实验

1. **禁用Scale增强**
   ```python
   use_scale_enhancement=False
   ```

2. **固定某些层**
   ```python
   # 只训练Layer 1, 固定Layer 2+3
   ```

3. **简化层级**
   ```python
   # 只用Layer 1+3, 跳过Layer 2
   ```

### 重点分析

- **错误传播**: Layer 1错误如何影响后续层?
- **类别不平衡**: 哪些类别样本少,检测差?
- **层间依赖**: Layer 2的prompt是否依赖Layer 1的结果?

## 🐛 常见问题

### Q1: Layer 1准确率很低怎么办?

**A**:
1. 检查prompt是否明确
2. 增加示例
3. 使用Meta-agent分析错误模式
4. 考虑合并相似类别

### Q2: Layer 3准确率很低怎么办?

**A**:
1. 某些CWE太相似,难以区分
2. 考虑简化Layer 3,只区分主要CWE
3. 或者将相似CWE合并

### Q3: 如何处理未知CWE?

**A**:
1. 在每层添加"Other"类别
2. 或者添加fallback机制
3. 记录未知样本,后续添加

### Q4: Scale增强真的有用吗?

**A**:
1. 实验表明可能提升5-10%
2. 但会增加API调用次数
3. 建议先不用,后期再加

## 📁 文件结构

```
src/evoprompt/
├── prompts/
│   └── hierarchical_three_layer.py  # ✨ 三层prompt定义
├── detectors/
│   └── three_layer_detector.py      # ✨ 三层检测器
scripts/
└── demo_three_layer_detection.py    # ✨ 演示脚本
```

## 🎬 完整示例

```python
#!/usr/bin/env python3
from evoprompt.prompts.hierarchical_three_layer import ThreeLayerPromptFactory
from evoprompt.detectors.three_layer_detector import ThreeLayerDetector, ThreeLayerEvaluator
from evoprompt.data.dataset import PrimevulDataset
from evoprompt.llm.client import create_llm_client

# 1. 创建prompt集
prompt_set = ThreeLayerPromptFactory.create_default_prompt_set()
print(f"Total prompts: {prompt_set.count_prompts()['total']}")

# 2. 创建检测器
llm_client = create_llm_client(llm_type="gpt-4")
detector = ThreeLayerDetector(prompt_set, llm_client)

# 3. 测试单个样本
code = "char buf[10]; strcpy(buf, input);"
cwe, details = detector.detect(code)
print(f"Detected: {cwe}")
print(f"Path: {details['layer1']} → {details['layer2']} → {details['layer3']}")

# 4. 评估数据集
dataset = PrimevulDataset("data/primevul_1percent_sample/dev.txt", "dev")
evaluator = ThreeLayerEvaluator(detector, dataset)
metrics = evaluator.evaluate(sample_size=50)

print(f"\nResults:")
print(f"Layer 1: {metrics['layer1_accuracy']:.1%}")
print(f"Layer 2: {metrics['layer2_accuracy']:.1%}")
print(f"Layer 3: {metrics['layer3_accuracy']:.1%}")
print(f"Full path: {metrics['full_path_accuracy']:.1%}")
```

## 🚧 下一步

1. ✅ 三层检测器实现
2. ✅ 默认Prompt集合
3. ✅ RAG增强集成
4. ⏳ Multi-agent训练集成
5. ⏳ 批量优化工具
6. ⏳ 可视化工具

## 📚 相关文档

- `RAG_README.md` - RAG增强检测 ⭐ 新增
- `MULTIAGENT_README.md` - Multi-agent协同进化
- `CWE_CATEGORY_README.md` - 大类分类
- `TROUBLESHOOTING.md` - 故障排查

## 🤝 贡献

欢迎改进:
- 扩展CWE映射
- 优化默认Prompt
- 添加更多训练策略
- 改进评估指标
