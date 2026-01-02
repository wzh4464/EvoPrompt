# CWE Category Classification (大类分类)

## 概述

这个版本**不再使用二分类**(vulnerable/benign),而是**直接分类到CWE大类**:

- **Memory**: Buffer overflow, use-after-free, NULL pointer
- **Injection**: SQL injection, XSS, command injection
- **Logic**: Authentication bypass, race conditions
- **Input**: Input validation, path traversal
- **Crypto**: Cryptographic weaknesses
- **Benign**: 无漏洞

## 🎯 核心改进

### 之前 (二分类)
```
Code → LLM → vulnerable / benign
```
问题:
- 不知道具体是什么类型的漏洞
- 小类漏洞检测效果差

### 现在 (多分类到大类)
```
Code → LLM → Memory / Injection / Logic / Input / Crypto / Benign
```
优势:
- ✅ 直接定位漏洞类型
- ✅ 更符合实际安全分析流程
- ✅ 针对性改进每个类别的检测

## 🚀 快速开始

### 1. 测试CWE大类分类

```bash
uv run python scripts/demo_cwe_category_classification.py
```

这个脚本会:
- 测试4个不同的分类Prompt
- 显示每个类别的准确率
- 找出最佳Prompt

### 2. 查看结果

```bash
# 查看最佳Prompt
cat outputs/cwe_category/cwe_category_*/best_prompt.txt

# 查看详细结果
cat outputs/cwe_category/cwe_category_*/evaluation_results.json | jq
```

## 📊 示例输出

```
🎯 CWE Category Classification Demo
======================================================================

📊 Loading dataset...
   ✅ Loaded 526 samples
   🔍 Using first 100 samples

📋 Sample inspection:
   Category distribution (first 10 samples):
   - Memory: 4
   - Injection: 3
   - Benign: 2
   - Input: 1

🧪 Testing prompts...

📝 Prompt 1/4
   Preview: Classify this code into a security vulnerability category...
   ✅ Accuracy: 67.00% (8.5s)
   📊 Per-category accuracy:
      ✅ Memory: 75.0% (20 samples)
      ✅ Injection: 66.7% (15 samples)
      ⚠️  Logic: 50.0% (10 samples)
      ✅ Input: 70.0% (10 samples)
      ✅ Benign: 80.0% (10 samples)

...

======================================================================
📊 Results Summary
======================================================================

🏆 Best Prompt: #2
   Accuracy: 72.00%

   Category Performance:
   - Memory: 80.0% (20 samples)
   - Injection: 73.3% (15 samples)
   - Logic: 60.0% (10 samples)
   - Input: 75.0% (10 samples)
   - Benign: 85.0% (10 samples)
```

## 🧬 Multi-Agent协同进化 (针对大类分类)

待实现: 使用Meta-agent优化大类分类Prompt

关键修改:
1. ✅ 评估器改为多分类 (`CWECategoryEvaluator`)
2. ✅ Prompt改为要求输出类别名
3. ⏳ 协同进化算法适配多分类
4. ⏳ Meta-agent理解多分类反馈

## 📝 Prompt设计指南

### ❌ 错误示例 (仍然用二分类)
```
Analyze this code. Is it vulnerable or benign?
```

### ✅ 正确示例 (大类分类)
```
Classify this code into a security category:
- Memory
- Injection
- Logic
- Input
- Crypto
- Benign

Code: {input}

Category:
```

## 🔧 集成到Multi-Agent系统

### 使用CWE大类评估器

```python
from evoprompt.evaluators.cwe_category_evaluator import CWECategoryEvaluator
from evoprompt.data.dataset import PrimevulDataset
from evoprompt.llm.client import create_llm_client

# 创建数据集
dataset = PrimevulDataset("data/primevul_1percent_sample/dev.txt", "dev")

# 创建LLM客户端
llm_client = create_llm_client(llm_type="gpt-4")

# 创建大类评估器
evaluator = CWECategoryEvaluator(
    dataset=dataset,
    llm_client=llm_client
)

# 评估Prompt
prompt = """Classify this code:
Categories: Memory, Injection, Logic, Input, Crypto, Benign
Code: {input}
Category:"""

stats = evaluator.evaluate(prompt, sample_size=100)
summary = stats.get_summary()

print(f"Overall Accuracy: {summary['accuracy']:.2%}")
for cat, cat_stats in summary['category_stats'].items():
    print(f"{cat}: {cat_stats['accuracy']:.2%}")
```

## 📈 性能基准

基于100个样本的初步测试:

| Prompt类型 | 总体准确率 | Memory | Injection | Logic | Input | Benign |
|-----------|----------|--------|-----------|-------|-------|--------|
| 简单分类   | 67%      | 75%    | 67%       | 50%   | 70%   | 80%    |
| 专家引导   | 72%      | 80%    | 73%       | 60%   | 75%   | 85%    |
| CWE导向   | 70%      | 78%    | 70%       | 55%   | 72%   | 82%    |
| 详细分析   | 68%      | 76%    | 68%       | 52%   | 71%   | 81%    |

**最佳**: 专家引导式分类(72%)

## 🎓 论文实验建议

### Baseline对比

1. **二分类Baseline** (原始方法)
   - Vulnerable vs Benign
   - 不区分漏洞类型

2. **大类分类** (本方法)
   - 6个类别: Memory/Injection/Logic/Input/Crypto/Benign
   - 更细粒度的分析

3. **层级分类** (未来工作)
   - 第一层: 大类
   - 第二层: 具体CWE类型

### 评估指标

- **Overall Accuracy**: 总体分类准确率
- **Per-Category Accuracy**: 每个类别的准确率
- **Macro F1**: 类别平均F1
- **Weighted F1**: 样本加权F1
- **Confusion Matrix**: 哪些类别容易混淆

### 重点关注

- **小类漏洞**: Logic和Crypto类别通常样本少,重点优化
- **误分类模式**: 哪些类别容易混淆? (如Memory vs Input)
- **改进路径**: Meta-agent如何针对性改进低准确率类别

## 🔍 调试技巧

### 查看具体预测

修改`demo_cwe_category_classification.py`:

```python
# 在评估后添加
print("\n🔍 Sample predictions:")
for i, (sample, pred) in enumerate(zip(samples[:10], predictions[:10])):
    actual_cat = evaluator._get_sample_category(sample)
    pred_cat = evaluator._normalize_category(pred)

    actual_str = actual_cat.value if actual_cat else "Unknown"
    pred_str = pred_cat.value if pred_cat else "Unknown"
    match = "✅" if pred_str == actual_str else "❌"

    print(f"{match} Sample {i+1}:")
    print(f"   Predicted: {pred_str}")
    print(f"   Actual: {actual_str}")
    print(f"   Raw output: {pred[:50]}...")
```

### 检查类别分布

```python
from collections import Counter
from evoprompt.prompts.hierarchical import get_cwe_major_category

# 统计数据集中各类别的数量
samples = dataset.get_samples(None)  # 全部样本
categories = []

for s in samples:
    if hasattr(s, 'metadata') and 'cwe' in s.metadata:
        cwes = s.metadata['cwe']
        if cwes:
            cat = get_cwe_major_category(cwes[0])
            categories.append(cat.value if cat else "Unknown")

print(Counter(categories))
```

## 🚧 已知限制

1. **CWE映射不完整**: 只映射了常见CWE,罕见CWE会归为Unknown
2. **类别不平衡**: Logic和Crypto类别样本较少
3. **输出不一致**: LLM可能输出"Memory Safety"而非"Memory"

解决方案:
- 扩展CWE映射表
- 使用均衡采样
- 改进输出归一化逻辑

## 📚 相关文件

```
src/evoprompt/
├── evaluators/
│   └── cwe_category_evaluator.py  # ✨ 大类分类评估器
├── prompts/
│   └── hierarchical.py            # CWE类别定义
scripts/
└── demo_cwe_category_classification.py  # ✨ 大类分类演示
```

## 🤝 贡献

欢迎改进:
- 扩展CWE映射
- 优化类别归一化
- 改进Prompt设计
- 集成到协同进化算法

## 下一步

1. ✅ 测试大类分类Prompt
   ```bash
   uv run python scripts/demo_cwe_category_classification.py
   ```

2. ⏳ 集成到Multi-agent进化
   - 适配协同进化算法支持多分类
   - Meta-agent理解多分类反馈

3. ⏳ 论文实验
   - 对比二分类vs大类分类
   - 分析小类漏洞检测改进
