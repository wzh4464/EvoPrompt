# 评估指标功能总结

## 🎯 核心改进

### 1. 新增多分类评估模块

**文件**: `src/evoprompt/evaluators/multiclass_metrics.py`

**功能**:
- ✅ `ClassMetrics`: 单个类别的TP/FP/TN/FN和P/R/F1
- ✅ `MultiClassMetrics`: 多分类场景的完整指标
- ✅ **Macro-F1**: 所有类别同等重要 (推荐用于漏洞检测)
- ✅ **Weighted-F1**: 按样本数加权
- ✅ **Micro-F1**: 全局计算 (等同于accuracy)
- ✅ Per-class详细指标
- ✅ 混淆矩阵
- ✅ 分类报告

**使用示例**:
```python
from evoprompt.evaluators.multiclass_metrics import MultiClassMetrics

metrics = MultiClassMetrics()
for pred, actual in predictions:
    metrics.add_prediction(pred, actual)

# 获取三种F1
macro_f1 = metrics.compute_macro_f1()      # ⭐ 推荐
weighted_f1 = metrics.compute_weighted_f1()
micro_f1 = metrics.compute_micro_f1()

# 打印完整报告
metrics.print_report()
```

---

### 2. 更新三层检测评估器

**文件**: `src/evoprompt/detectors/three_layer_detector.py`

**改进**:
- ✅ 集成`MultiClassMetrics`
- ✅ 每层都计算Macro/Weighted/Micro F1
- ✅ Per-class详细指标
- ✅ `verbose=True`模式打印详细报告

**输出格式**:
```json
{
  "layer1": {
    "accuracy": 0.80,
    "macro_f1": 0.65,       // ⭐ 推荐关注
    "weighted_f1": 0.75,
    "micro_f1": 0.80,
    "macro_precision": 0.63,
    "macro_recall": 0.67
  },
  "layer1_per_class": {
    "Memory": {
      "precision": 0.85,
      "recall": 0.80,
      "f1_score": 0.825,
      "support": 50
    },
    ...
  }
}
```

---

### 3. 更新主训练脚本

**文件**: `scripts/train_three_layer.py`

**改进**:
- ✅ 使用`verbose=True`评估
- ✅ 自动打印详细的Macro/Weighted/Micro F1
- ✅ 标注推荐指标 ⭐

**示例输出**:
```
======================================================================
EVALUATION RESULTS
======================================================================

Total Samples: 50
Full Path Accuracy: 0.4200

----------------------------------------------------------------------
Layer 1 (Major Category)
----------------------------------------------------------------------
  Accuracy:        0.8000
  Macro-F1:        0.6500 ⭐ (推荐)
  Weighted-F1:     0.7500
  Micro-F1:        0.8000
  Macro-Precision: 0.6300
  Macro-Recall:    0.6700

======================================================================
💡 推荐关注指标: Macro-F1
   原因: 漏洞检测中类别不平衡，Macro-F1确保所有类别都被重视
======================================================================
```

---

### 4. 新增演示脚本

**文件**: `scripts/demo_f1_metrics.py`

**功能**:
- ✅ 场景1: 平衡数据集
- ✅ 场景2: 不平衡数据集 - 多数类好
- ✅ 场景3: 不平衡数据集 - 少数类好
- ✅ 场景4: 三层检测实际应用
- ✅ 对比分析三种F1的差异

**运行**:
```bash
uv run python scripts/demo_f1_metrics.py
```

---

### 5. 新增文档

**文件**: `METRICS_GUIDE.md`

**内容**:
- ✅ Macro/Weighted/Micro F1的定义和公式
- ✅ 为什么在漏洞检测中必须使用Macro-F1
- ✅ 实际案例对比
- ✅ EvoPrompt中的实现
- ✅ 论文报告建议
- ✅ 相关文献

---

## 📊 三种F1对比

### 场景：不平衡数据集

```
Benign (安全代码):     900 samples, F1 = 0.95
Vulnerable (漏洞代码): 100 samples, F1 = 0.30
```

### 计算结果

| 指标 | 值 | 说明 |
|------|-----|------|
| **Macro-F1** | **0.625** | ✅ 揭示模型在Vulnerable上的差表现 |
| Weighted-F1 | 0.885 | ⚠️ 被多数类主导，产生误导性高分 |
| Micro-F1 | 0.840 | ℹ️ 等同于准确率 |

### 结论

在漏洞检测中**必须使用Macro-F1**，因为：
1. 数据严重不平衡 (安全代码 >> 漏洞代码)
2. 少数类（漏洞）同样重要，不能忽视
3. Weighted-F1会掩盖少数类的失败

---

## 🚀 使用指南

### 快速开始

```bash
# 1. 查看F1指标演示
uv run python scripts/demo_f1_metrics.py

# 2. 运行评估 (自动打印Macro/Weighted/Micro F1)
uv run python scripts/train_three_layer.py --eval-samples 50
```

### 在代码中使用

```python
from evoprompt.detectors.three_layer_detector import ThreeLayerDetector, ThreeLayerEvaluator
from evoprompt.data.dataset import PrimevulDataset

# 创建检测器
detector = ThreeLayerDetector(...)

# 创建评估器
dataset = PrimevulDataset("data/primevul_1percent_sample/dev.txt", "dev")
evaluator = ThreeLayerEvaluator(detector, dataset)

# 评估 (verbose=True打印详细指标)
metrics = evaluator.evaluate(sample_size=50, verbose=True)

# 获取Macro-F1
layer1_macro_f1 = metrics["layer1"]["macro_f1"]
layer2_macro_f1 = metrics["layer2"]["macro_f1"]
layer3_macro_f1 = metrics["layer3"]["macro_f1"]
```

---

## 📝 论文报告建议

### 表格格式

| 配置 | Layer 1<br>Macro-F1 | Layer 2<br>Macro-F1 | Layer 3<br>Macro-F1 | Full Path<br>Accuracy |
|------|---------------------|---------------------|---------------------|-----------------------|
| 基线 | 0.65 | 0.55 | 0.45 | 0.30 |
| + RAG | 0.72 (+11%) | 0.63 (+15%) | 0.52 (+16%) | 0.40 (+33%) |
| + 训练 | 0.80 (+23%) | 0.70 (+27%) | 0.60 (+33%) | 0.45 (+50%) |
| RAG+训练 | 0.88 (+35%) | 0.78 (+42%) | 0.68 (+51%) | 0.55 (+83%) |

### 说明文本

```
我们使用Macro-F1作为主要评估指标，因为漏洞检测数据集存在严重的
类别不平衡问题（安全代码占比 > 90%）。Macro-F1能够确保模型在
所有CWE类别（包括罕见但关键的少数类）上都保持良好性能，避免被
多数类主导的误导性高分。

此外，我们也报告了Weighted-F1和Accuracy作为辅助参考指标。
详细的Per-class F1分数见附录表X。
```

---

## 🔗 相关文件

### 代码

- `src/evoprompt/evaluators/multiclass_metrics.py` - 多分类指标模块 ⭐
- `src/evoprompt/detectors/three_layer_detector.py` - 更新的评估器
- `src/evoprompt/evaluators/__init__.py` - 导出模块

### 脚本

- `scripts/train_three_layer.py` - 主脚本 (verbose评估)
- `scripts/demo_f1_metrics.py` - F1指标演示 ⭐

### 文档

- `METRICS_GUIDE.md` - 详细指标指南 ⭐
- `START_HERE.md` - 更新的入口文档
- `README_INDEX.md` - 更新的文档索引

---

## 💡 关键要点

1. **Macro-F1 是漏洞检测的首选指标** ⭐
   - 所有类别同等重要
   - 避免被多数类误导

2. **系统自动计算三种F1**
   - 提供完整的性能视图
   - 便于论文报告

3. **Per-class指标帮助定位问题**
   - 找出表现差的类别
   - 针对性优化

4. **详细的可视化报告**
   - verbose=True模式
   - 清晰标注推荐指标

---

## 🎓 学习资源

### 运行演示

```bash
# F1指标对比演示 (推荐!)
uv run python scripts/demo_f1_metrics.py
```

### 阅读文档

1. `METRICS_GUIDE.md` - 完整的指标指南
2. `QUICKSTART.md` - 快速开始
3. `THREE_LAYER_README.md` - 三层检测详解

### 实际运行

```bash
# 评估并查看详细指标
uv run python scripts/train_three_layer.py --use-rag --eval-samples 50
```

---

## ✅ 总结

### 新增功能

1. ✅ 完整的多分类评估模块
2. ✅ Macro/Weighted/Micro F1自动计算
3. ✅ Per-class详细指标
4. ✅ 可视化报告
5. ✅ F1对比演示
6. ✅ 详细文档

### 使用建议

- **主要指标**: Macro-F1 ⭐
- **辅助指标**: Weighted-F1, Accuracy
- **详细分析**: Per-class F1

### 论文报告

- 报告所有三种F1
- 强调Macro-F1
- 说明选择原因

---

**开始使用**: 运行 `uv run python scripts/demo_f1_metrics.py` 查看演示！
