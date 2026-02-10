# EvoPrompt Project - Claude Development Notes

## 项目概述

EvoPrompt是一个专注于漏洞检测的prompt进化优化框架，使用进化算法自动优化安全代码分析的prompt性能。

## 环境管理

**重要**: 所有Python命令都使用`uv run`来执行。

```bash
uv run python main.py
uv run pytest tests/
```

## 项目结构

```
EvoPrompt/
├── src/evoprompt/           # 核心包
│   ├── algorithms/          # GA, DE, Coevolution
│   ├── core/                # 进化引擎, prompt追踪
│   ├── data/                # 数据集, 采样器, CWE分类
│   ├── llm/                 # LLM客户端 (同步/异步)
│   ├── utils/               # ResponseParser, checkpoint
│   ├── evaluators/          # 漏洞评估
│   ├── detectors/           # 三层检测器, 并行分层检测
│   ├── meta/                # Meta-learning (错误累积, prompt调优)
│   ├── rag/                 # RAG知识库
│   ├── multiagent/          # 多智能体协调
│   └── workflows/           # 检测工作流
├── scripts/                 # 工具脚本
│   ├── run_quick_training.sh
│   ├── run_full_training.sh
│   └── verify_*.py
├── tests/                   # 测试套件
├── docs/                    # 文档
├── data/                    # 数据集 (gitignored)
├── outputs/                 # 实验输出 (gitignored)
├── main.py                  # 主入口
└── pyproject.toml           # 项目配置
```

## API配置

通过`.env`文件管理：

```bash
API_BASE_URL=https://api-inference.modelscope.cn/v1/
API_KEY=your-key-here
MODEL_NAME=Qwen/Qwen3-Coder-480B-A35B-Instruct
```

## 响应解析 API

使用统一的 `ResponseParser` 类：

```python
from evoprompt.utils.response_parsing import ResponseParser

# 完整解析
result = ResponseParser.parse("CWE-120 buffer overflow vulnerability")
result.is_vulnerable      # True
result.cwe_category       # "Buffer Errors"
result.vulnerability_label # "1"

# 单独提取
ResponseParser.extract_vulnerability_label("benign")  # "0"
ResponseParser.extract_cwe_category("SQL injection")  # "Injection"
```

### 支持的 CWE 分类

- Benign, Buffer Errors, Injection, Memory Management
- Pointer Dereference, Integer Errors, Concurrency Issues
- Path Traversal, Cryptography Issues, Information Exposure, Other

## 常用命令

```bash
# 运行主程序
uv run python main.py

# 运行训练
./scripts/run_quick_training.sh
./scripts/run_full_training.sh

# 测试
uv run pytest tests/
RUN_RESPONSE_PARSING_TESTS=1 uv run pytest tests/test_response_parsing.py -v

# 类型检查
uv run mypy src/evoprompt

# Lint
uv run flake8 src/evoprompt
```

## 输出位置

实验结果保存在 `outputs/` 目录：

```
outputs/<experiment>/
├── experiment_summary.json
├── prompt_evolution.jsonl
├── best_prompts.txt
└── llm_call_history.json
```

## 开发注意事项

1. **必须使用 `uv run`**: 所有Python命令都通过uv执行
2. **API密钥安全**: .env 文件已在 .gitignore 中
3. **异常处理**: 开发阶段少用 try/except，让异常直接抛出便于调试
4. **测试**: 修改代码后运行 `uv run pytest tests/`

## 并行分层检测系统

### 架构

```
Code → Comment4Vul增强 → Layer1 (并行, top-k) → Layer2 (并行) → Layer3 (CWE) → 选择策略
                                                                              ↓
                                                          ErrorAccumulator → MetaLearning
```

### 核心组件

- `detectors/scoring.py` - ScoredPrediction, DetectionPath, SelectionStrategy
- `detectors/parallel_hierarchical_detector.py` - 并行分层检测器
- `detectors/hierarchical_coordinator.py` - 协调器 (含 meta-learning)
- `meta/error_accumulator.py` - 错误累积和模式识别
- `meta/prompt_tuner.py` - Meta-learning prompt 优化

### 使用示例

```python
from evoprompt.llm.async_client import AsyncLLMClient
from evoprompt.detectors import create_coordinator

# 创建协调器
client = AsyncLLMClient()
coordinator = create_coordinator(client, enable_meta_learning=True)

# 检测单个样本
result = coordinator.detect_single(code, ground_truth="CWE-120")

# 批量检测
results = coordinator.detect_batch(codes, ground_truths)

# 获取统计
print(coordinator.get_statistics_summary())
```

## TODO: Selection Strategies

- [ ] WeightedVotingSelection - 加权投票 (考虑不同层级、不同类别的权重)
- [ ] EnsembleSelection - 多策略组合 (组合多个策略的结果)
- [ ] Layer-specific weights - 层级权重调整 (基于历史准确率动态调整)
- [ ] ConfidenceCalibration - 置信度校准 (基于验证集校准输出概率)

## TODO: Meta-Learning Enhancements

- [ ] Contrastive examples - 在 tuning 时加入对比示例
- [ ] Few-shot prompt injection - 动态注入少量示例
- [ ] Prompt template evolution - 使用进化算法优化模板结构
- [ ] Multi-objective optimization - 同时优化准确率和特异性

## TODO: Code Enhancement

- [ ] Comment4Vul integration - 集成代码注释增强
- [ ] Static analysis hints - 集成静态分析结果
- [ ] Dataflow annotations - 添加数据流标注
