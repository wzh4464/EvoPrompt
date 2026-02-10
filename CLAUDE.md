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
│   ├── detectors/           # 三层检测器
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
