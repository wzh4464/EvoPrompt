# EvoPrompt: Evolutionary Prompt Optimization for Vulnerability Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EvoPrompt is a modern prompt evolution framework specifically designed for vulnerability detection tasks. It uses evolutionary algorithms to automatically optimize prompts for better performance on security code analysis.

## Key Features

- **Vulnerability Detection**: Primary focus on code security analysis with SVEN and Primevul datasets
- **Evolutionary Algorithms**: Genetic Algorithm (GA) and Differential Evolution (DE)
- **Unified Response Parsing**: `ResponseParser` class for consistent LLM output handling
- **Multi-Agent System**: Coordinated agents for complex detection tasks
- **RAG Integration**: Knowledge-base retrieval for enhanced detection

## Quick Start

### Installation

```bash
git clone <repository-url>
cd EvoPrompt

# Install with uv (recommended)
uv sync
```

### Configuration

Create a `.env` file:

```env
API_BASE_URL=https://api-inference.modelscope.cn/v1/
API_KEY=your-api-key-here
MODEL_NAME=Qwen/Qwen3-Coder-480B-A35B-Instruct
```

### Basic Usage

```bash
# Run main entry point
uv run python main.py

# Run training scripts
./scripts/run_quick_training.sh
./scripts/run_full_training.sh

# Run tests
uv run pytest tests/
```

## Project Structure

```
EvoPrompt/
├── src/evoprompt/           # Core package
│   ├── algorithms/          # GA, DE, Coevolution
│   ├── core/                # Evolution engine, prompt tracker
│   ├── data/                # Dataset, sampler, CWE categories
│   ├── llm/                 # LLM clients (sync & async)
│   ├── utils/               # ResponseParser, checkpoint, text
│   ├── evaluators/          # Vulnerability evaluation
│   ├── detectors/           # Three-layer detector, heuristic filter
│   ├── rag/                 # RAG knowledge base & retriever
│   ├── multiagent/          # Multi-agent coordination
│   └── workflows/           # Detection workflows
├── scripts/                 # Utility scripts
│   ├── run_quick_training.sh
│   ├── run_full_training.sh
│   ├── preprocess_primevul_comment4vul.py
│   └── verify_*.py          # Verification scripts
├── tests/                   # Test suite
├── docs/                    # Documentation
│   ├── QUICKSTART.md
│   ├── SYSTEM_OVERVIEW.md
│   └── images/              # Diagrams and charts
├── data/                    # Datasets (gitignored)
├── outputs/                 # Experiment outputs (gitignored)
├── main.py                  # Main entry point
└── pyproject.toml           # Project configuration
```

## Response Parsing API

The unified `ResponseParser` handles all LLM response parsing:

```python
from evoprompt.utils.response_parsing import ResponseParser, ParsedResponse

# Full parsing
result: ParsedResponse = ResponseParser.parse("CWE-120 buffer overflow vulnerability")
print(result.is_vulnerable)      # True
print(result.cwe_category)       # "Buffer Errors"
print(result.vulnerability_label) # "1"

# Individual extraction
label = ResponseParser.extract_vulnerability_label("The code is benign")  # "0"
category = ResponseParser.extract_cwe_category("SQL injection detected")  # "Injection"
```

### Supported CWE Categories

- Benign, Buffer Errors, Injection, Memory Management
- Pointer Dereference, Integer Errors, Concurrency Issues
- Path Traversal, Cryptography Issues, Information Exposure, Other

## Datasets

| Dataset | Samples | CWE Types | Location |
|---------|---------|-----------|----------|
| SVEN | ~1,000 | 9 | `data/vul_detection/sven/` |
| Primevul | 24,000+ | Multiple | `data/primevul/primevul/` |

## Evolutionary Algorithms

### Differential Evolution (DE)
- Continuous optimization for fine-tuning prompts
- Configurable mutation factor and crossover rate

### Genetic Algorithm (GA)
- Population-based optimization
- Tournament selection and crossover operators

### Coevolution
- Multi-population evolution for diverse solutions

## Development

```bash
# Run all tests
uv run pytest tests/

# Run response parsing tests
RUN_RESPONSE_PARSING_TESTS=1 uv run pytest tests/test_response_parsing.py -v

# Type checking
uv run mypy src/evoprompt

# Linting
uv run flake8 src/evoprompt
```

## Documentation

See `docs/` for detailed documentation:

- [QUICKSTART.md](docs/QUICKSTART.md) - Getting started guide
- [SYSTEM_OVERVIEW.md](docs/SYSTEM_OVERVIEW.md) - Architecture overview
- [WORKFLOW.md](docs/WORKFLOW.md) - Workflow documentation
- [INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md) - Integration guide

## Requirements

- Python 3.9+
- uv package manager
- API access (configured in .env)

## License

MIT License - see [LICENSE](LICENSE) file for details.
