# EvoPrompt: Evolutionary Prompt Optimization for Vulnerability Detection

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EvoPrompt is a modern prompt evolution framework specifically designed for vulnerability detection tasks. It uses evolutionary algorithms to automatically optimize prompts for better performance on security code analysis.

## 🎯 Focus Areas

- **Vulnerability Detection**: Primary focus on code security analysis
- **SVEN Dataset**: Support for CWE-based vulnerability classification
- **Primevul Dataset**: Large-scale vulnerability detection dataset support

## ✨ Key Features

- **Modern Architecture**: Built with modern Python packaging (pyproject.toml, src layout)
- **SVEN Integration**: Compatible with SVEN submodule API calls
- **Evolutionary Algorithms**: Genetic Algorithm (GA) and Differential Evolution (DE)
- **Real-time Tracking**: Complete prompt evolution tracking and analysis
- **Balanced Sampling**: Smart data sampling for imbalanced datasets

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd EvoPrompt

# Install with uv (recommended)
uv sync
```

### Configuration

Create a `.env` file with your API configuration:

```env
API_BASE_URL=https://newapi.pockgo.com/v1
API_KEY=your-api-key-here
BACKUP_API_BASE_URL=https://newapi.aicohere.org/v1
MODEL_NAME=kimi-k2-code
```

### Basic Usage

```bash
# Run main entry point
uv run python main.py

# Run Primevul 1% sampling experiment
uv run python scripts/run_primevul_1percent.py

# Run full pipeline
uv run python scripts/run_full_pipeline.py
```

## 📊 Supported Datasets

### SVEN Dataset
- **CWE Types**: 9 common vulnerability types
- **Format**: JSONL with function context
- **Location**: `data/vul_detection/sven/`

### Primevul Dataset  
- **Scale**: 24,000+ vulnerability samples
- **Format**: JSONL with code analysis
- **Location**: `data/primevul/primevul/`

## 🧬 Evolutionary Algorithms

### Differential Evolution (DE)
- Continuous optimization approach
- Good for fine-tuning prompts
- Configurable mutation and crossover rates

### Genetic Algorithm (GA)
- Population-based optimization
- Diverse prompt generation
- Tournament selection and crossover

## 📁 Project Structure

```
EvoPrompt/
├── src/evoprompt/           # Modern package structure
│   ├── algorithms/          # Evolutionary algorithms
│   ├── core/               # Core functionality
│   ├── data/               # Data processing
│   ├── llm/                # SVEN-compatible LLM client
│   └── workflows/          # Vulnerability detection workflow
├── data/                   # Datasets
│   ├── vul_detection/      # Processed vulnerability data
│   └── primevul/           # Primevul dataset
├── sven/                   # SVEN submodule
├── demo_primevul_1percent.py # Demo script
└── run_primevul_1percent.py  # Production script
```

## 🔧 API Usage

### SVEN-Compatible Client

```python
from evoprompt import sven_llm_init, sven_llm_query

# Initialize client
client = sven_llm_init()

# Single query
result = sven_llm_query("Analyze this code for vulnerabilities", client, task=True)

# Batch queries
results = sven_llm_query(["query1", "query2"], client, task=True)
```

### Vulnerability Detection Workflow

```python
from evoprompt import VulnerabilityDetectionWorkflow

# Configure experiment
config = {
    "algorithm": "de",
    "population_size": 10,
    "max_generations": 5,
    "llm_type": "sven"
}

# Run evolution
workflow = VulnerabilityDetectionWorkflow(config)
results = workflow.run()
```

## 📈 Performance Tracking

EvoPrompt provides comprehensive tracking of the evolution process:

- **Real-time Logging**: All prompt updates recorded in `prompt_evolution.jsonl`
- **Best Prompts**: History of best-performing prompts
- **Fitness Tracking**: Complete fitness evolution over generations
- **LLM Call History**: All API calls logged for analysis

## 🧪 Example Results

### Primevul 1% Demo Results
- **Total Samples**: 100 (balanced: 50 benign + 50 vulnerable)
- **Evolution Generations**: 4
- **LLM Calls**: 924
- **Output Files**: 5 detailed tracking files

## 🔄 Development Commands

```bash
# All Python commands use uv run
uv run python script_name.py

# Run tests
uv run pytest tests/

# Check code quality
uv run ruff check src/
uv run mypy src/
```

## 🧪 Response Parsing Harness

Use `scripts/verify_response_parsing.py` to run a handful of real LLM calls and
confirm that the parser recovers the intended labels:

```bash
uv run python scripts/verify_response_parsing.py \
  --llm-type openai \
  --model-name gpt-4o-mini \
  --sample-file data/primevul_1percent_sample/dev_sample.jsonl \
  --max-samples 3 \
  --temperature 0.0 \
  --verbose
```

- To run with full PrimeVul samples and output archiving:

```bash
uv run python scripts/verify_response_parsing.py \
  --model-name gpt-4o \
  --sample-file data/primevul_1percent_sample/dev_sample.jsonl \
  --max-samples 10 \
  --temperature 0.0 \
  --verbose \
  --use-cwe-major \
  --output-json result.json
```

- Runs against a real LLM by default; ensure `API_KEY`, `API_BASE_URL`, `MODEL_NAME` are set in `.env`.
- Use `--use-cwe-major` mode to verify major-category classification parsing.
- For offline debugging, pass `--mock-response "benign"` to reuse a single response (no real API validation).
- Use `--output-json` to save per-sample prompts, raw responses, and parsed results for further analysis.

> By default, `pytest` will not run `tests/test_response_parsing.py`.
> To include it in the test run, set `RUN_RESPONSE_PARSING_TESTS=1` before the command.

## 📋 Requirements

- Python 3.11+
- uv package manager
- API access (configured in .env)

## 🤝 Contributing

1. Focus on vulnerability detection improvements
2. Maintain SVEN compatibility
3. Follow modern Python practices
4. Add comprehensive tests

## 📄 License

MIT License - see LICENSE file for details

## 🔗 Related Projects

- **SVEN**: Vulnerability detection submodule
- **Primevul**: Large-scale vulnerability dataset

---

**Note**: This project is specifically focused on vulnerability detection. For other NLP tasks, consider using general-purpose prompt optimization frameworks.