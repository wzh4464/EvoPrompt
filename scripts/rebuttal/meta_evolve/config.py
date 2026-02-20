"""Configuration for meta-learning prompt evolution."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class MetaEvolveConfig:
    # Paths
    test_data: str = "data/primevul/primevul/primevul_test_fixed.jsonl"
    train_data: str = "data/primevul/primevul/primevul_train_fixed.jsonl"
    knowledge_cwe: str = "data/primevul/knowledge-cwe.jsonl"
    kb_output: str = "outputs/rebuttal/meta_evolve/cwe_knowledge.json"
    output_dir: str = "outputs/rebuttal/meta_evolve"

    # Training loop
    epochs: int = 3
    batch_size: int = 50
    meta_update_freq: int = 4  # update rules every N batches

    # kNN
    knn_k: int = 20
    candidate_cwes: int = 10  # max candidate CWEs per sample

    # Code
    max_code_chars: int = 6000

    # LLM - detection
    detect_model: str = "gpt-4o"
    detect_temperature: float = 0.0
    detect_max_tokens: int = 300
    max_workers: int = 16

    # LLM - meta prompter
    meta_model: Optional[str] = None  # defaults to detect_model
    meta_temperature: float = 0.3
    meta_max_tokens: int = 4000

    # Rules
    max_rules: int = 30
    initial_rules: List[str] = field(default_factory=list)

    # EMA for error analysis
    ema_alpha: float = 0.9

    # Failure buffer
    max_failures: int = 30

    # Resume
    resume_from: Optional[str] = None

    def __post_init__(self):
        if self.meta_model is None:
            self.meta_model = self.detect_model

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def resolve_path(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        return self.project_root / path
