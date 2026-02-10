"""Core components for EvoPrompt."""

from .evolution import EvolutionEngine
from .evaluator import Evaluator
from .dataset import Dataset
from .baseline import (
    BaselineConfig,
    BaselineManager,
    BaselineSnapshot,
    ComparisonResult,
)
from .experiment import (
    ExperimentConfig,
    ExperimentManager,
    ArtifactStore,
)

__all__ = [
    "EvolutionEngine",
    "Evaluator",
    "Dataset",
    # Baseline management
    "BaselineConfig",
    "BaselineManager",
    "BaselineSnapshot",
    "ComparisonResult",
    # Experiment management
    "ExperimentConfig",
    "ExperimentManager",
    "ArtifactStore",
]