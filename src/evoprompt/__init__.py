"""EvoPrompt: Evolutionary Prompt Optimization Framework."""

__version__ = "0.1.0"
__author__ = "EvoPrompt Team"

from .core.evolution import EvolutionEngine
from .core.evaluator import Evaluator  
from .algorithms.genetic import GeneticAlgorithm
from .algorithms.differential import DifferentialEvolution
from .llm.client import SVENLLMClient, sven_llm_init, sven_llm_query
from .workflows.vulnerability_detection import VulnerabilityDetectionWorkflow

__all__ = [
    "EvolutionEngine",
    "Evaluator", 
    "GeneticAlgorithm",
    "DifferentialEvolution",
    "SVENLLMClient",
    "sven_llm_init", 
    "sven_llm_query",
    "VulnerabilityDetectionWorkflow",
]