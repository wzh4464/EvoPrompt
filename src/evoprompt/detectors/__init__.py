"""Detectors module for vulnerability detection.

Provides three-layer hierarchical detection with optional RAG enhancement,
parallel hierarchical detection, and meta-learning optimization.
"""

from .three_layer_detector import ThreeLayerDetector, ThreeLayerEvaluator
from .rag_three_layer_detector import RAGThreeLayerDetector
from .topk_three_layer_detector import TopKThreeLayerDetector
from .heuristic_filter import VulnerabilityHeuristicFilter, HeuristicResult

# Parallel hierarchical detection
from .scoring import (
    ScoredPrediction,
    DetectionPath,
    SelectionStrategy,
    MaxConfidenceSelection,
    ThresholdSelection,
    create_selection_strategy,
)
from .parallel_hierarchical_detector import (
    ParallelHierarchicalDetector,
    ParallelDetectorConfig,
    HierarchicalPromptSet,
    CodeEnhancer,
    NoOpEnhancer,
    create_parallel_detector,
)
from .hierarchical_coordinator import (
    HierarchicalDetectionCoordinator,
    CoordinatorConfig,
    DetectionResult,
    create_coordinator,
)
from .code_enhancer import (
    CodeEnhancerBase,
    Comment4VulEnhancer,
    StaticAnalysisEnhancer,
    ChainedEnhancer,
    EnhancementConfig,
    create_comment4vul_enhancer,
)

# Unified detection pipeline
from .pipeline import (
    DetectionContext,
    PipelineStep,
    CodeEnhancementStep,
    PromptBuildStep,
    LLMInferenceStep,
    ResponseParseStep,
    PathSelectionStep,
    DetectionPipeline,
    SequentialStrategy,
    ParallelStrategy,
)

__all__ = [
    # Original detectors
    "ThreeLayerDetector",
    "ThreeLayerEvaluator",
    "RAGThreeLayerDetector",
    "TopKThreeLayerDetector",
    "VulnerabilityHeuristicFilter",
    "HeuristicResult",
    # Scoring and selection
    "ScoredPrediction",
    "DetectionPath",
    "SelectionStrategy",
    "MaxConfidenceSelection",
    "ThresholdSelection",
    "create_selection_strategy",
    # Parallel hierarchical detector
    "ParallelHierarchicalDetector",
    "ParallelDetectorConfig",
    "HierarchicalPromptSet",
    "CodeEnhancer",
    "NoOpEnhancer",
    "create_parallel_detector",
    # Coordinator
    "HierarchicalDetectionCoordinator",
    "CoordinatorConfig",
    "DetectionResult",
    "create_coordinator",
    # Code enhancers
    "CodeEnhancerBase",
    "Comment4VulEnhancer",
    "StaticAnalysisEnhancer",
    "ChainedEnhancer",
    "EnhancementConfig",
    "create_comment4vul_enhancer",
    # Unified detection pipeline
    "DetectionContext",
    "PipelineStep",
    "CodeEnhancementStep",
    "PromptBuildStep",
    "LLMInferenceStep",
    "ResponseParseStep",
    "PathSelectionStep",
    "DetectionPipeline",
    "SequentialStrategy",
    "ParallelStrategy",
]
