"""Prompt management for vulnerability detection."""

from .hierarchical import (
    HierarchicalPrompt,
    CWECategory,
    PromptHierarchy,
)
from .mop_manager import MoPromptManager

# Seed prompts for evolution
from .seed_prompts import (
    LAYER1_SEED_PROMPTS,
    LAYER2_SEED_PROMPTS,
    LAYER3_SEED_PROMPTS,
    get_seed_prompts_for_category,
    get_all_layer1_seeds,
)
from .seed_loader import (
    SeedPromptLoader,
    SeedPromptConfig,
    load_seeds_for_ga,
    get_hierarchical_seeds,
)

# Task-aware evolution prompts
from .evolution_prompts import (
    TaskContext,
    TASK_CONTEXTS,
    get_task_context,
    build_crossover_prompt,
    build_mutation_prompt,
    build_initialization_prompt,
)

# Prompt contract validation
from .contract import (
    PromptContract,
    PromptContractValidator,
    ValidationResult,
)

# Unified prompt template architecture
from .template import PromptTemplate, PromptSection, PromptMetadata
from .prompt_set import PromptSet

__all__ = [
    "HierarchicalPrompt",
    "CWECategory",
    "PromptHierarchy",
    "MoPromptManager",
    # Seed prompts
    "LAYER1_SEED_PROMPTS",
    "LAYER2_SEED_PROMPTS",
    "LAYER3_SEED_PROMPTS",
    "get_seed_prompts_for_category",
    "get_all_layer1_seeds",
    # Seed loader
    "SeedPromptLoader",
    "SeedPromptConfig",
    "load_seeds_for_ga",
    "get_hierarchical_seeds",
    # Evolution prompts
    "TaskContext",
    "TASK_CONTEXTS",
    "get_task_context",
    "build_crossover_prompt",
    "build_mutation_prompt",
    "build_initialization_prompt",
    # Prompt contract
    "PromptContract",
    "PromptContractValidator",
    "ValidationResult",
    # Unified prompt template
    "PromptTemplate",
    "PromptSection",
    "PromptMetadata",
    "PromptSet",
]
