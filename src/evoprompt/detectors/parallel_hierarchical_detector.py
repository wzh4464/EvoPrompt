"""Parallel hierarchical vulnerability detector with multi-layer classification.

Implements a three-layer parallel detection system:
  Layer 1: Parallel classification into major vulnerability categories (top-k selection)
  Layer 2: Parallel sub-classification within selected major categories
  Layer 3: CWE-specific classification

Supports optional code enhancement via Comment4Vul or similar enhancers.
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple, Any, runtime_checkable

from ..llm.async_client import AsyncLLMClient
from ..prompts.hierarchical_three_layer import (
    MajorCategory,
    MiddleCategory,
    MAJOR_TO_MIDDLE,
    MIDDLE_TO_CWE,
    ThreeLayerPromptSet,
)
from .scoring import ScoredPrediction, DetectionPath, SelectionStrategy, MaxConfidenceSelection


logger = logging.getLogger(__name__)


@runtime_checkable
class CodeEnhancer(Protocol):
    """Protocol for code enhancement (e.g., Comment4Vul).
    
    Code enhancers add analysis comments or annotations to source code
    to improve vulnerability detection.
    """
    
    def enhance(self, code: str) -> str:
        """Synchronously enhance code with analysis comments.
        
        Args:
            code: Original source code
            
        Returns:
            Enhanced code with added comments/annotations
        """
        ...
    
    async def enhance_async(self, code: str) -> str:
        """Asynchronously enhance code with analysis comments.
        
        Args:
            code: Original source code
            
        Returns:
            Enhanced code with added comments/annotations
        """
        ...


class NoOpEnhancer:
    """Default no-op enhancer that returns code unchanged."""
    
    def enhance(self, code: str) -> str:
        return code
    
    async def enhance_async(self, code: str) -> str:
        return code


@dataclass
class HierarchicalPromptSet:
    """Complete prompt set for parallel hierarchical detection.
    
    Organizes prompts by layer with support for trainable sections.
    
    Attributes:
        layer1_prompts: Mapping of MajorCategory → prompt string
        layer2_prompts: Nested mapping of MajorCategory → MiddleCategory → prompt
        layer3_prompts: Nested mapping of MiddleCategory → CWE → prompt
        shared_structure: Common output format constraints
        trainable_markers: Tuple of (start, end) markers for trainable sections
    """
    layer1_prompts: Dict[str, str] = field(default_factory=dict)
    layer2_prompts: Dict[str, Dict[str, str]] = field(default_factory=dict)
    layer3_prompts: Dict[str, Dict[str, str]] = field(default_factory=dict)
    shared_structure: str = ""
    trainable_markers: Tuple[str, str] = ("{{TRAINABLE_START}}", "{{TRAINABLE_END}}")
    
    @classmethod
    def from_three_layer_set(cls, prompt_set: ThreeLayerPromptSet) -> "HierarchicalPromptSet":
        """Convert from existing ThreeLayerPromptSet.
        
        Creates individual prompts for each category at each layer.
        """
        # Layer 1: Create individual prompts for each major category
        layer1_prompts = {}
        for major in MajorCategory:
            layer1_prompts[major.value] = cls._create_layer1_prompt(major)
        
        # Layer 2: Create prompts for each middle category
        layer2_prompts = {}
        for major, middles in MAJOR_TO_MIDDLE.items():
            layer2_prompts[major.value] = {}
            for middle in middles:
                layer2_prompts[major.value][middle.value] = cls._create_layer2_prompt(
                    major, middle
                )
        
        # Layer 3: Create prompts for each CWE
        layer3_prompts = {}
        for middle, cwes in MIDDLE_TO_CWE.items():
            layer3_prompts[middle.value] = {}
            for cwe in cwes:
                layer3_prompts[middle.value][cwe] = cls._create_layer3_prompt(
                    middle, cwe
                )
        
        return cls(
            layer1_prompts=layer1_prompts,
            layer2_prompts=layer2_prompts,
            layer3_prompts=layer3_prompts,
            shared_structure=cls._get_shared_structure(),
        )
    
    @staticmethod
    def _create_layer1_prompt(major: MajorCategory) -> str:
        """Create Layer 1 prompt for a specific major category."""
        descriptions = {
            MajorCategory.MEMORY: "memory safety issues (buffer overflow, use-after-free, null pointer, memory leaks)",
            MajorCategory.INJECTION: "injection vulnerabilities (SQL, XSS, command, LDAP injection)",
            MajorCategory.LOGIC: "logic flaws (authentication bypass, race conditions, insecure defaults)",
            MajorCategory.INPUT: "input handling issues (path traversal, validation errors, format strings)",
            MajorCategory.CRYPTO: "cryptographic weaknesses (weak algorithms, insecure random)",
            MajorCategory.BENIGN: "safe, non-vulnerable code patterns",
        }
        
        desc = descriptions.get(major, "vulnerability patterns")
        
        return f"""Analyze the following code and determine if it contains {desc}.

{{TRAINABLE_START}}
Focus on identifying patterns specific to {major.value} category vulnerabilities.
Consider both direct vulnerabilities and potential security weaknesses.
{{TRAINABLE_END}}

Respond with a confidence score between 0.0 and 1.0.
Format: CONFIDENCE: <score>
Example: CONFIDENCE: 0.85

Code to analyze:
{{CODE}}
"""
    
    @staticmethod
    def _create_layer2_prompt(major: MajorCategory, middle: MiddleCategory) -> str:
        """Create Layer 2 prompt for a specific middle category."""
        return f"""Given that the code may contain {major.value} vulnerabilities, 
analyze if it specifically exhibits {middle.value} patterns.

{{TRAINABLE_START}}
Look for specific indicators of {middle.value} vulnerabilities:
- Check for common patterns and anti-patterns
- Consider context and data flow
- Identify potential attack vectors
{{TRAINABLE_END}}

Respond with a confidence score between 0.0 and 1.0.
Format: CONFIDENCE: <score>

Code to analyze:
{{CODE}}
"""
    
    @staticmethod
    def _create_layer3_prompt(middle: MiddleCategory, cwe: str) -> str:
        """Create Layer 3 prompt for a specific CWE."""
        return f"""Given potential {middle.value} vulnerability, determine if this code 
matches the specific pattern of {cwe}.

{{TRAINABLE_START}}
Analyze for {cwe}-specific characteristics:
- Known vulnerable patterns
- CWE-specific triggers and conditions
- Standard remediation gaps
{{TRAINABLE_END}}

Respond with a confidence score between 0.0 and 1.0.
Format: CONFIDENCE: <score>

Code to analyze:
{{CODE}}
"""
    
    @staticmethod
    def _get_shared_structure() -> str:
        """Get shared output format constraints."""
        return """
Output format:
- Provide a confidence score between 0.0 and 1.0
- Use format: CONFIDENCE: <score>
- Only output the confidence line, no additional explanation
"""
    
    def get_layer1_prompts(self) -> Dict[str, str]:
        """Get all Layer 1 prompts."""
        return self.layer1_prompts
    
    def get_layer2_prompts_for_major(self, major: str) -> Dict[str, str]:
        """Get Layer 2 prompts for a major category."""
        return self.layer2_prompts.get(major, {})
    
    def get_layer3_prompts_for_middle(self, middle: str) -> Dict[str, str]:
        """Get Layer 3 prompts for a middle category."""
        return self.layer3_prompts.get(middle, {})
    
    def update_prompt(self, layer: int, category: str, new_prompt: str, 
                      parent: Optional[str] = None) -> None:
        """Update a specific prompt (for meta-learning).
        
        Args:
            layer: Layer number (1, 2, or 3)
            category: Category name
            new_prompt: New prompt content
            parent: Parent category (required for layer 2 and 3)
        """
        if layer == 1:
            self.layer1_prompts[category] = new_prompt
        elif layer == 2:
            if parent is None:
                raise ValueError("Parent required for layer 2")
            if parent not in self.layer2_prompts:
                self.layer2_prompts[parent] = {}
            self.layer2_prompts[parent][category] = new_prompt
        elif layer == 3:
            if parent is None:
                raise ValueError("Parent required for layer 3")
            if parent not in self.layer3_prompts:
                self.layer3_prompts[parent] = {}
            self.layer3_prompts[parent][category] = new_prompt


@dataclass
class ParallelDetectorConfig:
    """Configuration for parallel hierarchical detector.
    
    Attributes:
        layer1_top_k: Number of top categories to select from Layer 1
        layer2_top_k: Number of top categories to select from Layer 2
        layer3_top_k: Number of top CWEs to select from Layer 3
        max_concurrent_requests: Maximum concurrent LLM requests
        default_confidence: Default confidence for parse failures
        enable_enhancement: Whether to use code enhancement
    """
    layer1_top_k: int = 2
    layer2_top_k: int = 2
    layer3_top_k: int = 1
    max_concurrent_requests: int = 20
    default_confidence: float = 0.0
    enable_enhancement: bool = True


class ParallelHierarchicalDetector:
    """Parallel hierarchical vulnerability detector.
    
    Performs three-layer detection with parallel execution at each layer:
    1. Layer 1: Classify into major categories, select top-k
    2. Layer 2: For each selected major, classify into middle categories
    3. Layer 3: For each selected middle, classify into specific CWEs
    
    Results are aggregated using a configurable selection strategy.
    """
    
    def __init__(
        self,
        llm_client: AsyncLLMClient,
        prompt_set: HierarchicalPromptSet,
        config: Optional[ParallelDetectorConfig] = None,
        enhancer: Optional[CodeEnhancer] = None,
        selection_strategy: Optional[SelectionStrategy] = None,
    ):
        """Initialize the detector.
        
        Args:
            llm_client: Async LLM client for API calls
            prompt_set: Hierarchical prompt set
            config: Detector configuration
            enhancer: Optional code enhancer (e.g., Comment4Vul)
            selection_strategy: Strategy for final path selection
        """
        self.llm_client = llm_client
        self.prompt_set = prompt_set
        self.config = config or ParallelDetectorConfig()
        self.enhancer = enhancer or NoOpEnhancer()
        self.selection_strategy = selection_strategy or MaxConfidenceSelection()
        
        # Semaphore for controlling concurrent requests
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
    
    async def detect_async(self, code: str) -> List[DetectionPath]:
        """Perform hierarchical detection on code.
        
        Args:
            code: Source code to analyze
            
        Returns:
            List of detection paths sorted by confidence
        """
        # Step 1: Optionally enhance code
        if self.config.enable_enhancement:
            enhanced_code = await self.enhancer.enhance_async(code)
        else:
            enhanced_code = code
        
        # Step 2: Layer 1 - Parallel classification into major categories
        layer1_predictions = await self._classify_layer1_parallel(enhanced_code)
        
        # Select top-k from Layer 1
        top_layer1 = sorted(
            layer1_predictions, 
            key=lambda p: p.confidence, 
            reverse=True
        )[:self.config.layer1_top_k]
        
        logger.debug(f"Layer 1 top-{self.config.layer1_top_k}: {[p.category for p in top_layer1]}")
        
        # Step 3: Layer 2 - Parallel classification for each selected major category
        layer2_predictions = await self._classify_layer2_parallel(
            enhanced_code, 
            [p.category for p in top_layer1]
        )
        
        # Step 4: Layer 3 - Parallel CWE classification
        layer3_predictions = await self._classify_layer3_parallel(
            enhanced_code,
            layer2_predictions
        )
        
        # Step 5: Build detection paths
        paths = self._build_detection_paths(
            layer1_predictions,
            layer2_predictions,
            layer3_predictions
        )
        
        # Step 6: Apply selection strategy
        selected_paths = self.selection_strategy.select(paths, top_k=self.config.layer3_top_k)
        
        return selected_paths
    
    def detect(self, code: str) -> List[DetectionPath]:
        """Synchronous wrapper for detect_async."""
        return asyncio.run(self.detect_async(code))
    
    async def detect_batch_async(
        self, 
        codes: List[str],
        show_progress: bool = True
    ) -> List[List[DetectionPath]]:
        """Perform detection on multiple code samples.
        
        Args:
            codes: List of code samples
            show_progress: Whether to log progress
            
        Returns:
            List of detection results for each code sample
        """
        results = []
        for i, code in enumerate(codes):
            result = await self.detect_async(code)
            results.append(result)
            
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(codes)} samples")
        
        return results
    
    async def _classify_layer1_parallel(
        self, 
        code: str
    ) -> List[ScoredPrediction]:
        """Classify code into major categories in parallel.
        
        Args:
            code: Enhanced source code
            
        Returns:
            List of scored predictions for each major category
        """
        prompts = self.prompt_set.get_layer1_prompts()
        categories = list(prompts.keys())
        
        # Prepare prompts with code
        filled_prompts = [
            prompt.replace("{CODE}", code).replace("{{CODE}}", code)
            for prompt in prompts.values()
        ]
        
        # Execute in parallel
        responses = await self._batch_generate_with_semaphore(filled_prompts)
        
        # Parse responses
        predictions = []
        for category, response in zip(categories, responses):
            confidence = self._parse_confidence(response)
            predictions.append(ScoredPrediction(
                category=category,
                confidence=confidence,
                layer=1,
                raw_response=response
            ))
        
        return predictions
    
    async def _classify_layer2_parallel(
        self, 
        code: str,
        selected_majors: List[str]
    ) -> List[ScoredPrediction]:
        """Classify code into middle categories for selected majors.
        
        Args:
            code: Enhanced source code
            selected_majors: List of selected major category names
            
        Returns:
            List of scored predictions for middle categories
        """
        all_prompts = []
        all_categories = []
        parent_map = []
        
        for major in selected_majors:
            layer2_prompts = self.prompt_set.get_layer2_prompts_for_major(major)
            for middle, prompt in layer2_prompts.items():
                filled = prompt.replace("{CODE}", code).replace("{{CODE}}", code)
                all_prompts.append(filled)
                all_categories.append(middle)
                parent_map.append(major)
        
        if not all_prompts:
            return []
        
        # Execute in parallel
        responses = await self._batch_generate_with_semaphore(all_prompts)
        
        # Parse responses
        predictions = []
        for category, parent, response in zip(all_categories, parent_map, responses):
            confidence = self._parse_confidence(response)
            predictions.append(ScoredPrediction(
                category=category,
                confidence=confidence,
                layer=2,
                parent_category=parent,
                raw_response=response
            ))
        
        return predictions
    
    async def _classify_layer3_parallel(
        self, 
        code: str,
        layer2_predictions: List[ScoredPrediction]
    ) -> List[ScoredPrediction]:
        """Classify code into specific CWEs for relevant middle categories.
        
        Args:
            code: Enhanced source code
            layer2_predictions: Predictions from Layer 2
            
        Returns:
            List of scored predictions for CWEs
        """
        # Select top middle categories based on confidence
        top_middles = sorted(
            layer2_predictions,
            key=lambda p: p.confidence,
            reverse=True
        )[:self.config.layer2_top_k * self.config.layer1_top_k]  # Account for multiple majors
        
        all_prompts = []
        all_cwes = []
        parent_map = []
        
        for pred in top_middles:
            layer3_prompts = self.prompt_set.get_layer3_prompts_for_middle(pred.category)
            for cwe, prompt in layer3_prompts.items():
                filled = prompt.replace("{CODE}", code).replace("{{CODE}}", code)
                all_prompts.append(filled)
                all_cwes.append(cwe)
                parent_map.append(pred.category)
        
        if not all_prompts:
            return []
        
        # Execute in parallel
        responses = await self._batch_generate_with_semaphore(all_prompts)
        
        # Parse responses
        predictions = []
        for cwe, parent, response in zip(all_cwes, parent_map, responses):
            confidence = self._parse_confidence(response)
            predictions.append(ScoredPrediction(
                category=cwe,
                confidence=confidence,
                layer=3,
                parent_category=parent,
                raw_response=response
            ))
        
        return predictions
    
    async def _batch_generate_with_semaphore(
        self, 
        prompts: List[str]
    ) -> List[str]:
        """Execute batch generation with semaphore control.
        
        Args:
            prompts: List of prompts to process
            
        Returns:
            List of LLM responses
        """
        async def generate_one(prompt: str) -> str:
            async with self._semaphore:
                return await self.llm_client.generate_async(prompt)
        
        tasks = [generate_one(p) for p in prompts]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    def _parse_confidence(self, response: str) -> float:
        """Parse confidence score from LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed confidence score (0.0 to 1.0)
        """
        if not response or response == "error":
            return self.config.default_confidence
        
        # Try to find CONFIDENCE: <score> pattern
        patterns = [
            r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)",
            r"confidence:\s*([0-9]*\.?[0-9]+)",
            r"Score:\s*([0-9]*\.?[0-9]+)",
            r"(\d+\.?\d*)\s*$",  # Number at end
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    # Normalize if needed (e.g., percentage to decimal)
                    if score > 1.0:
                        score = score / 100.0
                    return max(0.0, min(1.0, score))
                except ValueError:
                    continue
        
        logger.warning(f"Could not parse confidence from response: {response[:100]}")
        return self.config.default_confidence
    
    def _build_detection_paths(
        self,
        layer1: List[ScoredPrediction],
        layer2: List[ScoredPrediction],
        layer3: List[ScoredPrediction]
    ) -> List[DetectionPath]:
        """Build complete detection paths from layer predictions.
        
        Args:
            layer1: Layer 1 predictions
            layer2: Layer 2 predictions
            layer3: Layer 3 predictions
            
        Returns:
            List of complete detection paths
        """
        # Create lookup maps
        l1_map = {p.category: p for p in layer1}
        l2_map = {p.category: p for p in layer2}
        l3_by_parent = {}
        for p in layer3:
            if p.parent_category not in l3_by_parent:
                l3_by_parent[p.parent_category] = []
            l3_by_parent[p.parent_category].append(p)
        
        paths = []
        
        # Build paths from Layer 3 back to Layer 1
        for l3_pred in layer3:
            middle = l3_pred.parent_category
            l2_pred = l2_map.get(middle)
            
            if l2_pred is None:
                continue
            
            major = l2_pred.parent_category
            l1_pred = l1_map.get(major)
            
            if l1_pred is None:
                continue
            
            path = DetectionPath(
                layer1_category=major,
                layer1_confidence=l1_pred.confidence,
                layer2_category=middle,
                layer2_confidence=l2_pred.confidence,
                layer3_cwe=l3_pred.category,
                layer3_confidence=l3_pred.confidence,
                metadata={
                    "layer1_raw": l1_pred.raw_response,
                    "layer2_raw": l2_pred.raw_response,
                    "layer3_raw": l3_pred.raw_response,
                }
            )
            paths.append(path)
        
        # Also include Layer 2 paths without Layer 3 (for categories without CWEs)
        for l2_pred in layer2:
            if l2_pred.category in l3_by_parent:
                continue  # Already handled via Layer 3
            
            major = l2_pred.parent_category
            l1_pred = l1_map.get(major)
            
            if l1_pred is None:
                continue
            
            path = DetectionPath(
                layer1_category=major,
                layer1_confidence=l1_pred.confidence,
                layer2_category=l2_pred.category,
                layer2_confidence=l2_pred.confidence,
                metadata={
                    "layer1_raw": l1_pred.raw_response,
                    "layer2_raw": l2_pred.raw_response,
                }
            )
            paths.append(path)
        
        return paths


# Factory function for creating detector with default configuration
def create_parallel_detector(
    llm_client: AsyncLLMClient,
    prompt_set: Optional[ThreeLayerPromptSet] = None,
    enhancer: Optional[CodeEnhancer] = None,
    **config_kwargs
) -> ParallelHierarchicalDetector:
    """Factory function to create a parallel hierarchical detector.
    
    Args:
        llm_client: Async LLM client
        prompt_set: Optional ThreeLayerPromptSet (creates default if not provided)
        enhancer: Optional code enhancer
        **config_kwargs: Configuration parameters
        
    Returns:
        Configured ParallelHierarchicalDetector
    """
    from ..prompts.hierarchical_three_layer import ThreeLayerPromptFactory
    
    if prompt_set is None:
        prompt_set = ThreeLayerPromptFactory.create_default_prompt_set()
    
    hierarchical_prompts = HierarchicalPromptSet.from_three_layer_set(prompt_set)
    config = ParallelDetectorConfig(**config_kwargs)
    
    return ParallelHierarchicalDetector(
        llm_client=llm_client,
        prompt_set=hierarchical_prompts,
        config=config,
        enhancer=enhancer,
    )
