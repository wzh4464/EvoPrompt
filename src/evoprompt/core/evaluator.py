"""Evaluator for prompt performance assessment."""

import json
import os
from typing import Dict, Any, List, Optional, Protocol
import numpy as np

from ..llm.client import LLMClient
from ..data.dataset import Dataset
from ..metrics.base import Metric


class EvaluationResult:
    """Container for evaluation results."""
    
    def __init__(self, score: float, details: Optional[Dict[str, Any]] = None):
        self.score = score
        self.details = details or {}
        
    def __repr__(self):
        return f"EvaluationResult(score={self.score}, details={self.details})"


class Evaluator:
    """Modern evaluator for prompt performance assessment."""
    
    def __init__(
        self,
        dataset: Dataset,
        metric: Metric,
        llm_client: LLMClient,
        template_config: Optional[Dict[str, Any]] = None
    ):
        self.dataset = dataset
        self.metric = metric
        self.llm_client = llm_client
        self.template_config = template_config or {}
        
    def evaluate(self, prompt: str, sample_size: Optional[int] = None) -> EvaluationResult:
        """Evaluate a prompt on the dataset."""
        samples = self.dataset.get_samples(sample_size)
        predictions = []
        targets = []
        
        for sample in samples:
            # Format prompt with sample
            formatted_prompt = self._format_prompt(prompt, sample)
            
            # Get prediction from LLM
            response = self.llm_client.generate(formatted_prompt)
            predictions.append(response)
            targets.append(sample.target)
            
        # Calculate metric
        score = self.metric.compute(predictions, targets)
        
        return EvaluationResult(
            score=score,
            details={
                "num_samples": len(samples),
                "predictions": predictions[:5],  # Store first 5 for debugging
                "targets": targets[:5]
            }
        )
        
    def _format_prompt(self, prompt: str, sample) -> str:
        """Format the prompt with sample data."""
        if hasattr(sample, 'input_text'):
            return prompt.replace("{input}", sample.input_text)
        return prompt


# Legacy evaluator class for backward compatibility
class LegacyEvaluator:
    """Legacy evaluator wrapper for backward compatibility."""
    
    def __init__(self, args):
        """Initialize legacy evaluator with old interface."""
        # Import legacy evaluator
        import sys
        sys.path.append("../../")
        from evaluator import Evaluator as OldEvaluator
        
        self._evaluator = OldEvaluator(args)
        
    def __getattr__(self, name):
        """Delegate to legacy evaluator."""
        return getattr(self._evaluator, name)