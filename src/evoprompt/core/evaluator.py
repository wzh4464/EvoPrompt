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
        
    def evaluate(self, prompt: str, sample_size: Optional[int] = None, filled_prompts_file: Optional[str] = None) -> EvaluationResult:
        """Evaluate a prompt on the dataset. Optionally log all filled prompt instances."""
        samples = self.dataset.get_samples(sample_size)
        predictions = []
        targets = []
        filled_examples = []

        for idx, sample in enumerate(samples):
            # Format prompt with sample
            formatted_prompt = self._format_prompt(prompt, sample)

            # 收集填充实例
            instance = {
                "template": prompt,
                "filled": formatted_prompt,
                "sample_id": getattr(sample, 'id', idx),
                "generation": getattr(self, 'generation', None),
                "target": getattr(sample, 'target', None)
            }
            filled_examples.append(instance)

            # Get prediction from LLM
            response = self.llm_client.generate(formatted_prompt)
            predictions.append(response)
            targets.append(sample.target)

        # 写入聚合的填充实例（如指定文件）
        if filled_prompts_file is not None and filled_examples:
            try:
                with open(filled_prompts_file, 'a', encoding='utf-8') as f:
                    for item in filled_examples:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"⚠️ 填充prompt样例保存失败: {e}")

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