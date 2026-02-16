"""Multi-agent coordinator for collaborative prompt evolution.

The coordinator manages the interaction between:
- DetectionAgent: Performs vulnerability detection
- MetaAgent: Optimizes prompts based on performance feedback
"""

from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from .agents import DetectionAgent, MetaAgent
from ..evaluators.statistics import DetectionStatistics, BatchStatistics, StatisticsCollector
from ..data.dataset import Dataset
from ..utils.trace import TraceManager, compute_text_hash
from ..core.prompt_change_logger import PromptChangeLogger


class CoordinationStrategy(Enum):
    """Strategy for coordinating agents."""
    SEQUENTIAL = "sequential"  # Meta-agent optimizes after each detection batch
    PARALLEL = "parallel"  # Multiple detection agents work in parallel
    HIERARCHICAL = "hierarchical"  # Hierarchical routing + category-specific detection


@dataclass
class CoordinatorConfig:
    """Configuration for multi-agent coordinator.

    Attributes:
        strategy: Coordination strategy
        batch_size: Size of evaluation batches
        enable_batch_feedback: Whether to provide batch-level feedback to meta-agent
        statistics_window: Number of recent batches to include in feedback
    """

    strategy: CoordinationStrategy = CoordinationStrategy.SEQUENTIAL
    batch_size: int = 32
    enable_batch_feedback: bool = True
    statistics_window: int = 5  # Track last 5 batches


class MultiAgentCoordinator:
    """Coordinates detection and meta-optimization agents.

    The coordinator implements a collaborative evolution loop:
    1. DetectionAgent evaluates prompts on a batch of data
    2. Statistics are collected and analyzed
    3. MetaAgent receives performance feedback and suggests improvements
    4. Process repeats with improved prompts
    """

    def __init__(
        self,
        detection_agent: DetectionAgent,
        meta_agent: MetaAgent,
        config: Optional[CoordinatorConfig] = None,
        trace_manager: Optional[TraceManager] = None,
        prompt_change_logger: Optional[PromptChangeLogger] = None,
    ):
        """Initialize coordinator.

        Args:
            detection_agent: Agent for vulnerability detection
            meta_agent: Agent for meta-level optimization
            config: Coordinator configuration
            prompt_change_logger: Always-on prompt change logger
        """
        self.detection_agent = detection_agent
        self.meta_agent = meta_agent
        self.config = config or CoordinatorConfig()
        self.trace = trace_manager
        self.prompt_change_logger = prompt_change_logger

        # Statistics tracking
        self.statistics_collector = StatisticsCollector()
        self.batch_history: List[BatchStatistics] = []

    def evaluate_prompt(
        self,
        prompt: str,
        dataset: Dataset,
        sample_size: Optional[int] = None
    ) -> DetectionStatistics:
        """Evaluate a prompt on a dataset.

        Args:
            prompt: Prompt to evaluate
            dataset: Dataset to evaluate on
            sample_size: Optional number of samples to use

        Returns:
            Detection statistics
        """
        samples = dataset.get_samples(sample_size)
        stats = DetectionStatistics()

        # Extract code and labels
        code_samples = [s.input_text for s in samples]
        labels = [s.target for s in samples]

        trace_enabled = bool(self.trace and self.trace.enabled)
        prompt_hash = compute_text_hash(prompt)

        # Get predictions from detection agent
        if trace_enabled:
            payload = self.detection_agent.detect(
                prompt,
                code_samples,
                return_raw=self.trace.config.store_raw_responses,
                return_prompts=self.trace.config.store_filled_prompts,
            )
            predictions = payload["predictions"]
            raw_responses = payload.get("responses", [])
            filled_prompts = payload.get("prompts", [])
        else:
            predictions = self.detection_agent.detect(prompt, code_samples)
            raw_responses = []
            filled_prompts = []

        # Collect statistics
        for idx, (pred, actual, sample) in enumerate(zip(predictions, labels, samples)):
            # Convert label to string
            actual_str = "vulnerable" if actual == 1 else "benign"

            # Get category/CWE if available
            category = None
            if hasattr(sample, 'metadata') and 'cwe' in sample.metadata:
                cwes = sample.metadata['cwe']
                category = cwes[0] if cwes else None

            stats.add_prediction(
                predicted=pred,
                actual=actual_str,
                category=category,
                sample_id=getattr(sample, 'id', None)
            )

            if trace_enabled:
                record: Dict[str, Any] = {
                    "prompt_hash": prompt_hash,
                    "prediction": pred,
                    "actual": actual_str,
                    "category": category,
                    "sample_id": getattr(sample, "id", None),
                }
                if self.trace.config.store_code:
                    record["input_code"] = sample.input_text
                if raw_responses and idx < len(raw_responses) and self.trace.config.store_raw_responses:
                    record["raw_response"] = raw_responses[idx]
                if filled_prompts and idx < len(filled_prompts) and self.trace.config.store_filled_prompts:
                    record["filled_prompt"] = filled_prompts[idx]
                self.trace.log_sample_trace(record)

        stats.compute_metrics()
        if trace_enabled:
            self.trace.log_event(
                "prompt_evaluation_summary",
                {
                    "prompt_hash": prompt_hash,
                    "total_samples": stats.total_samples,
                    "summary": stats.get_summary(),
                },
            )
        return stats

    def evaluate_with_batches(
        self,
        prompt: str,
        dataset: Dataset,
        sample_size: Optional[int] = None
    ) -> Tuple[DetectionStatistics, List[BatchStatistics]]:
        """Evaluate a prompt using batch processing.

        This enables batch-level feedback to the meta-agent.

        Args:
            prompt: Prompt to evaluate
            dataset: Dataset to evaluate on
            sample_size: Optional number of samples to use

        Returns:
            Tuple of (overall statistics, list of batch statistics)
        """
        samples = dataset.get_samples(sample_size)
        overall_stats = DetectionStatistics()
        batch_stats_list = []

        # Process in batches
        trace_enabled = bool(self.trace and self.trace.enabled)
        prompt_hash = compute_text_hash(prompt)

        for batch_id, i in enumerate(range(0, len(samples), self.config.batch_size)):
            batch_samples = samples[i:i + self.config.batch_size]

            # Evaluate batch
            batch_stats_obj = DetectionStatistics()
            code_samples = [s.input_text for s in batch_samples]
            labels = [s.target for s in batch_samples]

            if trace_enabled:
                payload = self.detection_agent.detect(
                    prompt,
                    code_samples,
                    return_raw=self.trace.config.store_raw_responses,
                    return_prompts=self.trace.config.store_filled_prompts,
                )
                predictions = payload["predictions"]
                raw_responses = payload.get("responses", [])
                filled_prompts = payload.get("prompts", [])
            else:
                predictions = self.detection_agent.detect(prompt, code_samples)
                raw_responses = []
                filled_prompts = []

            for idx, (pred, actual, sample) in enumerate(zip(predictions, labels, batch_samples)):
                actual_str = "vulnerable" if actual == 1 else "benign"
                category = None
                if hasattr(sample, 'metadata') and 'cwe' in sample.metadata:
                    cwes = sample.metadata['cwe']
                    category = cwes[0] if cwes else None

                # Update both overall and batch stats
                batch_stats_obj.add_prediction(pred, actual_str, category)
                overall_stats.add_prediction(pred, actual_str, category)

                if trace_enabled:
                    record: Dict[str, Any] = {
                        "prompt_hash": prompt_hash,
                        "batch_id": batch_id,
                        "prediction": pred,
                        "actual": actual_str,
                        "category": category,
                        "sample_id": getattr(sample, "id", None),
                    }
                    if self.trace.config.store_code:
                        record["input_code"] = sample.input_text
                    if raw_responses and idx < len(raw_responses) and self.trace.config.store_raw_responses:
                        record["raw_response"] = raw_responses[idx]
                    if filled_prompts and idx < len(filled_prompts) and self.trace.config.store_filled_prompts:
                        record["filled_prompt"] = filled_prompts[idx]
                    self.trace.log_sample_trace(record)

            batch_stats_obj.compute_metrics()

            # Create batch statistics record
            batch_stat = BatchStatistics(
                batch_id=batch_id,
                batch_size=len(batch_samples),
                statistics=batch_stats_obj,
                metadata={"start_idx": i, "end_idx": i + len(batch_samples)}
            )
            batch_stats_list.append(batch_stat)

        overall_stats.compute_metrics()
        if trace_enabled:
            self.trace.log_event(
                "prompt_evaluation_summary",
                {
                    "prompt_hash": prompt_hash,
                    "total_samples": overall_stats.total_samples,
                    "summary": overall_stats.get_summary(),
                },
            )
        return overall_stats, batch_stats_list

    def collaborative_improve(
        self,
        prompt: str,
        dataset: Dataset,
        generation: int = 0,
        historical_stats: Optional[List[DetectionStatistics]] = None,
        sample_size: Optional[int] = None,
    ) -> Tuple[str, DetectionStatistics]:
        """Collaboratively improve a prompt using detection + meta agents.

        Workflow:
        1. DetectionAgent evaluates current prompt
        2. Statistics are collected (overall + batches)
        3. MetaAgent analyzes performance and suggests improvements
        4. Return improved prompt

        Args:
            prompt: Current prompt to improve
            dataset: Dataset for evaluation
            generation: Current generation number
            historical_stats: Optional historical statistics

        Returns:
            Tuple of (improved prompt, evaluation statistics)
        """
        # Evaluate current prompt
        if self.config.enable_batch_feedback:
            stats, batch_stats = self.evaluate_with_batches(prompt, dataset, sample_size=sample_size)
            # Track batch statistics
            self.batch_history.extend(batch_stats)
            # Keep only recent batches
            if len(self.batch_history) > self.config.statistics_window * 10:
                self.batch_history = self.batch_history[-self.config.statistics_window * 10:]
        else:
            stats = self.evaluate_prompt(prompt, dataset, sample_size=sample_size)
            batch_stats = []

        # Get improvement suggestions from statistics
        self.statistics_collector.add_generation_stats(generation, stats)
        suggestions = self.statistics_collector.get_improvement_suggestions()

        # Meta-agent improves prompt
        improved_prompt = self.meta_agent.improve_prompt(
            current_prompt=prompt,
            current_stats=stats,
            historical_stats=historical_stats or [],
            improvement_suggestions=suggestions,
            generation=generation
        )
        # Evaluate improved prompt to get actual performance
        improved_stats = self.evaluate_prompt(improved_prompt, dataset, sample_size=sample_size)

        if self.trace and self.trace.enabled:
            payload = {
                "operation": "meta_improve",
                "generation": generation,
                "prompt_hash_before": compute_text_hash(prompt),
                "prompt_hash_after": compute_text_hash(improved_prompt),
                "metrics_before": stats.get_summary(),
                "metrics_after": improved_stats.get_summary(),
                "improvement_suggestions": suggestions,
            }
            if self.trace.config.store_prompts:
                payload["before_prompt"] = prompt
                payload["after_prompt"] = improved_prompt
            self.trace.log_prompt_update(payload)
            if self.trace.config.store_prompts:
                self.trace.save_prompt_snapshot(
                    f"gen{generation}_improve_{payload['prompt_hash_after']}",
                    improved_prompt,
                    metadata={"operation": "meta_improve", "generation": generation},
                )

        if self.prompt_change_logger:
            self.prompt_change_logger.log_change(
                operation="meta_improve",
                prompt_before=prompt,
                prompt_after=improved_prompt,
                generation=generation,
                trigger_reason="evolutionary_operator",
                context={"improvement_suggestions": suggestions},
                metrics_before=stats.get_summary(),
                metrics_after=improved_stats.get_summary(),
            )

        return improved_prompt, improved_stats

    def collaborative_crossover(
        self,
        parent1: str,
        parent2: str,
        dataset: Dataset,
        generation: int = 0,
        sample_size: Optional[int] = None,
    ) -> Tuple[str, DetectionStatistics]:
        """Perform collaborative crossover of two parent prompts.

        Args:
            parent1: First parent prompt
            parent2: Second parent prompt
            dataset: Dataset for evaluation
            generation: Current generation number

        Returns:
            Tuple of (offspring prompt, evaluation statistics)
        """
        # Evaluate both parents
        stats1 = self.evaluate_prompt(parent1, dataset, sample_size=sample_size)
        stats2 = self.evaluate_prompt(parent2, dataset, sample_size=sample_size)

        # Meta-agent creates offspring
        offspring = self.meta_agent.crossover_prompts(
            parent1, parent2, stats1, stats2, generation
        )

        # Evaluate offspring
        offspring_stats = self.evaluate_prompt(offspring, dataset, sample_size=sample_size)

        if self.trace and self.trace.enabled:
            payload = {
                "operation": "meta_crossover",
                "generation": generation,
                "prompt_hash_before": compute_text_hash(parent1),
                "prompt_hash_before_2": compute_text_hash(parent2),
                "prompt_hash_after": compute_text_hash(offspring),
                "metrics_parent1": stats1.get_summary(),
                "metrics_parent2": stats2.get_summary(),
                "metrics_after": offspring_stats.get_summary(),
            }
            if self.trace.config.store_prompts:
                payload["parent1"] = parent1
                payload["parent2"] = parent2
                payload["after_prompt"] = offspring
            self.trace.log_prompt_update(payload)
            if self.trace.config.store_prompts:
                self.trace.save_prompt_snapshot(
                    f"gen{generation}_crossover_{payload['prompt_hash_after']}",
                    offspring,
                    metadata={"operation": "meta_crossover", "generation": generation},
                )

        if self.prompt_change_logger:
            self.prompt_change_logger.log_change(
                operation="meta_crossover",
                prompt_before=parent1,
                prompt_after=offspring,
                generation=generation,
                trigger_reason="evolutionary_operator",
                context={
                    "parent1_hash": compute_text_hash(parent1),
                    "parent2_hash": compute_text_hash(parent2),
                },
                metrics_before=stats1.get_summary(),
                metrics_after=offspring_stats.get_summary(),
            )

        return offspring, offspring_stats

    def collaborative_mutate(
        self,
        prompt: str,
        dataset: Dataset,
        generation: int = 0,
        sample_size: Optional[int] = None,
    ) -> Tuple[str, DetectionStatistics]:
        """Perform collaborative mutation of a prompt.

        Args:
            prompt: Prompt to mutate
            dataset: Dataset for evaluation
            generation: Current generation number

        Returns:
            Tuple of (mutated prompt, evaluation statistics)
        """
        # Evaluate current prompt
        stats = self.evaluate_prompt(prompt, dataset, sample_size=sample_size)

        # Meta-agent mutates prompt
        mutated = self.meta_agent.mutate_prompt(prompt, stats, generation)

        # Evaluate mutated prompt
        mutated_stats = self.evaluate_prompt(mutated, dataset, sample_size=sample_size)

        if self.trace and self.trace.enabled:
            payload = {
                "operation": "meta_mutate",
                "generation": generation,
                "prompt_hash_before": compute_text_hash(prompt),
                "prompt_hash_after": compute_text_hash(mutated),
                "metrics_before": stats.get_summary(),
                "metrics_after": mutated_stats.get_summary(),
            }
            if self.trace.config.store_prompts:
                payload["before_prompt"] = prompt
                payload["after_prompt"] = mutated
            self.trace.log_prompt_update(payload)
            if self.trace.config.store_prompts:
                self.trace.save_prompt_snapshot(
                    f"gen{generation}_mutate_{payload['prompt_hash_after']}",
                    mutated,
                    metadata={"operation": "meta_mutate", "generation": generation},
                )

        if self.prompt_change_logger:
            self.prompt_change_logger.log_change(
                operation="meta_mutate",
                prompt_before=prompt,
                prompt_after=mutated,
                generation=generation,
                trigger_reason="evolutionary_operator",
                context={},
                metrics_before=stats.get_summary(),
                metrics_after=mutated_stats.get_summary(),
            )

        return mutated, mutated_stats

    def get_statistics_summary(self) -> Dict:
        """Get summary of collected statistics.

        Returns:
            Dictionary with statistics summary
        """
        return {
            "total_generations": len(self.statistics_collector.generation_stats),
            "total_batches": len(self.batch_history),
            "historical_trend": self.statistics_collector.get_historical_trend(),
            "category_error_patterns": self.statistics_collector.get_category_error_patterns(),
            "improvement_suggestions": self.statistics_collector.get_improvement_suggestions(),
        }

    def export_statistics(self, filepath: str):
        """Export all statistics to file.

        Args:
            filepath: Path to export JSON file
        """
        self.statistics_collector.export_to_json(filepath)
