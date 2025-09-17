"""Command-line interface for EvoPrompt."""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

from .core.evolution import EvolutionEngine
from .core.evaluator import Evaluator
from .algorithms.genetic import GeneticAlgorithm
from .algorithms.differential import DifferentialEvolution
from .llm.client import LLMClient


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="EvoPrompt: Evolutionary Prompt Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset and task
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name")
    parser.add_argument("--task", type=str, default="cls",
                       choices=["cls", "sim", "sum", "vul_detection"],
                       help="Task type")
    
    # Evolution settings
    parser.add_argument("--algorithm", type=str, default="de",
                       choices=["ga", "de"], 
                       help="Evolution algorithm")
    parser.add_argument("--population-size", type=int, default=20,
                       help="Population size")
    parser.add_argument("--generations", type=int, default=10,
                       help="Number of generations")
    parser.add_argument("--mutation-rate", type=float, default=0.1,
                       help="Mutation rate")
    
    # LLM settings
    parser.add_argument("--llm-type", type=str, default="kimi-k2-0711-preview",
                       help="LLM type for evolution")
    parser.add_argument("--evaluation-llm", type=str, default="kimi-k2-0711-preview",
                       help="LLM for evaluation")
    
    # Data settings
    parser.add_argument("--dev-file", type=str,
                       help="Development set file")
    parser.add_argument("--test-file", type=str,
                       help="Test set file")
    parser.add_argument("--sample-size", type=int,
                       help="Sample size for evaluation")
    
    # Output settings
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--save-population", action="store_true",
                       help="Save final population")
    
    # Legacy mode
    parser.add_argument("--legacy", action="store_true",
                       help="Use legacy implementation")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    return parser


def run_legacy_mode(args):
    """Run in legacy compatibility mode."""
    print("Running in legacy mode...")
    
    # Import legacy components
    from .legacy import parse_args, ga_evo, de_evo
    from .legacy.evaluator import Evaluator as LegacyEvaluator
    
    # Create legacy args object
    legacy_args = argparse.Namespace()
    for key, value in vars(args).items():
        setattr(legacy_args, key, value)
    
    # Set legacy-specific defaults
    if not hasattr(legacy_args, 'evo_mode'):
        legacy_args.evo_mode = args.algorithm
    if not hasattr(legacy_args, 'popsize'):
        legacy_args.popsize = args.population_size
    if not hasattr(legacy_args, 'budget'):
        legacy_args.budget = args.generations
        
    # Create evaluator
    evaluator = LegacyEvaluator(legacy_args)
    
    # Run evolution
    if args.algorithm == "ga":
        ga_evo(legacy_args, evaluator)
    elif args.algorithm == "de":
        de_evo(legacy_args, evaluator)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")


def run_modern_mode(args):
    """Run in modern mode with new architecture."""
    print("Running in modern mode...")
    
    # Create algorithm
    config = {
        "population_size": args.population_size,
        "max_generations": args.generations,
        "mutation_rate": args.mutation_rate,
    }
    
    if args.algorithm == "ga":
        algorithm = GeneticAlgorithm(config)
    elif args.algorithm == "de":
        algorithm = DifferentialEvolution(config)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # Create components (simplified for now)
    # In a full implementation, these would be properly configured
    llm_client = LLMClient(args.llm_type)
    
    # Create dummy evaluator and dataset for demonstration
    # In practice, these would be created based on args.dataset and args.task
    class DummyDataset:
        def get_samples(self, n=None):
            return [type('Sample', (), {'input_text': f'Sample {i}', 'target': f'Target {i}'})() 
                   for i in range(n or 10)]
    
    class DummyMetric:
        def compute(self, predictions, targets):
            return 0.85  # Dummy score
    
    evaluator = Evaluator(
        dataset=DummyDataset(),
        metric=DummyMetric(),
        llm_client=llm_client
    )
    
    # Create evolution engine
    engine = EvolutionEngine(
        algorithm=algorithm,
        evaluator=evaluator,
        llm_client=llm_client,
        config=config
    )
    
    # Run evolution
    initial_prompts = [
        "Solve this problem: {input}",
        "Answer the question: {input}",
        "Process this input: {input}",
    ]
    
    results = engine.evolve(initial_prompts=initial_prompts)
    
    print(f"Best prompt: {results['best_prompt']}")
    print(f"Best fitness: {results['best_fitness']}")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.verbose:
        print(f"EvoPrompt CLI - Args: {args}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.legacy:
            run_legacy_mode(args)
        else:
            run_modern_mode(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())