#!/usr/bin/env python3
"""CLI entry point for meta-learning prompt evolution."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from config import MetaEvolveConfig
from training_loop import MetaEvolveTrainer


def main():
    parser = argparse.ArgumentParser(description="Meta-Learning Prompt Evolution for CWE Classification")

    # Training params
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--meta-update-freq", type=int, default=4)

    # kNN params
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--candidate-cwes", type=int, default=10)

    # Model params
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--meta-model", default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-rules", type=int, default=30)

    # Code params
    parser.add_argument("--max-code-chars", type=int, default=6000)

    # EMA
    parser.add_argument("--ema-alpha", type=float, default=0.9)

    # Data paths
    parser.add_argument("--test-data", default="data/primevul/primevul/primevul_test_fixed.jsonl")
    parser.add_argument("--train-data", default="data/primevul/primevul/primevul_train_fixed.jsonl")
    parser.add_argument("--kb", default="outputs/rebuttal/meta_evolve/cwe_knowledge.json")
    parser.add_argument("--output-dir", default="outputs/rebuttal/meta_evolve")

    # Resume
    parser.add_argument("--resume", default=None, help="Resume from checkpoint directory")

    args = parser.parse_args()

    config = MetaEvolveConfig(
        test_data=args.test_data,
        train_data=args.train_data,
        kb_output=args.kb,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        meta_update_freq=args.meta_update_freq,
        knn_k=args.knn_k,
        candidate_cwes=args.candidate_cwes,
        max_code_chars=args.max_code_chars,
        detect_model=args.model,
        max_workers=args.workers,
        max_rules=args.max_rules,
        meta_model=args.meta_model,
        ema_alpha=args.ema_alpha,
        resume_from=args.resume,
    )

    trainer = MetaEvolveTrainer(config)
    best_f1, best_rules = trainer.run()

    print(f"\nFinal Best Macro-F1: {best_f1*100:.2f}%")
    print(f"Best rules ({len(best_rules)}):")
    for rule in best_rules:
        print(f"  {rule}")


if __name__ == "__main__":
    main()
