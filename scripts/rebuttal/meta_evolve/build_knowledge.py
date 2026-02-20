#!/usr/bin/env python3
"""Build CWE Knowledge Base from training data and CWE descriptions."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from cwe_knowledge import CWEKnowledgeBase


def main():
    parser = argparse.ArgumentParser(description="Build CWE Knowledge Base")
    parser.add_argument(
        "--knowledge-cwe",
        default="data/primevul/knowledge-cwe.jsonl",
        help="Path to knowledge-cwe.jsonl",
    )
    parser.add_argument(
        "--train-data",
        default="data/primevul/primevul/primevul_train_fixed.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--output",
        default="outputs/rebuttal/meta_evolve/cwe_knowledge.json",
        help="Output path for KB",
    )
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]

    def resolve(p):
        path = Path(p)
        return path if path.is_absolute() else project_root / path

    kb = CWEKnowledgeBase.build(
        knowledge_cwe_path=str(resolve(args.knowledge_cwe)),
        train_data_path=str(resolve(args.train_data)),
        output_path=str(resolve(args.output)),
        max_examples_per_cwe=args.max_examples,
    )

    print(f"\nKB Summary:")
    print(f"  CWE descriptions: {len(kb.descriptions)}")
    print(f"  CWEs with examples: {len(kb.examples)}")
    print(f"  Total examples: {sum(len(v) for v in kb.examples.values())}")
    print(f"  All known CWE IDs: {len(kb.all_cwe_ids)}")


if __name__ == "__main__":
    main()
