#!/usr/bin/env python3
"""Build knowledge base from Primevul dataset.

Creates a knowledge base with sampled examples from training data.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evoprompt.data.dataset import PrimevulDataset
from evoprompt.rag.knowledge_base import (
    KnowledgeBase,
    KnowledgeBaseBuilder,
    create_knowledge_base_from_dataset,
)


def build_from_default():
    """Build knowledge base with default examples."""
    print("🏗️  Building Knowledge Base from Default Examples")
    print("=" * 70)

    kb = KnowledgeBaseBuilder.create_default_kb()

    stats = kb.statistics()
    print(f"\n✅ Knowledge base created:")
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Major categories: {stats['major_categories']}")
    print(f"   Middle categories: {stats['middle_categories']}")
    print(f"   CWE types: {stats['cwe_types']}")

    print("\n📊 Examples per major category:")
    for cat, count in stats['examples_per_major'].items():
        print(f"   {cat}: {count}")

    print("\n📊 Examples per middle category:")
    for cat, count in stats['examples_per_middle'].items():
        print(f"   {cat}: {count}")

    return kb


def build_from_dataset(data_file: str, samples_per_category: int):
    """Build knowledge base from dataset.

    Args:
        data_file: Path to dataset file
        samples_per_category: Number of samples per category
    """
    print(f"🏗️  Building Knowledge Base from Dataset: {data_file}")
    print("=" * 70)

    # Load dataset
    dataset = PrimevulDataset(data_file, split="train")
    print(f"   ✅ Loaded {len(dataset)} samples")

    # Build KB
    print(f"\n📝 Sampling {samples_per_category} examples per category...")

    kb = KnowledgeBase()

    # Use the create function but don't save yet
    from evoprompt.rag.knowledge_base import create_knowledge_base_from_dataset
    import tempfile
    import os

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        kb = create_knowledge_base_from_dataset(
            dataset,
            temp_path,
            samples_per_category=samples_per_category
        )

        # Load it back
        kb = KnowledgeBase.load(temp_path)

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return kb


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build knowledge base for RAG-enhanced detection"
    )

    parser.add_argument(
        "--source",
        choices=["default", "dataset"],
        default="default",
        help="Source for building KB (default: default)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset file (required if source=dataset)"
    )

    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=2,
        help="Number of samples per category (default: 2)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/knowledge_base.json",
        help="Output path for knowledge base (default: ./outputs/knowledge_base.json)"
    )

    args = parser.parse_args()

    # Validate args
    if args.source == "dataset" and not args.dataset:
        parser.error("--dataset is required when source=dataset")

    # Build KB
    if args.source == "default":
        kb = build_from_default()
    else:
        if not Path(args.dataset).exists():
            print(f"❌ Dataset file not found: {args.dataset}")
            return 1

        kb = build_from_dataset(args.dataset, args.samples_per_category)

    # Save KB
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kb.save(str(output_path))

    print(f"\n💾 Knowledge base saved to: {output_path}")

    # Show final stats
    stats = kb.statistics()
    print(f"\n📊 Final Statistics:")
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Major categories: {stats['major_categories']}")
    print(f"   Middle categories: {stats['middle_categories']}")
    print(f"   CWE types: {stats['cwe_types']}")

    print("\n✨ Done!")
    print("\nNext steps:")
    print("   1. Use this KB with RAGThreeLayerDetector")
    print("   2. Run: uv run python scripts/demo_rag_detection.py")
    print("   3. Experiment with different retrieval strategies")

    return 0


if __name__ == "__main__":
    sys.exit(main())
