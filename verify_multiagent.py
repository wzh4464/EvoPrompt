#!/usr/bin/env python3
"""Quick verification script for multi-agent components."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("🔍 Verifying Multi-Agent Components...")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from evoprompt.prompts.hierarchical import (
        HierarchicalPrompt,
        PromptHierarchy,
        CWECategory,
        get_cwe_major_category
    )
    print("   ✅ Hierarchical prompts")
except Exception as e:
    print(f"   ❌ Hierarchical prompts: {e}")
    sys.exit(1)

try:
    from evoprompt.evaluators.statistics import (
        DetectionStatistics,
        BatchStatistics,
        StatisticsCollector
    )
    print("   ✅ Statistics collectors")
except Exception as e:
    print(f"   ❌ Statistics collectors: {e}")
    sys.exit(1)

try:
    from evoprompt.optimization.meta_optimizer import (
        MetaOptimizer,
        OptimizationContext
    )
    print("   ✅ Meta optimizer")
except Exception as e:
    print(f"   ❌ Meta optimizer: {e}")
    sys.exit(1)

try:
    from evoprompt.multiagent.agents import (
        DetectionAgent,
        MetaAgent,
        AgentRole,
        AgentConfig,
        create_detection_agent,
        create_meta_agent
    )
    print("   ✅ Multi-agent framework")
except Exception as e:
    print(f"   ❌ Multi-agent framework: {e}")
    sys.exit(1)

try:
    from evoprompt.multiagent.coordinator import (
        MultiAgentCoordinator,
        CoordinatorConfig,
        CoordinationStrategy
    )
    print("   ✅ Multi-agent coordinator")
except Exception as e:
    print(f"   ❌ Multi-agent coordinator: {e}")
    sys.exit(1)

try:
    from evoprompt.algorithms.coevolution import CoevolutionaryAlgorithm
    print("   ✅ Coevolutionary algorithm")
except Exception as e:
    print(f"   ❌ Coevolutionary algorithm: {e}")
    sys.exit(1)

# Test basic functionality
print("\n2. Testing basic functionality...")

# Test CWE categorization
cwe_120_cat = get_cwe_major_category("CWE-120")
assert cwe_120_cat == CWECategory.MEMORY, "CWE-120 should be Memory category"
print("   ✅ CWE categorization works")

# Test hierarchical prompt
hierarchy = PromptHierarchy()
hierarchy.initialize_with_defaults()
assert hierarchy.router_prompt is not None, "Router prompt should be initialized"
assert len(hierarchy.category_prompts) > 0, "Category prompts should be initialized"
print("   ✅ Hierarchical prompt initialization")

# Test statistics
stats = DetectionStatistics()
stats.add_prediction("vulnerable", "vulnerable", category="CWE-120")
stats.add_prediction("benign", "benign")
stats.add_prediction("vulnerable", "benign")  # False positive
stats.compute_metrics()
assert stats.total_samples == 3, "Should have 3 samples"
assert stats.accuracy > 0, "Accuracy should be calculated"
assert stats.f1_score > 0, "F1 score should be calculated"
print("   ✅ Statistics collection and computation")

# Test agent configuration
detection_config = AgentConfig(
    role=AgentRole.DETECTION,
    model_name="test-model",
    temperature=0.1
)
assert detection_config.role == AgentRole.DETECTION
print("   ✅ Agent configuration")

# Test statistics collector
collector = StatisticsCollector()
collector.add_generation_stats(0, stats)
summary = collector.get_generation_summary(0)
assert summary is not None
print("   ✅ Statistics collector")

# Test batch statistics
batch_stat = BatchStatistics(
    batch_id=0,
    batch_size=10,
    statistics=stats
)
batch_summary = batch_stat.get_summary()
assert batch_summary["batch_id"] == 0
print("   ✅ Batch statistics")

print("\n" + "=" * 60)
print("🎉 All verification tests passed!")
print("\n📝 Next steps:")
print("   1. Configure .env with API keys (API_KEY, META_API_KEY)")
print("   2. Prepare sample data: uv run python scripts/demo_primevul_1percent.py")
print("   3. Run demo: uv run python scripts/demo_multiagent_coevolution.py")
print("\n📚 See MULTIAGENT_README.md for detailed documentation")
