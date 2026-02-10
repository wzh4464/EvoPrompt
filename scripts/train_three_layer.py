#!/usr/bin/env python3
"""完整的三层检测训练脚本

支持:
- RAG增强 (可选)
- Scale增强 (可选)
- Multi-agent协同进化
- 层级训练策略
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evoprompt.prompts.hierarchical_three_layer import (
    ThreeLayerPromptFactory,
    ThreeLayerPromptSet,
)
from evoprompt.detectors.three_layer_detector import ThreeLayerDetector, ThreeLayerEvaluator
from evoprompt.detectors.rag_three_layer_detector import RAGThreeLayerDetector
from evoprompt.rag.knowledge_base import KnowledgeBase, KnowledgeBaseBuilder
from evoprompt.data.dataset import PrimevulDataset
from evoprompt.llm.client import load_env_vars, create_llm_client
from evoprompt.multiagent.agents import create_detection_agent, create_meta_agent
from evoprompt.multiagent.coordinator import MultiAgentCoordinator, CoordinatorConfig
from evoprompt.algorithms.coevolution import CoevolutionaryAlgorithm
from evoprompt.utils.trace import TraceManager, TraceConfig, trace_enabled_from_env


def setup_environment():
    """配置环境"""
    load_env_vars()

    api_key = os.getenv("API_KEY")
    if not api_key:
        print("❌ API_KEY not found in .env")
        return False

    print("✅ Environment configured:")
    print(f"   Detection Model: {os.getenv('MODEL_NAME', 'gpt-4')}")
    print(f"   Meta Model: {os.getenv('META_MODEL_NAME', 'claude-4.5')}")

    return True


def load_or_build_knowledge_base(args):
    """加载或构建知识库

    Args:
        args: 命令行参数

    Returns:
        KnowledgeBase or None
    """
    if not args.use_rag:
        print("\n⏭️  RAG disabled, skipping knowledge base")
        return None

    print("\n📚 Knowledge Base Setup")
    print("=" * 70)

    # 检查是否有已存在的知识库
    if args.kb_path and Path(args.kb_path).exists():
        print(f"   📖 Loading existing KB: {args.kb_path}")
        kb = KnowledgeBase.load(args.kb_path)
        stats = kb.statistics()
        print(f"   ✅ Loaded {stats['total_examples']} examples")
        return kb

    # 构建新知识库
    print("   🔨 Building new knowledge base...")

    if args.kb_from_dataset:
        # 从数据集构建
        print(f"   📂 Source: Dataset ({args.kb_samples_per_category} samples/category)")
        dataset = PrimevulDataset(args.train_file, "train")

        from evoprompt.rag.knowledge_base import create_knowledge_base_from_dataset
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            kb = create_knowledge_base_from_dataset(
                dataset,
                temp_path,
                samples_per_category=args.kb_samples_per_category
            )
            kb = KnowledgeBase.load(temp_path)
        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()
    else:
        # 使用默认示例
        print("   📦 Source: Default examples")
        kb = KnowledgeBaseBuilder.create_default_kb()

    stats = kb.statistics()
    print(f"   ✅ Built KB with {stats['total_examples']} examples")

    # 保存知识库
    if args.kb_path:
        Path(args.kb_path).parent.mkdir(parents=True, exist_ok=True)
        kb.save(args.kb_path)
        print(f"   💾 Saved to: {args.kb_path}")

    return kb


def create_detector(prompt_set, llm_client, kb, args):
    """创建检测器

    Args:
        prompt_set: Prompt集合
        llm_client: LLM客户端
        kb: 知识库 (可为None)
        args: 命令行参数

    Returns:
        检测器实例
    """
    print("\n🔧 Creating Detector")
    print("=" * 70)

    if args.use_rag and kb is not None:
        print(f"   🎯 Type: RAG-Enhanced Three-Layer")
        print(f"   📊 RAG top-k: {args.rag_top_k}")
        print(f"   🔍 Retriever: {args.rag_retriever_type}")
        print(f"   ⚡ Scale enhancement: {args.use_scale}")

        detector = RAGThreeLayerDetector(
            prompt_set=prompt_set,
            llm_client=llm_client,
            knowledge_base=kb,
            use_scale_enhancement=args.use_scale,
            retriever_type=args.rag_retriever_type,
            top_k=args.rag_top_k
        )
    else:
        print(f"   🎯 Type: Basic Three-Layer")
        print(f"   ⚡ Scale enhancement: {args.use_scale}")

        detector = ThreeLayerDetector(
            prompt_set=prompt_set,
            llm_client=llm_client,
            use_scale_enhancement=args.use_scale
        )

    return detector


def run_evaluation(detector, dataset, args, trace_manager: TraceManager = None):
    """运行评估

    Args:
        detector: 检测器
        dataset: 数据集
        args: 命令行参数

    Returns:
        评估指标字典
    """
    print("\n📊 Running Evaluation")
    print("=" * 70)

    evaluator = ThreeLayerEvaluator(detector, dataset)

    eval_count = args.eval_samples if args.eval_samples is not None else "all"
    print(f"   🔍 Evaluating on {eval_count} samples...")
    start = time.time()

    # 使用verbose=True打印详细的Macro/Weighted/Micro F1
    metrics = evaluator.evaluate(sample_size=args.eval_samples, verbose=True)

    elapsed = time.time() - start

    print(f"\n   ✅ Evaluation completed in {elapsed:.1f}s")

    if trace_manager and trace_manager.enabled:
        trace_manager.log_event(
            "evaluation",
            {
                "mode": "baseline" if not args.train else "evaluation",
                "metrics": metrics,
                "eval_samples": args.eval_samples,
            },
        )

    return metrics


def run_training(initial_prompt_set, detector, dataset, kb, args, trace_manager: TraceManager = None):
    """运行训练

    Args:
        initial_prompt_set: 初始prompt集合
        detector: 检测器
        dataset: 数据集
        kb: 知识库
        args: 命令行参数

    Returns:
        优化后的prompt集合
    """
    print("\n🚀 Starting Training")
    print("=" * 70)

    # 创建agents
    print("   🤖 Creating agents...")
    detection_agent = create_detection_agent(
        model_name=os.getenv("MODEL_NAME", "gpt-4")
    )
    meta_agent = create_meta_agent(
        model_name=os.getenv("META_MODEL_NAME", "claude-4.5")
    )

    # 创建协调器
    print("   🎯 Creating coordinator...")
    coordinator_config = CoordinatorConfig(
        batch_size=args.batch_size,
        enable_batch_feedback=True,
        statistics_window=5
    )
    coordinator = MultiAgentCoordinator(
        detection_agent=detection_agent,
        meta_agent=meta_agent,
        config=coordinator_config,
        trace_manager=trace_manager,
    )

    # 创建进化算法配置
    print("   🧬 Creating evolution algorithm...")
    config = {
        "population_size": args.population_size,
        "max_generations": args.max_generations,
        "elite_size": args.elite_size,
        "mutation_rate": args.mutation_rate,
        "meta_improve_interval": args.meta_improve_interval,
        "meta_improve_count": args.meta_improve_count,
        "top_k": args.elite_size,
        "enable_elitism": True,
        "meta_improvement_rate": 0.3
    }

    algorithm = CoevolutionaryAlgorithm(
        config=config,
        coordinator=coordinator,
        dataset=dataset
    )

    print()
    print(f"   📋 Configuration:")
    print(f"      Population: {args.population_size}")
    print(f"      Generations: {args.max_generations}")
    print(f"      Elite size: {args.elite_size}")
    print(f"      Mutation rate: {args.mutation_rate}")
    print(f"      Batch size: {args.batch_size}")
    print(f"      Meta improve interval: {args.meta_improve_interval}")
    print(f"      Meta improve count: {args.meta_improve_count}")

    # 运行进化
    print()
    print("   🎬 Starting evolution...")
    print("=" * 70)

    # 提取初始prompts - 使用layer1 prompt作为初始种群
    # TODO: 未来应该支持完整的三层prompt集合优化
    initial_prompts = [initial_prompt_set.layer1_prompt]

    if trace_manager and trace_manager.enabled:
        trace_manager.save_prompt_snapshot(
            "initial_layer1_prompt",
            initial_prompt_set.layer1_prompt,
            metadata={"stage": "initialization"},
        )

    best_individual = algorithm.evolve(initial_prompts=initial_prompts)

    print()
    print("   ✅ Training completed!")
    print(f"      Best fitness: {best_individual.fitness:.4f}")

    # TODO: 将best_individual.prompt转换回ThreeLayerPromptSet
    # 目前返回初始prompt集合
    if trace_manager and trace_manager.enabled:
        trace_manager.log_event(
            "training_complete",
            {
                "best_fitness": getattr(best_individual, "fitness", None),
            },
        )

    return initial_prompt_set


def save_results(output_dir, metrics, prompt_set, args):
    """保存结果

    Args:
        output_dir: 输出目录
        metrics: 评估指标
        prompt_set: Prompt集合
        args: 命令行参数
    """
    print(f"\n💾 Saving Results to: {output_dir}")
    print("=" * 70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config = {
        "use_rag": args.use_rag,
        "use_scale": args.use_scale,
        "rag_top_k": args.rag_top_k if args.use_rag else None,
        "rag_retriever_type": args.rag_retriever_type if args.use_rag else None,
        "train": args.train,
        "population_size": args.population_size if args.train else None,
        "max_generations": args.max_generations if args.train else None,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("   ✅ config.json")

    # 保存评估结果
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("   ✅ metrics.json")

    # 保存prompt集合
    with open(output_dir / "prompts.json", "w") as f:
        json.dump(prompt_set.to_dict(), f, indent=2, ensure_ascii=False)
    print("   ✅ prompts.json")

    # 保存可读的prompt文本
    with open(output_dir / "prompts.txt", "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("Three-Layer Prompts\n")
        f.write("="*70 + "\n\n")

        f.write("LAYER 1 PROMPT\n")
        f.write("-"*70 + "\n")
        f.write(prompt_set.layer1_prompt + "\n\n")

        f.write("LAYER 2 PROMPTS\n")
        f.write("-"*70 + "\n")
        for cat, prompt in prompt_set.layer2_prompts.items():
            f.write(f"\n[{cat.value}]\n")
            f.write(prompt + "\n")

        f.write("\nLAYER 3 PROMPTS\n")
        f.write("-"*70 + "\n")
        for cat, prompt in prompt_set.layer3_prompts.items():
            f.write(f"\n[{cat.value}]\n")
            f.write(prompt + "\n")
    print("   ✅ prompts.txt")

    print(f"\n📁 All results saved to: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="三层检测训练脚本 - 支持RAG和Scale增强"
    )

    # 数据集参数
    parser.add_argument(
        "--train-file",
        default="./data/primevul/primevul/dev.jsonl",
        help="训练数据文件"
    )
    parser.add_argument(
        "--eval-file",
        default="./data/primevul/primevul/primevul_test.jsonl",
        help="评估数据文件"
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="评估样本数量 (默认全量)"
    )

    # RAG参数
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="启用RAG增强"
    )
    parser.add_argument(
        "--kb-path",
        default="./outputs/knowledge_base.json",
        help="知识库路径"
    )
    parser.add_argument(
        "--kb-from-dataset",
        action="store_true",
        help="从数据集构建知识库"
    )
    parser.add_argument(
        "--kb-samples-per-category",
        type=int,
        default=3,
        help="每个类别采样数量"
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=2,
        help="RAG检索top-k"
    )
    parser.add_argument(
        "--rag-retriever-type",
        choices=["lexical", "embedding"],
        default="lexical",
        help="RAG检索器类型"
    )

    # Scale增强参数
    parser.add_argument(
        "--use-scale",
        action="store_true",
        help="启用Scale增强"
    )

    # 训练参数
    parser.add_argument(
        "--train",
        action="store_true",
        help="运行训练 (否则仅评估)"
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=5,
        help="种群大小"
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=10,
        help="最大代数"
    )
    parser.add_argument(
        "--elite-size",
        type=int,
        default=1,
        help="精英个体数量"
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.3,
        help="变异率"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="批处理大小"
    )
    parser.add_argument(
        "--meta-improve-interval",
        type=int,
        default=3,
        help="Meta优化间隔"
    )
    parser.add_argument(
        "--meta-improve-count",
        type=int,
        default=2,
        help="每次Meta优化个体数"
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="关闭详细追踪输出 (默认开启)",
    )

    # 输出参数
    parser.add_argument(
        "--output-dir",
        help="输出目录 (默认自动生成)"
    )

    args = parser.parse_args()

    if args.release:
        os.environ["EVOPROMPT_RELEASE"] = "1"

    # 设置输出目录
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "train" if args.train else "eval"
        rag_suffix = "_rag" if args.use_rag else ""
        scale_suffix = "_scale" if args.use_scale else ""
        args.output_dir = f"./outputs/three_layer_{mode}{rag_suffix}{scale_suffix}_{timestamp}"

    trace_enabled = not args.release if args.release else trace_enabled_from_env()
    trace_manager = TraceManager(
        TraceConfig(
            enabled=trace_enabled,
            base_dir=Path(args.output_dir),
            experiment_id=Path(args.output_dir).name,
        )
    )

    # 开始
    print("🏗️  Three-Layer Detection Training System")
    print("=" * 70)
    print()
    print("📋 Configuration:")
    print(f"   Mode: {'Training' if args.train else 'Evaluation Only'}")
    print(f"   RAG: {'✅ Enabled' if args.use_rag else '❌ Disabled'}")
    print(f"   Scale: {'✅ Enabled' if args.use_scale else '❌ Disabled'}")
    print(f"   Output: {args.output_dir}")

    # 环境设置
    if not setup_environment():
        return 1

    # 加载知识库
    kb = load_or_build_knowledge_base(args)

    # 加载数据集
    print("\n📂 Loading Dataset")
    print("=" * 70)
    print(f"   Training: {args.train_file}")
    print(f"   Evaluation: {args.eval_file}")

    train_dataset = PrimevulDataset(args.train_file, "train")
    eval_dataset = PrimevulDataset(args.eval_file, "dev")

    print(f"   ✅ Train: {len(train_dataset)} samples")
    print(f"   ✅ Eval: {len(eval_dataset)} samples")

    # 创建初始prompt集合
    print("\n📝 Creating Initial Prompts")
    print("=" * 70)
    prompt_set = ThreeLayerPromptFactory.create_default_prompt_set()
    counts = prompt_set.count_prompts()
    print(f"   ✅ Created {counts['total']} prompts")
    print(f"      Layer 1: {counts['layer1']}")
    print(f"      Layer 2: {counts['layer2']}")
    print(f"      Layer 3: {counts['layer3']}")

    # 创建LLM客户端
    llm_client = create_llm_client(llm_type=os.getenv("MODEL_NAME", "gpt-4"))

    # 创建检测器
    detector = create_detector(prompt_set, llm_client, kb, args)

    # 评估基线
    print("\n📊 Baseline Evaluation")
    print("=" * 70)
    baseline_metrics = run_evaluation(detector, eval_dataset, args, trace_manager=trace_manager)

    # 训练
    if args.train:
        prompt_set = run_training(prompt_set, detector, train_dataset, kb, args, trace_manager=trace_manager)

        # 重新创建检测器并评估
        print("\n📊 Final Evaluation")
        print("=" * 70)
        detector = create_detector(prompt_set, llm_client, kb, args)
        final_metrics = run_evaluation(detector, eval_dataset, args, trace_manager=trace_manager)

        # 保存最终结果
        save_results(args.output_dir, final_metrics, prompt_set, args)
    else:
        # 仅评估，保存基线结果
        save_results(args.output_dir, baseline_metrics, prompt_set, args)

    print("\n" + "=" * 70)
    print("✨ Completed!")
    print()
    print("📁 Results:")
    print(f"   {args.output_dir}/")
    print(f"   ├── config.json      # 配置")
    print(f"   ├── metrics.json     # 评估指标")
    print(f"   ├── prompts.json     # Prompt集合")
    print(f"   └── prompts.txt      # 可读Prompt")

    if kb and args.kb_path:
        print()
        print(f"📚 Knowledge Base: {args.kb_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
