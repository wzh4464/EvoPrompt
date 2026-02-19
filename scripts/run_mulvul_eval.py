#!/usr/bin/env python3
"""MulVul 多智能体系统评估脚本

使用完整的 Router-Detector-Aggregator 架构:
- RouterAgent: Top-k 路由 + 跨类型对比检索
- DetectorAgent: 类别专用检测器
- DecisionAggregator: 置信度加权聚合
- Recall@k / F1 评估指标
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, "src")

from tqdm import tqdm
from evoprompt.llm.client import create_llm_client, load_env_vars
from evoprompt.agents import MulVulDetector, DetectionResult
from evoprompt.rag.retriever import MulVulRetriever
from evoprompt.evaluators.multiclass_metrics import (
    MultiClassMetrics,
    RouterMetrics,
    recall_at_k,
)
from evoprompt.data.cwe_categories import map_cwe_to_major

# Major category mapping
CATEGORY_TO_MAJOR = {
    "Buffer Errors": "Memory",
    "Memory Management": "Memory",
    "Pointer Dereference": "Memory",
    "Integer Errors": "Memory",
    "Injection": "Injection",
    "Concurrency Issues": "Logic",
    "Path Traversal": "Input",
    "Cryptography Issues": "Crypto",
    "Information Exposure": "Logic",
    "Other": "Logic",
    "Benign": "Benign",
}


def load_jsonl_data(data_file: str) -> List[Dict]:
    """加载 JSONL 数据"""
    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return samples


def get_ground_truth(item: Dict) -> Tuple[str, str]:
    """获取样本的真实类别 (category, major)"""
    target = int(item.get("target", 0))
    if target == 0:
        return "Benign", "Benign"

    cwe_codes = item.get("cwe", [])
    if isinstance(cwe_codes, str):
        cwe_codes = [cwe_codes] if cwe_codes else []

    category = map_cwe_to_major(cwe_codes) if cwe_codes else "Other"
    major = CATEGORY_TO_MAJOR.get(category, "Logic")
    return category, major


def get_sample_id(item: Dict) -> str:
    if "idx" in item:
        return str(item["idx"])
    return str(hash(item.get("func", "")[:200]))


def balanced_sample(samples: List[Dict], seed: int = None) -> List[Dict]:
    """平衡采样: benign : all_vuls = 1:1"""
    if seed is not None:
        random.seed(seed)

    benign = [s for s in samples if int(s.get("target", 0)) == 0]
    vuls = [s for s in samples if int(s.get("target", 0)) == 1]

    n_vuls = len(vuls)
    n_benign = len(benign)

    # 采样使 benign:vul = 1:1
    if n_benign > n_vuls:
        benign = random.sample(benign, n_vuls)
    elif n_vuls > n_benign:
        vuls = random.sample(vuls, n_benign)

    result = benign + vuls
    random.shuffle(result)
    return result


def evaluate_single_sample(
    item: Dict,
    detector: MulVulDetector,
    expected_major: str,
) -> Dict:
    """使用 MulVul 评估单个样本"""
    code = item.get("func", "")

    try:
        # 获取完整检测结果
        details = detector.detect_with_details(code)

        # 提取路由结果
        routing = details["routing"]
        top_k_categories = routing["top_k"]

        # 提取最终预测
        final = details["final"]

        # 检查路由是否正确 (Recall@k)
        top_k_cats = [c[0] for c in top_k_categories]
        routing_correct = expected_major in top_k_cats

        # 检查最终预测是否正确
        final_correct = final["category"] == expected_major

        return {
            "expected_major": expected_major,
            "routing_top_k": top_k_categories,
            "routing_correct": routing_correct,
            "final_prediction": final["prediction"],
            "final_category": final["category"],
            "final_confidence": final["confidence"],
            "final_evidence": final["evidence"][:200] if final["evidence"] else "",
            "final_correct": final_correct,
            "detector_results": details["detectors"],
            "error": None,
        }

    except Exception as e:
        return {
            "expected_major": expected_major,
            "routing_top_k": [],
            "routing_correct": False,
            "final_prediction": "Error",
            "final_category": "Error",
            "final_confidence": 0.0,
            "final_evidence": "",
            "final_correct": False,
            "detector_results": [],
            "error": str(e),
        }


def run_mulvul_evaluation(
    data_file: str,
    max_workers: int = 32,
    max_samples: Optional[int] = None,
    output_dir: str = "./outputs",
    k: int = 3,
    balanced: bool = False,
    seed: int = None,
    kb_path: str = None,
) -> Dict[str, Any]:
    """运行 MulVul 多智能体评估"""

    load_env_vars()

    print("=" * 70)
    print("🔥 MulVul 多智能体系统评估")
    print("=" * 70)
    print(f"   🔀 Router: Top-{k} routing")
    print(f"   🔍 Detectors: Category-specific (Memory/Injection/Logic/Input/Crypto)")
    print(f"   📊 Aggregator: Confidence-based")
    if balanced:
        print(f"   ⚖️  Balanced: benign:vul = 1:1")
    if kb_path:
        print(f"   📚 Knowledge Base: {kb_path}")
    print("=" * 70)

    # 加载知识库
    retriever = None
    if kb_path and os.path.exists(kb_path):
        retriever = MulVulRetriever(knowledge_base_path=kb_path)

    # 加载数据
    print(f"\n📂 加载数据: {data_file}")
    samples = load_jsonl_data(data_file)
    print(f"   总样本数: {len(samples)}")

    # 平衡采样
    if balanced:
        samples = balanced_sample(samples, seed=seed)
        print(f"   平衡采样后: {len(samples)}")

    if max_samples:
        samples = samples[:max_samples]
        print(f"   评估样本数: {len(samples)}")

    # 按类别统计
    category_counts = defaultdict(int)
    for item in samples:
        _, major = get_ground_truth(item)
        category_counts[major] += 1

    print("\n📊 数据分布:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cat:12s}: {count:5d}")

    # 创建评估任务
    eval_tasks = [(item, get_ground_truth(item)[1]) for item in samples]

    print(f"\n🚀 启动并发评估 (workers={max_workers})")

    # 统计结构
    router_metrics = RouterMetrics(k=k)
    detector_metrics = MultiClassMetrics()
    major_stats = defaultdict(lambda: {"total": 0, "routing_correct": 0, "final_correct": 0})

    start_time = time.time()
    results = []

    # 并发评估
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 每个线程创建独立的 detector
        def evaluate_with_detector(args):
            item, expected_major = args
            llm_client = create_llm_client()
            detector = MulVulDetector.create_default(llm_client, retriever=retriever, k=k, parallel=False)
            return evaluate_single_sample(item, detector, expected_major)

        futures = {
            executor.submit(evaluate_with_detector, task): task
            for task in eval_tasks
        }

        with tqdm(total=len(eval_tasks), desc="MulVul 评估", unit="样本") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)

                    # 更新 Router 指标
                    router_metrics.add_prediction(
                        result["routing_top_k"],
                        result["expected_major"]
                    )

                    # 更新 Detector 指标
                    detector_metrics.add_prediction(
                        result["final_category"],
                        result["expected_major"]
                    )

                    # 更新分类统计
                    major = result["expected_major"]
                    major_stats[major]["total"] += 1
                    if result["routing_correct"]:
                        major_stats[major]["routing_correct"] += 1
                    if result["final_correct"]:
                        major_stats[major]["final_correct"] += 1

                    # 更新进度条
                    total = sum(s["total"] for s in major_stats.values())
                    correct = sum(s["final_correct"] for s in major_stats.values())
                    acc = correct / total if total > 0 else 0
                    pbar.set_postfix({"acc": f"{acc:.1%}"})

                except Exception as e:
                    tqdm.write(f"❌ 评估失败: {e}")

                pbar.update(1)

    elapsed = time.time() - start_time

    # 计算指标
    total = len(results)
    routing_correct = sum(1 for r in results if r["routing_correct"])
    final_correct = sum(1 for r in results if r["final_correct"])

    recall_k = routing_correct / total if total > 0 else 0
    accuracy = final_correct / total if total > 0 else 0

    # 打印结果
    print("\n" + "=" * 70)
    print("📊 MulVul 评估结果")
    print("=" * 70)

    print(f"\n🔀 Router Agent (Top-{k} Routing):")
    print(f"   Recall@{k}: {recall_k:.2%} ({routing_correct}/{total})")
    router_report = router_metrics.get_report()
    print(f"   MRR: {router_report['mrr']:.4f}")

    print(f"\n🎯 Final Detection:")
    print(f"   Accuracy: {accuracy:.2%} ({final_correct}/{total})")
    print(f"   Macro-F1: {detector_metrics.compute_macro_f1():.4f}")

    print(f"\n⏱️  Performance:")
    print(f"   耗时: {elapsed:.1f}秒")
    print(f"   吞吐量: {total / elapsed:.1f} 样本/秒")

    # 分类别统计
    print("\n📈 各类别统计:")
    print(f"   {'Category':<12} {'Recall@k':>10} {'Accuracy':>10} {'Total':>8}")
    print("   " + "-" * 44)
    for major in ["Memory", "Injection", "Input", "Crypto", "Logic", "Benign"]:
        stats = major_stats.get(major, {"total": 0, "routing_correct": 0, "final_correct": 0})
        if stats["total"] > 0:
            r_acc = stats["routing_correct"] / stats["total"]
            f_acc = stats["final_correct"] / stats["total"]
            print(f"   {major:<12} {r_acc:>10.2%} {f_acc:>10.2%} {stats['total']:>8}")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_file": data_file,
        "k": k,
        "total_samples": total,
        "elapsed_seconds": elapsed,
        "router": {
            "recall_at_k": recall_k,
            "mrr": router_report["mrr"],
        },
        "detector": {
            "accuracy": accuracy,
            "macro_f1": detector_metrics.compute_macro_f1(),
            "weighted_f1": detector_metrics.compute_weighted_f1(),
        },
        "per_category": {k: dict(v) for k, v in major_stats.items()},
    }

    output_file = Path(output_dir) / f"mulvul_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n💾 结果已保存: {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="MulVul 多智能体系统评估")
    parser.add_argument("--data", default="./data/primevul/primevul/primevul_valid.jsonl")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default="./outputs")
    parser.add_argument("--k", type=int, default=3, help="Top-k routing")
    parser.add_argument("--epochs", type=int, default=1, help="Number of evaluation runs")
    parser.add_argument("--balanced", action="store_true", help="Balance benign:vul = 1:1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balanced sampling")
    parser.add_argument("--kb", default="./data/knowledge_base_hierarchical.json", help="Knowledge base path")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"❌ 数据文件不存在: {args.data}")
        return 1

    all_results = []
    for epoch in range(1, args.epochs + 1):
        if args.epochs > 1:
            print(f"\n{'#'*70}")
            print(f"# Epoch {epoch}/{args.epochs}")
            print(f"{'#'*70}")

        summary = run_mulvul_evaluation(
            args.data,
            max_workers=args.workers,
            max_samples=args.max_samples,
            output_dir=args.output,
            k=args.k,
            balanced=args.balanced,
            seed=args.seed + epoch if args.balanced else None,
            kb_path=args.kb,
        )
        all_results.append(summary)

    # Print aggregate results for multiple epochs
    if args.epochs > 1:
        print("\n" + "=" * 70)
        print(f"📊 {args.epochs} Epochs 汇总")
        print("=" * 70)

        accuracies = [r["detector"]["accuracy"] for r in all_results]
        recall_ks = [r["router"]["recall_at_k"] for r in all_results]
        f1s = [r["detector"]["macro_f1"] for r in all_results]

        print(f"   Accuracy: {sum(accuracies)/len(accuracies):.2%} ± {max(accuracies)-min(accuracies):.2%}")
        print(f"   Recall@k: {sum(recall_ks)/len(recall_ks):.2%} ± {max(recall_ks)-min(recall_ks):.2%}")
        print(f"   Macro-F1: {sum(f1s)/len(f1s):.4f} ± {max(f1s)-min(f1s):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
