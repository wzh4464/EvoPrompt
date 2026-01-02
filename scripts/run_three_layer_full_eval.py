#!/usr/bin/env python3
"""并发全量评估三层检测器

支持:
- 全量 Primevul JSONL 数据
- 并发加速
- 三层检测 (Major → Middle → CWE)
- 详细结果输出
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, "src")

from evoprompt.llm.client import create_llm_client, load_env_vars
from evoprompt.prompts.hierarchical_three_layer import (
    ThreeLayerPromptFactory,
    MajorCategory,
    MAJOR_TO_MIDDLE,
    MIDDLE_TO_CWE,
)
from evoprompt.detectors.three_layer_detector import ThreeLayerDetector
from evoprompt.data.cwe_categories import CWE_MAJOR_CATEGORIES, map_cwe_to_major


def load_jsonl_data(data_file: str) -> List[Dict]:
    """加载 JSONL 数据"""
    print(f"📂 加载数据: {data_file}")

    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                samples.append(item)
            except json.JSONDecodeError:
                continue

    print(f"   总样本数: {len(samples)}")
    return samples


def get_ground_truth_category(item: Dict) -> Tuple[str, bool]:
    """获取样本的真实类别

    Returns:
        (category, is_vulnerable)
    """
    target = int(item.get("target", 0))

    if target == 0:
        return "Benign", False

    cwe_codes = item.get("cwe", [])
    if isinstance(cwe_codes, str):
        cwe_codes = [cwe_codes] if cwe_codes else []

    category = map_cwe_to_major(cwe_codes) if cwe_codes else "Other"
    return category, True


def evaluate_single_sample(
    item: Dict,
    prompt_set,
    expected_category: str,
    use_scale: bool = False
) -> Dict:
    """评估单个样本"""

    # 每个线程创建独立的 LLM 客户端
    llm_client = create_llm_client()

    detector = ThreeLayerDetector(
        prompt_set=prompt_set,
        llm_client=llm_client,
        use_scale_enhancement=use_scale
    )

    code = item.get("func", "")

    try:
        predicted_cwe, details = detector.detect(code, return_intermediate=True)

        layer1_pred = details.get("layer1", "Unknown")
        layer2_pred = details.get("layer2", "Unknown")
        layer3_pred = details.get("layer3", "Unknown")

        # 判断 Layer1 是否正确
        layer1_correct = False
        if expected_category == "Benign":
            layer1_correct = layer1_pred == "Benign"
        else:
            # 映射到我们的 Major Category
            major_mapping = {
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
            }
            expected_major = major_mapping.get(expected_category, "Logic")
            layer1_correct = layer1_pred == expected_major

        return {
            "expected_category": expected_category,
            "layer1_pred": layer1_pred,
            "layer2_pred": layer2_pred,
            "layer3_pred": layer3_pred,
            "layer1_correct": layer1_correct,
            "error": None
        }

    except Exception as e:
        return {
            "expected_category": expected_category,
            "layer1_pred": None,
            "layer2_pred": None,
            "layer3_pred": None,
            "layer1_correct": False,
            "error": str(e)
        }


def evaluate_category_batch(
    category: str,
    samples: List[Dict],
    prompt_set,
    max_samples: Optional[int] = None,
    use_scale: bool = False
) -> Dict[str, Any]:
    """评估单个类别的所有样本"""

    if not samples:
        return {"category": category, "total": 0, "layer1_correct": 0, "accuracy": 0.0}

    eval_samples = samples[:max_samples] if max_samples else samples

    results = []
    layer1_correct = 0

    for item in eval_samples:
        result = evaluate_single_sample(item, prompt_set, category, use_scale)
        results.append(result)
        if result.get("layer1_correct"):
            layer1_correct += 1

    accuracy = layer1_correct / len(eval_samples) if eval_samples else 0

    return {
        "category": category,
        "total": len(eval_samples),
        "layer1_correct": layer1_correct,
        "accuracy": accuracy,
        "sample_results": results[:5]  # 只保留前5个示例
    }


def run_concurrent_evaluation(
    data_file: str,
    max_workers: int = 64,
    max_samples_per_category: Optional[int] = None,
    output_dir: str = "./outputs",
    use_scale: bool = False
) -> Dict[str, Any]:
    """并发全量评估"""

    load_env_vars()

    print("=" * 70)
    print("🔥 三层检测器并发全量评估")
    if use_scale:
        print("   📊 SCALE Enhancement: ENABLED")
    print("=" * 70)

    # 加载数据
    samples = load_jsonl_data(data_file)

    # 按类别分组
    category_samples: Dict[str, List[Dict]] = {cat: [] for cat in CWE_MAJOR_CATEGORIES}
    category_samples["Benign"] = []

    for item in samples:
        category, is_vuln = get_ground_truth_category(item)
        if category in category_samples:
            category_samples[category].append(item)
        else:
            category_samples["Other"].append(item)

    print("\n📊 数据分布:")
    for cat, cat_samples in sorted(category_samples.items(), key=lambda x: len(x[1]), reverse=True):
        if cat_samples:
            print(f"   {cat:25s}: {len(cat_samples):5d} 样本")

    # 创建 prompt set
    prompt_set = ThreeLayerPromptFactory.create_default_prompt_set()

    # 并发评估
    print(f"\n🚀 启动并发评估 (workers={max_workers})")
    if max_samples_per_category:
        print(f"   每类最大样本: {max_samples_per_category}")

    start_time = time.time()
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for category, cat_samples in category_samples.items():
            if cat_samples:
                future = executor.submit(
                    evaluate_category_batch,
                    category, cat_samples, prompt_set, max_samples_per_category, use_scale
                )
                futures[future] = category

        for future in as_completed(futures):
            category = futures[future]
            try:
                result = future.result()
                results[category] = result
                print(f"   ✅ {category:25s}: {result['accuracy']:6.2%} ({result['layer1_correct']:4d}/{result['total']:4d})")
            except Exception as e:
                print(f"   ❌ {category:25s}: 失败 - {e}")
                results[category] = {"category": category, "error": str(e)}

    elapsed = time.time() - start_time

    # 汇总
    total_evaluated = sum(r.get("total", 0) for r in results.values())
    total_correct = sum(r.get("layer1_correct", 0) for r in results.values())
    overall_accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0

    category_accuracies = [r.get("accuracy", 0) for r in results.values() if r.get("total", 0) > 0]
    macro_accuracy = sum(category_accuracies) / len(category_accuracies) if category_accuracies else 0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_file": data_file,
        "use_scale": use_scale,
        "elapsed_seconds": elapsed,
        "total_samples": total_evaluated,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "macro_accuracy": macro_accuracy,
        "category_results": {k: {kk: vv for kk, vv in v.items() if kk != "sample_results"}
                            for k, v in results.items()},
    }

    # 打印结果
    print("\n" + "=" * 70)
    print("📊 三层检测评估结果")
    print("=" * 70)
    print(f"总样本数: {total_evaluated}")
    print(f"Layer1 正确数: {total_correct}")
    print(f"Layer1 准确率 (Micro): {overall_accuracy:.2%}")
    print(f"Layer1 宏平均准确率 (Macro): {macro_accuracy:.2%}")
    print(f"耗时: {elapsed:.1f}秒")
    print(f"吞吐量: {total_evaluated / elapsed:.1f} 样本/秒")

    print("\n📈 各类别 Layer1 准确率:")
    for cat, res in sorted(results.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True):
        if res.get('total', 0) > 0:
            print(f"   {cat:25s}: {res['accuracy']:6.2%} ({res['layer1_correct']:4d}/{res['total']:4d})")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / f"three_layer_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n💾 结果已保存: {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="三层检测器并发评估")
    parser.add_argument("--data", default="./data/primevul/primevul/primevul_valid.jsonl", help="JSONL 数据文件")
    parser.add_argument("--workers", type=int, default=64, help="并发线程数")
    parser.add_argument("--max-samples", type=int, default=None, help="每类最大样本数")
    parser.add_argument("--output", default="./outputs", help="输出目录")
    parser.add_argument("--use-scale", action="store_true", help="启用 SCALE Enhancement")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"❌ 数据文件不存在: {args.data}")
        return 1

    run_concurrent_evaluation(
        args.data,
        max_workers=args.workers,
        max_samples_per_category=args.max_samples,
        output_dir=args.output,
        use_scale=args.use_scale
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
