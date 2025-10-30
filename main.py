#!/usr/bin/env python3
"""
EvoPrompt Main Entry - PrimeVul Layer-1 并发漏洞分类

功能:
1. 从 init/ 文件夹读取初始化 prompts
2. 对 PrimeVul 数据集进行 CWE 大类分层漏洞检测
3. 每 16 条 code 为一个 batch 进行批量处理
4. Batch 级别的分析和反馈机制指导 prompt 进化
5. 输出最终 prompt 和各类别的 precision/recall/f1-score 到 result/ 文件夹
"""

import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

# 添加 src 路径
sys.path.insert(0, "src")

from evoprompt.data.sampler import sample_primevul_1percent
from evoprompt.data.dataset import PrimevulDataset
from evoprompt.data.cwe_categories import (
    CWE_MAJOR_CATEGORIES,
    map_cwe_to_major,
    canonicalize_category,
)
from evoprompt.llm.client import create_default_client
from evoprompt.algorithms.base import Individual, Population
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


class BatchAnalyzer:
    """Batch 级别的分析器，对比预测结果和 ground truth 并生成反馈"""

    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size
        self.analysis_history = []

    def analyze_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
        batch_idx: int
    ) -> Dict[str, Any]:
        """
        分析一个 batch 的预测结果

        Returns:
            analysis: 包含准确率、错误模式、改进建议的字典
        """
        correct = sum(p == g for p, g in zip(predictions, ground_truths))
        accuracy = correct / len(predictions) if predictions else 0.0

        # 统计错误类型
        error_patterns = {}
        for pred, truth in zip(predictions, ground_truths):
            if pred != truth:
                error_key = f"{truth} -> {pred}"
                error_patterns[error_key] = error_patterns.get(error_key, 0) + 1

        # 生成改进建议
        improvement_suggestions = self._generate_improvement_suggestions(
            error_patterns, ground_truths, predictions
        )

        analysis = {
            "batch_idx": batch_idx,
            "batch_size": len(predictions),
            "correct": correct,
            "accuracy": accuracy,
            "error_patterns": error_patterns,
            "improvement_suggestions": improvement_suggestions,
            "timestamp": datetime.now().isoformat(),
        }

        self.analysis_history.append(analysis)
        return analysis

    def _generate_improvement_suggestions(
        self,
        error_patterns: Dict[str, int],
        ground_truths: List[str],
        predictions: List[str]
    ) -> List[str]:
        """根据错误模式生成改进建议"""
        suggestions = []

        if not error_patterns:
            suggestions.append("This batch achieved perfect accuracy. Maintain current approach.")
            return suggestions

        # 分析最常见的错误
        sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)

        for error_pattern, count in sorted_errors[:3]:  # 只取前3个最常见错误
            true_cat, pred_cat = error_pattern.split(" -> ")

            suggestion = (
                f"Improve detection of '{true_cat}' (misclassified as '{pred_cat}' {count} times). "
                f"Focus on distinguishing {true_cat} characteristics from {pred_cat}."
            )
            suggestions.append(suggestion)

        # 统计每个类别的表现
        category_stats = {}
        for truth in set(ground_truths):
            category_stats[truth] = {
                "total": ground_truths.count(truth),
                "correct": sum(1 for p, g in zip(predictions, ground_truths) if g == truth and p == g)
            }

        # 找出表现最差的类别
        worst_category = None
        worst_accuracy = 1.0
        for cat, stats in category_stats.items():
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"]
                if acc < worst_accuracy:
                    worst_accuracy = acc
                    worst_category = cat

        if worst_category and worst_accuracy < 0.5:
            suggestions.append(
                f"Category '{worst_category}' has low accuracy ({worst_accuracy:.2%}). "
                f"Emphasize patterns specific to this vulnerability type."
            )

        return suggestions


class PromptEvolver:
    """基于 Batch 分析反馈的 Prompt 进化器"""

    def __init__(self, llm_client, config: Dict[str, Any]):
        self.llm_client = llm_client
        self.config = config
        self.evolution_history = []

    def evolve_with_feedback(
        self,
        current_prompt: str,
        batch_analysis: Dict[str, Any],
        generation: int
    ) -> str:
        """根据 batch 分析反馈进化 prompt"""

        # 如果准确率已经很高，不需要改进
        if batch_analysis["accuracy"] >= 0.95:
            return current_prompt

        # 构建进化指令
        improvement_text = "\n".join(
            f"- {sug}" for sug in batch_analysis["improvement_suggestions"]
        )

        error_text = "\n".join(
            f"- {pattern}: {count} occurrences"
            for pattern, count in batch_analysis["error_patterns"].items()
        )

        evolution_instruction = f"""
You are improving a vulnerability detection prompt based on batch analysis feedback.

Current Prompt:
{current_prompt}

Batch Analysis Results:
- Accuracy: {batch_analysis['accuracy']:.2%}
- Batch size: {batch_analysis['batch_size']}
- Correct predictions: {batch_analysis['correct']}

Common Error Patterns:
{error_text if error_text else "None - all predictions were correct"}

Improvement Suggestions:
{improvement_text}

Task: Create an improved prompt that:
1. Addresses the identified error patterns
2. Better distinguishes between the confused categories
3. Maintains the same output format (CWE major category or 'Benign')
4. Keeps the {{{{input}}}} placeholder for code insertion
5. Uses the following valid categories: {", ".join(CWE_MAJOR_CATEGORIES)}

Return ONLY the improved prompt text, nothing else:
"""

        try:
            improved_prompt = self.llm_client.generate(
                evolution_instruction,
                temperature=0.7,
                max_tokens=500
            )

            # 验证改进后的 prompt
            if "{input}" in improved_prompt and len(improved_prompt.strip()) > 50:
                self.evolution_history.append({
                    "generation": generation,
                    "batch_idx": batch_analysis["batch_idx"],
                    "old_accuracy": batch_analysis["accuracy"],
                    "prompt": improved_prompt,
                    "timestamp": datetime.now().isoformat(),
                })
                return improved_prompt.strip()
            else:
                print(f"    ⚠️ 进化后的 prompt 无效，保持原 prompt")
                return current_prompt

        except Exception as e:
            print(f"    ❌ Prompt 进化失败: {e}")
            return current_prompt


class PrimeVulLayer1Pipeline:
    """PrimeVul Layer-1 并发漏洞分类流水线"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config.get("batch_size", 16)
        self.init_dir = Path("init")
        self.result_dir = Path("result")

        # 创建必要的目录
        self.init_dir.mkdir(exist_ok=True)
        self.result_dir.mkdir(exist_ok=True)

        # 创建实验子目录
        self.exp_id = config.get("experiment_id", f"layer1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.exp_dir = self.result_dir / self.exp_id
        self.exp_dir.mkdir(exist_ok=True)

        # 初始化组件
        self.llm_client = create_default_client()
        self.batch_analyzer = BatchAnalyzer(batch_size=self.batch_size)
        self.prompt_evolver = PromptEvolver(self.llm_client, config)

        print(f"✅ 初始化 PrimeVul Layer-1 Pipeline")
        print(f"   实验 ID: {self.exp_id}")
        print(f"   Batch 大小: {self.batch_size}")
        print(f"   结果目录: {self.exp_dir}")

    def load_initial_prompts(self) -> List[str]:
        """从 init/ 文件夹加载初始 prompts"""
        prompts_file = self.init_dir / "layer1_prompts.txt"

        if not prompts_file.exists():
            print(f"⚠️ 未找到初始 prompts 文件: {prompts_file}")
            print(f"   使用默认 prompts 并保存到 {prompts_file}")
            default_prompts = self._create_default_prompts()

            # 保存默认 prompts
            with open(prompts_file, "w", encoding="utf-8") as f:
                f.write("# PrimeVul Layer-1 初始化 Prompts\n")
                f.write("# 每个 prompt 之间用空行分隔\n")
                f.write("# Prompt 中必须包含 {input} 占位符\n\n")
                for i, prompt in enumerate(default_prompts, 1):
                    f.write(f"# Prompt {i}\n")
                    f.write(prompt)
                    f.write("\n\n" + "="*80 + "\n\n")

            return default_prompts

        # 读取 prompts
        with open(prompts_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 按分隔符分割
        prompts = []
        for section in content.split("=" * 80):
            section = section.strip()
            if not section or section.startswith("#"):
                continue
            # 移除注释行
            lines = [line for line in section.split("\n") if not line.strip().startswith("#")]
            prompt = "\n".join(lines).strip()
            if prompt and "{input}" in prompt:
                prompts.append(prompt)

        print(f"✅ 从 {prompts_file} 加载了 {len(prompts)} 个初始 prompts")
        return prompts if prompts else self._create_default_prompts()

    def _create_default_prompts(self) -> List[str]:
        """创建默认的初始 prompts"""
        categories_text = ", ".join(f"'{cat}'" for cat in CWE_MAJOR_CATEGORIES)

        return [
            f"""Analyze this code for security vulnerabilities and classify it into one of these CWE major categories: {categories_text}.
If no vulnerability is found, respond with 'Benign'.
Respond ONLY with the category name.

Code to analyze:
{{input}}

CWE Major Category:""",

            f"""You are a security expert analyzing code for vulnerabilities.
Classify the code into ONE of these categories: {categories_text}.
For secure code, respond with 'Benign'.
Output ONLY the category name, nothing else.

Code:
{{input}}

Category:""",

            f"""Security vulnerability classification task.
Categories: {categories_text}

Examine the code and identify the PRIMARY vulnerability type.
If the code is secure, respond with 'Benign'.
Response format: Category name only.

Code to analyze:
{{input}}

Result:""",
        ]

    def batch_predict(
        self,
        prompt: str,
        samples: List[Any],
        batch_idx: int
    ) -> Tuple[List[str], List[str]]:
        """批量预测一个 batch 的样本"""
        predictions = []
        ground_truths = []

        # 准备批量查询
        queries = []
        for sample in samples:
            code = sample.input_text
            query = prompt.format(input=code)
            queries.append(query)

            # 获取 ground truth
            ground_truth_binary = int(sample.target)
            cwe_codes = sample.metadata.get("cwe", [])

            if ground_truth_binary == 1 and cwe_codes:
                ground_truth_category = map_cwe_to_major(cwe_codes)
            else:
                ground_truth_category = "Benign"

            ground_truths.append(ground_truth_category)

        # 批量调用 LLM
        print(f"      🔍 批量预测 {len(queries)} 个样本...")
        try:
            responses = self.llm_client.batch_generate(
                queries,
                temperature=0.1,
                max_tokens=20,
                batch_size=min(8, len(queries)),
                concurrent=True
            )

            # 规范化输出
            for response in responses:
                if response == "error":
                    predictions.append("Other")
                else:
                    predicted_category = canonicalize_category(response)
                    if predicted_category is None:
                        # 尝试从响应中提取
                        if "benign" in response.lower():
                            predicted_category = "Benign"
                        else:
                            predicted_category = "Other"
                    predictions.append(predicted_category)

        except Exception as e:
            print(f"      ❌ 批量预测失败: {e}")
            predictions = ["Other"] * len(samples)

        return predictions, ground_truths

    def evaluate_prompt_on_dataset(
        self,
        prompt: str,
        dataset,
        generation: int,
        prompt_id: str,
        enable_evolution: bool = False
    ) -> Dict[str, Any]:
        """在完整数据集上评估 prompt，使用 batch 处理"""
        samples = dataset.get_samples()
        total_samples = len(samples)

        all_predictions = []
        all_ground_truths = []
        batch_analyses = []

        current_prompt = prompt
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size

        print(f"    📊 评估 prompt (共 {num_batches} 个 batches, {total_samples} 个样本)")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_samples)
            batch_samples = samples[start_idx:end_idx]

            print(f"      Batch {batch_idx + 1}/{num_batches} (样本 {start_idx+1}-{end_idx})")

            # 批量预测
            predictions, ground_truths = self.batch_predict(
                current_prompt, batch_samples, batch_idx
            )

            # 分析 batch 结果
            batch_analysis = self.batch_analyzer.analyze_batch(
                predictions, ground_truths, batch_idx
            )

            print(f"        ✓ 准确率: {batch_analysis['accuracy']:.2%} ({batch_analysis['correct']}/{batch_analysis['batch_size']})")

            if batch_analysis["error_patterns"]:
                print(f"        ⚠️ 错误模式: {len(batch_analysis['error_patterns'])} 种")

            batch_analyses.append(batch_analysis)
            all_predictions.extend(predictions)
            all_ground_truths.extend(ground_truths)

            # 根据 batch 分析进化 prompt (仅在训练模式下)
            if enable_evolution and batch_analysis["accuracy"] < 0.95:
                print(f"        🧬 尝试进化 prompt...")
                new_prompt = self.prompt_evolver.evolve_with_feedback(
                    current_prompt, batch_analysis, generation
                )
                if new_prompt != current_prompt:
                    print(f"        ✅ Prompt 已进化")
                    current_prompt = new_prompt

        # 计算整体指标
        overall_accuracy = sum(p == g for p, g in zip(all_predictions, all_ground_truths)) / len(all_predictions)

        # 生成分类报告
        report = classification_report(
            all_ground_truths,
            all_predictions,
            labels=CWE_MAJOR_CATEGORIES,
            output_dict=True,
            zero_division=0
        )

        # 混淆矩阵
        cm = confusion_matrix(
            all_ground_truths,
            all_predictions,
            labels=CWE_MAJOR_CATEGORIES
        )

        return {
            "prompt_id": prompt_id,
            "generation": generation,
            "final_prompt": current_prompt,
            "accuracy": overall_accuracy,
            "total_samples": total_samples,
            "num_batches": num_batches,
            "batch_analyses": batch_analyses,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "predictions": all_predictions,
            "ground_truths": all_ground_truths,
        }

    def run_evolution(self) -> Dict[str, Any]:
        """运行完整的进化流程"""
        print("\n" + "="*80)
        print("🚀 开始 PrimeVul Layer-1 并发漏洞分类")
        print("="*80 + "\n")

        # 1. 准备数据
        print("📁 准备数据集...")
        primevul_dir = Path(self.config.get("primevul_dir", "./data/primevul/primevul"))
        sample_dir = Path(self.config.get("sample_dir", "./data/primevul_1percent_sample"))

        if not sample_dir.exists():
            print(f"   生成 1% 采样数据到 {sample_dir}")
            sample_primevul_1percent(str(primevul_dir), str(sample_dir), seed=42)

        train_file = sample_dir / "train.txt"
        dev_file = sample_dir / "dev.txt"

        train_dataset = PrimevulDataset(str(train_file), "train")
        dev_dataset = PrimevulDataset(str(dev_file), "dev")

        print(f"   ✅ 训练集: {len(train_dataset)} 样本")
        print(f"   ✅ 开发集: {len(dev_dataset)} 样本")

        # 2. 加载初始 prompts
        print("\n📝 加载初始 prompts...")
        initial_prompts = self.load_initial_prompts()

        # 3. 初始评估
        print(f"\n📊 初始评估 ({len(initial_prompts)} 个 prompts)...")
        population = []

        for i, prompt in enumerate(initial_prompts):
            print(f"\n  Prompt {i+1}/{len(initial_prompts)}")
            result = self.evaluate_prompt_on_dataset(
                prompt, dev_dataset, generation=0,
                prompt_id=f"initial_{i}", enable_evolution=False
            )
            individual = Individual(prompt)
            individual.fitness = result["accuracy"]
            population.append((individual, result))
            print(f"    ✓ 适应度: {individual.fitness:.4f}")

        # 4. 进化过程
        max_generations = self.config.get("max_generations", 5)
        print(f"\n🧬 开始进化 (共 {max_generations} 代)...")

        best_results = []

        for generation in range(1, max_generations + 1):
            print(f"\n{'='*80}")
            print(f"📈 第 {generation} 代进化")
            print(f"{'='*80}\n")

            # 选择最佳个体
            population.sort(key=lambda x: x[0].fitness, reverse=True)
            best_individual, best_result = population[0]
            best_results.append(best_result)

            print(f"  当前最佳适应度: {best_individual.fitness:.4f}")

            # 在训练集上进化最佳 prompt
            print(f"\n  在训练集上进化最佳 prompt...")
            evolved_result = self.evaluate_prompt_on_dataset(
                best_individual.prompt,
                train_dataset,
                generation=generation,
                prompt_id=f"gen{generation}_best",
                enable_evolution=True
            )

            # 创建进化后的个体并在开发集上评估
            evolved_prompt = evolved_result["final_prompt"]
            if evolved_prompt != best_individual.prompt:
                print(f"\n  在开发集上评估进化后的 prompt...")
                eval_result = self.evaluate_prompt_on_dataset(
                    evolved_prompt,
                    dev_dataset,
                    generation=generation,
                    prompt_id=f"gen{generation}_evolved",
                    enable_evolution=False
                )

                evolved_individual = Individual(evolved_prompt)
                evolved_individual.fitness = eval_result["accuracy"]

                print(f"    进化前适应度: {best_individual.fitness:.4f}")
                print(f"    进化后适应度: {evolved_individual.fitness:.4f}")

                if evolved_individual.fitness > best_individual.fitness:
                    print(f"    ✅ 接受进化后的 prompt!")
                    population[0] = (evolved_individual, eval_result)
                else:
                    print(f"    ❌ 保留原 prompt")

        # 5. 最终结果
        population.sort(key=lambda x: x[0].fitness, reverse=True)
        best_individual, best_result = population[0]

        print(f"\n{'='*80}")
        print(f"🎉 进化完成!")
        print(f"{'='*80}\n")
        print(f"  最终适应度: {best_individual.fitness:.4f}")

        # 6. 保存结果
        self.save_results(best_individual, best_result, best_results)

        return {
            "best_prompt": best_individual.prompt,
            "best_fitness": best_individual.fitness,
            "best_result": best_result,
            "evolution_history": best_results,
        }

    def save_results(
        self,
        best_individual: Individual,
        best_result: Dict[str, Any],
        evolution_history: List[Dict[str, Any]]
    ):
        """保存结果到 result/ 文件夹"""
        print(f"\n💾 保存结果到 {self.exp_dir}...")

        # 1. 保存最终 prompt
        prompt_file = self.exp_dir / "final_prompt.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(f"# 最终优化的 Prompt (适应度: {best_individual.fitness:.4f})\n")
            f.write(f"# 实验 ID: {self.exp_id}\n")
            f.write(f"# 生成时间: {datetime.now().isoformat()}\n\n")
            f.write(best_individual.prompt)
        print(f"  ✓ {prompt_file}")

        # 2. 保存分类报告 (precision, recall, f1-score)
        report = best_result["classification_report"]
        metrics_file = self.exp_dir / "classification_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  ✓ {metrics_file}")

        # 3. 保存分类报告的易读版本
        readable_report_file = self.exp_dir / "classification_report.txt"
        with open(readable_report_file, "w", encoding="utf-8") as f:
            f.write(f"PrimeVul Layer-1 分类报告\n")
            f.write(f"{'='*80}\n")
            f.write(f"实验 ID: {self.exp_id}\n")
            f.write(f"最终准确率: {best_individual.fitness:.4f}\n")
            f.write(f"总样本数: {best_result['total_samples']}\n")
            f.write(f"Batch 大小: {self.batch_size}\n")
            f.write(f"Batch 总数: {best_result['num_batches']}\n\n")

            f.write(f"各类别性能指标:\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{'Category':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n")
            f.write(f"{'-'*80}\n")

            for category in CWE_MAJOR_CATEGORIES:
                if category in report:
                    metrics = report[category]
                    f.write(f"{category:<25} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                           f"{metrics['f1-score']:>10.4f} {metrics['support']:>10}\n")

            f.write(f"{'-'*80}\n")
            f.write(f"{'Overall (macro avg)':<25} {report['macro avg']['precision']:>10.4f} "
                   f"{report['macro avg']['recall']:>10.4f} {report['macro avg']['f1-score']:>10.4f} "
                   f"{report['macro avg']['support']:>10}\n")
            f.write(f"{'Overall (weighted avg)':<25} {report['weighted avg']['precision']:>10.4f} "
                   f"{report['weighted avg']['recall']:>10.4f} {report['weighted avg']['f1-score']:>10.4f} "
                   f"{report['weighted avg']['support']:>10}\n")

        print(f"  ✓ {readable_report_file}")

        # 4. 保存混淆矩阵
        confusion_file = self.exp_dir / "confusion_matrix.json"
        with open(confusion_file, "w", encoding="utf-8") as f:
            json.dump({
                "labels": CWE_MAJOR_CATEGORIES,
                "matrix": best_result["confusion_matrix"]
            }, f, indent=2, ensure_ascii=False)
        print(f"  ✓ {confusion_file}")

        # 5. 保存 batch 分析历史
        batch_history_file = self.exp_dir / "batch_analyses.jsonl"
        with open(batch_history_file, "w", encoding="utf-8") as f:
            for analysis in best_result["batch_analyses"]:
                f.write(json.dumps(analysis, ensure_ascii=False) + "\n")
        print(f"  ✓ {batch_history_file}")

        # 6. 保存完整的实验配置和结果
        summary_file = self.exp_dir / "experiment_summary.json"
        summary = {
            "experiment_id": self.exp_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "best_fitness": best_individual.fitness,
            "best_prompt": best_individual.prompt,
            "total_samples": best_result["total_samples"],
            "num_batches": best_result["num_batches"],
            "batch_size": self.batch_size,
            "classification_report": report,
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  ✓ {summary_file}")

        # 7. 打印最终报告到控制台
        print(f"\n📊 最终分类性能:")
        print(f"{'-'*80}")
        print(f"{'Category':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print(f"{'-'*80}")

        for category in CWE_MAJOR_CATEGORIES:
            if category in report:
                metrics = report[category]
                print(f"{category:<25} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                     f"{metrics['f1-score']:>10.4f} {metrics['support']:>10.0f}")

        print(f"{'-'*80}")
        print(f"{'Overall (macro avg)':<25} {report['macro avg']['precision']:>10.4f} "
              f"{report['macro avg']['recall']:>10.4f} {report['macro avg']['f1-score']:>10.4f}")


def main():
    """主入口函数"""
    import argparse

    parser = argparse.ArgumentParser(description="EvoPrompt Main - PrimeVul Layer-1 并发漏洞分类")
    parser.add_argument("--batch-size", type=int, default=16, help="每个 batch 的样本数")
    parser.add_argument("--max-generations", type=int, default=5, help="最大进化代数")
    parser.add_argument("--primevul-dir", type=str, default="./data/primevul/primevul",
                       help="PrimeVul 数据集目录")
    parser.add_argument("--sample-dir", type=str, default="./data/primevul_1percent_sample",
                       help="采样数据目录")
    parser.add_argument("--experiment-id", type=str, default=None,
                       help="实验 ID (默认自动生成)")

    args = parser.parse_args()

    # 创建配置
    config = {
        "batch_size": args.batch_size,
        "max_generations": args.max_generations,
        "primevul_dir": args.primevul_dir,
        "sample_dir": args.sample_dir,
        "experiment_id": args.experiment_id,
    }

    # 创建并运行 pipeline
    pipeline = PrimeVulLayer1Pipeline(config)
    results = pipeline.run_evolution()

    print(f"\n✅ 实验完成!")
    print(f"📂 结果已保存到: {pipeline.exp_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
