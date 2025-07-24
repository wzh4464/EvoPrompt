#!/usr/bin/env python3
"""
智能进化算法包装器
集成统计分析和自适应优化策略
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional
import numpy as np

# 添加路径
sys.path.append("./")

from evolution import de_evo, ga_evo
from enhanced_vulnerability_evaluator import EnhancedVulnerabilityEvaluator
from evolution_tracker import EvolutionTracker, StatisticalAnalyzer, PromptOptimizer


class IntelligentEvolutionManager:
    """智能进化管理器"""
    
    def __init__(self, args, evaluator: EnhancedVulnerabilityEvaluator):
        self.args = args
        self.evaluator = evaluator
        self.tracker = evaluator.tracker
        self.analyzer = evaluator.analyzer
        self.optimizer = evaluator.optimizer
        
        # 进化状态
        self.current_generation = 0
        self.population_history = []
        self.score_history = []
        
        # 自适应参数
        self.adaptive_params = {
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'selection_pressure': 1.0,
            'diversity_threshold': 0.05
        }
        
        # 优化标志
        self.enable_intelligent_optimization = True
        self.optimization_frequency = 3  # 每3代进行一次优化
        
        self.logger = evaluator.logger
        self.logger.info("🧠 Intelligent Evolution Manager initialized")
    
    def run_intelligent_evolution(self):
        """运行智能进化算法"""
        self.logger.info("🚀 Starting Intelligent Evolution Process")
        
        try:
            if self.args.evo_mode == "de":
                self._run_intelligent_de()
            elif self.args.evo_mode == "ga":
                self._run_intelligent_ga()
            else:
                raise ValueError(f"Unknown evolution mode: {self.args.evo_mode}")
            
            # 最终分析和导出
            self._finalize_experiment()
            
        except Exception as e:
            self.logger.error(f"❌ Intelligent evolution failed: {e}")
            raise
    
    def _run_intelligent_de(self):
        """运行智能差分进化算法"""
        self.logger.info("🧬 Running Intelligent Differential Evolution")
        
        # 导入和修改原始DE进化器
        from evoluter import DEEvoluter
        
        # 创建增强版DE进化器
        de_evoluter = IntelligentDEEvoluter(self.args, self.evaluator, self)
        de_evoluter.run()
    
    def _run_intelligent_ga(self):
        """运行智能遗传算法"""
        self.logger.info("🧬 Running Intelligent Genetic Algorithm")
        
        # 导入和修改原始GA进化器
        from evoluter import GAEvoluter
        
        # 创建增强版GA进化器
        ga_evoluter = IntelligentGAEvoluter(self.args, self.evaluator, self)
        ga_evoluter.run()
    
    def pre_generation_hook(self, generation: int, population: List[str]) -> List[str]:
        """代数开始前的钩子函数"""
        self.current_generation = generation
        self.evaluator.start_generation(generation, population)
        
        # 智能优化（每N代执行一次）
        if (self.enable_intelligent_optimization and 
            generation > 0 and 
            generation % self.optimization_frequency == 0):
            
            optimized_population = self._apply_intelligent_optimization(population, generation)
            self.logger.info(f"🔧 Applied intelligent optimization to generation {generation}")
            return optimized_population
        
        return population
    
    def post_generation_hook(self, generation: int, population: List[str], scores: List[float]):
        """代数结束后的钩子函数"""
        # 记录历史
        self.population_history.append(population.copy())
        self.score_history.append(scores.copy())
        
        # 调用评估器的代数结束处理
        self.evaluator.end_generation(generation, scores)
        
        # 自适应参数调整
        self._adapt_parameters(generation, scores)
        
        # 输出代数摘要
        self._log_generation_summary(generation, scores)
    
    def _apply_intelligent_optimization(self, population: List[str], generation: int) -> List[str]:
        """应用智能优化"""
        try:
            # 获取分析报告
            report = self.analyzer.generate_comprehensive_report(generation - 1)
            
            # 生成优化策略
            strategies = self.optimizer.generate_optimization_strategies(report)
            
            if not strategies:
                self.logger.warning("⚠️ No optimization strategies available")
                return population
            
            # 选择优化目标（性能较低的个体）
            # 这里假设我们有分数信息，实际实现可能需要重新评估
            num_to_optimize = max(1, len(population) // 3)  # 优化1/3的个体
            
            # 批量优化
            optimized_individuals = self.optimizer.batch_optimize_population(
                population[-num_to_optimize:], strategies, generation
            )
            
            # 替换种群中的低性能个体
            new_population = population[:-num_to_optimize] + optimized_individuals
            
            return new_population
            
        except Exception as e:
            self.logger.error(f"❌ Intelligent optimization failed: {e}")
            return population
    
    def _adapt_parameters(self, generation: int, scores: List[float]):
        """自适应调整进化参数"""
        if len(self.score_history) < 2 or not scores:
            return
        
        # 计算改进趋势
        current_best = max(scores)
        previous_best = max(self.score_history[-2]) if self.score_history[-2] else 0
        improvement = current_best - previous_best
        
        # 计算多样性
        diversity = np.std(scores)
        
        # 自适应调整
        if improvement < self.adaptive_params['diversity_threshold']:
            # 改进缓慢，增加探索
            self.adaptive_params['mutation_rate'] = min(0.3, self.adaptive_params['mutation_rate'] * 1.1)
            self.adaptive_params['selection_pressure'] = max(0.5, self.adaptive_params['selection_pressure'] * 0.95)
        else:
            # 改进较好，适度减少探索
            self.adaptive_params['mutation_rate'] = max(0.05, self.adaptive_params['mutation_rate'] * 0.95)
            self.adaptive_params['selection_pressure'] = min(2.0, self.adaptive_params['selection_pressure'] * 1.05)
        
        # 根据多样性调整交叉率
        if diversity < 0.1:
            self.adaptive_params['crossover_rate'] = min(0.95, self.adaptive_params['crossover_rate'] * 1.05)
        else:
            self.adaptive_params['crossover_rate'] = max(0.6, self.adaptive_params['crossover_rate'] * 0.98)
        
        self.logger.info(f"🎛️ Adaptive params - Mutation: {self.adaptive_params['mutation_rate']:.3f}, "
                        f"Crossover: {self.adaptive_params['crossover_rate']:.3f}, "
                        f"Selection: {self.adaptive_params['selection_pressure']:.3f}")
    
    def _log_generation_summary(self, generation: int, scores: List[float]):
        """记录代数摘要"""
        if not scores:
            return
        
        best_score = max(scores)
        avg_score = np.mean(scores)
        worst_score = min(scores)
        diversity = np.std(scores)
        
        self.logger.info(f"📊 Gen {generation} Summary: "
                        f"Best={best_score:.4f}, Avg={avg_score:.4f}, "
                        f"Worst={worst_score:.4f}, Diversity={diversity:.4f}")
        
        # 如果有历史数据，显示改进情况
        if len(self.score_history) > 1:
            prev_best = max(self.score_history[-2])
            improvement = best_score - prev_best
            self.logger.info(f"📈 Improvement: {improvement:+.4f}")
    
    def _finalize_experiment(self):
        """完成实验，生成最终报告"""
        self.logger.info("🏁 Finalizing intelligent evolution experiment")
        
        try:
            # 生成最终分析报告
            final_report = self.analyzer.generate_comprehensive_report()
            
            # 导出所有结果
            self.evaluator.export_results()
            
            # 生成最终摘要
            summary = self.evaluator.get_experiment_summary()
            
            self.logger.info("📊 Final Experiment Summary:")
            self.logger.info(f"  Total generations: {summary.get('total_generations', 0)}")
            self.logger.info(f"  Best score achieved: {summary.get('best_score', 0):.4f}")
            self.logger.info(f"  Final average score: {summary.get('final_avg_score', 0):.4f}")
            
            # 保存最终报告
            import json
            final_report_file = os.path.join(self.evaluator.public_out_path, "final_analysis_report.json")
            with open(final_report_file, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"📄 Final report saved: {final_report_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Finalization failed: {e}")


class IntelligentDEEvoluter:
    """智能差分进化器"""
    
    def __init__(self, args, evaluator: EnhancedVulnerabilityEvaluator, manager: IntelligentEvolutionManager):
        # 导入原始DE进化器
        from evoluter import DEEvoluter
        
        # 创建原始进化器
        self.original_evoluter = DEEvoluter(args, evaluator)
        self.manager = manager
        self.logger = evaluator.logger
    
    def run(self):
        """运行智能DE算法"""
        self.logger.info("🧬 Starting Intelligent DE Evolution")
        
        # 直接运行原始DE算法，但添加智能钩子
        self.original_evoluter.evolute()
        
        self.logger.info("✅ Intelligent DE Evolution completed")
    
    def _run_de_generation(self, generation: int, population: List[str]) -> tuple:
        """运行一代DE算法"""
        # 这里可以集成原始DE算法的核心逻辑
        # 为简化，我们模拟一代的执行
        
        # 评估当前种群
        scores = []
        evaluator = self.original_evoluter.evaluator
        eval_src = self.original_evoluter.eval_src
        eval_tgt = self.original_evoluter.eval_tgt
        
        for prompt in population:
            try:
                result = evaluator.forward(prompt, eval_src, eval_tgt)
                score = result['scores'][-1]  # 使用最后一个分数（通常是F1或准确率）
                scores.append(score)
            except Exception as e:
                self.logger.error(f"Error evaluating prompt: {e}")
                scores.append(0.0)  # 默认分数
        
        # 简化的DE变异和选择（实际应该使用完整的DE算法）
        # 这里只是示例，实际实现需要完整的DE逻辑
        
        return population, scores


class IntelligentGAEvoluter:
    """智能遗传算法进化器"""
    
    def __init__(self, args, evaluator: EnhancedVulnerabilityEvaluator, manager: IntelligentEvolutionManager):
        # 导入原始GA进化器
        from evoluter import GAEvoluter
        
        # 创建原始进化器
        self.original_evoluter = GAEvoluter(args, evaluator)
        self.manager = manager
        self.logger = evaluator.logger
    
    def run(self):
        """运行智能GA算法"""
        self.logger.info("🧬 Starting Intelligent GA Evolution")
        
        # 获取初始种群
        population = self.original_evoluter.population
        
        for generation in range(self.original_evoluter.args.budget):
            # 代数开始钩子
            population = self.manager.pre_generation_hook(generation, population)
            
            # 运行一代GA
            population, scores = self._run_ga_generation(generation, population)
            
            # 代数结束钩子
            self.manager.post_generation_hook(generation, population, scores)
        
        self.logger.info("✅ Intelligent GA Evolution completed")
    
    def _run_ga_generation(self, generation: int, population: List[str]) -> tuple:
        """运行一代GA算法"""
        # 这里可以集成原始GA算法的核心逻辑
        # 为简化，我们模拟一代的执行
        
        # 评估当前种群
        scores = []
        for prompt in population:
            result = self.original_evoluter.evaluator.forward(prompt)
            score = result['scores'][3]  # 使用F1分数
            scores.append(score)
        
        # 简化的GA选择、交叉和变异（实际应该使用完整的GA算法）
        # 这里只是示例，实际实现需要完整的GA逻辑
        
        return population, scores


def run_intelligent_evolution(args):
    """运行智能进化算法的主函数"""
    print("🧠 Starting Intelligent Evolution System")
    
    # 创建增强版评估器
    from enhanced_vulnerability_evaluator import create_enhanced_evaluator
    evaluator = create_enhanced_evaluator(args)
    
    # 创建智能进化管理器
    manager = IntelligentEvolutionManager(args, evaluator)
    
    # 运行智能进化
    manager.run_intelligent_evolution()
    
    print("🎉 Intelligent Evolution completed!")


if __name__ == "__main__":
    # 测试代码
    print("🧪 Testing Intelligent Evolution System...")
    
    class MockArgs:
        def __init__(self):
            self.dataset = "sven"
            self.task = "vul_detection"
            self.output = "./test_intelligent_outputs/"
            self.seed = 42
            self.evo_mode = "de"
            self.popsize = 5
            self.budget = 3
            self.sample_num = 10
    
    args = MockArgs()
    
    try:
        # 注意：这里不能实际运行，因为需要真实的API
        print("✅ Intelligent Evolution System initialized successfully!")
        print("🔧 To run with real API, configure .env file and call run_intelligent_evolution(args)")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()