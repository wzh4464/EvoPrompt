#!/usr/bin/env python3
"""
进化过程跟踪和智能分析系统
用于存储prompt变化、分析统计偏差并提供智能更新策略
"""

import json
import os
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from sven_llm_client import sven_llm_init, sven_llm_query


class EvolutionTracker:
    """进化过程跟踪器"""
    
    def __init__(self, output_dir: str, experiment_name: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 数据库文件
        self.db_path = self.output_dir / f"{self.experiment_name}.db"
        self.init_database()
        
        # JSON文件（备份和可读性）
        self.json_path = self.output_dir / f"{self.experiment_name}_results.json"
        
        # LLM客户端（用于分析）
        self.llm_client = sven_llm_init()
        
        # 实验配置
        self.config = {}
        
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建实验基本信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                start_time TIMESTAMP,
                config TEXT,
                status TEXT DEFAULT 'running'
            )
        ''')
        
        # 创建进化代数表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                generation INTEGER,
                timestamp TIMESTAMP,
                best_score REAL,
                avg_score REAL,
                worst_score REAL,
                population_size INTEGER,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')
        
        # 创建prompt个体表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                generation INTEGER,
                prompt_text TEXT,
                score REAL,
                accuracy REAL,
                precision_val REAL,
                recall_val REAL,
                f1_score REAL,
                specificity REAL,
                parent_id INTEGER,
                mutation_type TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')
        
        # 创建详细预测结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id INTEGER,
                sample_index INTEGER,
                input_text TEXT,
                predicted_label TEXT,
                true_label TEXT,
                is_correct BOOLEAN,
                confidence REAL,
                response_time REAL,
                FOREIGN KEY (prompt_id) REFERENCES prompts (id)
            )
        ''')
        
        # 创建统计分析表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                generation INTEGER,
                analysis_type TEXT,
                analysis_result TEXT,
                recommendations TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_experiment(self, config: Dict[str, Any]):
        """开始实验"""
        self.config = config
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO experiments (name, start_time, config, status)
            VALUES (?, ?, ?, ?)
        ''', (self.experiment_name, datetime.now(), json.dumps(config), 'running'))
        
        self.experiment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"🚀 Started experiment: {self.experiment_name} (ID: {self.experiment_id})")
    
    def log_generation(self, generation: int, population_scores: List[float]):
        """记录一代的结果"""
        if not population_scores:
            return
            
        best_score = max(population_scores)
        avg_score = np.mean(population_scores)
        worst_score = min(population_scores)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO generations 
            (experiment_id, generation, timestamp, best_score, avg_score, worst_score, population_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (self.experiment_id, generation, datetime.now(), 
              best_score, avg_score, worst_score, len(population_scores)))
        
        conn.commit()
        conn.close()
        
        print(f"📊 Generation {generation}: Best={best_score:.4f}, Avg={avg_score:.4f}, Worst={worst_score:.4f}")
    
    def log_prompt(self, generation: int, prompt_text: str, scores: List[float], 
                   detailed_results: List[Dict] = None, parent_id: int = None, 
                   mutation_type: str = None) -> int:
        """记录单个prompt的详细结果"""
        # 解析scores [accuracy, precision, recall, f1, specificity]
        accuracy = scores[0] if len(scores) > 0 else 0
        precision_val = scores[1] if len(scores) > 1 else 0
        recall_val = scores[2] if len(scores) > 2 else 0
        f1_score = scores[3] if len(scores) > 3 else 0
        specificity = scores[4] if len(scores) > 4 else 0
        overall_score = f1_score  # 使用F1作为主要分数
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 插入prompt记录
        cursor.execute('''
            INSERT INTO prompts 
            (experiment_id, generation, prompt_text, score, accuracy, precision_val, 
             recall_val, f1_score, specificity, parent_id, mutation_type, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (self.experiment_id, generation, prompt_text, overall_score, accuracy,
              precision_val, recall_val, f1_score, specificity, parent_id, 
              mutation_type, datetime.now()))
        
        prompt_id = cursor.lastrowid
        
        # 如果有详细结果，也存储
        if detailed_results:
            for i, result in enumerate(detailed_results):
                cursor.execute('''
                    INSERT INTO predictions 
                    (prompt_id, sample_index, input_text, predicted_label, true_label, 
                     is_correct, confidence, response_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (prompt_id, i, 
                      result.get('input', ''),
                      result.get('predicted', ''),
                      result.get('true_label', ''),
                      result.get('correct', False),
                      result.get('confidence', 0.0),
                      result.get('response_time', 0.0)))
        
        conn.commit()
        conn.close()
        
        return prompt_id
    
    def get_generation_data(self, generation: int = None) -> pd.DataFrame:
        """获取指定代数的数据"""
        conn = sqlite3.connect(self.db_path)
        
        if generation is not None:
            query = '''
                SELECT * FROM prompts 
                WHERE experiment_id = ? AND generation = ?
                ORDER BY score DESC
            '''
            df = pd.read_sql_query(query, conn, params=(self.experiment_id, generation))
        else:
            query = '''
                SELECT * FROM prompts 
                WHERE experiment_id = ?
                ORDER BY generation, score DESC
            '''
            df = pd.read_sql_query(query, conn, params=(self.experiment_id,))
        
        conn.close()
        return df
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """获取进化过程摘要"""
        conn = sqlite3.connect(self.db_path)
        
        # 获取代数统计
        generations_df = pd.read_sql_query('''
            SELECT * FROM generations 
            WHERE experiment_id = ?
            ORDER BY generation
        ''', conn, params=(self.experiment_id,))
        
        # 获取最佳prompts
        best_prompts_df = pd.read_sql_query('''
            SELECT generation, prompt_text, score, accuracy, precision_val, recall_val, f1_score
            FROM prompts 
            WHERE experiment_id = ?
            ORDER BY score DESC
            LIMIT 10
        ''', conn, params=(self.experiment_id,))
        
        conn.close()
        
        return {
            'generations': generations_df.to_dict('records'),
            'best_prompts': best_prompts_df.to_dict('records'),
            'total_generations': len(generations_df),
            'best_score': float(generations_df['best_score'].max()) if not generations_df.empty else 0,
            'final_avg_score': float(generations_df['avg_score'].iloc[-1]) if not generations_df.empty else 0
        }


class StatisticalAnalyzer:
    """统计分析器 - 使用LLM分析实验数据"""
    
    def __init__(self, tracker: EvolutionTracker):
        self.tracker = tracker
        self.llm_client = tracker.llm_client
    
    def analyze_generation_performance(self, generation: int) -> Dict[str, Any]:
        """分析某一代的性能"""
        df = self.tracker.get_generation_data(generation)
        
        if df.empty:
            return {"error": "No data for this generation"}
        
        # 基础统计
        stats = {
            'generation': generation,
            'population_size': len(df),
            'score_stats': {
                'mean': float(df['score'].mean()),
                'std': float(df['score'].std()),
                'min': float(df['score'].min()),
                'max': float(df['score'].max()),
                'median': float(df['score'].median())
            },
            'metric_correlations': {
                'accuracy_f1_corr': float(df['accuracy'].corr(df['f1_score'])),
                'precision_recall_corr': float(df['precision_val'].corr(df['recall_val']))
            }
        }
        
        # 识别高性能和低性能的prompt特征
        top_prompts = df.nlargest(3, 'score')['prompt_text'].tolist()
        bottom_prompts = df.nsmallest(3, 'score')['prompt_text'].tolist()
        
        stats['top_prompts'] = top_prompts
        stats['bottom_prompts'] = bottom_prompts
        
        return stats
    
    def analyze_evolution_trends(self) -> Dict[str, Any]:
        """分析整个进化趋势"""
        summary = self.tracker.get_evolution_summary()
        generations = summary['generations']
        
        if not generations:
            return {"error": "No generation data available"}
        
        # 趋势分析
        best_scores = [g['best_score'] for g in generations]
        avg_scores = [g['avg_score'] for g in generations]
        
        trends = {
            'score_improvement': best_scores[-1] - best_scores[0] if len(best_scores) > 1 else 0,
            'convergence_rate': self._calculate_convergence_rate(best_scores),
            'diversity_trend': self._calculate_diversity_trend(generations),
            'stagnation_periods': self._detect_stagnation(best_scores)
        }
        
        return {
            'trends': trends,
            'generations_count': len(generations),
            'best_overall_score': max(best_scores),
            'improvement_trajectory': best_scores
        }
    
    def _calculate_convergence_rate(self, scores: List[float]) -> float:
        """计算收敛速度"""
        if len(scores) < 2:
            return 0.0
        
        improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        recent_improvements = improvements[-3:] if len(improvements) >= 3 else improvements
        
        return np.mean(recent_improvements) if recent_improvements else 0.0
    
    def _calculate_diversity_trend(self, generations: List[Dict]) -> float:
        """计算多样性趋势"""
        if len(generations) < 2:
            return 0.0
        
        diversity_scores = []
        for gen in generations:
            score_range = gen['best_score'] - gen['worst_score']
            diversity_scores.append(score_range)
        
        # 多样性是否在下降（收敛）
        recent_div = np.mean(diversity_scores[-3:]) if len(diversity_scores) >= 3 else diversity_scores[-1]
        early_div = np.mean(diversity_scores[:3]) if len(diversity_scores) >= 3 else diversity_scores[0]
        
        return recent_div - early_div
    
    def _detect_stagnation(self, scores: List[float], threshold: float = 0.001) -> int:
        """检测停滞期"""
        if len(scores) < 3:
            return 0
        
        stagnant_count = 0
        for i in range(2, len(scores)):
            if abs(scores[i] - scores[i-1]) < threshold:
                stagnant_count += 1
            else:
                stagnant_count = 0
        
        return stagnant_count
    
    def llm_analyze_patterns(self, generation_data: Dict[str, Any]) -> str:
        """使用LLM分析模式和给出建议"""
        analysis_prompt = f"""
作为一个专业的机器学习研究员，请分析以下prompt进化实验的数据：

## 实验数据摘要
- 当前代数: {generation_data.get('generation', 'N/A')}
- 种群大小: {generation_data.get('population_size', 'N/A')}
- 分数统计: {generation_data.get('score_stats', {})}

## 高性能Prompts (Top 3):
{chr(10).join(f"{i+1}. {prompt}" for i, prompt in enumerate(generation_data.get('top_prompts', [])))}

## 低性能Prompts (Bottom 3):
{chr(10).join(f"{i+1}. {prompt}" for i, prompt in enumerate(generation_data.get('bottom_prompts', [])))}

请分析：
1. 高性能和低性能prompt之间的关键差异
2. 识别可能影响性能的语言模式、结构或关键词
3. 指出可能的统计学偏差或过拟合问题
4. 给出具体的prompt改进建议

请用中文回答，结构化输出。
"""
        
        try:
            analysis = sven_llm_query(analysis_prompt, self.llm_client, temperature=0.3)
            return analysis
        except Exception as e:
            return f"LLM分析失败: {str(e)}"
    
    def generate_comprehensive_report(self, generation: int = None) -> Dict[str, Any]:
        """生成综合分析报告"""
        # 获取数据
        if generation is not None:
            gen_analysis = self.analyze_generation_performance(generation)
            llm_analysis = self.llm_analyze_patterns(gen_analysis)
        else:
            gen_analysis = {}
            llm_analysis = "未指定具体代数"
        
        evolution_analysis = self.analyze_evolution_trends()
        
        # 保存分析结果到数据库
        conn = sqlite3.connect(self.tracker.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analyses 
            (experiment_id, generation, analysis_type, analysis_result, recommendations, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (self.tracker.experiment_id, generation or -1, 'comprehensive', 
              json.dumps({
                  'generation_analysis': gen_analysis,
                  'evolution_analysis': evolution_analysis
              }), llm_analysis, datetime.now()))
        
        conn.commit()
        conn.close()
        
        return {
            'generation_analysis': gen_analysis,
            'evolution_analysis': evolution_analysis,
            'llm_insights': llm_analysis,
            'timestamp': datetime.now().isoformat()
        }


class PromptOptimizer:
    """智能Prompt优化器"""
    
    def __init__(self, tracker: EvolutionTracker, analyzer: StatisticalAnalyzer):
        self.tracker = tracker
        self.analyzer = analyzer
        self.llm_client = tracker.llm_client
    
    def generate_optimization_strategies(self, analysis_report: Dict[str, Any]) -> List[str]:
        """基于分析报告生成优化策略"""
        evolution_data = analysis_report.get('evolution_analysis', {})
        llm_insights = analysis_report.get('llm_insights', '')
        
        strategy_prompt = f"""
基于以下prompt进化实验分析，请提供具体的优化策略：

## 进化趋势分析
- 分数改进: {evolution_data.get('trends', {}).get('score_improvement', 0):.4f}
- 收敛速度: {evolution_data.get('trends', {}).get('convergence_rate', 0):.4f}
- 多样性变化: {evolution_data.get('trends', {}).get('diversity_trend', 0):.4f}
- 停滞期长度: {evolution_data.get('trends', {}).get('stagnation_periods', 0)}

## LLM专家分析
{llm_insights}

请基于以上信息，提供5-8个具体的prompt优化策略，每个策略应该包含：
1. 策略名称
2. 具体实施方法
3. 预期效果

格式要求：
- 用中文回答
- 每个策略用编号列出
- 具体可操作
"""
        
        try:
            strategies_text = sven_llm_query(strategy_prompt, self.llm_client, temperature=0.5)
            # 简单解析策略（实际可以更复杂）
            strategies = [line.strip() for line in strategies_text.split('\n') if line.strip() and any(char.isdigit() for char in line[:3])]
            return strategies
        except Exception as e:
            return [f"策略生成失败: {str(e)}"]
    
    def optimize_prompt(self, base_prompt: str, strategy: str, target_metrics: Dict[str, float] = None) -> str:
        """根据策略优化单个prompt"""
        optimization_prompt = f"""
你是一个专业的prompt工程师。请根据以下优化策略改进给定的prompt：

## 原始Prompt
{base_prompt}

## 优化策略
{strategy}

## 目标指标（如果有）
{target_metrics or "提高整体性能"}

请生成一个改进后的prompt，要求：
1. 保持原始prompt的核心功能
2. 应用指定的优化策略
3. 语言清晰、逻辑严密
4. 适合漏洞检测任务

只返回优化后的prompt文本，不要额外解释。
"""
        
        try:
            optimized_prompt = sven_llm_query(optimization_prompt, self.llm_client, temperature=0.3)
            return optimized_prompt.strip()
        except Exception as e:
            return base_prompt  # 如果优化失败，返回原prompt
    
    def batch_optimize_population(self, prompts: List[str], strategies: List[str], 
                                 generation: int) -> List[str]:
        """批量优化种群中的prompts"""
        optimized_prompts = []
        
        print(f"🔧 Optimizing {len(prompts)} prompts with {len(strategies)} strategies...")
        
        for i, prompt in enumerate(prompts):
            # 为每个prompt选择不同的策略
            strategy = strategies[i % len(strategies)]
            
            try:
                optimized = self.optimize_prompt(prompt, strategy)
                optimized_prompts.append(optimized)
                print(f"  ✅ Optimized prompt {i+1}/{len(prompts)}")
            except Exception as e:
                print(f"  ❌ Failed to optimize prompt {i+1}: {e}")
                optimized_prompts.append(prompt)  # 保持原prompt
        
        return optimized_prompts


def create_evolution_tracker(output_dir: str, experiment_name: str = None) -> EvolutionTracker:
    """创建进化跟踪器的工厂函数"""
    return EvolutionTracker(output_dir, experiment_name)


if __name__ == "__main__":
    # 测试代码
    print("🧪 Testing Evolution Tracker...")
    
    tracker = create_evolution_tracker("./test_outputs", "test_experiment")
    tracker.start_experiment({"popsize": 10, "budget": 5, "dataset": "sven"})
    
    # 模拟一些数据
    import random
    for gen in range(3):
        scores = [random.uniform(0.7, 0.95) for _ in range(10)]
        tracker.log_generation(gen, scores)
        
        for i, score in enumerate(scores):
            prompt_text = f"Test prompt {gen}-{i} for vulnerability detection"
            detailed_scores = [score, score+0.1, score-0.05, score+0.02, score]
            tracker.log_prompt(gen, prompt_text, detailed_scores)
    
    # 测试分析
    analyzer = StatisticalAnalyzer(tracker)
    report = analyzer.generate_comprehensive_report(generation=2)
    
    print("📊 Analysis Report Generated!")
    print(f"Generation analysis: {len(report['generation_analysis'])} metrics")
    print(f"LLM insights length: {len(report['llm_insights'])} characters")