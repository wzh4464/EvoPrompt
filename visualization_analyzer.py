#!/usr/bin/env python3
"""
可视化分析工具
用于分析和可视化智能进化实验的结果
"""

import os
import json
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EvolutionVisualizer:
    """进化过程可视化器"""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.db_files = list(self.experiment_dir.glob("*.db"))
        self.json_files = list(self.experiment_dir.glob("*.json"))
        
        # 创建图表输出目录
        self.plots_dir = self.experiment_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        print(f"📊 Initializing visualizer for: {experiment_dir}")
        print(f"   Found {len(self.db_files)} database files")
        print(f"   Found {len(self.json_files)} JSON files")
    
    def load_experiment_data(self, db_file: str = None) -> Dict[str, pd.DataFrame]:
        """加载实验数据"""
        if db_file is None and self.db_files:
            db_file = self.db_files[0]
        elif db_file is None:
            raise ValueError("No database file found")
        
        conn = sqlite3.connect(db_file)
        
        # 加载所有表
        tables = {}
        table_names = ['experiments', 'generations', 'prompts', 'predictions', 'analyses']
        
        for table_name in table_names:
            try:
                tables[table_name] = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                print(f"✅ Loaded {table_name}: {len(tables[table_name])} records")
            except Exception as e:
                print(f"⚠️ Failed to load {table_name}: {e}")
                tables[table_name] = pd.DataFrame()
        
        conn.close()
        return tables
    
    def plot_evolution_progress(self, data: Dict[str, pd.DataFrame], save: bool = True):
        """绘制进化进程图"""
        generations_df = data['generations']
        
        if generations_df.empty:
            print("⚠️ No generation data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Evolution Progress Analysis', fontsize=16, fontweight='bold')
        
        # 1. 分数进化趋势
        axes[0, 0].plot(generations_df['generation'], generations_df['best_score'], 
                       'o-', label='Best Score', linewidth=2, markersize=6)
        axes[0, 0].plot(generations_df['generation'], generations_df['avg_score'], 
                       's-', label='Average Score', linewidth=2, markersize=6)
        axes[0, 0].plot(generations_df['generation'], generations_df['worst_score'], 
                       '^-', label='Worst Score', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Score Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 多样性分析
        diversity = generations_df['best_score'] - generations_df['worst_score']
        axes[0, 1].plot(generations_df['generation'], diversity, 
                       'o-', color='purple', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Score Diversity')
        axes[0, 1].set_title('Population Diversity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 改进率
        if len(generations_df) > 1:
            improvement = generations_df['best_score'].diff()
            axes[1, 0].bar(generations_df['generation'][1:], improvement[1:], 
                          alpha=0.7, color='green')
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Score Improvement')
            axes[1, 0].set_title('Generation-wise Improvement')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 收敛分析
        axes[1, 1].plot(generations_df['generation'], generations_df['avg_score'], 
                       'o-', label='Average', linewidth=2)
        axes[1, 1].fill_between(generations_df['generation'], 
                               generations_df['worst_score'], 
                               generations_df['best_score'], 
                               alpha=0.3, label='Range')
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Convergence Analysis')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_file = self.plots_dir / "evolution_progress.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Evolution progress plot saved: {plot_file}")
        
        plt.show()
    
    def plot_prompt_performance(self, data: Dict[str, pd.DataFrame], save: bool = True):
        """绘制prompt性能分析"""
        prompts_df = data['prompts']
        
        if prompts_df.empty:
            print("⚠️ No prompt data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Prompt Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. 分数分布
        axes[0, 0].hist(prompts_df['score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Score Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 指标相关性热图
        metrics = ['accuracy', 'precision_val', 'recall_val', 'f1_score', 'specificity']
        available_metrics = [m for m in metrics if m in prompts_df.columns]
        
        if len(available_metrics) > 1:
            corr_matrix = prompts_df[available_metrics].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[0, 1], square=True)
            axes[0, 1].set_title('Metrics Correlation')
        
        # 3. 代数间性能比较
        if 'generation' in prompts_df.columns:
            gen_performance = prompts_df.groupby('generation')['score'].agg(['mean', 'std', 'max', 'min'])
            axes[1, 0].errorbar(gen_performance.index, gen_performance['mean'], 
                              yerr=gen_performance['std'], marker='o', capsize=5)
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Average Score')
            axes[1, 0].set_title('Performance by Generation')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 最佳prompt追踪
            best_per_gen = prompts_df.loc[prompts_df.groupby('generation')['score'].idxmax()]
            axes[1, 1].plot(best_per_gen['generation'], best_per_gen['score'], 
                          'ro-', linewidth=2, markersize=8)
            axes[1, 1].set_xlabel('Generation')
            axes[1, 1].set_ylabel('Best Score')
            axes[1, 1].set_title('Best Prompt Evolution')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_file = self.plots_dir / "prompt_performance.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Prompt performance plot saved: {plot_file}")
        
        plt.show()
    
    def plot_prediction_analysis(self, data: Dict[str, pd.DataFrame], save: bool = True):
        """绘制预测结果分析"""
        predictions_df = data['predictions']
        
        if predictions_df.empty:
            print("⚠️ No prediction data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Prediction Analysis', fontsize=16, fontweight='bold')
        
        # 1. 准确率分布
        if 'is_correct' in predictions_df.columns:
            accuracy_by_prompt = predictions_df.groupby('prompt_id')['is_correct'].mean()
            axes[0, 0].hist(accuracy_by_prompt, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 0].set_xlabel('Accuracy')
            axes[0, 0].set_ylabel('Number of Prompts')
            axes[0, 0].set_title('Accuracy Distribution')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 置信度分析
        if 'confidence' in predictions_df.columns:
            axes[0, 1].scatter(predictions_df['confidence'], predictions_df['is_correct'], 
                             alpha=0.6, s=30)
            axes[0, 1].set_xlabel('Confidence')
            axes[0, 1].set_ylabel('Correctness')
            axes[0, 1].set_title('Confidence vs Correctness')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 响应时间分析
        if 'response_time' in predictions_df.columns:
            axes[1, 0].hist(predictions_df['response_time'], bins=30, alpha=0.7, 
                          color='orange', edgecolor='black')
            axes[1, 0].set_xlabel('Response Time (seconds)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Response Time Distribution')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 错误分析
        if 'is_correct' in predictions_df.columns:
            error_rate = 1 - predictions_df['is_correct'].mean()
            correct_rate = predictions_df['is_correct'].mean()
            
            labels = ['Correct', 'Incorrect']
            sizes = [correct_rate, error_rate]
            colors = ['lightgreen', 'lightcoral']
            
            axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Overall Prediction Accuracy')
        
        plt.tight_layout()
        
        if save:
            plot_file = self.plots_dir / "prediction_analysis.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Prediction analysis plot saved: {plot_file}")
        
        plt.show()
    
    def generate_comprehensive_report(self, data: Dict[str, pd.DataFrame], save: bool = True):
        """生成综合可视化报告"""
        print("📊 Generating comprehensive visualization report...")
        
        # 创建所有图表
        self.plot_evolution_progress(data, save)
        self.plot_prompt_performance(data, save)
        self.plot_prediction_analysis(data, save)
        
        # 生成摘要统计
        summary = self._generate_summary_stats(data)
        
        if save:
            summary_file = self.plots_dir / "visualization_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"📄 Visualization summary saved: {summary_file}")
        
        print("✅ Comprehensive visualization report completed!")
        return summary
    
    def _generate_summary_stats(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """生成摘要统计"""
        summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'data_summary': {}
        }
        
        for table_name, df in data.items():
            if not df.empty:
                summary['data_summary'][table_name] = {
                    'record_count': len(df),
                    'columns': list(df.columns),
                    'date_range': {
                        'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                        'end': df['timestamp'].max() if 'timestamp' in df.columns else None
                    }
                }
        
        # 添加性能统计
        if not data['prompts'].empty:
            prompts_df = data['prompts']
            summary['performance_stats'] = {
                'best_score': float(prompts_df['score'].max()),
                'average_score': float(prompts_df['score'].mean()),
                'worst_score': float(prompts_df['score'].min()),
                'score_std': float(prompts_df['score'].std()),
                'total_prompts_evaluated': len(prompts_df)
            }
        
        return summary


def analyze_experiment_directory(experiment_dir: str):
    """分析整个实验目录"""
    print(f"🔍 Analyzing experiment directory: {experiment_dir}")
    
    if not os.path.exists(experiment_dir):
        print(f"❌ Directory not found: {experiment_dir}")
        return
    
    # 创建可视化器
    visualizer = EvolutionVisualizer(experiment_dir)
    
    # 加载数据
    try:
        data = visualizer.load_experiment_data()
        
        # 生成综合报告
        summary = visualizer.generate_comprehensive_report(data)
        
        print("📊 Analysis completed!")
        print(f"   Plots saved in: {visualizer.plots_dir}")
        print(f"   Best score achieved: {summary.get('performance_stats', {}).get('best_score', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        experiment_dir = sys.argv[1]
    else:
        experiment_dir = "./outputs/intelligent_vul_detection/sven/"
    
    analyze_experiment_directory(experiment_dir)