"""Data sampling utilities for balanced dataset creation."""

import random
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BalancedSampler:
    """Creates balanced samples from datasets."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        
    def sample_primevul_balanced(
        self, 
        data_file: str, 
        sample_ratio: float = 0.01,
        target_field: str = "target"
    ) -> Tuple[List[Dict], Dict[str, int]]:
        """
        从Primevul数据集中采样均衡的数据。
        
        Args:
            data_file: JSONL格式的数据文件路径
            sample_ratio: 采样比例 (0.01 = 1%)
            target_field: 目标标签字段名
            
        Returns:
            (sampled_data, statistics)
        """
        logger.info(f"Loading data from {data_file}")
        
        # 加载所有数据
        all_data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        all_data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {line[:100]}... Error: {e}")
        
        logger.info(f"Loaded {len(all_data)} total samples")
        
        # 按标签分组
        label_groups = {}
        for item in all_data:
            label = item.get(target_field, 0)
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(item)
        
        # 计算统计信息
        stats = {f"total_{label}": len(items) for label, items in label_groups.items()}
        stats["total_samples"] = len(all_data)
        
        logger.info("Original data distribution:")
        for label, count in stats.items():
            if label.startswith("total_") and label != "total_samples":
                percentage = (count / len(all_data)) * 100
                logger.info(f"  Label {label.split('_')[1]}: {count} samples ({percentage:.1f}%)")
        
        # 计算每个标签需要采样的数量
        total_target_samples = int(len(all_data) * sample_ratio)
        min_label_count = min(len(items) for items in label_groups.values())
        max_samples_per_label = int(total_target_samples / len(label_groups))
        
        # 确保不超过最小标签的样本数
        samples_per_label = min(max_samples_per_label, min_label_count)
        
        logger.info(f"Target total samples: {total_target_samples}")
        logger.info(f"Samples per label: {samples_per_label}")
        
        # 从每个标签组中均匀采样
        sampled_data = []
        sampled_stats = {}
        
        for label, items in label_groups.items():
            # 随机采样
            sampled_items = random.sample(items, samples_per_label)
            sampled_data.extend(sampled_items)
            sampled_stats[f"sampled_{label}"] = len(sampled_items)
            
            logger.info(f"Sampled {len(sampled_items)} samples from label {label}")
        
        # 打乱最终数据
        random.shuffle(sampled_data)
        
        # 更新统计信息
        sampled_stats["sampled_total"] = len(sampled_data)
        sampled_stats["sample_ratio"] = len(sampled_data) / len(all_data)
        sampled_stats.update(stats)  # 包含原始统计信息
        
        logger.info(f"Final balanced sample: {len(sampled_data)} samples ({sampled_stats['sample_ratio']:.3f} of original)")
        
        return sampled_data, sampled_stats
    
    def save_sampled_data(
        self, 
        sampled_data: List[Dict], 
        output_file: str,
        format: str = "jsonl"
    ):
        """保存采样数据到文件。"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in sampled_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif format.lower() == "tab":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in sampled_data:
                    func = item.get('func', '').strip()
                    target = str(item.get('target', 0))
                    # 清理函数代码为单行
                    func_clean = func.replace('\n', ' ').replace('\t', ' ')
                    while '  ' in func_clean:
                        func_clean = func_clean.replace('  ', ' ')
                    f.write(f"{func_clean}\t{target}\n")
        
        logger.info(f"Saved {len(sampled_data)} samples to {output_path}")
        return str(output_path)
    
    def create_train_dev_split(
        self,
        sampled_data: List[Dict],
        dev_ratio: float = 0.3,
        target_field: str = "target"
    ) -> Tuple[List[Dict], List[Dict]]:
        """将采样数据分割为训练集和开发集，保持平衡。"""
        
        # 按标签分组
        label_groups = {}
        for item in sampled_data:
            label = item.get(target_field, 0) 
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(item)
        
        train_data = []
        dev_data = []
        
        # 从每个标签组中分割
        for label, items in label_groups.items():
            random.shuffle(items)
            
            dev_count = int(len(items) * dev_ratio)
            train_count = len(items) - dev_count
            
            train_data.extend(items[:train_count])
            dev_data.extend(items[train_count:])
            
            logger.info(f"Label {label}: {train_count} train, {dev_count} dev")
        
        # 打乱数据
        random.shuffle(train_data)
        random.shuffle(dev_data)
        
        logger.info(f"Split complete: {len(train_data)} train, {len(dev_data)} dev samples")
        
        return train_data, dev_data


def sample_primevul_1percent(
    primevul_dir: str,
    output_dir: str,
    seed: int = 42
) -> Dict[str, Any]:
    """
    从Primevul数据集采样1%均衡数据的便捷函数。
    
    Args:
        primevul_dir: Primevul数据目录路径
        output_dir: 输出目录路径
        seed: 随机种子
        
    Returns:
        包含文件路径和统计信息的字典
    """
    sampler = BalancedSampler(seed=seed)
    
    # 检查数据文件
    primevul_path = Path(primevul_dir)
    dev_file = primevul_path / "dev.jsonl"
    
    if not dev_file.exists():
        # 尝试其他可能的文件名
        possible_files = [
            primevul_path / "primevul_train.jsonl",
            primevul_path / "train.jsonl",
            primevul_path / "primevul_valid.jsonl"
        ]
        
        for possible_file in possible_files:
            if possible_file.exists():
                dev_file = possible_file
                break
        else:
            raise FileNotFoundError(f"No Primevul data file found in {primevul_dir}")
    
    logger.info(f"Using data file: {dev_file}")
    
    # 采样1%数据
    sampled_data, stats = sampler.sample_primevul_balanced(
        data_file=str(dev_file),
        sample_ratio=0.01
    )
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 分割为训练和开发集
    train_data, dev_data = sampler.create_train_dev_split(
        sampled_data, 
        dev_ratio=0.3
    )
    
    # 保存数据文件
    files = {}
    
    # 保存JSONL格式（用于分析）
    files["train_jsonl"] = sampler.save_sampled_data(
        train_data, 
        str(output_path / "train_sample.jsonl"), 
        format="jsonl"
    )
    files["dev_jsonl"] = sampler.save_sampled_data(
        dev_data,
        str(output_path / "dev_sample.jsonl"),
        format="jsonl"
    )
    
    # 保存Tab格式（用于EvoPrompt）
    files["train_tab"] = sampler.save_sampled_data(
        train_data,
        str(output_path / "train.txt"),
        format="tab"
    )
    files["dev_tab"] = sampler.save_sampled_data(
        dev_data,
        str(output_path / "dev.txt"),
        format="tab"
    )
    
    # 保存统计信息
    stats_file = output_path / "sampling_stats.json"
    stats["files"] = files
    stats["sampling_config"] = {
        "sample_ratio": 0.01,
        "dev_ratio": 0.3,
        "seed": seed,
        "source_file": str(dev_file)
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    files["stats"] = str(stats_file)
    
    logger.info("Sampling complete!")
    logger.info(f"Files created:")
    for name, path in files.items():
        logger.info(f"  {name}: {path}")
    
    return {
        "files": files,
        "statistics": stats,
        "train_samples": len(train_data),
        "dev_samples": len(dev_data),
        "total_samples": len(sampled_data)
    }