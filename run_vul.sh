#!/bin/bash

# PrimeVul + EvoPrompt 运行脚本
# 用于在漏洞检测任务上运行提示词进化

set -ex

# 环境变量设置
export CUBLAS_WORKSPACE_CONFIG=:16:8  
export CUDA_VISIBLE_DEVICES=0

# 进化算法参数
BUDGET=15                # 进化轮数
POPSIZE=10              # 种群大小
SEED=42                 # 随机种子
TEMPLATE=v1             # 模板版本
LLM_TYPE=gpt-3.5-turbo  # 使用的LLM类型

# 数据集设置
DATASET=primevul        # 数据集名称
SAMPLE_NUM=100          # 样本数量

# 创建必要的目录结构
mkdir -p ./data/vul_detection/$DATASET
mkdir -p ./data/primevul/$DATASET

# 输出路径
OUT_PATH=outputs/vul_detection/$DATASET/de/bd${BUDGET}_pop${POPSIZE}_${TEMPLATE}/$LLM_TYPE

# 创建输出目录
mkdir -p $OUT_PATH

echo "=== PrimeVul + EvoPrompt 配置 ==="
echo "Dataset: $DATASET"
echo "Evolution Mode: DE (Differential Evolution)"
echo "Budget: $BUDGET"
echo "Population Size: $POPSIZE"
echo "LLM Type: $LLM_TYPE"
echo "Output Path: $OUT_PATH"
echo "================================="

# 检查是否存在必要的文件
if [ ! -f "./data/primevul/$DATASET/dev.jsonl" ]; then
    echo "警告: 未找到 PrimeVul 开发集数据"
    echo "请将 PrimeVul 数据放在 ./data/primevul/$DATASET/ 目录下"
    echo "需要的文件: dev.jsonl, test.jsonl"
fi

if [ ! -f "./auth.yaml" ]; then
    echo "警告: 未找到 auth.yaml 配置文件"
    echo "请确保已配置 OpenAI API 密钥"
fi

# 运行漏洞检测的提示词进化
echo "开始运行漏洞检测提示词进化..."

/Volumes/Mac_Ext/link_cache/codes/EvoPrompt/.venv/bin/python run_vulnerability_detection.py \
    --seed $SEED \
    --dataset $DATASET \
    --task vul_detection \
    --batch-size 16 \
    --sample_num $SAMPLE_NUM \
    --language_model gpt \
    --budget $BUDGET \
    --popsize $POPSIZE \
    --evo_mode de \
    --llm_type $LLM_TYPE \
    --template $TEMPLATE \
    --output $OUT_PATH/seed$SEED \
    --dev_file ./data/vul_detection/$DATASET/dev.txt \
    --test_file ./data/vul_detection/$DATASET/test.txt \
    --setting default

echo "进化完成，正在生成结果报告..."

# 生成结果报告
/Volumes/Mac_Ext/link_cache/codes/EvoPrompt/.venv/bin/python get_result.py -p $OUT_PATH > $OUT_PATH/result.txt

echo "结果已保存到: $OUT_PATH/result.txt"

# 可选：运行遗传算法进行对比
echo "是否运行遗传算法进行对比？ (y/n)"
read -r run_ga

if [ "$run_ga" = "y" ] || [ "$run_ga" = "Y" ]; then
    echo "开始运行遗传算法..."
    
    GA_OUT_PATH=outputs/vul_detection/$DATASET/ga/bd${BUDGET}_pop${POPSIZE}_${TEMPLATE}/$LLM_TYPE
    mkdir -p $GA_OUT_PATH
    
    /Volumes/Mac_Ext/link_cache/codes/EvoPrompt/.venv/bin/python run_vulnerability_detection.py \
        --seed $SEED \
        --dataset $DATASET \
        --task vul_detection \
        --batch-size 16 \
        --sample_num $SAMPLE_NUM \
        --language_model gpt \
        --budget $BUDGET \
        --popsize $POPSIZE \
        --evo_mode ga \
        --llm_type $LLM_TYPE \
        --template $TEMPLATE \
        --output $GA_OUT_PATH/seed$SEED \
        --dev_file ./data/vul_detection/$DATASET/dev.txt \
        --test_file ./data/vul_detection/$DATASET/test.txt \
        --setting default
    
    /Volumes/Mac_Ext/link_cache/codes/EvoPrompt/.venv/bin/python get_result.py -p $GA_OUT_PATH > $GA_OUT_PATH/result.txt
    echo "遗传算法结果已保存到: $GA_OUT_PATH/result.txt"
fi

echo "所有实验完成！"
echo "请查看以下文件获取详细结果："
echo "- DE算法结果: $OUT_PATH/result.txt"
if [ "$run_ga" = "y" ] || [ "$run_ga" = "Y" ]; then
    echo "- GA算法结果: $GA_OUT_PATH/result.txt"
fi