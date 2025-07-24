#!/bin/bash

# SVEN 漏洞检测数据集运行脚本
# 使用EvoPrompt进化算法优化漏洞检测提示词，采用SVEN风格的API调用

echo "Starting SVEN vulnerability detection with EvoPrompt..."

# 检查环境配置
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Using environment variables or defaults."
    echo "You can copy .env.example to .env and configure your API settings."
fi

# 设置数据集参数
DATASET="sven"
EVO_MODE="de"  # 差分进化算法

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Evolution Mode: $EVO_MODE"
echo "  Using SVEN-style API client"

# 运行漏洞检测任务
.venv/bin/python run_vulnerability_detection.py \
    --dataset $DATASET \
    --task vul_detection \
    --evo_mode $EVO_MODE \
    --popsize 10 \
    --budget 5 \
    --seed 42 \
    --output "./outputs/vul_detection/sven/" \
    --sample_num 50

if [ $? -eq 0 ]; then
    echo "✅ SVEN vulnerability detection completed successfully!"
    echo "📁 Results saved in ./outputs/vul_detection/sven/"
else
    echo "❌ SVEN vulnerability detection failed!"
    echo "💡 Tips:"
    echo "  - Check your .env configuration"
    echo "  - Ensure API_KEY is set correctly"
    echo "  - Verify API endpoints are accessible"
fi