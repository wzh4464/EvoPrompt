#!/bin/bash

# 智能SVEN漏洞检测运行脚本
# 集成结果存储、统计分析和智能prompt优化功能

echo "🧠 Starting Intelligent SVEN Vulnerability Detection System"
echo "============================================================"

# 检查环境配置
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Using environment variables or defaults."
    echo "💡 You can copy .env.example to .env and configure your API settings."
fi

# 设置数据集参数
DATASET="sven"
EVO_MODE="de"  # 差分进化算法

echo "📊 Configuration:"
echo "  Dataset: $DATASET"
echo "  Evolution Mode: $EVO_MODE"
echo "  Features: Intelligent Analysis + Auto Optimization"
echo ""

# 询问用户选择运行模式
echo "🎛️  Choose running mode:"
echo "  1) Quick Test (3 pop, 2 gen, 10 samples) - Fast"
echo "  2) Normal Run (10 pop, 5 gen, 50 samples) - Recommended"
echo "  3) Deep Analysis (20 pop, 10 gen, 100 samples) - Thorough"
echo "  4) Custom Configuration"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        POPSIZE=3
        BUDGET=2
        SAMPLE_NUM=10
        echo "⚡ Quick test mode selected"
        ;;
    2)
        POPSIZE=10
        BUDGET=5
        SAMPLE_NUM=50
        echo "🎯 Normal run mode selected"
        ;;
    3)
        POPSIZE=20
        BUDGET=10
        SAMPLE_NUM=100
        echo "🔬 Deep analysis mode selected"
        ;;
    4)
        read -p "Population size: " POPSIZE
        read -p "Number of generations: " BUDGET
        read -p "Sample size: " SAMPLE_NUM
        echo "⚙️  Custom configuration: pop=$POPSIZE, gen=$BUDGET, samples=$SAMPLE_NUM"
        ;;
    *)
        POPSIZE=10
        BUDGET=5
        SAMPLE_NUM=50
        echo "🎯 Using default configuration"
        ;;
esac

echo ""
echo "🚀 Starting intelligent vulnerability detection..."
echo "   Population Size: $POPSIZE"
echo "   Generations: $BUDGET"
echo "   Sample Size: $SAMPLE_NUM"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 运行智能漏洞检测任务
.venv/bin/python run_intelligent_vulnerability_detection.py \
    --dataset $DATASET \
    --task vul_detection \
    --evo_mode $EVO_MODE \
    --popsize $POPSIZE \
    --budget $BUDGET \
    --seed 42 \
    --output "./outputs/intelligent_vul_detection/sven/" \
    --sample_num $SAMPLE_NUM

# 检查运行结果
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "============================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Intelligent SVEN vulnerability detection completed successfully!"
    echo "⏱️  Total runtime: ${DURATION} seconds"
    echo "📁 Results saved in: ./outputs/intelligent_vul_detection/sven/"
    echo ""
    echo "📊 Generated Files:"
    echo "  - experiment_summary.json     # Overall results summary"
    echo "  - detailed_cache.json         # Detailed prediction results"
    echo "  - analysis_gen_*.json         # Generation-wise analysis"
    echo "  - strategies_gen_*.txt        # Optimization strategies"
    echo "  - final_analysis_report.json  # Complete analysis report"
    echo "  - *.db                        # SQLite database with all data"
    echo ""
    echo "💡 Analysis Features:"
    echo "  ✅ Statistical bias detection"
    echo "  ✅ Performance pattern analysis"
    echo "  ✅ LLM-powered insights"
    echo "  ✅ Adaptive optimization strategies"
    echo "  ✅ Comprehensive result tracking"
    echo ""
    echo "🎯 Next Steps:"
    echo "  1. Review the analysis reports for insights"
    echo "  2. Check optimization strategies for manual tuning"
    echo "  3. Use best prompts for production deployment"
    echo "  4. Run additional experiments with different parameters"
    
else
    echo "❌ Intelligent SVEN vulnerability detection failed!"
    echo "⏱️  Runtime before failure: ${DURATION} seconds"
    echo ""
    echo "🔧 Troubleshooting Tips:"
    echo "  - Check your .env configuration"
    echo "  - Ensure API_KEY is set correctly"
    echo "  - Verify API endpoints are accessible"
    echo "  - Try running with smaller parameters first"
    echo "  - Check ./outputs/intelligent_vul_detection/sven/error.log"
    echo ""
    echo "🧪 Quick Test Command:"
    echo "  ./run_intelligent_sven.sh  # Then select option 1"
    echo ""
    echo "📞 Get Help:"
    echo "  .venv/bin/python test_integration.py  # Test basic integration"
    echo "  .venv/bin/python test_sven_api.py     # Test API connection"
fi

echo "============================================================"