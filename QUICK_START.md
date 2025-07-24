# SVEN + EvoPrompt 快速开始指南

## 🚀 快速上手（4步搞定）

### 1. 测试集成
```bash
# 首先测试SVEN集成是否成功（不需要API）
.venv/bin/python test_integration.py
```

### 2. 配置API
```bash
# 复制配置模板
cp .env.example .env

# 编辑.env文件，填入你的API配置
vim .env
```

`.env`文件内容示例：
```bash
API_BASE_URL=https://api.openai.com/v1
API_KEY=your_openai_api_key_here
BACKUP_API_BASE_URL=https://newapi.aicohere.org/v1
MODEL_NAME=gpt-3.5-turbo
```

### 3. 测试配置
```bash
# 测试API连接
.venv/bin/python test_sven_api.py

# 测试参数解析
.venv/bin/python test_args.py
```

### 4. 运行智能SVEN系统
```bash
# 🆕 智能版本（推荐）- 包含结果分析和自动优化
./run_intelligent_sven.sh

# 或传统版本
./run_sven.sh
```

## 🧠 新功能：智能分析与优化

### 智能系统特性
- ✅ **结果追踪**: 完整记录每个prompt的变化和性能
- ✅ **统计分析**: LLM驱动的性能模式识别
- ✅ **智能优化**: 基于分析结果自动生成优化策略
- ✅ **自适应进化**: 动态调整进化参数
- ✅ **可视化报告**: 丰富的图表和分析报告

### 使用智能系统
```bash
# 交互式运行（推荐）
./run_intelligent_sven.sh

# 直接运行
.venv/bin/python run_intelligent_vulnerability_detection.py \
    --dataset sven \
    --evo_mode de \
    --popsize 10 \
    --budget 5 \
    --sample_num 50
```

### 分析结果
```bash
# 生成可视化报告
.venv/bin/python visualization_analyzer.py ./outputs/intelligent_vul_detection/sven/

# 查看生成的文件
ls ./outputs/intelligent_vul_detection/sven/
```

## 📋 自定义运行

```bash
.venv/bin/python run_vulnerability_detection.py \
    --dataset sven \
    --task vul_detection \
    --evo_mode de \
    --popsize 10 \
    --budget 5 \
    --seed 42 \
    --sample_num 50
```

## 🎯 关键参数说明

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--dataset` | 数据集名称 | - | `sven` |
| `--task` | 任务类型 | - | `vul_detection` |
| `--evo_mode` | 进化算法 | `de` | `de` |
| `--popsize` | 种群大小 | 10 | 10-20 |
| `--budget` | 进化代数 | 10 | 5-10 |
| `--sample_num` | 优化样本数 | 100 | 50-100 |

## 📊 输出结果

运行完成后，结果保存在：
- 📁 `./outputs/vul_detection/sven/`
- 📄 最优提示词
- 📈 性能指标
- 📝 进化日志

## ⚡ 性能优化建议

### 测试阶段
```bash
# 快速测试（小规模）
--popsize 5 --budget 3 --sample_num 20
```

### 正式运行
```bash
# 完整优化（推荐）
--popsize 10 --budget 5 --sample_num 50
```

### 深度优化
```bash
# 高质量结果（耗时较长）
--popsize 20 --budget 10 --sample_num 100
```

## 🔧 故障排除

### 常见问题

1. **API_KEY未设置**
   ```bash
   # 检查.env文件
   cat .env
   
   # 或设置环境变量
   export API_KEY=your_key_here
   ```

2. **参数错误**
   ```bash
   # 运行参数测试
   .venv/bin/python test_args.py
   ```

3. **API连接失败**
   ```bash
   # 运行API测试
   .venv/bin/python test_sven_api.py
   ```

4. **虚拟环境问题**
   ```bash
   # 重新创建虚拟环境
   uv venv --python 3.11
   uv add requests tqdm
   ```

### 调试模式

```bash
# 查看详细日志
.venv/bin/python run_vulnerability_detection.py \
    --dataset sven \
    --task vul_detection \
    --popsize 3 \
    --budget 2 \
    --sample_num 10
```

## 📞 获取帮助

- 📖 详细文档：`SVEN_INTEGRATION.md`
- 🧪 API测试：`.venv/bin/python test_sven_api.py`
- ⚙️ 参数测试：`.venv/bin/python test_args.py`