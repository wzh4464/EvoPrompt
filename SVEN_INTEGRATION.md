# SVEN数据集集成说明

## 概述

已成功将SVEN (Security Hardening and Adversarial Testing for Code LLMs) 数据集集成到EvoPrompt项目中，用于漏洞检测任务的提示词进化优化。

## SVEN数据集介绍

SVEN是一个专门用于代码安全漏洞检测的数据集，包含以下CWE类型的漏洞样本：

- **CWE-022**: Path Traversal (路径遍历)
- **CWE-078**: OS Command Injection (操作系统命令注入)  
- **CWE-079**: Cross-site Scripting (跨站脚本攻击)
- **CWE-089**: SQL Injection (SQL注入)
- **CWE-125**: Out-of-bounds Read (越界读取)
- **CWE-190**: Integer Overflow (整数溢出)
- **CWE-416**: Use After Free (释放后使用)
- **CWE-476**: NULL Pointer Dereference (空指针解引用)
- **CWE-787**: Out-of-bounds Write (越界写入)

## 数据集结构

```
sven/
├── data_train_val/
│   ├── train/          # 训练数据
│   │   ├── cwe-022.jsonl
│   │   ├── cwe-078.jsonl
│   │   └── ...
│   └── val/            # 验证数据
│       ├── cwe-022.jsonl
│       ├── cwe-078.jsonl
│       └── ...
├── data_eval/          # 评估数据
└── CLAUDE.md           # 使用说明
```

## 使用方法

### 1. 快速开始

使用提供的脚本运行SVEN数据集：

```bash
./run_sven.sh
```

### 2. 环境配置

**新特性：使用SVEN风格的API客户端**

```bash
# 复制配置模板
cp .env.example .env

# 编辑.env文件配置API
```

`.env`文件示例：
```bash
# 主要API配置
API_BASE_URL=https://api.openai.com/v1
API_KEY=your_openai_api_key_here

# 备用API配置（可选）
BACKUP_API_BASE_URL=https://newapi.aicohere.org/v1

# 模型配置
MODEL_NAME=gpt-3.5-turbo
```

### 3. 自定义运行

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

### 4. 参数说明

- `--dataset sven`: 指定使用SVEN数据集
- `--task vul_detection`: 指定任务类型为漏洞检测
- `--evo_mode`: 进化算法模式 (de=差分进化, ga=遗传算法)
- `--popsize`: 种群大小
- `--budget`: 进化代数（迭代次数）
- `--sample_num`: 用于优化的样本数量
- `--seed`: 随机种子

### 完整参数列表

```bash
# 基本参数
--dataset sven                    # 数据集名称
--task vul_detection             # 任务类型
--evo_mode de                    # 进化算法 (de/ga)
--popsize 10                     # 种群大小
--budget 5                       # 进化代数
--seed 42                        # 随机种子
--sample_num 50                  # 优化样本数

# 可选参数
--output "./outputs/sven/"       # 输出目录
--template v1                    # 提示词模板
--initial all                    # 初始化模式
--llm_type turbo                 # LLM类型（已废弃，使用.env配置）
```

**注意**：不再需要指定`--llm_type`和`--model_name`，这些在`.env`文件中配置

## SVEN API客户端特性

### 新集成的API调用方式

已将原来的EvoPrompt LLM客户端替换为SVEN风格的API客户端，提供以下改进：

#### 1. 多API支持
- **主备切换**：主API失败时自动切换到备用API
- **容错机制**：单个API故障不影响整体任务
- **配置灵活**：支持多种API提供商

#### 2. 批量处理优化
- **批量查询**：一次性处理多个提示词，提高效率
- **智能回退**：批量失败时自动回退到单个查询
- **进度显示**：实时显示处理进度

#### 3. 错误处理增强
- **重试机制**：失败时自动重试备用API
- **错误恢复**：单个查询失败使用默认值，不中断流程
- **详细日志**：提供详细的错误信息和调试信息

#### 4. 配置管理
- **环境变量**：支持`.env`文件和系统环境变量
- **动态配置**：运行时可切换不同API配置
- **安全性**：敏感信息不会泄露到版本控制

### 与原EvoPrompt的差异

| 特性 | 原EvoPrompt | SVEN风格 |
|------|-------------|----------|
| API配置 | YAML文件 | .env环境变量 |
| 错误处理 | 基础重试 | 多层容错 |
| 批量处理 | 固定批量 | 智能批量+回退 |
| API切换 | 手动 | 自动切换 |
| 进度显示 | 基础 | 详细进度 |

### 5. API测试

在开始正式使用前，建议先测试API配置：

```bash
# 测试API连接和基本功能
.venv/bin/python test_sven_api.py

# 测试参数解析
.venv/bin/python test_args.py
```

测试包括：
- ✅ 环境配置检查
- ✅ 单次查询测试
- ✅ 批量查询测试
- ✅ 漏洞检测场景测试
- ⚠️ API故障转移测试（可选）

## 数据转换

系统会自动将SVEN的JSONL格式转换为EvoPrompt需要的格式：

### SVEN原始格式
```json
{
    "func_name": "vulnerable_function",
    "func_src_before": "int vulnerable_code() { ... }",
    "func_src_after": "int fixed_code() { ... }",
    "vul_type": "cwe-089",
    "commit_link": "https://github.com/...",
    "file_name": "example.c"
}
```

### EvoPrompt格式
```
// cwe-089
int vulnerable_code() { ... }	vulnerable
```

## 提示词模板

为SVEN数据集定制了专门的安全漏洞检测提示词：

1. **基础分析型**: 分析C/C++代码的CWE安全漏洞
2. **专家角色型**: 作为安全研究员进行漏洞识别
3. **具体类型型**: 针对特定CWE类型的检测
4. **审计流程型**: 按照安全审计流程分析
5. **综合检查型**: 全面检查多种CWE弱点

## 输出结果

运行完成后，结果保存在 `./outputs/vul_detection/sven/` 目录：

- 最优提示词
- 性能指标 (准确率、精确率、召回率、F1分数)
- 进化过程日志
- 详细预测结果

## 评估指标

- **准确率 (Accuracy)**: 正确分类的比例
- **精确率 (Precision)**: 预测为漏洞中真正漏洞的比例
- **召回率 (Recall)**: 真实漏洞中被正确识别的比例
- **F1分数**: 精确率和召回率的调和平均
- **特异性 (Specificity)**: 真实安全代码被正确识别的比例

## 注意事项

1. **数据质量**: SVEN数据集主要包含有漏洞的代码，需要注意样本平衡
2. **模型选择**: 建议使用代码理解能力强的模型(如CodeT5, CodeBERT等)
3. **提示词设计**: 专门针对C/C++代码和CWE分类体系设计
4. **评估标准**: 使用安全领域专用的评估指标

## 扩展功能

1. **多语言支持**: 可扩展支持Python、Java等其他语言的漏洞检测
2. **细粒度分类**: 可进一步细分到具体的CWE子类型
3. **修复建议**: 结合修复后的代码提供漏洞修复建议
4. **置信度评估**: 为漏洞检测结果提供置信度分数

## 相关文件

- `run_vulnerability_detection.py`: 主要运行脚本
- `run_sven.sh`: 快速运行脚本
- `sven/`: SVEN数据集目录
- `data/vul_detection/sven/`: 转换后的数据存储目录

## 参考资料

- [SVEN论文](https://arxiv.org/abs/2302.05319)
- [CWE分类标准](https://cwe.mitre.org/)
- [EvoPrompt框架](https://github.com/beeevita/EvoPrompt)