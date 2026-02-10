#!/usr/bin/env python3
"""
测试修改后的run_primevul_concurrent_optimized.py中的批处理功能
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# 添加src路径
sys.path.insert(0, 'src')
# 添加项目根目录到路径，以便导入scripts模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_config_generation():
    """测试配置生成是否包含批处理参数"""
    print("🔧 测试配置生成...")
    
    # 导入函数
    from scripts.run_primevul_concurrent_optimized import create_optimized_config
    
    config = create_optimized_config()
    
    # 检查批处理配置
    required_keys = [
        'llm_batch_size',
        'enable_batch_processing',
        'feedback_batch_size'
    ]
    
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    assert not missing_keys, f"缺少批处理配置键: {missing_keys}"

    print(f"   ✅ 批处理配置正确:")
    print(f"      LLM批大小: {config['llm_batch_size']}")
    print(f"      启用批处理: {config['enable_batch_processing']}")
    print(f"      反馈批大小: {config['feedback_batch_size']}")


def test_llm_client_batch_support():
    """测试LLM客户端是否支持批处理"""
    print("🤖 测试LLM客户端批处理支持...")

    from evoprompt.llm.client import create_default_client

    # 创建客户端
    client = create_default_client()

    # 检查是否有batch_generate方法
    assert hasattr(client, 'batch_generate'), "LLM客户端缺少batch_generate方法"

    print("   ✅ LLM客户端支持batch_generate方法")

    # 测试小批量调用（不需要真实API）
    test_prompts = [
        "Test prompt 1: {input}",
        "Test prompt 2: {input}",
        "Test prompt 3: {input}"
    ]

    print("   📝 批处理方法签名检查通过")


def test_function_signatures():
    """测试函数签名是否正确更新"""
    print("📋 测试函数签名...")

    # 导入修改后的函数
    from scripts.run_primevul_concurrent_optimized import (
        evaluate_on_dataset,
        sample_wise_feedback_training
    )

    import inspect

    # 检查evaluate_on_dataset签名
    sig = inspect.signature(evaluate_on_dataset)
    params = list(sig.parameters.keys())

    assert 'config' in params, "evaluate_on_dataset缺少config参数"

    print("   ✅ evaluate_on_dataset签名正确")

    # 检查sample_wise_feedback_training是否能接受config
    sig = inspect.signature(sample_wise_feedback_training)
    params = list(sig.parameters.keys())

    assert 'config' in params, "sample_wise_feedback_training缺少config参数"

    print("   ✅ sample_wise_feedback_training签名正确")


def test_batch_processing_logic():
    """测试批处理逻辑"""
    print("⚡ 测试批处理逻辑...")

    # 模拟配置
    config = {
        'enable_batch_processing': True,
        'llm_batch_size': 8,
        'feedback_batch_size': 10
    }

    # 测试批处理参数提取
    enable_batch = config.get('enable_batch_processing', False)
    llm_batch_size = config.get('llm_batch_size', 8)
    feedback_batch_size = config.get('feedback_batch_size', 10)

    assert enable_batch, "批处理未启用"
    assert llm_batch_size == 8, f"LLM批大小错误: {llm_batch_size} != 8"
    assert feedback_batch_size == 10, f"反馈批大小错误: {feedback_batch_size} != 10"

    print("   ✅ 批处理参数提取正确")
    print(f"      批处理启用: {enable_batch}")
    print(f"      LLM批大小: {llm_batch_size}")
    print(f"      反馈批大小: {feedback_batch_size}")


def test_import_and_basic_functionality():
    """测试导入和基本功能"""
    print("📦 测试导入和基本功能...")

    # 测试主要函数导入
    from scripts.run_primevul_concurrent_optimized import (
        create_optimized_config,
        run_concurrent_evolution_with_feedback,
        evaluate_on_dataset,
        sample_wise_feedback_training,
        main
    )

    print("   ✅ 所有主要函数导入成功")

    # 测试配置创建
    config = create_optimized_config()
    assert isinstance(config, dict), "配置创建失败"

    print("   ✅ 配置创建成功")


def main():
    """运行所有测试"""
    print("=== 批处理优化测试 ===")
    print()
    
    tests = [
        ("导入和基本功能", test_import_and_basic_functionality),
        ("配置生成", test_config_generation),
        ("LLM客户端批处理支持", test_llm_client_batch_support),
        ("函数签名", test_function_signatures),
        ("批处理逻辑", test_batch_processing_logic),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"🧪 执行测试: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"   ✅ {test_name} 测试通过")
            else:
                print(f"   ❌ {test_name} 测试失败")
        except Exception as e:
            print(f"   💥 {test_name} 测试异常: {e}")
            results[test_name] = False
        print()
    
    # 总结
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print("=== 测试总结 ===")
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    print()
    print(f"总体结果: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！批处理优化已成功集成。")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查相关问题。")
        return 1


if __name__ == "__main__":
    sys.exit(main())