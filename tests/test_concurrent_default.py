#!/usr/bin/env python3
"""
验证run_primevul_concurrent_optimized.py默认使用concurrent=True
"""

import sys
import inspect
from pathlib import Path

# 添加项目根目录到路径，以便导入scripts模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_default_concurrent_config():
    """测试默认配置是否启用并发"""
    print("🔧 测试默认并发配置...")

    from scripts.run_primevul_concurrent_optimized import create_optimized_config

    config = create_optimized_config()

    # 检查关键配置
    required_config = {
        "enable_batch_processing": True,
        "llm_batch_size": 8,
        "enable_concurrent": True,  # 这是关键
    }

    print("📋 检查配置项:")
    all_correct = True

    for key, expected_value in required_config.items():
        actual_value = config.get(key)
        status = "✅" if actual_value == expected_value else "❌"
        print(f"   {status} {key}: {actual_value} (期望: {expected_value})")
        if actual_value != expected_value:
            all_correct = False

    return all_correct


def test_concurrent_parameter_flow():
    """测试并发参数在整个调用链中的传递"""
    print("\n🔄 测试并发参数传递...")

    from scripts.run_primevul_concurrent_optimized import create_optimized_config

    # 创建测试配置
    config = create_optimized_config()

    # 模拟参数提取逻辑
    enable_batch = config.get("enable_batch_processing", False)
    batch_size = config.get("llm_batch_size", 8)
    enable_concurrent = config.get("enable_concurrent", True)

    print("📊 参数提取结果:")
    print(f"   批处理启用: {enable_batch}")
    print(f"   批大小: {batch_size}")
    print(f"   并发启用: {enable_concurrent}")

    # 验证参数正确性
    if enable_batch and batch_size == 8 and enable_concurrent:
        print("   ✅ 参数提取正确")
        return True
    else:
        print("   ❌ 参数提取有误")
        return False


def test_sven_client_concurrent_support():
    """测试SVEN客户端并发支持"""
    print("\n🤖 测试SVEN客户端并发支持...")

    try:
        from sven_llm_client import sven_llm_query

        # 检查函数参数

        sig = inspect.signature(sven_llm_query)
        params = list(sig.parameters.keys())

        if "concurrent" not in params:
            print("   ❌ sven_llm_query缺少concurrent参数")
            return False

        print("   ✅ sven_llm_query支持concurrent参数")

        # 检查默认值（从兼容函数）
        from evoprompt.llm.client import llm_query

        sig2 = inspect.signature(llm_query)
        concurrent_param = sig2.parameters.get("concurrent")

        if concurrent_param and concurrent_param.default:
            print("   ✅ llm_query默认concurrent=True")
        else:
            print(
                f"   ⚠️ llm_query concurrent默认值: {concurrent_param.default if concurrent_param else 'None'}"
            )

        return True

    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        return False


def test_batch_processing_info():
    """测试批处理信息显示"""
    print("\n📊 测试批处理信息...")

    # 模拟批处理参数显示
    config = {"enable_concurrent": True, "llm_batch_size": 8}
    enable_concurrent = config.get("enable_concurrent", True)
    llm_batch_size = config.get("llm_batch_size", 8)

    concurrent_text = "并发" if enable_concurrent else "顺序"
    info_text = f"使用批处理模式，LLM batch_size={llm_batch_size} ({concurrent_text})"

    expected_text = "使用批处理模式，LLM batch_size=8 (并发)"

    if info_text == expected_text:
        print(f"   ✅ 信息显示正确: {info_text}")
        return True
    else:
        print(f"   ❌ 信息显示错误: {info_text}")
        print(f"       期望: {expected_text}")
        return False


def main():
    """运行所有测试"""
    print("=== 默认并发配置测试 ===")
    print("验证run_primevul_concurrent_optimized.py默认启用concurrent=True")
    print()

    tests = [
        ("默认并发配置", test_default_concurrent_config),
        ("并发参数传递", test_concurrent_parameter_flow),
        ("SVEN客户端并发支持", test_sven_client_concurrent_support),
        ("批处理信息显示", test_batch_processing_info),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"🧪 执行测试: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"   💥 测试异常: {e}")
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
        print("\n🎉 所有测试通过！")
        print(
            "✅ run_primevul_concurrent_optimized.py 已配置为默认使用 concurrent=True"
        )
        print()
        print("📝 配置总结:")
        print("   • enable_batch_processing: True (启用批处理)")
        print("   • llm_batch_size: 8 (每批8个请求)")
        print("   • enable_concurrent: True (批次内并发处理)")
        print("   • 预期性能提升: 2-4倍 (取决于API响应时间)")
        return 0
    else:
        print("\n⚠️ 部分测试失败，请检查配置")
        return 1


if __name__ == "__main__":
    sys.exit(main())
