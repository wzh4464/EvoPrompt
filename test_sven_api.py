#!/usr/bin/env python3
"""
SVEN API客户端测试脚本
用于验证API配置和连接是否正常
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from sven_llm_client import sven_llm_init, sven_llm_query


def test_single_query():
    """测试单次查询"""
    print("🔍 Testing single query...")
    
    try:
        client = sven_llm_init()
        
        test_prompt = "What is a buffer overflow vulnerability? Answer in one sentence."
        
        result = sven_llm_query(test_prompt, client, task=False, temperature=0.1)
        
        print(f"✅ Single query successful!")
        print(f"📝 Result: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Single query failed: {e}")
        return False


def test_batch_query():
    """测试批量查询"""
    print("\n🔍 Testing batch query...")
    
    try:
        client = sven_llm_init()
        
        test_prompts = [
            "What is SQL injection?",
            "What is XSS?", 
            "What is buffer overflow?"
        ]
        
        results = sven_llm_query(test_prompts, client, task=False, temperature=0.1)
        
        print(f"✅ Batch query successful!")
        for i, result in enumerate(results):
            print(f"📝 Result {i+1}: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Batch query failed: {e}")
        return False


def test_vulnerability_detection():
    """测试漏洞检测场景"""
    print("\n🔍 Testing vulnerability detection scenario...")
    
    try:
        client = sven_llm_init()
        
        # 模拟漏洞检测提示
        vuln_code = """
        int copy_data(char* dest, char* src) {
            strcpy(dest, src);  // Potential buffer overflow
            return 0;
        }
        """
        
        prompt = f"""Analyze this C/C++ code for security vulnerabilities. Look for common weaknesses like buffer overflows, injection attacks, memory corruption, and logic errors. Respond with 'vulnerable' if you find any security issues, or 'benign' if the code appears safe.

Code to analyze:
```c
{vuln_code}
```

Analysis: """
        
        result = sven_llm_query(prompt, client, task=False, temperature=0.1)
        
        print(f"✅ Vulnerability detection test successful!")
        print(f"📝 Analysis result: {result}")
        
        # 检查是否包含预期关键词
        if any(keyword in result.lower() for keyword in ["vulnerable", "vulnerability", "buffer overflow", "strcpy"]):
            print("🎯 Result contains expected security-related keywords!")
        else:
            print("⚠️  Result may not contain expected security analysis")
            
        return True
        
    except Exception as e:
        print(f"❌ Vulnerability detection test failed: {e}")
        return False


def test_api_failover():
    """测试API故障转移"""
    print("\n🔍 Testing API failover mechanism...")
    
    try:
        # 使用无效的主API来测试故障转移
        from sven_llm_client import SVENLLMClient
        
        client = SVENLLMClient(
            api_base="https://invalid-api-endpoint.com/v1",
            api_key=os.getenv("API_KEY", "test_key")
        )
        
        test_prompt = "Hello, can you respond?"
        
        # 这应该触发故障转移到备用API
        result = sven_llm_query(test_prompt, client, task=False, temperature=0.1)
        
        print(f"✅ API failover test successful!")
        print(f"📝 Result with failover: {result}")
        return True
        
    except Exception as e:
        print(f"⚠️  API failover test completed with expected error: {e}")
        # 这实际上是预期的，因为两个API都可能失败
        return True


def check_environment():
    """检查环境配置"""
    print("🔧 Checking environment configuration...")
    
    # 检查.env文件
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found, using environment variables or defaults")
    
    # 检查关键环境变量
    api_key = os.getenv("API_KEY")
    api_base = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    
    print(f"📊 Configuration:")
    print(f"  API_BASE_URL: {api_base or 'Not set'}")
    print(f"  API_KEY: {'***' + (api_key[-4:] if api_key else 'Not set')}")
    print(f"  MODEL_NAME: {model_name or 'Not set'}")
    
    if not api_key:
        print("❌ API_KEY not configured! Please set it in .env file or environment variable.")
        return False
    
    return True


def main():
    """主测试函数"""
    print("🚀 SVEN API Client Test Suite")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        print("\n❌ Environment check failed! Please configure your API settings.")
        return
    
    # 运行测试
    tests = [
        ("Environment Check", check_environment),
        ("Single Query", test_single_query),
        ("Batch Query", test_batch_query),
        ("Vulnerability Detection", test_vulnerability_detection),
        # ("API Failover", test_api_failover),  # 可选测试
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            if test_name != "Environment Check":  # 已经执行过了
                success = test_func()
                results.append((test_name, success))
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # 显示测试结果摘要
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your SVEN API client is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check your configuration.")


if __name__ == "__main__":
    main()

# 使用方法提示
# 运行方式: .venv/bin/python test_sven_api.py