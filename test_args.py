#!/usr/bin/env python3
"""
测试run_vulnerability_detection.py的参数是否正确
"""

import sys
import os
sys.path.append("./")

from args import parse_args


def test_args():
    """测试参数解析"""
    print("🔍 Testing argument parsing...")
    
    # 模拟命令行参数
    test_args = [
        "--dataset", "sven",
        "--task", "vul_detection", 
        "--evo_mode", "de",
        "--popsize", "10",
        "--budget", "5",
        "--seed", "42",
        "--sample_num", "50",
        "--output", "./outputs/vul_detection/sven/"
    ]
    
    # 备份原始sys.argv
    original_argv = sys.argv.copy()
    
    try:
        # 设置测试参数
        sys.argv = ["test_args.py"] + test_args
        
        # 解析参数
        args = parse_args()
        
        print("✅ Arguments parsed successfully!")
        print("📊 Parsed arguments:")
        print(f"  dataset: {args.dataset}")
        print(f"  task: {args.task}")
        print(f"  evo_mode: {args.evo_mode}")
        print(f"  popsize: {args.popsize}")
        print(f"  budget: {args.budget}")
        print(f"  seed: {args.seed}")
        print(f"  sample_num: {args.sample_num}")
        print(f"  output: {args.output}")
        
        return True
        
    except Exception as e:
        print(f"❌ Argument parsing failed: {e}")
        return False
        
    finally:
        # 恢复原始sys.argv
        sys.argv = original_argv


def check_output_dir():
    """检查输出目录"""
    output_dir = "./outputs/vul_detection/sven/"
    if os.path.exists(output_dir):
        print(f"✅ Output directory exists: {output_dir}")
        return True
    else:
        print(f"⚠️  Output directory does not exist: {output_dir}")
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"✅ Created output directory: {output_dir}")
            return True
        except Exception as e:
            print(f"❌ Failed to create output directory: {e}")
            return False


if __name__ == "__main__":
    print("🚀 Testing SVEN Arguments")
    print("=" * 40)
    
    success = True
    
    # 测试参数解析
    if not test_args():
        success = False
    
    print()
    
    # 检查输出目录
    if not check_output_dir():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed! You can now run the SVEN script.")
        print("💡 Run: ./run_sven.sh")
    else:
        print("⚠️  Some tests failed. Please check the configuration.")