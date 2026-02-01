#!/usr/bin/env python3
"""
自动运行推理并分析结果

用法:
  python examples/dobot_cr5/run_and_analyze.py \\
    --blocking-servo True \\
    --observation-delay-ms 10 \\
    --replan-steps 3 \\
    --max-steps 600 \\
    --prompt "Pick up the red block"
"""

import subprocess
import sys
import glob
import os
from pathlib import Path
import time


def run_inference(args):
    """运行推理并记录数据"""
    print("=" * 60)
    print("步骤 1/2: 运行推理并记录数据")
    print("=" * 60)

    # 构建 main.py 的命令
    cmd = [
        sys.executable,  # python
        "examples/dobot_cr5/main.py",
        "--record",  # 启用记录
    ]

    # 添加用户提供的参数
    cmd.extend(args)

    print(f"执行命令: {' '.join(cmd)}")
    print()

    # 记录开始时间
    start_time = time.time()

    # 运行推理
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("❌ 推理失败")
        sys.exit(1)

    print()
    print("✓ 推理完成")
    print()

    return start_time


def find_latest_h5(start_time):
    """查找最新生成的 h5 文件"""
    inference_logs_dir = Path("inference_logs")

    if not inference_logs_dir.exists():
        print("❌ 错误: inference_logs 目录不存在")
        sys.exit(1)

    # 查找所有 h5 文件
    h5_files = list(inference_logs_dir.glob("inference_log_*.h5"))

    if not h5_files:
        print("❌ 错误: 未找到任何 h5 文件")
        sys.exit(1)

    # 筛选出在 start_time 之后创建的文件
    recent_files = [f for f in h5_files if f.stat().st_mtime > start_time]

    if not recent_files:
        print("❌ 错误: 未找到新生成的 h5 文件")
        sys.exit(1)

    # 返回最新的文件
    latest_file = max(recent_files, key=lambda f: f.stat().st_mtime)
    return latest_file


def run_analysis(h5_file):
    """运行分析脚本"""
    print("=" * 60)
    print("步骤 2/2: 分析数据并生成轨迹图")
    print("=" * 60)
    print(f"数据文件: {h5_file}")
    print()

    # 运行分析脚本
    cmd = [
        sys.executable,
        "examples/dobot_cr5/analyze_inference_log.py",
        str(h5_file),
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("❌ 分析失败")
        sys.exit(1)

    print()
    print("✓ 分析完成")
    print()

    # 计算输出目录
    h5_filename = h5_file.stem
    output_name = h5_filename.replace('inference_log_', '')
    output_dir = Path(f'analysis_results/{output_name}')

    return output_dir


def main():
    """主函数"""
    # 获取传递给此脚本的所有参数
    user_args = sys.argv[1:]

    print()
    print("=" * 60)
    print("推理和分析自动化脚本")
    print("=" * 60)
    print()

    # 步骤 1: 运行推理
    start_time = run_inference(user_args)

    # 步骤 2: 查找生成的 h5 文件
    h5_file = find_latest_h5(start_time)

    # 步骤 3: 运行分析
    output_dir = run_analysis(h5_file)

    # 显示结果
    print("=" * 60)
    print("✓ 全部完成！")
    print("=" * 60)
    print()
    print(f"数据文件: {h5_file}")
    print(f"轨迹图像: {output_dir}/trajectory_axes.png")
    print()


if __name__ == "__main__":
    main()
