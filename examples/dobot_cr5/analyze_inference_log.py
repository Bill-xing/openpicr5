#!/usr/bin/env python3
"""
推理日志分析脚本

分析 DataLogger 记录的数据，用于定位 VLA 控制晃动问题的来源：
- 模型推理问题：动作输出跳变、不连续
- 推理客户端问题：跟踪误差大、延迟高

功能：
1. 跟踪误差分析 - 目标位置 vs 实际位置
2. 动作连续性分析 - 检测推理输出跳变
3. 按动作索引分组分析 - 每个 action_index 的误差分布
4. 推理延迟分析 - 延迟统计和时序图

输出：
- PNG 图表文件
- HTML 报告

使用方法：
python examples/dobot_cr5/analyze_inference_log.py \
    inference_logs/inference_log_xxx.h5 \
    --output_dir ./analysis_results
"""

import argparse
import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np


class InferenceLogAnalyzer:
    """推理日志分析器"""

    def __init__(self, h5_path: str, output_dir: str):
        """
        初始化分析器

        Args:
            h5_path: HDF5 文件路径
            output_dir: 输出目录
        """
        self.h5_path = Path(h5_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载数据
        self._load_data()

        # 存储分析结果
        self.results = {}
        self.figures = {}

    def _load_data(self):
        """加载 HDF5 数据"""
        print(f"加载数据: {self.h5_path}")

        with h5py.File(self.h5_path, 'r') as f:
            # 元数据
            self.metadata = {
                'prompt': f['metadata/prompt'][()].decode() if 'metadata/prompt' in f else 'N/A',
                'host': f['metadata/host'][()].decode() if 'metadata/host' in f else 'N/A',
                'port': int(f['metadata/port'][()]) if 'metadata/port' in f else 0,
                'replan_steps': int(f['metadata/replan_steps'][()]) if 'metadata/replan_steps' in f else 5,
            }

            # 推理数据
            self.inference = {
                'timestamps': f['inference/timestamps'][:] if 'inference/timestamps' in f else np.array([]),
                'steps': f['inference/steps'][:] if 'inference/steps' in f else np.array([]),
                'obs_states': f['inference/obs_states'][:] if 'inference/obs_states' in f else np.array([]),
                'actions': f['inference/actions'][:] if 'inference/actions' in f else np.array([]),
                'latencies_ms': f['inference/latencies_ms'][:] if 'inference/latencies_ms' in f else np.array([]),
            }

            # 命令数据
            self.commands = {
                'timestamps': f['commands/timestamps'][:] if 'commands/timestamps' in f else np.array([]),
                'target_poses': f['commands/target_poses'][:] if 'commands/target_poses' in f else np.array([]),
                'gripper_targets': f['commands/gripper_targets'][:] if 'commands/gripper_targets' in f else np.array([]),
                'action_indices': f['commands/action_indices'][:] if 'commands/action_indices' in f else np.array([]),
                'inference_steps': f['commands/inference_steps'][:] if 'commands/inference_steps' in f else np.array([]),
            }

            # 机械臂状态数据
            self.robot_states = {
                'timestamps': f['robot_states/timestamps'][:] if 'robot_states/timestamps' in f else np.array([]),
                'current_poses': f['robot_states/current_poses'][:] if 'robot_states/current_poses' in f else np.array([]),
                'gripper_states': f['robot_states/gripper_states'][:] if 'robot_states/gripper_states' in f else np.array([]),
                'pose_errors': f['robot_states/pose_errors'][:] if 'robot_states/pose_errors' in f else np.array([]),
                'frame_intervals_ms': f['robot_states/frame_intervals_ms'][:] if 'robot_states/frame_intervals_ms' in f else np.array([]),
            }

        print(f"  推理次数: {len(self.inference['timestamps'])}")
        print(f"  命令次数: {len(self.commands['timestamps'])}")
        print(f"  状态次数: {len(self.robot_states['timestamps'])}")

    def analyze_tracking_error(self):
        """分析跟踪误差"""
        print("\n=== 跟踪误差分析 ===")

        pose_errors = self.robot_states['pose_errors']
        if len(pose_errors) == 0:
            print("  无数据")
            return

        # 位置误差 (x, y, z) 单位: mm
        pos_errors = pose_errors[:, :3]
        # 姿态误差 (rx, ry, rz) 单位: 度
        rot_errors = pose_errors[:, 3:6]

        # 统计
        pos_stats = {
            'mean': np.mean(np.abs(pos_errors), axis=0),
            'std': np.std(pos_errors, axis=0),
            'max': np.max(np.abs(pos_errors), axis=0),
            'p95': np.percentile(np.abs(pos_errors), 95, axis=0),
        }
        rot_stats = {
            'mean': np.mean(np.abs(rot_errors), axis=0),
            'std': np.std(rot_errors, axis=0),
            'max': np.max(np.abs(rot_errors), axis=0),
            'p95': np.percentile(np.abs(rot_errors), 95, axis=0),
        }

        self.results['tracking_error'] = {
            'position': pos_stats,
            'rotation': rot_stats,
        }

        # 打印统计
        print("  位置误差 (mm):")
        print(f"    均值: x={pos_stats['mean'][0]:.2f}, y={pos_stats['mean'][1]:.2f}, z={pos_stats['mean'][2]:.2f}")
        print(f"    标准差: x={pos_stats['std'][0]:.2f}, y={pos_stats['std'][1]:.2f}, z={pos_stats['std'][2]:.2f}")
        print(f"    最大值: x={pos_stats['max'][0]:.2f}, y={pos_stats['max'][1]:.2f}, z={pos_stats['max'][2]:.2f}")
        print(f"    P95: x={pos_stats['p95'][0]:.2f}, y={pos_stats['p95'][1]:.2f}, z={pos_stats['p95'][2]:.2f}")

        print("  姿态误差 (度):")
        print(f"    均值: rx={rot_stats['mean'][0]:.2f}, ry={rot_stats['mean'][1]:.2f}, rz={rot_stats['mean'][2]:.2f}")
        print(f"    最大值: rx={rot_stats['max'][0]:.2f}, ry={rot_stats['max'][1]:.2f}, rz={rot_stats['max'][2]:.2f}")

        # 绘制位置误差时序图
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # 位置误差
        ax = axes[0]
        steps = np.arange(len(pos_errors))
        ax.plot(steps, pos_errors[:, 0], label='X', alpha=0.8)
        ax.plot(steps, pos_errors[:, 1], label='Y', alpha=0.8)
        ax.plot(steps, pos_errors[:, 2], label='Z', alpha=0.8)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Step')
        ax.set_ylabel('Error (mm)')
        ax.set_title('Position Tracking Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 姿态误差
        ax = axes[1]
        ax.plot(steps, rot_errors[:, 0], label='RX', alpha=0.8)
        ax.plot(steps, rot_errors[:, 1], label='RY', alpha=0.8)
        ax.plot(steps, rot_errors[:, 2], label='RZ', alpha=0.8)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Step')
        ax.set_ylabel('Error (deg)')
        ax.set_title('Rotation Tracking Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = self.output_dir / 'tracking_error.png'
        fig.savefig(fig_path, dpi=150)
        self.figures['tracking_error'] = fig
        print(f"  图表已保存: {fig_path}")

    def analyze_action_continuity(self):
        """分析动作连续性（检测推理输出跳变）"""
        print("\n=== 动作连续性分析 ===")

        actions = self.inference['actions']
        if len(actions) < 2:
            print("  推理次数不足，无法分析")
            return

        # 计算相邻推理之间的动作跳变
        # 比较上一次推理的最后一个动作与这次推理的第一个动作
        jumps = []
        jump_indices = []

        for i in range(1, len(actions)):
            prev_last_action = actions[i-1, -1, :]  # 上次最后一个动作
            curr_first_action = actions[i, 0, :]   # 这次第一个动作
            diff = np.abs(curr_first_action - prev_last_action)
            jumps.append(diff)

            # 检测显著跳变（位置 > 5mm 或 姿态 > 2度）
            if np.max(diff[:3]) > 5 or np.max(diff[3:6]) > 2:
                jump_indices.append(i)

        jumps = np.array(jumps)

        # 统计
        jump_stats = {
            'mean': np.mean(jumps, axis=0),
            'max': np.max(jumps, axis=0),
            'std': np.std(jumps, axis=0),
            'num_large_jumps': len(jump_indices),
            'large_jump_indices': jump_indices,
        }

        self.results['action_continuity'] = jump_stats

        print(f"  动作跳变统计 (相邻推理之间):")
        print(f"    位置跳变均值: x={jump_stats['mean'][0]:.2f}, y={jump_stats['mean'][1]:.2f}, z={jump_stats['mean'][2]:.2f} mm")
        print(f"    位置跳变最大: x={jump_stats['max'][0]:.2f}, y={jump_stats['max'][1]:.2f}, z={jump_stats['max'][2]:.2f} mm")
        print(f"    显著跳变次数: {len(jump_indices)} (位置>5mm 或 姿态>2度)")

        if jump_indices:
            print(f"    跳变发生在推理步: {jump_indices[:10]}{'...' if len(jump_indices) > 10 else ''}")

        # 绘制跳变图
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # 位置跳变
        ax = axes[0]
        inference_steps = np.arange(1, len(actions))
        ax.plot(inference_steps, jumps[:, 0], label='X', alpha=0.8)
        ax.plot(inference_steps, jumps[:, 1], label='Y', alpha=0.8)
        ax.plot(inference_steps, jumps[:, 2], label='Z', alpha=0.8)
        ax.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Threshold (5mm)')

        # 标记显著跳变
        for idx in jump_indices:
            ax.axvline(x=idx, color='r', alpha=0.3, linewidth=0.5)

        ax.set_xlabel('Inference Step')
        ax.set_ylabel('Jump (mm)')
        ax.set_title('Action Position Jump Between Consecutive Inferences')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 夹爪跳变
        ax = axes[1]
        ax.plot(inference_steps, jumps[:, 6], label='Gripper', alpha=0.8, color='purple')
        ax.set_xlabel('Inference Step')
        ax.set_ylabel('Jump')
        ax.set_title('Gripper Action Jump Between Consecutive Inferences')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = self.output_dir / 'action_continuity.png'
        fig.savefig(fig_path, dpi=150)
        self.figures['action_continuity'] = fig
        print(f"  图表已保存: {fig_path}")

    def analyze_by_action_index(self):
        """按动作索引分组分析误差"""
        print("\n=== 按动作索引分组分析 ===")

        action_indices = self.commands['action_indices']
        pose_errors = self.robot_states['pose_errors']

        if len(action_indices) == 0 or len(pose_errors) == 0:
            print("  无数据")
            return

        # 确保长度一致
        min_len = min(len(action_indices), len(pose_errors))
        action_indices = action_indices[:min_len]
        pose_errors = pose_errors[:min_len]

        # 按 action_index 分组
        unique_indices = np.unique(action_indices)
        index_stats = {}

        print("  各动作索引的位置误差 (mm):")
        for idx in unique_indices:
            mask = action_indices == idx
            errors = pose_errors[mask, :3]  # 只取位置误差

            stats = {
                'count': int(np.sum(mask)),
                'mean': np.mean(np.abs(errors), axis=0),
                'std': np.std(errors, axis=0),
                'max': np.max(np.abs(errors), axis=0),
            }
            index_stats[int(idx)] = stats

            total_mean = np.mean(stats['mean'])
            print(f"    Index {idx}: count={stats['count']}, "
                  f"mean={total_mean:.2f}mm "
                  f"(x={stats['mean'][0]:.2f}, y={stats['mean'][1]:.2f}, z={stats['mean'][2]:.2f})")

        self.results['by_action_index'] = index_stats

        # 绘制分组误差箱线图
        # 根据数据量动态调整图表宽度
        num_indices = len(unique_indices)
        fig_width = max(12, num_indices * 0.5)  # 每个索引至少0.5英寸宽度
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        # 准备箱线图数据
        box_data = []
        labels = []
        for idx in sorted(unique_indices):
            mask = action_indices == idx
            errors = np.linalg.norm(pose_errors[mask, :3], axis=1)  # 位置误差范数
            box_data.append(errors)
            labels.append(f'{int(idx)}')  # 只显示数字，去掉"Index"前缀

        bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True)

        # 设置颜色
        colors = plt.cm.viridis(np.linspace(0, 1, len(box_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel('Action Index', fontsize=12)
        ax.set_ylabel('Position Error Norm (mm)', fontsize=12)
        ax.set_title('Position Error Distribution by Action Index', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        # 旋转X轴标签，避免重叠
        plt.xticks(rotation=45, ha='right', fontsize=10)

        plt.tight_layout()
        fig_path = self.output_dir / 'error_by_action_index.png'
        fig.savefig(fig_path, dpi=150)
        self.figures['by_action_index'] = fig
        print(f"  图表已保存: {fig_path}")

    def analyze_inference_latency(self):
        """分析推理延迟"""
        print("\n=== 推理延迟分析 ===")

        latencies = self.inference['latencies_ms']
        if len(latencies) == 0:
            print("  无数据")
            return

        # 统计
        latency_stats = {
            'mean': float(np.mean(latencies)),
            'std': float(np.std(latencies)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies)),
            'p50': float(np.percentile(latencies, 50)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
        }

        self.results['inference_latency'] = latency_stats

        print(f"  推理延迟统计 (ms):")
        print(f"    均值: {latency_stats['mean']:.1f}")
        print(f"    标准差: {latency_stats['std']:.1f}")
        print(f"    最小值: {latency_stats['min']:.1f}")
        print(f"    最大值: {latency_stats['max']:.1f}")
        print(f"    P50: {latency_stats['p50']:.1f}")
        print(f"    P95: {latency_stats['p95']:.1f}")
        print(f"    P99: {latency_stats['p99']:.1f}")

        # 绘制延迟图
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # 时序图
        ax = axes[0]
        inference_steps = np.arange(len(latencies))
        ax.plot(inference_steps, latencies, alpha=0.8)
        ax.axhline(y=latency_stats['mean'], color='r', linestyle='--',
                   label=f'Mean ({latency_stats["mean"]:.1f}ms)', alpha=0.7)
        ax.axhline(y=latency_stats['p95'], color='orange', linestyle='--',
                   label=f'P95 ({latency_stats["p95"]:.1f}ms)', alpha=0.7)
        ax.set_xlabel('Inference Step')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Inference Latency Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 直方图
        ax = axes[1]
        ax.hist(latencies, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=latency_stats['mean'], color='r', linestyle='--',
                   label=f'Mean ({latency_stats["mean"]:.1f}ms)')
        ax.axvline(x=latency_stats['p95'], color='orange', linestyle='--',
                   label=f'P95 ({latency_stats["p95"]:.1f}ms)')
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Count')
        ax.set_title('Inference Latency Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = self.output_dir / 'inference_latency.png'
        fig.savefig(fig_path, dpi=150)
        self.figures['inference_latency'] = fig
        print(f"  图表已保存: {fig_path}")

    def analyze_frame_rate(self):
        """分析帧率/帧间隔"""
        print("\n=== 帧率分析 ===")

        frame_intervals = self.robot_states['frame_intervals_ms']
        if len(frame_intervals) == 0:
            print("  无帧间隔数据（旧版本记录不包含此数据）")
            return

        # 计算帧率
        frame_rates = 1000.0 / frame_intervals  # Hz

        # 统计
        frame_stats = {
            'interval_mean': float(np.mean(frame_intervals)),
            'interval_std': float(np.std(frame_intervals)),
            'interval_min': float(np.min(frame_intervals)),
            'interval_max': float(np.max(frame_intervals)),
            'rate_mean': float(np.mean(frame_rates)),
            'rate_min': float(np.min(frame_rates)),
            'rate_max': float(np.max(frame_rates)),
            'below_25hz_count': int(np.sum(frame_rates < 25)),
            'below_25hz_pct': float(np.mean(frame_rates < 25) * 100),
        }

        self.results['frame_rate'] = frame_stats

        print(f"  帧间隔统计 (ms):")
        print(f"    均值: {frame_stats['interval_mean']:.1f} (期望33.3)")
        print(f"    标准差: {frame_stats['interval_std']:.1f}")
        print(f"    范围: {frame_stats['interval_min']:.1f} ~ {frame_stats['interval_max']:.1f}")
        print(f"  帧率统计 (Hz):")
        print(f"    均值: {frame_stats['rate_mean']:.1f} (期望30)")
        print(f"    范围: {frame_stats['rate_min']:.1f} ~ {frame_stats['rate_max']:.1f}")
        print(f"    低于25Hz的比例: {frame_stats['below_25hz_pct']:.1f}% ({frame_stats['below_25hz_count']}帧)")

        # 分析帧率与误差的相关性
        pose_errors = self.robot_states['pose_errors']
        if len(pose_errors) == len(frame_intervals):
            error_norms = np.linalg.norm(pose_errors[:, :3], axis=1)
            correlation = np.corrcoef(frame_intervals, error_norms)[0, 1]
            frame_stats['error_correlation'] = float(correlation)
            print(f"  帧间隔与位置误差的相关系数: {correlation:.3f}")

        # 绘制帧率图
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        steps = np.arange(len(frame_intervals))

        # 帧间隔时序图
        ax = axes[0]
        ax.plot(steps, frame_intervals, alpha=0.8)
        ax.axhline(y=33.3, color='g', linestyle='--', label='Target (33.3ms = 30Hz)', alpha=0.7)
        ax.axhline(y=40, color='orange', linestyle='--', label='25Hz threshold (40ms)', alpha=0.7)
        ax.axhline(y=50, color='r', linestyle='--', label='20Hz threshold (50ms)', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Frame Interval (ms)')
        ax.set_title('Frame Interval Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 帧率时序图
        ax = axes[1]
        ax.plot(steps, frame_rates, alpha=0.8)
        ax.axhline(y=30, color='g', linestyle='--', label='Target (30Hz)', alpha=0.7)
        ax.axhline(y=25, color='orange', linestyle='--', label='25Hz threshold', alpha=0.7)
        ax.axhline(y=20, color='r', linestyle='--', label='20Hz threshold', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Frame Rate (Hz)')
        ax.set_title('Frame Rate Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 帧间隔直方图
        ax = axes[2]
        ax.hist(frame_intervals, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=33.3, color='g', linestyle='--', label='Target (33.3ms)')
        ax.axvline(x=frame_stats['interval_mean'], color='r', linestyle='--',
                   label=f'Mean ({frame_stats["interval_mean"]:.1f}ms)')
        ax.set_xlabel('Frame Interval (ms)')
        ax.set_ylabel('Count')
        ax.set_title('Frame Interval Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = self.output_dir / 'frame_rate.png'
        fig.savefig(fig_path, dpi=150)
        self.figures['frame_rate'] = fig
        print(f"  图表已保存: {fig_path}")

    def analyze_trajectory(self):
        """分析目标轨迹 vs 实际轨迹"""
        print("\n=== 轨迹分析 ===")

        target_poses = self.commands['target_poses']
        current_poses = self.robot_states['current_poses']

        if len(target_poses) == 0 or len(current_poses) == 0:
            print("  无数据")
            return

        # 确保长度一致
        min_len = min(len(target_poses), len(current_poses))
        target_poses = target_poses[:min_len]
        current_poses = current_poses[:min_len]

        # 只绘制各轴对比图
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        steps = np.arange(min_len)
        axis_labels = ['X', 'Y', 'Z']

        for i, (ax, label) in enumerate(zip(axes, axis_labels)):
            ax.plot(steps, target_poses[:, i], 'b-', label='Target', alpha=0.8)
            ax.plot(steps, current_poses[:, i], 'r-', label='Actual', alpha=0.8)
            ax.set_xlabel('Step')
            ax.set_ylabel(f'{label} (mm)')
            ax.set_title(f'{label} Position: Target vs Actual')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = self.output_dir / 'trajectory_axes.png'
        fig.savefig(fig_path, dpi=150)
        self.figures['trajectory_axes'] = fig
        print(f"  图表已保存: {fig_path}")

    def generate_html_report(self):
        """生成 HTML 报告"""
        print("\n=== 生成 HTML 报告 ===")

        # 将图表转为 base64
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')

        # 构建 HTML
        html_parts = []
        html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>推理日志分析报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .metadata { background: #e8f5e9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .stats-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        .stats-table th, .stats-table td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        .stats-table th { background: #4CAF50; color: white; }
        .stats-table tr:nth-child(even) { background: #f9f9f9; }
        .figure { text-align: center; margin: 20px 0; }
        .figure img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }
        .warning { color: #f44336; font-weight: bold; }
        .good { color: #4CAF50; font-weight: bold; }
        .summary { background: #fff3e0; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
<div class="container">
""")

        # 标题和元数据
        html_parts.append(f"""
<h1>推理日志分析报告</h1>
<div class="metadata">
    <strong>文件:</strong> {self.h5_path.name}<br>
    <strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
    <strong>任务提示:</strong> {self.metadata['prompt']}<br>
    <strong>服务器:</strong> {self.metadata['host']}:{self.metadata['port']}<br>
    <strong>重规划步数:</strong> {self.metadata['replan_steps']}<br>
    <strong>推理次数:</strong> {len(self.inference['timestamps'])}<br>
    <strong>执行步数:</strong> {len(self.commands['timestamps'])}
</div>
""")

        # 跟踪误差
        if 'tracking_error' in self.results:
            pos = self.results['tracking_error']['position']
            rot = self.results['tracking_error']['rotation']

            # 判断误差是否过大
            pos_mean_norm = np.linalg.norm(pos['mean'])
            status = '正常' if pos_mean_norm < 5 else '偏大'
            status_class = 'good' if pos_mean_norm < 5 else 'warning'

            html_parts.append(f"""
<h2>1. 跟踪误差分析</h2>
<p>状态: <span class="{status_class}">{status}</span> (位置误差均值: {pos_mean_norm:.2f}mm)</p>
<table class="stats-table">
    <tr><th>指标</th><th>X (mm)</th><th>Y (mm)</th><th>Z (mm)</th></tr>
    <tr><td>均值</td><td>{pos['mean'][0]:.2f}</td><td>{pos['mean'][1]:.2f}</td><td>{pos['mean'][2]:.2f}</td></tr>
    <tr><td>标准差</td><td>{pos['std'][0]:.2f}</td><td>{pos['std'][1]:.2f}</td><td>{pos['std'][2]:.2f}</td></tr>
    <tr><td>最大值</td><td>{pos['max'][0]:.2f}</td><td>{pos['max'][1]:.2f}</td><td>{pos['max'][2]:.2f}</td></tr>
    <tr><td>P95</td><td>{pos['p95'][0]:.2f}</td><td>{pos['p95'][1]:.2f}</td><td>{pos['p95'][2]:.2f}</td></tr>
</table>
""")

            if 'tracking_error' in self.figures:
                img_data = fig_to_base64(self.figures['tracking_error'])
                html_parts.append(f'<div class="figure"><img src="data:image/png;base64,{img_data}"></div>')

        # 动作连续性
        if 'action_continuity' in self.results:
            stats = self.results['action_continuity']
            num_jumps = stats['num_large_jumps']
            status = '良好' if num_jumps < 5 else '存在问题'
            status_class = 'good' if num_jumps < 5 else 'warning'

            html_parts.append(f"""
<h2>2. 动作连续性分析</h2>
<p>状态: <span class="{status_class}">{status}</span> (显著跳变次数: {num_jumps})</p>
<table class="stats-table">
    <tr><th>指标</th><th>X (mm)</th><th>Y (mm)</th><th>Z (mm)</th></tr>
    <tr><td>跳变均值</td><td>{stats['mean'][0]:.2f}</td><td>{stats['mean'][1]:.2f}</td><td>{stats['mean'][2]:.2f}</td></tr>
    <tr><td>跳变最大</td><td>{stats['max'][0]:.2f}</td><td>{stats['max'][1]:.2f}</td><td>{stats['max'][2]:.2f}</td></tr>
</table>
""")
            if num_jumps > 0:
                jump_indices = stats['large_jump_indices'][:10]
                html_parts.append(f'<p>跳变发生在推理步: {jump_indices}{"..." if len(stats["large_jump_indices"]) > 10 else ""}</p>')

            if 'action_continuity' in self.figures:
                img_data = fig_to_base64(self.figures['action_continuity'])
                html_parts.append(f'<div class="figure"><img src="data:image/png;base64,{img_data}"></div>')

        # 按动作索引分组
        if 'by_action_index' in self.results:
            html_parts.append('<h2>3. 按动作索引分组分析</h2>')
            html_parts.append('<table class="stats-table">')
            html_parts.append('<tr><th>Index</th><th>Count</th><th>Mean X</th><th>Mean Y</th><th>Mean Z</th><th>Total Mean</th></tr>')

            for idx in sorted(self.results['by_action_index'].keys()):
                stats = self.results['by_action_index'][idx]
                total_mean = np.mean(stats['mean'])
                html_parts.append(f'<tr><td>{idx}</td><td>{stats["count"]}</td>'
                                  f'<td>{stats["mean"][0]:.2f}</td><td>{stats["mean"][1]:.2f}</td>'
                                  f'<td>{stats["mean"][2]:.2f}</td><td>{total_mean:.2f}</td></tr>')
            html_parts.append('</table>')

            if 'by_action_index' in self.figures:
                img_data = fig_to_base64(self.figures['by_action_index'])
                html_parts.append(f'<div class="figure"><img src="data:image/png;base64,{img_data}"></div>')

        # 推理延迟
        if 'inference_latency' in self.results:
            stats = self.results['inference_latency']
            status = '正常' if stats['p95'] < 100 else '偏高'
            status_class = 'good' if stats['p95'] < 100 else 'warning'

            html_parts.append(f"""
<h2>4. 推理延迟分析</h2>
<p>状态: <span class="{status_class}">{status}</span> (P95: {stats['p95']:.1f}ms)</p>
<table class="stats-table">
    <tr><th>指标</th><th>值 (ms)</th></tr>
    <tr><td>均值</td><td>{stats['mean']:.1f}</td></tr>
    <tr><td>标准差</td><td>{stats['std']:.1f}</td></tr>
    <tr><td>最小值</td><td>{stats['min']:.1f}</td></tr>
    <tr><td>最大值</td><td>{stats['max']:.1f}</td></tr>
    <tr><td>P50</td><td>{stats['p50']:.1f}</td></tr>
    <tr><td>P95</td><td>{stats['p95']:.1f}</td></tr>
    <tr><td>P99</td><td>{stats['p99']:.1f}</td></tr>
</table>
""")

            if 'inference_latency' in self.figures:
                img_data = fig_to_base64(self.figures['inference_latency'])
                html_parts.append(f'<div class="figure"><img src="data:image/png;base64,{img_data}"></div>')

        # 帧率分析
        if 'frame_rate' in self.results:
            stats = self.results['frame_rate']
            status = '正常' if stats['below_25hz_pct'] < 10 else '偏低'
            status_class = 'good' if stats['below_25hz_pct'] < 10 else 'warning'

            html_parts.append(f"""
<h2>5. 帧率分析</h2>
<p>状态: <span class="{status_class}">{status}</span> (低于25Hz比例: {stats['below_25hz_pct']:.1f}%)</p>
<table class="stats-table">
    <tr><th>指标</th><th>值</th></tr>
    <tr><td>帧间隔均值</td><td>{stats['interval_mean']:.1f} ms (期望33.3)</td></tr>
    <tr><td>帧间隔标准差</td><td>{stats['interval_std']:.1f} ms</td></tr>
    <tr><td>帧间隔范围</td><td>{stats['interval_min']:.1f} ~ {stats['interval_max']:.1f} ms</td></tr>
    <tr><td>帧率均值</td><td>{stats['rate_mean']:.1f} Hz (期望30)</td></tr>
    <tr><td>帧率范围</td><td>{stats['rate_min']:.1f} ~ {stats['rate_max']:.1f} Hz</td></tr>
    <tr><td>低于25Hz帧数</td><td>{stats['below_25hz_count']} ({stats['below_25hz_pct']:.1f}%)</td></tr>
</table>
""")
            if 'error_correlation' in stats:
                corr = stats['error_correlation']
                corr_status = '强相关' if abs(corr) > 0.5 else ('弱相关' if abs(corr) > 0.2 else '无明显相关')
                html_parts.append(f'<p>帧间隔与位置误差的相关系数: {corr:.3f} ({corr_status})</p>')

            if 'frame_rate' in self.figures:
                img_data = fig_to_base64(self.figures['frame_rate'])
                html_parts.append(f'<div class="figure"><img src="data:image/png;base64,{img_data}"></div>')

        # 轨迹图
        if 'trajectory_3d' in self.figures:
            html_parts.append('<h2>6. 轨迹对比</h2>')
            img_data = fig_to_base64(self.figures['trajectory_3d'])
            html_parts.append(f'<div class="figure"><img src="data:image/png;base64,{img_data}"></div>')

        if 'trajectory_axes' in self.figures:
            img_data = fig_to_base64(self.figures['trajectory_axes'])
            html_parts.append(f'<div class="figure"><img src="data:image/png;base64,{img_data}"></div>')

        # 总结
        html_parts.append("""
<div class="summary">
    <h2>分析总结</h2>
    <p><strong>晃动问题定位建议:</strong></p>
    <ul>
        <li>如果<strong>动作连续性分析</strong>显示大量跳变，问题可能在<strong>模型推理</strong>层面</li>
        <li>如果<strong>跟踪误差</strong>大但动作连续，问题可能在<strong>机械臂控制</strong>层面</li>
        <li>如果某个 <strong>action_index</strong> 误差明显大于其他，可能是重规划时机问题</li>
        <li>如果<strong>推理延迟</strong>波动大，可能影响实时控制性能</li>
        <li>如果<strong>帧率</strong>经常低于25Hz，会导致动作执行时间不匹配，加剧晃动</li>
    </ul>
</div>
""")

        html_parts.append('</div></body></html>')

        # 保存 HTML
        html_path = self.output_dir / 'report.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_parts))

        print(f"  HTML 报告已保存: {html_path}")

    def run_all_analysis(self):
        """运行所有分析"""
        # 只生成轨迹图
        self.analyze_trajectory()

        print(f"\n=== 分析完成 ===")
        print(f"输出目录: {self.output_dir}")

        # 关闭所有图表
        plt.close('all')


def main():
    parser = argparse.ArgumentParser(description='分析推理日志')
    parser.add_argument('h5_file', type=str, help='HDF5 文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认: 根据输入文件名自动生成)')

    args = parser.parse_args()

    # 如果没有指定输出目录，根据输入文件名自动生成
    if args.output_dir is None:
        # 从文件名中提取目录名
        # 支持两种格式：
        # 1. 旧格式: inference_log_20260201_205009.h5 -> 20260201_205009
        # 2. 新格式: inference_log_20260202_040122_bsT_od10_rp3_ms600.h5 -> 20260202_040122_bsT_od10_rp3_ms600
        h5_filename = Path(args.h5_file).stem  # 去掉扩展名

        # 去掉 inference_log_ 前缀，保留所有参数
        if h5_filename.startswith('inference_log_'):
            output_name = h5_filename.replace('inference_log_', '')
        else:
            output_name = h5_filename

        output_dir = f'/home/hit/openpi/analysis_results/{output_name}'
    else:
        output_dir = args.output_dir

    analyzer = InferenceLogAnalyzer(args.h5_file, output_dir)
    analyzer.run_all_analysis()


if __name__ == '__main__':
    main()
