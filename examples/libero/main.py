"""
LIBERO 任务评估脚本

功能：
1. 在 LIBERO 模拟环境中加载不同的任务套件
2. 连接到远程策略服务器并执行推理
3. 评估策略网络的成功率
4. 生成每个任务的执行视频

支持的任务套件：
- libero_spatial: 空间操作任务（220步）
- libero_object: 物体交互任务（280步）
- libero_goal: 目标驱动任务（300步）
- libero_10: 10个任务混合（520步）
- libero_90: 90个任务混合（400步）

使用示例：
    python examples/libero/main.py --host localhost --port 8000 --task_suite_name libero_spatial
"""

import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

# ========== 常量定义 ==========
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]  # 虚拟动作（6个0 + 1个-1），用于初始化等待阶段
LIBERO_ENV_RESOLUTION = 256  # 环境渲染分辨率，与训练数据一致


@dataclasses.dataclass
class Args:
    """命令行参数配置"""

    # ========== 策略服务器参数 ==========
    host: str = "0.0.0.0"  # WebSocket 服务器地址
    port: int = 8000  # WebSocket 服务器端口
    resize_size: int = 224  # 图像尺寸（策略模型期望的输入大小）
    replan_steps: int = 5  # 重新规划步数：每执行N步后重新调用策略推理

    # ========== LIBERO 环境参数 ==========
    task_suite_name: str = "libero_spatial"  # 任务套件名称
    # 选项: libero_spatial, libero_object, libero_goal, libero_10, libero_90

    num_steps_wait: int = 10  # 初始等待步数：让模拟环境中的物体稳定下落
    num_trials_per_task: int = 50  # 每个任务的试验次数（rollout次数）

    # ========== 输出配置 ==========
    video_out_path: str = "data/libero/videos"  # 视频输出路径
    seed: int = 7  # 随机种子（保证可重现性）


def eval_libero(args: Args) -> None:
    """
    主评估函数：在LIBERO环境中评估策略网络

    流程：
    1. 初始化LIBERO任务套件
    2. 为每个任务创建模拟环境
    3. 对每个任务执行多次试验
    4. 评估策略推理并执行动作
    5. 生成执行视频和成功率报告
    """

    # ========== 初始化随机种子 ==========
    np.random.seed(args.seed)  # 设置NumPy随机种子保证可重现性

    # ========== 初始化任务套件 ==========
    benchmark_dict = benchmark.get_benchmark_dict()  # 加载所有可用任务套件
    task_suite = benchmark_dict[args.task_suite_name]()  # 获取指定任务套件
    num_tasks_in_suite = task_suite.n_tasks  # 获取任务套件中的任务总数
    logging.info(f"Task suite: {args.task_suite_name}")

    # ========== 创建输出目录 ==========
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)  # 创建视频输出目录

    # ========== 根据任务套件设置最大步数 ==========
    # 最大步数 = 最长演示步数 + 安全余量
    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # 最长训练演示: 193步
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # 最长训练演示: 254步
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # 最长训练演示: 270步
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # 最长训练演示: 505步
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # 最长训练演示: 373步
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # ========== 连接到策略服务器 ==========
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # ========== 评估所有任务 ==========
    total_episodes, total_successes = 0, 0  # 累计统计：总试验数、成功数

    # 使用tqdm显示进度条：遍历任务套件中的每个任务
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # 获取当前任务
        task = task_suite.get_task(task_id)

        # 获取任务的初始状态集合（多个不同起点）
        initial_states = task_suite.get_task_init_states(task_id)

        # 初始化LIBERO环境和获取任务描述
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # 对当前任务执行多次试验
        task_episodes, task_successes = 0, 0  # 当前任务的试验数和成功数

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # ========== 重置环境准备新试验 ==========
            env.reset()  # 重置环境状态
            action_plan = collections.deque()  # 动作队列（用于缓存预取的多个动作）

            # ========== 设置初始状态 ==========
            obs = env.set_init_state(initial_states[episode_idx])  # 设置本次试验的初始状态

            # ========== 初始化试验变量 ==========
            t = 0  # 当前时步计数
            replay_images = []  # 存储每一步的图像用于生成视频

            logging.info(f"Starting episode {task_episodes+1}...")

            # ========== 主执行循环 ==========
            # 限制最大步数：max_steps + num_steps_wait（等待物体稳定）
            while t < max_steps + args.num_steps_wait:
                try:
                    # ========== 初始等待阶段 ==========
                    # 重要：模拟器初期会掉物体，需要等待它们落稳
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # ========== 获取和预处理图像 ==========
                    # 获取主相机图像（agentview）
                    # 重要：旋转180度以匹配训练数据的预处理
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

                    # 获取腕部相机图像（wrist camera）
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    # 预处理主相机图像：转换为uint8 + 填充resize到224x224
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )

                    # 预处理腕部相机图像
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # ========== 保存图像用于回放视频 ==========
                    replay_images.append(img)

                    # ========== 检查是否需要重新规划动作 ==========
                    if not action_plan:
                        # 动作队列为空，表示需要调用策略推理获取新的动作序列

                        # 构建观测字典（策略网络输入）
                        element = {
                            "observation/image": img,  # 主相机图像
                            "observation/wrist_image": wrist_img,  # 腕部相机图像
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],  # 末端执行器位置 (3,)
                                    _quat2axisangle(obs["robot0_eef_quat"]),  # 末端执行器方向 (3,)
                                    obs["robot0_gripper_qpos"],  # 夹爪状态 (1,)
                                )
                            ),  # 总计 (7,) 的状态向量
                            "prompt": str(task_description),  # 任务描述文本
                        }

                        # 调用策略服务器推理
                        action_chunk = client.infer(element)["actions"]  # 返回动作序列

                        # 验证策略返回足够的动作步数
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."

                        # 将前replan_steps个动作加入队列（其余动作丢弃）
                        # 这样每执行5步就重新推理一次，使策略能适应环境变化
                        action_plan.extend(action_chunk[: args.replan_steps])

                    # ========== 执行动作 ==========
                    action = action_plan.popleft()  # 从队列取出一个动作

                    # 在环境中执行动作
                    obs, reward, done, info = env.step(action.tolist())

                    # 检查任务是否成功完成
                    if done:
                        task_successes += 1  # 当前任务成功数加1
                        total_successes += 1  # 总成功数加1
                        break  # 任务完成，结束本次试验

                    t += 1  # 时步计数加1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break  # 发生异常，结束本次试验

            # ========== 试验结束，更新统计 ==========
            task_episodes += 1  # 当前任务试验数加1
            total_episodes += 1  # 总试验数加1

            # ========== 生成执行视频 ==========
            suffix = "success" if done else "failure"  # 视频标签：成功或失败
            task_segment = task_description.replace(" ", "_")  # 将空格替换为下划线

            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,  # 视频帧率10fps
            )

            # ========== 打印当前进度 ==========
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # ========== 任务完成，打印任务级别统计 ==========
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    # ========== 最终统计信息 ==========
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """
    初始化并返回LIBERO环境

    参数：
    - task: LIBERO任务对象
    - resolution: 相机分辨率（像素）
    - seed: 随机种子

    返回：
    - env: 初始化后的LIBERO环境
    - task_description: 任务的自然语言描述
    """

    task_description = task.language  # 获取任务的自然语言描述

    # 构造BDDL文件路径（BDDL是行为描述语言）
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    # 环境配置参数
    env_args = {
        "bddl_file_name": task_bddl_file,  # 任务定义文件
        "camera_heights": resolution,  # 相机高度（像素）
        "camera_widths": resolution,  # 相机宽度（像素）
    }

    # 创建离屏渲染环境（无需显示界面）
    env = OffScreenRenderEnv(**env_args)

    # 设置随机种子
    # 重要：种子会影响物体初始位置，即使使用固定的初始状态也需要设置
    env.seed(seed)

    return env, task_description


def _quat2axisangle(quat):
    """
    将四元数转换为轴角表示法 (axis-angle)

    来源：https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    参数：
    - quat: 四元数 [x, y, z, w] 形式

    返回：
    - axis_angle: 轴角向量 (3,)，方向为旋转轴，大小为旋转角度

    解释：
    轴角是一种3参数的旋转表示，优于四元数的原因：
    - 参数更少（3vs4）
    - 物理意义清晰（轴+角度）
    """

    # ========== 检查和裁剪四元数的w分量 ==========
    # w分量范围应为[-1, 1]，可能因数值误差超出范围
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    # ========== 计算旋转轴的长度（正弦） ==========
    # sin(θ/2) = sqrt(1 - cos²(θ/2)) = sqrt(1 - w²)
    den = np.sqrt(1.0 - quat[3] * quat[3])

    # ========== 处理零旋转特殊情况 ==========
    # 当 den ≈ 0 时，说明θ ≈ 0（接近零旋转），直接返回零向量
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    # ========== 四元数转轴角公式 ==========
    # axis_angle = (xyz_component * 2 * acos(w)) / sin(θ/2)
    # = (xyz_component * 2 * acos(w)) / den
    #
    # 其中：
    # - acos(w) = θ/2 （w是四元数的标量部分）
    # - xyz_component 是旋转轴的单位向量
    # - 乘以2*acos(w)得到轴角表示（轴向量*旋转角度）
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    # ========== 配置日志 ==========
    logging.basicConfig(level=logging.INFO)  # 设置日志级别为INFO

    # ========== 使用tyro解析命令行参数并运行 ==========
    # tyro会自动将命令行参数解析为Args数据类的实例
    tyro.cli(eval_libero)
