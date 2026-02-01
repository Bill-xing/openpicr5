#!/usr/bin/env python3
"""
CR5 真机推理客户端

功能概述：
连接到远程策略服务器，订阅ROS2话题获取图像和机器人状态，
以30Hz频率执行策略推理，控制Dobot机械臂和夹爪执行动作。

主要组件：
1. DobotRosWrapper: ROS接口封装，提供服务客户端和话题发布器
2. GripperComm: 夹爪Modbus通信管理，控制夹爪开合
3. GripperStateFeedback: 夹爪状态反馈线程，使用时间插补将10Hz采样提升到100Hz发布
4. InferenceClient: 主推理类，连接策略服务器并执行推理循环

技术特点：
- 高频控制：使用ServoP伺服模式，30Hz控制频率（与训练数据一致）
- 时间插补：夹爪状态使用线性插补，将低频Modbus读取平滑到高频发布
- 异步非阻塞：ServoP和夹爪控制使用异步调用，避免阻塞推理循环

ROS话题：
- 订阅 /camera/color/image_raw: RGB图像（30Hz）
- 订阅 /dobot_msgs_v3/msg/ToolVectorActual: 机器人当前位姿
- 发布 /robot/target_pose: 机械臂目标姿态
- 发布 /gripper/state_feedback: 夹爪实际位置（100Hz插补）
- 发布 /gripper/command_update: 夹爪目标位置（100Hz插补）

使用方法：
python examples/dobot_cr5/main.py --host localhost --port 8000 --prompt "pick up the red block"
"""

import dataclasses
from datetime import datetime
import logging
import pathlib
import re
import threading
import time
from collections import deque

import cv2
import h5py
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

# ROS消息类型
from dobot_msgs_v3.msg import ToolVectorActual
from dobot_msgs_v3.srv import (
    ClearError,
    EnableRobot,
    GetHoldRegs,
    GetPose,
    ModbusClose,
    ModbusCreate,
    ServoP,
    SetHoldRegs,
    SpeedFactor,
)
from geometry_msgs.msg import PointStamped
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from sensor_msgs.msg import Image
import tyro


@dataclasses.dataclass
class Args:
    """CR5推理客户端命令行参数配置"""

    # 策略服务器参数
    host: str = "localhost"  # 策略服务器地址
    port: int = 8000  # 策略服务器端口
    resize_size: int = 224  # 图像尺寸（策略模型期望的图像大小）
    replan_steps: int = 5  # 重新规划步数：每执行N步动作后重新调用策略推理

    # 任务参数
    prompt: str = "pick up the object"  # 任务描述

    # 执行参数
    max_steps: int = 500  # 最大执行步数
    dry_run: bool = False  # 空运行模式：只打印动作，不执行
    blocking_servo: bool = True  # 阻塞执行模式：等待 ServoP 服务调用完成

    # 数据记录参数
    record: bool = False  # 是否启用数据记录
    record_dir: str = "./inference_logs"  # 记录数据保存目录
    record_images: bool = False  # 是否记录图像（禁用可显著提升性能）


class DataLogger:
    """
    数据记录器（优化版）

    记录推理过程中的三类数据：
    1. 推理结果：模型输出的动作序列和输入图像
    2. 发送命令：实际发送给机械臂的 ServoP 命令
    3. 机械臂状态：机械臂实时位姿反馈

    用于分析 VLA 控制晃动问题，定位是模型推理问题还是推理客户端问题。

    优化措施：
    - 使用 Python 原生类型存储，仅在保存时转换为 numpy
    - 使用预分配列表减少内存分配
    - 可选禁用图像记录以减少开销
    """

    def __init__(self, record_dir: str, args: Args):
        """
        初始化数据记录器

        Args:
            record_dir: 记录数据保存目录
            args: 命令行参数配置
        """
        self.record_dir = pathlib.Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.args = args
        self.record_images = args.record_images  # 是否记录图像

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = self.record_dir / f"inference_log_{timestamp}.h5"

        # 推理记录缓冲区（低频，约 6Hz）
        self.inference_data = {
            "timestamps": [],
            "steps": [],
            "obs_images": [],  # 仅当 record_images=True 时记录
            "obs_states": [],
            "actions": [],
            "latencies_ms": [],
        }

        # 命令记录缓冲区（30Hz）- 使用 Python 原生类型
        self.command_data = {
            "timestamps": [],
            "target_poses": [],  # 存储为 tuple，保存时转换
            "gripper_targets": [],
            "action_indices": [],
            "inference_steps": [],
        }

        # 机械臂状态记录缓冲区（30Hz）- 使用 Python 原生类型
        self.robot_state_data = {
            "timestamps": [],
            "current_poses": [],  # 存储为 tuple，保存时转换
            "gripper_states": [],
            "pose_errors": [],    # 存储为 tuple，保存时转换
            "frame_intervals_ms": [],
        }

        # 推理计数器
        self.inference_count = 0
        self.last_image_timestamp = None  # 用于计算帧间隔（使用图像时间戳）

        logging.info(f"[DataLogger] 初始化完成, 保存路径: {self.filename}")
        logging.info(f"[DataLogger] 图像记录: {'启用' if self.record_images else '禁用'}")

    def log_inference(
        self,
        step: int,
        obs_image: np.ndarray,
        obs_state: np.ndarray,
        actions: np.ndarray,
        latency_ms: float,
    ):
        """
        记录推理调用

        Args:
            step: 当前执行步数
            obs_image: 输入图像 (224, 224, 3)
            obs_state: 输入状态 (7,)
            actions: 推理输出动作 (horizon, 7)
            latency_ms: 推理延迟 (毫秒)
        """
        self.inference_data["timestamps"].append(time.time())
        self.inference_data["steps"].append(step)
        # 仅当启用图像记录时才复制图像
        if self.record_images:
            self.inference_data["obs_images"].append(obs_image.copy())
        self.inference_data["obs_states"].append(obs_state.copy())
        self.inference_data["actions"].append(actions.copy())
        self.inference_data["latencies_ms"].append(latency_ms)
        self.inference_count += 1

    def log_command(
        self,
        target_pose: np.ndarray,
        gripper_target: float,
        action_index: int,
        inference_step: int,
    ):
        """
        记录发送给机械臂的命令（优化版：避免 numpy 数组创建）

        Args:
            target_pose: 目标位姿 (6,) [x,y,z,rx,ry,rz]
            gripper_target: 夹爪目标位置 [0-1000]
            action_index: 当前执行的是第几个动作 (0~replan_steps-1)
            inference_step: 该动作来自哪次推理
        """
        self.command_data["timestamps"].append(time.time())
        # 使用 tuple 存储，避免创建 numpy 数组
        self.command_data["target_poses"].append(tuple(target_pose))
        self.command_data["gripper_targets"].append(float(gripper_target))
        self.command_data["action_indices"].append(int(action_index))
        self.command_data["inference_steps"].append(int(inference_step))

    def log_robot_state(
        self,
        current_pose: np.ndarray,
        gripper_state: float,
        target_pose: np.ndarray,
        image_timestamp: float,
    ):
        """
        记录机械臂实时状态（优化版：避免 numpy 数组创建）

        Args:
            current_pose: 当前位姿 (6,) [x,y,z,rx,ry,rz]
            gripper_state: 夹爪当前位置
            target_pose: 目标位姿 (6,)，用于计算跟踪误差
            image_timestamp: 图像时间戳，用于计算帧间隔
        """
        current_time = time.time()
        self.robot_state_data["timestamps"].append(current_time)
        # 使用 tuple 存储，避免创建 numpy 数组
        self.robot_state_data["current_poses"].append(tuple(current_pose))
        self.robot_state_data["gripper_states"].append(float(gripper_state))

        # 计算跟踪误差（使用列表推导式，避免 numpy 开销）
        pose_error = tuple(float(t - c) for t, c in zip(target_pose, current_pose))
        self.robot_state_data["pose_errors"].append(pose_error)

        # 使用图像时间戳计算帧间隔（反映真实的图像到达间隔）
        if self.last_image_timestamp is not None:
            frame_interval_ms = (image_timestamp - self.last_image_timestamp) * 1000
        else:
            frame_interval_ms = 33.3  # 默认30Hz
        self.robot_state_data["frame_intervals_ms"].append(frame_interval_ms)
        self.last_image_timestamp = image_timestamp

    def save(self):
        """保存数据到 HDF5 文件"""
        if not self.inference_data["timestamps"]:
            logging.warning("[DataLogger] 没有数据可保存")
            return

        logging.info(f"[DataLogger] 保存数据到 {self.filename}")
        logging.info(f"  - 推理次数: {len(self.inference_data['timestamps'])}")
        logging.info(f"  - 命令次数: {len(self.command_data['timestamps'])}")
        logging.info(f"  - 状态次数: {len(self.robot_state_data['timestamps'])}")

        with h5py.File(self.filename, "w") as f:
            # 保存元数据
            meta = f.create_group("metadata")
            meta.create_dataset("prompt", data=self.args.prompt)
            meta.create_dataset("host", data=self.args.host)
            meta.create_dataset("port", data=self.args.port)
            meta.create_dataset("replan_steps", data=self.args.replan_steps)
            meta.create_dataset("resize_size", data=self.args.resize_size)

            # 保存推理数据
            inf = f.create_group("inference")
            inf.create_dataset("timestamps", data=np.array(self.inference_data["timestamps"]))
            inf.create_dataset("steps", data=np.array(self.inference_data["steps"]))
            if self.inference_data["obs_images"]:
                inf.create_dataset(
                    "obs_images",
                    data=np.stack(self.inference_data["obs_images"]),
                    compression="gzip",
                    compression_opts=4,
                )
            if self.inference_data["obs_states"]:
                inf.create_dataset("obs_states", data=np.stack(self.inference_data["obs_states"]))
            if self.inference_data["actions"]:
                inf.create_dataset("actions", data=np.stack(self.inference_data["actions"]))
            inf.create_dataset("latencies_ms", data=np.array(self.inference_data["latencies_ms"]))

            # 保存命令数据
            cmd = f.create_group("commands")
            cmd.create_dataset("timestamps", data=np.array(self.command_data["timestamps"]))
            if self.command_data["target_poses"]:
                cmd.create_dataset("target_poses", data=np.stack(self.command_data["target_poses"]))
            cmd.create_dataset("gripper_targets", data=np.array(self.command_data["gripper_targets"]))
            cmd.create_dataset("action_indices", data=np.array(self.command_data["action_indices"]))
            cmd.create_dataset("inference_steps", data=np.array(self.command_data["inference_steps"]))

            # 保存机械臂状态数据
            state = f.create_group("robot_states")
            state.create_dataset("timestamps", data=np.array(self.robot_state_data["timestamps"]))
            if self.robot_state_data["current_poses"]:
                state.create_dataset("current_poses", data=np.stack(self.robot_state_data["current_poses"]))
            state.create_dataset("gripper_states", data=np.array(self.robot_state_data["gripper_states"]))
            if self.robot_state_data["pose_errors"]:
                state.create_dataset("pose_errors", data=np.stack(self.robot_state_data["pose_errors"]))
            state.create_dataset("frame_intervals_ms", data=np.array(self.robot_state_data["frame_intervals_ms"]))

        logging.info(f"[DataLogger] 数据保存完成: {self.filename}")

    def close(self):
        """清理资源"""
        self.save()


class DobotRosWrapper(Node):
    """
    机械臂ROS接口封装

    提供与Dobot机械臂交互的ROS服务客户端和话题发布器。
    封装了机械臂控制、夹爪通信、状态发布等功能。

    主要功能：
    - 机械臂使能、清除错误、速度设置
    - ServoP伺服运动控制
    - 获取当前位姿
    - Modbus通信（用于夹爪控制）
    - 发布机械臂目标姿态和夹爪状态
    - 订阅相机图像和机器人状态（用于推理）
    """

    def __init__(self, node_name="inference_client"):
        """
        初始化ROS节点和服务客户端

        Args:
            node_name: ROS节点名称，默认为"inference_client"
        """
        super().__init__(node_name)

        # === 机械臂控制服务客户端 ===
        self.cli_enable = self.create_client(
            EnableRobot, "/dobot_bringup_v3/srv/EnableRobot"
        )
        self.cli_clear_error = self.create_client(
            ClearError, "/dobot_bringup_v3/srv/ClearError"
        )
        self.cli_speed_factor = self.create_client(
            SpeedFactor, "/dobot_bringup_v3/srv/SpeedFactor"
        )
        self.cli_servo_p = self.create_client(ServoP, "/dobot_bringup_v3/srv/ServoP")
        self.cli_get_pose = self.create_client(GetPose, "/dobot_bringup_v3/srv/GetPose")

        # === Modbus通信服务客户端（用于夹爪控制）===
        self.cli_modbus_create = self.create_client(
            ModbusCreate, "/dobot_bringup_v3/srv/ModbusCreate"
        )
        self.cli_modbus_close = self.create_client(
            ModbusClose, "/dobot_bringup_v3/srv/ModbusClose"
        )
        self.cli_set_hold_regs = self.create_client(
            SetHoldRegs, "/dobot_bringup_v3/srv/SetHoldRegs"
        )
        self.cli_get_hold_regs = self.create_client(
            GetHoldRegs, "/dobot_bringup_v3/srv/GetHoldRegs"
        )

        # === 话题发布器 ===
        self.pub_robot_target = self.create_publisher(
            ToolVectorActual, "/robot/target_pose", 10
        )
        self.pub_gripper_state = self.create_publisher(
            PointStamped, "/gripper/state_feedback", 10
        )
        self.pub_gripper_update = self.create_publisher(
            PointStamped, "/gripper/command_update", 10
        )

        # === QoS配置 ===
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        robot_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # === 话题订阅器（用于推理）===
        self.robot_state = None
        self.latest_image = None
        self.bridge = CvBridge()

        self.sub_image = self.create_subscription(
            Image, "/camera/color/image_raw", self._image_callback, sensor_qos
        )
        self.sub_robot_current = self.create_subscription(
            ToolVectorActual,
            "/dobot_msgs_v3/msg/ToolVectorActual",
            self._robot_current_callback,
            robot_qos,
        )

        self.get_logger().info("等待Dobot服务...")
        if not self.cli_servo_p.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("Dobot服务未就绪")

    def _image_callback(self, msg: Image):
        """图像回调 - 保存最新图像"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def _robot_current_callback(self, msg: ToolVectorActual):
        """机器人当前位姿回调"""
        self.robot_state = np.array(
            [msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz], dtype=np.float32
        )

    def call_service(self, client, request):
        """
        同步调用ROS服务并等待结果

        Args:
            client: ROS服务客户端
            request: 服务请求对象

        Returns:
            服务响应结果，如果服务未就绪则返回None
        """
        if not client.service_is_ready():
            return None
        future = client.call_async(request)
        while not future.done():
            time.sleep(0.001)
        return future.result()

    def call_service_async_no_wait(self, client, request):
        """
        异步发送服务请求，不等待结果

        用于高频控制场景（如ServoP），避免阻塞主循环。

        Args:
            client: ROS服务客户端
            request: 服务请求对象
        """
        if client.service_is_ready():
            client.call_async(request)


class GripperComm:
    """
    夹爪Modbus通信管理类

    通过Modbus RTU协议与夹爪通信，控制夹爪开合和读取状态。
    使用机械臂控制器作为Modbus主站，通过127.0.0.1:60000端口通信。

    关键寄存器地址：
    - 256: 使能寄存器 (1=使能)
    - 257: 力度/速度寄存器 (0-100)
    - 259: 目标位置寄存器 (0-1000，0=全开，1000=全闭)
    - 514: 当前位置反馈寄存器 (0-1000)
    """

    def __init__(self, wrapper: DobotRosWrapper):
        """
        初始化夹爪通信

        Args:
            wrapper: DobotRosWrapper实例，用于调用Modbus服务
        """
        self.wrapper = wrapper
        self.id = 0
        self.init_connection()

    def init_connection(self):
        """
        初始化Modbus连接

        步骤：
        1. 关闭可能存在的旧连接（索引1-4）
        2. 创建新的Modbus RTU连接
        3. 初始化夹爪寄存器（使能、力度/速度）
        """
        logging.info("[Gripper] 关闭旧连接...")
        for i in range(1, 5):
            req = ModbusClose.Request()
            req.index = i
            self.wrapper.call_service(self.wrapper.cli_modbus_close, req)

        logging.info("[Gripper] 创建Modbus连接...")
        req = ModbusCreate.Request()
        req.ip = "127.0.0.1"
        req.port = 60000
        req.slave_id = 1
        req.is_rtu = 1
        res = self.wrapper.call_service(self.wrapper.cli_modbus_create, req)

        if res and res.res == 0:
            match = re.search(r"(\d+)", str(res.index))
            self.id = int(match.group(1)) if match else int(res.index)
            logging.info(f"[Gripper] 连接成功, ID: {self.id}")
        else:
            logging.error(f"[Gripper] 连接失败! Response: {res}")
            self.id = 0

        if self.id > 0:
            logging.info("[Gripper] 初始化寄存器...")
            self.write_reg(256, 1, "1", wait=True)  # 使能夹爪
            self.write_reg(257, 1, "60", wait=True)  # 设置力度/速度为60%

    def write_reg(self, addr, count, val_str, wait=False):
        """
        写Modbus保持寄存器

        Args:
            addr: 寄存器起始地址
            count: 要写入的寄存器数量
            val_str: 要写入的值（字符串格式）
            wait: 是否等待写入完成（True=同步，False=异步）
        """
        if self.id <= 0:
            return
        req = SetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = count
        req.val_tab = val_str
        if wait:
            self.wrapper.call_service(self.wrapper.cli_set_hold_regs, req)
        else:
            self.wrapper.call_service_async_no_wait(self.wrapper.cli_set_hold_regs, req)

    def read_reg(self, addr):
        """
        读取Modbus保持寄存器

        Args:
            addr: 寄存器地址

        Returns:
            寄存器值（整数），失败返回None
        """
        if self.id <= 0:
            return None
        req = GetHoldRegs.Request()
        req.index = self.id
        req.addr = addr
        req.count = 1
        res = self.wrapper.call_service(self.wrapper.cli_get_hold_regs, req)
        if res and res.res == 0:
            try:
                return int(res.value)
            except Exception:
                pass
        return None


class GripperStateFeedback(threading.Thread):
    """
    夹爪状态反馈线程

    定期读取夹爪实际位置并发布当前状态和目标状态。
    使用时间插补技术将低频Modbus读取（10Hz）插补到高频发布（100Hz）。

    工作原理：
    1. 以10Hz频率读取夹爪实际位置（寄存器514）
    2. 以10Hz频率采样目标位置（从InferenceClient）
    3. 使用线性插补算法，在两次采样之间进行时间插值
    4. 以100Hz频率发布插补后的状态和目标位置
    """

    def __init__(self, wrapper: DobotRosWrapper, comm: GripperComm, target_pos_ref: list):
        """
        初始化夹爪状态反馈线程

        Args:
            wrapper: DobotRosWrapper实例，用于发布话题
            comm: GripperComm实例，用于读取夹爪位置
            target_pos_ref: 目标位置引用（列表），与InferenceClient共享
        """
        super().__init__(daemon=True)
        self.wrapper = wrapper
        self.comm = comm
        self.target_pos_ref = target_pos_ref
        self.running = True

        # 频率设置
        self.feedback_rate = 100.0  # 发布频率 100Hz
        self.modbus_read_rate = 10.0  # Modbus读取频率 10Hz
        self.read_interval = 1.0 / self.modbus_read_rate

        # 插补状态变量
        self.last_read_time = time.time()
        self.last_real_pos = None
        self.current_real_pos = None
        self.last_real_read_time = None
        self.current_real_read_time = None
        self.last_target_pos = None
        self.current_target_pos = None
        self.last_target_read_time = None
        self.current_target_read_time = None

    def _interpolate(self, last_val, current_val, last_time, current_time, now):
        """
        基于时间的线性插补算法

        Args:
            last_val: 上一次采样的值
            current_val: 当前采样的值
            last_time: 上一次采样的时间戳
            current_time: 当前采样的时间戳
            now: 当前时刻

        Returns:
            插补后的值（float）
        """
        if last_val is None or current_val is None:
            return current_val if current_val is not None else 0.0

        if last_time is None or current_time is None:
            return current_val

        time_span = current_time - last_time
        if time_span <= 0:
            return current_val

        elapsed = now - last_time
        alpha = min(1.0, elapsed / time_span)
        return last_val + alpha * (current_val - last_val)

    def run(self):
        """核心循环：100Hz发布夹爪状态"""
        while self.running:
            try:
                now = time.time()
                timestamp = self.wrapper.get_clock().now().to_msg()

                # 每0.1秒读取一次真实值（10Hz采样）
                if now - self.last_read_time >= self.read_interval:
                    self.last_read_time = now

                    # 读取夹爪实际位置
                    real_pos = self.comm.read_reg(514)
                    if real_pos is not None:
                        self.last_real_pos = self.current_real_pos
                        self.last_real_read_time = self.current_real_read_time
                        self.current_real_pos = float(real_pos)
                        self.current_real_read_time = now

                    # 读取目标位置
                    target = self.target_pos_ref[0] if self.target_pos_ref else 0.0
                    self.last_target_pos = self.current_target_pos
                    self.last_target_read_time = self.current_target_read_time
                    self.current_target_pos = target
                    self.current_target_read_time = now

                # 计算插补后的值
                interpolated_current = self._interpolate(
                    self.last_real_pos,
                    self.current_real_pos,
                    self.last_real_read_time,
                    self.current_real_read_time,
                    now,
                )
                interpolated_target = self._interpolate(
                    self.last_target_pos,
                    self.current_target_pos,
                    self.last_target_read_time,
                    self.current_target_read_time,
                    now,
                )

                # 发布实际位置反馈
                msg_current = PointStamped()
                msg_current.header.stamp = timestamp
                msg_current.header.frame_id = "gripper_feedback"
                msg_current.point.x = float(interpolated_current) if interpolated_current else 0.0
                self.wrapper.pub_gripper_state.publish(msg_current)

                # 发布目标位置
                msg_target = PointStamped()
                msg_target.header.stamp = timestamp
                msg_target.header.frame_id = "gripper_command"
                msg_target.point.x = float(interpolated_target) if interpolated_target else 0.0
                self.wrapper.pub_gripper_update.publish(msg_target)

            except Exception as e:
                logging.error(f"[ERROR] GripperStateFeedback: {e}")

            time.sleep(1.0 / self.feedback_rate)


class InferenceClient:
    """
    推理客户端主类

    连接到远程策略服务器，订阅ROS2话题获取图像和机器人状态，
    执行策略推理并控制机械臂和夹爪执行动作。

    主要功能：
    1. 初始化ROS节点和机械臂通信
    2. 初始化夹爪通信和状态反馈
    3. 连接策略服务器
    4. 执行推理循环
    5. 发布目标姿态供评估对比
    """

    def __init__(self, args: Args):
        """
        初始化推理客户端

        Args:
            args: 命令行参数配置
        """
        self.args = args

        # 初始化ROS
        if not rclpy.ok():
            rclpy.init()

        self.node = DobotRosWrapper()

        # ROS Spin线程
        self.spin_thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True
        )
        self.spin_thread.start()

        # 初始化机械臂
        if not args.dry_run:
            self._enable_robot()

        # 初始化夹爪
        self.gripper_comm = GripperComm(self.node)

        # 夹爪目标位置引用（用于状态反馈线程读取）
        self.gripper_target_pos = [1000.0]  # 初始为张开状态

        # 启动夹爪状态反馈线程
        self.gripper_feedback = GripperStateFeedback(
            self.node, self.gripper_comm, self.gripper_target_pos
        )
        self.gripper_feedback.start()

        # 推理相关状态
        self.policy_client = None
        self.action_queue = deque()
        self.step_count = 0
        self.inference_enabled = False

        # 性能监控
        self.perf_stats = {
            'inference_count': 0,                    # 推理调用次数
            'inference_times': deque(maxlen=100),    # 最近100次推理延迟
            'last_inference_time': None,             # 上次推理时间戳
        }

        # 数据记录器
        self.data_logger = None
        if args.record:
            self.data_logger = DataLogger(args.record_dir, args)

        # 动作队列跟踪（用于记录）
        self.current_action_index = 0  # 当前执行的动作在队列中的索引
        self.current_inference_step = 0  # 当前动作来自哪次推理

        # 多维度插值步长参数（基于 Dobot CR5 特性）
        self.step_lengths = np.array([
            5.0,    # x (mm) - 30Hz × 5mm = 150mm/s 最大速度
            5.0,    # y (mm)
            5.0,    # z (mm)
            3.0,    # rx (度) - 30Hz × 3° = 90°/s 最大旋转
            3.0,    # ry (度)
            3.0,    # rz (度)
            20.0    # gripper (0-1000) - 跳变200需10步
        ], dtype=np.float32)

        logging.info("=== 推理客户端初始化完成 ===")
        logging.info(f"策略服务器: {args.host}:{args.port}")
        logging.info(f"任务提示: {args.prompt}")
        logging.info(f"重规划步数: {args.replan_steps}")
        logging.info(f"最大步数: {args.max_steps}")
        logging.info(f"空运行模式: {args.dry_run}")
        logging.info(f"数据记录: {args.record}")

    def _enable_robot(self):
        """使能机器人"""
        logging.info("Waiting for robot services...")
        self.node.cli_enable.wait_for_service(timeout_sec=10.0)
        self.node.cli_clear_error.wait_for_service(timeout_sec=10.0)
        self.node.cli_speed_factor.wait_for_service(timeout_sec=10.0)
        self.node.cli_servo_p.wait_for_service(timeout_sec=10.0)
        self.node.cli_get_pose.wait_for_service(timeout_sec=10.0)

        logging.info("Clearing errors...")
        self.node.call_service(self.node.cli_clear_error, ClearError.Request())

        logging.info("Setting speed factor to 100%...")
        req = SpeedFactor.Request()
        req.ratio = 100
        self.node.call_service(self.node.cli_speed_factor, req)

        logging.info("Enabling robot...")
        self.node.call_service(self.node.cli_enable, EnableRobot.Request())

        logging.info("Robot enabled")

    def connect_policy_server(self):
        """连接策略服务器"""
        logging.info(f"Connecting to policy server at {self.args.host}:{self.args.port}...")
        self.policy_client = _websocket_client_policy.WebsocketClientPolicy(
            self.args.host, self.args.port
        )
        logging.info("Connected to policy server")

        # 获取服务器元数据
        metadata = self.policy_client.get_server_metadata()
        logging.info(f"Server metadata: {metadata}")

    def ServoP(self, x, y, z, rx, ry, rz):
        """
        发送ServoP伺服运动指令

        根据 args.blocking_servo 决定是否阻塞等待执行完成。

        Args:
            x, y, z: 目标位置 (mm)
            rx, ry, rz: 目标姿态 (度)
        """
        req = ServoP.Request()
        req.x, req.y, req.z = float(x), float(y), float(z)
        req.rx, req.ry, req.rz = float(rx), float(ry), float(rz)

        if self.args.blocking_servo:
            # 阻塞执行：等待 ServoP 完成
            self.node.call_service(self.node.cli_servo_p, req)
        else:
            # 非阻塞执行：立即返回
            self.node.call_service_async_no_wait(self.node.cli_servo_p, req)

    def set_gripper(self, position):
        """
        设置夹爪目标位置

        Args:
            position: 目标位置，范围0-1000 (0=全开，1000=全闭)
        """
        self.gripper_target_pos[0] = float(position)
        self.gripper_comm.write_reg(259, 1, str(int(position)), wait=False)

    def get_current_pose(self):
        """
        获取机械臂当前位姿

        Returns:
            numpy array [x, y, z, rx, ry, rz] 或 None
        """
        req = GetPose.Request()
        res = self.node.call_service(self.node.cli_get_pose, req)
        if res:
            try:
                # 解析响应字符串
                matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", str(res.pose))
                if len(matches) >= 6:
                    return np.array([float(x) for x in matches[:6]], dtype=np.float32)
            except Exception:
                pass
        return None

    def _get_current_state(self) -> np.ndarray:
        """
        获取当前实际状态（用于插值）

        Returns:
            当前状态向量 [x, y, z, rx, ry, rz, gripper]，如果数据不可用则返回None
        """
        if self.node.robot_state is None:
            return None

        # 机械臂实际反馈位姿
        robot_pose = self.node.robot_state.copy()  # (6,)

        # 夹爪实际反馈位置（0-1000）
        gripper_pos = self.gripper_feedback.current_real_pos
        if gripper_pos is None:
            # 降级：使用目标位置
            gripper_pos = self.gripper_target_pos[0]

        return np.concatenate([robot_pose, [gripper_pos]]).astype(np.float32)

    def publish_robot_target(self, pose):
        """
        发布机械臂目标姿态

        Args:
            pose: 目标姿态 [x, y, z, rx, ry, rz]
        """
        msg = ToolVectorActual()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.x = float(pose[0])
        msg.y = float(pose[1])
        msg.z = float(pose[2])
        msg.rx = float(pose[3])
        msg.ry = float(pose[4])
        msg.rz = float(pose[5])
        self.node.pub_robot_target.publish(msg)

    def _build_observation(self):
        """构建观测数据（简化版，无时间同步）"""
        if self.node.robot_state is None or self.node.latest_image is None:
            return None

        cv_image = self.node.latest_image

        # 图像预处理
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img = image_tools.resize_with_pad(img, self.args.resize_size, self.args.resize_size)
        img = image_tools.convert_to_uint8(img)

        # 使用实际反馈位置（关键！）
        gripper_pos = self.gripper_feedback.current_real_pos
        if gripper_pos is None:
            # 降级：使用目标位置
            gripper_pos = self.gripper_target_pos[0]

        gripper_normalized = gripper_pos / 1000.0

        # 拼接状态
        state = np.concatenate([self.node.robot_state, [gripper_normalized]]).astype(np.float32)

        return {
            "observation/image": img,
            "observation/wrist_image": img,
            "observation/state": state,
            "prompt": self.args.prompt,
        }

    def _should_interpolate(
        self,
        actions: np.ndarray,
        current_state: np.ndarray
    ) -> tuple[bool, int]:
        """
        判断是否需要插值及所需步数

        Args:
            actions: 原始动作序列 (N, 7)
            current_state: 当前实际状态 (7,)

        Returns:
            (是否需要插值, 总步数)
        """
        if len(actions) == 0:
            return False, 0

        # 计算差距
        first_action = actions[0]
        diffs = np.abs(first_action - current_state)

        # 各维度所需步数
        steps_needed = np.ceil(diffs / self.step_lengths).astype(int)
        steps_needed = np.maximum(steps_needed, 1)

        # 总步数
        max_dim_steps = np.max(steps_needed)
        total_steps = max(max_dim_steps, len(actions))

        needs_interpolation = total_steps > len(actions)

        if needs_interpolation:
            # 记录触发维度
            dim_names = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
            triggers = [f"{dim_names[i]}={diffs[i]:.1f}"
                       for i in range(7) if steps_needed[i] == max_dim_steps]
            logging.info(f"触发插值: {', '.join(triggers)} | {len(actions)}步→{total_steps}步")

        return needs_interpolation, total_steps

    def _generate_interpolated_actions(
        self,
        actions: np.ndarray,
        current_state: np.ndarray
    ) -> np.ndarray:
        """
        基于物理约束的多维度轨迹插值

        参考: template.py 的 generate_trajectory

        当任一维度跳变超过step_length时，扩展动作步数以实现平滑过渡。
        所有维度都进行线性插值，在整个轨迹上均匀重新采样。

        Args:
            actions: 原始动作序列 (N, 7)
            current_state: 当前实际状态 (7,)

        Returns:
            插值后的动作序列，步数可能大于原始 N
        """
        if len(actions) == 0:
            return actions

        # 判断是否需要插值
        needs_interpolation, total_steps = self._should_interpolate(actions, current_state)

        if not needs_interpolation:
            return actions

        # 执行插值
        original_steps = len(actions)
        interpolated = np.zeros((total_steps, 7), dtype=np.float32)

        for step in range(total_steps):
            if total_steps == 1:
                progress = 1.0
            else:
                progress = step / (total_steps - 1)

            if progress == 0:
                # 第一步：当前状态
                interpolated[step] = current_state
            else:
                # 在原始动作序列中插值
                action_progress = progress * (original_steps - 1)
                idx_low = int(np.floor(action_progress))
                idx_high = min(idx_low + 1, original_steps - 1)
                alpha = action_progress - idx_low

                if idx_high == idx_low:
                    interpolated[step] = actions[idx_low]
                else:
                    interpolated[step] = (
                        actions[idx_low] * (1 - alpha) +
                        actions[idx_high] * alpha
                    )

        return interpolated

    def _get_action(self, obs: dict):
        """获取动作（使用动作队列 + 重规划）"""
        if not self.action_queue:
            try:
                if self.step_count == 0:
                    logging.info("Calling policy inference for first time...")

                # 记录推理开始时间
                t0 = time.time()
                result = self.policy_client.infer(obs)
                inference_time = time.time() - t0

                # 更新统计
                self.perf_stats['inference_count'] += 1
                self.perf_stats['inference_times'].append(inference_time)
                self.perf_stats['last_inference_time'] = time.time()

                if self.step_count == 0:
                    logging.info(f"First inference completed! Latency: {inference_time*1000:.1f}ms")

                actions = result["actions"]

                # 记录推理结果
                if self.data_logger is not None:
                    self.data_logger.log_inference(
                        step=self.step_count,
                        obs_image=obs["observation/image"],
                        obs_state=obs["observation/state"],
                        actions=actions,
                        latency_ms=inference_time * 1000,
                    )

                # 更新推理步数计数器
                self.current_inference_step = self.perf_stats['inference_count']
                self.current_action_index = 0

                # 取前 replan_steps 个动作
                original_actions = actions[:min(self.args.replan_steps, len(actions))]

                # 获取当前实际状态
                current_state = self._get_current_state()

                # 基于当前状态进行插值
                if current_state is not None:
                    interpolated_actions = self._generate_interpolated_actions(
                        original_actions,
                        current_state
                    )
                else:
                    # 降级：使用原始动作
                    interpolated_actions = original_actions

                # 填充动作队列
                for i in range(len(interpolated_actions)):
                    self.action_queue.append(interpolated_actions[i])

            except Exception as e:
                logging.error(f"Inference failed: {e}")
                return None

        if self.action_queue:
            action = self.action_queue.popleft()
            # 更新动作索引（从当前推理批次中的第几个动作）
            action_index = self.current_action_index
            self.current_action_index += 1
            # 返回动作和索引信息
            return action, action_index, self.current_inference_step
        return None

    def _execute_action(self, action: np.ndarray, action_index: int, inference_step: int):
        """
        执行动作并记录

        Args:
            action: 动作向量 (7,) [x,y,z,rx,ry,rz,gripper]
            action_index: 当前执行的是第几个动作 (0~replan_steps-1)
            inference_step: 该动作来自哪次推理
        """
        # action: (7,) [x,y,z,rx,ry,rz,gripper]
        target_pose = [
            float(action[0]),
            float(action[1]),
            float(action[2]),
            float(action[3]),
            float(action[4]),
            float(action[5]),
        ]

        gripper_raw = float(action[6])
        gripper_target = int(np.clip(gripper_raw, 0, 1000))

        if self.step_count % 100 == 0:
            logging.info(
                f"Action[{self.step_count}]: gripper_raw={gripper_raw:.1f}, gripper_target={gripper_target}"
            )

        # 获取当前机械臂状态（用于记录）
        current_pose = self.node.robot_state

        # 获取当前夹爪状态（使用实际反馈）
        gripper_state = self.gripper_feedback.current_real_pos
        if gripper_state is None:
            gripper_state = self.gripper_target_pos[0]

        # 记录数据
        if self.data_logger is not None and current_pose is not None:
            # 记录发送的命令
            self.data_logger.log_command(
                target_pose=np.array(target_pose, dtype=np.float32),
                gripper_target=gripper_target,
                action_index=action_index,
                inference_step=inference_step,
            )
            # 记录机械臂状态
            self.data_logger.log_robot_state(
                current_pose=current_pose,
                gripper_state=gripper_state,
                target_pose=np.array(target_pose, dtype=np.float32),
                image_timestamp=time.time(),  # 使用当前时间作为时间戳
            )

        if self.args.dry_run:
            logging.info(f"[DRY RUN] pose={target_pose}, gripper={gripper_target}")
            return

        # 发送ServoP指令
        self.ServoP(*target_pose)

        # 发布目标位姿话题
        self.publish_robot_target(target_pose)

        # 设置夹爪目标
        self.set_gripper(gripper_target)

    def run(self):
        """
        运行推理主循环

        按30Hz频率（图像到达频率）执行推理：
        1. 等待新图像到达
        2. 构建观测数据
        3. 获取动作（调用策略服务器或从队列取出）
        4. 执行动作（ServoP + 夹爪控制）
        5. 发布目标状态话题
        """
        logging.info("\n=== 开始推理 ===")
        logging.info("按 Ctrl+C 停止\n")

        # 等待初始数据
        logging.info("Waiting for camera and robot data...")
        wait_start = time.time()
        while self.node.robot_state is None or self.node.latest_image is None:
            if time.time() - wait_start > 10.0:
                logging.error("Timeout waiting for sensor data")
                raise RuntimeError("Sensor data not available")
            time.sleep(0.1)
        logging.info("Sensor data available")

        # 获取初始位姿
        if not self.args.dry_run:
            initial_pose = self.get_current_pose()
            if initial_pose is not None:
                logging.info(f"Initial pose: {initial_pose}")
            else:
                logging.warning("Could not get initial pose")

        # 连接策略服务器
        self.connect_policy_server()

        self.inference_enabled = True
        start_time = time.time()

        # 30Hz 速率控制
        target_dt = 1.0 / 30.0  # 33.3ms per step
        last_step_time = time.time()

        try:
            while self.inference_enabled and self.step_count < self.args.max_steps:
                # 速率控制: 确保至少间隔 33.3ms
                current_time = time.time()
                elapsed = current_time - last_step_time
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
                last_step_time = time.time()

                # 构建观测
                obs = self._build_observation()
                if obs is None:
                    time.sleep(0.001)
                    continue

                # 获取动作
                action_result = self._get_action(obs)
                if action_result is None:
                    continue

                action, action_index, inference_step = action_result

                # 执行动作
                self._execute_action(action, action_index, inference_step)
                self.step_count += 1

                # 打印进度（每30帧）
                if self.step_count % 30 == 0:
                    elapsed = time.time() - start_time
                    action_fps = self.step_count / elapsed if elapsed > 0 else 0
                    inf_count = self.perf_stats['inference_count']
                    inf_fps = inf_count / elapsed if elapsed > 0 else 0

                    # 推理延迟统计
                    if self.perf_stats['inference_times']:
                        inf_times = list(self.perf_stats['inference_times'])
                        inf_mean = np.mean(inf_times) * 1000
                        inf_max = np.max(inf_times) * 1000
                    else:
                        inf_mean = inf_max = 0

                    logging.info(
                        f"Step {self.step_count} | "
                        f"动作频率: {action_fps:.1f}Hz (目标30) | "
                        f"推理频率: {inf_fps:.1f}Hz (预期{30/self.args.replan_steps:.0f}) | "
                        f"推理延迟: {inf_mean:.0f}ms (max:{inf_max:.0f}ms) | "
                        f"队列: {len(self.action_queue)}"
                    )

            logging.info(f"\n=== 推理完成 ===")
            logging.info(f"总步数: {self.step_count}")
            logging.info(f"总耗时: {time.time() - start_time:.2f}s")

        except KeyboardInterrupt:
            logging.info("\n\n=== 推理被中断 ===")

        # 保存记录数据
        if self.data_logger is not None:
            self.data_logger.save()

        # 等待夹爪到达最终位置
        logging.info("等待夹爪到达最终位置...")
        time.sleep(1.0)

    def close(self):
        """
        关闭推理客户端并清理资源

        步骤：
        1. 停止夹爪状态反馈线程
        2. 等待线程结束
        3. 关闭数据记录器
        4. 关闭ROS节点
        """
        self.gripper_feedback.running = False
        self.gripper_feedback.join(timeout=1.0)

        # 关闭数据记录器（会自动保存未保存的数据）
        if self.data_logger is not None:
            self.data_logger.close()

        self.node.destroy_node()
        rclpy.shutdown()


def main():
    """
    主函数：命令行入口
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = tyro.cli(Args)

    # 创建推理客户端
    client = InferenceClient(args)

    # 运行推理
    client.run()

    # 关闭
    client.close()

    logging.info("Done")


if __name__ == "__main__":
    main()
