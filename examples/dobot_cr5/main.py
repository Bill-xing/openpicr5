"""
CR5 真机推理客户端

功能：
1. 连接到远程策略服务器（WebSocket）
2. 订阅ROS2话题获取图像和机器人状态
3. 以30Hz频率执行策略推理
4. 发送ServoP指令控制机器人
5. 控制Modbus夹爪

架构：
- 图像到达触发推理（30Hz，与训练数据一致）
- 机器人状态订阅用于构建观测
- 发布目标位姿和夹爪状态（与data_collector4兼容）

使用：
    python examples/dobot_cr5/main.py --host localhost --port 8000 --prompt "pick up the red block"
"""

import dataclasses
import logging
import queue
import re
import threading
import time
from collections import deque

import cv2
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
    max_steps: int = 1000  # 最大执行步数
    dry_run: bool = False  # 空运行模式：只打印动作，不执行


class GripperController(threading.Thread):
    """
    夹爪控制线程 - 参考data_collector4.py的GripperManager

    功能：
    1. 初始化Modbus连接
    2. 以50Hz频率发送夹爪目标位置
    3. 以10Hz频率读取夹爪实际位置
    """

    def __init__(self, node: Node):
        super().__init__(daemon=True)
        self.node = node
        self.modbus_id = 0
        self.target_pos = 1000.0  # 目标位置 0-1000（1000=完全张开）
        self.current_pos = 1000.0  # 当前位置 0-1000
        self.running = True
        self._init_modbus()

    def _call_service(self, client, request):
        """同步调用服务"""
        if not client.service_is_ready():
            return None
        future = client.call_async(request)
        while not future.done():
            time.sleep(0.001)
        return future.result()

    def _init_modbus(self):
        """初始化Modbus连接"""
        # 创建服务客户端
        self.cli_modbus_create = self.node.create_client(
            ModbusCreate, "/dobot_bringup_v3/srv/ModbusCreate"
        )
        self.cli_modbus_close = self.node.create_client(
            ModbusClose, "/dobot_bringup_v3/srv/ModbusClose"
        )
        self.cli_set_hold_regs = self.node.create_client(
            SetHoldRegs, "/dobot_bringup_v3/srv/SetHoldRegs"
        )
        self.cli_get_hold_regs = self.node.create_client(
            GetHoldRegs, "/dobot_bringup_v3/srv/GetHoldRegs"
        )

        # 等待服务可用
        logging.info("Waiting for Modbus services...")
        self.cli_modbus_create.wait_for_service(timeout_sec=5.0)

        # 关闭旧连接
        logging.info("Closing old Modbus connections...")
        for i in range(1, 5):
            req = ModbusClose.Request()
            req.index = i
            self._call_service(self.cli_modbus_close, req)

        # 创建新连接
        logging.info("Creating Modbus connection...")
        req = ModbusCreate.Request()
        req.ip = "127.0.0.1"
        req.port = 60000
        req.slave_id = 1
        req.is_rtu = 1
        res = self._call_service(self.cli_modbus_create, req)

        if res and res.res == 0:
            match = re.search(r"(\d+)", str(res.index))
            self.modbus_id = int(match.group(1)) if match else int(res.index)
            logging.info(f"Gripper connected, Modbus ID: {self.modbus_id}")

            # 初始化夹爪
            self._write_reg(256, 1, "1")  # Enable
            self._write_reg(257, 1, "60")  # Force/Speed
        else:
            logging.error(f"Gripper ModbusCreate failed: {res}")
            self.modbus_id = 0

        # 读取初始位置
        if self.modbus_id > 0:
            init_val = self._read_reg(514)
            if init_val is not None:
                self.current_pos = float(init_val)
                logging.info(f"Gripper initial position: {self.current_pos}")

    def _write_reg(self, addr: int, count: int, val_str: str):
        """写入Modbus寄存器"""
        if self.modbus_id <= 0:
            return
        req = SetHoldRegs.Request()
        req.index = self.modbus_id
        req.addr = addr
        req.count = count
        req.val_tab = val_str
        self._call_service(self.cli_set_hold_regs, req)

    def _read_reg(self, addr: int):
        """读取Modbus寄存器"""
        if self.modbus_id <= 0:
            return None
        req = GetHoldRegs.Request()
        req.index = self.modbus_id
        req.addr = addr
        req.count = 1
        res = self._call_service(self.cli_get_hold_regs, req)
        if res and res.res == 0:
            try:
                return int(res.value)
            except Exception:
                pass
        return None

    def set_target(self, pos: float):
        """设置目标位置 (0-1000)"""
        self.target_pos = max(0.0, min(1000.0, float(pos)))

    def run(self):
        """控制循环 (50Hz发送指令，10Hz读取实际位置)"""
        read_counter = 0
        while self.running:
            # 发送目标位置
            self._write_reg(259, 1, str(int(self.target_pos)))

            # 每5次读取一次实际位置（10Hz）
            read_counter += 1
            if read_counter >= 5:
                read_counter = 0
                real_pos = self._read_reg(514)
                if real_pos is not None:
                    self.current_pos = float(real_pos)

            time.sleep(0.02)  # 50Hz

    def stop(self):
        """停止控制线程"""
        self.running = False


class CR5InferenceNode(Node):
    """
    CR5推理ROS2节点

    订阅：
    - /camera/color/image_raw - RGB图像（30Hz，触发推理）
    - /dobot_msgs_v3/msg/ToolVectorActual - 机器人当前位姿

    发布：
    - /robot/target_pose - 机器人目标位姿
    - /gripper/command_update - 夹爪目标状态
    - /gripper/state_feedback - 夹爪实际状态反馈
    """

    def __init__(self, args: Args):
        super().__init__("cr5_inference_node")
        self.args = args

        # 状态变量
        self.robot_state = None  # 机器人当前位姿 (6,)
        self.latest_image = None  # 最新图像
        self.action_queue = deque()  # 动作队列
        self.inference_enabled = False  # 推理使能标志
        self.step_count = 0  # 执行步数计数
        self.bridge = CvBridge()

        # 推理队列（线程安全）
        self.inference_queue = queue.Queue(maxsize=2)  # 最多缓存2帧图像
        self.inference_thread = None
        self.inference_running = False

        # ========== 服务客户端 ==========
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

        # ========== QoS配置 ==========
        # 相机话题使用BEST_EFFORT（RealSense默认配置）
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # 机器人状态话题使用RELIABLE（匹配dobot_feedback发布者）
        robot_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,  # 必须与发布者匹配
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ========== 订阅器 ==========
        # 相机图像 - 30Hz，触发推理
        self.sub_image = self.create_subscription(
            Image, "/camera/color/image_raw", self.image_callback, sensor_qos
        )
        # 机器人当前位姿 - 100Hz，由驱动发布
        self.sub_robot_current = self.create_subscription(
            ToolVectorActual,
            "/dobot_msgs_v3/msg/ToolVectorActual",
            self.robot_current_callback,
            robot_qos,
        )

        # ========== 发布器（与data_collector4兼容）==========
        # 机器人目标位姿
        self.pub_robot_target = self.create_publisher(
            ToolVectorActual, "/robot/target_pose", 10
        )
        # 夹爪目标状态
        self.pub_gripper_target = self.create_publisher(
            PointStamped, "/gripper/command_update", 5
        )
        # 夹爪实际状态反馈
        self.pub_gripper_state = self.create_publisher(
            PointStamped, "/gripper/state_feedback", 5
        )

        logging.info("CR5InferenceNode initialized")

    def call_service(self, client, request):
        """同步调用服务"""
        if not client.service_is_ready():
            return None
        future = client.call_async(request)
        while not future.done():
            time.sleep(0.001)
        return future.result()

    def robot_current_callback(self, msg: ToolVectorActual):
        """机器人当前位姿回调"""
        if self.robot_state is None:
            logging.info("Robot state callback received first message")
        self.robot_state = np.array(
            [msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz], dtype=np.float32
        )

    def image_callback(self, msg: Image):
        """图像回调 - 非阻塞，只负责数据采集（30Hz）"""
        if self.latest_image is None:
            logging.info("Image callback received first message")
        if not self.inference_enabled:
            self.latest_image = True  # 标记已收到图像
            return

        # 检查是否达到最大步数
        if self.step_count >= self.args.max_steps:
            logging.info(f"Reached max steps ({self.args.max_steps}), stopping...")
            self.inference_enabled = False
            return

        # 获取图像
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            logging.error(f"Image conversion failed: {e}")
            return

        self.latest_image = cv_image

        # 非阻塞入队：如果队列满了，丢弃旧帧
        try:
            self.inference_queue.put_nowait(cv_image)
        except queue.Full:
            # 队列满，丢弃当前帧（推理线程处理不过来）
            pass

    def _build_observation(self, cv_image: np.ndarray):
        """构建观测数据"""
        # 检查机器人状态是否就绪
        if self.robot_state is None:
            logging.warning("Robot state not available")
            return None

        # 检查夹爪控制器是否就绪
        if not hasattr(self, "gripper_controller"):
            logging.warning("Gripper controller not available")
            return None

        # 图像预处理：BGR→RGB，resize到224x224
        if self.step_count == 0:
            logging.info("Building first observation...")
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img = image_tools.resize_with_pad(img, self.args.resize_size, self.args.resize_size)
        img = image_tools.convert_to_uint8(img)

        # 夹爪状态归一化 (0-1000 -> 0-1)
        gripper_normalized = self.gripper_controller.current_pos / 1000.0

        # 状态向量：[x,y,z,rx,ry,rz,gripper_normalized]
        # 位置单位：mm，旋转单位：度
        state = np.concatenate(
            [self.robot_state, [gripper_normalized]]  # (6,)  # (1,)
        ).astype(np.float32)

        return {
            "observation/image": img,
            "observation/wrist_image": img,  # 复用主相机（没有腕部相机）
            "observation/state": state,
            "prompt": self.args.prompt,
        }

    def _get_action(self, obs: dict):
        """获取动作（使用动作队列 + 重规划）"""
        # 如果动作队列为空，调用策略推理
        if not self.action_queue:
            try:
                if self.step_count == 0:
                    logging.info("Calling policy inference for first time...")
                result = self.policy_client.infer(obs)
                if self.step_count == 0:
                    logging.info("First inference completed!")
                actions = result["actions"]  # (action_horizon, 7)

                # 只取前replan_steps个动作
                for i in range(min(self.args.replan_steps, len(actions))):
                    self.action_queue.append(actions[i])

            except Exception as e:
                logging.error(f"Inference failed: {e}")
                return None

        # 从队列取出一个动作
        if self.action_queue:
            return self.action_queue.popleft()
        return None

    def _execute_action(self, action: np.ndarray):
        """执行动作"""
        # action: (7,) [x,y,z,rx,ry,rz,gripper]
        # 注意：模型输出的动作已经是反归一化后的原始值！

        # 1. 提取目标位姿（单位：mm, 度）
        target_pose = [
            float(action[0]),
            float(action[1]),
            float(action[2]),
            float(action[3]),
            float(action[4]),
            float(action[5]),
        ]

        # 2. 提取夹爪目标（模型输出已经是原始值 280-1000，不需要再乘1000）
        gripper_raw = float(action[6])
        gripper_target = int(np.clip(gripper_raw, 0, 1000))

        # 每100步打印一次原始动作值用于调试
        if self.step_count % 100 == 0:
            logging.info(f"Action[{self.step_count}]: gripper_raw={gripper_raw:.1f}, gripper_target={gripper_target}")

        if self.args.dry_run:
            # 空运行模式：只打印
            logging.info(
                f"[DRY RUN] pose={target_pose}, gripper={gripper_target}"
            )
            return

        # 3. 发送ServoP指令
        self._servo_p(*target_pose)

        # 4. 发布目标位姿话题
        self._publish_robot_target(*target_pose)

        # 5. 设置夹爪目标
        self.gripper_controller.set_target(gripper_target)

        # 6. 发布夹爪状态话题
        self._publish_gripper_state()

    def _servo_p(self, x, y, z, rx, ry, rz):
        """发送ServoP指令"""
        req = ServoP.Request()
        req.x, req.y, req.z = float(x), float(y), float(z)
        req.rx, req.ry, req.rz = float(rx), float(ry), float(rz)
        self.call_service(self.cli_servo_p, req)

    def _publish_robot_target(self, x, y, z, rx, ry, rz):
        """发布机器人目标位姿"""
        msg = ToolVectorActual()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.x, msg.y, msg.z = float(x), float(y), float(z)
        msg.rx, msg.ry, msg.rz = float(rx), float(ry), float(rz)
        self.pub_robot_target.publish(msg)

    def _publish_gripper_state(self):
        """发布夹爪状态（目标和实际）"""
        now = self.get_clock().now().to_msg()

        # 发布目标状态
        msg_target = PointStamped()
        msg_target.header.stamp = now
        msg_target.header.frame_id = "gripper_command"
        msg_target.point.x = float(self.gripper_controller.target_pos)
        self.pub_gripper_target.publish(msg_target)

        # 发布实际状态
        msg_current = PointStamped()
        msg_current.header.stamp = now
        msg_current.header.frame_id = "gripper_feedback"
        msg_current.point.x = float(self.gripper_controller.current_pos)
        self.pub_gripper_state.publish(msg_current)

    def _inference_loop(self):
        """推理线程主循环 - 独立线程中运行"""
        logging.info("Inference thread started")

        while self.inference_running and self.inference_enabled:
            try:
                # 从队列获取图像（阻塞，超时1秒）
                try:
                    cv_image = self.inference_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # 检查是否需要停止
                if self.step_count >= self.args.max_steps:
                    logging.info(f"Reached max steps ({self.args.max_steps}), stopping...")
                    self.inference_enabled = False
                    break

                # 构建观测
                obs = self._build_observation(cv_image)
                if obs is None:
                    continue

                # 获取动作
                action = self._get_action(obs)
                if action is None:
                    continue

                # 执行动作
                self._execute_action(action)
                self.step_count += 1

                # 定期打印状态
                if self.step_count % 30 == 0:
                    logging.info(
                        f"Step {self.step_count}, "
                        f"action queue: {len(self.action_queue)}, "
                        f"image queue: {self.inference_queue.qsize()}"
                    )

            except Exception as e:
                logging.error(f"Error in inference loop: {e}", exc_info=True)
                time.sleep(0.1)

        logging.info("Inference thread stopped")

    def start_inference_thread(self):
        """启动推理线程"""
        if self.inference_thread is not None:
            logging.warning("Inference thread already running")
            return

        self.inference_running = True
        self.inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="InferenceThread"
        )
        self.inference_thread.start()
        logging.info("Inference thread launched")

    def stop_inference_thread(self):
        """停止推理线程"""
        if self.inference_thread is None:
            return

        logging.info("Stopping inference thread...")
        self.inference_running = False

        # 等待线程结束，设置合理的超时
        if self.inference_thread.is_alive():
            self.inference_thread.join(timeout=3.0)
            if self.inference_thread.is_alive():
                logging.warning("Inference thread did not stop gracefully")

        self.inference_thread = None
        logging.info("Inference thread stopped")

    def enable_robot(self):
        """使能机器人"""
        logging.info("Waiting for robot services...")
        self.cli_enable.wait_for_service(timeout_sec=10.0)
        self.cli_clear_error.wait_for_service(timeout_sec=10.0)
        self.cli_speed_factor.wait_for_service(timeout_sec=10.0)
        self.cli_servo_p.wait_for_service(timeout_sec=10.0)
        self.cli_get_pose.wait_for_service(timeout_sec=10.0)

        logging.info("Clearing errors...")
        self.call_service(self.cli_clear_error, ClearError.Request())

        logging.info("Setting speed factor to 100%...")
        req = SpeedFactor.Request()
        req.ratio = 100
        self.call_service(self.cli_speed_factor, req)

        logging.info("Enabling robot...")
        self.call_service(self.cli_enable, EnableRobot.Request())

        logging.info("Robot enabled")

    def get_current_pose(self):
        """获取当前位姿"""
        res = self.call_service(self.cli_get_pose, GetPose.Request())
        if res and hasattr(res, "pose"):
            matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", res.pose)
            if len(matches) >= 6:
                return [float(x) for x in matches[:6]]
        return None


def run_inference(args: Args):
    """运行推理主循环"""
    logging.info("=" * 60)
    logging.info("CR5 Inference Client")
    logging.info("=" * 60)
    logging.info(f"Server: {args.host}:{args.port}")
    logging.info(f"Prompt: {args.prompt}")
    logging.info(f"Replan steps: {args.replan_steps}")
    logging.info(f"Max steps: {args.max_steps}")
    logging.info(f"Dry run: {args.dry_run}")
    logging.info("=" * 60)

    # 初始化ROS2
    rclpy.init()

    # 创建节点
    node = CR5InferenceNode(args)

    # 启动ROS2 spin线程
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        # 使能机器人
        if not args.dry_run:
            node.enable_robot()

        # 初始化夹爪控制器
        logging.info("Initializing gripper controller...")
        node.gripper_controller = GripperController(node)
        node.gripper_controller.start()

        # 等待初始数据
        logging.info("Waiting for camera and robot data...")
        wait_start = time.time()
        while node.robot_state is None or node.latest_image is None:
            if time.time() - wait_start > 10.0:
                logging.error("Timeout waiting for sensor data")
                raise RuntimeError("Sensor data not available")
            time.sleep(0.1)
        logging.info("Sensor data available")

        # 获取初始位姿
        if not args.dry_run:
            initial_pose = node.get_current_pose()
            if initial_pose:
                logging.info(f"Initial pose: {initial_pose}")
            else:
                logging.warning("Could not get initial pose")

        # 连接策略服务器
        logging.info(f"Connecting to policy server at {args.host}:{args.port}...")
        node.policy_client = _websocket_client_policy.WebsocketClientPolicy(
            args.host, args.port
        )
        logging.info("Connected to policy server")

        # 获取服务器元数据
        metadata = node.policy_client.get_server_metadata()
        logging.info(f"Server metadata: {metadata}")

        # 开始推理
        logging.info("=" * 60)
        logging.info("Starting inference... Press Ctrl+C to stop")
        logging.info("=" * 60)
        node.inference_enabled = True

        # 启动推理线程
        node.start_inference_thread()

        # 主循环 - 等待推理完成或用户中断
        while node.inference_enabled and rclpy.ok():
            time.sleep(0.1)

        logging.info(f"Inference stopped after {node.step_count} steps")

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error(f"Error: {e}")
        raise
    finally:
        # 停止推理线程
        if hasattr(node, "inference_thread"):
            node.stop_inference_thread()

        # 停止夹爪控制器
        if hasattr(node, "gripper_controller"):
            node.gripper_controller.stop()
            node.gripper_controller.join(timeout=1.0)

        # 关闭节点
        node.destroy_node()
        rclpy.shutdown()

    logging.info("Done")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = tyro.cli(Args)
    run_inference(args)
