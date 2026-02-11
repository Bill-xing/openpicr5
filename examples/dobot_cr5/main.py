"""
CR5 真机推理客户端（单线程架构）

功能：
1. 连接到远程策略服务器（WebSocket）
2. 订阅ROS2话题获取图像和机器人状态
3. 以30Hz频率执行动作
4. 发送ServoP指令控制机器人
5. 控制Modbus夹爪

架构：
- 单线程主循环
- 推理获取动作块后，以30Hz频率逐个执行

使用：
    python examples/dobot_cr5/main.py --host localhost --port 8000 --prompt "pick up the red block"
"""

import dataclasses
import logging
import re
import time

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
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from sensor_msgs.msg import Image
import tyro


@dataclasses.dataclass
class Args:
    """CR5推理客户端命令行参数配置"""

    host: str = "localhost"
    port: int = 8000
    resize_size: int = 224
    action_horizon: int = 10  # 每次推理返回的动作数量（与训练配置一致）

    prompt: str = "pick up the object"

    max_steps: int = 1000
    dry_run: bool = False


class GripperController:
    """夹爪控制器（同步Modbus调用）"""

    def __init__(self, node: Node):
        self.node = node
        self.modbus_id = 0
        self.target_pos = 1000.0
        self.current_pos = 1000.0

        # 创建Modbus服务客户端
        self.cli_modbus_create = node.create_client(
            ModbusCreate, "/dobot_bringup_v3/srv/ModbusCreate"
        )
        self.cli_modbus_close = node.create_client(
            ModbusClose, "/dobot_bringup_v3/srv/ModbusClose"
        )
        self.cli_set_hold_regs = node.create_client(
            SetHoldRegs, "/dobot_bringup_v3/srv/SetHoldRegs"
        )
        self.cli_get_hold_regs = node.create_client(
            GetHoldRegs, "/dobot_bringup_v3/srv/GetHoldRegs"
        )

        self._init_modbus()

    def _call_service(self, client, request):
        if not client.service_is_ready():
            return None
        future = client.call_async(request)
        while not future.done():
            rclpy.spin_once(self.node, timeout_sec=0.001)
        return future.result()

    def _init_modbus(self):
        logging.info("Waiting for Modbus services...")
        self.cli_modbus_create.wait_for_service(timeout_sec=5.0)

        logging.info("Closing old Modbus connections...")
        for i in range(1, 5):
            req = ModbusClose.Request()
            req.index = i
            self._call_service(self.cli_modbus_close, req)

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
            self._write_reg(256, 1, "1")  # Enable
            self._write_reg(257, 1, "60")  # Force/Speed
        else:
            logging.error(f"Gripper ModbusCreate failed: {res}")
            self.modbus_id = 0

        if self.modbus_id > 0:
            init_val = self._read_reg(514)
            if init_val is not None:
                self.current_pos = float(init_val)
                logging.info(f"Gripper initial position: {self.current_pos}")

    def _write_reg(self, addr: int, count: int, val_str: str):
        if self.modbus_id <= 0:
            return
        req = SetHoldRegs.Request()
        req.index = self.modbus_id
        req.addr = addr
        req.count = count
        req.val_tab = val_str
        self._call_service(self.cli_set_hold_regs, req)

    def _read_reg(self, addr: int):
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
        self.target_pos = max(0.0, min(1000.0, float(pos)))
        self._write_reg(259, 1, str(int(self.target_pos)))

    def read_position(self) -> float:
        real_pos = self._read_reg(514)
        if real_pos is not None:
            self.current_pos = float(real_pos)
        return self.current_pos


class CR5InferenceNode(Node):
    """CR5推理ROS2节点"""

    def __init__(self, args: Args):
        super().__init__("cr5_inference_node")
        self.args = args

        self.robot_state = None
        self.latest_image = None
        self.step_count = 0
        self.bridge = CvBridge()

        self.gripper: GripperController | None = None
        self.policy_client = None

        # 服务客户端
        self.cli_enable = self.create_client(EnableRobot, "/dobot_bringup_v3/srv/EnableRobot")
        self.cli_clear_error = self.create_client(ClearError, "/dobot_bringup_v3/srv/ClearError")
        self.cli_speed_factor = self.create_client(SpeedFactor, "/dobot_bringup_v3/srv/SpeedFactor")
        self.cli_servo_p = self.create_client(ServoP, "/dobot_bringup_v3/srv/ServoP")
        self.cli_get_pose = self.create_client(GetPose, "/dobot_bringup_v3/srv/GetPose")

        # QoS配置（depth=1 保证数据最新）
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        robot_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # 订阅器
        self.sub_image = self.create_subscription(
            Image, "/camera/color/image_raw", self.image_callback, sensor_qos
        )
        self.sub_robot_current = self.create_subscription(
            ToolVectorActual,
            "/dobot_msgs_v3/msg/ToolVectorActual",
            self.robot_current_callback,
            robot_qos,
        )

        logging.info("CR5InferenceNode initialized")

    def call_service(self, client, request):
        if not client.service_is_ready():
            return None
        future = client.call_async(request)
        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.001)
        return future.result()

    def robot_current_callback(self, msg: ToolVectorActual):
        if self.robot_state is None:
            logging.info("Robot state callback received first message")
        self.robot_state = np.array(
            [msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz], dtype=np.float32
        )

    def image_callback(self, msg: Image):
        if self.latest_image is None:
            logging.info("Image callback received first message")
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            logging.error(f"Image conversion failed: {e}")

    def build_observation(self) -> dict | None:
        """构建观测数据（与 DobotCR3Inputs 期望格式一致）"""
        if self.robot_state is None or self.latest_image is None:
            return None

        # 图像预处理：BGR→RGB, resize到224x224, uint8
        img = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB)
        img = image_tools.resize_with_pad(img, self.args.resize_size, self.args.resize_size)
        img = image_tools.convert_to_uint8(img)

        # 读取夹爪位置（原始值 0-1000）
        gripper_pos = self.gripper.read_position() if self.gripper else 1000.0

        # 状态向量：7D = 6D末端位姿(mm,度) + 1D夹爪开度(原始值)
        state = np.concatenate([self.robot_state, [gripper_pos]]).astype(np.float32)

        return {
            "observation/image": img,
            "observation/state": state,
            "prompt": self.args.prompt,
        }

    def execute_action(self, action: np.ndarray, update_gripper: bool = True):
        """执行单个动作，返回耗时(ms)"""
        target_pose = [float(action[i]) for i in range(6)]
        gripper_target = int(np.clip(float(action[6]), 0, 1000))

        if self.args.dry_run:
            logging.info(f"[DRY RUN] pose={target_pose}, gripper={gripper_target}")
            return 0.0, 0.0

        # ServoP
        t0 = time.perf_counter()
        req = ServoP.Request()
        req.x, req.y, req.z = target_pose[0], target_pose[1], target_pose[2]
        req.rx, req.ry, req.rz = target_pose[3], target_pose[4], target_pose[5]
        self.call_service(self.cli_servo_p, req)
        servo_ms = (time.perf_counter() - t0) * 1000

        # 夹爪（仅在需要时更新）
        gripper_ms = 0.0
        if update_gripper and self.gripper:
            t1 = time.perf_counter()
            self.gripper.set_target(gripper_target)
            gripper_ms = (time.perf_counter() - t1) * 1000

        return servo_ms, gripper_ms

    def enable_robot(self):
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
        res = self.call_service(self.cli_get_pose, GetPose.Request())
        if res and hasattr(res, "pose"):
            matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", res.pose)
            if len(matches) >= 6:
                return [float(x) for x in matches[:6]]
        return None


def run_inference(args: Args):
    """运行推理主循环"""
    logging.info("=" * 60)
    logging.info("CR5 Inference Client (Single-threaded)")
    logging.info("=" * 60)
    logging.info(f"Server: {args.host}:{args.port}")
    logging.info(f"Prompt: {args.prompt}")
    logging.info(f"Action horizon: {args.action_horizon}")
    logging.info(f"Max steps: {args.max_steps}")
    logging.info(f"Dry run: {args.dry_run}")
    logging.info("=" * 60)

    rclpy.init()
    node = CR5InferenceNode(args)

    FPS = 30
    dt = 1.0 / FPS

    try:
        if not args.dry_run:
            node.enable_robot()

        logging.info("Initializing gripper controller...")
        node.gripper = GripperController(node)

        logging.info("Waiting for camera and robot data...")
        wait_start = time.time()
        while node.robot_state is None or node.latest_image is None:
            rclpy.spin_once(node, timeout_sec=0.01)
            if time.time() - wait_start > 10.0:
                raise RuntimeError("Sensor data not available")
        logging.info("Sensor data available")

        if not args.dry_run:
            initial_pose = node.get_current_pose()
            if initial_pose:
                logging.info(f"Initial pose: {initial_pose}")

        logging.info(f"Connecting to policy server at {args.host}:{args.port}...")
        node.policy_client = _websocket_client_policy.WebsocketClientPolicy(
            args.host, args.port
        )
        logging.info("Connected to policy server")

        metadata = node.policy_client.get_server_metadata()
        logging.info(f"Server metadata: {metadata}")

        logging.info("=" * 60)
        logging.info("Starting inference... Press Ctrl+C to stop")
        logging.info("=" * 60)

        # 主循环
        while node.step_count < args.max_steps:
            # 1. spin_once更新传感器数据
            rclpy.spin_once(node, timeout_sec=0.001)

            # 2. 构建观测
            obs = node.build_observation()
            if obs is None:
                continue

            # 3. 推理获取动作块
            t_infer_start = time.perf_counter()
            action_chunk = node.policy_client.infer(obs)["actions"]
            infer_ms = (time.perf_counter() - t_infer_start) * 1000

            # 4. 执行动作（每5次ServoP更新一次夹爪，保持30Hz频率）
            t_exec_start = time.perf_counter()
            actions_executed = 0
            gripper_update_interval = 5
            for t in range(min(args.action_horizon, len(action_chunk))):
                if node.step_count >= args.max_steps:
                    break

                start_loop_t = time.perf_counter()

                action = action_chunk[t]
                # 每5次更新一次夹爪，使用当前动作的最新夹爪目标值
                update_gripper = ((t + 1) % gripper_update_interval == 0) or (t == min(args.action_horizon, len(action_chunk)) - 1)
                servo_ms, gripper_ms = node.execute_action(action, update_gripper=update_gripper)
                logging.info(f"[Action {node.step_count}] servo={servo_ms:.1f}ms, gripper={gripper_ms:.1f}ms")
                node.step_count += 1
                actions_executed += 1

                # 严格保持30Hz
                elapsed = time.perf_counter() - start_loop_t
                remaining = dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)

            exec_ms = (time.perf_counter() - t_exec_start) * 1000
            logging.info(f"[Step {node.step_count}] infer={infer_ms:.1f}ms, exec={exec_ms:.1f}ms ({actions_executed} actions)")

        logging.info(f"Inference stopped after {node.step_count} steps")

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error(f"Error: {e}")
        raise
    finally:
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
