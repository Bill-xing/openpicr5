# CR5 推理客户端设计文档

## 概述

`main.py` 实现了一个用于 Dobot CR5 机械臂的实时推理客户端，将训练好的策略模型部署到真实机器人上执行任务。

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              系统架构                                    │
└─────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────┐
                         │   策略服务器      │
                         │  (WebSocket)     │
                         │  localhost:8000  │
                         └────────▲─────────┘
                                  │
                    观测数据上传 │ │ 动作序列返回
                    (obs dict)   │ │ (actions)
                                  │ │
┌─────────────────────────────────┴─┴─────────────────────────────────────┐
│                         CR5 推理客户端                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     CR5InferenceNode                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   │
│  │  │ 图像订阅    │  │ 状态订阅    │  │ 动作执行                 │  │   │
│  │  │ 30Hz       │  │ 100Hz      │  │ ServoP + 话题发布        │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   GripperController                              │   │
│  │                   独立线程 50Hz Modbus通信                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                    ROS2服务调用  │  ROS2话题发布
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           硬件层                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ RealSense   │  │ CR5机械臂   │  │ Modbus夹爪  │  │ Dobot驱动   │    │
│  │ 相机 30Hz   │  │ ServoP控制  │  │ 127.0.0.1   │  │ ROS2服务    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 线程模型

### 三线程架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            线程模型                                      │
└─────────────────────────────────────────────────────────────────────────┘

线程1: 主线程 (run_inference)
├── 初始化ROS2
├── 创建节点
├── 启动其他线程
├── 连接策略服务器
├── 监控推理状态
└── 清理资源

线程2: ROS2 Spin线程 (daemon=True)
├── 处理 /camera/color/image_raw 回调 (30Hz)
│   └── image_callback() → 触发推理流程
├── 处理 /dobot_msgs_v3/msg/ToolVectorActual 回调 (100Hz)
│   └── robot_current_callback() → 更新机器人状态
└── 持续运行直到节点销毁

线程3: 夹爪控制线程 (daemon=True)
├── 50Hz 发送目标位置到Modbus寄存器
├── 10Hz 读取实际位置反馈
└── 独立于主推理循环，保证夹爪控制稳定性
```

### 线程间通信

```python
# 共享变量（无锁，原子操作）
self.robot_state          # 线程2写，线程2读（image_callback中）
self.latest_image         # 线程2写，线程1读（等待初始数据）
self.inference_enabled    # 线程1写，线程2读
self.step_count          # 线程2写，线程1读

# GripperController共享变量
gripper_controller.target_pos   # 线程2写，线程3读
gripper_controller.current_pos  # 线程3写，线程2读
```

### 为什么使用三线程？

| 线程 | 原因 |
|------|------|
| 主线程 | 负责初始化和资源管理，保持程序入口清晰 |
| ROS2 Spin线程 | `rclpy.spin()` 是阻塞调用，必须独立线程运行 |
| 夹爪控制线程 | Modbus通信与推理解耦，保证50Hz稳定控制频率 |

---

## 数据流详解

### 推理触发流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         推理触发流程 (30Hz)                              │
└─────────────────────────────────────────────────────────────────────────┘

相机发布图像 (/camera/color/image_raw, 30Hz)
            │
            ▼
    image_callback(msg)  [第304行]
            │
            ├── 检查 inference_enabled 标志
            │
            ├── 检查 step_count < max_steps
            │
            ├── cv_bridge 转换图像 (ROS Image → OpenCV BGR)
            │
            ▼
    _build_observation(cv_image)  [第341行]
            │
            ├── BGR → RGB 颜色转换
            ├── resize_with_pad(224, 224)
            ├── 读取 robot_state (6维位姿)
            ├── 读取 gripper_controller.current_pos
            ├── 拼接状态向量 (7维)
            │
            ▼
        返回 obs 字典:
        {
            "observation/image": (224,224,3) uint8,
            "observation/wrist_image": (224,224,3) uint8,
            "observation/state": (7,) float32,
            "prompt": str
        }
            │
            ▼
    _get_action(obs)  [第374行]
            │
            ├── 检查 action_queue 是否为空
            │   │
            │   ├── 空 → policy_client.infer(obs)
            │   │        返回 actions (10, 7)
            │   │        取前5个放入队列
            │   │
            │   └── 非空 → 跳过推理
            │
            ├── action_queue.popleft()
            │
            ▼
        返回 action (7,)
            │
            ▼
    _execute_action(action)  [第395行]
            │
            ├── 提取位姿 [x,y,z,rx,ry,rz]
            ├── 提取夹爪目标 (归一化→0-1000)
            │
            ├── _servo_p() → 发送ServoP服务请求
            ├── _publish_robot_target() → 发布目标位姿话题
            ├── gripper_controller.set_target() → 设置夹爪目标
            └── _publish_gripper_state() → 发布夹爪状态话题
```

### 观测数据格式

```python
obs = {
    # 主相机图像
    "observation/image": np.ndarray,      # shape=(224,224,3), dtype=uint8, RGB

    # 腕部相机图像（本系统无腕部相机，复用主相机）
    "observation/wrist_image": np.ndarray, # shape=(224,224,3), dtype=uint8, RGB

    # 状态向量
    "observation/state": np.ndarray,       # shape=(7,), dtype=float32
    # [0:3] = x,y,z (mm)         - 末端位置
    # [3:6] = rx,ry,rz (度)      - 末端姿态（欧拉角）
    # [6]   = gripper (0-1)      - 夹爪开度（归一化）

    # 任务描述
    "prompt": str                          # 例如 "pick up the red block"
}
```

### 动作数据格式

```python
# 服务器返回
result = {
    "actions": np.ndarray  # shape=(10,7), dtype=float32
}

# 单个动作
action = np.ndarray  # shape=(7,), dtype=float32
# [0:3] = x,y,z (mm)         - 目标末端位置
# [3:6] = rx,ry,rz (度)      - 目标末端姿态
# [6]   = gripper (0-1)      - 目标夹爪开度
```

---

## 动作队列与重规划机制

### 设计原理

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      动作队列与重规划机制                                │
└─────────────────────────────────────────────────────────────────────────┘

问题：
  - 策略模型每次推理返回 action_horizon=10 个动作
  - 但每帧只能执行1个动作
  - 如果每帧都调用推理，浪费计算资源且延迟高

解决方案：动作队列 + 重规划
  - 每次推理后，取前 replan_steps=5 个动作放入队列
  - 每帧从队列取1个动作执行
  - 队列空时重新调用推理

时序图：
  帧1  帧2  帧3  帧4  帧5  帧6  帧7  帧8  帧9  帧10
   │    │    │    │    │    │    │    │    │    │
   ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
  推理  执行  执行  执行  执行  推理  执行  执行  执行  执行
   │                          │
   └─ 返回[a0,a1,a2,a3,a4]    └─ 返回[b0,b1,b2,b3,b4]
      执行a0                     执行b0
         执行a1                     执行b1
            执行a2                     ...
               执行a3
                  执行a4
```

### 代码实现

```python
def _get_action(self, obs: dict):
    # 队列为空时才调用推理
    if not self.action_queue:
        result = self.policy_client.infer(obs)
        actions = result["actions"]  # (10, 7)

        # 只取前 replan_steps 个动作
        for i in range(min(self.args.replan_steps, len(actions))):
            self.action_queue.append(actions[i])

    # 从队列取出一个动作
    if self.action_queue:
        return self.action_queue.popleft()
    return None
```

### 参数选择

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `action_horizon` | 10 | 模型输出的动作序列长度（由训练配置决定） |
| `replan_steps` | 5 | 每次取多少个动作放入队列 |

**为什么 replan_steps=5？**
- 太小（如1）：每帧都推理，延迟高
- 太大（如10）：动作过时，无法响应环境变化
- 5是折中：约166ms重规划一次（5帧×33ms/帧）

---

## 类设计详解

### Args 数据类

```python
@dataclasses.dataclass
class Args:
    """命令行参数配置"""

    # 策略服务器
    host: str = "localhost"    # 服务器地址
    port: int = 8000           # 服务器端口

    # 图像处理
    resize_size: int = 224     # 模型期望的图像尺寸

    # 重规划
    replan_steps: int = 5      # 每次取多少个动作

    # 任务
    prompt: str = "pick up the object"  # 任务描述

    # 执行控制
    max_steps: int = 1000      # 最大执行步数
    dry_run: bool = False      # 空运行模式
```

### GripperController 类

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GripperController                                 │
│                        (threading.Thread)                                │
└─────────────────────────────────────────────────────────────────────────┘

职责：
  - 管理与夹爪的Modbus通信
  - 独立于主推理循环运行
  - 保证稳定的控制频率

初始化流程 (_init_modbus):
  1. 创建Modbus服务客户端
  2. 关闭旧连接 (索引1-4)
  3. 创建新连接 (ip=127.0.0.1, port=60000, slave_id=1)
  4. 初始化夹爪 (Enable + Force/Speed)
  5. 读取初始位置

控制循环 (run):
  while running:
      _write_reg(259, target_pos)     # 发送目标位置
      if counter % 5 == 0:
          current_pos = _read_reg(514) # 读取实际位置
      sleep(0.02)                      # 50Hz

Modbus寄存器映射：
  | 地址 | 功能 | 值范围 |
  |------|------|--------|
  | 256  | Enable | 0/1 |
  | 257  | Force/Speed | 0-100 |
  | 259  | 目标位置(写) | 0-1000 |
  | 514  | 实际位置(读) | 0-1000 |
```

### CR5InferenceNode 类

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CR5InferenceNode                                  │
│                        (rclpy.node.Node)                                 │
└─────────────────────────────────────────────────────────────────────────┘

订阅话题：
  ┌────────────────────────────────────┬─────────────┬─────────────────┐
  │ 话题                               │ 消息类型     │ 频率   │ 回调   │
  ├────────────────────────────────────┼─────────────┼────────┼────────┤
  │ /camera/color/image_raw            │ Image       │ 30Hz   │ image_callback │
  │ /dobot_msgs_v3/msg/ToolVectorActual│ ToolVector  │ 100Hz  │ robot_current_callback │
  └────────────────────────────────────┴─────────────┴────────┴────────┘

发布话题：
  ┌────────────────────────────────────┬─────────────┬─────────────────┐
  │ 话题                               │ 消息类型     │ 说明            │
  ├────────────────────────────────────┼─────────────┼─────────────────┤
  │ /robot/target_pose                 │ ToolVector  │ 目标位姿        │
  │ /gripper/command_update            │ PointStamped│ 夹爪目标        │
  │ /gripper/state_feedback            │ PointStamped│ 夹爪实际状态    │
  └────────────────────────────────────┴─────────────┴─────────────────┘

服务客户端：
  ┌────────────────────────────────────┬─────────────────────────────────┐
  │ 服务                               │ 用途                            │
  ├────────────────────────────────────┼─────────────────────────────────┤
  │ /dobot_bringup_v3/srv/EnableRobot  │ 使能机器人                      │
  │ /dobot_bringup_v3/srv/ClearError   │ 清除错误                        │
  │ /dobot_bringup_v3/srv/SpeedFactor  │ 设置速度因子                    │
  │ /dobot_bringup_v3/srv/ServoP       │ 位置伺服控制                    │
  │ /dobot_bringup_v3/srv/GetPose      │ 获取当前位姿                    │
  └────────────────────────────────────┴─────────────────────────────────┘

核心方法：
  image_callback()       - 图像回调，触发推理流程
  robot_current_callback() - 更新机器人状态
  _build_observation()   - 构建观测字典
  _get_action()          - 获取动作（含重规划逻辑）
  _execute_action()      - 执行动作
  _servo_p()             - 发送ServoP指令
  enable_robot()         - 使能机器人
```

---

## 与训练数据的一致性

### 数据对比

| 项目 | 训练时 (recorder_optimized.py) | 推理时 (main.py) |
|------|-------------------------------|------------------|
| 图像频率 | 30Hz | 30Hz（相机触发） |
| 图像原始尺寸 | 640×480 | 640×480 |
| 图像处理后尺寸 | 224×224 | 224×224 |
| 状态维度 | 7D (6D位姿 + 1D夹爪) | 7D |
| 动作维度 | 7D (6D位姿 + 1D夹爪) | 7D |
| 夹爪值范围 | 0-1 (归一化) | 0-1 (归一化) |
| 位姿单位 | mm, 度 | mm, 度 |

### 图像处理流程对比

```
训练时 (transforms):
  原始图像 → resize_with_pad(224,224) → normalize → 模型输入

推理时 (main.py):
  原始图像 → BGR→RGB → resize_with_pad(224,224) → uint8 → 发送服务器
                                                           ↓
                                                    服务器端normalize
```

---

## 话题兼容性设计

### 与 data_collector4.py 的兼容

本客户端发布的话题与遥操作系统 (data_collector4.py) 相同：

```
data_collector4.py (遥操作) 发布:
  /robot/target_pose        ← 本客户端也发布
  /gripper/command_update   ← 本客户端也发布
  /gripper/state_feedback   ← 本客户端也发布

recorder_optimized.py (录制) 订阅:
  /robot/target_pose        ← 可以录制本客户端的动作
  /gripper/command_update   ← 可以录制本客户端的夹爪指令
  /gripper/state_feedback   ← 可以录制本客户端的夹爪状态
```

**好处**：可以用 recorder_optimized.py 录制推理过程，用于调试或数据增强。

---

## 错误处理与安全机制

### 超时保护

```python
# 等待传感器数据超时
wait_start = time.time()
while node.robot_state is None or node.latest_image is None:
    if time.time() - wait_start > 10.0:
        raise RuntimeError("Sensor data not available")
    time.sleep(0.1)
```

### 最大步数限制

```python
if self.step_count >= self.args.max_steps:
    logging.info(f"Reached max steps ({self.args.max_steps}), stopping...")
    self.inference_enabled = False
    return
```

### 空运行模式

```python
if self.args.dry_run:
    logging.info(f"[DRY RUN] pose={target_pose}, gripper={gripper_target}")
    return  # 不发送实际指令
```

### 资源清理

```python
finally:
    # 停止夹爪控制器
    if hasattr(node, "gripper_controller"):
        node.gripper_controller.stop()
        node.gripper_controller.join(timeout=1.0)

    # 关闭节点
    node.destroy_node()
    rclpy.shutdown()
```

---

## 性能考虑

### 推理延迟

```
单次推理流程耗时估计：
  图像转换 (cv_bridge):     ~1-2ms
  图像预处理 (resize):      ~2-3ms
  WebSocket通信:           ~5-10ms
  模型推理 (服务器端):      ~20-50ms
  动作执行 (ServoP):        ~1-2ms
  ────────────────────────────────
  总计:                    ~30-70ms

30Hz要求: 33ms/帧
实际: 每5帧推理一次，所以平均 (30-70)/5 + 5 ≈ 11-19ms/帧 ✓
```

### 夹爪控制独立性

夹爪控制线程独立运行，不受推理延迟影响：
- 推理慢 → 夹爪仍保持50Hz控制
- 推理失败 → 夹爪保持当前目标位置

---

## 使用示例

### 基本使用

```bash
python examples/dobot_cr5/main.py \
    --host localhost \
    --port 8000 \
    --prompt "pick up the red block"
```

### 空运行测试

```bash
python examples/dobot_cr5/main.py \
    --dry-run \
    --prompt "test task"
```

### 自定义参数

```bash
python examples/dobot_cr5/main.py \
    --host 192.168.1.100 \
    --port 8080 \
    --replan-steps 3 \
    --max-steps 500 \
    --prompt "place the cup on the table"
```

---

## 调试方法

### 话题监控

```bash
# 查看目标位姿
ros2 topic echo /robot/target_pose

# 检查发布频率
ros2 topic hz /robot/target_pose

# 查看夹爪状态
ros2 topic echo /gripper/state_feedback
```

### 日志输出

程序会输出：
- 初始化进度
- 连接状态
- 每30步的执行状态
- 错误信息

```
2024-01-28 10:00:00 [INFO] CR5 Inference Client
2024-01-28 10:00:00 [INFO] Server: localhost:8000
2024-01-28 10:00:01 [INFO] Robot enabled
2024-01-28 10:00:02 [INFO] Gripper connected, Modbus ID: 1
2024-01-28 10:00:03 [INFO] Connected to policy server
2024-01-28 10:00:03 [INFO] Starting inference...
2024-01-28 10:00:04 [INFO] Step 30, queue size: 2
2024-01-28 10:00:05 [INFO] Step 60, queue size: 4
...
```
