
使用的命令
一个终端
python examples/dobot_cr5/main.py       --args.prompt "Pick up the red block and place it on the plate"       --args.max-steps 400
另一个终端
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_dobot_cr3 --policy.dir=/home/hit/openpi/checkpoints/pi0_dobot_cr3/21000



# CR5 真机推理客户端





用于在 Dobot CR5 真机上执行策略推理的客户端。

## 功能

- 连接到远程策略服务器（WebSocket）
- 以 30Hz 频率执行策略推理（与训练数据一致）
- 控制机器人和夹爪
- 发布状态话题（与数据采集兼容）

## 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                       CR5 推理客户端                                  │
└─────────────────────────────────────────────────────────────────────┘

                    订阅                          发布
                    ────                          ────
/camera/color/image_raw (30Hz) ────┐      ┌──► /robot/target_pose
/dobot_msgs_v3/msg/ToolVectorActual ──┐  │    /gripper/command_update
                                      │  │    /gripper/state_feedback
                                      ▼  ▼
                              ┌──────────────┐
         图像到达触发推理 ───►│  推理循环     │───► ServoP服务调用
                              │  (30Hz)      │     Modbus夹爪控制
                              └──────────────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │ WebSocket    │
                              │ 策略服务器    │
                              └──────────────┘
```

## 启动步骤

### 1. 启动 ROS2 驱动

```bash
cd /home/xing/openpi-main/dobot_ws
source install/setup.bash
ros2 launch dobot_bringup_v3 dobot_bringup_ros2.launch.py
```

### 2. 启动相机

```bash
./start_camera_shm.sh
```

### 3. 启动策略服务器

```bash
cd /home/xing/openpi-main
  uv run python scripts/serve_policy.py policy:checkpoint \
      --policy.config pi0_dobot_cr3 \
      --policy.dir /home/hit/openpi/checkpoints/pi0_dobot_cr3/21000
```

### 4. 启动推理客户端

```bash
cd /home/xing/openpi-main
source /home/xing/openpi-main/dobot_ws/install/setup.bash
python examples/dobot_cr5/main.py \
    --host localhost \
    --port 8000 \
    --prompt "Pick up the red block and place it on the plate"
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | localhost | 策略服务器地址 |
| `--port` | 8000 | 策略服务器端口 |
| `--resize-size` | 224 | 图像尺寸 |
| `--replan-steps` | 5 | 每N步重新规划 |
| `--max-steps` | 1000 | 最大执行步数 |
| `--prompt` | "pick up the object" | 任务描述 |
| `--dry-run` | False | 空运行模式（只打印不执行）|

## 空运行测试

在不连接机器人的情况下测试推理流程：

```bash
python examples/dobot_cr5/main.py --dry-run --prompt "test task"
```

## 验证

```bash
# 检查话题发布
ros2 topic echo /robot/target_pose

# 检查频率
ros2 topic hz /robot/target_pose  # 应该约30Hz
```

## 数据一致性

| 数据 | 训练时 | 推理时 |
|------|--------|--------|
| 图像频率 | 30Hz | 30Hz（相机触发） |
| 图像原始大小 | 640x480 | 640x480 |
| 图像处理后 | 224x224 | 224x224 |
| 状态维度 | 7D (6D位姿+1D夹爪) | 7D |
| 动作维度 | 7D (6D位姿+1D夹爪) | 7D |
| 夹爪范围 | 0-1 (归一化) | 0-1 (归一化) |
