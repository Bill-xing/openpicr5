# ServoP 延迟问题调查报告

## 问题描述

在 VLA 推理客户端 `main.py` 中，ServoP 服务调用延迟约为 **90-100ms**，导致控制频率只能达到 **~10Hz**。

而在数据采集脚本 `data_collector4.py` 中，相同的 ServoP 服务调用延迟仅为 **~11ms**，可以实现 **~90Hz** 的控制频率。

## 环境信息

- 机器人：Dobot CR5
- ROS 版本：ROS2
- 控制方式：ServoP 伺服模式
- 目标频率：30Hz（与训练数据一致）

## 代码架构对比

### data_collector4.py（11ms 延迟）

```
单进程架构：
├── 主线程：控制循环（100Hz）+ ServoP 调用
├── spin 线程：rclpy.spin(node)
├── GripperManager 线程：夹爪控制
└── GripperStateFeedback 线程：夹爪状态发布

关键特点：
- 只有一个 ROS 节点
- 节点只有服务客户端和发布器，无订阅器
- 主循环以 100Hz 持续调用 ServoP（即使位姿不变）
```

### main.py（90-100ms 延迟）

```
单进程架构：
├── 主线程：推理循环（目标30Hz）+ ServoP 调用
├── MultiThreadedExecutor：4个线程处理订阅和服务
├── 图像订阅：30Hz RGB 图像
├── 状态订阅：机器人位姿反馈
├── GripperStateFeedback 线程：夹爪状态发布
└── 其他服务客户端

关键特点：
- 有图像和状态订阅器（大数据量）
- 推理延迟约 60-80ms，导致 ServoP 调用频率较低
```

## 尝试的方案

### 方案 1：MultiThreadedExecutor + 回调组分离

**思路**：将 ServoP 服务放入独立的回调组，与图像/状态订阅分离

**实现**：
```python
self.servo_cb_group = MutuallyExclusiveCallbackGroup()
self.image_cb_group = MutuallyExclusiveCallbackGroup()
self.state_cb_group = MutuallyExclusiveCallbackGroup()
```

**结果**：❌ 无效，ServoP 延迟仍为 ~100ms

**原因**：回调组只影响回调执行的并发性，不影响服务调用的网络延迟

---

### 方案 2：ServoNode 独立节点

**思路**：创建专用的 ServoNode 节点，只包含 ServoP 服务客户端，无订阅器

**实现**：
```python
class ServoNode(Node):
    def __init__(self):
        # 只有 ServoP 服务客户端，无订阅器
        self.cli_servo_p = self.create_client(ServoP, "/dobot_bringup_v3/srv/ServoP")
```

**结果**：❌ 无效，ServoP 延迟仍为 ~100ms

**原因**：虽然 ServoNode 独立，但仍在同一进程中，共享 rclpy 上下文和网络栈

---

### 方案 3：spin_until_future_complete vs 手动轮询

**思路**：尝试不同的服务调用方式

**实现**：
```python
# 方式 A：spin_until_future_complete
rclpy.spin_until_future_complete(self, future)

# 方式 B：手动轮询（data_collector4.py 使用的方式）
while not future.done():
    time.sleep(0.001)
```

**结果**：❌ 两种方式延迟相似，都是 ~100ms

---

### 方案 4：ServoPThread 高频发送线程

**思路**：在独立线程中持续高频发送 ServoP，主循环只更新目标位姿（非阻塞）

**实现**：
```python
class ServoPThread(threading.Thread):
    def run(self):
        while self.running:
            # 持续发送当前目标位姿
            self.servo_node.call_service(self.servo_node.cli_servo_p, req)
```

**结果**：⚠️ ServoP 延迟 ~140ms，且产生严重的锯齿轨迹

**原因**：
- 延迟更高（可能是 GIL 竞争）
- 非阻塞模式导致模型看到运动中的中间状态，预测不连续

---

### 方案 5：独立进程（multiprocessing）

**思路**：将 ServoP 放入完全独立的 Python 进程，拥有独立的 rclpy 上下文

**实现**：
```python
def servo_process_worker(cmd_queue, result_queue, stop_event):
    rclpy.init()  # 独立的 rclpy 上下文
    node = rclpy.create_node("servo_process_node")
    # 持续高频发送 ServoP
    while not stop_event.is_set():
        if cli_servo_p.service_is_ready():
            future = cli_servo_p.call_async(req)
            while not future.done():
                time.sleep(0.001)
```

**结果**：⚠️ 部分改善
- 独立进程 ServoP 延迟：~48ms（从 90ms 降低）
- 主进程等待时间：~85ms（队列同步开销）
- 控制频率仍为 ~10Hz

---

### 方案 6：独立进程 + 高频持续发送

**思路**：独立进程即使没有新命令也持续发送当前位姿，保持控制器在"连续伺服模式"

**实现**：
```python
while not stop_event.is_set():
    # 非阻塞检查新命令
    try:
        new_pose = cmd_queue.get_nowait()
        current_pose = new_pose
        pending_result = True
    except:
        pass  # 继续使用当前位姿

    # 持续发送 ServoP
    cli_servo_p.call_async(req)
```

**结果**：⚠️ 独立进程 ServoP 延迟降至 ~48ms，但未达到预期的 11ms

## 关键发现

### 1. Dobot 控制器的"连续伺服模式"

data_collector4.py 能达到 11ms 延迟的可能原因：
- 主循环以 100Hz 持续调用 ServoP
- 控制器检测到高频指令后进入"快速响应模式"
- 没有图像订阅等大数据量操作的干扰

### 2. 图像订阅的影响

main.py 中的 30Hz 图像订阅（RGB，约 1MB/帧）可能导致：
- DDS 网络栈拥塞
- rclpy 内部资源竞争
- 即使使用独立进程，网络层仍可能受影响

### 3. 调用频率的影响

| 调用频率 | ServoP 延迟 |
|---------|------------|
| ~100Hz（data_collector4.py） | ~11ms |
| ~20Hz（独立进程持续发送） | ~48ms |
| ~10Hz（按需调用） | ~90ms |

这表明 Dobot 控制器的响应时间与调用频率相关。

## 性能对比总结

| 方案 | ServoP 延迟 | 控制频率 | 轨迹质量 | 任务完成 |
|------|------------|---------|---------|---------|
| 原始阻塞模式 | ~90ms | ~10Hz | ✅ 好 | ✅ 是 |
| 回调组分离 | ~100ms | ~10Hz | ✅ 好 | ✅ 是 |
| ServoNode 独立节点 | ~93ms | ~10Hz | ✅ 好 | ✅ 是 |
| ServoPThread（非阻塞） | ~140ms | ~26Hz | ❌ 锯齿 | ❌ 差 |
| 独立进程（阻塞） | ~48ms (进程内) / ~85ms (主进程) | ~10Hz | ✅ 好 | ✅ 是 |

## 结论

1. **无法在 main.py 中复现 data_collector4.py 的 11ms 延迟**
   - 根本原因可能是图像订阅导致的网络/资源干扰
   - Dobot 控制器需要高频持续调用才能进入快速响应模式

2. **阻塞模式是最佳选择**
   - 虽然只有 10Hz，但轨迹质量好，任务能完成
   - 非阻塞模式会导致锯齿轨迹

3. **独立进程方案有一定改善**
   - ServoP 延迟从 90ms 降至 48ms
   - 但队列同步开销抵消了部分收益

## 后续建议

1. **短期**：接受 10Hz 的控制频率，使用简单的阻塞模式

2. **中期**：
   - 研究 Dobot 固件是否有快速响应模式的配置
   - 尝试降低图像分辨率或帧率
   - 尝试使用 UDP 而非 TCP 的 DDS 配置

3. **长期**：
   - 考虑使用 C++ 实现 ServoP 调用部分
   - 研究是否可以使用 Dobot 的实时以太网接口（如 EtherCAT）
   - 评估是否需要 30Hz 控制频率，或 10Hz 已足够

## 相关文件

- `examples/dobot_cr5/main.py` - VLA 推理客户端
- `dobot_demo/dobot_demo/data_collector4.py` - 数据采集脚本（参考实现）
