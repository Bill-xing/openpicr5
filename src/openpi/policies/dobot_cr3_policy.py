import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """将输入图像统一转换为 uint8 的 HWC 格式。

    兼容情况：
    - 输入可能是 float32（通常范围在 [0, 1]），这里转换到 [0, 255] 并转为 uint8。
    - 输入可能是 CHW 格式（3, H, W），这里转换为 HWC 格式（H, W, 3）。
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DobotCR3Inputs(transforms.DataTransformFn):
    """Dobot CR3 数据集的输入转换（训练与推理共用）。

    依据录制与转换脚本（recorder_optimized.py / convert_to_lerobot.py）的输出格式：
    - observation.state: 7D (6D 末端位姿 + 1D 夹爪开度)
    - observation.image: 顶视角图像（由 observation.images.top 重映射而来）
    - prompt: 由 tasks.jsonl 生成
    """

    # 模型类型（影响 image_mask 的占位策略）
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        """将单条样本 dict 转为模型期望的输入结构。"""
        # 图像解析：LeRobot 常以 float32 + CHW 存储
        base_image = _parse_image(data["observation/image"])

        inputs = {
            # 机器人本体状态（7D）
            "state": data["observation/state"],
            # 图像输入：主视角 + 腕部视角（零数组占位）
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            # 图像有效性掩码
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,  # 屏蔽，因为没有腕部图像
                "right_wrist_0_rgb": np.False_,
            },
        }

        # 训练阶段才有 actions；推理阶段通常无此字段
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # 透传 prompt（由 tasks.jsonl 生成）
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DobotCR3Outputs(transforms.DataTransformFn):
    """Dobot CR3 的输出转换（仅推理使用）。

    模型输出动作会被 padding 到统一维度，这里裁剪回 7D：
    (x, y, z, rx, ry, rz, gripper)
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}