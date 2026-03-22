"""
Teacher模型
使用YOLOv8官方模型作为Teacher，提供特征提取和监督信号
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class TeacherModel(nn.Module):
    """
    Teacher模型

    使用YOLOv8官方模型作为Teacher，提供特征提取和监督信号

    注意：
    1. Teacher模型使用YOLOv8官方实现
    2. 通过ultralytics库加载
    3. 参数应该被冻结（在Student模型中处理）
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = 'yolov8n.pt',
        device: str = 'cpu',
    ):
        """
        参数:
            model_path: 模型权重路径（如果提供，则加载权重）
            model_name: 模型名称（用于ultralytics加载）
            device: 设备
        """
        super().__init__()

        self.device = device
        self.model_name = model_name

        # 加载YOLOv8模型
        if model_path is not None and Path(model_path).exists():
            # 从权重文件加载
            self.model = self._load_from_path(model_path)
        else:
            # 从ultralytics加载
            self.model = self._load_from_ultralytics(model_name)

        # 获取模型结构
        self.model = self.model.model

    def _load_from_path(self, model_path: str) -> nn.Module:
        """
        从权重文件加载模型

        参数:
            model_path: 模型权重路径

        返回:
            model: YOLO模型
        """
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            print(f"✓ 成功加载Teacher模型: {model_path}")
            return model
        except ImportError:
            raise ImportError(
                "ultralytics库未安装，请先安装：pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")

    def _load_from_ultralytics(self, model_name: str) -> nn.Module:
        """
        从ultralytics加载预训练模型

        参数:
            model_name: 模型名称

        返回:
            model: YOLO模型
        """
        try:
            from ultralytics import YOLO
            model = YOLO(model_name)
            print(f"✓ 成功加载预训练模型: {model_name}")
            return model
        except ImportError:
            raise ImportError(
                "ultralytics库未安装，请先安装：pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"加载预训练模型失败: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入图像 [B, 3, H, W]

        返回:
            output: 模型输出
        """
        return self.model(x)

    def get_model(self) -> nn.Module:
        """
        获取原始模型

        返回:
            model: YOLOv8模型
        """
        return self.model

    def to(self, device):
        """
        移动模型到指定设备

        参数:
            device: 目标设备
        """
        self.device = device
        self.model = self.model.to(device)
        return super().to(device)

    @staticmethod
    def load_from_checkpoint(checkpoint_path: str, device: str = 'cpu') -> 'TeacherModel':
        """
        从检查点加载Teacher模型

        参数:
            checkpoint_path: 检查点路径
            device: 设备

        返回:
            teacher: Teacher模型实例
        """
        teacher = TeacherModel(model_path=checkpoint_path, device=device)
        return teacher

    def freeze(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False

        # 冻结BatchNorm统计量
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.track_running_stats = False

    def unfreeze(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True

        # 解冻BatchNorm统计量
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()
                module.track_running_stats = True


def create_teacher_model(
    model_path: str,
    device: str = 'cpu',
    freeze: bool = True,
) -> TeacherModel:
    """
    创建Teacher模型的便捷函数

    参数:
        model_path: 模型权重路径
        device: 设备
        freeze: 是否冻结参数

    返回:
        teacher: Teacher模型
    """
    teacher = TeacherModel.load_from_checkpoint(model_path, device=device)

    if freeze:
        teacher.freeze()

    return teacher