"""
损失函数模块
支持检测损失和蒸馏损失
"""

from .detection_loss import DetectionLoss
from .distill_loss import DistillationLoss, TotalLoss

__all__ = ['DetectionLoss', 'DistillationLoss', 'TotalLoss']