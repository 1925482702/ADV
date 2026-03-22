"""
残差学习模块
用于学习Teacher未捕捉的跨模态关联信息
"""

import torch
import torch.nn as nn


class ResidualFusionBlock(nn.Module):
    """
    残差融合块

    功能：
    1. 学习Teacher未捕捉的信息
    2. 通过残差连接实现"青出于蓝而胜于蓝"的潜力
    """

    def __init__(self, channels: int):
        """
        参数:
            channels: 输入输出通道数
        """
        super().__init__()

        self.block = nn.Sequential(
            # 1x1 卷积降维
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.SiLU(inplace=True),
            # 3x3 卷积提取特征
            nn.Conv2d(channels // 4, channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.SiLU(inplace=True),
            # 1x1 卷积升维
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),  # 🚨 关键修复：输出约束在[-0.278, +∞)范围，与Teacher特征分布一致
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入特征 [B, C, H, W]

        返回:
            output: 残差输出 [B, C, H, W]
        """
        residual = self.block(x)
        return residual