"""
残差学习模块
用于学习Teacher未捕捉的跨模态关联信息

核心设计理念：
1. 零初始化：训练初期残差输出≈0，不破坏Teacher的融合特征
2. 渐进学习：随着训练进行，逐渐学习"误差修正"能力
3. 青出于蓝：最终突破Teacher上限
"""

import torch
import torch.nn as nn


class ResidualFusionBlock(nn.Module):
    """
    残差融合块

    功能：
    1. 学习Teacher未捕捉的信息
    2. 通过残差连接实现"青出于蓝而胜于蓝"的潜力
    
    🚨 核心改进：零初始化
    - 最后一个BN层的权重初始化为0
    - 训练初期残差输出≈0，不破坏Teacher的融合特征
    - 随着训练进行，逐渐学习有用的"误差修正"
    """

    def __init__(self, channels: int, zero_init: bool = True):
        """
        参数:
            channels: 输入输出通道数
            zero_init: 是否使用零初始化（默认True）
        """
        super().__init__()
        
        self.zero_init = zero_init

        # 🚨 分开定义各层，便于零初始化最后一个BN
        self.conv1 = nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels // 4)
        self.act1 = nn.SiLU(inplace=True)
        
        self.conv2 = nn.Conv2d(channels // 4, channels // 4, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels // 4)
        self.act2 = nn.SiLU(inplace=True)
        
        self.conv3 = nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)  # 🚨 关键：最后一个BN层，用于零初始化
        self.act3 = nn.SiLU(inplace=True)
        
        # 🚨 零初始化：让残差块初始输出为零
        if zero_init:
            self._zero_init_last_bn()
    
    def _zero_init_last_bn(self):
        """
        零初始化最后一个BN层
        
        原理：
        - BN输出 = gamma * (x - mean) / sqrt(var + eps) + beta
        - 当 gamma = 0 时，输出 = beta = 0
        - 这样残差块初始输出为0，不破坏Teacher特征
        
        效果：
        - 训练初期：fused = base_sum + 0 = base_sum（等于Teacher目标）
        - 训练中期：残差逐渐学习有用的修正信息
        - 训练后期：残差可能突破Teacher上限
        """
        nn.init.zeros_(self.bn3.weight)  # gamma = 0
        nn.init.zeros_(self.bn3.bias)    # beta = 0
        print(f"  ✓ ResidualFusionBlock 零初始化完成（最后一层BN权重和偏置设为0）")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入特征 [B, C, H, W]

        返回:
            output: 残差输出 [B, C, H, W]
        """
        # 逐层前向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)  # 🚨 零初始化的BN层
        x = self.act3(x)
        
        return x