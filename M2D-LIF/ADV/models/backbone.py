"""
简单的双流Backbone模块
用于Student模型
"""

import torch
import torch.nn as nn
from typing import List

# 从M2D-LIF导入必要的模块
import sys
sys.path.insert(0, '/root/autodl-tmp/ADV/M2D-LIF')
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f


class SimpleBackbone(nn.Module):
    """
    简单的Backbone，用于RGB或IR特征提取
    输出P3/P4/P5三个尺度的特征
    """
    def __init__(self, in_channels=3, out_channels=[192, 384, 576]):
        """
        参数:
            in_channels: 输入通道数（3 for RGB/IR）
            out_channels: 输出特征通道数 [P3, P4, P5]
        """
        super().__init__()

        # P1/2 (下采样 1 次)
        self.conv1 = Conv(in_channels, 64, 3, 2)  # 640 -> 320
        
        # P2/4 (下采样 1 次)
        self.conv2 = Conv(64, 128, 3, 2)          # 320 -> 160
        self.c2f1 = C2f(128, 128, 2, True)
        
        # P3/8 (下采样 1 次，第一个输出节点)
        self.conv3 = Conv(128, out_channels[0], 3, 2)  # 160 -> 80, 通道: 128 -> 192
        self.c2f2 = C2f(out_channels[0], out_channels[0], 3, True)
        
        # P4/16 (下采样 1 次，第二个输出节点)
        self.conv4 = Conv(out_channels[0], out_channels[1], 3, 2)  # 80 -> 40, 通道: 192 -> 384
        self.c2f3 = C2f(out_channels[1], out_channels[1], 3, True)
        
        # P5/32 (下采样 1 次，第三个输出节点)
        self.conv5 = Conv(out_channels[1], out_channels[2], 3, 2)  # 40 -> 20, 通道: 384 -> 576
        self.c2f4 = C2f(out_channels[2], out_channels[2], 3, True)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入 [B, 3, 640, 640]

        返回:
            features: [P3, P4, P5] 特征列表
        """
        # P1/2
        x = self.conv1(x)  # [B, 64, 320, 320]
        
        # P2/4
        x = self.conv2(x)  # [B, 128, 160, 160]
        x = self.c2f1(x)   # [B, 128, 160, 160]
        
        # P3/8 (输出)
        x = self.conv3(x)  # [B, 192, 80, 80]
        p3 = self.c2f2(x)  # [B, 192, 80, 80] ← 直接输出
        
        # P4/16 (输出)
        x = self.conv4(p3) # [B, 384, 40, 40]
        p4 = self.c2f3(x)  # [B, 384, 40, 40] ← 直接输出
        
        # P5/32 (输出)
        x = self.conv5(p4) # [B, 576, 20, 20]
        p5 = self.c2f4(x)  # [B, 576, 20, 20] ← 直接输出
        
        return [p3, p4, p5]