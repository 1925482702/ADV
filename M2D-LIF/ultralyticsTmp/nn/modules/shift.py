# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Shift head for cross-modal alignment prediction."""

import torch
import torch.nn as nn
from .conv import Conv

__all__ = ['ShiftHead']


class ShiftHead(nn.Module):
    """Head for predicting relative shift between RGB and IR modalities."""

    def __init__(self, in_channels):
        """Initialize ShiftHead.

        Args:
            in_channels (int): Number of input channels from fusion feature
        """
        super().__init__()
        c2 = max(in_channels // 4, 32)  # intermediate channels

        # Predict relative shift: dx and dy
        self.conv = nn.Sequential(
            Conv(in_channels, c2, 3),
            Conv(c2, c2, 3),
            nn.Conv2d(c2, 2, 1)  # output: [dx, dy]
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input feature from fusion layer, shape (B, C, H, W)

        Returns:
            torch.Tensor: Predicted shift, shape (B, 2, H, W)
        """
        return self.conv(x)
