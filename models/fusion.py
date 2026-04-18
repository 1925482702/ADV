"""
特征融合模块
实现逐尺度特征融合，在每个检测尺度上独立融合RGB和IR特征
"""

import torch
import torch.nn as nn
from typing import List

from .residual import ResidualFusionBlock

# 🚨 全局调试开关
DEBUG_MODE = True
DEBUG_FUSION_COUNT = 0  # 用于限制输出次数


def _debug_tensor(name: str, tensor: torch.Tensor, enabled: bool = True):
    """统一的张量调试输出"""
    if not enabled or not DEBUG_MODE:
        return
    global DEBUG_FUSION_COUNT
    if DEBUG_FUSION_COUNT > 5:  # 只输出前5次
        return
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    print(f"    {name}: shape={list(tensor.shape)}, "
          f"range=[{tensor.min().item():.4f}, {tensor.max().item():.4f}], "
          f"mean={tensor.mean().item():.4f}, "
          f"NaN={has_nan}, Inf={has_inf}")


class MultiScaleFusion(nn.Module):
    """
    多尺度特征融合模块

    功能：
    1. 在每个尺度上独立进行跨模态融合
    2. 输出通道数保持递增，完全兼容YOLOv8检测头
    3. 支持残差学习，学习Teacher未捕捉的信息
    4. 支持相加和拼接两种融合模式
    """

    def __init__(
        self,
        in_channels: List[int] = [256, 512, 1024],
        out_channels: List[int] = None,
        use_residual: bool = True,
        fusion_mode: str = 'concat',
    ):
        """
        参数:
            in_channels: 每个尺度的输入通道数 [P3, P4, P5]
            out_channels: 每个尺度的输出通道数 [P3, P4, P5]
                         如果为None，则使用in_channels
            use_residual: 是否使用残差学习
            fusion_mode: 融合模式，'concat'（拼接）或 'add'（相加）
        """
        super().__init__()

        # 🚨 关键检查：out_channels 必须是列表
        if out_channels is None:
            out_channels = in_channels
        else:
            assert isinstance(out_channels, list), "out_channels 必须是列表！"
            assert len(in_channels) == len(out_channels), "输入输出通道数长度不匹配！"

        assert fusion_mode in ['concat', 'add'], f"fusion_mode 必须是 'concat' 或 'add'，但得到 {fusion_mode}"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = len(in_channels)
        self.use_residual = use_residual
        self.fusion_mode = fusion_mode

        # 为每个尺度创建融合模块
        self.fuse_layers = nn.ModuleList([
            self._make_fuse_block(in_ch, out_ch, fusion_mode)
            for in_ch, out_ch in zip(in_channels, out_channels)
        ])

        # 残差学习块（每个尺度对应一个）
        if use_residual:
            self.residual_blocks = nn.ModuleList([
                ResidualFusionBlock(out_ch)
                for out_ch in out_channels
            ])
        else:
            self.residual_blocks = None

    def _make_fuse_block(self, in_ch: int, out_ch: int, fusion_mode: str) -> nn.Module:
        """
        创建单个尺度的融合块

        参数:
            in_ch: 输入通道数（单个模态）
            out_ch: 输出通道数
            fusion_mode: 融合模式

        返回:
            block: 融合块
        """
        if fusion_mode == 'concat':
            # 拼接模式：输入是 in_ch * 2（RGB + IR）
            return nn.Sequential(
                # 1x1 卷积：拼接后压缩通道
                nn.Conv2d(in_ch * 2, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                # 3x3 卷积：增强特征融合
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            # 🚨 简化的相加模式：直接恒等映射，不做任何额外处理
            # 让融合变成真正的"直接相加"，类似Baseline的方式
            # 这样可以公平对比"双路特征提取"vs"早期融合"
            return nn.Identity()

    def forward(
        self,
        rgb_features: List[torch.Tensor],
        ir_features: List[torch.Tensor],
        apply_residual: bool = True,
    ) -> List[torch.Tensor]:
        """
        前向传播

        参数:
            rgb_features: RGB 特征列表 [P3_rgb, P4_rgb, P5_rgb]
            ir_features: IR 特征列表 [P3_ir, P4_ir, P5_ir]
            apply_residual: 是否应用残差学习（默认True）

        返回:
            fused: 融合后的特征列表 [P3_fuse, P4_fuse, P5_fuse]

        🚨 关键：
        1. 强制断言检查特征尺寸是否一致
        2. 不做动态插值对齐
        3. 根据 fusion_mode 使用不同的融合方式
        4. apply_residual控制是否应用Residual（用于特征蒸馏）
        """
        global DEBUG_FUSION_COUNT
        
        # ======== DEBUG: 融合模块输入检查 ========
        if DEBUG_MODE and DEBUG_FUSION_COUNT < 5:
            print(f"\n{'='*20} 🔍 融合模块 (#{DEBUG_FUSION_COUNT}) {'='*20}")
            print(f"  fusion_mode: {self.fusion_mode}")
            print(f"  apply_residual: {apply_residual}")
            print(f"  输入特征:")
            for i, (rgb, ir) in enumerate(zip(rgb_features, ir_features)):
                print(f"    P{i+3}:")
                _debug_tensor("rgb", rgb)
                _debug_tensor("ir", ir)
        
        assert len(rgb_features) == len(ir_features) == self.num_scales, \
            f"特征数量不匹配！RGB: {len(rgb_features)}, IR: {len(ir_features)}, 期望: {self.num_scales}"

        fused = []
        for i in range(self.num_scales):
            # 🚨 关键：断言检查特征尺寸是否一致
            assert rgb_features[i].shape[-2:] == ir_features[i].shape[-2:], \
                f"尺度 {i} 的 RGB 和 IR 特征图尺寸不匹配！" \
                f"RGB: {rgb_features[i].shape[-2:]}, IR: {ir_features[i].shape[-2:]}"

            # 根据融合模式选择融合方式
            if self.fusion_mode == 'concat':
                # 拼接模式：拼接 RGB 和 IR 特征
                concat = torch.cat([rgb_features[i], ir_features[i]], dim=1)
                fused_feat = self.fuse_layers[i](concat)
            else:
                # 🚨 简化的相加模式：直接相加，不做额外处理
                # 类似Baseline的Add层，保持数值量级
                fused_feat = rgb_features[i] + ir_features[i]

            # 残差学习（可选，由apply_residual控制）
            if apply_residual and self.residual_blocks is not None:
                residual = self.residual_blocks[i](fused_feat)
                fused_feat = fused_feat + residual

            fused.append(fused_feat)
            
            # ======== DEBUG: 融合后特征检查 ========
            if DEBUG_MODE and DEBUG_FUSION_COUNT < 5:
                print(f"  P{i+3}融合后:")
                _debug_tensor("fused", fused_feat)

        if DEBUG_MODE and DEBUG_FUSION_COUNT < 5:
            print(f"{'='*60}")
            DEBUG_FUSION_COUNT += 1

        return fused

    def get_output_channels(self) -> List[int]:
        """
        获取输出通道数列表

        返回:
            out_channels: 输出通道数列表
        """
        return self.out_channels


class SimpleFusion(nn.Module):
    """
    简化版融合（无残差）
    用于 ablation study
    """

    def __init__(
        self,
        in_channels: List[int] = [256, 512, 1024],
        out_channels: List[int] = None,
    ):
        """
        参数:
            in_channels: 每个尺度的输入通道数 [P3, P4, P5]
            out_channels: 每个尺度的输出通道数 [P3, P4, P5]
        """
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        assert isinstance(out_channels, list), "out_channels 必须是列表！"
        assert len(in_channels) == len(out_channels), "输入输出通道数长度不匹配！"

        self.fuse_layers = nn.ModuleList([
            nn.Conv2d(in_ch * 2, out_ch, kernel_size=1, bias=False)
            for in_ch, out_ch in zip(in_channels, out_channels)
        ])

        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(out_ch)
            for out_ch in out_channels
        ])

    def forward(
        self,
        rgb_features: List[torch.Tensor],
        ir_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        前向传播

        参数:
            rgb_features: RGB 特征列表
            ir_features: IR 特征列表

        返回:
            fused: 融合后的特征列表
        """
        fused = []
        for i, (rgb_f, ir_f) in enumerate(zip(rgb_features, ir_features)):
            # 🚨 断言检查
            assert rgb_f.shape[-2:] == ir_f.shape[-2:], \
                f"尺度 {i} 的 RGB 和 IR 特征图尺寸不匹配！"

            concat = torch.cat([rgb_f, ir_f], dim=1)
            fused_feat = self.fuse_layers[i](concat)
            fused_feat = self.bn_layers[i](fused_feat)
            fused.append(fused_feat)

        return fused