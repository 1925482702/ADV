"""
多模态（RGB + IR）目标检测模型
架构：双 Backbone → 多层特征提取 → 对抗蒸馏融合 → 检测头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


# ─────────────────────────────────────────────
# 1. 单模态 Backbone（提取 P3/P4/P5 三层特征）
# ─────────────────────────────────────────────

class ModalityBackbone(nn.Module):
    """
    基于 ResNet50 的单模态特征提取器，输出 P3/P4/P5 三个尺度特征。
    输入: [B, 3, H, W]  (RGB 或 IR 的 3 通道)
    输出: dict {'p3': ..., 'p4': ..., 'p5': ...}
    """
    def __init__(self, pretrained=True):
        super().__init__()
        base = resnet50(pretrained=pretrained)

        # stem: conv1 + bn + relu + maxpool
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )
        self.layer1 = base.layer1   # stride 4,  C=256
        self.layer2 = base.layer2   # stride 8,  C=512  → P3
        self.layer3 = base.layer3   # stride 16, C=1024 → P4
        self.layer4 = base.layer4   # stride 32, C=2048 → P5

        # 统一通道数到 256，方便后续处理
        self.proj_p3 = nn.Conv2d(512,  256, 1)
        self.proj_p4 = nn.Conv2d(1024, 256, 1)
        self.proj_p5 = nn.Conv2d(2048, 256, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        p3 = self.proj_p3(self.layer2(x))
        p4 = self.proj_p4(self.layer3(self.layer2(x)))  # 重新走
        # 上面写法有重复计算，修正如下：
        return p3, p4, None  # 占位，下面用正确版本

    def forward(self, x):
        x  = self.stem(x)
        c1 = self.layer1(x)     # stride 4
        c2 = self.layer2(c1)    # stride 8  → P3
        c3 = self.layer3(c2)    # stride 16 → P4
        c4 = self.layer4(c3)    # stride 32 → P5

        p3 = self.proj_p3(c2)
        p4 = self.proj_p4(c3)
        p5 = self.proj_p5(c4)
        return {'p3': p3, 'p4': p4, 'p5': p5}


# ─────────────────────────────────────────────
# 2. 特征对齐模块（上采样到 P3 尺寸后拼接）
# ─────────────────────────────────────────────

class FeatureAlignFusion(nn.Module):
    """
    将 RGB 和 IR 的 P3/P4/P5 特征统一上采样到 P3 尺寸，
    按 [RGB_p3, RGB_p4, RGB_p5, IR_p3, IR_p4, IR_p5] 拼接。
    然后用一个卷积层降维，得到融合特征 F_fused。

    拼接顺序（对应文档描述）：
      [RGB层1, RGB层2, ..., RGB层k ; IR层1, IR层2, ..., IR层k]

    输入通道: 256 * 3 * 2 = 1536
    输出通道: fusion_channels（默认 512）
    """
    def __init__(self, per_modal_channels=256, num_scales=3, fusion_channels=512):
        super().__init__()
        in_ch = per_modal_channels * num_scales * 2  # 1536
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_ch, fusion_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feats: dict, ir_feats: dict):
        """
        rgb_feats / ir_feats: {'p3':..., 'p4':..., 'p5':...}
        返回融合特征 [B, fusion_channels, H/8, W/8]（P3 尺寸）
        """
        target_size = rgb_feats['p3'].shape[-2:]  # H/8, W/8

        def upsample(t):
            if t.shape[-2:] != target_size:
                return F.interpolate(t, size=target_size, mode='bilinear', align_corners=False)
            return t

        # [RGB_p3, RGB_p4, RGB_p5, IR_p3, IR_p4, IR_p5]
        parts = [
            upsample(rgb_feats['p3']),
            upsample(rgb_feats['p4']),
            upsample(rgb_feats['p5']),
            upsample(ir_feats['p3']),
            upsample(ir_feats['p4']),
            upsample(ir_feats['p5']),
        ]
        cat = torch.cat(parts, dim=1)   # [B, 1536, H/8, W/8]
        return self.fuse_conv(cat)       # [B, 512, H/8, W/8]


# ─────────────────────────────────────────────
# 3. 残差头（学习 Teacher 检测误差）
# ─────────────────────────────────────────────

class ResidualHead(nn.Module):
    """
    目的：学习 baseline teacher 检测时的"误差"，即查缺补漏。
    结构上只是一个轻量残差块 + 1x1 投影。

    设计选择（回答你的问题）：
      只在融合后的特征上加一个残差头，而不是每层都加。
      原因：如果在每层都加，对抗噪声注入后残差块会直接补偿掉噪声，
           相当于给噪声"打补丁"，模型就不再被迫压榨另一模态了，
           对抗训练的效果会大打折扣。所以残差块只放在融合之后，
           此时两模态特征已经合并，残差块学习的是融合层面的误差修正。
    """
    def __init__(self, channels=512):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


# ─────────────────────────────────────────────
# 4. 简易检测头（Anchor-Free 风格）
# ─────────────────────────────────────────────

class DetectionHead(nn.Module):
    """
    简单的单尺度检测头，输入融合特征，输出 cls + reg。
    实际使用时可替换为 YOLOv8 的 DFL Head 或其他。
    """
    def __init__(self, in_channels=512, num_classes=80):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 4, 1)   # [x, y, w, h] offset
        )

    def forward(self, x):
        return self.cls_head(x), self.reg_head(x)


# ─────────────────────────────────────────────
# 5. 单模态检测模型（阶段一预训练用）
# ─────────────────────────────────────────────

class SingleModalDetector(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone  = ModalityBackbone(pretrained=True)
        self.fusion    = FeatureAlignFusion(per_modal_channels=256,
                                            num_scales=3, fusion_channels=512)
        self.head      = DetectionHead(512, num_classes)

    def forward(self, x_rgb, x_ir=None):
        """单模态时 x_ir=None，用零特征占位"""
        rgb_f = self.backbone(x_rgb)
        if x_ir is not None:
            ir_f = self.backbone(x_ir)
        else:
            # 用零张量占位，保持接口一致
            ir_f = {k: torch.zeros_like(v) for k, v in rgb_f.items()}
        fused = self.fusion(rgb_f, ir_f)
        return self.head(fused)


# ─────────────────────────────────────────────
# 6. 双模态学生模型（阶段二 & 三）
# ─────────────────────────────────────────────

class BiModalStudentDetector(nn.Module):
    """
    双模态学生模型，包含：
      - RGB backbone
      - IR  backbone
      - 特征对齐融合模块
      - 残差头（学习 teacher 误差）
      - 检测头
    """
    def __init__(self, num_classes=80, fusion_channels=512):
        super().__init__()
        self.rgb_backbone  = ModalityBackbone(pretrained=True)
        self.ir_backbone   = ModalityBackbone(pretrained=True)
        self.align_fusion  = FeatureAlignFusion(256, 3, fusion_channels)
        self.residual_head = ResidualHead(fusion_channels)
        self.det_head      = DetectionHead(fusion_channels, num_classes)

    def forward(self, x_rgb, x_ir,
                adv_noise_rgb=None, adv_noise_ir=None):
        """
        x_rgb: [B, 3, H, W]
        x_ir:  [B, 3, H, W]
        adv_noise_rgb / adv_noise_ir: 对抗噪声字典 {'p3':..., 'p4':..., 'p5':...}
            只有其中一个不为 None（硬币决定污染哪个模态）
        """
        rgb_feats = self.rgb_backbone(x_rgb)
        ir_feats  = self.ir_backbone(x_ir)

        # 对抗噪声注入（加在 backbone 第一层输出，即 P3/P4/P5 之前已经映射的特征上）
        # 这里按文档讨论：噪声加在进入 backbone 后的最早特征（P3/P4/P5），
        # 而不是原始输入，这样噪声通过后续卷积累积效应更真实。
        if adv_noise_rgb is not None:
            for k in rgb_feats:
                rgb_feats[k] = rgb_feats[k] + adv_noise_rgb[k]
        if adv_noise_ir is not None:
            for k in ir_feats:
                ir_feats[k] = ir_feats[k] + adv_noise_ir[k]

        fused = self.align_fusion(rgb_feats, ir_feats)
        fused = self.residual_head(fused)          # 残差修正
        cls_pred, reg_pred = self.det_head(fused)

        # 同时返回中间特征，供蒸馏 loss 使用
        return cls_pred, reg_pred, fused, rgb_feats, ir_feats