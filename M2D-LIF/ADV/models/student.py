"""
双模态 Student 模型
基于 YOLOv8 的多模态目标检测器

核心设计理念：
1. Teacher使用Baseline模型（已训练好的双模态融合模型）
2. 对抗噪声注入在backbone早期层，让噪声随网络传播
3. 支持两次前向传播：干净前向获取梯度 → 加噪前向真正训练
4. 残差块学习"误差修补"能力，打破Teacher上限
"""

import copy
import random
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple

from .fusion import MultiScaleFusion
from .adversarial import AdversarialNoiseGenerator, set_bn_eval, set_bn_train
from ..utils.hooks import HookExtractor

# 🚨 全局调试开关
DEBUG_MODE = False  # 设置为 True 启用详细调试输出
DEBUG_FORWARD_COUNT = 0


def _debug_tensor(name: str, tensor: torch.Tensor, enabled: bool = True):
    """统一的张量调试输出"""
    if not enabled or not DEBUG_MODE:
        return
    global DEBUG_FORWARD_COUNT
    if DEBUG_FORWARD_COUNT > 5:
        return
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    print(f"  {name}: shape={list(tensor.shape)}, "
          f"range=[{tensor.min().item():.4f}, {tensor.max().item():.4f}], "
          f"mean={tensor.mean().item():.4f}, "
          f"NaN={has_nan}, Inf={has_inf}")


class DualModalStudent(nn.Module):
    """
    双模态学生模型
    
    架构：
    - 双路Backbone（RGB和IR各一路，可共享权重）
    - 早期层噪声注入点
    - 多尺度融合层 + 残差块
    - Neck (PAFPN)
    - Detect Head
    
    Teacher：
    - 使用Baseline双模态模型，提供融合后的P3/P4/P5特征作为监督
    """
    
    def __init__(
        self,
        baseline_teacher: nn.Module,
        num_classes: int = 3,
        fusion_channels: List[int] = None,
        use_residual: bool = True,
        fusion_mode: str = 'add',
        early_layer_idx: int = 2,  # 早期层索引，用于噪声注入
    ):
        """
        参数:
            baseline_teacher: Baseline双模态Teacher模型（已训练好）
            num_classes: 类别数
            fusion_channels: 融合后的通道数 [P3, P4, P5]
            use_residual: 是否使用残差块
            fusion_mode: 融合模式 ('add' 或 'concat')
            early_layer_idx: 早期层索引，噪声注入位置（默认第2层）
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_residual = use_residual
        self.fusion_mode = fusion_mode
        self.early_layer_idx = early_layer_idx
        
        # ==================== Teacher 设置 ====================
        # 🚨 核心改动：使用Baseline Teacher（双模态融合模型）
        self.teacher = self._setup_baseline_teacher(baseline_teacher)
        
        # Teacher特征层（Baseline融合后的P3/P4/P5）
        # 根据yolov8_naive_add.yaml: 层13=P3融合, 层18=P4融合, 层23=P5融合
        self.teacher_fusion_layers = [13, 18, 23]
        self.hook_teacher = HookExtractor(
            self.teacher, 
            layer_names=[f'model.{i}' for i in self.teacher_fusion_layers]
        )
        
        # 探测特征通道数
        self.feature_channels = self._probe_feature_channels()
        
        if fusion_channels is None:
            self.fusion_channels = self.feature_channels.copy()
        else:
            self.fusion_channels = fusion_channels
        
        # ==================== Student Backbone ====================
        # 从Baseline Teacher提取backbone结构
        # 🚨 使用 self.teacher（已处理的 DetectionModel），而不是原始 baseline_teacher
        self.backbone_rgb, self.backbone_ir = self._create_dual_backbones(self.teacher)
        
        # ==================== 融合模块 + 残差块 ====================
        self.fusion = MultiScaleFusion(
            in_channels=self.feature_channels,
            out_channels=self.fusion_channels,
            use_residual=use_residual,
            fusion_mode=fusion_mode,
        )
        
        # ==================== 对抗噪声生成器 ====================
        self.adv_gen = AdversarialNoiseGenerator(epsilon=0.03, normalize_noise=True)
        
        # ==================== Neck ====================
        self.neck_layers = self._extract_neck(self.teacher)
        
        # ==================== Detect Head ====================
        self.detect_head = self._create_detect_head()
        
        # ==================== 特征缓存 ====================
        self.student_base_sum = None      # 基础和（无残差），用于蒸馏
        self.teacher_features = None      # Teacher融合特征
        self.fused_features = None        # Student融合特征
        
        # 早期特征缓存（用于两次前向传播）
        self.early_rgb = None
        self.early_ir = None
        
        # 🚨 核心修复：暴露 YOLO 引擎强依赖的顶级属性，让它完美伪装成官方模型
        self.stride = self.detect_head.stride
        self.pt = True  # 告诉引擎这是一个纯 PyTorch 模型
        self.names = {i: str(i) for i in range(num_classes)}  # 占位符，后续会被 dataset 的真实名称覆盖
        
        # 🚨 Bug 4 修复：使用 dummy forward 验证通道数对齐
        # 特别是在 fusion_mode='add' 时，确保融合后的通道数与 Neck/Detect Head 兼容
        self._validate_channel_alignment()
    
    def _validate_channel_alignment(self):
        """
        验证通道数对齐
        
        🚨 Bug 4 修复：fusion_mode='add' 时，通道数必须与 Neck 和 Detect Head 兼容
        
        执行 dummy forward 来验证：
        1. Backbone 输出通道数
        2. 融合后通道数
        3. SPPF 输入/输出通道数
        4. Detect Head 输入通道数
        """
        print("\n🔍 [Channel Validation] 验证通道数对齐...")
        
        device = next(self.teacher.parameters()).device
        dtype = next(self.teacher.parameters()).dtype
        
        # 创建 dummy 输入
        dummy_input = torch.randn(1, 6, 640, 640, device=device, dtype=dtype)
        
        try:
            # 1. 验证 Backbone 输出
            x_rgb = dummy_input[:, :3, :, :]
            x_ir = dummy_input[:, 3:, :, :]
            
            y_rgb = [None] * 10
            y_ir = [None] * 10
            
            rgb_features = []
            ir_features = []
            
            rgb_curr = x_rgb
            ir_curr = x_ir
            
            for i, (rgb_layer, ir_layer) in enumerate(zip(self.backbone_rgb, self.backbone_ir)):
                rgb_curr = rgb_layer(rgb_curr)
                ir_curr = ir_layer(ir_curr)
                y_rgb[i] = rgb_curr
                y_ir[i] = ir_curr
                
                if i in [4, 6, 8]:
                    rgb_features.append(rgb_curr)
                    ir_features.append(ir_curr)
            
            print(f"  ✓ Backbone 输出:")
            for i, (rf, irf) in enumerate(zip(rgb_features, ir_features)):
                print(f"    P{i+3}: RGB={rf.shape[1]}, IR={irf.shape[1]}")
            
            # 2. 验证融合输出
            fused_features = self.fusion(rgb_features, ir_features, apply_residual=False)
            print(f"  ✓ 融合输出 (fusion_mode='{self.fusion_mode}'):")
            for i, ff in enumerate(fused_features):
                print(f"    P{i+3}: channels={ff.shape[1]}")
            
            # 3. 验证 SPPF
            sppf_input = fused_features[2]  # P5 特征
            sppf_output = self.backbone_rgb[9](sppf_input)
            print(f"  ✓ SPPF:")
            print(f"    输入: {sppf_input.shape[1]} channels")
            print(f"    输出: {sppf_output.shape[1]} channels")
            
            # 4. 验证 Detect Head 输入通道
            print(f"  ✓ Detect Head 期望通道: {self.fusion_channels}")
            
            # 5. 尝试完整 forward（验证 Neck）
            # 构建缓存
            y_fused = [None] * 40
            y_fused[13] = fused_features[0]  # P3
            y_fused[18] = fused_features[1]  # P4
            y_fused[24] = sppf_output        # SPPF
            
            curr_feat = y_fused[24]
            
            for i, layer in enumerate(self.neck_layers):
                idx = i + 25
                
                if hasattr(layer, 'f') and layer.f != -1:
                    if isinstance(layer.f, int):
                        curr_feat = y_fused[layer.f]
                    else:
                        curr_feat = [curr_feat if j == -1 else y_fused[j] for j in layer.f]
                
                curr_feat = layer(curr_feat)
                y_fused[idx] = curr_feat
            
            # 收集 Neck 输出
            neck_outputs = [y_fused[30], y_fused[33], y_fused[36]]
            print(f"  ✓ Neck 输出:")
            for i, no in enumerate(neck_outputs):
                print(f"    P{i+3}: channels={no.shape[1]}")
            
            # 6. 验证 Detect Head
            predictions = self.detect_head(neck_outputs)
            print(f"  ✓ Detect Head 输出成功")
            
            print("✅ 通道对齐验证通过！\n")
            
        except Exception as e:
            print(f"\n❌ 通道对齐验证失败: {e}")
            print(f"  fusion_mode: {self.fusion_mode}")
            print(f"  feature_channels (Teacher): {self.feature_channels}")
            print(f"  fusion_channels: {self.fusion_channels}")
            raise RuntimeError(
                f"通道数不匹配！请检查 fusion_channels 配置。"
                f"fusion_mode='add' 时，fusion_channels 应该与 Teacher 融合层输出通道数一致。"
            ) from e
    
    def _setup_baseline_teacher(self, teacher_model: nn.Module) -> nn.Module:
        """
        设置Baseline Teacher
        
        Baseline Teacher是已训练好的双模态融合模型，
        其融合后的P3/P4/P5特征作为Student的监督信号
        """
        # 🚨 极其关键：必须保留 DetectionModel 这一层壳，因为它重写了 forward 处理跳跃连接
        # 不能直接提取内部的 Sequential！
        
        # 如果传入的是 Ultralytics 的 YOLO 外壳类，剥掉最外层拿到 DetectionModel
        if type(teacher_model).__name__ == 'YOLO':
            teacher = teacher_model.model
        else:
            # 如果已经是 DetectionModel，直接使用
            teacher = teacher_model
        
        # 冻结所有参数
        for param in teacher.parameters():
            param.requires_grad = False
        
        # 冻结BN统计量
        for module in teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.track_running_stats = False
        
        teacher.eval()
        print("✓ Baseline Teacher 已加载并冻结，保留了原生的跳跃连接前向逻辑")
        
        return teacher
    
    def _probe_feature_channels(self) -> List[int]:
        """探测Baseline Teacher融合层的通道数"""
        device = next(self.teacher.parameters()).device
        dtype = next(self.teacher.parameters()).dtype
        
        # 使用双模态dummy输入
        dummy_input = torch.randn(1, 6, 640, 640, device=device).to(dtype)
        
        self.hook_teacher.clear()
        with torch.no_grad():
            # 🚨 类型对齐：确保输入类型和 teacher 权重一致
            teacher_dtype = next(self.teacher.parameters()).dtype
            _ = self.teacher(dummy_input.to(teacher_dtype))
        
        features_dict = self.hook_teacher.get_features()
        channels = []
        
        for layer_idx in self.teacher_fusion_layers:
            layer_name = f'model.{layer_idx}'
            if layer_name in features_dict:
                feat = features_dict[layer_name]
                channels.append(feat.shape[1])
            else:
                raise ValueError(f"无法从Teacher提取层 {layer_name} 的特征")
        
        print(f"✓ Teacher融合层通道数: {channels}")
        return channels
    
    def _create_dual_backbones(self, baseline_teacher: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """
        创建双路Backbone
        
        从Baseline Teacher提取backbone结构，创建RGB和IR两个独立分支
        
        🚨 关键：根据 yolov8_naive_add.yaml 的层交替结构：
        - RGB 分支对应 yaml 层：3, 5, 7, 9, 11, 14, 16, 19, 21, 24
        - IR  分支对应 yaml 层：4, 6, 8, 10, 12, 15, 17, 20, 22, 24(SPPF共享)
        
        单路纯净 Backbone 结构（10层，标准 YOLOv8）：
        0: Conv(P1)  -> P1/2
        1: Conv(P2)  -> P2/4
        2: C2f       -> 特征处理
        3: Conv(P3)  -> P3/8
        4: C2f       -> P3特征输出（融合前）
        5: Conv(P4)  -> P4/16
        6: C2f       -> P4特征输出（融合前）
        7: Conv(P5)  -> P5/32
        8: C2f       -> P5特征处理
        9: SPPF      -> P5特征输出
        
        这样索引 [4, 6, 9] 对应 P3, P4, P5 特征输出，与标准 YOLOv8 一致
        """
        # 获取所有层
        if hasattr(baseline_teacher, 'model'):
            source_layers = list(baseline_teacher.model.children())
        else:
            source_layers = list(baseline_teacher.children())
        
        # 🚨 定义 RGB 和 IR 分支对应的 yaml 层索引
        # RGB: 层 3, 5, 7, 9, 11, 14, 16, 19, 21, 24
        # IR:  层 4, 6, 8, 10, 12, 15, 17, 20, 22, 24(SPPF共享)
        rgb_layer_indices = [3, 5, 7, 9, 11, 14, 16, 19, 21, 24]
        ir_layer_indices = [4, 6, 8, 10, 12, 15, 17, 20, 22, 24]
        
        def create_single_backbone(name: str, layer_indices: List[int]) -> nn.Module:
            """创建单个纯净backbone，只包含属于该模态的层"""
            layers = []
            
            for idx in layer_indices:
                if idx < len(source_layers):
                    layer = copy.deepcopy(source_layers[idx])
                    layers.append(layer)
                else:
                    print(f"⚠️ 警告: 层索引 {idx} 超出范围，跳过")
            
            # 重新初始化权重（除了SPPF层，因为它应该是共享的预训练权重）
            for i, layer in enumerate(layers):
                # SPPF层（最后一层）不重新初始化，保留预训练权重
                if i == len(layers) - 1:
                    continue
                    
                for module in layer.modules():
                    if isinstance(module, nn.Conv2d):
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.BatchNorm2d):
                        nn.init.ones_(module.weight)
                        nn.init.zeros_(module.bias)
                        module.reset_running_stats()
                        module.train()
            
            backbone = nn.Sequential(*layers)
            print(f"✓ {name} Backbone 创建完成（{len(layers)}层），权重已重新初始化")
            return backbone
        
        backbone_rgb = create_single_backbone("RGB", rgb_layer_indices)
        backbone_ir = create_single_backbone("IR", ir_layer_indices)
        
        return backbone_rgb, backbone_ir
    
    def _extract_neck(self, baseline_teacher: nn.Module) -> nn.ModuleList:
        """
        提取Neck层（PAFPN）
        
        🚨 根据 yolov8_naive_add.yaml：
        - Neck/Head 从第 25 层开始
        - P3出口: 第 30 层
        - P4出口: 第 33 层  
        - P5出口: 第 36 层
        - Detect头: 第 37 层
        """
        if hasattr(baseline_teacher, 'model'):
            source_layers = list(baseline_teacher.model.children())
        else:
            source_layers = list(baseline_teacher.children())
        
        # 🚨 Neck层（层25-36，不包含Detect头）
        neck_layers = [copy.deepcopy(layer) for layer in source_layers[25:37]]
        
        # 解冻并重置权重
        for layer in neck_layers:
            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    module.weight.requires_grad = True
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                    module.reset_running_stats()
                    module.train()
        
        print(f"✓ Neck 层提取完成，共 {len(neck_layers)} 层")
        return nn.ModuleList(neck_layers)
    
    def _create_detect_head(self) -> nn.Module:
        """创建检测头"""
        from ultralytics.nn.modules.head import Detect
        
        detect_head = Detect(nc=self.num_classes, ch=self.fusion_channels)
        detect_head.stride = torch.tensor([8.0, 16.0, 32.0])
        detect_head.bias_init()
        
        print(f"✓ 检测头创建完成，nc={self.num_classes}, stride={detect_head.stride.tolist()}")
        return detect_head
    
    # ==================== 核心方法：早期特征提取 ====================
    
    def extract_early_features(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List, List]:
        """
        提取早期特征（用于对抗噪声注入）
        
        这是两次前向传播的关键：
        1. 第一次前向：提取早期特征，获取梯度，生成噪声
        2. 第二次前向：在早期特征上加噪，继续传播
        
        🚨 Backbone 使用独立的缓存（索引 0-9）
        
        参数:
            x: 输入图像 [B, 6, H, W]，前3通道RGB，后3通道IR
        
        返回:
            rgb_early: RGB早期特征 [B, C, H', W']
            ir_early: IR早期特征 [B, C, H', W']
            y_rgb: RGB分支的特征缓存（用于跳跃连接）
            y_ir: IR分支的特征缓存（用于跳跃连接）
        """
        x_rgb = x[:, :3, :, :]
        x_ir = x[:, 3:, :, :]
        
        # 🚨 Backbone 缓存：只有 10 层
        y_rgb = [None] * 10
        y_ir = [None] * 10
        
        # 逐层前向传播，同时缓存每层输出
        rgb_curr = x_rgb
        ir_curr = x_ir
        
        for i in range(self.early_layer_idx):
            rgb_curr = self.backbone_rgb[i](rgb_curr)
            ir_curr = self.backbone_ir[i](ir_curr)
            y_rgb[i] = rgb_curr
            y_ir[i] = ir_curr
        
        return rgb_curr, ir_curr, y_rgb, y_ir
    
    def forward_from_early(
        self,
        rgb_early: torch.Tensor,
        ir_early: torch.Tensor,
        y_rgb: List = None,
        y_ir: List = None,
        original_x: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        从早期特征继续前向传播
        
        用于第二次前向传播：加噪后的早期特征 → 剩余backbone → 融合 → Neck → Head
        
        🚨 关键修复：接收并维护特征缓存 y，保持跳跃连接完整性
        
        参数:
            rgb_early: RGB早期特征（可能已加噪）
            ir_early: IR早期特征（可能已加噪）
            y_rgb: RGB分支的特征缓存（从前半段传递过来）
            y_ir: IR分支的特征缓存（从前半段传递过来）
            original_x: 原始输入图像（用于 Teacher 特征提取）
        
        返回:
            predictions: 检测预测结果
            features_dict: 特征字典（用于损失计算）
        """
        # 初始化缓存（如果没传）- Backbone 只有 10 层
        if y_rgb is None:
            y_rgb = [None] * 10
        if y_ir is None:
            y_ir = [None] * 10
        
        # 继续跑剩余的backbone层，同时更新缓存
        rgb_features = self._forward_backbone_remaining(rgb_early, is_rgb=True, y=y_rgb)
        ir_features = self._forward_backbone_remaining(ir_early, is_rgb=False, y=y_ir)
        
        # 计算基础和（无残差），用于蒸馏
        self.student_base_sum = [r + i for r, i in zip(rgb_features, ir_features)]
        
        # 融合（包含残差）
        self.fused_features = self.fusion(rgb_features, ir_features, apply_residual=True)
        
        # 🚨 关键：构建融合后的全局缓存 y_fused
        # 使用与yaml一致的索引，这样跳跃连接可以直接使用原始索引
        y_fused = [None] * 40
        
        # 将融合后的特征放到对应的全局索引位置
        y_fused[13] = self.fused_features[0]  # P3融合层
        y_fused[18] = self.fused_features[1]  # P4融合层
        # 🚨 P5 需要经过 SPPF 层处理（从 576 通道降到 384 通道）
        y_fused[24] = self.backbone_rgb[9](self.fused_features[2])  # SPPF 层
        
        # Neck + Head（传入完整缓存）
        predictions = self._forward_neck_and_head(self.fused_features, y_fused)
        
        # 🚨 提取 Teacher 特征（用于蒸馏损失）
        # 🚨 核心修复：如果有原始输入，传给 Teacher；否则从 y_rgb/y_ir 拼接
        if self.training:
            self.hook_teacher.clear()
            with torch.no_grad():
                if original_x is not None:
                    teacher_input = original_x
                else:
                    # 尝试用当前的 y_rgb 和 y_ir 拼接回 6 通道
                    teacher_input = torch.cat([y_rgb[0], y_ir[0]], dim=1)
                # 🚨 类型对齐：确保输入类型和 teacher 权重一致
                teacher_dtype = next(self.teacher.parameters()).dtype
                _ = self.teacher(teacher_input.to(teacher_dtype))
            self.teacher_features = [
                self.hook_teacher.get_features()[f'model.{i}'].detach().clone()
                for i in self.teacher_fusion_layers
            ]
            self.hook_teacher.clear()
        else:
            self.teacher_features = None
        
        # 返回预测和特征
        features_dict = {
            'student_base_sum': self.student_base_sum,
            'fused_features': self.fused_features,
            'teacher_features': self.teacher_features,
        }
        
        return predictions, features_dict
    
    def _forward_backbone_remaining(
        self,
        early_feat: torch.Tensor,
        is_rgb: bool = True,
        y: List = None,
    ) -> List[torch.Tensor]:
        """
        从早期特征继续跑剩余的backbone层
        
        🚨 关键修复：同时更新特征缓存 y，保持跳跃连接完整性
        
        参数:
            early_feat: 早期特征
            is_rgb: 是否为RGB分支
            y: 特征缓存（会被原地更新）
        
        返回:
            features: P3/P4/P5三个尺度的特征列表
        """
        backbone = self.backbone_rgb if is_rgb else self.backbone_ir
        
        # 初始化缓存（如果没传）- Backbone 只有 10 层
        if y is None:
            y = [None] * 10
        
        # 继续前向传播
        x = early_feat
        features = []
        
        # 🚨 逐层前向传播，同时缓存每层输出
        for i, layer in enumerate(backbone[self.early_layer_idx:], start=self.early_layer_idx):
            x = layer(x)
            y[i] = x  # 缓存当前层输出
            
            # 🚨 纯净Backbone：索引 4, 6, 8 对应 P3, P4, P5 输出 (P5在SPPF之前)
            if i in [4, 6, 8]:
                features.append(x)
        
        return features
    
    def _forward_neck_and_head(
        self,
        fused_features: List[torch.Tensor],
        y: List = None,
    ) -> torch.Tensor:
        """
        Neck和Head前向传播
        
        🚨 根据 yolov8_naive_add.yaml 的跳跃连接：
        - 第26层 Concat: 连接第18层(P4融合) → fused_features[1]
        - 第29层 Concat: 连接第13层(P3融合) → fused_features[0]
        - 第32层 Concat: 连接第27层(Neck中间层)
        - 第35层 Concat: 连接第24层(SPPF) → fused_features[2]
        
        🚨 核心修复：neck_layers 中的 layer.f 保留的是原始 yaml 索引
        我们需要使用相同的索引体系来填充 y 缓存
        
        参数:
            fused_features: 融合后的P3/P4/P5特征
            y: 全局特征缓存（使用原始yaml索引）
        
        返回:
            predictions: 检测预测结果
        """
        # 🚨 构建全局缓存，使用与yaml一致的索引
        if y is None:
            y = [None] * 40
            y[13] = fused_features[0]  # P3融合层
            y[18] = fused_features[1]  # P4融合层
            # y[24] 在外部已设置（SPPF输出）
        
        # 🚨 核心修复：确保 y[24] 已设置
        if y[24] is None:
            # 如果 y[24] 没有设置，使用 fused_features[2] 并经过 SPPF
            y[24] = self.backbone_rgb[9](fused_features[2])
        
        # 从 P5 SPPF 输出开始
        curr_feat = y[24]
        neck_outputs = []
        
        for i, layer in enumerate(self.neck_layers):
            # 🚨 核心修复：idx 使用原始 yaml 索引，与 layer.f 的索引体系一致
            idx = i + 25  # Neck 从第25层开始
            
            # 处理跳跃连接
            if hasattr(layer, 'f') and layer.f != -1:
                if isinstance(layer.f, int):
                    # 🚨 单个跳跃连接：检查 y[layer.f] 是否已填充
                    if y[layer.f] is None:
                        raise RuntimeError(
                            f"跳跃连接索引错误！层 {idx} 需要层 {layer.f} 的输出，"
                            f"但 y[{layer.f}] 尚未填充。可用索引: {[j for j, v in enumerate(y) if v is not None]}"
                        )
                    curr_feat = y[layer.f]
                else:
                    # 多个跳跃连接（如 Concat）
                    # 🚨 核心修复：检查所有需要的特征是否都已填充
                    for j in layer.f:
                        if j != -1 and y[j] is None:
                            raise RuntimeError(
                                f"跳跃连接索引错误！层 {idx} 需要层 {j} 的输出，"
                                f"但 y[{j}] 尚未填充。可用索引: {[k for k, v in enumerate(y) if v is not None]}"
                            )
                    # -1 表示上一层的输出（curr_feat），其他索引从 y 中获取
                    curr_feat = [curr_feat if j == -1 else y[j] for j in layer.f]
            
            # 执行当前层
            curr_feat = layer(curr_feat)
            
            # 🚨 核心修复：使用原始索引存储输出，供后续跳跃连接使用
            y[idx] = curr_feat
            
            # Neck 出口（P3/P4/P5）
            if idx in [30, 33, 36]:
                neck_outputs.append(curr_feat)
        
        # 检测头
        predictions = self.detect_head(neck_outputs)
        
        return predictions
    
    # ==================== 完整前向传播 ====================
    
    # 🚨 核心修复：添加 *args, **kwargs，无缝吸收 YOLO 官方验证器的所有额外参数
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        完整前向传播（用于推理或干净训练）
        
        🚨 关键修复：维护全局特征缓存 y，保持跳跃连接完整性
        
        参数:
            x: 输入图像 [B, 6, H, W]
            return_features: 是否返回特征字典
        
        返回:
            predictions: 检测预测结果
            features_dict: 特征字典（如果return_features=True）
        """
        global DEBUG_FORWARD_COUNT
        
        # 分离RGB和IR
        x_rgb = x[:, :3, :, :]
        x_ir = x[:, 3:, :, :]
        
        # ==================== Teacher特征提取 ====================
        # 🚨 核心修复：只在训练阶段（或明确要求返回特征时）才惊动 Teacher！
        # 在正常的验证/推理阶段，Student 必须独立行走，彻底绕开钩子深拷贝断裂的问题！
        if self.training or return_features:
            self.hook_teacher.clear()
            with torch.no_grad():
                # 🚨 类型对齐：确保输入类型和 teacher 权重一致
                teacher_dtype = next(self.teacher.parameters()).dtype
                _ = self.teacher(x.to(teacher_dtype))
            self.teacher_features = [
                self.hook_teacher.get_features()[f'model.{i}'].detach().clone()
                for i in self.teacher_fusion_layers
            ]
            self.hook_teacher.clear()
        else:
            self.teacher_features = None  # 验证推理时，Teacher 特征置空
        
        # ==================== Student Backbone ====================
        # 🚨 Backbone 使用独立的缓存，索引 0-9
        y_rgb = [None] * 10
        y_ir = [None] * 10
        
        rgb_features = self._forward_full_backbone(x_rgb, self.backbone_rgb, y_rgb)
        ir_features = self._forward_full_backbone(x_ir, self.backbone_ir, y_ir)
        
        # 基础和（无残差）
        self.student_base_sum = [r + i for r, i in zip(rgb_features, ir_features)]
        
        # 融合（包含残差）
        self.fused_features = self.fusion(rgb_features, ir_features, apply_residual=True)
        
        # 🚨 构建融合后的全局缓存（使用与yaml一致的索引）
        y_fused = [None] * 40
        y_fused[13] = self.fused_features[0]  # P3融合层
        y_fused[18] = self.fused_features[1]  # P4融合层
        # 🚨 P5 需要经过 SPPF 层处理（从 576 通道降到 384 通道）
        y_fused[24] = self.backbone_rgb[9](self.fused_features[2])  # SPPF 层
        
        # ==================== Neck + Head ====================
        predictions = self._forward_neck_and_head(self.fused_features, y_fused)
        
        DEBUG_FORWARD_COUNT += 1
        
        if return_features:
            features_dict = {
                'student_base_sum': self.student_base_sum,
                'teacher_features': self.teacher_features,
                'fused_features': self.fused_features,
            }
            return predictions, features_dict
        
        return predictions
    
    def _forward_full_backbone(
        self,
        x: torch.Tensor,
        backbone: nn.Module,
        y: List = None,
    ) -> List[torch.Tensor]:
        """
        完整backbone前向传播，返回P3/P4/P5特征
        
        🚨 关键修复：同时更新特征缓存 y
        
        参数:
            x: 输入张量
            backbone: Backbone模块
            y: 特征缓存（会被原地更新）
        
        返回:
            features: P3/P4/P5特征列表
        """
        if y is None:
            y = [None] * 10  # 🚨 Backbone 只有 10 层
        
        features = []
        for i, layer in enumerate(backbone):
            x = layer(x)
            y[i] = x  # 缓存当前层输出
            
            # 🚨 纯净Backbone：索引 4, 6, 8 对应 P3, P4, P5 输出 (P5在SPPF之前)
            if i in [4, 6, 8]:
                features.append(x)
        return features
    
    # ==================== 对抗训练接口 ====================
    
    def forward_clean_for_adv(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List, List]:
        """
        干净前向传播（用于生成对抗噪声）
        
        返回早期特征用于梯度计算
        
        🚨 关键修复：同时返回特征缓存 y，保持跳跃连接完整性
        
        参数:
            x: 输入图像 [B, 6, H, W]
        
        返回:
            predictions: 检测预测结果
            rgb_early: RGB早期特征
            ir_early: IR早期特征
            y_rgb: RGB特征缓存
            y_ir: IR特征缓存
        """
        # 提取早期特征和缓存
        rgb_early, ir_early, y_rgb, y_ir = self.extract_early_features(x)
        
        # 标记需要保留梯度
        rgb_early.requires_grad_(True)
        ir_early.requires_grad_(True)
        rgb_early.retain_grad()
        ir_early.retain_grad()
        
        # 🚨 核心修复：传入原始输入 x，让 Teacher 能拿到真实输入
        predictions, _ = self.forward_from_early(rgb_early, ir_early, y_rgb, y_ir, original_x=x)
        
        # 这里不计算loss，由外部调用者计算
        return predictions, rgb_early, ir_early, y_rgb, y_ir
    
    def forward_with_noise(
        self,
        x: torch.Tensor,
        noise_rgb: Optional[torch.Tensor] = None,
        noise_ir: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        带噪声的前向传播（第二次前向）
        
        🚨 关键修复：正确传递特征缓存 y
        
        参数:
            x: 输入图像 [B, 6, H, W]
            noise_rgb: RGB早期特征的对抗噪声
            noise_ir: IR早期特征的对抗噪声
        
        返回:
            predictions: 检测预测结果
            features_dict: 特征字典
        """
        # 提取早期特征和缓存
        rgb_early, ir_early, y_rgb, y_ir = self.extract_early_features(x)
        
        # 应用噪声
        if noise_rgb is not None:
            rgb_early = rgb_early + noise_rgb.to(rgb_early.dtype)
        if noise_ir is not None:
            ir_early = ir_early + noise_ir.to(ir_early.dtype)
        
        # 🚨 核心修复：传入原始输入 x，让 Teacher 能拿到真实输入
        return self.forward_from_early(rgb_early, ir_early, y_rgb, y_ir, original_x=x)
    
    # ==================== 特征获取 ====================
    
    def get_latest_features(self) -> Dict:
        """获取最新的特征字典"""
        return {
            'student_base_sum': self.student_base_sum,
            'teacher_features': self.teacher_features,
            'fused_features': self.fused_features,
        }
    
    def clear_features(self):
        """清空特征缓存"""
        self.student_base_sum = None
        self.teacher_features = None
        self.fused_features = None
        self.early_rgb = None
        self.early_ir = None
    
    # ==================== 工具方法 ====================
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """获取可训练参数"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def freeze_bn_for_adv(self):
        """冻结BN用于对抗噪声生成"""
        set_bn_eval(self)
    
    def unfreeze_bn(self):
        """解冻BN用于正常训练"""
        set_bn_train(self)
    
    def forward_inference(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        独立推理方法（不需要 Teacher）
        
        用于训练后的 Student 模型独立验证，不计算蒸馏损失。
        
        参数:
            x: 输入图像 [B, 6, H, W]
        
        返回:
            predictions: 检测预测结果
        """
        # 🚨 关键修复：确保 detect_head 处于 eval 模式
        # Detect.forward() 在 training 模式下返回原始特征图，不会解码 bbox 和 sigmoid 置信度
        was_training = self.detect_head.training
        self.detect_head.eval()
        
        # 分离RGB和IR
        x_rgb = x[:, :3, :, :]
        x_ir = x[:, 3:, :, :]
        
        # ==================== Student Backbone（不需要 Teacher）====================
        y_rgb = [None] * 10
        y_ir = [None] * 10
        
        rgb_features = self._forward_full_backbone(x_rgb, self.backbone_rgb, y_rgb)
        ir_features = self._forward_full_backbone(x_ir, self.backbone_ir, y_ir)
        
        # 融合（包含残差）
        fused_features = self.fusion(rgb_features, ir_features, apply_residual=True)
        
        # 构建融合后的全局缓存
        y_fused = [None] * 40
        y_fused[13] = fused_features[0]  # P3融合层
        y_fused[18] = fused_features[1]  # P4融合层
        y_fused[24] = self.backbone_rgb[9](fused_features[2])  # SPPF 层
        
        # Neck + Head
        predictions = self._forward_neck_and_head(fused_features, y_fused)
        
        # 恢复 detect_head 的训练状态
        if was_training:
            self.detect_head.train()
        
        return predictions
    
    def loss(self, batch, preds=None):
        """
        🚨 专门用来敷衍 YOLO 官方验证器的 val_loss 计算接口
        """
        if not hasattr(self, 'criterion'):
            from ultralytics.utils.loss import v8DetectionLoss
            
            # 官方的 v8DetectionLoss 有个癖好，喜欢从 model.model[-1] 获取检测头
            # 为了满足它，我们随手捏造一个 self.model 属性骗过它
            if not hasattr(self, 'model'):
                self.model = [None, self.detect_head]
            elif isinstance(self.model, list):
                self.model[-1] = self.detect_head
                
            self.criterion = v8DetectionLoss(self)
        
        # 如果验证器没传 preds 过来，我们自己跑一遍前向
        preds = self.forward(batch['img']) if preds is None else preds
        
        # 返回标准的 (loss, loss_items)
        return self.criterion(preds, batch)


# ==================== 工厂函数 ====================

def create_student_model(
    baseline_teacher: nn.Module,
    num_classes: int = 3,
    use_residual: bool = True,
    fusion_mode: str = 'add',
) -> DualModalStudent:
    """
    创建学生模型的便捷函数
    
    参数:
        baseline_teacher: Baseline双模态Teacher模型
        num_classes: 类别数
        use_residual: 是否使用残差块
        fusion_mode: 融合模式
    
    返回:
        student: 学生模型
    """
    return DualModalStudent(
        baseline_teacher=baseline_teacher,
        num_classes=num_classes,
        use_residual=use_residual,
        fusion_mode=fusion_mode,
    )
