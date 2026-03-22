"""
双模态 Student 模型
基于 YOLOv8 的多模态目标检测器
"""

import copy
import torch
import torch.nn as nn
from typing import List, Optional, Dict
from .fusion import MultiScaleFusion
from .adversarial import AdversarialNoiseGenerator
from utils.hooks import HookExtractor


class DualModalStudent(nn.Module):
    def __init__(
        self,
        teacher_rgb: nn.Module,
        teacher_ir: nn.Module,
        num_classes: int = 80,
        fusion_channels: List[int] = None,
        use_residual: bool = True,
        feature_layers: List[str] = None,
        fusion_mode: str = 'add',
    ):
        super().__init__()

        # 🚨 分别设置Teacher和Student的特征层名称
        # Teacher是完整模型，层名格式为 'model.X'
        # Student的backbone是Sequential，层名格式为纯数字 'X'

        # 🚨 Teacher的特征层名（完整模型中的命名）
        self.teacher_feature_layers = ['model.4', 'model.6', 'model.9']

        # 🚨 Student的特征层名（经过nn.Sequential截断后的纯数字命名）
        # 层4 (C2f): P3特征 (stride=8)
        # 层6 (C2f): P4特征 (stride=16)
        # 层9 (SPPF): P5特征 (stride=32)
        if feature_layers is None:
            self.student_feature_layers = ['4', '6', '9']
        else:
            self.student_feature_layers = feature_layers

        self.num_classes = num_classes
        self.use_residual = use_residual
        self.fusion_mode = fusion_mode

        self.teacher_rgb_detect_head_weights = None
        self.teacher_ir_detect_head_weights = None

        self.teacher_rgb_wrapper = teacher_rgb
        self.teacher_ir_wrapper = teacher_ir

        self.teacher_rgb = self._setup_teacher(teacher_rgb, is_rgb=True)
        self.teacher_ir = self._setup_teacher(teacher_ir, is_rgb=False)

        # 🚨 Teacher使用teacher_feature_layers
        self.hook_rgb = HookExtractor(self.teacher_rgb, layer_names=self.teacher_feature_layers)
        self.hook_ir = HookExtractor(self.teacher_ir, layer_names=self.teacher_feature_layers)

        self.feature_channels = self._get_feature_channels()

        if fusion_channels is None:
            self.fusion_channels = self.feature_channels.copy()
        else:
            if not isinstance(fusion_channels, list):
                raise TypeError(f"fusion_channels 必须是列表，但得到的是 {type(fusion_channels)}！")
            if len(fusion_channels) != len(self.feature_channels):
                raise ValueError("fusion_channels 长度与特征层数量不匹配！")
            self.fusion_channels = fusion_channels

        self.fusion = MultiScaleFusion(
            in_channels=self.feature_channels,
            out_channels=self.fusion_channels,
            use_residual=use_residual,
            fusion_mode=self.fusion_mode,
        )

        self.adv_gen = AdversarialNoiseGenerator(epsilon=0.01)

        # 🚨 创建Student的Backbone（从Teacher复制，但可训练）
        # 传入student_feature_layers
        self.backbone_rgb = self._create_backbone(teacher_rgb, self.student_feature_layers)
        self.backbone_ir = self._create_backbone(teacher_ir, self.student_feature_layers)

        self.detect_head = self._create_detect_head(self.teacher_rgb)

        # 存放特征供 Loss 调用
        self.student_base_sum = None
        self.teacher_rgb_features = None
        self.teacher_ir_features = None
        self.fused_features = None

    def _setup_teacher(self, teacher_model: nn.Module, is_rgb: bool = True) -> nn.Module:
        teacher_model.eval()

        if hasattr(teacher_model, 'model'):
            detection_model = teacher_model.model
        else:
            detection_model = teacher_model

        for param in detection_model.parameters():
            param.requires_grad = False

        for module in detection_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.track_running_stats = False

        from ultralytics.nn.modules.head import Detect
        if hasattr(detection_model, 'model') and isinstance(detection_model.model, nn.Sequential):
            if isinstance(detection_model.model[-1], Detect):
                if is_rgb:
                    self.teacher_rgb_detect_head_weights = detection_model.model[-1].state_dict()
                    print(f"✓ 已保存 RGB Teacher 检测头权重")
                else:
                    self.teacher_ir_detect_head_weights = detection_model.model[-1].state_dict()
                    print(f"✓ 已保存 IR Teacher 检测头权重")

        class DummyHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.f = -1 
                self.i = -1 
                self.type = 'detect' 
            def forward(self, x):
                return [] 

        if hasattr(detection_model, 'model') and isinstance(detection_model.model, nn.Sequential):
            detection_model.model[-1] = DummyHead()
            print("✓ 成功给 Teacher 摘除检测头，显存黑洞已封闭")

        return detection_model

    def _get_feature_channels(self) -> List[int]:
        self.teacher_rgb.eval()
        dtype = next(self.teacher_rgb.parameters()).dtype
        device = next(self.teacher_rgb.parameters()).device

        dummy_input = torch.randn(1, 3, 640, 640, device=device).to(dtype)

        self.hook_rgb.clear()
        with torch.no_grad():
            _ = self.teacher_rgb(dummy_input)

        features_dict = self.hook_rgb.get_features()
        channels = []

        # 🚨 使用teacher_feature_layers（包含'model.'前缀）
        for layer_name in self.teacher_feature_layers:
            if layer_name in features_dict:
                feat = features_dict[layer_name]
                channels.append(feat.shape[1])
            else:
                raise ValueError(f"Hook 提取失败：找不到层 '{layer_name}'")

        print(f"✓ 成功探测到 Teacher 特征通道数: {channels}")
        return channels
    
    def _create_backbone(self, teacher_model: nn.Module, feature_layers: List[str]) -> nn.Module:
        """
        创建Student的Backbone（只提取Teacher的特征提取部分，不包含neck和head）

        YOLOv8架构：
        - Backbone: 层0-9（特征提取，输出P3/P4/P5）
        - Neck: 层10-21（FPN+PAN）
        - Head: 层22（Detect）

        只提取backbone部分，约12M参数
        """
        import copy

        # 获取teacher的model
        source_model = teacher_model.model if hasattr(teacher_model, 'model') else teacher_model

        # 获取model的所有层
        if hasattr(source_model, 'model'):
            model_layers = list(source_model.model.children())
        else:
            model_layers = list(source_model.children())

        # 🚨 只提取backbone部分（前10层）
        # 层0-9是特征提取部分，输出P3/P4/P5三个尺度的特征
        backbone_layers = model_layers[:10]

        # 🚨 打印backbone层的结构
        print(f"✓ Backbone结构分析（共{len(backbone_layers)}层）：")
        for i, layer in enumerate(backbone_layers):
            layer_type = type(layer).__name__
            # 尝试获取更多详细信息
            if hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
                print(f"  层{i}: {layer_type} (in={layer.in_channels}, out={layer.out_channels})")
            else:
                print(f"  层{i}: {layer_type}")

        # 🚨 关键：deepcopy选定的backbone层，确保Student拥有独立的权重副本
        backbone_layers = [copy.deepcopy(layer) for layer in backbone_layers]

        # 🚨 重新初始化所有权重，避免"作弊"（直接复制 Teacher 权重）
        def _reset_weights(model):
            """重新初始化模型的所有权重"""
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    # Kaiming 初始化（YOLOv8 默认）
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm2d):
                    # BatchNorm 重新初始化
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                    module.reset_running_stats()

        for layer in backbone_layers:
            _reset_weights(layer)

        print(f"✓ Student backbone 权重已重新初始化（只提取特征提取部分，不包含neck和head）")

        # 🚨 创建一个包装器，使用HookExtractor提取P3/P4/P5特征
        class BackboneWrapper(nn.Module):
            def __init__(self, layers, feature_layers, backbone_name=""):
                super().__init__()
                self.feature_layers = feature_layers  # 使用传入的feature_layers
                self.backbone_name = backbone_name

                # 将backbone层组合成Sequential
                self.model = nn.Sequential(*layers)

                # 注册HookExtractor（使用纯数字层名）
                self.hook = HookExtractor(self.model, layer_names=feature_layers)

                # 解冻参数
                for param in self.model.parameters():
                    param.requires_grad = True
                for module in self.model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        # 重置 BatchNorm 的统计量
                        module.reset_running_stats()
                        module.train()
                        module.track_running_stats = True

            def forward(self, x):
                # 清空Hook缓存
                self.hook.clear()

                # 前向传播
                _ = self.model(x)

                # 提取特征
                features_dict = self.hook.get_features()
                features = []
                for layer_name in self.feature_layers:
                    if layer_name in features_dict:
                        features.append(features_dict[layer_name])
                    else:
                        raise ValueError(f"层 '{layer_name}' 的特征未找到")

                return features

        return BackboneWrapper(backbone_layers, feature_layers, backbone_name="backbone")

    def _create_detect_head(self, teacher_rgb: nn.Module) -> nn.Module:
        from ultralytics.nn.modules.head import Detect
        detect_head = Detect(nc=self.num_classes, ch=self.fusion_channels)
        detect_head.stride = torch.tensor([8.0, 16.0, 32.0])

        success = False
        if self.teacher_rgb_detect_head_weights is not None:
            print(f"✓ 找到 RGB Teacher 检测头权重，正在加载...")
            try:
                detect_head.load_state_dict(self.teacher_rgb_detect_head_weights, strict=False)
                print(f"✓ 检测头权重加载完成")
                success = True
            except Exception as e:
                print(f"⚠️  检测头权重加载失败: {e}")

        if not success:
            print(f"⚠️  检测头将使用随机初始化")
            detect_head.bias_init()

        print(f"✓ 检测头创建完成，stride: {detect_head.stride.tolist()}")
        return detect_head

    def forward(
        self,
        x: torch.Tensor,
        noises_rgb: List[torch.Tensor] = None,
        noises_ir: List[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        老老实实地完整跑一次前向传播，把理论落地的稳定版。
        """
        x_rgb = x[:, :3, :, :] 
        x_ir = x[:, 3:, :, :]  
        
        # 1. Teacher 提取特征 (剥离并固化)
        self.hook_rgb.clear()
        self.hook_ir.clear()
        with torch.no_grad():
            # 加 autocast 兜底精度转换问题
            with torch.cuda.amp.autocast(enabled=True):
                _ = self.teacher_rgb_wrapper(x_rgb)
                _ = self.teacher_ir_wrapper(x_ir)
        
        # 拿到 Teacher 特征后立刻清空 Hook
        teacher_rgb_features = [f.detach().clone() for f in self._extract_features(self.hook_rgb, self.teacher_feature_layers)]
        teacher_ir_features = [f.detach().clone() for f in self._extract_features(self.hook_ir, self.teacher_feature_layers)]
        self.hook_rgb.clear()
        self.hook_ir.clear()
        
        # 2. Student 提取特征
        rgb_features = self.backbone_rgb(x_rgb)
        ir_features = self.backbone_ir(x_ir)
        
        # 3. 🚨【核心创新】计算基础和（无残差），专门存给蒸馏使用
        # 用没有加噪声的干净特征算，保证蒸馏的纯粹性
        self.student_base_sum = [r + i for r, i in zip(rgb_features, ir_features)] 
        
        # 保存 Teacher 特征供外部调用
        self.teacher_rgb_features = teacher_rgb_features
        self.teacher_ir_features = teacher_ir_features
        
        # 4. 应用对抗噪声 (如果有的话)
        if noises_rgb is not None:
            rgb_features = [f + n.to(f.dtype) for f, n in zip(rgb_features, noises_rgb)]
        if noises_ir is not None:
            ir_features = [f + n.to(f.dtype) for f, n in zip(ir_features, noises_ir)]
        
        # 5. 🚨【核心创新】融合（必须包含残差），用于检测任务冲上限
        fused_features = self.fusion(rgb_features, ir_features, apply_residual=True)
        self.fused_features = fused_features # 为了兼容 TotalLoss 存一份
        
        # 6. 送入检测头
        predictions = self.detect_head(fused_features)
        
        return predictions

    def _extract_features(self, hook_extractor: HookExtractor, feature_layers: List[str] = None) -> List[torch.Tensor]:
        """从 Hook 提取器获取特征"""
        if feature_layers is None:
            feature_layers = self.teacher_feature_layers  # 默认使用teacher的层名

        features_dict = hook_extractor.get_features()
        features = []
        for layer_name in feature_layers:
            if layer_name in features_dict:
                features.append(features_dict[layer_name])
            else:
                raise ValueError(f"层 '{layer_name}' 的特征未找到")
        return features

    def get_latest_features(self) -> Dict[str, List[torch.Tensor]]:
        """获取最新的特征字典，把 Base Sum 送出去算 Loss"""
        return {
            'student_base_sum': self.student_base_sum,
            'teacher_rgb_features': self.teacher_rgb_features,
            'teacher_ir_features': self.teacher_ir_features,
            'fused_features': self.fused_features,
        }

    def clear_features(self):
        self.student_base_sum = None
        self.teacher_rgb_features = None
        self.teacher_ir_features = None
        self.fused_features = None

    def get_trainable_params(self) -> List[nn.Parameter]:
        """获取可训练参数"""
        return [p for p in self.parameters() if p.requires_grad]


def create_student_model(
    teacher_rgb: nn.Module,
    teacher_ir: nn.Module,
    num_classes: int = 80,
    fusion_channels: List[int] = None,
    use_residual: bool = True,
) -> DualModalStudent:
    student = DualModalStudent(
        teacher_rgb=teacher_rgb,
        teacher_ir=teacher_ir,
        num_classes=num_classes,
        fusion_channels=fusion_channels,
        use_residual=use_residual,
    )
    return student