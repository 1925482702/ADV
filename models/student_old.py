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
    """
    双模态 Student 模型

    核心功能：
    1. 接收 [B, 6, H, W] 格式的多模态输入
    2. 使用冻结的 Teacher 提取 RGB 和 IR 特征
    3. 逐尺度融合特征
    4. 接收外部传入的对抗噪声
    5. 完全兼容 YOLOv8 检测头
    """

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
        """
        参数:
            teacher_rgb: RGB Teacher 模型（已预训练）
            teacher_ir: IR Teacher 模型（已预训练）
            num_classes: 类别数量
            fusion_channels: 融合后特征通道数列表 [P3, P4, P5]，必须与 feature_channels 长度一致
            use_residual: 是否使用残差块
            feature_layers: 提取特征的层名称列表
            fusion_mode: 融合模式 ('concat' 或 'add')
        """
        super().__init__()

        # 默认特征层（YOLOv8n 的 P3/P4/P5 层）
        # 注意：不同大小的YOLOv8模型（n/s/m/l/x）的层名可能不同
        # YOLOv8n: model.15 (P3), model.18 (P4), model.21 (P5)
        # YOLOv8s: model.9 (P3), model.12 (P4), model.15 (P5)
        if feature_layers is None:
            feature_layers = ['model.15', 'model.18', 'model.21']

        self.num_classes = num_classes
        self.use_residual = use_residual
        self.feature_layers = feature_layers
        self.fusion_mode = fusion_mode

        # 🚨 保存原始检测头权重（在替换之前）
        self.teacher_rgb_detect_head_weights = None
        self.teacher_ir_detect_head_weights = None

        # 🚨 关键修正：保存对原始TeacherModel的引用，用于前向传播
        self.teacher_rgb_wrapper = teacher_rgb
        self.teacher_ir_wrapper = teacher_ir

        # 1. 设置 Teacher 模型（冻结参数）
        # 这里使用内部的DetectionModel，以便HookExtractor可以直接访问层
        self.teacher_rgb = self._setup_teacher(teacher_rgb, is_rgb=True)
        self.teacher_ir = self._setup_teacher(teacher_ir, is_rgb=False)

        # 2. Hook 特征提取器
        self.hook_rgb = HookExtractor(
            self.teacher_rgb,
            layer_names=feature_layers,
        )
        self.hook_ir = HookExtractor(
            self.teacher_ir,
            layer_names=feature_layers,
        )

        # 3. 获取特征通道数
        self.feature_channels = self._get_feature_channels()

        # 4. 设置 fusion_channels（如果未提供，则使用 feature_channels）
        if fusion_channels is None:
            # 🚨 关键修改：使用相加融合，融合后的通道数保持不变
            self.fusion_channels = self.feature_channels.copy()
        else:
            # 验证 fusion_channels 是否为列表
            if not isinstance(fusion_channels, list):
                raise TypeError(f"fusion_channels 必须是列表，但得到的是 {type(fusion_channels)}！")
            # 验证长度是否匹配
            if len(fusion_channels) != len(self.feature_channels):
                raise ValueError(
                    f"fusion_channels 长度与特征层数量不匹配！"
                    f"feature_channels: {self.feature_channels}, "
                    f"fusion_channels: {fusion_channels}"
                )
            self.fusion_channels = fusion_channels

        # 4. 特征融合模块
        self.fusion = MultiScaleFusion(
            in_channels=self.feature_channels,
            out_channels=self.fusion_channels,
            use_residual=use_residual,
            fusion_mode=self.fusion_mode,  # 使用指定的融合模式
        )

        # 5. 对抗噪声生成器（用于随机噪声生成，FGSM 在 Trainer 中完成）
        self.adv_gen = AdversarialNoiseGenerator(epsilon=0.01)

        # 6. 检测头（通过深拷贝 Teacher 的检测头）
        self.detect_head = self._create_detect_head(self.teacher_rgb)

        # 初始化特征缓存
        self.latest_rgb_features = None
        self.latest_ir_features = None
        self.latest_fused_features = None

    def _setup_teacher(self, teacher_model: nn.Module, is_rgb: bool = True) -> nn.Module:
        """
        设置 Teacher 模型（冻结所有参数并摘除检测头）

        参数:
            teacher_model: Teacher 模型
            is_rgb: 是否为 RGB Teacher（用于保存检测头权重）
        """
        teacher_model.eval()

        if hasattr(teacher_model, 'model'):
            detection_model = teacher_model.model
        else:
            detection_model = teacher_model

        # 1. 冻结所有参数
        for param in detection_model.parameters():
            param.requires_grad = False

        # 2. 冻结 BatchNorm 统计量
        for module in detection_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.track_running_stats = False

        # 🚨 3. 在替换检测头之前保存其权重
        from ultralytics.nn.modules.head import Detect
        if hasattr(detection_model, 'model') and isinstance(detection_model.model, nn.Sequential):
            # 检查最后一层是否是 Detect
            if isinstance(detection_model.model[-1], Detect):
                if is_rgb:
                    # 保存 RGB Teacher 的检测头权重
                    self.teacher_rgb_detect_head_weights = detection_model.model[-1].state_dict()
                    print(f"✓ 已保存 RGB Teacher 检测头权重")
                else:
                    # 保存 IR Teacher 的检测头权重
                    self.teacher_ir_detect_head_weights = detection_model.model[-1].state_dict()
                    print(f"✓ 已保存 IR Teacher 检测头权重")

        # 🚨 4. 核弹级显存优化：切断 Teacher 的检测头推理
        # Teacher 的 Detect 头在 eval 模式下会疯狂进行 DFL 解码榨干显存
        # 我们通过 Hook 已经拿到了 P3/P4/P5，直接让它在最后一层返回空！
        class DummyHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.f = -1  # YOLOv8 检测头需要的属性（输入来源）
                self.i = -1  # YOLOv8 检测头需要的属性（层索引）
                self.type = 'detect'  # YOLOv8 检测头需要的属性（层类型）

            def forward(self, x):
                return [] # 直接返回空列表，截断所有耗时运算

        # 将 Teacher 的最后一层（检测头）强行替换为 DummyHead
        if hasattr(detection_model, 'model') and isinstance(detection_model.model, nn.Sequential):
            detection_model.model[-1] = DummyHead()
            print("✓ 成功给 Teacher 摘除检测头，显存黑洞已封闭")

        return detection_model

    def _get_feature_channels(self) -> List[int]:
        """
        动态获取特征通道数：跑一次前向传播，抓取所有层信息
        """
        self.teacher_rgb.eval()
        
        # 1. 获取精度和设备（这一步你改得很棒！）
        dtype = next(self.teacher_rgb.parameters()).dtype
        device = next(self.teacher_rgb.parameters()).device
        
        # 2. 准备虚拟输入
        dummy_input = torch.randn(1, 3, 640, 640, device=device).to(dtype)
        
        # 3. 跑一次前向传播，让 Hook 自动抓取特征
        self.hook_rgb.clear()
        with torch.no_grad():
            _ = self.teacher_rgb(dummy_input)
            
        # 4. 从 Hook 结果中按需提取
        features_dict = self.hook_rgb.get_features()
        channels = []
        
        for layer_name in self.feature_layers:
            if layer_name in features_dict:
                feat = features_dict[layer_name]
                channels.append(feat.shape[1])  # 获取 C 维度
            else:
                raise ValueError(f"Hook 提取失败：在模型中找不到层 '{layer_name}'。请检查 layer_names 配置。")
        
        print(f"✓ 成功探测到 Teacher 特征通道数: {channels}")
        return channels
    
    def _create_detect_head(self, teacher_rgb: nn.Module) -> nn.Module:
        """
        创建检测头并从 RGB Teacher 复制权重

        参数:
            teacher_rgb: RGB Teacher 模型

        返回:
            detect_head: 检测头（原始的 Detect 类实例）
        """
        from ultralytics.nn.modules.head import Detect
        import copy

        # 🚨 关键修正：创建新的 Detect 头，使用融合后的通道数
        detect_head = Detect(nc=self.num_classes, ch=self.fusion_channels)

        # 验证检测头类型
        assert isinstance(detect_head, nn.Module), "检测头必须是一个 nn.Module"

        # 🚨 验证是否是真正的 Detect 类（包含必要的属性）
        required_attrs = ['nc', 'reg_max', 'stride']
        for attr in required_attrs:
            assert hasattr(detect_head, attr), f"检测头缺少必要属性: {attr}"

        # 🚨 关键修正：强制写入 YOLO 标准的 P3/P4/P5 步长
        # 避免使用全0的stride导致Loss计算时出现NaN或inf
        detect_head.stride = torch.tensor([8.0, 16.0, 32.0])

        # 🚨 关键修改：从保存的检测头权重加载
        # 由于我们使用相加融合，融合后的通道数与 RGB Teacher 的特征通道数相同
        # 所以可以直接复制 RGB Teacher 的检测头权重

        success = False

        if self.teacher_rgb_detect_head_weights is not None:
            print(f"✓ 找到 RGB Teacher 检测头权重，正在加载...")

            try:
                # 直接使用 load_state_dict 加载权重
                detect_head.load_state_dict(self.teacher_rgb_detect_head_weights, strict=False)
                print(f"✓ 检测头权重加载完成")
                success = True
            except Exception as e:
                print(f"⚠️  检测头权重加载失败: {e}")
                print(f"⚠️  尝试逐层加载...")

                # 如果整体加载失败，尝试逐层加载
                try:
                    for name, param in detect_head.named_parameters():
                        if name in self.teacher_rgb_detect_head_weights:
                            param.data.copy_(self.teacher_rgb_detect_head_weights[name])
                    print(f"✓ 检测头权重逐层加载完成")
                    success = True
                except Exception as e2:
                    print(f"⚠️  检测头权重逐层加载也失败: {e2}")

        if not success:
            print(f"⚠️  检测头将使用随机初始化")
            # 初始化偏置，避免 Loss 爆炸
            detect_head.bias_init()

        # 检测头会在第一次前向传播时自动初始化 anchors
        # 不需要显式初始化以节省内存
        print(f"✓ 检测头创建完成，stride: {detect_head.stride.tolist()}")

        return detect_head

    def forward(
        self,
        x: torch.Tensor,
        noises_rgb: List[torch.Tensor] = None,
        noises_ir: List[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        🚨 关键：前向传播（纯前向计算，不涉及对抗梯度计算）

        参数:
            x: 多模态输入张量 [B, 6, H, W]
               - 前 3 通道: RGB 图像
               - 后 3 通道: IR 图像（已对齐）
            noises_rgb: RGB 分支的对抗噪声列表 [noise_P3, noise_P4, noise_P5]
                        如果为 None，则不加噪声
            noises_ir: IR 分支的对抗噪声列表 [noise_P3, noise_P4, noise_P5]
                       如果为 None，则不加噪声

        返回:
            predictions: 检测预测结果（纯净格式，兼容 YOLOv8）
                        列表格式: [pred_P3, pred_P4, pred_P5]

        🚨 致命修正点：
        1. 绝不能在 forward 中计算对抗梯度（会导致计算图断裂或重复 backward）
        2. 对抗噪声必须从外部（Trainer）传入
        3. 特征信息存储在内部属性中，用于蒸馏
        4. 只返回纯净的 predictions，不返回特征字典
        """
        device = x.device
        dtype = x.dtype

        # 1. 分离 RGB 和 IR 通道
        x_rgb = x[:, :3, :, :]   # [B, 3, H, W]
        x_ir = x[:, 3:, :, :]    # [B, 3, H, W]

        # 2. 清空 Hook 缓存
        self.hook_rgb.clear()
        self.hook_ir.clear()

        # 3. Teacher 提取特征（无梯度）
        # 🚨 关键修正：使用TeacherModel的引用进行前向传播
        with torch.no_grad():
            _ = self.teacher_rgb_wrapper(x_rgb)
            _ = self.teacher_ir_wrapper(x_ir)

        # 4. 获取特征
        # 🚨 保存Teacher特征（用于蒸馏）
        teacher_rgb_features = [f.detach().clone() for f in self._extract_features(self.hook_rgb)]
        teacher_ir_features = [f.detach().clone() for f in self._extract_features(self.hook_ir)]

        # 清空Hook缓存
        self.hook_rgb.clear()
        self.hook_ir.clear()

        # 5. Student提取特征（用于后续处理）
        rgb_features = teacher_rgb_features  # Teacher的特征就是Student的初始特征
        ir_features = teacher_ir_features

        # 5. 应用对抗噪声（如果传入）
        if noises_rgb is not None:
            rgb_features = [
                f + n.to(device=f.device, dtype=f.dtype)
                for f, n in zip(rgb_features, noises_rgb)
            ]

        if noises_ir is not None:
            ir_features = [
                f + n.to(device=f.device, dtype=f.dtype)
                for f, n in zip(ir_features, noises_ir)
            ]

        # 6. 逐尺度融合
        fused_features = self.fusion(rgb_features, ir_features)

        # 7. 检测头预测
        # 🚨 关键修正：严禁解包参数！YOLOv8 的 Detect 头期望接收完整的 List[torch.Tensor]
        # 绝对不能写成 self.detect_head(*fused_features)
        # 必须直接将列表传入：self.detect_head(fused_features)

        # 🚨 断言防护：确保输入格式正确
        assert isinstance(fused_features, list), f"检测头输入必须是List，但得到 {type(fused_features)}"
        assert len(fused_features) == 3, f"检测头输入必须包含3个尺度特征，但得到 {len(fused_features)} 个"
        assert isinstance(fused_features[0], torch.Tensor), f"检测头输入元素必须是Tensor，但得到 {type(fused_features[0])}"

        # 🚨 关键修正：Detect 头在训练模式下会修改输入特征！
        # 所以在传递给 detect_head 之前，需要先备份 fused_features
        fused_features_backup = [f.clone() for f in fused_features]

        # 直接传递列表给检测头
        predictions = self.detect_head(fused_features)

        # 8. 🚨 关键：存储特征到内部属性（用于蒸馏）
        # 使用备份的 fused_features，而不是被 Detect 头修改后的版本
        self.latest_rgb_features = rgb_features
        self.latest_ir_features = ir_features
        self.latest_fused_features = fused_features_backup

        # 9. 🚨 关键：只返回纯净的 predictions（兼容 YOLOv8）
        # 不返回特征字典，避免破坏 YOLOv8 的 loss 计算流程
        return predictions

    def _extract_features(self, hook_extractor: HookExtractor) -> List[torch.Tensor]:
        """
        从 Hook 提取器获取特征

        参数:
            hook_extractor: Hook 特征提取器

        返回:
            features: 特征列表 [P3, P4, P5]
        """
        features_dict = hook_extractor.get_features()

        # 按照原始顺序提取特征
        features = []
        for layer_name in self.feature_layers:
            if layer_name in features_dict:
                features.append(features_dict[layer_name])
            else:
                raise ValueError(f"层 '{layer_name}' 的特征未找到")

        return features

    def get_latest_features(self) -> Dict[str, List[torch.Tensor]]:
        """
        获取最新的特征（用于蒸馏）

        返回:
            features_dict: 特征字典
                {
                    'teacher_rgb_features': [P3_rgb, P4_rgb, P5_rgb],
                    'teacher_ir_features': [P3_ir, P4_ir, P5_ir],
                    'fused_features': [P3_fuse, P4_fuse, P5_fuse],
                }
        """
        return {
            'teacher_rgb_features': self.latest_rgb_features,
            'teacher_ir_features': self.latest_ir_features,
            'fused_features': self.latest_fused_features,
        }

    def clear_features(self):
        """清空特征缓存"""
        self.latest_rgb_features = None
        self.latest_ir_features = None
        self.latest_fused_features = None

    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        获取可训练参数

        返回:
            params: 可训练参数列表
        """
        return [p for p in self.parameters() if p.requires_grad]

    def set_train_mode(self, mode: str):
        """
        设置训练模式

        参数:
            mode: 训练模式 ('normal' or 'adversarial')
        """
        assert mode in ['normal', 'adversarial'], f"未知模式: {mode}"
        self.training_mode = mode


def create_student_model(
    teacher_rgb: nn.Module,
    teacher_ir: nn.Module,
    num_classes: int = 80,
    fusion_channels: List[int] = None,
    use_residual: bool = True,
) -> DualModalStudent:
    """
    创建 Student 模型的便捷函数

    参数:
        teacher_rgb: RGB Teacher 模型
        teacher_ir: IR Teacher 模型
        num_classes: 类别数量
        fusion_channels: 融合后特征通道数列表 [P3, P4, P5]，如果为None则自动使用Teacher的特征通道数
        use_residual: 是否使用残差块

    返回:
        student: Student 模型
    """
    student = DualModalStudent(
        teacher_rgb=teacher_rgb,
        teacher_ir=teacher_ir,
        num_classes=num_classes,
        fusion_channels=fusion_channels,
        use_residual=use_residual,
    )
    return student