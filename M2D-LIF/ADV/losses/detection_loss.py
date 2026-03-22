"""
检测损失（复用YOLOv8）
"""

import torch
import torch.nn as nn
from ultralytics.utils.loss import v8DetectionLoss


class DetectionLoss(nn.Module):
    """
    检测损失（复用YOLOv8）
    去除所有 try-except 备用逻辑，强制使用原生 Loss 暴露真实问题
    """

    def __init__(self, model):
        super().__init__()

        # 1. 提取 detect_head
        if hasattr(model, 'detect_head'):
            detect_head = model.detect_head
        elif hasattr(model, 'model') and hasattr(model.model, 'detect_head'):
            detect_head = model.model.detect_head
        else:
            raise ValueError("无法找到detect_head，请检查模型结构")

        # 2. 提取必要的属性
        self.stride = detect_head.stride
        self.reg_max = getattr(detect_head, 'reg_max', 16)
        self.nc = detect_head.nc

        # 3. 构造 LossModel 并强制初始化官方 Loss
        loss_model = self._create_loss_model(detect_head, model)
        self.loss_fn = v8DetectionLoss(loss_model)
        print("✓ 成功挂载 YOLOv8 官方检测损失")

    def _create_loss_model(self, detect_head, original_model):
        """构造一个假model对象，包含v8DetectionLoss需要的属性"""

        # 🚨 补充 YOLOv8 官方默认的损失放缩超参数
        class MockArgs:
            def __init__(self):
                self.box = 7.5   # 边框损失权重
                self.cls = 0.5   # 分类损失权重
                self.dfl = 4.0   # DFL分布损失权重 (官方默认4.0，不是1.5！)
                self.reg_max = getattr(detect_head, 'reg_max', 16)

        class LossModel:
            def __init__(self, detect_head, original_model):
                self.model = [detect_head]
                # 注入带有完整属性的 MockArgs
                self.args = getattr(original_model, 'args', MockArgs())
                try:
                    params = list(original_model.parameters())
                    self.device = params[0].device if params else 'cpu'
                except:
                    self.device = 'cpu'

            def parameters(self):
                for param in self.model[-1].parameters():
                    yield param

        return LossModel(detect_head, original_model)

    def forward(self, predictions, targets):
        """计算检测损失"""
        # 直接调用官方 Loss，如果有格式不对，让它直接报错，不要吞异常！
        out = self.loss_fn(predictions, targets)

        # YOLOv8 返回格式: (total_loss, loss_components_tensor)
        # loss_components_tensor 包含 [box_loss, cls_loss, dfl_loss]
        total_loss = out[0]
        loss_components = out[1]

        loss_dict = {
            'total': total_loss,
            'bbox': loss_components[0],
            'cls': loss_components[1],
            'dfl': loss_components[2],
        }
        return loss_dict

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)