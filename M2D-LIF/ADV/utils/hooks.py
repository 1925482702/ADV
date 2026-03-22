"""
Hook 特征提取器
用于无损提取模型中间层的特征
"""

import torch
from typing import Dict, List, Optional, Callable


class HookExtractor:
    """
    Hook 特征提取器

    功能：
    1. 在指定层注册 forward hook
    2. 自动提取前向传播时的中间层特征
    3. 不破坏原始模型结构
    """

    def __init__(self, model: torch.nn.Module, layer_names: List[str]):
        """
        参数:
            model: 要提取特征的模型
            layer_names: 要提取特征的层名称列表
        """
        self.model = model
        self.layer_names = layer_names
        self.features = {}
        self.handles = {}

        # 注册 hooks
        self._register_hooks()

    def _register_hooks(self):
        """注册 forward hooks"""
        def get_hook_fn(name: str) -> Callable:
            """创建 hook 函数"""
            def hook(module, input, output):
                self.features[name] = output
            return hook

        # 遍历模型的所有层
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                # 注册 hook
                handle = module.register_forward_hook(get_hook_fn(name))
                self.handles[name] = handle
                print(f"✓ Hook 已注册到层: {name}")

        # 检查是否所有层都找到了
        for layer_name in self.layer_names:
            if layer_name not in self.handles:
                print(f"⚠ 警告：未找到层 '{layer_name}'，请检查层名称是否正确")

    def get_features(self) -> Dict[str, torch.Tensor]:
        """
        获取提取的特征

        返回:
            features: 特征字典 {layer_name: feature_tensor}
        """
        return self.features

    def clear(self):
        """清空特征缓存"""
        self.features.clear()

    def remove_hooks(self):
        """移除所有 hooks"""
        for handle in self.handles.values():
            handle.remove()
        self.handles.clear()

    def __del__(self):
        """析构时自动移除 hooks"""
        self.remove_hooks()