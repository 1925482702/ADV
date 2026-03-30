"""
单模态对抗训练模块
实现特征层早期噪声注入，增强模型鲁棒性

核心设计理念：
1. 噪声注入在backbone早期层，让噪声随网络传播
2. L2归一化噪声，防止特征量级差异导致训练崩溃
3. 保护BatchNorm统计量，生成噪声时冻结BN更新
"""

import torch
import torch.nn as nn
from typing import Optional


def set_bn_eval(model: nn.Module):
    """
    临时冻结所有BatchNorm层的running统计量更新
    
    在生成对抗噪声时使用，防止噪声污染BN统计量
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            module.eval()


def set_bn_train(model: nn.Module):
    """
    恢复所有BatchNorm层的训练模式
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            module.train()


class SingleModalAdversarialNoiseGenerator:
    """
    单模态对抗噪声生成器
    
    功能：
    1. 在早期特征层生成对抗噪声
    2. L2归一化噪声，确保噪声强度受控
    3. 支持FGSM攻击
    4. 保护BatchNorm统计量
    """
    
    def __init__(
        self,
        epsilon: float = 0.03,
        normalize_noise: bool = True,
    ):
        """
        参数:
            epsilon: 噪声强度（归一化后的比例）
            normalize_noise: 是否对噪声进行L2归一化
        """
        self.epsilon = epsilon
        self.normalize_noise = normalize_noise
    
    def generate_noise(
        self,
        feature: torch.Tensor,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        """
        根据梯度生成对抗噪声
        
        参数:
            feature: 早期特征层 [B, C, H, W]
            grad: 该特征层的梯度（从最终Loss反向传播得到）
        
        返回:
            noise: 归一化后的对抗噪声 [B, C, H, W]
        """
        return self._fgsm_noise(feature, grad)
    
    def _fgsm_noise(
        self,
        feature: torch.Tensor,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        """
        FGSM噪声生成（单步攻击）
        
        归一化模式（推荐）：
            noise = epsilon * feat_scale * grad_normalized
            保留梯度的相对大小信息，贡献大的像素点噪声多，贡献小的噪声少
        
        非归一化模式（原始FGSM）：
            noise = epsilon * sign(grad)
            一刀切，所有非零梯度位置都加最大噪声
        """
        if self.normalize_noise:
            # L2归一化 + 特征尺度缩放
            # 1. 计算特征尺度因子，确保噪声相对于特征是有意义的
            feat_scale = feature.abs().mean().detach()
            
            # 2. L2归一化梯度（保留相对大小信息，不用sign抹杀）
            grad_norm = grad / (grad.norm() + 1e-8)
            
            # 3. 生成噪声：强度 = epsilon * 特征尺度 * 归一化梯度
            noise = self.epsilon * feat_scale * grad_norm
        else:
            # 原始FGSM：只取梯度方向
            noise = self.epsilon * grad.sign()
        
        return noise.detach()


class EarlyFeatureExtractor:
    """
    早期特征提取器
    
    用于在模型的早期层截断前向传播，提取特征用于对抗噪声生成
    """
    
    def __init__(self, early_layer_idx: int = 2):
        """
        参数:
            early_layer_idx: 早期层索引，噪声注入位置（默认第2层）
        """
        self.early_layer_idx = early_layer_idx
        self.feature_cache = []
        self.hook_handles = []
    
    def register_hooks(self, model: nn.Module):
        """
        注册forward hook来捕获早期特征
        
        参数:
            model: 目标模型
        """
        self.clear()
        
        # 获取模型的层列表
        if hasattr(model, 'model'):
            layers = list(model.model.children())
        else:
            layers = list(model.children())
        
        # 注册hook到早期层
        if self.early_layer_idx < len(layers):
            handle = layers[self.early_layer_idx].register_forward_hook(self._forward_hook)
            self.hook_handles.append(handle)
    
    def _forward_hook(self, module, input, output):
        """Forward hook回调函数"""
        self.feature_cache.append(output)
    
    def get_early_feature(self):
        """获取捕获的早期特征"""
        if len(self.feature_cache) > 0:
            return self.feature_cache[0]
        return None
    
    def clear(self):
        """清除缓存和hooks"""
        self.feature_cache = []
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


def apply_noise_to_feature(
    feature: torch.Tensor,
    noise: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """
    将噪声应用到特征上
    
    参数:
        feature: 原始特征 [B, C, H, W]
        noise: 对抗噪声 [B, C, H, W]
        inplace: 是否原地修改
    
    返回:
        noisy_feature: 加噪后的特征
    """
    if inplace:
        feature.add_(noise.to(feature.dtype))
        return feature
    else:
        return feature + noise.to(feature.dtype)


# ==================== 便捷函数 ====================

def create_single_adv_generator(
    epsilon: float = 0.03,
) -> SingleModalAdversarialNoiseGenerator:
    """
    创建单模态对抗噪声生成器的便捷函数
    """
    return SingleModalAdversarialNoiseGenerator(
        epsilon=epsilon,
        normalize_noise=True,
    )
