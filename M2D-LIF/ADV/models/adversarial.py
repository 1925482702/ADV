"""
对抗训练模块
实现特征层早期噪声注入，模拟单模态受损场景

核心设计理念：
1. 噪声注入在backbone早期层，让噪声随网络传播污染整个模态
2. L2归一化噪声，防止特征量级差异导致训练崩溃
3. 单侧特征污染（只污染RGB或IR），强迫另一模态补偿
4. 保护BatchNorm统计量，生成噪声时冻结BN更新
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


class AdversarialNoiseGenerator:
    """
    对抗噪声生成器
    
    功能：
    1. 在早期特征层生成对抗噪声
    2. L2归一化噪声，确保噪声强度受控
    3. 支持FGSM和PGD攻击
    4. 保护BatchNorm统计量
    """
    
    def __init__(
        self,
        attack_method: str = 'fgsm',
        epsilon: float = 0.03,
        pgd_steps: int = 3,
        pgd_alpha: float = 0.01,
        normalize_noise: bool = True,
    ):
        """
        参数:
            attack_method: 攻击方法 ('fgsm' 或 'pgd')
            epsilon: 噪声强度（归一化后的比例）
            pgd_steps: PGD迭代步数
            pgd_alpha: PGD步长
            normalize_noise: 是否对噪声进行L2归一化
        """
        self.attack_method = attack_method
        self.epsilon = epsilon
        self.pgd_steps = pgd_steps
        self.pgd_alpha = pgd_alpha
        self.normalize_noise = normalize_noise
    
    def generate_noise(
        self,
        feature: torch.Tensor,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        """
        根据梯度生成对抗噪声
        
        核心改进：
        - 只需要单层特征和梯度
        - L2归一化噪声，确保强度受控
        
        参数:
            feature: 早期特征层 [B, C, H, W]
            grad: 该特征层的梯度（从最终Loss反向传播得到）
        
        返回:
            noise: 归一化后的对抗噪声 [B, C, H, W]
        """
        if self.attack_method == 'fgsm':
            return self._fgsm_noise(feature, grad)
        elif self.attack_method == 'pgd':
            raise NotImplementedError("PGD需要完整的前向传播循环，请使用generate_pgd_noise方法")
        else:
            raise ValueError(f"未知的攻击方法: {self.attack_method}")
    
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
            # 🚨 推荐模式：L2归一化 + 特征尺度缩放
            # 1. 计算特征尺度因子，确保噪声相对于特征是有意义的
            feat_scale = feature.abs().mean().detach()
            
            # 2. L2归一化梯度（保留相对大小信息，不用sign抹杀）
            #    这样梯度大的位置噪声多，梯度小的位置噪声少，更平滑
            grad_norm = grad / (grad.norm() + 1e-8)
            
            # 3. 生成噪声：强度 = epsilon * 特征尺度 * 归一化梯度
            noise = self.epsilon * feat_scale * grad_norm
        else:
            # 原始FGSM：只取梯度方向，所有位置都加最大噪声
            # （不推荐，可能导致训练不稳定）
            noise = self.epsilon * grad.sign()
        
        return noise.detach()
    
    def generate_noise_with_loss(
        self,
        feature: torch.Tensor,
        model: nn.Module,
        loss_fn,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        """
        完整的噪声生成流程：从特征计算损失 -> 获取梯度 -> 生成噪声
        
        这个方法封装了梯度计算过程，适合简单场景。
        对于复杂的两次前向传播场景，建议使用 generate_noise() 方法。
        
        参数:
            feature: 早期特征层 [B, C, H, W]（必须requires_grad=True）
            model: 学生模型（用于冻结BN）
            loss_fn: 损失计算函数，接收特征并返回损失
            retain_graph: 是否保留计算图
        
        返回:
            noise: 对抗噪声
        """
        # 🚨 关键：保存训练状态
        was_training = model.training
        
        try:
            # 冻结BN统计量更新
            set_bn_eval(model)
            
            # 确保特征可求导
            if not feature.requires_grad:
                feature = feature.detach().requires_grad_(True)
            
            # 计算损失
            loss = loss_fn(feature)
            
            # 反向传播获取梯度
            loss.backward(retain_graph=retain_graph)
            
            # 提取梯度
            grad = feature.grad.clone()
            
            # 生成噪声
            noise = self._fgsm_noise(feature.detach(), grad)
            
            return noise
            
        finally:
            # 恢复BN训练模式
            if was_training:
                set_bn_train(model)
            
            # 清空梯度
            model.zero_grad()
    
    def generate_pgd_noise(
        self,
        feature: torch.Tensor,
        loss_fn,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        PGD噪声生成（多步迭代攻击）
        
        参数:
            feature: 原始早期特征 [B, C, H, W]
            loss_fn: 损失函数
            model: 学生模型（可选，用于冻结BN）
        
        返回:
            noise: PGD生成的对抗噪声
        """
        was_training = model.training if model is not None else True
        
        try:
            if model is not None:
                set_bn_eval(model)
            
            # 初始化对抗特征
            feat_adv = feature.clone().detach().requires_grad_(True)
            
            for step in range(self.pgd_steps):
                # 计算损失
                loss = loss_fn(feat_adv)
                loss.backward()
                
                # 提取梯度
                grad = feat_adv.grad.clone()
                
                # 更新对抗特征
                with torch.no_grad():
                    # 归一化梯度方向
                    if self.normalize_noise:
                        feat_scale = feature.abs().mean()
                        grad_norm = grad / (grad.norm() + 1e-8)
                        delta = self.pgd_alpha * feat_scale * grad_norm.sign()
                    else:
                        delta = self.pgd_alpha * grad.sign()
                    
                    feat_adv = feat_adv + delta
                    
                    # 投影到epsilon球内
                    total_delta = feat_adv - feature
                    if self.normalize_noise:
                        # 按特征尺度约束
                        feat_scale = feature.abs().mean()
                        max_delta = self.epsilon * feat_scale
                        total_delta = torch.clamp(total_delta, -max_delta, max_delta)
                    else:
                        total_delta = torch.clamp(total_delta, -self.epsilon, self.epsilon)
                    
                    feat_adv = feature + total_delta
                
                # 重新创建可求导的张量
                feat_adv = feat_adv.detach().requires_grad_(True)
            
            # 计算最终噪声
            noise = feat_adv.detach() - feature
            
            return noise
            
        finally:
            if model is not None and was_training:
                set_bn_train(model)
            if model is not None:
                model.zero_grad()


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


def random_modality_selection(batch_size: int, device: str = 'cuda') -> torch.Tensor:
    """
    随机选择要攻击的模态
    
    返回:
        mask: [B] 布尔张量，True表示攻击RGB，False表示攻击IR
    """
    return torch.rand(batch_size, device=device) > 0.5


# ==================== 便捷函数 ====================

def create_adversarial_generator(
    epsilon: float = 0.03,
    attack_method: str = 'fgsm',
) -> AdversarialNoiseGenerator:
    """
    创建对抗噪声生成器的便捷函数
    """
    return AdversarialNoiseGenerator(
        attack_method=attack_method,
        epsilon=epsilon,
        normalize_noise=True,
    )
