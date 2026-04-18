"""
对抗训练模块
实现对抗噪声生成，在特征级别添加对抗扰动
"""

import torch
import torch.nn as nn
from typing import List, Callable


class AdversarialNoiseGenerator:
    """
    对抗噪声生成器（纯净工具类）

    功能：
    1. 在特征级别添加对抗扰动
    2. 支持 FGSM 和 PGD 攻击
    3. 保护 BatchNorm 统计量不被污染
    4. 维护 PGD 梯度链条完整性
    """

    def __init__(
        self,
        attack_method: str = 'fgsm',
        epsilon: float = 0.03,
        pgd_steps: int = 10,
        pgd_alpha: float = 0.01,
    ):
        """
        参数:
            attack_method: 攻击方法 ('fgsm' 或 'pgd')
            epsilon: 噪声强度
            pgd_steps: PGD 迭代步数
            pgd_alpha: PGD 步长
        """
        self.attack_method = attack_method
        self.epsilon = epsilon
        self.pgd_steps = pgd_steps
        self.pgd_alpha = pgd_alpha

    def generate_noise(
        self,
        features: List[torch.Tensor],
        loss_fn: Callable,
        model: nn.Module,
    ) -> List[torch.Tensor]:
        """
        生成对抗噪声

        参数:
            features: 特征列表 [F1, F2, ..., Fk]
            loss_fn: 损失函数（仅检测损失）
            model: 学生模型

        返回:
            noises: 噪声列表 [noise1, noise2, ..., noisek]
        """
        if self.attack_method == 'fgsm':
            return self._fgsm_attack(features, loss_fn, model)
        elif self.attack_method == 'pgd':
            return self._pgd_attack(features, loss_fn, model)
        else:
            raise ValueError(f"Unknown attack method: {self.attack_method}")

    def _fgsm_attack(
        self,
        features: List[torch.Tensor],
        loss_fn: Callable,
        model: nn.Module,
    ) -> List[torch.Tensor]:
        """
        FGSM（Fast Gradient Sign Method）攻击
        单步攻击，速度快
        """
        # 🚨 关键修正1：保存训练模式
        original_training_mode = model.training

        # 保存原始梯度状态
        original_requires_grad = {}
        for name, param in model.named_parameters():
            original_requires_grad[name] = param.requires_grad

        try:
            # 🚨 关键修正1：切换到 eval 模式，防止 BatchNorm 统计量污染
            model.eval()

            # 冻结模型参数
            for param in model.parameters():
                param.requires_grad = False

            # 创建可梯度的特征副本
            features_adv = [
                f.clone().detach().requires_grad_(True)
                for f in features
            ]

            # 🚨 使用 autocast 处理混合精度训练
            device = features[0].device if isinstance(features, list) else features.device
            with torch.cuda.amp.autocast(enabled=True):
                # 计算损失和梯度
                loss = loss_fn(features_adv)

            loss.backward()

            # 生成噪声
            noises = [
                self.epsilon * torch.sign(f.grad)
                for f in features_adv
            ]

            return noises

        finally:
            # 🚨 关键修正1：恢复训练模式
            model.train(original_training_mode)

            # 恢复参数梯度状态
            for name, param in model.named_parameters():
                param.requires_grad = original_requires_grad[name]

            # 清空梯度
            model.zero_grad()

    def _pgd_attack(
        self,
        features: List[torch.Tensor],
        loss_fn: Callable,
        model: nn.Module,
    ) -> List[torch.Tensor]:
        """
        PGD（Projected Gradient Descent）攻击
        多步迭代，更强的对抗性
        """
        # 🚨 关键修正1：保存训练模式
        original_training_mode = model.training

        # 保存原始梯度状态
        original_requires_grad = {}
        for name, param in model.named_parameters():
            original_requires_grad[name] = param.requires_grad

        try:
            # 🚨 关键修正1：切换到 eval 模式，防止 BatchNorm 统计量污染
            model.eval()

            # 冻结模型参数
            for param in model.parameters():
                param.requires_grad = False

            # 创建可梯度的特征副本
            features_adv = [
                f.clone().detach().requires_grad_(True)
                for f in features
            ]

            # PGD 迭代
            for step in range(self.pgd_steps):
                # 计算损失和梯度
                loss = loss_fn(features_adv)
                loss.backward()

                # 生成梯度
                gradients = [f.grad for f in features_adv]

                # 更新特征（添加扰动）
                for i in range(len(features_adv)):
                    features_adv[i] = features_adv[i] + self.pgd_alpha * torch.sign(gradients[i])

                    # 投影到 epsilon 范围内
                    delta = features_adv[i] - features[i]
                    delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                    features_adv[i] = features[i] + delta

                    # 🚨 关键修正2：重新创建可梯度的叶子节点，防止梯度链条断裂
                    features_adv[i] = features_adv[i].detach().requires_grad_(True)

                # 清空梯度
                for f in features_adv:
                    if f.grad is not None:
                        f.grad.zero_()

            # 计算最终噪声
            noises = [
                features_adv[i] - features[i]
                for i in range(len(features))
            ]

            return noises

        finally:
            # 🚨 关键修正1：恢复训练模式
            model.train(original_training_mode)

            # 恢复参数梯度状态
            for name, param in model.named_parameters():
                param.requires_grad = original_requires_grad[name]

            # 清空梯度
            model.zero_grad()