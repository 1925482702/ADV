"""
训练调度器模块

包含：
1. AdversarialScheduler - 对抗强度调度器
2. LossWeightScheduler - 损失权重调度器
3. LRScheduler - 学习率调度器
4. CombinedScheduler - 组合调度器
"""

import torch
import math
from typing import Dict, Optional


class AdversarialScheduler:
    """
    对抗强度调度器
    根据epoch调整对抗噪声强度
    """
    def __init__(
        self,
        warmup_epochs: int = 20,
        max_epsilon: float = 0.03,
        total_epochs: int = 100,
        scheduler_type: str = 'progressive',
    ):
        """
        参数:
            warmup_epochs: warm-up轮数
            max_epsilon: 最大对抗强度
            total_epochs: 总训练轮数
            scheduler_type: 调度器类型
                - 'progressive': 渐进式（推荐）
                - 'step': 阶梯式
                - 'cosine': 余弦式
        """
        self.warmup_epochs = warmup_epochs
        self.max_epsilon = max_epsilon
        self.total_epochs = total_epochs
        self.scheduler_type = scheduler_type

    def get_epsilon(self, epoch: int) -> float:
        """
        获取当前epoch的对抗强度

        参数:
            epoch: 当前epoch

        返回:
            epsilon: 对抗噪声强度
        """
        if self.scheduler_type == 'progressive':
            return self._progressive_schedule(epoch)
        elif self.scheduler_type == 'step':
            return self._step_schedule(epoch)
        elif self.scheduler_type == 'cosine':
            return self._cosine_schedule(epoch)
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

    def _progressive_schedule(self, epoch: int) -> float:
        """
        渐进式调度
        
        🚨 修改：去掉对抗训练的warmup，从第0个epoch就开始
        """
        if epoch < 5:
            # 前5个epoch渐进增加：0.00 → max_epsilon
            progress = epoch / 5.0
            return progress * self.max_epsilon
        elif epoch >= self.total_epochs - 10:
            # 最后10轮逐渐减弱：max_epsilon → 0.00
            progress = (self.total_epochs - epoch) / 10.0
            return progress * self.max_epsilon
        else:
            # 稳定期：最大强度
            return self.max_epsilon

    def _step_schedule(self, epoch: int) -> float:
        """
        阶梯式调度
        """
        if epoch < self.warmup_epochs:
            return 0.0
        elif epoch < self.warmup_epochs + 20:
            return 0.01
        elif epoch < self.total_epochs - 20:
            return self.max_epsilon
        else:
            return 0.01

    def _cosine_schedule(self, epoch: int) -> float:
        """
        余弦式调度
        """
        if epoch < self.warmup_epochs:
            return 0.0

        # 余弦下降 (从 max_epsilon 降到 0)
        t = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        epsilon = 0.5 * self.max_epsilon * (1 + math.cos(math.pi * t))

        return epsilon

    def get_epsilon_schedule(self) -> list:
        """获取完整的调度曲线"""
        return [self.get_epsilon(e) for e in range(self.total_epochs + 1)]


class LossWeightScheduler:
    """
    损失权重调度器
    动态调整检测、蒸馏损失权重

    注意：对抗损失权重已废弃，对抗训练通过输入端的噪声注入实现，
    对抗强度由AdversarialScheduler控制，而不是损失权重。
    """
    def __init__(
        self,
        total_epochs: int = 100,
        schedule_type: str = 'stage',
    ):
        """
        参数:
            total_epochs: 总训练轮数
            schedule_type: 调度类型
                - 'stage': 分阶段
                - 'adaptive': 自适应
                - 'fixed': 固定
        """
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type

        # 默认权重
        self.default_weights = {
            'lambda_det': 1.0,
            'lambda_distill': 0.5,
        }

    def get_loss_weights(self, epoch: int) -> Dict[str, float]:
        """
        获取当前epoch的损失权重

        参数:
            epoch: 当前epoch

        返回:
            weights: 损失权重字典
        """
        if self.schedule_type == 'stage':
            return self._stage_schedule(epoch)
        elif self.schedule_type == 'adaptive':
            return self._adaptive_schedule(epoch)
        elif self.schedule_type == 'fixed':
            return self.default_weights
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def _stage_schedule(self, epoch: int) -> Dict[str, float]:
        """
        分阶段调度
        """
        if epoch < 20:
            # 阶段1：主要学习检测
            return {
                'lambda_det': 1.0,
                'lambda_distill': 0.5,
            }
        elif epoch < 50:
            # 阶段2：引入蒸馏
            return {
                'lambda_det': 1.0,
                'lambda_distill': 0.5,
            }
        elif epoch < 80:
            # 阶段3：对抗训练（对抗强度由AdversarialScheduler控制）
            return {
                'lambda_det': 1.0,
                'lambda_distill': 0.3,
            }
        else:
            # 阶段4：精细调优
            return {
                'lambda_det': 1.0,
                'lambda_distill': 0.3,
            }

    def _adaptive_schedule(self, epoch: int) -> Dict[str, float]:
        """
        自适应调度
        根据epoch动态调整权重
        """
        # 线性插值
        progress = epoch / self.total_epochs

        # 检测损失权重保持不变
        lambda_det = 1.0

        # 蒸馏损失权重逐渐降低
        lambda_distill = 0.5 * (1 - 0.4 * progress)

        return {
            'lambda_det': lambda_det,
            'lambda_distill': lambda_distill,
        }


class LRScheduler:
    """
    学习率调度器
    使用PyTorch原生调度器（LinearLR + SequentialLR + CosineAnnealingLR）

    实现warm-up + cosine annealing的组合策略，完全遵循PyTorch官方API，
    不进行任何手工修改optimizer.param_groups的操作。
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        initial_lr: float = 0.001,
        min_lr: float = 0.00001,
        scheduler_type: str = 'cosine',
        warmup_epochs: int = 5,
    ):
        """
        参数:
            optimizer: 优化器
            total_epochs: 总训练轮数
            initial_lr: 初始学习率
            min_lr: 最小学习率
            scheduler_type: 调度器类型
                - 'cosine': Cosine Annealing（推荐）
                - 'step': Step LR
                - 'exponential': Exponential LR
            warmup_epochs: warm-up轮数（使用PyTorch原生LinearLR）
        """
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs

        # 创建主调度器
        if scheduler_type == 'cosine':
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=min_lr,
            )
        elif scheduler_type == 'step':
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=(total_epochs - warmup_epochs) // 3,
                gamma=0.1,
            )
        elif scheduler_type == 'exponential':
            main_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95,
            )
        elif scheduler_type == 'constant':
            # 固定学习率：不做任何调整
            main_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=total_epochs,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        # 创建warm-up调度器
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )

            # 使用SequentialLR组合warm-up和主调度器
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            # 无warm-up，直接使用主调度器
            self.scheduler = main_scheduler

    def step(self):
        """更新学习率"""
        self.scheduler.step()

    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


class CombinedScheduler:
    """
    组合调度器
    组合多个调度器
    """
    def __init__(
        self,
        adversarial_scheduler: AdversarialScheduler,
        loss_weight_scheduler: LossWeightScheduler,
        lr_scheduler: LRScheduler,
    ):
        """
        参数:
            adversarial_scheduler: 对抗强度调度器
            loss_weight_scheduler: 损失权重调度器
            lr_scheduler: 学习率调度器
        """
        self.adv_scheduler = adversarial_scheduler
        self.loss_scheduler = loss_weight_scheduler
        self.lr_scheduler = lr_scheduler

    def step(self, epoch: int):
        """
        更新所有调度器

        参数:
            epoch: 当前epoch
        """
        # 更新学习率
        self.lr_scheduler.step()

        # 其他调度器不需要step，直接查询

    def get_state(self, epoch: int) -> Dict:
        """
        获取当前调度状态

        参数:
            epoch: 当前epoch

        返回:
            state: 调度状态字典
        """
        return {
            'epoch': epoch,
            'epsilon': self.adv_scheduler.get_epsilon(epoch),
            'loss_weights': self.loss_scheduler.get_loss_weights(epoch),
            'lr': self.lr_scheduler.get_lr(),
        }

    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.lr_scheduler.get_lr()
