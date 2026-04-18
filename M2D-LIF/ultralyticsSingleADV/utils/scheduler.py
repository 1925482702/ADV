"""
单模态对抗训练调度器
实现 epsilon 的动态调度

三阶段训练策略：
  Stage 0 - Warmup:     纯检测训练，不加噪，先建立基本检测能力
  Stage 1 - ADV Train:  对抗训练，epsilon 线性增长
  Stage 2 - Clean Finetune: 干净微调，去掉噪声，巩固检测能力
"""

import math
from typing import Dict


class SingleModalADVScheduler:
    """
    单模态对抗训练调度器

    三阶段策略：
      [0, warmup)            → warmup:     eps=0
      [warmup, stage2_start) → adv_train:  eps 线性增长
      [stage2_start, total)  → finetune:   eps=0
    """

    def __init__(
        self,
        total_epochs: int,
        epsilon_start: float = 0.0,
        epsilon_max: float = 0.05,
        warmup_epochs: int = 5,
        stage2_ratio: float = 0.2,
    ):
        """
        参数:
            total_epochs: 总训练轮数
            epsilon_start: 初始噪声强度
            epsilon_max: 最大噪声强度
            warmup_epochs: 纯检测 warmup 轮数
            stage2_ratio: Stage 2（干净微调）占总训练的比例
        """
        self.total_epochs = total_epochs
        self.epsilon_start = epsilon_start
        self.epsilon_max = epsilon_max
        self.warmup_epochs = warmup_epochs
        self.stage2_ratio = stage2_ratio

        # 计算阶段边界
        self.stage2_start = int(total_epochs * (1 - stage2_ratio))

        # 当前状态
        self.current_epoch = 0
        self.current_epsilon = epsilon_start
        self.current_stage = "warmup"

    def step(self, epoch: int) -> Dict[str, float]:
        """
        根据当前 epoch 更新调度参数

        参数:
            epoch: 当前 epoch

        返回:
            dict: {
                'epsilon': 当前噪声强度,
                'stage': 当前阶段名称
            }
        """
        self.current_epoch = epoch

        # Stage 0: Warmup — 纯检测训练
        if epoch < self.warmup_epochs:
            self.current_stage = "warmup"
            self.current_epsilon = 0.0

        # Stage 1: 对抗训练
        elif epoch < self.stage2_start:
            self.current_stage = "adv_train"
            # epsilon: 线性增长
            progress = (epoch - self.warmup_epochs) / max(self.stage2_start - self.warmup_epochs, 1)
            self.current_epsilon = self.epsilon_start + (self.epsilon_max - self.epsilon_start) * progress

        # Stage 2: 干净微调
        else:
            self.current_stage = "clean_finetune"
            self.current_epsilon = 0.0

        return {
            'epsilon': self.current_epsilon,
            'stage': self.current_stage,
        }

    def get_state(self) -> Dict:
        """获取当前调度状态"""
        return {
            'epoch': self.current_epoch,
            'epsilon': self.current_epsilon,
            'stage': self.current_stage,
        }

    def is_warmup(self) -> bool:
        """是否在 warmup 阶段"""
        return self.current_stage == "warmup"

    def is_adv_train(self) -> bool:
        """是否在对抗训练阶段"""
        return self.current_stage == "adv_train"

    def is_clean_finetune(self) -> bool:
        """是否在干净微调阶段"""
        return self.current_stage == "clean_finetune"

    def should_apply_noise(self) -> bool:
        """是否应该应用对抗噪声"""
        return self.current_epsilon > 0.001 and self.is_adv_train()

    def __repr__(self):
        return (f"SingleModalADVScheduler(epoch={self.current_epoch}/{self.total_epochs}, "
                f"eps={self.current_epsilon:.4f}, "
                f"stage={self.current_stage})")


# ==================== 便捷函数 ====================

def create_single_adv_scheduler(
    total_epochs: int,
    epsilon_max: float = 0.05,
) -> SingleModalADVScheduler:
    """
    创建单模态对抗训练调度器

    三阶段策略：
      [0,  5)   Warmup:     eps=0
      [5,  80)  ADV Train:  eps 从 0 开始增到 max
      [80, 100) Finetune:   eps=0
    """
    return SingleModalADVScheduler(
        total_epochs=total_epochs,
        epsilon_start=0.0,
        epsilon_max=epsilon_max,
        warmup_epochs=5,
        stage2_ratio=0.2,
    )
