"""
ADV训练调度器
实现 epsilon 和 lambda_distill 的动态调度

三阶段训练策略：
  Stage 0 - Warmup:     纯检测训练，不蒸馏不加噪，先建立基本检测能力
  Stage 1 - ADV Distill: 对抗蒸馏，epsilon 线性增长，lambda_distill 从 0 ramp-up 到目标值
  Stage 2 - Clean Finetune: 干净微调，去掉噪声，蒸馏权重线性衰减到 0
"""

import math
from typing import Dict


class ADVScheduler:
    """
    ADV 三阶段训练调度器

    关键改进（对比旧版）：
    1. Warmup 阶段 lambda_distill=0，让模型先学会基本检测
    2. 进入 Stage 1 后 lambda_distill 从 0 线性 ramp-up 到目标值，避免突变
    3. Stage 2 仍然是干净微调 + 蒸馏衰减
    """

    def __init__(
        self,
        total_epochs: int,
        epsilon_start: float = 0.0,
        epsilon_max: float = 0.05,
        lambda_distill_max: float = 0.5,
        lambda_distill_end: float = 0.0,
        warmup_epochs: int = 5,
        rampup_epochs: int = 10,
        stage2_ratio: float = 0.2,
    ):
        """
        参数:
            total_epochs: 总训练轮数
            epsilon_start: 初始噪声强度
            epsilon_max: 最大噪声强度
            lambda_distill_max: 蒸馏权重最大值（ramp-up 的目标）
            lambda_distill_end: 最终蒸馏权重（Stage 2 结束时）
            warmup_epochs: 纯检测 warmup 轮数
            rampup_epochs: 蒸馏权重 ramp-up 轮数（warmup 结束后）
            stage2_ratio: Stage 2 占总训练的比例
        """
        self.total_epochs = total_epochs
        self.epsilon_start = epsilon_start
        self.epsilon_max = epsilon_max
        self.lambda_distill_max = lambda_distill_max
        self.lambda_distill_end = lambda_distill_end
        self.warmup_epochs = warmup_epochs
        self.rampup_epochs = rampup_epochs
        self.stage2_ratio = stage2_ratio

        # 计算阶段边界
        self.rampup_end = warmup_epochs + rampup_epochs
        self.stage2_start = int(total_epochs * (1 - stage2_ratio))

        # 确保 rampup_end 不超过 stage2_start
        if self.rampup_end > self.stage2_start:
            self.rampup_end = self.stage2_start

        # 当前状态
        self.current_epoch = 0
        self.current_epsilon = epsilon_start
        self.current_lambda_distill = 0.0
        self.current_stage = "warmup"

    def step(self, epoch: int) -> Dict[str, float]:
        """
        根据当前 epoch 更新调度参数

        调度逻辑:
          [0, warmup)              → warmup:     eps=0, lambda=0
          [warmup, rampup_end)     → ramp-up:    eps 线性增, lambda 从 0 线性增到 max
          [rampup_end, stage2)     → full distill: eps 继续增, lambda=max
          [stage2, total)          → clean finetune: eps=0, lambda 从 max 线性降到 0
        """
        self.current_epoch = epoch

        # Stage 0: Warmup — 纯检测训练
        if epoch < self.warmup_epochs:
            self.current_stage = "warmup"
            self.current_epsilon = 0.0
            self.current_lambda_distill = 0.0

        # Stage 1: 对抗蒸馏（含 ramp-up 子阶段）
        elif epoch < self.stage2_start:
            self.current_stage = "adversarial_distill"

            # epsilon: 从 warmup 结束后线性增长到 max
            adv_progress = (epoch - self.warmup_epochs) / max(self.stage2_start - self.warmup_epochs, 1)
            self.current_epsilon = self.epsilon_start + (self.epsilon_max - self.epsilon_start) * adv_progress

            # lambda_distill: ramp-up 阶段从 0 线性增到 max，之后保持 max
            if epoch < self.rampup_end:
                rampup_progress = (epoch - self.warmup_epochs) / max(self.rampup_epochs, 1)
                self.current_lambda_distill = self.lambda_distill_max * rampup_progress
            else:
                self.current_lambda_distill = self.lambda_distill_max

        # Stage 2: 干净微调
        else:
            self.current_stage = "clean_finetune"
            self.current_epsilon = 0.0

            # lambda_distill: 从 max 线性衰减到 end
            decay_progress = (epoch - self.stage2_start) / max(self.total_epochs - self.stage2_start, 1)
            self.current_lambda_distill = self.lambda_distill_max + \
                (self.lambda_distill_end - self.lambda_distill_max) * decay_progress

        return {
            'epsilon': self.current_epsilon,
            'lambda_distill': self.current_lambda_distill,
            'stage': self.current_stage,
        }

    def get_state(self) -> Dict:
        """获取当前调度状态"""
        return {
            'epoch': self.current_epoch,
            'epsilon': self.current_epsilon,
            'lambda_distill': self.current_lambda_distill,
            'stage': self.current_stage,
        }

    def is_warmup(self) -> bool:
        return self.current_stage == "warmup"

    def is_adversarial(self) -> bool:
        return self.current_stage == "adversarial_distill"

    def is_clean_finetune(self) -> bool:
        return self.current_stage == "clean_finetune"

    def should_apply_noise(self) -> bool:
        return self.current_epsilon > 0.001 and not self.is_warmup()

    def __repr__(self):
        return (f"ADVScheduler(epoch={self.current_epoch}/{self.total_epochs}, "
                f"eps={self.current_epsilon:.4f}, "
                f"lambda={self.current_lambda_distill:.4f}, "
                f"stage={self.current_stage})")


# ==================== 便捷函数 ====================

def create_adv_scheduler(
    total_epochs: int,
    epsilon_max: float = 0.05,
    lambda_distill_start: float = 0.5,
) -> ADVScheduler:
    """
    创建 ADV 调度器

    三阶段策略：
      [0,  5)   Warmup:     纯检测，lambda=0, eps=0
      [5, 15)   Ramp-up:    lambda 从 0 → 0.5, eps 从 0 开始增
      [15, 80)  Full:       lambda=0.5, eps 继续增到 max
      [80, 100) Finetune:   lambda 0.5→0, eps=0
    """
    return ADVScheduler(
        total_epochs=total_epochs,
        epsilon_start=0.0,
        epsilon_max=epsilon_max,
        lambda_distill_max=lambda_distill_start,
        lambda_distill_end=0.0,
        warmup_epochs=5,
        rampup_epochs=10,
        stage2_ratio=0.2,
    )
