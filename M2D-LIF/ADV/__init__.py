"""
ADV - 对抗蒸馏融合模块
Adversarial Distillation for Multi-Modal Fusion

核心组件：
1. models: 学生模型、教师模型、融合模块、对抗噪声生成器、残差块
2. losses: 检测损失、蒸馏损失、总损失
3. trainer: 调度器（epsilon和lambda_distill）
4. utils: Hook提取器等工具
"""

# Models
from .models import (
    TeacherModel,
    create_teacher_model,
    DualModalStudent,
    create_student_model,
    MultiScaleFusion,
    AdversarialNoiseGenerator,
    set_bn_eval,
    set_bn_train,
    apply_noise_to_feature,
    random_modality_selection,
    create_adversarial_generator,
    ResidualFusionBlock,
)

# Losses
from .losses import DetectionLoss, DistillationLoss, TotalLoss

# Trainer
from .trainer import ADVScheduler, create_adv_scheduler

# Utils
from .utils import HookExtractor

# Validator
from .validator import DualModalValidator, create_dual_modal_validator

__all__ = [
    # Models - Teacher
    'TeacherModel',
    'create_teacher_model',
    # Models - Student
    'DualModalStudent',
    'create_student_model',
    # Models - Fusion
    'MultiScaleFusion',
    # Models - Adversarial
    'AdversarialNoiseGenerator',
    'set_bn_eval',
    'set_bn_train',
    'apply_noise_to_feature',
    'random_modality_selection',
    'create_adversarial_generator',
    # Models - Residual
    'ResidualFusionBlock',
    # Losses
    'DetectionLoss',
    'DistillationLoss',
    'TotalLoss',
    # Trainer
    'ADVScheduler',
    'create_adv_scheduler',
    # Utils
    'HookExtractor',
    # Validator
    'DualModalValidator',
    'create_dual_modal_validator',
]