"""
Models 模块
包含 Student、Teacher、Fusion、Adversarial、Residual 等模型定义
"""

from .teacher import TeacherModel, create_teacher_model
from .student import DualModalStudent, create_student_model
from .fusion import MultiScaleFusion
from .adversarial import (
    AdversarialNoiseGenerator,
    set_bn_eval,
    set_bn_train,
    apply_noise_to_feature,
    random_modality_selection,
    create_adversarial_generator,
)
from .residual import ResidualFusionBlock

__all__ = [
    # Teacher
    'TeacherModel',
    'create_teacher_model',
    # Student
    'DualModalStudent',
    'create_student_model',
    # Fusion
    'MultiScaleFusion',
    # Adversarial
    'AdversarialNoiseGenerator',
    'set_bn_eval',
    'set_bn_train',
    'apply_noise_to_feature',
    'random_modality_selection',
    'create_adversarial_generator',
    # Residual
    'ResidualFusionBlock',
]
