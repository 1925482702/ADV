"""
Models 模块
包含 Student、Teacher、Fusion、Adversarial 等模型定义
"""

from .teacher import TeacherModel, create_teacher_model

__all__ = [
    'TeacherModel',
    'create_teacher_model',
]