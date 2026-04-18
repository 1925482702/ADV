"""
数据处理模块
支持双模态（RGB + IR）数据加载和增强
"""

from .dataset import DualModalityDataset
from .augment import (
    PairedLetterBox,
    PairedMosaic,
    PairedMixUp,
    PairedAlbumentations,
    PairedRandomFlip,
    PairedRandomPerspective,
    PairedCopyPaste,
    build_paired_transforms,
    Instances,
    Compose,
    Format
)

__all__ = [
    'DualModalityDataset',
    'PairedLetterBox',
    'PairedMosaic',
    'PairedMixUp',
    'PairedAlbumentations',
    'PairedRandomFlip',
    'PairedRandomPerspective',
    'PairedCopyPaste',
    'build_paired_transforms',
    'Instances',
    'Compose',
    'Format'
]