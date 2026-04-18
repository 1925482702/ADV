"""
数据加载器
创建训练和验证数据加载器
"""

import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, Optional
import yaml


def create_train_loader(
    config: Dict,
    distributed: bool = False,
    rank: int = 0,
) -> DataLoader:
    """
    创建训练数据加载器

    参数:
        config: 配置字典，包含：
            - train_yaml: 训练数据集YAML配置文件路径
            - img_size: 图像大小
            - batch_size: 批次大小
            - num_workers: 工作进程数
            - hyp: 数据增强超参数
        distributed: 是否使用分布式训练
        rank: 当前进程的rank

    返回:
        train_loader: 训练数据加载器
    """
    # 加载数据集配置
    data_config = load_data_config(config.get('train_yaml', 'data/FLIR.yaml'))

    # 创建数据集
    from .dataset import DualModalityDataset

    train_dataset = DualModalityDataset(
        img_path=data_config['train'],
        imgsz=config.get('img_size', 640),
        cache=config.get('cache', False),
        augment=config.get('augment', False),  # 🚨 默认关闭数据增强，与Baseline一致
        hyp=config.get('hyp', {}),
        prefix='train: ',
        rect=config.get('rect', False),
        batch_size=config.get('batch_size', 16),
        stride=32,
        single_cls=config.get('single_cls', False),
        classes=config.get('classes', None),
        fraction=config.get('fraction', 1.0),
        data=data_config,
    )

    # 创建采样器
    if distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.get('world_size', 1),
            rank=rank,
            shuffle=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.get('num_workers', 8),
        pin_memory=True,
        collate_fn=DualModalityDataset.collate_fn,
        drop_last=True,
    )

    print(f"✓ 训练数据加载器创建成功: {len(train_loader)} 个批次")
    return train_loader


def create_val_loader(
    config: Dict,
    distributed: bool = False,
    rank: int = 0,
) -> DataLoader:
    """
    创建验证数据加载器

    参数:
        config: 配置字典，包含：
            - val_yaml: 验证数据集YAML配置文件路径
            - img_size: 图像大小
            - batch_size: 批次大小
            - num_workers: 工作进程数
        distributed: 是否使用分布式训练
        rank: 当前进程的rank

    返回:
        val_loader: 验证数据加载器
    """
    # 加载数据集配置
    data_config = load_data_config(config.get('val_yaml', 'data/FLIR.yaml'))

    # 创建数据集
    from .dataset import DualModalityDataset

    val_dataset = DualModalityDataset(
        img_path=data_config.get('val', data_config.get('train')),
        imgsz=config.get('img_size', 640),
        cache=config.get('cache', False),
        augment=False,
        hyp={},
        prefix='val: ',
        rect=False,
        batch_size=config.get('batch_size', 16),
        stride=32,
        single_cls=config.get('single_cls', False),
        classes=config.get('classes', None),
        fraction=config.get('fraction', 1.0),
        data=data_config,
    )

    # 创建采样器
    if distributed:
        sampler = DistributedSampler(
            val_dataset,
            num_replicas=config.get('world_size', 1),
            rank=rank,
            shuffle=False,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = False

    # 创建数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.get('num_workers', 8),
        pin_memory=True,
        collate_fn=DualModalityDataset.collate_fn,
        drop_last=False,
    )

    print(f"✓ 验证数据加载器创建成功: {len(val_loader)} 个批次")
    return val_loader


def load_data_config(yaml_path: str) -> Dict:
    """
    加载数据集配置文件

    参数:
        yaml_path: YAML配置文件路径

    返回:
        data_config: 数据集配置字典
    """
    try:
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        # 🚨 关键修正：处理相对路径
        # 如果配置文件中包含'path'键，需要将'train'和'val'路径转换为绝对路径
        if 'path' in data_config:
            base_path = data_config['path']
            # 将相对路径转换为绝对路径
            if 'train' in data_config:
                data_config['train'] = os.path.join(base_path, data_config['train'])
            if 'val' in data_config:
                data_config['val'] = os.path.join(base_path, data_config['val'])

        print(f"✓ 成功加载数据集配置: {yaml_path}")
        return data_config
    except FileNotFoundError:
        # 如果文件不存在，返回默认配置
        print(f"⚠ 警告：未找到配置文件 {yaml_path}，使用默认配置")
        return {
            'train': 'datasets/mock_data/images',
            'val': 'datasets/mock_data/images',
            'nc': 1,
            'names': ['object'],
        }
    except Exception as e:
        raise RuntimeError(f"加载配置文件失败: {e}")


def create_data_loaders(
    config: Dict,
    distributed: bool = False,
    rank: int = 0,
) -> tuple:
    """
    创建训练和验证数据加载器

    参数:
        config: 配置字典
        distributed: 是否使用分布式训练
        rank: 当前进程的rank

    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    train_loader = create_train_loader(config, distributed, rank)
    val_loader = create_val_loader(config, distributed, rank)

    return train_loader, val_loader