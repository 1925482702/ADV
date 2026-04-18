#!/usr/bin/env python3
"""
Teacher 模型训练脚本
支持通过命令行参数指定训练 RGB 或 IR 模型
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Teacher Model (RGB or IR)')
    
    # 模态选择
    parser.add_argument(
        '--modality',
        type=str,
        required=True,
        choices=['rgb', 'ir'],
        help='模态类型: rgb 或 ir'
    )
    
    # 模型配置
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8m.yaml',
        help='模型配置文件（相对于 ultralytics/cfg/models/）'
    )
    
    # 数据配置
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='数据配置文件（如果在 teacherTraining 目录下则不需要指定完整路径）'
    )
    
    # 训练参数
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='训练轮数'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='批次大小'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='图像尺寸'
    )
    
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.001,
        help='初始学习率'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='训练设备（如 0, 1, cpu）'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='数据加载线程数'
    )
    
    parser.add_argument(
        '--amp',
        action='store_true',
        help='启用混合精度训练'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='保存目录'
    )
    
    return parser.parse_args()


def get_data_config(modality, data_arg=None):
    """
    获取数据配置文件路径
    
    参数:
        modality: 'rgb' 或 'ir'
        data_arg: 用户指定的数据配置文件
    
    返回:
        data_config: 数据配置文件路径
    """
    if data_arg is not None:
        # 用户指定了数据配置文件
        return data_arg
    
    # 根据模态选择默认配置文件
    if modality == 'rgb':
        data_config = './FLIR_RGB.yaml'
    elif modality == 'ir':
        data_config = './FLIR_IR.yaml'
    else:
        raise ValueError(f"不支持的模态: {modality}")
    
    return data_config


def get_save_dir(modality, save_dir_arg=None):
    """
    获取保存目录
    
    参数:
        modality: 'rgb' 或 'ir'
        save_dir_arg: 用户指定的保存目录
    
    返回:
        save_dir: 保存目录
    """
    if save_dir_arg is not None:
        return save_dir_arg
    
    # 默认保存目录
    if modality == 'rgb':
        save_dir = 'runs/teacher/FLIR_RGB'
    elif modality == 'ir':
        save_dir = 'runs/teacher/FLIR_IR'
    else:
        raise ValueError(f"不支持的模态: {modality}")
    
    return save_dir


def main():
    """主函数"""
    args = parse_args()
    
    print("="*60)
    print(f"训练 Teacher 模型")
    print("="*60)
    print(f"模态: {args.modality.upper()}")
    print(f"模型: {args.model}")
    print(f"轮数: {args.epochs}")
    print(f"批次大小: {args.batch}")
    print(f"图像尺寸: {args.imgsz}")
    print(f"学习率: {args.lr0}")
    print(f"设备: {args.device}")
    print(f"工作线程: {args.workers}")
    print("="*60)
    
    # 获取数据配置
    data_config = get_data_config(args.modality, args.data)
    print(f"\n数据配置: {data_config}")
    
    # 检查数据配置文件是否存在
    data_config_path = Path(__file__).parent / data_config
    if not data_config_path.exists():
        print(f"\n❌ 错误：数据配置文件不存在: {data_config_path}")
        print(f"请确保 {data_config} 存在于 {Path(__file__).parent}")
        sys.exit(1)
    
    # 获取保存目录
    save_dir = get_save_dir(args.modality, args.save_dir)
    print(f"保存目录: {save_dir}")
    
    # 构建训练参数
    train_args = {
        'model': f"./ultralytics/cfg/models/v8/{args.model}",
        'data': data_config,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'lr0': args.lr0,
        'device': args.device,
        'workers': args.workers,
        'amp': args.amp,
        'augment': True,
        'rect': False,
        'cache': False,
        'patience': 50,
        'save': True,
        'save_period': -1,
        'val': True,
    }
    
    # 设置保存目录
    DEFAULT_CFG.save_dir = save_dir
    
    print("\n开始训练...")
    
    try:
        # 使用 DetectionTrainer 训练
        trainer = DetectionTrainer(overrides=train_args)
        trainer.train()
        
        print("\n" + "="*60)
        print("✓ 训练完成！")
        print(f"模型保存在: {save_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()