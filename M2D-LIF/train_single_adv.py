#!/usr/bin/env python3
"""
单模态对抗训练脚本

使用方法:
    # 基本用法
    python train_single_adv.py --data FLIR_IR.yaml --epochs 100

    # 指定 GPU 和批量大小
    python train_single_adv.py --data FLIR_IR.yaml --epochs 100 --device 0 --batch 16

    # 自定义对抗训练参数
    python train_single_adv.py --data FLIR_IR.yaml --epochs 100 --epsilon-max 0.05 --early-layer-idx 2

训练流程:
    Stage 0 (0~5 epochs): Warmup
        - 纯检测训练，不加噪

    Stage 1 (5~80 epochs): 对抗训练
        - 干净前向 → 获取梯度 → 生成噪声
        - 加噪前向 → 真正训练
        - epsilon 线性增长

    Stage 2 (80~100 epochs): 干净微调
        - 不加噪，巩固检测能力
"""

import argparse
import os
import sys

# 添加 ultralyticsSingle 到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='单模态对抗训练')
    
    # 数据相关
    parser.add_argument('--data', type=str, required=True,
                        help='数据集配置文件路径
    
    # 模型相关
    parser.add_argument('--model', type=str, default='yolov8s.yaml',
                        help='模型配置文件或预训练权重路径')
    parser.add_argument('--weights', type=str, default=None,
                        help='预训练权重路径（可选）')
    
    # 训练相关
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=16,
                        help='批量大小')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像大小')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU 设备
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载线程数')
    parser.add_argument('--project', type=str, default='./runs/single_adv',
                        help='保存目录')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    
    # 对抗训练参数
    parser.add_argument('--adv-enabled', action='store_true', default=True,
                        help='是否启用对抗训练')
    parser.add_argument('--no-adv', action='store_true',
                        help='禁用对抗训练（使用标准训练）')
    parser.add_argument('--epsilon-max', type=float, default=0.05,
                        help='最大噪声强度 (default: 0.05)')
    parser.add_argument('--early-layer-idx', type=int, default=2,
                        help='早期层索引，噪声注入位置 (default: 2)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='对抗训练 warmup 轮数 (default: 5)')
    parser.add_argument('--stage2-ratio', type=float, default=0.2,
                        help='干净微调阶段占比 (default: 0.2)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 处理对抗训练开关
    if args.no_adv:
        args.adv_enabled = False
    
    print("=" * 60)
    print("单模态对抗训练配置:")
    print(f"  数据集: {args.data}")
    print(f"  模型: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch}")
    print(f"  Image Size: {args.imgsz}")
    print(f"  Device: {args.device}")
    print("-" * 60)
    print("对抗训练参数:")
    print(f"  启用对抗: {args.adv_enabled}")
    print(f"  Epsilon Max: {args.epsilon_max}")
    print(f"  Early Layer Index: {args.early_layer_idx}")
    print(f"  Warmup Epochs: {args.warmup_epochs}")
    print(f"  Stage2 Ratio: {args.stage2_ratio}")
    print("=" * 60)
    
    # 加载模型
    model = YOLO(args.model)
    
    # 如果提供了预训练权重
    if args.weights:
        print(f"加载预训练权重: {args.weights}")
        model.load(args.weights)
    
    # 准备训练参数
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        # 对抗训练参数
        'adv_enabled': args.adv_enabled,
        'epsilon_max': args.epsilon_max,
        'early_layer_idx': args.early_layer_idx,
        'warmup_epochs': args.warmup_epochs,
        'stage2_ratio': args.stage2_ratio,
    }
    
    # 开始训练
    # 使用自定义的单模态对抗训练器
    from ultralytics.models.yolo.detect.train_adv import SingleModalADVTrainer
    
    trainer = SingleModalADVTrainer(overrides=train_args)
    trainer.train()
    
    print("\n训练完成！")
    print(f"结果保存在: {args.project}/{args.name}")


if __name__ == '__main__':
    main()
