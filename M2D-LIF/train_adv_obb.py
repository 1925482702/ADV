#!/usr/bin/env python3
"""
ADV 对抗蒸馏训练 (OBB版本) —— 基于 ultralytics 插件方案

用法:
    python train_adv_obb.py
    python train_adv_obb.py --scale s --epochs 100 --batch 16
"""

import argparse
import sys
from pathlib import Path

# 确保能找到 ultralytics
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description='ADV 对抗蒸馏训练 (OBB)')
    parser.add_argument('--scale', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--data', type=str, default='./data/DroneVehicle.yaml',
                        help='数据集配置文件 (OBB格式)')
    parser.add_argument('--model', type=str, default='./model_yaml_obb/yolov8_obb_fusion.yaml',
                        help='模型配置文件')
    parser.add_argument('--teacher', type=str,
                        default='./runs/baseline/yolov8s_obb/weights/best.pt',
                        help='Baseline teacher 权重路径')
    parser.add_argument('--project', type=str, default='./runs/adv_obb',
                        help='项目保存路径')
    parser.add_argument('--name', type=str, default='adv_obb',
                        help='实验名称')
    # ADV 专用参数
    parser.add_argument('--warmup_frac', type=float, default=0.10, help='Warmup 占比（纯检测）')
    parser.add_argument('--finetune_frac', type=float, default=0.10, help='Finetune 占比（纯检测）')
    parser.add_argument('--epsilon_max', type=float, default=0.016, help='FGSM 噪声最大值 (4/255≈0.016)')
    parser.add_argument('--lambda_max', type=float, default=0.5, help='蒸馏权重最大值')
    args = parser.parse_args()

    # ultralytics 训练参数
    overrides = dict(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=8,
        project=args.project,
        name=args.name,
    )

    # ADV 调度参数
    adv_cfg = dict(
        warmup_frac=args.warmup_frac,
        finetune_frac=args.finetune_frac,
        epsilon_max=args.epsilon_max,
        lambda_max=args.lambda_max,
    )

    from ultralytics.models.yolo.obb.train_adv import ADVOBBTrainer

    print('=' * 60)
    print('ADV 对抗蒸馏训练 (OBB版本)')
    print('=' * 60)
    print(f'  Model:   {args.model}')
    print(f'  Data:    {args.data}')
    print(f'  Teacher: {args.teacher}')
    print(f'  Epochs:  {args.epochs}')
    print(f'  ADV cfg: {adv_cfg}')
    print('=' * 60)

    trainer = ADVOBBTrainer(
        overrides=overrides,
        teacher_weights=args.teacher,
        adv_cfg=adv_cfg,
    )
    trainer.train()

    print(f'\nTraining complete. Results saved to {trainer.save_dir}')


if __name__ == '__main__':
    main()
