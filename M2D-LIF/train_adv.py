#!/usr/bin/env python3
"""
ADV 对抗蒸馏训练 —— 基于 ultralytics 插件方案

用法:
    python train_adv.py
    python train_adv.py --scale s --epochs 100 --batch 16
"""

import argparse
import sys
from pathlib import Path

# 确保能找到 ultralytics
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description='ADV 对抗蒸馏训练')
    parser.add_argument('--scale', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--teacher', type=str,
                        default='./runs/baseline/yolov8s_naive_add2/weights/best.pt',
                        help='Baseline teacher 权重路径')
    # ADV 专用参数
    parser.add_argument('--warmup_frac', type=float, default=0.10, help='Warmup 占比（纯检测）')
    parser.add_argument('--finetune_frac', type=float, default=0.10, help='Finetune 占比（纯检测）')
    parser.add_argument('--epsilon_max', type=float, default=0.016, help='FGSM 噪声最大值 (4/255≈0.016)')
    parser.add_argument('--lambda_max', type=float, default=0.5, help='蒸馏权重最大值')
    args = parser.parse_args()

    # 模型 yaml
    model_yaml = f'./model_yaml/yolov8_naive_add.yaml'

    # ultralytics 训练参数
    overrides = dict(
        model=model_yaml,
        data='./data/FLIR.yaml',
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=8,
        project='./runs/adv',
        name=f'yolov8{args.scale}_adv',
    )

    # ADV 调度参数
    adv_cfg = dict(
        warmup_frac=args.warmup_frac,
        finetune_frac=args.finetune_frac,
        epsilon_max=args.epsilon_max,
        lambda_max=args.lambda_max,
    )

    from ultralytics.models.yolo.detect.train_adv import ADVDetectionTrainer

    print('=' * 60)
    print('ADV 对抗蒸馏训练（ultralytics 插件方案）')
    print('=' * 60)
    print(f'  Model:   {model_yaml}')
    print(f'  Teacher: {args.teacher}')
    print(f'  Epochs:  {args.epochs}')
    print(f'  ADV cfg: {adv_cfg}')
    print('=' * 60)

    trainer = ADVDetectionTrainer(
        overrides=overrides,
        teacher_weights=args.teacher,
        adv_cfg=adv_cfg,
    )
    trainer.train()

    print(f'\nTraining complete. Results saved to {trainer.save_dir}')


if __name__ == '__main__':
    main()
