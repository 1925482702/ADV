"""
Shift训练启动脚本

跨模态物体平移预测训练
用于验证"强迫模型进行跨模态对比"的idea
"""

import argparse
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics.models.yolo.detect import ShiftDetectionTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Shift Detection Training')
    parser.add_argument('--model', type=str, default='./model_yaml/yolov8_naive_add.yaml',
                        help='model config file path')
    parser.add_argument('--epochs', type=int, default=100,
                        help='total training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='batch size')
    parser.add_argument('--device', type=str, default='0',
                        help='cuda device, e.g. 0 or cpu')
    parser.add_argument('--data', type=str, default='./data/FLIR.yaml',
                        help='dataset config file')
    
    # Shift 参数
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='warmup epochs before shift training')
    parser.add_argument('--p-start', type=float, default=0.3,
                        help='initial shift probability')
    parser.add_argument('--p-end', type=float, default=0.8,
                        help='final shift probability (max 0.8, keep 20% no-shift samples)')
    parser.add_argument('--lambda-start', type=float, default=0.1,
                        help='initial shift loss weight')
    parser.add_argument('--lambda-end', type=float, default=0.5,
                        help='final shift loss weight')
    parser.add_argument('--shift-range-min', type=int, default=10,
                        help='min shift in pixels')
    parser.add_argument('--shift-range-max', type=int, default=50,
                        help='max shift in pixels')
    parser.add_argument('--name', type=str, default='shift_exp',
                        help='experiment name')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查文件存在
    if not os.path.exists(args.model):
        print(f"❌ 模型配置文件不存在: {args.model}")
        return
    
    if not os.path.exists(args.data):
        print(f"❌ 数据集配置文件不存在: {args.data}")
        return
    
    print("=" * 60)
    print("  Shift Detection Training")
    print("  跨模态物体平移预测训练")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch: {args.batch}")
    print(f"Device: {args.device}")
    print("-" * 60)
    print("Shift Config:")
    print(f"  Warmup Epochs: {args.warmup_epochs}")
    print(f"  Shift Prob: {args.p_start} -> {args.p_end}")
    print(f"  Lambda: {args.lambda_start} -> {args.lambda_end}")
    print(f"  Shift Range: [{args.shift_range_min}, {args.shift_range_max}] pixels")
    print(f"  Fusion Layer: auto-detect")
    print("=" * 60)
    
    # Shift 配置（融合层自动检测）
    shift_cfg = {
        'warmup_epochs': args.warmup_epochs,
        'p_start': args.p_start,
        'p_end': args.p_end,
        'lambda_start': args.lambda_start,
        'lambda_end': args.lambda_end,
    }
    
    # 创建训练器
    trainer = ShiftDetectionTrainer(
        overrides={
            'model': args.model,
            'data': args.data,
            'epochs': args.epochs,
            'imgsz': 640,
            'batch': args.batch,
            'device': args.device,
            'workers': 4,  # WSL 环境下不要设太高
            'project': './runs/shift',
            'name': args.name,
            'lr0': 0.01,
            'visualize': False,  # 禁用 TensorBoard 图可视化，避免 0 channels 警告
        },
        shift_cfg=shift_cfg,
    )
    
    # 开始训练
    trainer.train()
    
    print("\n" + "=" * 60)
    print("  Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
