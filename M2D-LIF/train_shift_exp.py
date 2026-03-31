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

from ultralytics.models.yolo.detect import DetectionTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Shift Detection Training')
    parser.add_argument('--model', type=str, default='./model_yaml/yolov8_shift.yaml',
                        help='model config file path')
    parser.add_argument('--epochs', type=int, default=100,
                        help='total training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='batch size')
    parser.add_argument('--device', type=str, default='0',
                        help='cuda device, e.g. 0 or cpu')
    parser.add_argument('--data', type=str, default='./data/FLIR.yaml',
                        help='dataset config file')
    parser.add_argument('--name', type=str, default='shift_exp',
                        help='experiment name')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='image size')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--workers', type=int, default=4,
                        help='dataloader workers')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查文件存在
    if not os.path.exists(args.model):
        print(f"Model config not found: {args.model}")
        return
    
    if not os.path.exists(args.data):
        print(f"Dataset config not found: {args.data}")
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
    print(f"Image Size: {args.imgsz}")
    print("=" * 60)
    
    # 创建训练器
    trainer = DetectionTrainer(
        overrides={
            'model': args.model,
            'data': args.data,
            'epochs': args.epochs,
            'imgsz': args.imgsz,
            'batch': args.batch,
            'device': args.device,
            'workers': args.workers,
            'project': './runs/shift',
            'name': args.name,
            'lr0': args.lr0,
            'visualize': False,  # 禁用 TensorBoard 图可视化
        },
    )
    
    # 开始训练
    trainer.train()
    
    print("\n" + "=" * 60)
    print("  Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
