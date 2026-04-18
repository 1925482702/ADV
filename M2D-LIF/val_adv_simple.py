#!/usr/bin/env python3
"""
ADV 双模态模型验证脚本（使用官方验证器）

由于 DualModalStudent 已经完全兼容 YOLO 官方验证器，
可以直接使用官方的 DetectionValidator 进行 mAP 计算。
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description='ADV 模型验证')
    parser.add_argument('--model', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--teacher', type=str, default=None, help='Teacher 模型路径（加载 Student 时需要）')
    parser.add_argument('--data', type=str, default='/root/autodl-tmp/ADV/yaml/data/FLIR.yaml', help='数据集配置')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--device', type=str, default='0', help='设备')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.7, help='NMS IoU 阈值')
    parser.add_argument('--project', type=str, default='./runs/val', help='输出目录')
    parser.add_argument('--name', type=str, default='adv_val', help='实验名称')
    return parser.parse_args()


def load_model(model_path: str, teacher_path: str, device: str):
    """加载模型（支持 Student 和 Teacher/Baseline）"""
    print(f"加载模型: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    print(f"检查点键: {list(checkpoint.keys())}")
    
    # 提取模型对象
    if 'model' in checkpoint:
        model = checkpoint['model']
        print(f"检查点['model'] 类型: {type(model)}")
        if hasattr(model, 'detect_head'):
            print(f"✓ 模型有 detect_head 属性")
        if hasattr(model, 'state_dict'):
            print(f"✓ 模型有 state_dict 方法")
    else:
        model = checkpoint
        print(f"检查点类型: {type(model)}")
        # 如果是 state_dict 形式，需要重建模型
        if not hasattr(model, 'forward'):
            print("  检测到 state_dict 格式，需要重建模型...")
            
            # 需要 Teacher 来初始化 Student 结构
            if teacher_path is None:
                teacher_path = '/root/autodl-tmp/ADV/runs/baseline/yolov8s_naive_add5/weights/best.pt'
            
            print(f"  加载 Teacher: {teacher_path}")
            teacher_ckpt = torch.load(teacher_path, map_location=device)
            teacher = teacher_ckpt['model'] if 'model' in teacher_ckpt else teacher_ckpt
            
            # 创建 Student
            from ADV.models.student import DualModalStudent
            model = DualModalStudent(
                baseline_teacher=teacher,
                num_classes=3,
                early_layer_idx=2,
                use_residual=True,
                fusion_mode='add',
            )
            
            # 加载权重
            state_dict = checkpoint['model']
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
            model.load_state_dict(state_dict, strict=False)
            print("  ✓ Student 权重加载完成")
    else:
        model = checkpoint
    
    model = model.to(device)
    model.eval()
    
    return model


def main():
    args = parse_args()
    
    print("=" * 60)
    print("ADV 模型验证（官方验证器）")
    print("=" * 60)
    
    device = f'cuda:{args.device}' if args.device.isdigit() else args.device
    print(f"设备: {device}")
    
    # 加载模型
    model = load_model(args.model, args.teacher, device)
    
    # 加载数据配置
    with open(args.data, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    print(f"\n数据集: {args.data}")
    print(f"类别数: {data_cfg.get('nc', 3)}")
    print(f"通道数: {data_cfg.get('ch', 3)}")
    
    # 创建保存目录
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用官方验证器
    print("\n使用官方 DetectionValidator 验证...")
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.data import build_dataloader, YOLODataset
    from ultralytics.cfg import get_cfg
    
    # 构建验证参数
    val_args = get_cfg(overrides={
        'conf': args.conf,
        'iou': args.iou,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'plots': False,
        'save_json': False,
        'half': False,
        'device': device,
    })
    
    # 构建数据集
    data_path = Path(data_cfg.get('path', ''))
    val_path = data_cfg.get('val', data_cfg.get('valid', 'images'))
    if not os.path.isabs(val_path):
        val_path = str(data_path / val_path)
    
    dataset = YOLODataset(
        img_path=val_path,
        imgsz=args.imgsz,
        data=data_cfg,
        task='detect',
        augment=False,
        rect=True,
    )
    
    dataloader = build_dataloader(
        dataset,
        batch=args.batch,
        workers=0,
        shuffle=False,
        rank=-1,
    )
    
    print(f"验证样本数: {len(dataset)}")
    
    # 创建验证器
    validator = DetectionValidator(
        dataloader=dataloader,
        save_dir=save_dir,
        args=val_args,
    )
    validator.data = data_cfg
    
    # 执行验证
    print("\n开始验证...")
    validator(model=model)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)
    print(f"  P (Precision):     {validator.metrics.p:.4f}")
    print(f"  R (Recall):        {validator.metrics.r:.4f}")
    print(f"  mAP50:             {validator.metrics.map50:.4f}")
    print(f"  mAP50-95:          {validator.metrics.map:.4f}")
    print(f"  Fitness:           {validator.metrics.fitness:.4f}")
    print(f"\n结果保存至: {save_dir}")
    print("=" * 60)
    
    return validator.metrics


if __name__ == '__main__':
    main()
