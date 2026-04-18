#!/usr/bin/env python3
"""
ADV 双模态验证脚本

验证 DualModalStudent 模型在 FLIR 数据集上的 mAP

使用方法:
    python val_adv.py --student path/to/student.pt --teacher path/to/baseline.pt
    
    # 或直接使用 Teacher 模型验证（不需要单独的 Student）
    python val_adv.py --teacher path/to/baseline.pt
"""

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch

# 确保 ADV 模块可导入
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ADV 双模态模型验证')
    
    # 模型参数
    parser.add_argument(
        '--student', 
        type=str, 
        default=None,
        help='Student 模型检查点路径（如果提供，使用训练好的 Student）'
    )
    parser.add_argument(
        '--teacher', 
        type=str, 
        default='/root/autodl-tmp/ADV/runs/baseline/yolov8s_naive_add5/weights/best.pt',
        help='Baseline Teacher 模型路径（用于构建 Student 结构）'
    )
    
    # 数据参数
    parser.add_argument(
        '--data', 
        type=str, 
        default='/root/autodl-tmp/ADV/yaml/data/FLIR.yaml',
        help='数据集配置文件'
    )
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--device', type=str, default='0', help='设备')
    parser.add_argument('--workers', type=int, default=0, help='数据加载线程数（0 禁用多进程）')
    
    # 验证参数（使用官方默认值）
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.7, help='NMS IoU 阈值')
    parser.add_argument('--max_det', type=int, default=300, help='每张图最大检测数')
    parser.add_argument('--plots', action='store_true', help='保存验证结果图')
    parser.add_argument('--save_json', action='store_true', help='保存 COCO JSON 结果')
    parser.add_argument('--rebuild_cache', action='store_true', help='重建数据缓存')
    
    # 输出参数
    parser.add_argument('--project', type=str, default='./runs/val', help='输出目录')
    parser.add_argument('--name', type=str, default='adv_val', help='实验名称')
    
    return parser.parse_args()


def load_teacher(teacher_path: str, device: str):
    """加载 Teacher 模型（直接加载检查点，保留自定义结构）"""
    print(f"加载 Teacher 模型: {teacher_path}")
    
    # 直接加载检查点
    checkpoint = torch.load(teacher_path, map_location=device)
    
    if 'model' in checkpoint:
        model = checkpoint['model']
    else:
        model = checkpoint
    
    # 设置为评估模式
    model.eval()
    model = model.to(device)
    
    # 如果是 fp16 模型，保持 fp16
    if next(model.parameters()).dtype == torch.float16:
        model = model.half()
    
    print(f"✓ Teacher 模型加载完成")
    return model


def load_student(student_path: str, device: str, teacher_model=None):
    """
    加载训练好的 Student 模型
    
    参数:
        student_path: Student 检查点路径
        device: 设备
        teacher_model: 可选的 Teacher 模型。如果为 None，会自动加载或创建临时 Teacher
    
    注意:
        推理时使用 forward_inference() 方法，不需要 Teacher 特征
    """
    from ADV.models.student import DualModalStudent
    
    print(f"加载 Student 模型: {student_path}")
    
    # 从检查点加载
    checkpoint = torch.load(student_path, map_location=device)
    
    # 提取 state_dict（可能是模型对象或 state_dict）
    if 'model' in checkpoint:
        model_or_state = checkpoint['model']
        # 检查是模型对象还是 state_dict
        if hasattr(model_or_state, 'state_dict'):
            # 是模型对象，提取 state_dict
            state_dict = model_or_state.state_dict()
            print("  从检查点提取模型 state_dict")
        elif hasattr(model_or_state, 'items'):
            # 已经是 state_dict (dict-like)
            state_dict = model_or_state
        else:
            raise TypeError(f"未知的模型格式: {type(model_or_state)}")
    else:
        state_dict = checkpoint
    
    # 如果没有提供 Teacher，使用 Teacher 路径加载
    if teacher_model is None:
        print("  需要 Teacher 模型来初始化 Student 结构...")
        # 尝试使用默认 Teacher 路径
        default_teacher = '/root/autodl-tmp/ADV/runs/baseline/yolov8s_naive_add5/weights/best.pt'
        if os.path.exists(default_teacher):
            teacher_model = load_teacher(default_teacher, device)
        else:
            raise ValueError(
                "加载 Student 需要 Teacher 模型来初始化结构。\n"
                "请提供 --teacher 参数，或确保默认 Teacher 路径存在：\n"
                f"  {default_teacher}"
            )
    
    # 创建 Student 模型（需要 Teacher 初始化结构）
    student = DualModalStudent(
        baseline_teacher=teacher_model,
        num_classes=3,
        early_layer_idx=2,
        use_residual=True,
        fusion_mode='add',
    )
    
    # 加载权重
    student.load_state_dict(state_dict, strict=False)
    student = student.to(device)
    student.eval()
    
    print(f"✓ Student 模型加载完成（使用 forward_inference 进行独立推理）")
    return student


def build_val_dataloader(args, data_cfg):
    """构建验证数据加载器"""
    from ultralytics.data import build_dataloader, YOLODataset
    from pathlib import Path
    
    # 获取数据路径
    data_path = Path(data_cfg.get('path', ''))
    val_path = data_cfg.get('val', data_cfg.get('valid', data_cfg.get('test', '')))
    
    if not os.path.isabs(val_path):
        val_path = str(data_path / val_path)
    
    print(f"验证数据路径: {val_path}")
    
    # 使用官方 YOLODataset（支持 6 通道）
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
        workers=args.workers,
        shuffle=False,
        rank=-1,
    )
    
    return dataloader


def validate(student_model, dataloader, args, save_dir, data_cfg):
    """执行验证"""
    from ADV.validator import DualModalValidator
    from ultralytics.utils import LOGGER
    from ultralytics.cfg import get_cfg
    
    # 创建验证参数（使用 get_cfg 处理）
    val_args = get_cfg(overrides={
        'conf': args.conf,
        'iou': args.iou,
        'max_det': args.max_det,
        'save_hybrid': False,
        'half': False,
        'single_cls': False,
        'plots': args.plots,
        'save_json': args.save_json,
        'split': 'val',
        'task': 'detect',
    })
    
    # 创建验证器
    validator = DualModalValidator(
        dataloader=dataloader,
        save_dir=save_dir,
        args=val_args,
    )
    validator.device = f'cuda:{args.device}' if args.device.isdigit() else args.device
    validator.data = data_cfg  # 传入数据配置
    
    # 执行验证
    LOGGER.info("开始验证...")
    results = validator(model=student_model)
    
    return results


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("ADV 双模态模型验证")
    print("=" * 60)
    
    # 设置设备
    device = f'cuda:{args.device}' if args.device.isdigit() else args.device
    print(f"设备: {device}")
    
    # 加载 Teacher（如果需要）
    teacher_model = None
    if args.teacher:
        teacher_model = load_teacher(args.teacher, device)
    
    # 加载 Student 或使用 Teacher
    if args.student:
        student_model = load_student(args.student, device, teacher_model)
    elif teacher_model:
        # 直接使用 Teacher 模型（已经是 6 通道双模态模型）
        print("未提供 Student 模型，直接使用 Teacher 进行验证")
        student_model = teacher_model
    else:
        raise ValueError("请提供 --teacher 或 --student 参数")
    
    # 加载数据配置
    import yaml
    with open(args.data, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    print(f"\n数据集: {args.data}")
    print(f"类别: {data_cfg.get('names', {})}")
    print(f"通道数: {data_cfg.get('ch', 3)}")
    
    # 构建数据加载器
    print("\n构建验证数据加载器...")
    dataloader = build_val_dataloader(args, data_cfg)
    print(f"验证样本数: {len(dataloader.dataset)}")
    
    # 创建保存目录
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 执行验证
    results = validate(student_model, dataloader, args, save_dir, data_cfg)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)
    
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, (int, float, str)):
                print(f"  {key}: {value}")
            elif isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item() if value.numel() == 1 else value.shape}")
    
    print(f"\n结果保存至: {save_dir}")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
