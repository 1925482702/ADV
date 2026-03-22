#!/usr/bin/env python3
"""
测试脚本
评估模型性能
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher import TeacherModel
from models.student import DualModalStudent
from data.loader import create_val_loader


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Test Model')

    # 模型路径
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Student模型检查点路径')
    parser.add_argument('--teacher_rgb', type=str, required=True,
                        help='RGB Teacher模型路径')
    parser.add_argument('--teacher_ir', type=str, required=True,
                        help='IR Teacher模型路径')

    # 数据集配置
    parser.add_argument('--data', type=str, default='data/FLIR.yaml',
                        help='数据集配置文件')
    parser.add_argument('--img_size', type=int, default=640,
                        help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='数据加载工作进程数')

    # 模型配置
    parser.add_argument('--num_classes', type=int, default=80,
                        help='类别数量')
    parser.add_argument('--fusion_channels', type=int, nargs='+', default=None,
                        help='融合后特征通道数列表 [P3, P4, P5]')

    # 设备配置
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')

    return parser.parse_args()


def load_teacher(teacher_path: str, device: torch.device) -> TeacherModel:
    """
    加载Teacher模型

    参数:
        teacher_path: Teacher模型路径
        device: 设备

    返回:
        teacher: Teacher模型
    """
    print(f"正在加载Teacher模型: {teacher_path}")
    teacher = TeacherModel.load_from_checkpoint(teacher_path, device=str(device))
    teacher.eval()
    teacher.freeze()
    print(f"✓ Teacher模型加载成功")
    return teacher


def load_student(
    checkpoint_path: str,
    teacher_rgb: TeacherModel,
    teacher_ir: TeacherModel,
    num_classes: int,
    fusion_channels: list,
    device: torch.device,
) -> DualModalStudent:
    """
    加载Student模型

    参数:
        checkpoint_path: Student检查点路径
        teacher_rgb: RGB Teacher模型
        teacher_ir: IR Teacher模型
        num_classes: 类别数量
        fusion_channels: 融合后特征通道数
        device: 设备

    返回:
        student: Student模型
    """
    print(f"正在加载Student模型: {checkpoint_path}")

    # 创建Student模型
    student = DualModalStudent(
        teacher_rgb=teacher_rgb,
        teacher_ir=teacher_ir,
        num_classes=num_classes,
        fusion_channels=fusion_channels,
        use_residual=True,
    ).to(device)

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student.load_state_dict(checkpoint['student_state_dict'])
    student.eval()

    print(f"✓ Student模型加载成功")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    return student


def compute_mAP(predictions, targets, num_classes: int = 80, conf_thres: float = 0.25, iou_thres: float = 0.45):
    """
    计算mAP（使用Ultralytics的后处理工具）

    参数:
        predictions: 预测结果列表，每个元素是 [pred_P3, pred_P4, pred_P5]
        targets: 目标标注列表，每个元素是 {'bboxes': ..., 'labels': ...}
        num_classes: 类别数量
        conf_thres: 置信度阈值
        iou_thres: IOU阈值

    返回:
        mAP: mAP@0.5:0.95值
    """
    try:
        from ultralytics.utils.metrics import Metrics
        from ultralytics.utils.ops import non_max_suppression
    except ImportError:
        print("警告：无法导入Ultralytics的Metrics和ops，返回占位值0.0")
        return 0.0

    # 初始化Metrics
    metrics = Metrics(names=['mAP50-95', 'mAP50', 'mAP75', 'mAPs', 'mAPm', 'mAPl'])

    # 遍历所有batch
    for batch_idx, (batch_preds, batch_targets) in enumerate(zip(predictions, targets)):
        # batch_preds 是 [pred_P3, pred_P4, pred_P5]
        # 需要将它们合并成YOLOv8的格式

        # YOLOv8的输出格式是 [B, 4 + num_classes, 8400]
        # 但我们的预测是列表格式，需要转换
        # 这里假设student的detect_head已经输出了YOLOv8格式

        # 处理predictions
        if isinstance(batch_preds, list) and len(batch_preds) == 3:
            # 合并三个尺度的预测
            # 每个尺度是 [B, 4 + num_classes, H, W]
            pred_P3, pred_P4, pred_P5 = batch_preds

            # 将每个尺度的预测展平成 [B, 4 + num_classes, H*W]
            pred_P3_flat = pred_P3.permute(0, 2, 3, 1).reshape(pred_P3.shape[0], -1, pred_P3.shape[1])
            pred_P4_flat = pred_P4.permute(0, 2, 3, 1).reshape(pred_P4.shape[0], -1, pred_P4.shape[1])
            pred_P5_flat = pred_P5.permute(0, 2, 3, 1).reshape(pred_P5.shape[0], -1, pred_P5.shape[1])

            # 拼接所有尺度的预测
            preds = torch.cat([pred_P3_flat, pred_P4_flat, pred_P5_flat], dim=1)
        else:
            # 如果已经是展平的格式，直接使用
            preds = batch_preds

        # NMS过滤
        preds = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres)

        # 准备targets格式
        # Ultralytics的Metrics需要targets格式: [batch_idx, class_id, x, y, w, h]
        tboxes = batch_targets['bboxes']  # [N, 4] (x_center, y_center, w, h, normalized)
        tcls = batch_targets['labels']    # [N,]

        # 转换为Ultralytics格式
        tboxes_ultralytics = torch.zeros((len(tboxes), 6), device=tboxes.device)
        tboxes_ultralytics[:, 0] = batch_idx  # batch_idx
        tboxes_ultralytics[:, 1] = tcls       # class_id
        tboxes_ultralytics[:, 2:] = tboxes    # x, y, w, h

        # 更新metrics
        metrics.update(preds, tboxes_ultralytics)

    # 计算mAP
    results = metrics.results
    mAP = results[0]  # mAP50-95

    return mAP


def test(
    student: DualModalStudent,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    测试模型

    参数:
        student: Student模型
        val_loader: 验证数据加载器
        device: 设备

    返回:
        metrics: 指标字典
    """
    print("\n开始测试...")

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Testing'):
            # 🚨 关键：Student接口必须正确
            # 数据已经在 dataset.py 的 Format 类中拼接成 [B, 6, H, W] 格式
            # 直接使用即可，无需分离和拼接
            x = batch['img'].to(device)  # [B, 6, H, W]

            # 准备目标
            targets = {
                'bboxes': batch['bboxes'],
                'labels': batch['cls'],
            }

            # 前向传播
            # 🚨 关键：Student接口不接受epsilon，只接受6通道输入
            predictions = student(x)

            # 保存预测和目标
            all_predictions.append(predictions)
            all_targets.append(targets)

    # 计算mAP
    print("\n计算mAP...")
    mAP = compute_mAP(all_predictions, all_targets)

    metrics = {
        'mAP': mAP,
    }

    return metrics


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("Testing Model")
    print("=" * 60)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载Teacher模型
    print("\n" + "=" * 60)
    print("Loading Teacher models...")
    print("=" * 60)
    teacher_rgb = load_teacher(args.teacher_rgb, device)
    teacher_ir = load_teacher(args.teacher_ir, device)

    # 加载Student模型
    print("\n" + "=" * 60)
    print("Loading Student model...")
    print("=" * 60)
    student = load_student(
        checkpoint_path=args.checkpoint,
        teacher_rgb=teacher_rgb,
        teacher_ir=teacher_ir,
        num_classes=args.num_classes,
        fusion_channels=args.fusion_channels,
        device=device,
    )

    # 创建数据加载器
    print("\n" + "=" * 60)
    print("Creating data loader...")
    print("=" * 60)
    val_loader = create_val_loader({
        'val_yaml': args.data,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
    })

    print(f"验证样本数: {len(val_loader.dataset)}")

    # 测试
    metrics = test(student, val_loader, device)

    # 打印结果
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"mAP@0.5: {metrics['mAP']:.4f}")

    print("\nTesting completed!")


if __name__ == '__main__':
    main()
