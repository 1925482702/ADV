#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 ObjectLevelShiftAugment 增强效果的可视化脚本
"""

import os
import sys
import random
import numpy as np
import cv2

# 添加路径
sys.path.insert(0, '/mnt/home/pyq_code/ADV/M2D-LIF')

from ultralytics.data.augment import ObjectLevelShiftAugment


def create_dummy_labels(h=640, w=640, n_objects=3):
    """创建模拟的labels数据"""
    # 创建RGB图像（随机颜色背景+物体）
    img_rgb = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
    img_ir = np.random.randint(80, 180, (h, w, 3), dtype=np.uint8)
    
    # 创建模拟的bboxes和instances
    bboxes = []
    
    # 创建几个不同大小的物体
    object_colors_rgb = [
        (255, 0, 0),    # 红色物体（大）
        (0, 255, 0),    # 绿色物体（中）
        (0, 0, 255),    # 蓝色物体（小）
    ]
    object_colors_ir = [
        (200, 50, 50),
        (50, 200, 50),
        (50, 50, 200),
    ]
    
    sizes = [
        (100, 120),  # 大物体
        (60, 80),    # 中物体
        (40, 50),    # 小物体
    ]
    positions = [
        (100, 100),
        (300, 250),
        (500, 450),
    ]
    
    for i, ((obj_w, obj_h), (cx, cy), color_rgb, color_ir) in enumerate(
        zip(sizes, positions, object_colors_rgb, object_colors_ir)
    ):
        x1 = cx - obj_w // 2
        y1 = cy - obj_h // 2
        x2 = cx + obj_w // 2
        y2 = cy + obj_h // 2
        
        # 确保在图像内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # 在RGB和IR上绘制物体（矩形）
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color_rgb, -1)
        cv2.rectangle(img_ir, (x1, y1), (x2, y2), color_ir, -1)
        
        # 归一化坐标
        bboxes.append([x1/w, y1/h, x2/w, y2/h])
    
    # 创建Instances对象
    from ultralytics.data.augment import Instances
    bboxes = np.array(bboxes)
    instances = Instances(bboxes=bboxes, segments=[], keypoints=None, bbox_format='xyxy', normalized=True)
    
    labels = {
        'img': img_rgb,
        'img_lwir': img_ir,
        'instances': instances,
    }
    
    return labels


def visualize_shift(labels_original, labels_shifted, save_path='shift_visualization.png'):
    """可视化平移效果"""
    h, w = labels_original['img'].shape[:2]
    
    # 创建画布
    canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    
    # 原始RGB（左上）
    canvas[:h, :w] = labels_original['img']
    cv2.putText(canvas, 'Original RGB', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 原始IR（右上）
    canvas[:h, w:] = labels_original['img_lwir']
    cv2.putText(canvas, 'Original IR', (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 平移后RGB（左下）
    canvas[h:, :w] = labels_shifted['img']
    cv2.putText(canvas, 'Shifted RGB', (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 平移后IR（右下）
    canvas[h:, w:] = labels_shifted['img_lwir']
    cv2.putText(canvas, 'Shifted IR', (w+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 绘制信息
    shift_applied = labels_shifted.get('shift_applied', False)
    if shift_applied:
        shift_dx = labels_shifted.get('shift_dx', 0) * w
        shift_dy = labels_shifted.get('shift_dy', 0) * h
        shift_modality = labels_shifted.get('shift_modality', -1)
        modality_name = 'RGB' if shift_modality == 0 else 'IR' if shift_modality == 1 else 'None'
        
        info_text = f'Shift: dx={shift_dx:.1f}px, dy={shift_dy:.1f}px, Modality: {modality_name}'
        cv2.putText(canvas, info_text, (10, h*2-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 绘制目标bbox（平移前位置）
        if 'shift_target_bbox' in labels_shifted:
            x1, y1, x2, y2 = labels_shifted['shift_target_bbox'].astype(int)
            # 在对应模态上画框（原位置，用虚线表示）
            if shift_modality == 0:
                # RGB被平移，在原始RGB上画框
                cv2.rectangle(canvas[:h, :w], (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(canvas[:h, :w], 'Target (orig pos)', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                # IR被平移，在原始IR上画框
                cv2.rectangle(canvas[:h, w:], (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(canvas[:h, w:], 'Target (orig pos)', (w+x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.putText(canvas, 'No shift applied', (10, h*2-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 绘制所有bbox
    instances = labels_original['instances']
    bboxes = instances.bboxes.copy()
    bboxes[:, [0, 2]] *= w
    bboxes[:, [1, 3]] *= h
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox.astype(int)
        # 在原始图像上画框
        cv2.rectangle(canvas[:h, :w], (x1, y1), (x2, y2), (255, 255, 0), 1)
        cv2.rectangle(canvas[:h, w:], (x1, y1), (x2, y2), (255, 255, 0), 1)
    
    cv2.imwrite(save_path, canvas)
    print(f'Saved to: {save_path}')
    return canvas


def test_multiple_cases(n_tests=5):
    """测试多种情况"""
    augment = ObjectLevelShiftAugment(shift_range=(10, 50), shift_prob=1.0)  # 100%概率测试
    
    os.makedirs('/mnt/home/pyq_code/ADV/M2D-LIF/shift_test_results', exist_ok=True)
    
    for i in range(n_tests):
        print(f'\n=== Test {i+1} ===')
        
        # 创建模拟数据
        labels = create_dummy_labels()
        
        # 保存原始
        labels_original = {
            'img': labels['img'].copy(),
            'img_lwir': labels['img_lwir'].copy(),
            'instances': labels['instances'],
        }
        
        # 应用增强
        labels_shifted = augment(labels)
        
        # 打印信息
        print(f"  shift_applied: {labels_shifted.get('shift_applied', False)}")
        print(f"  shift_modality: {labels_shifted.get('shift_modality', -1)}")
        print(f"  shift_dx: {labels_shifted.get('shift_dx', 0):.4f}")
        print(f"  shift_dy: {labels_shifted.get('shift_dy', 0):.4f}")
        
        # 可视化
        save_path = f'/mnt/home/pyq_code/ADV/M2D-LIF/shift_test_results/test_{i+1}.png'
        visualize_shift(labels_original, labels_shifted, save_path)


def test_with_real_data():
    """用真实数据测试"""
    import glob
    
    # 查找FLIR数据集
    rgb_pattern = '/mnt/home/pyq_code/ADV/datasets/FLIR_yolo/images/train/*.jpg'
    label_pattern = '/mnt/home/pyq_code/ADV/datasets/FLIR_yolo/labels/train/*.txt'
    
    rgb_files = sorted(glob.glob(rgb_pattern))
    label_files = sorted(glob.glob(label_pattern))
    
    if not rgb_files:
        print("未找到FLIR数据集，跳过真实数据测试")
        return
    
    print(f"找到 {len(rgb_files)} RGB图像和 {len(label_files)} 标签文件")
    
    augment = ObjectLevelShiftAugment(shift_range=(10, 50), shift_prob=1.0)
    
    os.makedirs('/mnt/home/pyq_code/ADV/M2D-LIF/shift_test_results/real', exist_ok=True)
    
    test_count = 0
    for rgb_path in rgb_files:
        if test_count >= 5:
            break
        
        # 获取对应的标签文件路径
        basename = os.path.basename(rgb_path).replace('.jpg', '.txt')
        label_path = f'/mnt/home/pyq_code/ADV/datasets/FLIR_yolo/labels/train/{basename}'
        ir_path = rgb_path.replace('images', 'images_ir')
        
        if not os.path.exists(label_path) or not os.path.exists(ir_path):
            continue
        
        # 读取图像
        img_rgb = cv2.imread(rgb_path)
        img_ir = cv2.imread(ir_path)
        
        if img_rgb is None or img_ir is None:
            continue
        
        # 读取标签文件
        bboxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO格式: class x_center y_center width height (归一化)
                    _, xc, yc, w, h = map(float, parts[:5])
                    # 转换为xyxy格式
                    x1 = xc - w / 2
                    y1 = yc - h / 2
                    x2 = xc + w / 2
                    y2 = yc + h / 2
                    bboxes.append([x1, y1, x2, y2])
        
        if len(bboxes) == 0:
            continue
        
        test_count += 1
        print(f'\n处理: {os.path.basename(rgb_path)}, {len(bboxes)}个目标')
        
        # 调整大小
        orig_h, orig_w = img_rgb.shape[:2]
        img_rgb = cv2.resize(img_rgb, (640, 640))
        img_ir = cv2.resize(img_ir, (640, 640))
        
        from ultralytics.data.augment import Instances
        instances = Instances(
            bboxes=np.array(bboxes),
            segments=[],
            keypoints=None,
            bbox_format='xyxy',
            normalized=True
        )
        
        labels = {
            'img': img_rgb,
            'img_lwir': img_ir,
            'instances': instances,
        }
        
        labels_original = {
            'img': img_rgb.copy(),
            'img_lwir': img_ir.copy(),
            'instances': instances,
        }
        
        labels_shifted = augment(labels)
        
        print(f"  shift_modality: {labels_shifted.get('shift_modality', -1)}")
        print(f"  shift_dx: {labels_shifted.get('shift_dx', 0):.4f}")
        print(f"  shift_dy: {labels_shifted.get('shift_dy', 0):.4f}")
        
        save_path = f'/mnt/home/pyq_code/ADV/M2D-LIF/shift_test_results/real/test_{test_count}.png'
        visualize_shift(labels_original, labels_shifted, save_path)


if __name__ == '__main__':
    print("=" * 60)
    print("ObjectLevelShiftAugment 测试")
    print("=" * 60)
    
    # 测试模拟数据
    print("\n[1] 测试模拟数据...")
    test_multiple_cases(n_tests=5)
    
    # 测试真实数据
    print("\n[2] 测试真实数据...")
    test_with_real_data()
    
    print("\n" + "=" * 60)
    print("测试完成！结果保存在 shift_test_results/ 目录")
    print("=" * 60)
