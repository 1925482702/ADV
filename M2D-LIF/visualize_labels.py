#!/usr/bin/env python3
"""
可视化 DroneVehicle 数据集的标注框
从 train 目录随机选择图片，绘制标注框并保存
"""

import os
import random
import cv2
import numpy as np
from pathlib import Path

# 数据集路径
DATASET_ROOT = "/mnt/home/pyq_code/ADV/datasets/DroneVehicle_yolo"

# 要处理的图像目录（RGB和IR）
IMAGE_DIRS = ["images", "images_ir"]

# 每个目录随机选择的图片数量
NUM_SAMPLES = 5

# 输出目录
OUTPUT_DIR = "./label_visualization"


def yolo_to_bbox(yolo_box, img_w, img_h):
    """
    将 YOLO 格式转换为 bbox 坐标
    YOLO: class_id x_center y_center width height (归一化)
    bbox: x1, y1, x2, y2 (像素坐标)
    """
    x_center, y_center, w, h = yolo_box
    x1 = int((x_center - w / 2) * img_w)
    y1 = int((y_center - h / 2) * img_h)
    x2 = int((x_center + w / 2) * img_w)
    y2 = int((y_center + h / 2) * img_h)
    return x1, y1, x2, y2


def draw_boxes(image, label_path, class_names=None):
    """
    在图像上绘制标注框
    """
    if not os.path.exists(label_path):
        print(f"  警告: 标签文件不存在: {label_path}")
        return image, 0
    
    img_h, img_w = image.shape[:2]
    box_count = 0
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:5])
            
            # 转换为像素坐标
            x1, y1, x2, y2 = yolo_to_bbox([x_center, y_center, w, h], img_w, img_h)
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))
            
            # 绘制红色框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 添加类别标签
            label_text = f"class_{class_id}" if class_names is None else class_names.get(class_id, f"class_{class_id}")
            cv2.putText(image, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            box_count += 1
    
    return image, box_count


def visualize_dataset():
    """
    主函数：可视化数据集标注
    """
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for img_dir_name in IMAGE_DIRS:
        print(f"\n{'='*50}")
        print(f"处理目录: {img_dir_name}")
        print(f"{'='*50}")
        
        # 构建路径
        train_img_dir = os.path.join(DATASET_ROOT, img_dir_name, "train")
        train_label_dir = os.path.join(DATASET_ROOT, "labels", "train")
        
        if not os.path.exists(train_img_dir):
            print(f"  警告: 图像目录不存在: {train_img_dir}")
            continue
        
        # 获取所有图片文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(train_img_dir).glob(ext))
        
        if not image_files:
            print(f"  警告: 未找到图片文件")
            continue
        
        # 随机选择图片
        selected = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))
        
        print(f"  找到 {len(image_files)} 张图片，随机选择 {len(selected)} 张")
        
        for img_path in selected:
            img_name = img_path.stem
            img_ext = img_path.suffix
            
            print(f"\n  处理: {img_path.name}")
            
            # 读取图片
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"    错误: 无法读取图片")
                continue
            
            # 查找对应的标签文件
            label_path = os.path.join(train_label_dir, f"{img_name}.txt")
            
            # 绘制标注框
            image_with_boxes, box_count = draw_boxes(image, label_path)
            
            print(f"    图像尺寸: {image.shape[1]} x {image.shape[0]}")
            print(f"    标注框数量: {box_count}")
            
            # 保存结果
            output_name = f"{img_dir_name}_{img_path.name}"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            cv2.imwrite(output_path, image_with_boxes)
            print(f"    保存到: {output_path}")
    
    print(f"\n{'='*50}")
    print(f"完成！结果保存在: {OUTPUT_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    # 设置随机种子以便复现
    random.seed(42)
    visualize_dataset()