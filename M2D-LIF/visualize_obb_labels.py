#!/usr/bin/env python3
"""
可视化 OBB (旋转目标检测) 标注
从 DroneVehicle 数据集中随机选择图片并绘制旋转边界框
"""

import os
import random
import cv2
import numpy as np
from pathlib import Path


def remove_white_border(image, white_threshold=250):
    """
    去除图像四周的白色边框
    white_threshold: RGB值大于此阈值视为白色
    返回: (裁剪后的图像, (x_offset, y_offset)) 偏移量用于调整label坐标
    """
    h, w = image.shape[:2]
    
    # 创建白色像素掩码（三个通道都接近255）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white_mask = gray >= white_threshold
    
    # 找到非白色像素的位置
    non_white_positions = np.where(~white_mask)
    
    if len(non_white_positions[0]) == 0:
        # 整张图都是白色，不裁剪
        print("  Warning: Entire image is white, no cropping")
        return image, (0, 0)
    
    # 计算非白色区域的边界
    y_min, y_max = non_white_positions[0].min(), non_white_positions[0].max()
    x_min, x_max = non_white_positions[1].min(), non_white_positions[1].max()
    
    # 计算裁剪掉的边框大小
    top_crop = y_min
    bottom_crop = h - 1 - y_max
    left_crop = x_min
    right_crop = w - 1 - x_max
    
    # 只有当边框存在时才裁剪
    if top_crop > 0 or bottom_crop > 0 or left_crop > 0 or right_crop > 0:
        print(f"  White border detected: top={top_crop}, bottom={bottom_crop}, left={left_crop}, right={right_crop}")
        cropped = image[y_min:y_max+1, x_min:x_max+1]
        return cropped, (x_min, y_min)
    
    return image, (0, 0)


def read_obb_label(label_path):
    """
    读取 OBB label 文件
    格式: class x1 y1 x2 y2 x3 y3 x4 y4 (归一化坐标)
    返回: list of (class_id, points) where points is (4, 2) array
    """
    boxes = []
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:
                class_id = int(parts[0])
                # 4个角点的归一化坐标
                coords = [float(x) for x in parts[1:9]]
                points = np.array([
                    [coords[0], coords[1]],  # (x1, y1)
                    [coords[2], coords[3]],  # (x2, y2)
                    [coords[4], coords[5]],  # (x3, y3)
                    [coords[6], coords[7]],  # (x4, y4)
                ])
                boxes.append((class_id, points))
    return boxes


def draw_obb_boxes(image, boxes, color=(0, 0, 255), thickness=2):
    """
    在图片上绘制旋转边界框
    boxes: list of (class_id, points), points 是归一化坐标
    """
    h, w = image.shape[:2]
    
    for class_id, points in boxes:
        # 将归一化坐标转换为像素坐标
        points_pixel = points.copy()
        points_pixel[:, 0] *= w  # x
        points_pixel[:, 1] *= h  # y
        points_pixel = points_pixel.astype(np.int32)
        
        # 绘制四边形（闭合）
        pts = points_pixel.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
        
        # 在框的左上角标注类别
        label_pos = (int(points_pixel[0, 0]), int(points_pixel[0, 1]) - 5)
        cv2.putText(image, str(class_id), label_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return image


def main():
    # 数据集路径
    dataset_root = "/mnt/home/pyq_code/ADV/datasets/DroneVehicle_yolo"
    
    # 图像目录
    rgb_train_dir = os.path.join(dataset_root, "images", "train")
    ir_train_dir = os.path.join(dataset_root, "images_ir", "train")
    
    # 标签目录
    label_dir = os.path.join(dataset_root, "labels", "train")
    
    # 输出目录
    output_dir = "./visualized_labels"
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有图片
    all_images = []
    
    # RGB 图片
    if os.path.exists(rgb_train_dir):
        rgb_images = [f for f in os.listdir(rgb_train_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        for img in rgb_images:
            all_images.append(('rgb', os.path.join(rgb_train_dir, img)))
    
    # IR 图片
    if os.path.exists(ir_train_dir):
        ir_images = [f for f in os.listdir(ir_train_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        for img in ir_images:
            all_images.append(('ir', os.path.join(ir_train_dir, img)))
    
    print(f"Total images found: {len(all_images)}")
    
    if len(all_images) == 0:
        print("No images found! Please check the dataset path.")
        return
    
    # 随机选择5张图片
    num_samples = min(5, len(all_images))
    selected = random.sample(all_images, num_samples)
    
    print(f"\nRandomly selected {num_samples} images:")
    
    for i, (img_type, img_path) in enumerate(selected, 1):
        print(f"\n[{i}] Type: {img_type}, Image: {os.path.basename(img_path)}")
        
        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            print(f"  Failed to read image: {img_path}")
            continue
        
        print(f"  Original size: {image.shape[1]} x {image.shape[0]}")
        
        # 去除白边
        image_cropped, (x_off, y_off) = remove_white_border(image)
        print(f"  Cropped size: {image_cropped.shape[1]} x {image_cropped.shape[0]}")
        
        # 找到对应的 label 文件
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{img_name}.txt")
        
        print(f"  Label file: {label_path}")
        
        # 读取并打印 label 内容
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                content = f.read()
            print(f"  Label content:\n{content}")
        else:
            print(f"  Label file not found!")
        
        # 读取 OBB boxes
        boxes = read_obb_label(label_path)
        print(f"  Number of boxes: {len(boxes)}")
        
        # 绘制（使用裁剪后的图片）
        image_vis = image_cropped.copy()
        draw_obb_boxes(image_vis, boxes, color=(0, 0, 255), thickness=2)
        
        # 在图片上显示信息
        orig_h, orig_w = image.shape[:2]
        crop_h, crop_w = image_cropped.shape[:2]
        info_text = f"{img_type.upper()} - {img_name} - {len(boxes)} boxes"
        size_text = f"Original: {orig_w}x{orig_h} -> Cropped: {crop_w}x{crop_h}"
        cv2.putText(image_vis, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image_vis, size_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 保存
        output_path = os.path.join(output_dir, f"vis_{i}_{img_type}_{img_name}.jpg")
        cv2.imwrite(output_path, image_vis)
        print(f"  Saved to: {output_path}")
    
    print(f"\nAll visualized images saved to: {output_dir}")


if __name__ == "__main__":
    main()