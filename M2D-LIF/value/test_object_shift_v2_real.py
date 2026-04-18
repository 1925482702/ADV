"""
使用真实 FLIR 数据集测试 ObjectShift V2 功能

测试内容：
1. 读取真实的 RGB 和 IR 图像对
2. 读取对应的 YOLO 格式标签
3. 应用 ObjectShift V2 增强
4. 可视化并保存结果
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import random
import glob
from ultralytics.data.augment import ObjectShift


def read_flir_image_pair(rgb_path, ir_path, label_path, img_size=640):
    """
    读取 FLIR 数据集的图像对和标签
    
    Args:
        rgb_path: RGB 图像路径
        ir_path: IR 图像路径
        label_path: 标签文件路径
        img_size: 目标图像大小
    
    Returns:
        img: [H, W, 6] 双模态图像
        bboxes: [N, 4] xywh 归一化坐标
    """
    # 读取图像
    img_rgb = cv2.imread(rgb_path)
    img_ir = cv2.imread(ir_path)
    
    if img_rgb is None or img_ir is None:
        return None, None
    
    # 调整大小
    img_rgb = cv2.resize(img_rgb, (img_size, img_size))
    img_ir = cv2.resize(img_ir, (img_size, img_size))
    
    # 读取标签文件 (YOLO 格式: class x_center y_center width height)
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, xc, yc, w, h = map(float, parts[:5])
                    bboxes.append([xc, yc, w, h])
    
    # 合并双模态图像 [H, W, 6]
    img = np.concatenate([img_rgb, img_ir], axis=2)
    
    return img, np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32)


def draw_bboxes_and_shift(img, bboxes, shift_gt=None, shift_mask=None, color=(0, 255, 0)):
    """在图像上绘制 bboxes 和 shift 箭头"""
    h, w = img.shape[:2]
    
    for i, bbox in enumerate(bboxes):
        cx, cy, bw, bh = bbox
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        
        # 绘制 bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制 shift 箭头（如果有）
        if shift_gt is not None and shift_mask is not None and shift_mask[i] > 0.5:
            cx_px = int(cx * w)
            cy_px = int(cy * h)
            dx_px = int(shift_gt[i, 0] * w * 3)  # 放大3倍便于可视化
            dy_px = int(shift_gt[i, 1] * h * 3)
            
            end_x = cx_px + dx_px
            end_y = cy_px + dy_px
            cv2.arrowedLine(img, (cx_px, cy_px), (end_x, end_y), (255, 0, 255), 2)
            
            # 显示 shift 值
            shift_text = f"({shift_gt[i, 0]:.2f}, {shift_gt[i, 1]:.2f})"
            cv2.putText(img, shift_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    return img


def test_with_flir_dataset(n_samples=5, output_dir=None):
    """
    使用 FLIR 数据集测试 ObjectShift V2
    
    Args:
        n_samples: 测试样本数量
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # FLIR 数据集路径
    data_path = '/mnt/home/pyq_code/ADV/datasets/FLIR_yolo'
    rgb_pattern = os.path.join(data_path, 'images/train/*.jpg')
    ir_dir = os.path.join(data_path, 'images_ir/train')
    label_dir = os.path.join(data_path, 'labels/train')
    
    rgb_files = sorted(glob.glob(rgb_pattern))
    
    if not rgb_files:
        print(f"未找到 FLIR 数据集图像: {rgb_pattern}")
        return
    
    print(f"找到 {len(rgb_files)} 张图像")
    
    # 随机选择样本
    random.seed(42)
    selected_files = random.sample(rgb_files, min(n_samples, len(rgb_files)))
    
    # 创建 ObjectShift 实例
    shift = ObjectShift(max_shift=0.1, shift_ratio=0.5, prob=1.0, bg_max_attempts=100)
    
    for idx, rgb_path in enumerate(selected_files):
        basename = os.path.basename(rgb_path)
        name_no_ext = os.path.splitext(basename)[0]
        
        # 构建对应路径
        ir_path = os.path.join(ir_dir, basename)
        if not os.path.exists(ir_path):
            # 尝试 PNG 格式
            ir_path = os.path.join(ir_dir, name_no_ext + '.png')
        label_path = os.path.join(label_dir, name_no_ext + '.txt')
        
        if not os.path.exists(ir_path):
            print(f"跳过 {basename}: IR 图像不存在")
            continue
        
        print(f"\n处理 [{idx+1}/{len(selected_files)}]: {basename}")
        
        # 读取图像和标签
        img, bboxes = read_flir_image_pair(rgb_path, ir_path, label_path)
        
        if img is None:
            print(f"  跳过: 无法读取图像")
            continue
        
        if len(bboxes) == 0:
            print(f"  跳过: 没有标注")
            continue
        
        print(f"  图像尺寸: {img.shape}, 物体数量: {len(bboxes)}")
        
        # 保存原始图像（用于对比）
        orig_rgb = img[:, :, :3].copy()
        orig_ir = img[:, :, 3:6].copy()
        
        # 应用 ObjectShift V2
        labels = {
            'img': img.copy(),
            'bboxes': bboxes.copy()
        }
        
        shift.apply_image(labels)
        
        result_img = labels['img']
        shift_gt = labels['shift_gt']
        shift_mask = labels['shift_mask']
        
        # 分离 RGB 和 IR
        result_rgb = result_img[:, :, :3]
        result_ir = result_img[:, :, 3:6]
        
        # 绘制结果
        vis_rgb = result_rgb.copy()
        vis_ir = result_ir.copy()
        
        # 绘制 GT bbox 和 shift 箭头
        vis_rgb = draw_bboxes_and_shift(vis_rgb, bboxes, shift_gt, shift_mask, (0, 255, 0))
        vis_ir = draw_bboxes_and_shift(vis_ir, bboxes, shift_gt, shift_mask, (0, 255, 0))
        
        # 创建对比图
        # 上排: 原始 RGB | 原始 IR
        # 下排: 处理后 RGB | 处理后 IR
        top_row = np.hstack([orig_rgb, orig_ir])
        bottom_row = np.hstack([vis_rgb, vis_ir])
        compare = np.vstack([top_row, bottom_row])
        
        # 添加标签
        h, w = orig_rgb.shape[:2]
        cv2.putText(compare, 'Original RGB', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(compare, 'Original IR', (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(compare, 'Shifted RGB (with GT bbox)', (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(compare, 'Shifted IR', (w + 10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 添加统计信息
        n_shifted = int(shift_mask.sum())
        info_text = f'Shifted objects: {n_shifted}/{len(bboxes)}'
        cv2.putText(compare, info_text, (10, h * 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 保存结果
        save_path = os.path.join(output_dir, f'flir_shift_v2_{idx+1}.png')
        cv2.imwrite(save_path, compare)
        
        print(f"  被平移物体: {n_shifted}/{len(bboxes)}")
        print(f"  shift_gt (归一化值):")
        for i in range(len(bboxes)):
            if shift_mask[i] > 0.5:
                print(f"    物体 {i}: dx={shift_gt[i, 0]:.4f}, dy={shift_gt[i, 1]:.4f}")
        print(f"  结果保存到: {save_path}")


def test_boundary_with_flir(output_dir=None):
    """
    测试边界情况：选择靠近边缘的物体
    
    Args:
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    data_path = '/mnt/home/pyq_code/ADV/datasets/FLIR_yolo'
    rgb_pattern = os.path.join(data_path, 'images/train/*.jpg')
    ir_dir = os.path.join(data_path, 'images_ir/train')
    label_dir = os.path.join(data_path, 'labels/train')
    
    rgb_files = sorted(glob.glob(rgb_pattern))
    
    if not rgb_files:
        print(f"未找到 FLIR 数据集图像")
        return
    
    # 查找含有边缘物体的图像
    edge_images = []
    
    for rgb_path in rgb_files[:100]:  # 检查前100张
        basename = os.path.basename(rgb_path)
        name_no_ext = os.path.splitext(basename)[0]
        label_path = os.path.join(label_dir, name_no_ext + '.txt')
        
        if not os.path.exists(label_path):
            continue
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, xc, yc, w, h = map(float, parts[:5])
                    # 检查是否靠近边缘
                    if xc < 0.15 or xc > 0.85 or yc < 0.15 or yc > 0.85:
                        edge_images.append(rgb_path)
                        break
        
        if len(edge_images) >= 3:
            break
    
    if not edge_images:
        print("未找到含有边缘物体的图像")
        return
    
    print(f"找到 {len(edge_images)} 张含有边缘物体的图像")
    
    # 测试这些图像
    shift = ObjectShift(max_shift=0.1, shift_ratio=1.0, prob=1.0)  # 100%概率，所有物体
    
    for idx, rgb_path in enumerate(edge_images[:3]):
        basename = os.path.basename(rgb_path)
        name_no_ext = os.path.splitext(basename)[0]
        ir_path = os.path.join(ir_dir, basename)
        if not os.path.exists(ir_path):
            ir_path = os.path.join(ir_dir, name_no_ext + '.png')
        label_path = os.path.join(label_dir, name_no_ext + '.txt')
        
        img, bboxes = read_flir_image_pair(rgb_path, ir_path, label_path)
        
        if img is None or len(bboxes) == 0:
            continue
        
        print(f"\n边界测试 [{idx+1}]: {basename}, 物体数: {len(bboxes)}")
        
        orig_rgb = img[:, :, :3].copy()
        orig_ir = img[:, :, 3:6].copy()
        
        labels = {
            'img': img.copy(),
            'bboxes': bboxes.copy()
        }
        
        shift.apply_image(labels)
        
        result_rgb = labels['img'][:, :, :3]
        result_ir = labels['img'][:, :, 3:6]
        shift_gt = labels['shift_gt']
        shift_mask = labels['shift_mask']
        
        vis_rgb = draw_bboxes_and_shift(result_rgb.copy(), bboxes, shift_gt, shift_mask, (0, 255, 0))
        vis_ir = draw_bboxes_and_shift(result_ir.copy(), bboxes, shift_gt, shift_mask, (0, 255, 0))
        
        compare = np.vstack([
            np.hstack([orig_rgb, orig_ir]),
            np.hstack([vis_rgb, vis_ir])
        ])
        
        h, w = orig_rgb.shape[:2]
        cv2.putText(compare, 'Original RGB', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(compare, 'Original IR', (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(compare, 'Shifted RGB (Edge Test)', (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(compare, 'Shifted IR', (w + 10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        save_path = os.path.join(output_dir, f'flir_shift_v2_edge_{idx+1}.png')
        cv2.imwrite(save_path, compare)
        
        print(f"  shift_gt:")
        for i in range(len(bboxes)):
            print(f"    物体 {i}: dx={shift_gt[i, 0]:.4f}, dy={shift_gt[i, 1]:.4f}, mask={shift_mask[i]:.0f}")
        print(f"  结果保存到: {save_path}")


if __name__ == '__main__':
    print("=" * 60)
    print("ObjectShift V2 真实数据测试")
    print("=" * 60)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 测试基本功能
    print("\n[1] 测试基本功能...")
    test_with_flir_dataset(n_samples=5, output_dir=output_dir)
    
    # 测试边界情况
    print("\n[2] 测试边界情况...")
    test_boundary_with_flir(output_dir=output_dir)
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 60)
