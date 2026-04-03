"""
测试 ObjectShift V2 功能

测试内容：
1. 基本平移功能 - 验证放大+复制粘贴是否正确
2. 边界检查 - 验证边缘物体的处理
3. 背景打乱 - 验证背景区域是否正确处理
4. shift_gt 记录 - 验证归一化值是否正确
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import random
from ultralytics.data.augment import ObjectShift


def create_test_image(h=480, w=640, n_objects=5):
    """
    创建测试用的双模态图像和 bboxes
    
    Returns:
        img: [H, W, 6] 双模态图像
        bboxes: [N, 4] xywh 归一化坐标
    """
    # 创建背景
    rgb = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
    ir = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
    
    # 添加一些纹理避免纯随机
    for i in range(0, h, 50):
        for j in range(0, w, 50):
            cv2.rectangle(rgb, (j, i), (j+30, i+30), (200, 150, 100), -1)
            cv2.rectangle(ir, (j, i), (j+30, i+30), (100, 150, 200), -1)
    
    bboxes = []
    
    # 在不同位置添加物体
    for i in range(n_objects):
        # 随机位置和大小
        cx = random.uniform(0.2, 0.8)
        cy = random.uniform(0.2, 0.8)
        bw = random.uniform(0.08, 0.15)
        bh = random.uniform(0.08, 0.15)
        
        # 转换为像素坐标
        cx_px = int(cx * w)
        cy_px = int(cy * h)
        bw_px = int(bw * w)
        bh_px = int(bh * h)
        
        x1 = max(0, cx_px - bw_px // 2)
        y1 = max(0, cy_px - bh_px // 2)
        x2 = min(w, cx_px + bw_px // 2)
        y2 = min(h, cy_px + bh_px // 2)
        
        # 在 RGB 上画彩色矩形
        color_rgb = (random.randint(0, 100), random.randint(100, 255), random.randint(150, 255))
        cv2.rectangle(rgb, (x1, y1), (x2, y2), color_rgb, -1)
        
        # 在 IR 上画灰色矩形（模拟红外）
        color_ir = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))
        cv2.rectangle(ir, (x1, y1), (x2, y2), color_ir, -1)
        
        # 添加边框便于识别
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.rectangle(ir, (x1, y1), (x2, y2), (0, 0, 0), 2)
        
        bboxes.append([cx, cy, bw, bh])
    
    # 合并双模态
    img = np.concatenate([rgb, ir], axis=2)
    
    return img, np.array(bboxes, dtype=np.float32)


def draw_bboxes(img, bboxes, color=(0, 255, 0), label="GT"):
    """在图像上绘制 bboxes"""
    h, w = img.shape[:2]
    for i, bbox in enumerate(bboxes):
        cx, cy, bw, bh = bbox
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label}{i}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def draw_shift_arrows(img, bboxes, shift_gt, shift_mask, color=(255, 0, 255)):
    """绘制 shift 箭头"""
    h, w = img.shape[:2]
    for i, (bbox, shift, mask) in enumerate(zip(bboxes, shift_gt, shift_mask)):
        if mask > 0.5:
            cx, cy, bw, bh = bbox
            cx_px = int(cx * w)
            cy_px = int(cy * h)
            
            # shift 是归一化值，转换为像素
            dx_px = int(shift[0] * w)
            dy_px = int(shift[1] * h)
            
            # 绘制箭头
            end_x = cx_px + dx_px * 5  # 放大5倍便于可视化
            end_y = cy_px + dy_px * 5
            cv2.arrowedLine(img, (cx_px, cy_px), (end_x, end_y), color, 2)
            cv2.putText(img, f"({shift[0]:.2f},{shift[1]:.2f})", (cx_px + 10, cy_px - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img


def test_basic_shift():
    """测试基本平移功能"""
    print("=" * 60)
    print("测试 1: 基本平移功能")
    print("=" * 60)
    
    random.seed(42)
    np.random.seed(42)
    
    # 创建测试数据
    img, bboxes = create_test_image(h=480, w=640, n_objects=5)
    
    # 创建 labels 字典
    labels = {
        'img': img.copy(),
        'bboxes': bboxes.copy()
    }
    
    # 应用 ObjectShift
    shift = ObjectShift(max_shift=0.1, shift_ratio=0.5, prob=1.0)
    shift.apply_image(labels)
    
    # 获取结果
    result_img = labels['img']
    shift_gt = labels['shift_gt']
    shift_mask = labels['shift_mask']
    
    # 分离 RGB 和 IR
    result_rgb = result_img[:, :, :3]
    result_ir = result_img[:, :, 3:6]
    orig_rgb = img[:, :, :3]
    orig_ir = img[:, :, 3:6]
    
    # 绘制结果
    vis_rgb = result_rgb.copy()
    vis_ir = result_ir.copy()
    
    # 绘制 GT bbox（黄色）
    vis_rgb = draw_bboxes(vis_rgb, bboxes, (0, 255, 255), "GT")
    vis_ir = draw_bboxes(vis_ir, bboxes, (0, 255, 255), "GT")
    
    # 绘制 shift 箭头
    vis_rgb = draw_shift_arrows(vis_rgb, bboxes, shift_gt, shift_mask)
    vis_ir = draw_shift_arrows(vis_ir, bboxes, shift_gt, shift_mask)
    
    # 保存结果
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 拼接原图和处理后的图
    compare_rgb = np.hstack([orig_rgb, vis_rgb])
    compare_ir = np.hstack([orig_ir, vis_ir])
    compare = np.vstack([compare_rgb, compare_ir])
    
    cv2.imwrite(os.path.join(output_dir, 'test_shift_v2_basic.png'), compare)
    
    print(f"物体数量: {len(bboxes)}")
    print(f"被平移的物体数量: {int(shift_mask.sum())}")
    print(f"shift_gt (归一化值):\n{shift_gt}")
    print(f"shift_mask:\n{shift_mask}")
    print(f"结果保存到: {os.path.join(output_dir, 'test_shift_v2_basic.png')}")
    print()


def test_boundary_cases():
    """测试边界情况"""
    print("=" * 60)
    print("测试 2: 边界情况处理")
    print("=" * 60)
    
    random.seed(123)
    np.random.seed(123)
    
    h, w = 480, 640
    
    # 创建背景
    rgb = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
    ir = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
    
    # 创建边缘物体
    bboxes = [
        [0.1, 0.5, 0.1, 0.1],   # 左边缘
        [0.9, 0.5, 0.1, 0.1],   # 右边缘
        [0.5, 0.1, 0.1, 0.1],   # 上边缘
        [0.5, 0.9, 0.1, 0.1],   # 下边缘
        [0.5, 0.5, 0.1, 0.1],   # 中心
    ]
    
    for i, bbox in enumerate(bboxes):
        cx, cy, bw, bh = bbox
        cx_px = int(cx * w)
        cy_px = int(cy * h)
        bw_px = int(bw * w)
        bh_px = int(bh * h)
        
        x1 = cx_px - bw_px // 2
        y1 = cy_px - bh_px // 2
        x2 = cx_px + bw_px // 2
        y2 = cy_px + bh_px // 2
        
        color = ((i * 50) % 256, (i * 100) % 256, (i * 150) % 256)
        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(ir, (x1, y1), (x2, y2), (200, 200, 200), -1)
        cv2.putText(rgb, str(i), (cx_px - 10, cy_px + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(ir, str(i), (cx_px - 10, cy_px + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    img = np.concatenate([rgb, ir], axis=2)
    
    # 应用 ObjectShift
    labels = {
        'img': img.copy(),
        'bboxes': np.array(bboxes, dtype=np.float32)
    }
    
    shift = ObjectShift(max_shift=0.1, shift_ratio=1.0, prob=1.0)
    shift.apply_image(labels)
    
    result_img = labels['img']
    shift_gt = labels['shift_gt']
    shift_mask = labels['shift_mask']
    
    # 可视化
    result_rgb = result_img[:, :, :3]
    result_ir = result_img[:, :, 3:6]
    orig_rgb = img[:, :, :3]
    
    vis_rgb = result_rgb.copy()
    vis_rgb = draw_bboxes(vis_rgb, np.array(bboxes), (0, 255, 255), "GT")
    vis_rgb = draw_shift_arrows(vis_rgb, np.array(bboxes), shift_gt, shift_mask)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    compare = np.hstack([orig_rgb, vis_rgb])
    cv2.imwrite(os.path.join(output_dir, 'test_shift_v2_boundary.png'), compare)
    
    print(f"shift_gt (归一化值):\n{shift_gt}")
    print(f"shift_mask:\n{shift_mask}")
    print(f"结果保存到: {os.path.join(output_dir, 'test_shift_v2_boundary.png')}")
    print()


def test_background_shuffle():
    """测试背景打乱功能"""
    print("=" * 60)
    print("测试 3: 背景打乱功能")
    print("=" * 60)
    
    random.seed(456)
    np.random.seed(456)
    
    h, w = 480, 640
    
    # 创建有纹理的背景
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    ir = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 创建网格纹理
    for i in range(0, h, 40):
        for j in range(0, w, 40):
            color_val = ((i + j) % 200) + 50
            rgb[i:i+40, j:j+40] = (color_val, color_val // 2, color_val)
            ir[i:i+40, j:j+40] = (color_val, color_val, color_val)
    
    # 添加几个物体
    bboxes = [
        [0.3, 0.3, 0.15, 0.15],
        [0.7, 0.7, 0.15, 0.15],
    ]
    
    for i, bbox in enumerate(bboxes):
        cx, cy, bw, bh = bbox
        cx_px = int(cx * w)
        cy_px = int(cy * h)
        bw_px = int(bw * w)
        bh_px = int(bh * h)
        
        x1 = cx_px - bw_px // 2
        y1 = cy_px - bh_px // 2
        x2 = cx_px + bw_px // 2
        y2 = cy_px + bh_px // 2
        
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 0, 255), -1)
        cv2.rectangle(ir, (x1, y1), (x2, y2), (0, 0, 255), -1)
    
    img = np.concatenate([rgb, ir], axis=2)
    
    # 应用 ObjectShift
    labels = {
        'img': img.copy(),
        'bboxes': np.array(bboxes, dtype=np.float32)
    }
    
    shift = ObjectShift(max_shift=0.1, shift_ratio=1.0, prob=1.0, bg_max_attempts=100)
    shift.apply_image(labels)
    
    result_img = labels['img']
    
    # 可视化
    result_rgb = result_img[:, :, :3]
    result_ir = result_img[:, :, 3:6]
    orig_rgb = img[:, :, :3]
    orig_ir = img[:, :, 3:6]
    
    # 对比 RGB 和 IR（检查是否只有一个模态被修改）
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    compare = np.vstack([
        np.hstack([orig_rgb, result_rgb]),
        np.hstack([orig_ir, result_ir])
    ])
    cv2.imwrite(os.path.join(output_dir, 'test_shift_v2_background.png'), compare)
    
    print(f"结果保存到: {os.path.join(output_dir, 'test_shift_v2_background.png')}")
    print("注意: 只有一个模态应该被修改（物体平移 + 背景打乱）")
    print()


def test_shift_gt_values():
    """测试 shift_gt 的值范围"""
    print("=" * 60)
    print("测试 4: shift_gt 值范围检查")
    print("=" * 60)
    
    random.seed(789)
    np.random.seed(789)
    
    all_shifts = []
    
    for trial in range(10):
        img, bboxes = create_test_image(h=480, w=640, n_objects=5)
        
        labels = {
            'img': img.copy(),
            'bboxes': bboxes.copy()
        }
        
        shift = ObjectShift(max_shift=0.1, shift_ratio=0.5, prob=1.0)
        shift.apply_image(labels)
        
        shift_gt = labels['shift_gt']
        shift_mask = labels['shift_mask']
        
        for i in range(len(bboxes)):
            if shift_mask[i] > 0.5:
                all_shifts.append(shift_gt[i])
    
    all_shifts = np.array(all_shifts)
    
    print(f"总平移次数: {len(all_shifts)}")
    print(f"shift_dx 范围: [{all_shifts[:, 0].min():.4f}, {all_shifts[:, 0].max():.4f}]")
    print(f"shift_dy 范围: [{all_shifts[:, 1].min():.4f}, {all_shifts[:, 1].max():.4f}]")
    print(f"预期范围: [-0.1, 0.1]")
    
    # 检查是否在预期范围内
    if np.all(np.abs(all_shifts) <= 0.1):
        print("✓ 所有 shift 值都在预期范围内")
    else:
        print("✗ 存在超出范围的 shift 值！")
    
    # 检查是否有非零值
    non_zero_count = np.sum(np.any(all_shifts != 0, axis=1))
    print(f"非零 shift 数量: {non_zero_count} / {len(all_shifts)}")
    print()


if __name__ == '__main__':
    test_basic_shift()
    test_boundary_cases()
    test_background_shuffle()
    test_shift_gt_values()
    
    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60)
