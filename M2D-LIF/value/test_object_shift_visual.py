"""
测试 ObjectShift 数据增强效果

从 FLIR_yolo 数据集随机抽取图片，应用 ObjectShift 增强，可视化结果
"""

import os
import sys
import random
import cv2
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics.data.augment import ObjectShift


def draw_bbox(img, bbox, color=(0, 255, 0), thickness=2, label=""):
    """在图像上绘制边界框

    Args:
        img: 图像 [H, W, 3]
        bbox: 边界框 [x1, y1, x2, y2] 归一化坐标
        color: 颜色
        thickness: 线宽
        label: 标签文本
    """
    h, w = img.shape[:2]
    x1 = int(bbox[0] * w)
    y1 = int(bbox[1] * h)
    x2 = int(bbox[2] * w)
    y2 = int(bbox[3] * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(img, (x1, y1 - text_h - baseline - 5),
                      (x1 + text_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - baseline - 2),
                    font, font_scale, (255, 255, 255), font_thickness)


def load_flir_image(rgb_path, ir_path, h, w):
    """加载 FLIR 数据集的双模态图像

    FLIR 数据集的 RGB 和 IR 图像是分开存储的
    """
    # 读取 RGB 图像
    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        raise ValueError(f"无法读取 RGB 图像: {rgb_path}")

    # 读取 IR 图像
    ir = cv2.imread(str(ir_path))
    if ir is None:
        raise ValueError(f"无法读取 IR 图像: {ir_path}")

    # 调整大小
    rgb = cv2.resize(rgb, (w, h))
    ir = cv2.resize(ir, (w, h))

    # 拼接成 6 通道（RGB 在前，IR 在后）
    img_6ch = np.concatenate([rgb, ir], axis=2)

    return img_6ch


def load_labels(label_path):
    """加载 YOLO 格式的标签文件

    Returns:
        bboxes: 边界框列表 [x1, y1, x2, y2] 归一化坐标
        classes: 类别列表
    """
    bboxes = []
    classes = []

    if not os.path.exists(label_path):
        return bboxes, classes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])

                # 转换为 xyxy 格式
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                bboxes.append([x1, y1, x2, y2])
                classes.append(cls)

    return bboxes, classes


def visualize_shift(img_6ch, bboxes, shift_gt, shift_mask, shift_modality, save_path):
    """可视化 shift 效果

    Args:
        img_6ch: 6 通道图像 [H, W, 6]
        bboxes: 边界框列表
        shift_gt: shift GT
        shift_mask: shift mask
        shift_modality: 被平移的模态 ('rgb' 或 'ir')
        save_path: 保存路径
    """
    h, w = img_6ch.shape[:2]

    # 分离 RGB 和 IR
    rgb = img_6ch[:, :, :3].copy()
    ir = img_6ch[:, :, 3:6].copy()

    # 绘制原始边界框（绿色）
    for i, bbox in enumerate(bboxes):
        draw_bbox(rgb, bbox, color=(0, 255, 0), thickness=2)
        draw_bbox(ir, bbox, color=(0, 255, 0), thickness=2)

        # 如果有 shift，绘制平移后的边界框（红色）
        if shift_mask[i] > 0.5:
            dx, dy = shift_gt[i]
            x1, y1, x2, y2 = bbox

            # shift_gt 是像素值，需要归一化到 [0, 1]
            dx_norm = dx / w
            dy_norm = dy / h

            # 计算平移后的边界框
            new_x1 = np.clip(x1 + dx_norm, 0, 1)
            new_y1 = np.clip(y1 + dy_norm, 0, 1)
            new_x2 = np.clip(x2 + dx_norm, 0, 1)
            new_y2 = np.clip(y2 + dy_norm, 0, 1)

            # 绘制平移后的框（红色）
            draw_bbox(rgb, [new_x1, new_y1, new_x2, new_y2], color=(0, 0, 255), thickness=2)
            draw_bbox(ir, [new_x1, new_y1, new_x2, new_y2], color=(0, 0, 255), thickness=2)

            # 绘制箭头表示 shift 方向
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            new_center_x = center_x + dx_norm
            new_center_y = center_y + dy_norm

            cv2.arrowedLine(rgb, (int(center_x * w), int(center_y * h)),
                           (int(new_center_x * w), int(new_center_y * h)),
                           (255, 0, 0), 3, tipLength=0.3)
            cv2.arrowedLine(ir, (int(center_x * w), int(center_y * h)),
                           (int(new_center_x * w), int(new_center_y * h)),
                           (255, 0, 0), 3, tipLength=0.3)

    # 拼接 RGB 和 IR
    result = np.hstack([rgb, ir])

    # 添加文字说明
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, f"RGB (Shifted)" if shift_modality == 'rgb' else "RGB", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(result, f"IR (Shifted)" if shift_modality == 'ir' else "IR", (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(result, "Green: Original, Red: Shifted, Blue Arrow: Shift Direction",
                (10, h - 10), font, 0.6, (255, 255, 255), 1)

    # 保存
    cv2.imwrite(str(save_path), result)
    print(f"保存结果到: {save_path}")


def main():
    # 配置
    data_root = Path("/mnt/home/pyq_code/ADV/datasets/FLIR_yolo")
    rgb_dir = data_root / "images/train"
    ir_dir = data_root / "images_ir/train"
    label_dir = data_root / "labels/train"
    output_dir = Path("/mnt/home/pyq_code/ADV/M2D-LIF/value/shift_test_results")
    output_dir.mkdir(exist_ok=True)

    # ObjectShift 参数
    min_shift_pixels = 8
    max_shift_pixels = 40
    shift_ratio = 0.5  # 50% 的物体被平移
    prob = 1.0  # 100% 应用增强

    # 获取所有 RGB 图片
    image_files = sorted(list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg")))
    if len(image_files) == 0:
        print("没有找到图片文件")
        return

    print(f"找到 {len(image_files)} 张 RGB 图片")

    # 随机选择 5 张图片进行测试
    n_test = min(5, len(image_files))
    selected_images = random.sample(image_files, n_test)

    # 创建 ObjectShift
    object_shift = ObjectShift(
        min_shift_pixels=min_shift_pixels,
        max_shift_pixels=max_shift_pixels,
        shift_ratio=shift_ratio,
        prob=prob
    )

    img_size = 640

    for i, rgb_path in enumerate(selected_images):
        print(f"\n========== 测试图片 {i+1}/{n_test}: {rgb_path.name} ==========")

        # 对应的 IR 图像路径
        ir_path = ir_dir / rgb_path.name
        if not ir_path.exists():
            print(f"对应的 IR 图像不存在: {ir_path}")
            continue

        # 对应的标签文件
        label_path = label_dir / (rgb_path.stem + ".txt")

        # 加载图片
        try:
            img_6ch = load_flir_image(rgb_path, ir_path, img_size, img_size)
        except Exception as e:
            print(f"加载图片失败: {e}")
            continue

        # 加载标签
        bboxes, classes = load_labels(label_path)
        print(f"找到 {len(bboxes)} 个物体")

        if len(bboxes) == 0:
            print("没有物体，跳过")
            continue

        # 准备 labels 字典
        labels = {
            'img': img_6ch.copy(),
            'bboxes': np.array(bboxes, dtype=np.float32),
            'cls': np.array(classes, dtype=np.float32)
        }

        # 应用 ObjectShift
        print(f"应用 ObjectShift (min_shift={min_shift_pixels}px, max_shift={max_shift_pixels}px)")
        object_shift.apply_image(labels)

        # 获取结果
        shifted_img = labels['img']
        shift_gt = labels.get('shift_gt', np.zeros((len(bboxes), 2)))
        shift_mask = labels.get('shift_mask', np.zeros(len(bboxes)))

        # 统计
        n_shifted = int(shift_mask.sum())
        print(f"被平移的物体数: {n_shifted}/{len(bboxes)}")

        for j, (bbox, mask, gt) in enumerate(zip(bboxes, shift_mask, shift_gt)):
            if mask > 0.5:
                dx_px = gt[0]  # 已经是像素值
                dy_px = gt[1]  # 已经是像素值
                print(f"  物体 {j}: shift=({dx_px:.1f}px, {dy_px:.1f}px)")

        # 获取被平移的模态
        shift_modality = labels.get('shift_modality', 'unknown')
        print(f"被平移的模态: {shift_modality}")

        # 可视化
        output_path = output_dir / f"{rgb_path.stem}_shifted.png"
        visualize_shift(shifted_img, bboxes, shift_gt, shift_mask, shift_modality, output_path)

    print(f"\n测试完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()