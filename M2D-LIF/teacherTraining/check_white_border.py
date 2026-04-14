"""
检查白边裁剪是否合适
"""
import cv2
import numpy as np
from pathlib import Path

def detect_white_border(img, threshold=250):
    """
    检测图像的白边
    返回: (top, bottom, left, right) 各边的白边像素数
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 从上往下检查
    top = 0
    for i in range(gray.shape[0]):
        if gray[i, :].mean() < threshold:
            top = i
            break

    # 从下往上检查
    bottom = 0
    for i in range(gray.shape[0]-1, -1, -1):
        if gray[i, :].mean() < threshold:
            bottom = gray.shape[0] - 1 - i
            break

    # 从左往右检查
    left = 0
    for i in range(gray.shape[1]):
        if gray[:, i].mean() < threshold:
            left = i
            break

    # 从右往左检查
    right = 0
    for i in range(gray.shape[1]-1, -1, -1):
        if gray[:, i].mean() < threshold:
            right = gray.shape[1] - 1 - i
            break

    return top, bottom, left, right

# 检查数据集
data_config = "./Drone_RGB.yaml"
with open(data_config, 'r') as f:
    config_lines = f.readlines()
    for line in config_lines:
        if line.startswith('path:'):
            dataset_path = line.split(':')[1].strip()
            break

image_dir = Path(dataset_path) / "images/val"
image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

print(f"找到 {len(image_files)} 张图像")
print(f"检查前10张图像的白边情况...\n")

for i, img_path in enumerate(image_files[:10]):
    img = cv2.imread(str(img_path))
    top, bottom, left, right = detect_white_border(img)

    print(f"{img_path.name}:")
    print(f"  上: {top}px, 下: {bottom}px, 左: {left}px, 右: {right}px")
    print(f"  原始尺寸: {img.shape[:2]}")

    # 如果有白边，可视化
    if max(top, bottom, left, right) > 5:
        img_viz = img.copy()
        # 标记白边区域
        if top > 0:
            cv2.rectangle(img_viz, (0, 0), (img.shape[1], top), (0, 0, 255), 2)
        if bottom > 0:
            cv2.rectangle(img_viz, (0, img.shape[0]-bottom), (img.shape[1], img.shape[0]), (0, 0, 255), 2)
        if left > 0:
            cv2.rectangle(img_viz, (0, 0), (left, img.shape[0]), (0, 0, 255), 2)
        if right > 0:
            cv2.rectangle(img_viz, (img.shape[1]-right, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 2)

        save_path = f"./check_white_border_{i}.jpg"
        cv2.imwrite(save_path, img_viz)
        print(f"  保存可视化: {save_path}")
    print()

print("\n如果白边数值较大，说明裁剪可能不够或过度")