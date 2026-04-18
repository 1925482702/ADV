"""
检查原始标注是否对齐
"""
import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics.data import YOLODataset
from ultralytics.data.utils import check_det_dataset

# 加载数据集配置
data_config = "./Drone_RGB.yaml"  # 或 Drone_IR.yaml
data_dict = check_det_dataset(data_config)

# 创建数据集（不使用增强）
dataset = YOLODataset(
    img_path=data_dict['train'],
    imgsz=640,
    cache=False,
    augment=False,  # 关键：关闭增强
    hyp={'mosaic': 0.0, 'mixup': 0.0},
    rect=False,
    batch_size=1,
    stride=32,
    pad=0.5,
    prefix='check_align',
    classes=None,
    fraction=0.01  # 只检查1%的数据
)

print(f"共 {len(dataset)} 张图像")

# 检查前10张图像的标注对齐情况
for i in range(min(10, len(dataset))):
    label = dataset[i]
    img = label['img']
    instances = label['instances']
    
    if len(instances.segments) == 0:
        continue
    
    # 可视化
    img_viz = img.copy()
    h, w = img.shape[:2]
    
    # 绘制 segments
    for seg in instances.segments:
        # 转换为像素坐标
        seg_px = seg.copy()
        seg_px[:, 0] *= w
        seg_px[:, 1] *= h
        seg_px = seg_px.astype(np.int32)
        
        # 绘制多边形
        cv2.polylines(img_viz, [seg_px], True, (0, 255, 0), 2)
        
        # 绘制每个角点
        for pt in seg_px:
            cv2.circle(img_viz, tuple(pt), 3, (0, 0, 255), -1)
    
    # 保存结果
    save_path = f"./check_alignment_{i}.jpg"
    cv2.imwrite(save_path, img_viz)
    print(f"保存: {save_path}")

print("\n请检查生成的图像，观察标注框是否紧贴物体边缘")