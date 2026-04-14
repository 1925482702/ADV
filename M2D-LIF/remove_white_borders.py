#!/usr/bin/env python3
"""
批量去除 DroneVehicle 数据集中所有图像的白边
"""

import os
import cv2
import numpy as np
from tqdm import tqdm


def remove_white_border(image, threshold=250):
    """去除图像四周的白色边框"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    non_white = np.where(gray < threshold)
    
    if len(non_white[0]) == 0:
        return image, False
    
    y_min, y_max = non_white[0].min(), non_white[0].max()
    x_min, x_max = non_white[1].min(), non_white[1].max()
    
    if y_min == 0 and y_max == h-1 and x_min == 0 and x_max == w-1:
        return image, False
    
    return image[y_min:y_max+1, x_min:x_max+1], True


def main():
    dataset_root = "/mnt/home/pyq_code/ADV/datasets/DroneVehicle_yolo"
    
    # 找到所有图像目录
    img_dirs = []
    for root, dirs, files in os.walk(dataset_root):
        if 'images' in root or 'images_ir' in root:
            img_dirs.append(root)
    
    print(f"Found {len(img_dirs)} image directories")
    
    total, cropped = 0, 0
    
    for img_dir in img_dirs:
        images = [f for f in os.listdir(img_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"\nProcessing: {img_dir} ({len(images)} images)")
        
        for img_name in tqdm(images):
            path = os.path.join(img_dir, img_name)
            image = cv2.imread(path)
            if image is None:
                continue
            
            total += 1
            result, was_cropped = remove_white_border(image)
            
            if was_cropped:
                cropped += 1
                cv2.imwrite(path, result)
    
    print(f"\nDone! Cropped {cropped}/{total} images")


if __name__ == "__main__":
    main()