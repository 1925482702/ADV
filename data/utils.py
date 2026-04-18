"""
数据处理辅助函数
"""

import hashlib
import os
import numpy as np
from pathlib import Path


def get_hash(paths):
    """计算文件路径的哈希值"""
    h = hashlib.sha256()
    for path in sorted(paths):
        p = str(path).encode()
        h.update(p)
    return h.hexdigest()


def img2label_paths(img_paths):
    """根据图像路径生成标签路径"""
    labels = []
    for img_path in img_paths:
        # 将 images/train 替换为 labels/train
        label_path = img_path.replace('/images/', '/labels/')
        # 将 .jpg/.png 等替换为 .txt
        label_path = os.path.splitext(label_path)[0] + '.txt'
        labels.append(label_path)
    return labels


def verify_image_label(im_file, lb_file, prefix, use_keypoints, num_classes, nkpt, ndim):
    """
    验证图像和标签文件
    
    Returns:
        tuple: (im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg)
    """
    import cv2
    
    # 读取图像
    im = cv2.imread(im_file)
    if im is None:
        nm_f = 1
        nf_f, ne_f, nc_f = 0, 0, 0
        msg = f'{prefix}WARNING ⚠️ image read failure: {im_file}'
        return None, None, None, None, None, nm_f, nf_f, ne_f, nc_f, msg
    
    h, w = im.shape[:2]
    
    # 读取标签
    try:
        with open(lb_file) as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x) >= 6]
    except FileNotFoundError:
        nf_f = 1
        nm_f, ne_f, nc_f = 0, 0, 0
        msg = f'{prefix}WARNING ⚠️ label file missing: {lb_file}'
        return im_file, None, (h, w), None, None, nm_f, nf_f, ne_f, nc_f, msg
    
    if len(lb) == 0:
        ne_f = 1
        nm_f, nf_f, nc_f = 0, 1, 0
        msg = f'{prefix}WARNING ⚠️ no labels found in {lb_file}'
        return im_file, None, (h, w), None, None, nm_f, nf_f, ne_f, nc_f, msg
    
    # 解析标签
    lb = np.array(lb, dtype=np.float32)
    nc_f = 0
    nm_f, ne_f, nf_f = 0, 0, 1
    
    # 检查类别
    classes = lb[:, 0].astype(int)
    if classes.max() >= num_classes:
        nc_f = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: Label class {classes.max()} exceeds dataset class count {num_classes}'
        return im_file, None, (h, w), None, None, nm_f, nf_f, ne_f, nc_f, msg
    
    # 检查坐标范围
    if (lb[:, 1:] < 0).any() or (lb[:, 1:] > 1).any():
        nc_f = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: non-normalized or out of bounds coordinates'
        return im_file, None, (h, w), None, None, nm_f, nf_f, ne_f, nc_f, msg
    
    # 检查重复标签
    if len(lb) > len(np.unique(lb[:, 1:], axis=0)):
        msg = f'{prefix}WARNING ⚠️ {im_file}: duplicate labels removed'
    else:
        msg = None
    
    # 转换坐标格式
    segments = None
    keypoint = None
    
    # 返回标签
    return im_file, lb, (h, w), segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg


def verify_image(path, prefix):
    """
    验证图像文件
    
    Returns:
        tuple: (path, nf_f, nc_f, msg)
    """
    import cv2
    
    try:
        im = cv2.imread(path)
        if im is None:
            return path, 0, 1, f'{prefix}WARNING ⚠️ image read failure: {path}'
        
        h, w = im.shape[:2]
        if h < 1 or w < 1:
            return path, 0, 1, f'{prefix}WARNING ⚠️ invalid image size: {path}'
        
        return path, 1, 0, None
    
    except Exception as e:
        return path, 0, 1, f'{prefix}WARNING ⚠️ {path}: {str(e)}'