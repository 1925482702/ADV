#!/usr/bin/env python3
"""
测试 cv2.merge 的行为
"""

import cv2
import numpy as np

# 创建一个单通道图像
gray = np.ones((100, 100), dtype=np.uint8) * 128

# 创建一个3通道图像
bgr = np.ones((100, 100, 3), dtype=np.uint8) * 255

print(f"gray shape: {gray.shape}")
print(f"bgr shape: {bgr.shape}")

# 测试 cv2.merge
try:
    # 尝试合并单通道图像
    result1 = cv2.merge((gray, gray, gray))
    print(f"merge (gray, gray, gray) shape: {result1.shape}")
except Exception as e:
    print(f"merge (gray, gray, gray) failed: {e}")

try:
    # 尝试合并单通道和3通道图像
    result2 = cv2.merge((gray, bgr))
    print(f"merge (gray, bgr) shape: {result2.shape}")
except Exception as e:
    print(f"merge (gray, bgr) failed: {e}")

# 测试 cv2.split
try:
    channels = cv2.split(bgr)
    print(f"split bgr: {len(channels)} channels")
    for i, ch in enumerate(channels):
        print(f"  channel {i} shape: {ch.shape}")
except Exception as e:
    print(f"split bgr failed: {e}")

# 测试 np.concatenate
try:
    result3 = np.concatenate((gray[:, :, np.newaxis], bgr), axis=2)
    print(f"concatenate (gray[:, :, np.newaxis], bgr) shape: {result3.shape}")
except Exception as e:
    print(f"concatenate (gray[:, :, np.newaxis], bgr) failed: {e}")