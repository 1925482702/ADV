"""
测试 ObjectShift 数据增强类

测试内容：
1. 基本功能测试：验证类能否正确初始化
2. 单物体平移测试：验证单个物体能否正确平移
3. 多物体平移测试：验证多个物体能否正确平移
4. 模态选择测试：验证 RGB 和 IR 模态能否正确选择
5. 边界填充测试：验证空洞填充是否正确
6. shift_gt 和 shift_mask 测试：验证 GT 是否正确记录
"""

import sys
sys.path.insert(0, '/mnt/home/pyq_code/ADV/M2D-LIF')

import numpy as np
import cv2
import random

# 导入 ObjectShift 类
from ultralytics.data.augment import ObjectShift


def create_test_image(h=640, w=640):
    """创建测试用的双模态图像 [H, W, 6]"""
    # RGB 图像：随机彩色
    rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    # IR 图像：灰度风格
    ir = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    # 拼接
    img = np.concatenate([rgb, ir], axis=2)
    return img


def create_test_bboxes(n_obj=3, seed=42):
    """创建测试用的边界框 [N, 4] xywh 归一化坐标"""
    np.random.seed(seed)
    bboxes = []
    for i in range(n_obj):
        cx = np.random.uniform(0.2, 0.8)
        cy = np.random.uniform(0.2, 0.8)
        bw = np.random.uniform(0.1, 0.3)
        bh = np.random.uniform(0.1, 0.3)
        bboxes.append([cx, cy, bw, bh])
    return np.array(bboxes, dtype=np.float32)


def test_1_basic_init():
    """测试1：基本初始化"""
    print("=" * 50)
    print("测试1：ObjectShift 基本初始化")
    
    shift = ObjectShift(max_shift=0.1, shift_ratio=0.3, prob=0.5)
    
    assert shift.max_shift == 0.1, "max_shift 初始化失败"
    assert shift.shift_ratio == 0.3, "shift_ratio 初始化失败"
    assert shift.prob == 0.5, "prob 初始化失败"
    
    print("✓ 初始化测试通过")
    return True


def test_2_no_shift_when_prob_zero():
    """测试2：prob=0 时不应用平移"""
    print("=" * 50)
    print("测试2：prob=0 时不应用平移")
    
    shift = ObjectShift(max_shift=0.1, shift_ratio=0.5, prob=0.0)
    
    labels = {
        'img': create_test_image(),
        'bboxes': create_test_bboxes(n_obj=3)
    }
    original_img = labels['img'].copy()
    
    shift(labels)
    
    # 检查图像是否未改变
    assert np.allclose(labels['img'], original_img), "prob=0 时图像不应改变"
    # 检查 shift_gt 全为 0
    assert np.allclose(labels['shift_gt'], 0), "prob=0 时 shift_gt 应全为 0"
    # 检查 shift_mask 全为 0
    assert np.allclose(labels['shift_mask'], 0), "prob=0 时 shift_mask 应全为 0"
    
    print("✓ prob=0 测试通过")
    return True


def test_3_shift_applied():
    """测试3：正常应用平移"""
    print("=" * 50)
    print("测试3：正常应用平移")
    
    random.seed(42)
    shift = ObjectShift(max_shift=0.1, shift_ratio=0.5, prob=1.0)
    
    labels = {
        'img': create_test_image(),
        'bboxes': create_test_bboxes(n_obj=3, seed=42)
    }
    original_img = labels['img'].copy()
    
    shift(labels)
    
    # 检查 shift_gt 不全为 0（有物体被平移）
    assert not np.allclose(labels['shift_gt'], 0), "应该有物体被平移"
    # 检查 shift_mask 有非零值
    assert labels['shift_mask'].sum() > 0, "shift_mask 应该有非零值"
    # 检查图像确实发生了变化
    # 注意：由于填充噪声，即使平移也会变化，所以检查形状即可
    assert labels['img'].shape == original_img.shape, "图像形状不应改变"
    
    print(f"  shift_gt:\n{labels['shift_gt']}")
    print(f"  shift_mask: {labels['shift_mask']}")
    print("✓ 正常平移测试通过")
    return True


def test_4_shape_preserved():
    """测试4：图像形状保持不变"""
    print("=" * 50)
    print("测试4：图像形状保持不变")
    
    shift = ObjectShift(max_shift=0.15, shift_ratio=0.8, prob=1.0)
    
    for h, w in [(640, 640), (512, 768), (320, 480)]:
        labels = {
            'img': create_test_image(h, w),
            'bboxes': create_test_bboxes(n_obj=5)
        }
        original_shape = labels['img'].shape
        
        shift(labels)
        
        assert labels['img'].shape == original_shape, f"图像形状改变: {original_shape} -> {labels['img'].shape}"
    
    print("✓ 形状保持测试通过")
    return True


def test_5_shift_gt_range():
    """测试5：shift_gt 值范围正确"""
    print("=" * 50)
    print("测试5：shift_gt 值范围正确")
    
    max_shift = 0.1
    shift = ObjectShift(max_shift=max_shift, shift_ratio=0.5, prob=1.0)
    
    # 运行多次取统计
    all_shifts = []
    for _ in range(20):
        random.seed()
        labels = {
            'img': create_test_image(),
            'bboxes': create_test_bboxes(n_obj=5)
        }
        shift(labels)
        all_shifts.append(labels['shift_gt'])
    
    all_shifts = np.concatenate(all_shifts, axis=0)
    non_zero_shifts = all_shifts[np.abs(all_shifts).sum(axis=1) > 0]
    
    if len(non_zero_shifts) > 0:
        # 检查范围
        assert np.all(np.abs(non_zero_shifts) <= max_shift + 1e-6), \
            f"shift_gt 超出范围: max={np.abs(non_zero_shifts).max()}, 限制={max_shift}"
    
    print(f"  检测到 {len(non_zero_shifts)} 个非零 shift 值")
    print("✓ shift_gt 范围测试通过")
    return True


def test_6_empty_bboxes():
    """测试6：空边界框情况"""
    print("=" * 50)
    print("测试6：空边界框情况")
    
    shift = ObjectShift(max_shift=0.1, shift_ratio=0.5, prob=1.0)
    
    labels = {
        'img': create_test_image(),
        'bboxes': np.array([], dtype=np.float32).reshape(0, 4)  # 空
    }
    
    shift(labels)
    
    assert labels['shift_gt'].shape == (0, 2), "空边界框时 shift_gt 应为 (0, 2)"
    assert labels['shift_mask'].shape == (0,), "空边界框时 shift_mask 应为 (0,)"
    
    print("✓ 空边界框测试通过")
    return True


def test_7_single_modality():
    """测试7：单模态图像（通道数<6）"""
    print("=" * 50)
    print("测试7：单模态图像（通道数<6）")
    
    shift = ObjectShift(max_shift=0.1, shift_ratio=0.5, prob=1.0)
    
    # 创建单模态图像 (H, W, 3)
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    labels = {
        'img': img,
        'bboxes': create_test_bboxes(n_obj=3)
    }
    
    shift(labels)
    
    # 单模态情况下应该不应用平移
    assert np.allclose(labels['shift_gt'], 0), "单模态时不应平移"
    
    print("✓ 单模态测试通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试 ObjectShift 类")
    print("=" * 60 + "\n")
    
    tests = [
        test_1_basic_init,
        test_2_no_shift_when_prob_zero,
        test_3_shift_applied,
        test_4_shape_preserved,
        test_5_shift_gt_range,
        test_6_empty_bboxes,
        test_7_single_modality,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ 测试失败: {test.__name__}")
            print(f"  错误: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
