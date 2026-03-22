#!/usr/bin/env python3
"""
数据增强详细测试脚本
用于测试 FLIR 数据集的输入和增强逻辑，输出增强的中间过程的内容
"""

import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from copy import deepcopy

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import v8_Pairedtransforms, Compose, PairedLetterBox, PairedMosaic, PairedRandomPerspective
from ultralytics.utils import LOGGER, IterableSimpleNamespace

def visualize_paired_image(img, title, save_path=None):
    """
    可视化双模态图像（RGB + 红外）
    img: numpy array of shape (H, W, 6) where [:, :, :3] is IR and [:, :, 3:] is RGB
    """
    # 注意：根据 load_image 方法，前3个通道是红外，后3个通道是RGB
    ir = img[:, :, :3]
    rgb = img[:, :, 3:]
    
    # 转换RGB从BGR到RGB显示
    rgb_display = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    # 红外图像是3通道，取第一个通道显示
    ir_display = cv2.cvtColor(ir[:, :, 0], cv2.COLOR_GRAY2RGB)
    
    # 合并显示
    combined = np.hstack([rgb_display, ir_display])
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"保存可视化图像到: {save_path}")
    
    return combined

def print_separator(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def main():
    print_separator("开始测试数据增强逻辑")
    
    # 设置路径
    data_path = Path(__file__).parent / "datasets" / "FLIR_yolo"
    
    # 配置数据集参数
    data_config = {
        'path': str(data_path),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,
        'ch': 6,
        'names': ['car', 'person', 'bicycle']
    }
    
    # 设置超参数
    hyp_dict = {
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.0,
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        'crop_fraction': 1.0,
        'patience': 50,
        'save': False,
        'cache': False,
        'device': '0',
        'workers': 8,
        'project': 'runs/train',
        'name': 'exp',
        'exist_ok': False,
        'pretrained': True,
        'optimizer': 'SGD',
        'verbose': True,
        'seed': 0,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,
        'mask_ratio': 4,
        'overlap_mask': True,
        'ch': 6  # 通道数，6表示双模态（RGB+IR）
    }
    hyp = IterableSimpleNamespace(**hyp_dict)
    
    # 创建数据集
    print_separator("创建数据集")
    dataset = YOLODataset(
        img_path=str(data_path / "images" / "train"),
        imgsz=640,
        batch_size=16,
        augment=True,
        hyp=hyp,
        rect=False,
        cache=False,
        single_cls=False,
        task='detect',
        data=data_config
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"数据集类型: {type(dataset)}")
    
    # 选择几张图片进行测试
    test_indices = [0, 1, 2]  # 测试前3张图片
    
    # 创建输出目录
    output_dir = Path(__file__).parent / "augmentation_test_output_detailed"
    output_dir.mkdir(exist_ok=True)
    
    # 测试数据增强流程
    print_separator("测试数据增强流程")
    
    for idx in test_indices:
        print(f"\n{'='*60}")
        print(f"测试图片索引: {idx}")
        print(f"{'='*60}\n")
        
        # 获取原始数据
        print("步骤 1: 获取原始数据...")
        label = dataset.get_image_and_label(idx)
        
        print(f"   - 原始图像路径: {label['im_file']}")
        print(f"   - 原始图像形状: {label['img'].shape}")
        print(f"   - 原始图像形状 (resized_shape): {label.get('resized_shape', 'N/A')}")
        print(f"   - 原始图像形状 (ori_shape): {label.get('ori_shape', 'N/A')}")
        print(f"   - 类别数量: {len(label['cls'])}")
        print(f"   - 类别标签: {label['cls'].flatten()}")
        print(f"   - Instances 数量: {len(label['instances'])}")
        print(f"   - Instances bboxes shape: {label['instances'].bboxes.shape}")
        print(f"   - Instances bboxes (前3个):")
        for i, bbox in enumerate(label['instances'].bboxes[:3]):
            print(f"     {i+1}. {bbox}")
        
        # 保存原始图像
        if 'img' in label and label['img'] is not None:
            original_img = label['img']
            if len(original_img.shape) == 3 and original_img.shape[2] == 6:
                save_path = output_dir / f"original_{idx:04d}.jpg"
                visualize_paired_image(original_img, f"原始图像 {idx}", str(save_path))
        
        # 测试各个增强步骤
        print("\n步骤 2: 测试各个增强步骤...")
        
        # 步骤 2.1: PairedLetterBox
        print("\n  2.1: 测试 PairedLetterBox...")
        label_copy = deepcopy(label)
        letterbox = PairedLetterBox(new_shape=(640, 640), scaleup=False)
        label_letterbox = letterbox(label_copy)
        print(f"     - LetterBox 后图像形状: {label_letterbox['img'].shape}")
        print(f"     - LetterBox 后 resized_shape: {label_letterbox['resized_shape']}")
        print(f"     - LetterBox 后 ratio_pad: {label_letterbox.get('ratio_pad', 'N/A')}")
        print(f"     - LetterBox 后 instances bboxes shape: {label_letterbox['instances'].bboxes.shape}")
        print(f"     - LetterBox 后 instances bboxes (前3个):")
        for i, bbox in enumerate(label_letterbox['instances'].bboxes[:3]):
            print(f"       {i+1}. {bbox}")
        
        # 保存 LetterBox 后的图像
        if 'img' in label_letterbox:
            save_path = output_dir / f"letterbox_{idx:04d}.jpg"
            visualize_paired_image(label_letterbox['img'], f"LetterBox 后图像 {idx}", str(save_path))
        
        # 步骤 2.2: PairedMosaic
        print("\n  2.2: 测试 PairedMosaic...")
        label_copy = deepcopy(label)
        mosaic = PairedMosaic(dataset, imgsz=640, p=1.0)
        # 为了测试 Mosaic，我们需要确保有足够的 buffer
        if len(dataset.buffer) >= 3:
            label_mosaic = mosaic(label_copy)
            print(f"     - Mosaic 后图像形状: {label_mosaic['img'].shape}")
            print(f"     - Mosaic 后 resized_shape: {label_mosaic['resized_shape']}")
            print(f"     - Mosaic 后 instances 数量: {len(label_mosaic['instances'])}")
            print(f"     - Mosaic 后 instances bboxes shape: {label_mosaic['instances'].bboxes.shape}")
            print(f"     - Mosaic 后 instances bboxes (前3个):")
            for i, bbox in enumerate(label_mosaic['instances'].bboxes[:3]):
                print(f"       {i+1}. {bbox}")
            
            # 保存 Mosaic 后的图像
            if 'img' in label_mosaic:
                save_path = output_dir / f"mosaic_{idx:04d}.jpg"
                visualize_paired_image(label_mosaic['img'], f"Mosaic 后图像 {idx}", str(save_path))
        else:
            print(f"     - Buffer 大小不足 ({len(dataset.buffer)} < 3)，跳过 Mosaic 测试")
        
        # 步骤 2.3: 完整的增强流水线
        print("\n  2.3: 测试完整的增强流水线...")
        label_copy = deepcopy(label)
        transforms = v8_Pairedtransforms(dataset, imgsz=640, hyp=hyp)
        
        print(f"     - 增强流水线包含 {len(transforms.transforms)} 个变换:")
        for i, transform in enumerate(transforms.transforms):
            print(f"       {i+1}. {transform.__class__.__name__}")
        
        try:
            label_augmented = transforms(label_copy)
            
            print(f"     - 增强后图像形状: {label_augmented['img'].shape}")
            print(f"     - 增强后 instances 数量: {len(label_augmented['instances'])}")
            print(f"     - 增强后 instances bboxes shape: {label_augmented['instances'].bboxes.shape}")
            print(f"     - 增强后 instances bboxes (前3个):")
            for i, bbox in enumerate(label_augmented['instances'].bboxes[:3]):
                print(f"       {i+1}. {bbox}")
            
            # 保存增强后的图像
            if 'img' in label_augmented:
                save_path = output_dir / f"augmented_{idx:04d}.jpg"
                visualize_paired_image(label_augmented['img'], f"增强后图像 {idx}", str(save_path))
        
        except Exception as e:
            print(f"     [ERROR] 增强过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'='*60}")
    
    print_separator("测试完成")
    print(f"\n所有输出图像已保存到: {output_dir}")
    print("\n请检查以下内容:")
    print("1. 原始图像是否正确加载（RGB和红外通道都存在）")
    print("2. LetterBox 后图像是否正确调整大小")
    print("3. Mosaic 后图像是否正确拼接多张图像")
    print("4. 增强后图像形状是否正确")
    print("5. 边界框坐标是否在图像范围内")
    print("6. 类别标签是否保持一致")

if __name__ == "__main__":
    main()