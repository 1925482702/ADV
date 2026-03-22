#!/usr/bin/env python3
"""
测试 Student 模型输出

使用方法:
    python test_student_output.py --model path/to/model.pt --rgb path/to/rgb.jpg --ir path/to/ir.jpg
    
    # 批量测试验证集
    python test_student_output.py --model path/to/model.pt --val_dir path/to/val/images --num 10
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description='测试 Student 模型输出')
    parser.add_argument('--model', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--rgb', type=str, default=None, help='RGB 图像路径')
    parser.add_argument('--ir', type=str, default=None, help='IR 图像路径')
    parser.add_argument('--val_dir', type=str, default=None, help='验证集目录（批量测试）')
    parser.add_argument('--num', type=int, default=10, help='批量测试数量')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--save_dir', type=str, default=None, help='结果保存目录')
    return parser.parse_args()


def load_model(model_path, device):
    """加载模型"""
    print(f"加载模型: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    
    # 优先使用 EMA 模型
    if 'ema' in ckpt and ckpt['ema'] is not None:
        model = ckpt['ema']
        print("使用 EMA 模型")
    elif 'model' in ckpt:
        model = ckpt['model']
        print("使用普通模型")
    else:
        model = ckpt
    
    model = model.to(device).float()
    model.eval()
    
    return model


def load_image_pair(rgb_path, ir_path, imgsz=640):
    """加载并预处理图像对"""
    # 加载图像
    rgb = cv2.imread(rgb_path)
    ir = cv2.imread(ir_path)
    
    if rgb is None:
        raise ValueError(f"无法加载 RGB 图像: {rgb_path}")
    if ir is None:
        raise ValueError(f"无法加载 IR 图像: {ir_path}")
    
    # 记录原始尺寸
    ori_shape = rgb.shape[:2]  # (H, W)
    
    # Letterbox resize
    h, w = rgb.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    rgb_resized = cv2.resize(rgb, (new_w, new_h))
    ir_resized = cv2.resize(ir, (new_w, new_h))
    
    # Pad
    pad_h = (imgsz - new_h) // 2
    pad_w = (imgsz - new_w) // 2
    
    rgb_padded = cv2.copyMakeBorder(rgb_resized, pad_h, imgsz - new_h - pad_h, 
                                     pad_w, imgsz - new_w - pad_w, cv2.BORDER_CONSTANT, value=114)
    ir_padded = cv2.copyMakeBorder(ir_resized, pad_h, imgsz - new_h - pad_h, 
                                    pad_w, imgsz - new_w - pad_w, cv2.BORDER_CONSTANT, value=114)
    
    # 合并为 6 通道
    combined = np.concatenate([rgb_padded, ir_padded], axis=2)  # [H, W, 6]
    
    # 转换为 tensor
    combined_rgb = combined[:, :, ::-1].copy()  # BGR -> RGB
    x = torch.from_numpy(combined_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # [1, 6, H, W]
    
    return x, ori_shape, (scale, pad_h, pad_w)


def postprocess(preds, ori_shape, pad_info, conf_thres=0.25, iou_thres=0.7):
    """后处理：NMS + 坐标还原"""
    from ultralytics.utils import ops
    
    # NMS
    preds_nms = ops.non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres)
    
    # 坐标还原
    scale, pad_h, pad_w = pad_info
    
    results = []
    for pred in preds_nms:
        if len(pred) == 0:
            results.append([])
            continue
        
        # 还原到原始图像坐标
        boxes = pred[:, :4].clone()
        boxes[:, [0, 2]] -= pad_w  # x 去除 padding
        boxes[:, [1, 3]] -= pad_h  # y 去除 padding
        boxes /= scale  # 还原缩放
        
        # 裁剪到图像范围内
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, ori_shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, ori_shape[0])
        
        detections = []
        for i in range(len(pred)):
            x1, y1, x2, y2 = boxes[i].tolist()
            conf = pred[i, 4].item()
            cls = int(pred[i, 5].item())
            detections.append({
                'bbox': [x1, y1, x2 - x1, y2 - y1],  # xywh
                'conf': conf,
                'cls': cls,
            })
        results.append(detections)
    
    return results


def draw_detections(image, detections, names=None):
    """在图像上绘制检测结果"""
    for det in detections:
        x, y, w, h = det['bbox']
        conf = det['conf']
        cls = det['cls']
        
        # 颜色
        color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)][cls % 3]
        
        # 绘制框
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        
        # 绘制标签
        label = f"{names[cls] if names else cls}: {conf:.2f}"
        cv2.putText(image, label, (int(x), int(y) - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image


def test_single(model, rgb_path, ir_path, args, names=None):
    """测试单张图像对"""
    device = next(model.parameters()).device
    
    # 加载图像
    x, ori_shape, pad_info = load_image_pair(rgb_path, ir_path, args.imgsz)
    x = x.to(device)
    
    # 推理
    with torch.no_grad():
        preds = model(x)
    
    # 后处理
    detections = postprocess(preds, ori_shape, pad_info, args.conf)[0]
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"RGB: {rgb_path}")
    print(f"IR:  {ir_path}")
    print(f"检测到 {len(detections)} 个目标:")
    for det in detections:
        print(f"  类别={det['cls']}, 置信度={det['conf']:.4f}, bbox={det['bbox']}")
    
    # 保存结果
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 读取原始图像
        rgb_img = cv2.imread(rgb_path)
        ir_img = cv2.imread(ir_path)
        
        # 绘制检测框
        rgb_vis = draw_detections(rgb_img.copy(), detections, names)
        ir_vis = draw_detections(ir_img.copy(), detections, names)
        
        # 拼接保存
        combined = np.hstack([rgb_vis, ir_vis])
        save_path = os.path.join(args.save_dir, Path(rgb_path).stem + '_det.jpg')
        cv2.imwrite(save_path, combined)
        print(f"结果已保存: {save_path}")
    
    return detections


def test_batch(model, val_dir, args, names=None):
    """批量测试验证集"""
    val_dir = Path(val_dir)
    rgb_dir = val_dir / 'images' / 'val'
    ir_dir = val_dir / 'images_ir' / 'val'
    
    if not rgb_dir.exists():
        rgb_dir = val_dir
        ir_dir = Path(str(val_dir).replace('/images/', '/images_ir/'))
    
    # 获取图像列表
    rgb_files = sorted(list(rgb_dir.glob('*.jpg')))[:args.num]
    
    if not rgb_files:
        print(f"未找到图像文件: {rgb_dir}")
        return
    
    print(f"测试 {len(rgb_files)} 张图像...")
    
    total_dets = 0
    conf_list = []
    
    for rgb_path in rgb_files:
        ir_path = ir_dir / rgb_path.name
        
        if not ir_path.exists():
            print(f"IR 图像不存在: {ir_path}")
            continue
        
        dets = test_single(model, str(rgb_path), str(ir_path), args, names)
        total_dets += len(dets)
        conf_list.extend([d['conf'] for d in dets])
    
    # 统计
    print(f"\n{'='*60}")
    print(f"统计结果:")
    print(f"  测试图像: {len(rgb_files)}")
    print(f"  总检测数: {total_dets}")
    print(f"  平均检测数: {total_dets / len(rgb_files):.2f}")
    if conf_list:
        print(f"  置信度范围: [{min(conf_list):.4f}, {max(conf_list):.4f}]")
        print(f"  平均置信度: {np.mean(conf_list):.4f}")


def main():
    args = parse_args()
    
    # 类别名称
    names = {0: 'car', 1: 'person', 2: 'bicycle'}
    
    # 加载模型
    model = load_model(args.model, args.device)
    
    # 测试
    if args.val_dir:
        test_batch(model, args.val_dir, args, names)
    elif args.rgb and args.ir:
        test_single(model, args.rgb, args.ir, args, names)
    else:
        print("请指定 --rgb 和 --ir 参数，或使用 --val_dir 进行批量测试")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
