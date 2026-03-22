#!/usr/bin/env python3
"""测试模型输出格式"""

import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def test_model_output(model_path, teacher_path=None):
    """测试模型输出"""
    print("=" * 60)
    print("测试模型输出格式")
    print("=" * 60)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 加载检查点
    print(f"\n加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    print(f"检查点键: {list(checkpoint.keys())}")
    
    if 'model' in checkpoint:
        model = checkpoint['model']
        print(f"模型类型: {type(model).__name__}")
        
        # 检查模型属性
        if hasattr(model, 'detect_head'):
            print(f"✓ 有 detect_head 属性")
        if hasattr(model, 'backbone_rgb'):
            print(f"✓ 有 backbone_rgb 属性")
        if hasattr(model, 'backbone_ir'):
            print(f"✓ 有 backbone_ir 属性")
        if hasattr(model, 'fusion'):
            print(f"✓ 有 fusion 属性")
        if hasattr(model, 'teacher'):
            print(f"✓ 有 teacher 属性")
    else:
        print("检查点中没有 'model' 键！")
        return
    
    # 设置为 eval 模式
    model.eval()
    model = model.to(device)
    model = model.float()  # 转为 fp32
    
    # 创建测试输入
    dummy_input = torch.randn(1, 6, 640, 640, device=device)
    print(f"\n输入: {dummy_input.shape}")
    
    # 前向传播
    print("\n运行前向传播...")
    with torch.no_grad():
        output = model(dummy_input)
    
    # 检查输出
    print(f"\n输出类型: {type(output)}")
    if isinstance(output, (list, tuple)):
        print(f"输出长度: {len(output)}")
        for i, o in enumerate(output):
            if isinstance(o, torch.Tensor):
                print(f"  [{i}] shape={o.shape}, min={o.min():.4f}, max={o.max():.4f}, mean={o.mean():.4f}")
                # 检查是否有预测
                if o.dim() == 3 and o.shape[1] >= 4:
                    # 检查置信度
                    if o.shape[1] > 4:
                        conf = o[:, 4:].sigmoid().max()
                        print(f"       最大置信度: {conf:.4f}")
    elif isinstance(output, torch.Tensor):
        print(f"输出 shape: {output.shape}")
        print(f"min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
    
    # 🚨 测试 NMS
    print("\n" + "=" * 60)
    print("测试 NMS")
    print("=" * 60)
    
    from ultralytics.utils.ops import non_max_suppression
    
    # 取第一个输出
    preds = output[0] if isinstance(output, (list, tuple)) else output
    print(f"NMS 输入 shape: {preds.shape}")
    
    # 检查置信度分布
    if preds.shape[1] > 4:
        confs = preds[:, 4:].sigmoid()
        print(f"置信度分布:")
        print(f"  min={confs.min():.4f}, max={confs.max():.4f}, mean={confs.mean():.4f}")
        print(f"  > 0.25 的数量: {(confs > 0.25).sum().item()}")
        print(f"  > 0.5 的数量: {(confs > 0.5).sum().item()}")
    
    # 运行 NMS
    nms_preds = non_max_suppression(
        preds,
        conf_thres=0.25,
        iou_thres=0.7,
        max_det=300,
    )
    
    print(f"\nNMS 输出:")
    for i, pred in enumerate(nms_preds):
        print(f"  图像 {i}: {pred.shape[0]} 个检测框")
        if pred.shape[0] > 0:
            print(f"    前3个: {pred[:3]}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--teacher', type=str, default=None)
    args = parser.parse_args()
    
    test_model_output(args.model, args.teacher)
