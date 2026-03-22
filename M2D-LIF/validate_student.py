#!/usr/bin/env python3
"""
验证 DualModalStudent 模型的层索引修改是否正确

测试内容：
1. 模型创建和结构验证
2. Backbone 层数量和索引
3. 前向传播测试（推理模式）
4. 对抗训练接口测试
5. 特征维度验证
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/root/autodl-tmp/ADV/M2D-LIF')

def load_baseline_teacher():
    """加载 baseline teacher 模型"""
    from ultralytics import YOLO
    import os
    
    # 使用预训练的 baseline teacher
    checkpoint_path = '/root/autodl-tmp/ADV/runs/baseline/yolov8s_naive_add5/weights/best.pt'
    
    print(f"  加载预训练 baseline teacher: {checkpoint_path}")
    model = YOLO(checkpoint_path)
    
    # 🚨 返回 YOLO 对象本身，让 DualModalStudent 内部提取 DetectionModel
    return model


def test_model_creation():
    """测试模型创建"""
    print("=" * 60)
    print("测试 1: 模型创建")
    print("=" * 60)
    
    from ADV.models.student import DualModalStudent, create_student_model
    
    # 加载 baseline teacher
    print("  加载 baseline teacher...")
    baseline_teacher = load_baseline_teacher()
    
    # 创建 student 模型
    model = DualModalStudent(
        baseline_teacher=baseline_teacher,
        num_classes=3,
        early_layer_idx=2,
    )
    
    print(f"✓ 模型创建成功")
    print(f"  - num_classes: {model.num_classes}")
    print(f"  - early_layer_idx: {model.early_layer_idx}")
    print(f"  - Backbone RGB 层数: {len(model.backbone_rgb)}")
    print(f"  - Backbone IR 层数: {len(model.backbone_ir)}")
    print(f"  - Neck 层数: {len(model.neck_layers)}")
    
    # 验证层数量
    assert len(model.backbone_rgb) == 10, f"RGB Backbone 应为 10 层，实际 {len(model.backbone_rgb)}"
    assert len(model.backbone_ir) == 10, f"IR Backbone 应为 10 层，实际 {len(model.backbone_ir)}"
    assert len(model.neck_layers) == 12, f"Neck 应为 12 层，实际 {len(model.neck_layers)}"
    
    print("✓ 层数量验证通过")
    return model


def test_forward_inference(model):
    """测试推理模式前向传播"""
    print("\n" + "=" * 60)
    print("测试 2: 推理模式前向传播")
    print("=" * 60)
    
    model.eval()
    
    # 创建测试输入 [B, 6, H, W]
    batch_size = 2
    img_size = 640
    x = torch.randn(batch_size, 6, img_size, img_size)
    
    with torch.no_grad():
        predictions = model(x)
    
    print(f"✓ 前向传播成功")
    print(f"  - 输入形状: {x.shape}")
    print(f"  - 输出类型: {type(predictions)}")
    
    if isinstance(predictions, (list, tuple)):
        for i, pred in enumerate(predictions):
            if isinstance(pred, torch.Tensor):
                print(f"  - 输出[{i}] 形状: {pred.shape}")
    
    return predictions


def test_forward_with_features(model):
    """测试返回特征的前向传播"""
    print("\n" + "=" * 60)
    print("测试 3: 返回特征的前向传播")
    print("=" * 60)
    
    model.eval()
    batch_size = 2
    x = torch.randn(batch_size, 6, 640, 640)
    
    with torch.no_grad():
        predictions, features = model(x, return_features=True)
    
    print(f"✓ 前向传播成功（带特征返回）")
    
    # 验证特征
    if 'student_base_sum' in features:
        base_sum = features['student_base_sum']
        print(f"  - student_base_sum: {len(base_sum)} 个尺度")
        for i, f in enumerate(base_sum):
            print(f"    - P{i+3} 形状: {f.shape}")
    
    if 'teacher_features' in features:
        teacher_f = features['teacher_features']
        print(f"  - teacher_features: {len(teacher_f)} 个尺度")
        for i, f in enumerate(teacher_f):
            print(f"    - P{i+3} 形状: {f.shape}")
    
    if 'fused_features' in features:
        fused = features['fused_features']
        print(f"  - fused_features: {len(fused)} 个尺度")
        for i, f in enumerate(fused):
            print(f"    - P{i+3} 形状: {f.shape}")
    
    return features


def test_adversarial_interface(model):
    """测试对抗训练接口"""
    print("\n" + "=" * 60)
    print("测试 4: 对抗训练接口")
    print("=" * 60)
    
    model.train()
    batch_size = 2
    x = torch.randn(batch_size, 6, 640, 640)
    
    # 测试 forward_clean_for_adv
    print("\n  测试 forward_clean_for_adv...")
    with torch.no_grad():
        predictions, rgb_early, ir_early, y_rgb, y_ir = model.forward_clean_for_adv(x)
    
    print(f"  ✓ forward_clean_for_adv 成功")
    print(f"    - rgb_early 形状: {rgb_early.shape}")
    print(f"    - ir_early 形状: {ir_early.shape}")
    print(f"    - y_rgb 长度: {len(y_rgb)}")
    print(f"    - y_ir 长度: {len(y_ir)}")
    
    # 验证缓存大小
    assert len(y_rgb) == 10, f"y_rgb 长度应为 10，实际 {len(y_rgb)}"
    assert len(y_ir) == 10, f"y_ir 长度应为 10，实际 {len(y_ir)}"
    print("  ✓ 缓存大小验证通过（10层）")
    
    # 测试 forward_with_noise
    print("\n  测试 forward_with_noise...")
    noise_rgb = torch.randn_like(rgb_early) * 0.01
    noise_ir = torch.randn_like(ir_early) * 0.01
    
    with torch.no_grad():
        predictions_noisy, features_noisy = model.forward_with_noise(
            x, noise_rgb=noise_rgb, noise_ir=noise_ir
        )
    
    print(f"  ✓ forward_with_noise 成功")
    print(f"    - predictions 类型: {type(predictions_noisy)}")
    
    return True


def test_feature_dimensions(model):
    """测试特征维度"""
    print("\n" + "=" * 60)
    print("测试 5: 特征维度验证")
    print("=" * 60)
    
    model.eval()
    batch_size = 1
    x = torch.randn(batch_size, 6, 640, 640)
    
    with torch.no_grad():
        predictions, features = model(x, return_features=True)
    
    # 检查融合特征维度
    fused = features['fused_features']
    
    # 预期通道数（根据 yolov8_naive_add.yaml）
    expected_channels = [256, 512, 1024]  # P3, P4, P5
    
    print(f"  融合特征维度检查:")
    for i, (f, exp_ch) in enumerate(zip(fused, expected_channels)):
        actual_ch = f.shape[1]
        status = "✓" if actual_ch == exp_ch else "✗"
        print(f"    P{i+3}: {f.shape} (期望通道 {exp_ch}) {status}")
    
    return features


def test_gradient_flow(model):
    """测试梯度流"""
    print("\n" + "=" * 60)
    print("测试 6: 梯度流验证")
    print("=" * 60)
    
    model.train()
    batch_size = 2
    x = torch.randn(batch_size, 6, 640, 640, requires_grad=True)
    
    # 前向传播
    predictions = model(x)
    
    # 创建伪损失
    if isinstance(predictions, (list, tuple)):
        loss = sum(p.sum() for p in predictions if isinstance(p, torch.Tensor))
    else:
        loss = predictions.sum()
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_grad = x.grad is not None
    print(f"  输入梯度存在: {has_grad}")
    
    if has_grad:
        grad_norm = x.grad.norm().item()
        print(f"  输入梯度范数: {grad_norm:.6f}")
    
    # 检查模型参数梯度
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    print(f"  有梯度的参数: {params_with_grad}/{total_params}")
    
    print("✓ 梯度流测试完成")
    return True


def main():
    print("=" * 60)
    print("DualModalStudent 模型验证脚本")
    print("=" * 60)
    
    try:
        # 测试 1: 模型创建
        model = test_model_creation()
        
        # 测试 2: 推理模式
        test_forward_inference(model)
        
        # 测试 3: 返回特征
        test_forward_with_features(model)
        
        # 测试 4: 对抗训练接口
        test_adversarial_interface(model)
        
        # 测试 5: 特征维度
        test_feature_dimensions(model)
        
        # 测试 6: 梯度流
        test_gradient_flow(model)
        
        print("\n" + "=" * 60)
        print("所有测试通过！ ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
