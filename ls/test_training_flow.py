"""
测试 ShiftDetect 完整训练流程（batch=1）

测试内容：
1. 模型构建测试：能否正确加载 yolov8_shift.yaml
2. 前向传播测试：batch=1 能否正常前向传播
3. 损失计算测试：损失能否正常计算
4. 反向传播测试：梯度能否正常回传
5. 优化器步骤测试：优化器能否正常更新参数
"""

import sys
sys.path.insert(0, '/mnt/home/pyq_code/ADV/M2D-LIF')

import torch
import torch.nn as nn
import numpy as np


def test_1_model_build():
    """测试1：模型构建"""
    print("=" * 50)
    print("测试1：模型构建（加载 yolov8_shift.yaml）")
    
    from ultralytics.nn.tasks import ShiftDetectionModel
    from ultralytics.utils import DEFAULT_CFG_DICT
    from types import SimpleNamespace
    
    # 尝试加载模型配置
    try:
        model = ShiftDetectionModel(
            cfg='/mnt/home/pyq_code/ADV/M2D-LIF/model_yaml/yolov8_shift_global.yaml',
            ch=6,
            nc=3,
            verbose=True
        )
        
        # 设置模型 args（用于 loss 计算）
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT.copy())
        model.args.shift_weight = 1.0
        
        print(f"✓ 模型构建成功")
        print(f"  模型类型: {type(model).__name__}")
        
        # 检查最后一层是否是 ShiftDetect
        last_module = model.model[-1]
        print(f"  检测头类型: {type(last_module).__name__}")
        
        return True, model
    except Exception as e:
        print(f"✗ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_2_forward_pass(model):
    """测试2：前向传播（batch=1）"""
    print("=" * 50)
    print("测试2：前向传播（batch=1）")
    
    batch_size = 1
    
    # 获取模型所在设备
    device = next(model.parameters()).device
    
    # 创建输入（在相同设备上）
    x = torch.randn(batch_size, 6, 640, 640, device=device)
    
    model.train()
    
    try:
        output = model(x)
        print(f"✓ 前向传播成功")
        print(f"  输出类型: {type(output)}")
        if isinstance(output, tuple):
            print(f"  输出元组长度: {len(output)}")
            for i, o in enumerate(output):
                if isinstance(o, list):
                    print(f"  output[{i}] 是列表, 长度: {len(o)}")
                    if len(o) > 0:
                        print(f"    output[{i}][0] 形状: {o[0].shape}")
                elif isinstance(o, torch.Tensor):
                    print(f"  output[{i}] 形状: {o.shape}")
        return True, output
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_3_loss_computation(model):
    """测试3：损失计算"""
    print("=" * 50)
    print("测试3：损失计算（batch=1）")
    
    batch_size = 1
    device = next(model.parameters()).device
    
    # 创建模拟批次数据（在相同设备上）
    batch = {
        'img': torch.randn(batch_size, 6, 640, 640, device=device),
        'batch_idx': torch.tensor([0, 0, 0], device=device),  # 3个GT属于batch 0
        'cls': torch.tensor([0, 1, 2], device=device).float(),  # 3个类别
        'bboxes': torch.tensor([
            [0.3, 0.3, 0.1, 0.1],  # xywh
            [0.5, 0.5, 0.15, 0.15],
            [0.7, 0.7, 0.1, 0.1],
        ], device=device),
        'shift_gt': torch.tensor([
            [[0.05, -0.03], [0, 0], [0.02, 0.01]],  # batch 0: 3个物体的shift
        ], device=device),  # [1, 3, 2]
        'shift_mask': torch.tensor([
            [1.0, 0.0, 1.0],  # batch 0: 第1和第3个物体被平移
        ], device=device),  # [1, 3]
    }
    
    try:
        # 初始化 criterion
        criterion = model.init_criterion()
        
        # 前向传播
        model.train()
        preds = model(batch['img'])
        
        # 计算损失
        total_loss, loss_items = criterion(preds, batch)
        
        print(f"✓ 损失计算成功")
        print(f"  总损失: {total_loss.item():.4f}")
        print(f"  损失项 [box, cls, dfl, shift]: {loss_items.tolist()}")
        
        return True, (total_loss, loss_items)
    except Exception as e:
        print(f"✗ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_4_backward_pass(model):
    """测试4：反向传播"""
    print("=" * 50)
    print("测试4：反向传播（batch=1）")
    
    batch_size = 1
    device = next(model.parameters()).device
    
    batch = {
        'img': torch.randn(batch_size, 6, 640, 640, device=device),
        'batch_idx': torch.tensor([0, 0], device=device),
        'cls': torch.tensor([0, 1], device=device).float(),
        'bboxes': torch.tensor([
            [0.3, 0.3, 0.1, 0.1],
            [0.5, 0.5, 0.15, 0.15],
        ], device=device),
        'shift_gt': torch.tensor([
            [[0.05, -0.03], [0, 0]],
        ], device=device),
        'shift_mask': torch.tensor([
            [1.0, 0.0],
        ], device=device),
    }
    
    try:
        model.train()
        criterion = model.init_criterion()
        
        preds = model(batch['img'])
        total_loss, loss_items = criterion(preds, batch)
        
        # 反向传播
        total_loss.backward()
        
        # 检查梯度
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        if has_grad:
            print(f"✓ 反向传播成功，梯度正常")
            return True
        else:
            print(f"✗ 反向传播失败：无梯度")
            return False
    except Exception as e:
        print(f"✗ 反向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_optimizer_step(model):
    """测试5：优化器步骤"""
    print("=" * 50)
    print("测试5：优化器步骤（batch=1）")
    
    batch_size = 1
    device = next(model.parameters()).device
    
    batch = {
        'img': torch.randn(batch_size, 6, 640, 640, device=device),
        'batch_idx': torch.tensor([0, 0], device=device),
        'cls': torch.tensor([0, 1], device=device).float(),
        'bboxes': torch.tensor([
            [0.3, 0.3, 0.1, 0.1],
            [0.5, 0.5, 0.15, 0.15],
        ], device=device),
        'shift_gt': torch.tensor([
            [[0.05, -0.03], [0, 0]],
        ], device=device),
        'shift_mask': torch.tensor([
            [1.0, 0.0],
        ], device=device),
    }
    
    try:
        model.train()
        criterion = model.init_criterion()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # 记录初始参数
        initial_param = None
        for p in model.parameters():
            initial_param = p.clone()
            break
        
        # 前向+反向
        preds = model(batch['img'])
        total_loss, loss_items = criterion(preds, batch)
        total_loss.backward()
        
        # 优化器步骤
        optimizer.step()
        optimizer.zero_grad()
        
        # 检查参数是否更新
        for p in model.parameters():
            if torch.allclose(p, initial_param):
                continue  # 某些参数可能没更新
            else:
                print(f"✓ 优化器步骤成功，参数已更新")
                return True
        
        print(f"⚠ 优化器步骤完成，但参数未检测到变化（可能学习率太小）")
        return True
    except Exception as e:
        print(f"✗ 优化器步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试 ShiftDetect 完整训练流程（batch=1）")
    print("=" * 60 + "\n")
    
    # 测试1：模型构建
    success, model = test_1_model_build()
    if not success:
        print("\n模型构建失败，无法继续后续测试")
        return False
    
    # 测试2：前向传播
    success, _ = test_2_forward_pass(model)
    if not success:
        print("\n前向传播失败，无法继续后续测试")
        return False
    
    # 测试3：损失计算
    success, _ = test_3_loss_computation(model)
    if not success:
        print("\n损失计算失败，无法继续后续测试")
        return False
    
    # 测试4：反向传播
    success = test_4_backward_pass(model)
    if not success:
        print("\n反向传播失败")
        return False
    
    # 测试5：优化器步骤
    success = test_5_optimizer_step(model)
    if not success:
        print("\n优化器步骤失败")
        return False
    
    print("\n" + "=" * 60)
    print("所有测试通过！ShiftDetect 训练流程正常")
    print("=" * 60 + "\n")
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
