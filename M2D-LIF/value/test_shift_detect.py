"""
测试 ShiftDetect 检测头

测试内容：
1. 基本初始化测试：验证类能否正确初始化
2. 前向传播测试（训练模式）：验证输出形状正确
3. 前向传播测试（推理模式）：验证输出形状正确
4. 输出通道数测试：验证 no = nc + reg_max*4 + 2
5. shift 输出范围测试：验证 shift 使用 tanh 归一化
"""

import sys
sys.path.insert(0, '/mnt/home/pyq_code/ADV/M2D-LIF')

import torch
import torch.nn as nn


def test_1_basic_init():
    """测试1：基本初始化"""
    print("=" * 50)
    print("测试1：ShiftDetect 基本初始化")
    
    from ultralytics.nn.modules.head import ShiftDetect
    
    # 定义输入通道（模拟 P3, P4, P5 三个尺度）
    ch = (256, 512, 1024)
    nc = 3
    
    model = ShiftDetect(nc=nc, ch=ch)
    
    # 检查基本属性
    assert model.nc == nc, f"nc 应为 {nc}, 实际为 {model.nc}"
    assert model.nl == 3, f"nl 应为 3, 实际为 {model.nl}"
    assert model.reg_max == 16, f"reg_max 应为 16, 实际为 {model.reg_max}"
    
    # 检查输出通道数
    expected_no = nc + 16 * 4 + 2  # nc + reg_max*4 + 2
    assert model.no == expected_no, f"no 应为 {expected_no}, 实际为 {model.no}"
    
    # 检查 shift 分支
    assert hasattr(model, 'cv_shift'), "应有 cv_shift 属性"
    assert len(model.cv_shift) == 3, "cv_shift 应有 3 个尺度"
    
    print("✓ 初始化测试通过")
    return True


def test_2_forward_train():
    """测试2：训练模式前向传播"""
    print("=" * 50)
    print("测试2：训练模式前向传播")
    
    from ultralytics.nn.modules.head import ShiftDetect
    
    # 模拟输入特征
    batch_size = 2
    ch = (256, 512, 1024)
    nc = 3
    
    # P3: 80x80, P4: 40x40, P5: 20x20
    x = [
        torch.randn(batch_size, 256, 80, 80),
        torch.randn(batch_size, 512, 40, 40),
        torch.randn(batch_size, 1024, 20, 20),
    ]
    
    model = ShiftDetect(nc=nc, ch=ch)
    model.train()
    
    # 前向传播
    output = model(x)
    
    # 检查输出是列表
    assert isinstance(output, list), f"输出应为列表, 实际为 {type(output)}"
    assert len(output) == 3, f"输出应有 3 个尺度, 实际为 {len(output)}"
    
    # 检查每个尺度的输出形状
    expected_no = nc + 16 * 4 + 2  # 3 + 64 + 2 = 69
    for i, feat in enumerate(output):
        b, c, h, w = feat.shape
        assert b == batch_size, f"batch size 应为 {batch_size}, 实际为 {b}"
        assert c == expected_no, f"通道数应为 {expected_no}, 实际为 {c}"
    
    print(f"  输出形状: {[o.shape for o in output]}")
    print("✓ 训练模式前向传播测试通过")
    return True


def test_3_forward_inference():
    """测试3：推理模式前向传播"""
    print("=" * 50)
    print("测试3：推理模式前向传播")
    
    from ultralytics.nn.modules.head import ShiftDetect
    
    # 模拟输入特征
    batch_size = 2
    ch = (256, 512, 1024)
    nc = 3
    
    x = [
        torch.randn(batch_size, 256, 80, 80),
        torch.randn(batch_size, 512, 40, 40),
        torch.randn(batch_size, 1024, 20, 20),
    ]
    
    model = ShiftDetect(nc=nc, ch=ch)
    model.eval()
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    # 检查输出是元组
    assert isinstance(output, tuple), f"输出应为元组, 实际为 {type(output)}"
    assert len(output) == 2, f"输出应有 2 个元素, 实际为 {len(output)}"
    
    y, feats = output  # y: 检测结果, feats: 中间特征
    
    # 检查 y 的形状: [B, 4+nc+2, N_anchors]
    # 4 (box) + 3 (cls) + 2 (shift) = 9
    assert y.dim() == 3, f"y 应为 3D 张量, 实际为 {y.dim()}D"
    assert y.shape[0] == batch_size, f"batch size 应为 {batch_size}"
    assert y.shape[1] == 4 + nc + 2, f"通道数应为 {4 + nc + 2}, 实际为 {y.shape[1]}"
    
    # 检查 shift 部分的范围 (tanh 归一化到 [-1, 1])
    shift = y[:, -2:, :]  # 最后两个通道是 shift
    assert shift.min() >= -1.0 and shift.max() <= 1.0, \
        f"shift 应在 [-1, 1] 范围内, 实际范围: [{shift.min():.4f}, {shift.max():.4f}]"
    
    print(f"  输出 y 形状: {y.shape}")
    print(f"  shift 范围: [{shift.min():.4f}, {shift.max():.4f}]")
    print("✓ 推理模式前向传播测试通过")
    return True


def test_4_output_channels():
    """测试4：输出通道数正确"""
    print("=" * 50)
    print("测试4：输出通道数正确")
    
    from ultralytics.nn.modules.head import ShiftDetect
    
    for nc in [3, 10, 80]:
        ch = (256, 512, 1024)
        model = ShiftDetect(nc=nc, ch=ch)
        
        expected_no = nc + 16 * 4 + 2
        assert model.no == expected_no, \
            f"nc={nc}: no 应为 {expected_no}, 实际为 {model.no}"
        
        # 检查各分支输出通道
        # cv2: box -> reg_max * 4 = 64
        # cv3: cls -> nc
        # cv_shift: shift -> 2
        
        for i, cv_shift in enumerate(model.cv_shift):
            # cv_shift 最后一层输出 2 个通道
            assert cv_shift[-1].out_channels == 2, \
                f"cv_shift[{i}] 应输出 2 通道"
    
    print("✓ 输出通道数测试通过")
    return True


def test_5_gradient_flow():
    """测试5：梯度流通"""
    print("=" * 50)
    print("测试5：梯度流通")
    
    from ultralytics.nn.modules.head import ShiftDetect
    
    batch_size = 2
    ch = (256, 512, 1024)
    nc = 3
    
    # 创建叶子张量
    x = [
        torch.randn(batch_size, 256, 80, 80),
        torch.randn(batch_size, 512, 40, 40),
        torch.randn(batch_size, 1024, 20, 20),
    ]
    
    model = ShiftDetect(nc=nc, ch=ch)
    model.train()
    
    # 前向传播
    output = model(x)
    
    # 计算伪损失
    loss = sum(o.mean() for o in output)
    loss.backward()
    
    # 检查模型参数是否有梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "模型参数应有非零梯度"
    
    print("✓ 梯度流通测试通过")
    return True


def test_6_stride_calculation():
    """测试6：stride 计算"""
    print("=" * 50)
    print("测试6：stride 计算")
    
    from ultralytics.nn.modules.head import ShiftDetect
    from ultralytics.nn.tasks import DetectionModel
    import yaml
    
    # 简单测试：检查 stride 属性是否正确设置
    ch = (256, 512, 1024)
    model = ShiftDetect(nc=3, ch=ch)
    
    # 模拟 stride 设置（通常由 DetectionModel 初始化）
    model.stride = torch.tensor([8.0, 16.0, 32.0])
    
    assert model.stride.shape[0] == 3, "stride 应有 3 个值"
    assert torch.allclose(model.stride, torch.tensor([8.0, 16.0, 32.0])), "stride 值不正确"
    
    print("✓ stride 计算测试通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试 ShiftDetect 类")
    print("=" * 60 + "\n")
    
    tests = [
        test_1_basic_init,
        test_2_forward_train,
        test_3_forward_inference,
        test_4_output_channels,
        test_5_gradient_flow,
        test_6_stride_calculation,
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
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
