"""
测试 v8ShiftDetectionLoss 损失函数

测试内容：
1. 基本初始化测试：验证类能否正确初始化
2. 前向传播测试（无 shift_gt）：验证没有 shift_gt 时能正常工作
3. 前向传播测试（有 shift_gt）：验证有 shift_gt 时能正常计算 shift loss
4. 损失值范围测试：验证各项损失值合理
5. 梯度流通测试：验证梯度能正确回传
6. 边界情况测试：空目标、无正样本等
"""

import sys
sys.path.insert(0, '/mnt/home/pyq_code/ADV/M2D-LIF')

import torch
import torch.nn as nn
import numpy as np


class MockDetectHead(nn.Module):
    """模拟 Detect 检测头"""
    def __init__(self, nc=3, reg_max=16, nl=3):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.nl = nl  # number of detection layers
        self.no = nc + reg_max * 4 + 2  # 输出通道数
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        
    def forward(self, x):
        return x


class MockModel:
    """模拟模型类，用于测试 Loss"""
    def __init__(self, nc=3, reg_max=16):
        self.nc = nc
        self.reg_max = reg_max
        
        # 模拟 args
        class Args:
            box = 7.5
            cls = 0.5
            dfl = 1.5
            shift_weight = 1.0
            overlap_mask = True
            
        self.args = Args()
        
        # 模拟检测头
        self.model = [MockDetectHead(nc=nc, reg_max=reg_max)]
        
        # 模拟模型参数
        self.param = nn.Parameter(torch.zeros(1))
    
    def parameters(self):
        yield self.param


def create_mock_preds(batch_size=2, nc=3, reg_max=16, strides=[8, 16, 32]):
    """创建模拟的预测输出"""
    no = nc + reg_max * 4 + 2  # cls + box_reg + shift
    
    feats = []
    for stride in strides:
        h = w = 640 // stride
        feat = torch.randn(batch_size, no, h, w)
        feats.append(feat)
    
    return feats


def create_mock_batch(batch_size=2, n_obj=5, nc=3, imgsz=640):
    """创建模拟的批次数据"""
    batch = {}
    
    # batch_idx: 每个 GT 属于哪个样本
    batch_idx = torch.cat([torch.full((n_obj,), i) for i in range(batch_size)])
    
    # cls: 类别
    cls = torch.randint(0, nc, (batch_size * n_obj,)).float()
    
    # bboxes: xywh 归一化坐标
    bboxes = torch.rand(batch_size * n_obj, 4)
    bboxes[:, 2:] = bboxes[:, 2:] * 0.3 + 0.1  # w, h 在 [0.1, 0.4] 范围
    
    # shift_gt: 每个 GT 的平移量
    shift_gt = torch.zeros(batch_size, n_obj, 2)
    # 随机设置部分物体被平移
    for b in range(batch_size):
        n_shift = np.random.randint(1, n_obj + 1)
        shift_idx = np.random.choice(n_obj, n_shift, replace=False)
        shift_gt[b, shift_idx] = torch.rand(n_shift, 2) * 0.2 - 0.1  # [-0.1, 0.1]
    
    # shift_mask: 哪些物体被平移
    shift_mask = (shift_gt.abs().sum(dim=-1) > 0).float()
    
    # img: 图像（占位）
    batch['img'] = torch.randn(batch_size, 6, imgsz, imgsz)
    batch['batch_idx'] = batch_idx
    batch['cls'] = cls
    batch['bboxes'] = bboxes
    batch['shift_gt'] = shift_gt
    batch['shift_mask'] = shift_mask
    
    return batch


def test_1_basic_init():
    """测试1：基本初始化"""
    print("=" * 50)
    print("测试1：v8ShiftDetectionLoss 基本初始化")
    
    from ultralytics.utils.loss import v8ShiftDetectionLoss
    
    model = MockModel(nc=3)
    loss_fn = v8ShiftDetectionLoss(model)
    
    assert loss_fn.nc == 3, "nc 应为 3"
    assert loss_fn.reg_max == 16, "reg_max 应为 16"
    assert loss_fn.shift_weight == 1.0, "shift_weight 应为 1.0"
    
    print("✓ 初始化测试通过")
    return True


def test_2_forward_no_shift():
    """测试2：无 shift_gt 的前向传播"""
    print("=" * 50)
    print("测试2：无 shift_gt 的前向传播")
    
    from ultralytics.utils.loss import v8ShiftDetectionLoss
    
    model = MockModel(nc=3)
    loss_fn = v8ShiftDetectionLoss(model)
    
    batch_size = 2
    feats = create_mock_preds(batch_size=batch_size, nc=3)
    batch = create_mock_batch(batch_size=batch_size, n_obj=5, nc=3)
    
    # 移除 shift_gt
    batch_no_shift = {k: v for k, v in batch.items() if k not in ['shift_gt', 'shift_mask']}
    
    # 前向传播
    total_loss, loss_items = loss_fn(feats, batch_no_shift)
    
    # 检查输出
    assert total_loss.dim() == 0, "total_loss 应为标量"
    assert loss_items.shape == (4,), f"loss_items 应为 4 元素, 实际为 {loss_items.shape}"
    
    # shift loss 应为 0
    assert loss_items[3] == 0, f"无 shift_gt 时 shift loss 应为 0, 实际为 {loss_items[3]}"
    
    print(f"  total_loss: {total_loss.item():.4f}")
    print(f"  loss_items: {loss_items.tolist()}")
    print("✓ 无 shift_gt 测试通过")
    return True


def test_3_forward_with_shift():
    """测试3：有 shift_gt 的前向传播"""
    print("=" * 50)
    print("测试3：有 shift_gt 的前向传播")
    
    from ultralytics.utils.loss import v8ShiftDetectionLoss
    
    model = MockModel(nc=3)
    model.args.shift_weight = 1.0  # 设置 shift 权重
    loss_fn = v8ShiftDetectionLoss(model)
    
    batch_size = 2
    feats = create_mock_preds(batch_size=batch_size, nc=3)
    batch = create_mock_batch(batch_size=batch_size, n_obj=5, nc=3)
    
    # 前向传播
    total_loss, loss_items = loss_fn(feats, batch)
    
    # 检查输出
    assert total_loss.dim() == 0, "total_loss 应为标量"
    assert loss_items.shape == (4,), f"loss_items 应为 4 元素"
    
    print(f"  total_loss: {total_loss.item():.4f}")
    print(f"  loss_items (box, cls, dfl, shift): {loss_items.tolist()}")
    print(f"  shift_gt 非零数量: {batch['shift_mask'].sum().item()}")
    print("✓ 有 shift_gt 测试通过")
    return True


def test_4_loss_range():
    """测试4：损失值范围合理"""
    print("=" * 50)
    print("测试4：损失值范围合理")
    
    from ultralytics.utils.loss import v8ShiftDetectionLoss
    
    model = MockModel(nc=3)
    loss_fn = v8ShiftDetectionLoss(model)
    
    # 运行多次测试
    for _ in range(5):
        feats = create_mock_preds(batch_size=2, nc=3)
        batch = create_mock_batch(batch_size=2, n_obj=5, nc=3)
        
        total_loss, loss_items = loss_fn(feats, batch)
        
        # 损失值应为有限值
        assert torch.isfinite(total_loss), f"total_loss 应为有限值, 实际为 {total_loss}"
        assert torch.all(torch.isfinite(loss_items)), f"loss_items 应为有限值"
        
        # 损失值应为非负
        assert total_loss >= 0, f"total_loss 应非负"
        assert torch.all(loss_items >= 0), f"loss_items 应非负"
    
    print("✓ 损失值范围测试通过")
    return True


def test_5_gradient_flow():
    """测试5：梯度流通"""
    print("=" * 50)
    print("测试5：梯度流通")
    
    from ultralytics.utils.loss import v8ShiftDetectionLoss
    
    model = MockModel(nc=3)
    loss_fn = v8ShiftDetectionLoss(model)
    
    # 创建需要梯度的预测
    feats = create_mock_preds(batch_size=2, nc=3)
    feats = [f.requires_grad_(True) for f in feats]
    
    batch = create_mock_batch(batch_size=2, n_obj=5, nc=3)
    
    # 前向传播
    total_loss, loss_items = loss_fn(feats, batch)
    
    # 反向传播
    total_loss.backward()
    
    # 检查梯度
    has_grad = False
    for f in feats:
        if f.grad is not None and f.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "应有非零梯度"
    
    print("✓ 梯度流通测试通过")
    return True


def test_6_empty_targets():
    """测试6：空目标情况"""
    print("=" * 50)
    print("测试6：空目标情况")
    
    from ultralytics.utils.loss import v8ShiftDetectionLoss
    
    model = MockModel(nc=3)
    loss_fn = v8ShiftDetectionLoss(model)
    
    # 创建空目标的 batch
    feats = create_mock_preds(batch_size=2, nc=3)
    batch = {
        'img': torch.randn(2, 6, 640, 640),
        'batch_idx': torch.tensor([], dtype=torch.long),
        'cls': torch.tensor([]),
        'bboxes': torch.tensor([]).reshape(0, 4),
    }
    
    # 前向传播（应该不崩溃）
    total_loss, loss_items = loss_fn(feats, batch)
    
    assert torch.isfinite(total_loss), "空目标时损失应为有限值"
    
    print(f"  total_loss: {total_loss.item():.4f}")
    print("✓ 空目标测试通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试 v8ShiftDetectionLoss 类")
    print("=" * 60 + "\n")
    
    tests = [
        test_1_basic_init,
        test_2_forward_no_shift,
        test_3_forward_with_shift,
        test_4_loss_range,
        test_5_gradient_flow,
        test_6_empty_targets,
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
