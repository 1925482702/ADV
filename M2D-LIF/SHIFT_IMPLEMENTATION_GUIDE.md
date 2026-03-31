# 跨模态物体平移预测（Cross-Modal Shift Prediction）实现指南

## 概述

本实现提供了跨模态物体平移预测的基础架构，用于强制RGB和IR模态之间的对齐。核心思想是通过独立平移两个模态中的物体，然后让模型预测相对偏移量，从而强制模型同时在两个模态中定位物体。

## 已实现的组件

### 1. ShiftHead（在 `ultralytics/nn/modules/shift.py`）
- 预测RGB和IR之间的相对偏移（dx, dy）
- 输入：融合层特征 (B, C, H, W)
- 输出：相对偏移量 (B, 2, H, W)

### 2. ShiftLoss（在 `ultralytics/utils/shift.py`）
- 使用 Smooth L1 Loss 计算偏移预测误差
- 支持权重调整 (默认 0.1)

### 3. 数据增强（在 `ultralytics/data/augment.py`）
- `ShiftAugment`: 单模态版本
- `PairedShiftAugment`: 成对数据版本
  - 独立对RGB和IR应用随机平移
  - 计算相对偏移GT
  - 使用边界像素均值+高斯模糊填充空洞

### 4. 集成工具（在 `ultralytics/models/shift_integration.py`）
- `ShiftAugmentationManager`: 管理增强的启用/禁用
- `DetectionLossWithShift`: 组合检测和shift loss
- `enable_shift_head_in_model()`: 动态启用模型中的ShiftHead

## 使用步骤

### 步骤 1: 在数据加载时启用Shift增强

在 `ultralytics/data/dataset.py` 的 `YOLODataset.build_transforms()` 中添加：

```python
def build_transforms(self, hyp=None):
    """Builds and appends transforms to the list."""
    if self.augment:
        hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
        hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
        transforms = v8_Pairedtransforms(self, self.imgsz, hyp)
        
        # ✓ 添加这行来启用shift增强
        # transforms.append(PairedShiftAugment(shift_range=(10, 50), shift_prob=0.3))
```

### 步骤 2: 在模型中启用ShiftHead

修改 YAML 配置或在训练器中：

```python
# 在训练器的 get_model() 或 set_model_attributes() 中
from ultralytics.models.shift_integration import enable_shift_head_in_model
enable_shift_head_in_model(self.model, enabled=True)
```

### 步骤 3: 修改Loss计算

在 `ultralytics/models/yolo/detect/train.py` 的 `DetectionTrainer` 中：

```python
def init_criterion(self):
    """Initialize loss criterion with shift support."""
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.models.shift_integration import DetectionLossWithShift
    
    base_loss = v8DetectionLoss(self.model)
    # 包装基础loss来支持shift
    return DetectionLossWithShift(base_loss, shift_weight=0.1, enabled=True)
```

### 步骤 4: 动态调度（可选）

根据训练进度调整shift概率和loss权重：

```python
# 在训练循环中（trainer.py的train_one_epoch或类似位置）
epoch = current_epoch
total_epochs = self.args.epochs

# Warmup: 前20%的epoch不使用shift
# Early: 线性增长
# Late: 达到最大值

if epoch < total_epochs * 0.2:
    shift_prob = 0.0
    shift_loss_weight = 0.0
elif epoch < total_epochs * 0.8:
    progress = (epoch - total_epochs * 0.2) / (total_epochs * 0.6)  # 0->1
    shift_prob = 0.3 * progress
    shift_loss_weight = 0.1 * progress
else:
    shift_prob = 0.3
    shift_loss_weight = 0.1

# 更新augmentation和loss权重
augment_manager.augment.shift_prob = shift_prob
loss_fn.shift_loss.weight = shift_loss_weight
```

## 关键实现细节

### 坐标归一化检测
```python
# ShiftAugment 自动检测坐标格式
bbox_max = bboxes.max()
is_normalized = bbox_max < 2.0  # 归一化坐标 < 2.0
```

### 空洞填补
1. 使用边界像素的平均值填充
2. 应用5x5高斯模糊来平滑过渡
3. 检查最小尺寸（5x5）后再应用模糊

### Loss不归一化
保持loss在正常量级（~1），不除以任何数值

## 调试建议

### 1. 验证增强是否工作
```python
# 在数据加载后检查batch中的shift字段
if 'shift_dx' in batch:
    print(f"Shift applied: dx={batch['shift_dx']}, dy={batch['shift_dy']}")
```

### 2. 监控loss值
```python
# 在训练日志中查看
# train/shift_loss 应该逐渐减小
# 或在动态调度中逐渐增加权重后减小
```

### 3. 可视化平移效果
```python
import cv2
import numpy as np

# 在ShiftAugment中添加保存功能
# 保存平移前后的图像对来验证
```

## 参数建议

- **平移范围**: [10, 50] 像素（可根据图像大小调整）
- **平移概率**: 早期0.3, 后期0.8（动态调度）
- **Shift Loss权重**: 0.05-0.5（根据检测loss调整）
- **area阈值**: 
  - 归一化坐标: 0.001 * w * h
  - 像素坐标: 100

## 下一步

1. **集成到YAML**: 可以在YAML中添加shift参数
2. **融合位置优化**: 当前使用第一个检测层的通道，可以尝试其他融合点
3. **多目标支持**: 目前只处理面积最大+随机的2个目标，可扩展为全部目标
4. **验证**: 在FLIR等多模态数据集上验证效果

## 常见问题

Q: 为什么使用相对偏移而不是绝对偏移？
A: 相对偏移 (RGB shift - IR shift) 强制模型同时在两个模态中定位，相对偏移为0时表示完美对齐。

Q: 高斯模糊为什么重要？
A: 防止插值痕迹，使填充区域更自然，减少模型学习到伪特征。

Q: 为什么要保留20%的无平移样本？
A: 让模型保持对原始对齐数据的敏感度，防止完全依赖平移预测。

## 注意事项

- ⚠️ 确保RGB和IR图像完全配对
- ⚠️ 平移范围不要超过图像大小的1/4
- ⚠️ 确保bboxes正确标准化或转换
- ⚠️ Smooth L1 Loss对异常值更鲁棒，但不使用权重可能导致loss过小

## 引用

此实现基于以下思想：
- 跨模态对齐通过强制同时定位
- 物体平移作为自监督信号
- 相对偏移作为对齐度量
