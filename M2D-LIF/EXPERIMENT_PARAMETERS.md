# 🧪 shift_training_example.py 中的可配置参数详解

## 📊 参数分类与实验设置

### 【第一类】数据增强参数 - PairedShiftAugment

#### 1. `shift_range` - 平移像素范围
- **当前值**: `(10, 50)`
- **含义**: RGB和IR分别在10-50像素范围内随机平移
- **可尝试范围**: 
  - `(5, 30)` - 小平移（温和）
  - `(10, 50)` - 中平移（推荐）
  - `(20, 80)` - 大平移（激进）
  - `(15, 40)` - 中小平移（微调）
- **影响**: 
  - 更大的平移 → 更强的对齐约束，但可能导致过度增强
  - 更小的平移 → 更温和，但可能不足以强制对齐
- **位置**: 
  - 第49行: `shift_range=(10, 50)`
  - 第97行: `shift_range=(10, 50)`
  - 第145行: `shift_range=(10, 50)`

#### 2. `shift_prob` - 平移概率
- **当前值**: `0.3`（30%的概率应用平移）
- **含义**: 在0-1之间，表示数据中有多少比例被平移
- **可尝试范围**:
  - `0.0` - 无平移（baseline）
  - `0.1` - 10% 数据被平移
  - `0.3` - 30% 数据被平移（推荐）
  - `0.5` - 50% 数据被平移（激进）
  - `0.8` - 80% 数据被平移（非常激进）
  - `1.0` - 100% 数据被平移（完全增强）
- **影响**:
  - 概率越高 → 模型看到更多平移数据，更强的对齐训练
  - 概率越低 → 对原始数据影响小，但可能不足以学习对齐
- **位置**:
  - 第50行: `shift_prob=0.3`
  - 第97行: `shift_prob=0.0`
  - 第145行: `shift_prob=1.0`

---

### 【第二类】模型训练参数 - example_basic_training()

#### 3. `epochs` - 训练轮数
- **当前值**: `100`
- **可尝试范围**: `50, 75, 100, 150, 200`
- **影响**: 更多epoch可能获得更好性能，但训练时间更长
- **位置**: 第25行

#### 4. `imgsz` - 输入图像大小
- **当前值**: `640`
- **可尝试范围**: `320, 480, 640, 800, 1024`
- **影响**: 
  - 更大 → 更多细节，更好的小目标检测
  - 更小 → 训练更快，显存占用少
- **位置**: 第26行

#### 5. `batch` - 批大小
- **当前值**: `16`
- **可尝试范围**: `4, 8, 16, 32, 64`
- **影响**:
  - 更大 → 更稳定的梯度，但显存占用多
  - 更小 → 显存占用少，但可能噪声大
- **位置**: 第27行

#### 6. `device` - GPU设备号
- **当前值**: `0`
- **可尝试范围**: `0, 1, 2, 3` 或 `[0, 1]`（多GPU）
- **影响**: 选择使用哪个GPU
- **位置**: 第28行

---

### 【第三类】Loss参数 - example_with_loss_monitoring()

#### 7. `shift_weight` - Shift Loss权重
- **当前值**: `0.1`
- **含义**: Shift loss在总loss中的权重（相对于detection loss）
- **可尝试范围**:
  - `0.01` - 非常小（shift loss几乎被忽视）
  - `0.05` - 小（轻微shift约束）
  - `0.1` - 中（推荐）
  - `0.2` - 大（强shift约束）
  - `0.5` - 非常大（可能压制detection loss）
- **影响**:
  - 权重越大 → shift学习约束越强，但可能影响检测性能
  - 权重越小 → 对检测影响小，但可能不足以学习对齐
- **位置**: 第78行

#### 8. `shift_loss_enabled` - 是否启用Shift Loss
- **当前值**: `True`
- **可尝试范围**: `True, False`
- **影响**: 
  - `True` → 使用shift loss
  - `False` → 只用detection loss（ablation实验）
- **位置**: 第78行（改为 `enabled=False` 测试ablation）

---

### 【第四类】动态调度参数 - DynamicShiftTrainer

#### 9. `total_epochs` - 动态调度的总epoch数
- **当前值**: `100`
- **含义**: 用于计算warmup/early/late阶段的分割点
- **可尝试范围**: 与实际训练epoch相同
- **位置**: 第94行

#### 10. 三阶段划分 - Warmup/Early/Late
- **当前值**:
  - Warmup: 0-20% (0%)
  - Early: 20-80% (线性 0→0.3)
  - Late: 80-100% (0.3)
- **可调参数**:
  ```python
  # Warmup 阶段百分比
  warmup_percent = 0.2  # 当前值，可改为 0.1, 0.15, 0.25, 0.3
  
  # Early 阶段百分比
  early_percent = 0.6  # 当前值（20%-80%），可改为 0.5, 0.4, 0.7
  
  # 最大平移概率（Late阶段）
  max_shift_prob = 0.3  # 当前值，可改为 0.2, 0.4, 0.5, 0.8
  ```

- **影响**:
  - Warmup更长 → 模型有更多时间适应，但learning窗口小
  - Early更长 → 缓慢过渡，但learning周期长
  - 最大平移概率更高 → 更强的对齐约束

- **位置**: 第99-116行

---

### 【第五类】数据参数 - example_verify_augmentation()

#### 11. `img_height, img_width` - 虚拟图像尺寸
- **当前值**: `640, 640`
- **可尝试范围**: `320, 480, 640, 800, 1024`
- **影响**: 影响shift范围相对于图像的比例
- **位置**: 第138-139行
- **实验建议**: 保持与 `imgsz` 一致

#### 12. `num_bboxes` - 虚拟框数量
- **当前值**: `2`（只有2个框在虚拟数据中）
- **可尝试范围**: `1, 2, 5, 10, 20`
- **影响**: 影响面积阈值过滤
- **位置**: 第141行
- **实验建议**: 改为 `np.array([[0.2, 0.2, 0.4, 0.4], [0.5, 0.5, 0.7, 0.7], ...])`

#### 13. `num_augment_iterations` - 增强重复次数
- **当前值**: `3`
- **含义**: 在虚拟数据上重复应用增强的次数
- **可尝试范围**: `1, 3, 5, 10`
- **影响**: 用于验证增强的随机性
- **位置**: 第148行

---

## 🧬 完整实验矩阵模板

### 实验1: Shift参数敏感性分析
```python
# 修改第49行（或第97、145行）
shift_configs = [
    {"shift_range": (5, 30),   "shift_prob": 0.1},
    {"shift_range": (10, 50),  "shift_prob": 0.3},  # baseline
    {"shift_range": (15, 60),  "shift_prob": 0.5},
    {"shift_range": (20, 80),  "shift_prob": 0.3},
    {"shift_range": (10, 50),  "shift_prob": 0.8},
]

for config in shift_configs:
    augment = PairedShiftAugment(**config)
    # 运行训练并记录结果
```

### 实验2: Loss权重扫描
```python
# 修改第78行
loss_weights = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

for weight in loss_weights:
    shift_loss = DetectionLossWithShift(base_loss, shift_weight=weight, enabled=True)
    # 运行训练并记录结果
```

### 实验3: 动态调度对比
```python
# 修改 DynamicShiftTrainer 的三阶段参数
schedules = [
    {"warmup": 0.0,   "early_max": 0.3, "late_max": 0.3},  # baseline
    {"warmup": 0.0,   "early_max": 0.5, "late_max": 0.5},  # 更激进
    {"warmup": 0.3,   "early_max": 0.2, "late_max": 0.3},  # 更长warmup
    {"warmup": 0.0,   "early_max": 0.2, "late_max": 0.2},  # 更保守
]
```

### 实验4: Ablation实验
```python
# 修改第78行
ablations = [
    {"enabled": False},  # 无shift loss
    {"enabled": True, "shift_weight": 0.0},  # shift head但权重为0
    {"enabled": True, "shift_weight": 0.1},  # 完整shift
]
```

---

## 🎯 推荐实验流程

### 第1阶段: 基础验证
- 运行 `example_verify_augmentation()` 验证增强效果
- 保持默认参数，确保代码工作

### 第2阶段: Shift参数优化
- **固定**: Loss权重=0.1，其他默认
- **扫描**: shift_range 和 shift_prob
- **目标**: 找到最佳的增强强度

### 第3阶段: Loss权重优化
- **固定**: 最优shift_range和shift_prob
- **扫描**: shift_weight in [0.01, 0.05, 0.1, 0.2, 0.5]
- **目标**: 找到检测和对齐的平衡点

### 第4阶段: 动态调度优化
- **固定**: 最优shift参数和loss权重
- **扫描**: 三阶段划分和概率曲线
- **目标**: 找到最优的训练进度表

### 第5阶段: 最终验证
- 运行最优配置的完整训练
- 对比baseline（无shift）和最优配置
- 评估检测性能和跨模态一致性

---

## 📈 关键观察指标

运行实验时应监控：

1. **检测Loss**: box_loss, cls_loss, dfl_loss
   - 应该平稳下降
   - 如果增加可能shift_weight太大

2. **Shift Loss**: 应该从高降低
   - 在early阶段应该快速下降
   - 在late阶段应该稳定

3. **总Loss**: 应该平滑下降
   - 不应有剧烈波动

4. **精度指标**: mAP, mAP50, mAP75
   - 应该相比baseline提升或至少不下降

5. **跨模态一致性**: （需自定义指标）
   - RGB和IR的检测结果相关性

---

## 💻 实验运行示例

```python
# 快速实验脚本模板
from examples.shift_training_example import *

# 实验1: 测试不同shift_range
for shift_range in [(5, 30), (10, 50), (20, 80)]:
    augment = PairedShiftAugment(shift_range=shift_range, shift_prob=0.3)
    print(f"Testing shift_range={shift_range}")
    # 运行训练: example_basic_training()
    # 保存结果

# 实验2: 测试不同loss权重
for weight in [0.05, 0.1, 0.2]:
    print(f"Testing shift_weight={weight}")
    # 在 example_with_loss_monitoring 中修改权重
    # 运行训练
    # 保存结果

# 实验3: 测试动态调度
trainer = DynamicShiftTrainer(model, total_epochs=100)
for epoch in range(100):
    shift_prob = trainer.get_shift_prob_for_epoch(epoch)
    print(f"Epoch {epoch}: shift_prob={shift_prob:.3f}")
```

---

## 📋 参数总结表

| 参数 | 当前值 | 推荐范围 | 影响程度 | 实验优先级 |
|------|--------|---------|--------|-----------|
| shift_range | (10,50) | (5,80) | 高 | ⭐⭐⭐ |
| shift_prob | 0.3 | 0.1-0.8 | 高 | ⭐⭐⭐ |
| shift_weight | 0.1 | 0.01-0.5 | 高 | ⭐⭐⭐ |
| epochs | 100 | 50-200 | 中 | ⭐⭐ |
| imgsz | 640 | 320-1024 | 中 | ⭐⭐ |
| batch | 16 | 4-64 | 中 | ⭐⭐ |
| warmup% | 20% | 10%-30% | 中 | ⭐⭐ |
| max_prob | 0.3 | 0.2-0.8 | 中 | ⭐⭐ |

---

**💡 快速开始**: 建议先用默认参数跑通，然后优先扫描 **shift_range** 和 **shift_prob**！
